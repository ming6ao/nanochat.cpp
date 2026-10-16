// Supplementary model gradient check.
//
// The committed fixture zero-initializes `attn.c_proj` and `mlp.c_proj`
// (gpt.py does), which makes the loss independent of the attention and MLP
// sub-graphs; every gradient behind those projections is therefore exactly
// zero in the fixture, and the oracle test cannot exercise their backward.
// It also zero-initializes `smear_lambda`, which disables the smear gate and
// hides SmearBackward's input gradient.
//
// This test closes those gaps: it builds the same tiny architecture, gives the
// projections small nonzero weights and `smear_lambda` a nonzero value so the
// whole graph participates, and verifies the analytic backward against a
// central finite difference of the loss along several random directions over
// every parameter. A directional derivative covers all gradient entries at
// once, which keeps the check cheap on the tiny configuration. See
// docs/testing.md, "Finite-difference checks".

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "src/model_impl.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::Model;
using nanochat::ParamView;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

inline float AsF32(ComputeType value) { return static_cast<float>(value); }

// Writes fp32 host values into the parameter's fp32 master. The caller then
// calls `Model::SyncCompute()` to rebuild the compute copy
// (docs/precision.md section 5). In the fp32 build the master aliases the
// compute copy, so the pair of calls is a plain write.
void StoreValues(const ParamView& view, const float* values) {
  if (view.master == nullptr || view.count <= 0 || values == nullptr) return;
  nanochat::kernels::Memcpy(
      view.master, values, static_cast<std::size_t>(view.count) * sizeof(float),
      nanochat::CopyDir::kHostToDevice);
}

struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed) {}
  std::uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  float Uniform() {
    return static_cast<float>((Next() >> 11) * (1.0 / 9007199254740992.0)) *
               2.0f -
           1.0f;
  }
};

int Run() {
  // The check is a central finite difference with a step of 1e-2. A bf16 store
  // has 7 mantissa bits, so the perturbed loss is a step function of the
  // perturbation and the difference is meaningless. The bf16 parity gate is the
  // bf16 numeric trace (docs/precision.md section 6, Stage 2).
#if !defined(NANOCHAT_PRECISION_FP32)
  std::printf(
      "model_gradient: skipped (bf16 build; a finite difference needs fp32)\n");
  return 0;
#endif
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";

  const int batch = 2;
  const int seq = 8;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(20241001);

  // Give both projections small nonzero weights so the gradient flows through
  // attention, QkPrep, and the MLP.
  Rng rng(0xabcdef1234567890ull);
  {
    std::vector<ParamView> views = model->params();
    for (const ParamView& view : views) {
      const std::string name(view.name);
      if (name.find("c_proj.weight") == std::string::npos) continue;
      std::vector<float> host(static_cast<std::size_t>(view.count));
      for (std::int64_t i = 0; i < view.count; ++i) {
        host[static_cast<std::size_t>(i)] = 0.05f * rng.Uniform();
      }
      StoreValues(view, host.data());
    }
    // `smear_lambda` is initialized to zero (gate disabled). Set it nonzero so
    // the smear gate participates and the finite difference covers
    // SmearBackward's elementwise input gradient.
    for (const ParamView& view : views) {
      if (std::string(view.name) == "smear_lambda") {
        const float value = 0.7f;
        StoreValues(view, &value);
      }
    }
    model->SyncCompute();
  }

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 7 + 3) % config.vocab_size);
    targets[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 11 + 5) % config.vocab_size);
  }

  model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  model->Backward();

  std::vector<ParamView> views = model->params();
  std::vector<std::vector<float>> base(views.size());
  std::vector<std::vector<float>> analytic(views.size());
  for (std::size_t k = 0; k < views.size(); ++k) {
    const std::int64_t count = views[k].count;
    base[k].resize(static_cast<std::size_t>(count));
    analytic[k].resize(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
      base[k][static_cast<std::size_t>(i)] = views[k].master[i];
      analytic[k][static_cast<std::size_t>(i)] = AsF32(views[k].grad[i]);
    }
  }

  const float h = 1e-2f;
  // Tight enough that a wrong elementwise contribution to the smear input
  // gradient (the previous `dgate * gate` bug) fails the check, while leaving
  // room for the central-difference truncation error. The measured errors are
  // ~1.6e-3 with the correct backward and ~3.9e-3 with the bug.
  const double tolerance = 2e-3;
  const int directions = 4;
  for (int direction = 0; direction < directions; ++direction) {
    std::vector<std::vector<float>> dir(views.size());
    double expected = 0.0;
    for (std::size_t k = 0; k < views.size(); ++k) {
      const std::int64_t count = views[k].count;
      dir[k].resize(static_cast<std::size_t>(count));
      for (std::int64_t i = 0; i < count; ++i) {
        const float value = rng.Uniform();
        dir[k][static_cast<std::size_t>(i)] = value;
        expected +=
            static_cast<double>(analytic[k][static_cast<std::size_t>(i)]) *
            static_cast<double>(value);
      }
    }

    // p + h*d
    for (std::size_t k = 0; k < views.size(); ++k) {
      std::vector<float> host(static_cast<std::size_t>(views[k].count));
      for (std::int64_t i = 0; i < views[k].count; ++i) {
        host[static_cast<std::size_t>(i)] =
            base[k][static_cast<std::size_t>(i)] +
            h * dir[k][static_cast<std::size_t>(i)];
      }
      StoreValues(views[k], host.data());
    }
    model->SyncCompute();
    const float loss_plus =
        model->ForwardLoss(tokens.data(), targets.data(), batch, seq);

    // p - h*d
    for (std::size_t k = 0; k < views.size(); ++k) {
      std::vector<float> host(static_cast<std::size_t>(views[k].count));
      for (std::int64_t i = 0; i < views[k].count; ++i) {
        host[static_cast<std::size_t>(i)] =
            base[k][static_cast<std::size_t>(i)] -
            h * dir[k][static_cast<std::size_t>(i)];
      }
      StoreValues(views[k], host.data());
    }
    model->SyncCompute();
    const float loss_minus =
        model->ForwardLoss(tokens.data(), targets.data(), batch, seq);

    // Restore exactly.
    for (std::size_t k = 0; k < views.size(); ++k) {
      StoreValues(views[k], base[k].data());
    }
    model->SyncCompute();

    const double numeric =
        (static_cast<double>(loss_plus) - static_cast<double>(loss_minus)) /
        (2.0 * h);
    const double error = std::fabs(numeric - expected);
    std::printf(
        "model_gradient_test: direction %d analytic %.6g numeric %.6g "
        "|diff| %.3g\n",
        direction, expected, numeric, error);
    if (error > tolerance * (1.0 + std::fabs(numeric))) {
      Fail(Format("direction %d mismatch: analytic %.6g numeric %.6g",
                  direction, expected, numeric));
    }
  }
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main() { return Run(); }
