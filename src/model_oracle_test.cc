// Model oracle test: proves the train graph in `src/model.cc` against the
// committed CPU fixture `tests/data/debug_state.bin`.
//
//   1. build a Config from the fixture's `config/*` records;
//   2. load every `param/<name>` into the model's parameter views;
//   3. run ForwardLoss on `input/tokens` + `input/targets` and compare the raw
//      logits, the soft-capped logits, and the loss;
//   4. run Backward and compare every `grad/<name>`.
//
// The fixture came from nanochat's `gpt.py` (PyTorch, CPU). The model mirrors
// its math and is expected to agree to fp32 roundoff (well under 1e-5). This is
// the proof the P0 model-skeleton node requires; //tests:oracle_test only
// checks the fixture's own algebra.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "src/model_impl.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::Model;
using nanochat::TrainModel;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

constexpr double kForwardTolerance = 1e-5;
constexpr double kBackwardTolerance = 1e-5;
constexpr float kSoftcap = 15.0f;

int g_failures = 0;
double g_max_forward_error = 0.0;
double g_max_backward_error = 0.0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

template <typename T = ComputeType>
float AsFloat32(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return nanochat::Fp16ToFloat(value);
  }
}

std::string Join(const std::vector<std::string>& parts) {
  std::string joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) joined += ", ";
    joined += parts[i];
  }
  return joined;
}

std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/debug_state.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/debug_state.bin");
    candidates.emplace_back(std::string(src_dir) +
                            "/nanochat_cpp/tests/data/debug_state.bin");
  }
  candidates.emplace_back("tests/data/debug_state.bin");
  candidates.emplace_back("nanochat.cpp/tests/data/debug_state.bin");
  candidates.emplace_back("../tests/data/debug_state.bin");

  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error("oracle fixture not found; tried: " +
                           Join(candidates));
}

Config ConfigFromFixture(const Fixture& fixture) {
  Config config;
  config.num_layers =
      static_cast<int>(fixture.Get("config/layers").scalar_int());
  config.num_heads = static_cast<int>(fixture.Get("config/heads").scalar_int());
  config.num_kv_heads =
      static_cast<int>(fixture.Get("config/kv_heads").scalar_int());
  config.hidden_dim = static_cast<int>(fixture.Get("config/embd").scalar_int());
  config.seq_len = static_cast<int>(fixture.Get("config/seq").scalar_int());
  config.vocab_size =
      static_cast<int>(fixture.Get("config/vocab").scalar_int());
  config.padded_vocab_size =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  config.rope_base = 100000.0f;

  const Tensor& pattern = fixture.Get("config/window_pattern");
  std::string window(pattern.data, pattern.data + pattern.numel());
  config.window_pattern = window;
  return config;
}

template <typename T = ComputeType>
void StoreFloat(T* dst, float value) {
  if constexpr (std::is_same_v<T, float>) {
    *dst = value;
  } else {
    *dst = nanochat::Fp16FromFloat(value);
  }
}

void LoadParameters(const Fixture& fixture, Model* model) {
  std::vector<nanochat::ParamView> views = model->params();
  for (const nanochat::ParamView& view : views) {
    const std::string record = std::string("param/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing parameter record '" + record + "'");
      continue;
    }
    const Tensor& source = fixture.Get(record);
    if (source.numel() != view.count) {
      Fail(Format("parameter '%s' count mismatch: model %lld fixture %lld",
                  view.name, static_cast<long long>(view.count),
                  static_cast<long long>(source.numel())));
      continue;
    }
    if (source.dtype != nanochat::oracle::DType::kFp32) {
      Fail("parameter '" + record + "' is not fp32 in the fixture");
      continue;
    }
    const float* src = source.f32();
    // Stage the fixture values on the host, then upload in one copy so the same
    // helper works whether `view.value` is host (CPU) or device (CUDA).
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i, src[i]);
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        nanochat::CopyDir::kHostToDevice);
  }
}

void CopyCompute(const ComputeType* src, std::int64_t count,
                 std::vector<float>* out) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src, static_cast<std::size_t>(count) * sizeof(ComputeType),
        nanochat::CopyDir::kDeviceToHost);
  }
  out->resize(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        AsFloat32(host[static_cast<std::size_t>(i)]);
  }
}

void RunChecks(const Fixture& fixture) {
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int vocab = static_cast<int>(fixture.Get("config/vocab").scalar_int());
  const int padded =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  const Config config = ConfigFromFixture(fixture);
  std::unique_ptr<Model> model = Model::Create(config);
  auto* impl = static_cast<TrainModel*>(model.get());

  LoadParameters(fixture, model.get());

  const Tensor& tokens = fixture.Get("input/tokens");
  const Tensor& targets = fixture.Get("input/targets");
  const int* token_data = reinterpret_cast<const int*>(tokens.i32());
  const int* target_data = reinterpret_cast<const int*>(targets.i32());

  const float loss = model->ForwardLoss(token_data, target_data, batch, seq);
  const float want_loss = fixture.Get("forward/loss").scalar_f32();
  if (std::fabs(static_cast<double>(loss) - static_cast<double>(want_loss)) >
      kForwardTolerance) {
    Fail(Format(
        "loss: got %.9g want %.9g (|diff|=%.3g)", loss, want_loss,
        std::fabs(static_cast<double>(loss) - static_cast<double>(want_loss))));
  }
  g_max_forward_error = std::max(
      g_max_forward_error,
      std::fabs(static_cast<double>(loss) - static_cast<double>(want_loss)));

  // Raw logits: model [rows, padded], fixture [batch, seq, vocab]. Stage the
  // whole device buffer once.
  const ComputeType* raw = impl->raw_logits();
  const Tensor& want_raw = fixture.Get("forward/raw_logits");
  std::vector<ComputeType> raw_host(static_cast<std::size_t>(rows) * padded);
  nanochat::kernels::Memcpy(
      raw_host.data(), raw,
      static_cast<std::size_t>(rows) * padded * sizeof(ComputeType),
      nanochat::CopyDir::kDeviceToHost);
  std::vector<float> raw_flat;
  raw_flat.resize(static_cast<std::size_t>(rows) * vocab);
  for (std::int64_t r = 0; r < rows; ++r) {
    for (int v = 0; v < vocab; ++v) {
      raw_flat[static_cast<std::size_t>(r) * vocab + v] =
          AsFloat32(raw_host[static_cast<std::size_t>(r) * padded + v]);
    }
  }
  // Track the raw-logit error too.
  {
    double max_error = 0.0;
    for (std::int64_t i = 0; i < rows * vocab; ++i) {
      max_error = std::max(
          max_error,
          std::fabs(static_cast<double>(raw_flat[static_cast<std::size_t>(i)]) -
                    static_cast<double>(want_raw.f32()[i])));
    }
    g_max_forward_error = std::max(g_max_forward_error, max_error);
    if (max_error > kForwardTolerance) {
      Fail(Format("forward/raw_logits: max |diff| = %.3g", max_error));
    }
  }

  // Post-softcap logits, reconstructed the same way gpt.py produces them.
  const Tensor& want_logits = fixture.Get("forward/logits");
  {
    double max_error = 0.0;
    for (std::int64_t i = 0; i < rows * vocab; ++i) {
      const float got =
          kSoftcap *
          std::tanh(raw_flat[static_cast<std::size_t>(i)] / kSoftcap);
      max_error = std::max(
          max_error, std::fabs(static_cast<double>(got) -
                               static_cast<double>(want_logits.f32()[i])));
    }
    g_max_forward_error = std::max(g_max_forward_error, max_error);
    if (max_error > kForwardTolerance) {
      Fail(Format("forward/logits: max |diff| = %.3g", max_error));
    }
  }

  // Backward and per-parameter gradient comparison.
  model->Backward();
  std::vector<nanochat::ParamView> views = model->params();
  for (const nanochat::ParamView& view : views) {
    const std::string record = std::string("grad/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing gradient record '" + record + "'");
      continue;
    }
    const Tensor& want = fixture.Get(record);
    if (want.numel() != view.count) {
      Fail(Format("gradient '%s' count mismatch: model %lld fixture %lld",
                  view.name, static_cast<long long>(view.count),
                  static_cast<long long>(want.numel())));
      continue;
    }
    std::vector<float> got;
    CopyCompute(view.grad, view.count, &got);
    double max_error = 0.0;
    std::int64_t worst = -1;
    for (std::int64_t i = 0; i < view.count; ++i) {
      const double diff = std::fabs(static_cast<double>(got[i]) -
                                    static_cast<double>(want.f32()[i]));
      if (diff > max_error) {
        max_error = diff;
        worst = i;
      }
    }
    g_max_backward_error = std::max(g_max_backward_error, max_error);
    if (max_error > kBackwardTolerance) {
      Fail(Format("grad/%s: max |diff| = %.3g at %lld (got %.9g want %.9g)",
                  view.name, max_error, static_cast<long long>(worst),
                  static_cast<double>(got[worst]),
                  static_cast<double>(want.f32()[worst])));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  try {
    path = LocateFixture(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  Fixture fixture;
  try {
    fixture = Fixture::Load(path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  std::printf("model_oracle_test: loaded %s (%zu records)\n", path.c_str(),
              fixture.size());
  RunChecks(fixture);

  std::printf(
      "model_oracle_test: max forward error %.3g, max backward error %.3g\n",
      g_max_forward_error, g_max_backward_error);
  if (g_failures != 0) {
    std::fprintf(stderr, "model_oracle_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("model_oracle_test: all checks passed\n");
  return 0;
}
