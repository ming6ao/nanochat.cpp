// tests/post_training_core_test.cc -- the whole-campaign gate for
// docs/post-training.md.
//
// The test builds a tiny model on the CPU reference backend and proves the
// three properties that SFT and RL share:
//
//   1. `ForwardLoss` divides the sum of the per-row losses by the number of
//      targets that are not -1 (docs/post-training.md section 2.1). The
//      reference per-row losses are recomputed independently from the saved
//      raw logits.
//   2. `BackwardWeighted` with all-ones row weights matches `Backward`
//      (docs/post-training.md section 2.2).
//   3. Two `BackwardAccumulate` passes over micro-batches that share a token
//      id sum to one `Backward` over the concatenated batch. The shared
//      embedding row accumulates both contributions.
//
// The model is built with `Model::Create` from a fixed seed, so the test needs
// no fixture and no torch. The same source links the CUDA backend, but the
// production proof is the CPU run under the `t0-cpu` sandbox.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "src/model_impl.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::ParamView;
using nanochat::TrainModel;

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

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

void ExpectNear(double got, double want, double tolerance,
                const std::string& what) {
  const double diff = std::fabs(got - want);
  if (!(diff <= tolerance)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g > tol=%.3g)", what.c_str(),
                got, want, diff, tolerance));
  }
}

// Copies a (possibly device-resident) buffer to host floats.
std::vector<float> ReadCompute(const ComputeType* src, std::int64_t count) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src, static_cast<std::size_t>(count) * sizeof(ComputeType),
        CopyDir::kDeviceToHost);
  }
  std::vector<float> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] =
        nanochat::AsF(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

// The tiny but structurally complete shape shared by every case: two blocks,
// grouped-query attention, a sliding-window pattern, and a value embedding on
// the last layer. `head_dim` is 4, the shape the CPU reference attention
// serves.
Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.rope_base = 100000.0f;
  config.window_pattern = "SL";
  return config;
}

constexpr std::uint64_t kSeed = 20240607;

// Recomputes one row's cross-entropy in double precision from the saved raw
// logits, independent of the classifier's own loss buffer. A target of -1
// yields zero.
std::vector<double> ReferenceRowLosses(const TrainModel* impl,
                                       const std::vector<int>& targets,
                                       int rows, int vocab, int padded) {
  const std::vector<float> raw =
      ReadCompute(impl->raw_logits(), static_cast<std::int64_t>(rows) * padded);
  std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
  const double cap = static_cast<double>(nanochat::kLogitSoftcap);
  for (int r = 0; r < rows; ++r) {
    if (targets[r] == -1) continue;
    const float* row = raw.data() + static_cast<std::size_t>(r) * padded;
    double row_max = -std::numeric_limits<double>::infinity();
    std::vector<double> z(static_cast<std::size_t>(vocab));
    for (int v = 0; v < vocab; ++v) {
      z[static_cast<std::size_t>(v)] =
          cap * std::tanh(static_cast<double>(row[v]) / cap);
      if (z[static_cast<std::size_t>(v)] > row_max) {
        row_max = z[static_cast<std::size_t>(v)];
      }
    }
    double sum_exp = 0.0;
    for (int v = 0; v < vocab; ++v) {
      sum_exp += std::exp(z[static_cast<std::size_t>(v)] - row_max);
    }
    out[static_cast<std::size_t>(r)] =
        std::log(sum_exp) + row_max - z[static_cast<std::size_t>(targets[r])];
  }
  return out;
}

ParamView FindParam(Model* model, const std::string& name) {
  for (const ParamView& view : model->params()) {
    if (name == view.name) return view;
  }
  return ParamView{};
}

// Compares every parameter gradient of two identically initialized models.
// Returns true when all gradients agree within `atol + rtol * max(|a|, |b|)`.
bool CompareGradients(Model* a, Model* b, double atol, double rtol,
                      const std::string& what) {
  const std::vector<ParamView> pa = a->params();
  const std::vector<ParamView> pb = b->params();
  if (pa.size() != pb.size()) {
    Fail(what + ": parameter count differs");
    return false;
  }
  bool ok = true;
  double max_relative = 0.0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    if (std::string(pa[i].name) != pb[i].name) {
      Fail(what + ": parameter order differs at index " + std::to_string(i));
      ok = false;
      continue;
    }
    const std::vector<float> ga = ReadCompute(pa[i].grad, pa[i].count);
    const std::vector<float> gb = ReadCompute(pb[i].grad, pb[i].count);
    for (std::size_t j = 0; j < ga.size(); ++j) {
      const double da = ga[j];
      const double db = gb[j];
      const double limit =
          atol + rtol * std::max({std::fabs(da), std::fabs(db)});
      const double diff = std::fabs(da - db);
      if (std::fabs(da) > 0.0 || std::fabs(db) > 0.0) {
        max_relative =
            std::max(max_relative,
                     diff / std::max({std::fabs(da), std::fabs(db), 1e-12}));
      }
      if (diff > limit) {
        Fail(Format("%s: grad/%s[%zu] %.9g vs %.9g (|diff|=%.3g)", what.c_str(),
                    pa[i].name, j, da, db, diff));
        ok = false;
        break;
      }
    }
  }
  std::printf("  %-42s max relative grad diff %.3g\n", what.c_str(),
              max_relative);
  return ok;
}

// Property 1: the valid-token normalization. The divisor is the number of
// targets that are not -1, shared by pretraining (no ignored target) and by
// SFT and RL (masked targets).
void CheckValidTokenNormalization() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const int vocab = config.vocab_size;
  const int padded = config.padded_vocab_size;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(kSeed);
  auto* impl = static_cast<TrainModel*>(model.get());

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    tokens[i] = (i * 7 + 3) % vocab;
    targets[i] = (i * 11 + 5) % vocab;
  }

  // 1a. All targets valid: the divisor is `rows`, exactly as pretraining.
  const float loss_all =
      model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::vector<double> ref_all =
      ReferenceRowLosses(impl, targets, rows, vocab, padded);
  double sum_all = 0.0;
  for (double value : ref_all) sum_all += value;
  ExpectNear(loss_all, sum_all / rows, 1e-4,
             "all-valid ForwardLoss divides by rows");

  // 1b. Mask every third target. `ForwardLoss` must divide by the valid
  // count, not by `rows`.
  std::vector<int> masked = targets;
  int valid = 0;
  for (int i = 0; i < rows; ++i) {
    if (i % 3 == 2) {
      masked[i] = -1;
    } else {
      ++valid;
    }
  }
  ExpectTrue(valid > 0 && valid < rows,
             "the mask leaves both valid and ignored targets");
  const float loss_masked =
      model->ForwardLoss(tokens.data(), masked.data(), batch, seq);
  const std::vector<double> ref_masked =
      ReferenceRowLosses(impl, masked, rows, vocab, padded);
  double sum_valid = 0.0;
  for (int i = 0; i < rows; ++i) {
    if (masked[i] != -1) sum_valid += ref_masked[i];
  }
  ExpectNear(loss_masked, sum_valid / valid, 1e-4,
             "masked ForwardLoss divides by the valid count");
  ExpectTrue(std::fabs(loss_masked - sum_valid / rows) > 1e-4,
             "masked ForwardLoss is not the all-row mean");
  std::printf(
      "  valid-token normalization: rows=%d valid=%d loss=%.6g ref=%.6g\n",
      rows, valid, loss_masked, sum_valid / valid);
}

// Property 2: `BackwardWeighted` with all-ones weights is `Backward`. The
// batch has ignored targets, so this also pins that a zero-weight ignored row
// stays zero.
void CheckWeightedMatchesBackward() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const int vocab = config.vocab_size;

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    tokens[i] = (i * 5 + 1) % vocab;
    targets[i] = (i % 4 == 3) ? -1 : (i * 13 + 2) % vocab;
  }

  std::unique_ptr<Model> uniform = Model::Create(config);
  uniform->InitWeights(kSeed);
  std::unique_ptr<Model> weighted = Model::Create(config);
  weighted->InitWeights(kSeed);

  uniform->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  uniform->Backward();

  weighted->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::vector<float> ones(static_cast<std::size_t>(rows), 1.0f);
  weighted->ZeroGrad();
  weighted->BackwardWeighted(ones.data(), 1.0f);

  CompareGradients(uniform.get(), weighted.get(), 1e-6, 1e-5,
                   "ones-weighted == Backward");
}

// Property 3: gradient accumulation over micro-batches. Two micro-batches that
// share a token id must sum to one backward over the concatenated batch, and
// the shared embedding row must accumulate both contributions.
void CheckAccumulateMatchesConcatenated() {
  const Config config = TinyConfig();
  const int seq = 8;
  const int micro = 1;
  const int rows = micro * seq;
  const int vocab = config.vocab_size;
  const int hidden = config.hidden_dim;
  const int shared = 5;

  std::vector<int> tokens0(static_cast<std::size_t>(rows));
  std::vector<int> targets0(static_cast<std::size_t>(rows));
  std::vector<int> tokens1(static_cast<std::size_t>(rows));
  std::vector<int> targets1(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    tokens0[i] = (i * 7 + 3) % vocab;
    targets0[i] = (i * 11 + 5) % vocab;
    tokens1[i] = (i * 5 + 9) % vocab;
    targets1[i] = (i * 13 + 7) % vocab;
  }
  // The same token id appears in both micro-batches.
  tokens0[0] = shared;
  tokens1[3] = shared;

  std::vector<int> concat_tokens = tokens0;
  concat_tokens.insert(concat_tokens.end(), tokens1.begin(), tokens1.end());
  std::vector<int> concat_targets = targets0;
  concat_targets.insert(concat_targets.end(), targets1.begin(), targets1.end());

  // One backward over the concatenated batch, divided by 2 * seq valid tokens.
  std::unique_ptr<Model> single = Model::Create(config);
  single->InitWeights(kSeed);
  single->ForwardLoss(concat_tokens.data(), concat_targets.data(), 2, seq);
  single->Backward();

  // Two passes, each divided by `seq` valid tokens and scaled by 1/2, so the
  // accumulated gradient is `(g0 + g1) / (2 * seq)`.
  std::unique_ptr<Model> accum = Model::Create(config);
  accum->InitWeights(kSeed);
  accum->ZeroGrad();
  accum->ForwardLoss(tokens0.data(), targets0.data(), micro, seq);
  accum->BackwardAccumulate(0.5f);
  accum->ForwardLoss(tokens1.data(), targets1.data(), micro, seq);
  accum->BackwardAccumulate(0.5f);

  CompareGradients(single.get(), accum.get(), 1e-6, 1e-4,
                   "micro-batch accumulation == concatenated");

  // The shared embedding row must be nonzero in the accumulated model. This is
  // the embedding accumulation the micro-batch path exists for.
  const ParamView wte = FindParam(accum.get(), "transformer.wte.weight");
  ExpectTrue(wte.value != nullptr, "the model exposes transformer.wte.weight");
  if (wte.value != nullptr) {
    const std::vector<float> grad = ReadCompute(wte.grad, wte.count);
    const std::size_t base = static_cast<std::size_t>(shared) * hidden;
    double norm2 = 0.0;
    for (int d = 0; d < hidden; ++d) {
      norm2 += static_cast<double>(grad[base + d]) * grad[base + d];
    }
    ExpectTrue(norm2 > 0.0,
               "the shared token id accumulates a nonzero embedding gradient");
    std::printf("  shared token id %d: embedding row norm %.6g\n", shared,
                std::sqrt(norm2));
  }
}

}  // namespace

int main() {
  std::printf("post_training_core_test: whole-campaign gate\n");
  CheckValidTokenNormalization();
  CheckWeightedMatchesBackward();
  CheckAccumulateMatchesConcatenated();

  if (g_failures != 0) {
    std::fprintf(stderr, "post_training_core_test: %d check(s) failed\n",
                 g_failures);
    return 1;
  }
  std::printf("post_training_core_test: all checks passed\n");
  return 0;
}
