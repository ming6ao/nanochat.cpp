// src/masked_loss_test.cc -- the masked, mean-normalized cross-entropy and the
// weighted backward (docs/post-training.md section 2).
//
// The test proves four things on the tiny architecture used by
// `src/harness_test.cc` and `src/optim_test.cc`:
//   1. `ForwardLoss` divides by the number of targets that are not -1, not by
//      the row count;
//   2. an ignored row carries a zero per-row loss;
//   3. `BackwardWeighted` with all-ones weights agrees with
//      `BackwardAccumulate` on a masked batch, so the weighted path shares the
//      valid-token divisor;
//   4. a row weight of 0 in `BackwardWeighted` is equivalent to an ignored
//      target, so a masked position contributes no gradient.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/tensor.h"
#include "src/model_impl.h"

namespace {

using nanochat::AsF;
using nanochat::ComputeType;
using nanochat::Config;
using nanochat::Model;
using nanochat::ParamView;
using nanochat::TrainModel;

constexpr std::uint64_t kSeed = 20241001;

// The weighted and uniform backward paths reach the same value through
// different arithmetic. fp32 agrees to a tight tolerance; the fp16 storage
// rounds the per-row scale, so it needs a wider one.
#if defined(NANOCHAT_PRECISION_FP16)
constexpr double kGradientTolerance = 5e-2;
#else
constexpr double kGradientTolerance = 1e-4;
#endif

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& what) {
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(stderr, "FAIL: %s: got %.10g want %.10g\n", what.c_str(),
                 actual, expected);
    ++g_failures;
  }
}

Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

// Stages the per-row losses that the most recent forward saved. The CUDA
// backend keeps them in device memory.
std::vector<ComputeType> ReadLosses(TrainModel* model, std::int64_t rows) {
  std::vector<ComputeType> host(static_cast<std::size_t>(rows));
  const ComputeType* losses = model->losses();
  nanochat::kernels::Memcpy(
      host.data(), losses, static_cast<std::size_t>(rows) * sizeof(ComputeType),
      nanochat::CopyDir::kDeviceToHost);
  return host;
}

void ExpectSameGradients(Model* a, Model* b, const std::string& what) {
  const std::vector<ParamView> pa = a->params();
  const std::vector<ParamView> pb = b->params();
  if (pa.size() != pb.size()) {
    Fail(what + ": parameter count mismatch");
    return;
  }
  double worst = 0.0;
  for (std::size_t k = 0; k < pa.size(); ++k) {
    if (pa[k].count != pb[k].count) {
      Fail(what + ": parameter count mismatch");
      return;
    }
    for (std::int64_t i = 0; i < pa[k].count; ++i) {
      worst =
          std::max(worst, std::fabs(static_cast<double>(AsF(pa[k].grad[i])) -
                                    static_cast<double>(AsF(pb[k].grad[i]))));
    }
  }
  if (worst > kGradientTolerance) {
    std::fprintf(stderr, "FAIL: %s: worst gradient difference %.3g\n",
                 what.c_str(), worst);
    ++g_failures;
  }
}

// The loss is the mean over the valid targets only. The ignored rows carry a
// zero loss, so an all-rows mean would be strictly smaller.
void TestValidCountDenominator() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(kSeed);
  TrainModel* impl = static_cast<TrainModel*>(model.get());

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 7 + 3) % config.vocab_size);
    targets[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 11 + 5) % config.vocab_size);
  }
  const std::vector<std::int64_t> masked = {0, 5, rows - 1};
  for (std::int64_t m : masked) targets[static_cast<std::size_t>(m)] = -1;
  const std::int64_t valid = rows - static_cast<std::int64_t>(masked.size());

  const float loss =
      model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::vector<ComputeType> losses = ReadLosses(impl, rows);

  double valid_sum = 0.0;
  for (std::int64_t i = 0; i < rows; ++i) {
    const double value = AsF(losses[static_cast<std::size_t>(i)]);
    if (targets[static_cast<std::size_t>(i)] != -1) valid_sum += value;
  }

  for (std::int64_t m : masked) {
    if (AsF(losses[static_cast<std::size_t>(m)]) != 0.0f) {
      Fail("masked row carries a nonzero per-row loss");
    }
  }

  ExpectNear(loss, valid_sum / static_cast<double>(valid), 1e-5,
             "ForwardLoss uses the valid count as the divisor");

  // The all-rows mean divides by more rows, so it must differ from the valid
  // count mean. This makes the previous check meaningful.
  if (valid_sum > 0.0) {
    const double all_rows_mean = valid_sum / static_cast<double>(rows);
    if (std::fabs(static_cast<double>(loss) - all_rows_mean) < 1e-6) {
      Fail("ForwardLoss equals the all-rows mean; the divisor is wrong");
    }
  }
}

// On a masked batch, `BackwardWeighted` with all-ones weights is the uniform
// `BackwardAccumulate` pass. Both divide by the valid count.
void TestWeightedMatchesUniformOnMaskedBatch() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  std::unique_ptr<Model> uniform = Model::Create(config);
  uniform->InitWeights(kSeed);
  std::unique_ptr<Model> weighted = Model::Create(config);
  weighted->InitWeights(kSeed);

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 7 + 3) % config.vocab_size);
    targets[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 11 + 5) % config.vocab_size);
  }
  targets[2] = -1;
  targets[9] = -1;

  uniform->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  uniform->ZeroGrad();
  uniform->BackwardAccumulate(1.0f);

  std::vector<float> weights(static_cast<std::size_t>(rows), 1.0f);
  weighted->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  weighted->ZeroGrad();
  weighted->BackwardWeighted(weights.data(), 1.0f);

  ExpectSameGradients(uniform.get(), weighted.get(),
                      "BackwardWeighted all-ones matches BackwardAccumulate");
}

// A row weight of 0 removes a row exactly like an ignored target. Model A
// masks the targets and runs the plain backward; model B keeps every target
// valid, zeroes the same rows in the weight vector, and scales the kept rows
// by rows / valid so the weighted backward reproduces the 1 / valid divisor.
void TestZeroWeightMatchesMaskedTarget() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  std::unique_ptr<Model> masked = Model::Create(config);
  masked->InitWeights(kSeed);
  std::unique_ptr<Model> weighted = Model::Create(config);
  weighted->InitWeights(kSeed);

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets_all_valid(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 7 + 3) % config.vocab_size);
    targets_all_valid[static_cast<std::size_t>(i)] =
        static_cast<int>((i * 11 + 5) % config.vocab_size);
  }

  std::vector<int> targets_masked = targets_all_valid;
  const std::vector<std::int64_t> masked_rows = {1, 4, 10, 15};
  for (std::int64_t m : masked_rows) {
    targets_masked[static_cast<std::size_t>(m)] = -1;
  }
  const std::int64_t valid =
      rows - static_cast<std::int64_t>(masked_rows.size());

  std::vector<float> weights(static_cast<std::size_t>(rows), 1.0f);
  for (std::int64_t m : masked_rows) {
    weights[static_cast<std::size_t>(m)] = 0.0f;
  }

  masked->ForwardLoss(tokens.data(), targets_masked.data(), batch, seq);
  masked->ZeroGrad();
  masked->Backward();

  weighted->ForwardLoss(tokens.data(), targets_all_valid.data(), batch, seq);
  weighted->ZeroGrad();
  weighted->BackwardWeighted(
      weights.data(), static_cast<float>(rows) / static_cast<float>(valid));

  ExpectSameGradients(masked.get(), weighted.get(),
                      "zero row weight matches an ignored target");
}

}  // namespace

int main() {
  TestValidCountDenominator();
  TestWeightedMatchesUniformOnMaskedBatch();
  TestZeroWeightMatchesMaskedTarget();
  if (g_failures != 0) {
    std::fprintf(stderr, "masked_loss_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("masked_loss_test: ok\n");
  return 0;
}
