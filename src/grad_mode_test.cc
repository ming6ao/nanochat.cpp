// T0 test for grad mode (docs/grad-mode.md).
//
// It proves two things. First, the grad-mode forward reproduces the training
// forward row for row. It uses the full classifier, and then a forced small
// chunk that exercises the tiling path.
//
// Second, the grad-mode arena is much smaller than the training arena. The eval
// path saves no layer activations and no backward scratch. The tiny
// architecture keeps the graph cheap. The fixture accessors stage through the
// kernel seam, so the test is backend-agnostic.

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
#include "src/ops.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::NoGradGuard;
using nanochat::TrainModel;

constexpr double kLossTolerance = 1e-3;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

Config TinyConfig() {
  Config config;
  config.num_layers = 12;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SSSL";
  return config;
}

std::vector<ComputeType> StageLosses(Model* model, int rows) {
  auto* impl = static_cast<TrainModel*>(model);
  std::vector<ComputeType> host(static_cast<std::size_t>(rows));
  const ComputeType* losses = impl->losses();
  if (losses != nullptr) {
    nanochat::kernels::Memcpy(
        host.data(), losses,
        static_cast<std::size_t>(rows) * sizeof(ComputeType),
        CopyDir::kDeviceToHost);
  }
  return host;
}

std::vector<int> MakeTokens(int batch, int seq, int vocab) {
  std::vector<int> tokens(static_cast<std::size_t>(batch) * seq);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int>((i * 7 + 3) % static_cast<std::size_t>(vocab));
  }
  return tokens;
}

std::vector<int> MakeTargets(int batch, int seq, int vocab) {
  std::vector<int> targets(static_cast<std::size_t>(batch) * seq);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    targets[i] =
        static_cast<int>((i * 11 + 5) % static_cast<std::size_t>(vocab));
  }
  return targets;
}

void CompareRows(const std::vector<ComputeType>& want,
                 const std::vector<ComputeType>& got, const std::string& what) {
  if (want.size() != got.size()) {
    Fail(what + ": row count mismatch");
    return;
  }
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double a = nanochat::AsF(want[i]);
    const double b = nanochat::AsF(got[i]);
    if (std::fabs(a - b) > kLossTolerance * (1.0 + std::fabs(a))) {
      Fail(what + ": row " + std::to_string(i) + " got " + std::to_string(b) +
           " want " + std::to_string(a));
      return;
    }
  }
}

// The grad-mode forward reproduces the training forward on the same weights.
void CheckEvalParity() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(2024);

  const int batch = 2;
  const int seq = 6;
  const int rows = batch * seq;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  const std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);

  const float train_loss =
      model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::vector<ComputeType> train_rows = StageLosses(model.get(), rows);

  float eval_loss = 0.0f;
  std::vector<ComputeType> eval_rows;
  {
    NoGradGuard guard(model.get());
    eval_loss = model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    eval_rows = StageLosses(model.get(), rows);
  }

  if (std::fabs(static_cast<double>(train_loss) -
                static_cast<double>(eval_loss)) > kLossTolerance) {
    Fail("mean loss differs: got " + std::to_string(eval_loss) + " want " +
         std::to_string(train_loss));
  }
  CompareRows(train_rows, eval_rows, "per-row loss");

  // The guard restores grad mode, so a later training forward still works.
  if (!model->grad_enabled()) {
    Fail("NoGradGuard did not restore grad mode");
  }
}

// A forced small classifier chunk must give the same rows as the full pass.
void CheckEvalTiling() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(99);
  auto* impl = static_cast<TrainModel*>(model.get());
  // Two rows per chunk: per_row = padded_vocab * sizeof(ComputeType).
  const std::int64_t budget =
      2 * static_cast<std::int64_t>(config.padded_vocab_size) *
      static_cast<std::int64_t>(sizeof(ComputeType));
  impl->SetEvalLogitsBudgetForTest(budget);

  const int batch = 3;
  const int seq = 5;
  const int rows = batch * seq;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  const std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);

  model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::vector<ComputeType> train_rows = StageLosses(model.get(), rows);

  std::vector<ComputeType> eval_rows;
  {
    NoGradGuard guard(model.get());
    model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    eval_rows = StageLosses(model.get(), rows);
  }
  CompareRows(train_rows, eval_rows, "tiled per-row loss");
}

// The eval arena drops the layer saves and the backward scratch.
void CheckEvalWorkspaceSmaller() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(11);
  auto* impl = static_cast<TrainModel*>(model.get());

  const int batch = 2;
  const int seq = 6;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  const std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);

  model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::size_t train_bytes = impl->workspace_bytes();

  std::size_t eval_bytes = 0;
  {
    NoGradGuard guard(model.get());
    model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    eval_bytes = impl->workspace_bytes();
  }

  if (train_bytes == 0 || eval_bytes == 0) {
    Fail("workspace was not built");
    return;
  }
  if (eval_bytes * 5 > train_bytes) {
    Fail("eval arena is not much smaller: eval " + std::to_string(eval_bytes) +
         " bytes, train " + std::to_string(train_bytes) + " bytes");
  }
}

}  // namespace

int main() {
  CheckEvalParity();
  CheckEvalTiling();
  CheckEvalWorkspaceSmaller();

  if (g_failures != 0) {
    std::fprintf(stderr, "grad_mode_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("grad_mode_test: all checks passed\n");
  return 0;
}
