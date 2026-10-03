// T0 unit test for `ScoreBatch` (docs/eval.md section 3.2).
//
// It checks the shape contract (`nll` and `argmax` are one entry per padded
// position), the masking contract (a row's final valid position and all padding
// carry a zero loss and the ignore-index argmax), that the reported loss and
// argmax agree with a hand-built `ForwardLoss` over the same shifted targets,
// and that the focused logits are exactly the raw logits at the requested
// position for the requested ids. The same tiny architecture as the other src
// tests keeps the whole graph cheap; the fixture accessors stage through the
// kernel seam, so the test is backend-agnostic.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "model_impl.h"
#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/tensor.h"
#include "ops.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::ScoreFocus;
using nanochat::ScoreResult;
using nanochat::TrainModel;

constexpr double kTolerance = 1e-4;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
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

std::vector<int> ExpectedTargets(const std::vector<int>& tokens,
                                 const std::vector<int>& lengths, int batch,
                                 int seq) {
  std::vector<int> targets(static_cast<std::size_t>(batch) * seq, -1);
  for (int b = 0; b < batch; ++b) {
    const int length =
        lengths.empty() ? seq : lengths[static_cast<std::size_t>(b)];
    for (int p = 0; p + 1 < length; ++p) {
      targets[static_cast<std::size_t>(b) * seq + p] =
          tokens[static_cast<std::size_t>(b) * seq + p + 1];
    }
  }
  return targets;
}

int Argmax(const ComputeType* row, int vocab) {
  int best = 0;
  float best_value = nanochat::AsF(row[0]);
  for (int j = 1; j < vocab; ++j) {
    const float value = nanochat::AsF(row[j]);
    if (value > best_value) {
      best_value = value;
      best = j;
    }
  }
  return best;
}

// Runs `ForwardLoss` on the hand-built targets and stages the saved losses and
// raw logits, the reference the primitive must reproduce.
struct Reference {
  std::vector<ComputeType> losses;
  std::vector<ComputeType> logits;
};

Reference RunReference(Model* model, const std::vector<int>& tokens,
                       const std::vector<int>& targets, int batch, int seq) {
  auto* impl = static_cast<TrainModel*>(model);
  model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  const int padded_vocab = model->config().padded_vocab_size;
  Reference reference;
  reference.losses.resize(static_cast<std::size_t>(rows));
  nanochat::kernels::Memcpy(
      reference.losses.data(), impl->losses(),
      static_cast<std::size_t>(rows) * sizeof(ComputeType),
      CopyDir::kDeviceToHost);
  reference.logits.resize(static_cast<std::size_t>(rows) * padded_vocab);
  nanochat::kernels::Memcpy(
      reference.logits.data(), impl->raw_logits(),
      static_cast<std::size_t>(rows) * padded_vocab * sizeof(ComputeType),
      CopyDir::kDeviceToHost);
  return reference;
}

void CheckClosed(double got, double want, const std::string& what) {
  if (std::fabs(got - want) > kTolerance * (1.0 + std::fabs(want))) {
    Fail(what + ": got " + std::to_string(got) + " want " +
         std::to_string(want));
  }
}

// Shapes, masking, and agreement with a hand-built forward, with a focus
// request at a valid position and at a row's masked final position.
void CheckShapeMaskingAndFocus() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(1234);

  const int batch = 2;
  const int seq = 6;
  const std::vector<int> tokens = {5, 9, 2, 7, 11, 3, 4, 8, 1, 0, 0, 0};
  const std::vector<int> lengths = {6, 3};

  const std::vector<int> focus0 = {1, 5, 63};
  const std::vector<int> focus1 = {7};
  std::vector<ScoreFocus> focus(batch);
  focus[0].position = 2;  // valid prediction position in row 0
  focus[0].ids = focus0.data();
  focus[0].count = static_cast<int>(focus0.size());
  focus[1].position = lengths[1] - 1;  // masked loss, but logits are defined
  focus[1].ids = focus1.data();
  focus[1].count = static_cast<int>(focus1.size());

  std::vector<ScoreResult> results;
  nanochat::ScoreBatch(model.get(), tokens.data(), batch, seq, lengths.data(),
                       focus.data(), &results);

  if (results.size() != static_cast<std::size_t>(batch)) {
    Fail("ScoreBatch returned the wrong row count");
    return;
  }
  for (int b = 0; b < batch; ++b) {
    if (results[static_cast<std::size_t>(b)].nll.size() !=
            static_cast<std::size_t>(seq) ||
        results[static_cast<std::size_t>(b)].argmax.size() !=
            static_cast<std::size_t>(seq)) {
      Fail("result row " + std::to_string(b) + " does not have seq entries");
      return;
    }
  }
  if (results[0].focus_logits.size() != focus0.size() ||
      results[1].focus_logits.size() != focus1.size()) {
    Fail("ScoreBatch returned the wrong number of focused logits");
    return;
  }

  const std::vector<int> targets = ExpectedTargets(tokens, lengths, batch, seq);
  const Reference reference =
      RunReference(model.get(), tokens, targets, batch, seq);
  const int padded = config.padded_vocab_size;
  const int vocab = config.vocab_size;

  for (int b = 0; b < batch; ++b) {
    const ScoreResult& result = results[static_cast<std::size_t>(b)];
    for (int p = 0; p < seq; ++p) {
      const bool valid = p + 1 < lengths[static_cast<std::size_t>(b)];
      const std::size_t index =
          static_cast<std::size_t>(b) * seq + static_cast<std::size_t>(p);
      if (valid) {
        CheckClosed(result.nll[static_cast<std::size_t>(p)],
                    nanochat::AsF(reference.losses[index]),
                    "nll row " + std::to_string(b) + " position " +
                        std::to_string(p));
        const ComputeType* lrow =
            reference.logits.data() + index * static_cast<std::size_t>(padded);
        if (result.argmax[static_cast<std::size_t>(p)] !=
            Argmax(lrow, vocab)) {
          Fail("argmax row " + std::to_string(b) + " position " +
               std::to_string(p) + " disagrees with the raw logits");
        }
      } else {
        if (result.nll[static_cast<std::size_t>(p)] != 0.0f ||
            result.argmax[static_cast<std::size_t>(p)] != -1) {
          Fail("masked position (" + std::to_string(b) + ", " +
               std::to_string(p) + ") is not zero/-1");
        }
      }
    }
  }

  // Row 0 focus at prediction position 2.
  for (std::size_t k = 0; k < focus0.size(); ++k) {
    const std::size_t base = static_cast<std::size_t>(2) * padded;
    CheckClosed(results[0].focus_logits[k],
                nanochat::AsF(reference.logits[base +
                                                static_cast<std::size_t>(
                                                    focus0[k])]),
                "row 0 focus " + std::to_string(k));
  }
  // Row 1 focus at the masked final position (index 2).
  {
    const std::size_t base =
        static_cast<std::size_t>(1) * seq * padded +
        static_cast<std::size_t>(2) * padded;
    CheckClosed(results[1].focus_logits[0],
                nanochat::AsF(reference.logits[base +
                                                static_cast<std::size_t>(
                                                    focus1[0])]),
                "row 1 focus");
  }
}

// A null `lengths` treats every row as full length; a null `focus` leaves every
// `focus_logits` empty.
void CheckFullLengthNoFocus() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(7);

  const int batch = 2;
  const int seq = 5;
  const std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<ScoreResult> results;
  nanochat::ScoreBatch(model.get(), tokens.data(), batch, seq, nullptr, nullptr,
                       &results);

  if (results.size() != static_cast<std::size_t>(batch)) {
    Fail("full-length batch returned the wrong row count");
    return;
  }
  const std::vector<int> targets = ExpectedTargets(tokens, {}, batch, seq);
  const Reference reference =
      RunReference(model.get(), tokens, targets, batch, seq);

  for (int b = 0; b < batch; ++b) {
    const ScoreResult& result = results[static_cast<std::size_t>(b)];
    if (!result.focus_logits.empty()) {
      Fail("null focus produced focused logits");
    }
    if (result.nll.size() != static_cast<std::size_t>(seq) ||
        result.argmax.size() != static_cast<std::size_t>(seq)) {
      Fail("full-length row does not have seq entries");
      continue;
    }
    for (int p = 0; p < seq; ++p) {
      const std::size_t index =
          static_cast<std::size_t>(b) * seq + static_cast<std::size_t>(p);
      if (p == seq - 1) {
        if (result.nll[static_cast<std::size_t>(p)] != 0.0f ||
            result.argmax[static_cast<std::size_t>(p)] != -1) {
          Fail("full-length final position is not masked");
        }
      } else {
        CheckClosed(result.nll[static_cast<std::size_t>(p)],
                    nanochat::AsF(reference.losses[index]),
                    "full-length nll");
      }
    }
  }
}

// A degenerate but legal input: every row length 0, every position masked.
void CheckEmptyRows() {
  const Config config = TinyConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(3);

  const int batch = 1;
  const int seq = 4;
  const std::vector<int> tokens = {9, 9, 9, 9};
  const std::vector<int> lengths = {0};

  std::vector<ScoreResult> results;
  nanochat::ScoreBatch(model.get(), tokens.data(), batch, seq, lengths.data(),
                       nullptr, &results);

  if (results.size() != 1) {
    Fail("empty-row batch returned the wrong row count");
    return;
  }
  for (int p = 0; p < seq; ++p) {
    if (results[0].nll[static_cast<std::size_t>(p)] != 0.0f ||
        results[0].argmax[static_cast<std::size_t>(p)] != -1) {
      Fail("an empty row has an unmasked position");
    }
  }
}

}  // namespace

int main() {
  CheckShapeMaskingAndFocus();
  CheckFullLengthNoFocus();
  CheckEmptyRows();

  if (g_failures != 0) {
    std::fprintf(stderr, "eval_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("eval_test: all checks passed\n");
  return 0;
}
