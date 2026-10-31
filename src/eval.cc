// Forward-only bits-per-byte evaluation (mirrors nanochat's `evaluate_bpb`).
//
// The metric is vocabulary-size independent: sum the per-token cross-entropy
// (nats) over the batches, sum the number of source bytes the target tokens
// represent, and divide. Special tokens and masked ids (byte length 0) are
// skipped. When the loader has no byte table, every counted token contributes
// one byte, so the result reduces to mean-nats / ln(2).

#include "nanochat/model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"
#include "nanochat/tokenizer.h"
#include "src/eval.h"
#include "src/model_impl.h"
#include "src/ops.h"

namespace nanochat {

float EvalBpb(Model* model, DataLoader* loader, int steps) {
  if (model == nullptr || loader == nullptr || steps <= 0) {
    return std::numeric_limits<float>::infinity();
  }
  const int batch = loader->batch();
  const int seq = loader->seq();
  if (batch <= 0 || seq <= 0) {
    return std::numeric_limits<float>::infinity();
  }

  // Forward-only path (docs/eval.md): this mirrors the reference
  // `@torch.no_grad()`. The guard restores the previous grad mode on return.
  NoGradGuard guard(model);

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));

  int byte_vocab = 0;
  const std::uint8_t* byte_table = loader->token_bytes(&byte_vocab);

  double total_nats = 0.0;
  std::int64_t total_bytes = 0;
  for (int s = 0; s < steps; ++s) {
    if (!loader->Next(tokens.data(), targets.data())) break;
    model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    auto* impl = static_cast<TrainModel*>(model);
    const float* losses = impl->losses();
    if (losses == nullptr) break;
    // Stage the per-token losses once; the CUDA backend keeps them in device
    // memory.
    std::vector<float> losses_host(static_cast<std::size_t>(rows));
    kernels::Memcpy(losses_host.data(), losses,
                    static_cast<std::size_t>(rows) * sizeof(float),
                    CopyDir::kDeviceToHost);
    for (std::int64_t i = 0; i < rows; ++i) {
      const int target = targets[static_cast<std::size_t>(i)];
      if (target < 0) continue;  // ignore index
      std::int64_t bytes = 1;
      if (byte_table != nullptr) {
        bytes = target < byte_vocab ? byte_table[target] : 0;
      }
      if (bytes <= 0) continue;
      total_nats +=
          static_cast<double>(losses_host[static_cast<std::size_t>(i)]);
      total_bytes += bytes;
    }
  }

  if (total_bytes == 0) return std::numeric_limits<float>::infinity();
  const double denominator = std::log(2.0) * static_cast<double>(total_bytes);
  return static_cast<float>(total_nats / denominator);
}

// Forward-only scoring over a padded batch (docs/eval.md section 3.2). One
// `ForwardLoss` call covers the whole batch: the per-position targets are the
// shifted ids, and every position at or past a row's valid length carries the
// ignore index so the classifier writes a zero loss there. The saved per-token
// losses and the raw (pre-softcap) logits are staged to the host, where the
// argmax and the focused logits are read off. No gradients are touched.
void ScoreBatch(Model* model, const int* tokens, int batch, int seq,
                const int* lengths, const ScoreFocus* focus,
                std::vector<ScoreResult>* out) {
  if (out == nullptr) return;
  out->clear();
  if (model == nullptr || tokens == nullptr || batch <= 0 || seq <= 0) {
    return;
  }

  auto* impl = static_cast<TrainModel*>(model);
  const int vocab = model->config().vocab_size;
  const int padded_vocab = model->config().padded_vocab_size;
  if (vocab <= 0 || padded_vocab <= 0) return;

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);

  // Shifted targets with the ignore index at masked positions. Position `p`
  // predicts the token at `p + 1`, so the last valid position of each row and
  // every padding position contribute no loss.
  std::vector<int> targets(static_cast<std::size_t>(rows), -1);
  for (int b = 0; b < batch; ++b) {
    int length = lengths != nullptr ? lengths[b] : seq;
    if (length < 0) length = 0;
    if (length > seq) length = seq;
    const std::int64_t base = static_cast<std::int64_t>(b) * seq;
    for (int p = 0; p + 1 < length; ++p) {
      targets[static_cast<std::size_t>(base + p)] =
          tokens[static_cast<std::size_t>(base + p + 1)];
    }
  }

  model->ForwardLoss(tokens, targets.data(), batch, seq);

  // Stage the per-token losses (the CUDA backend keeps them in device memory).
  std::vector<float> losses_host(static_cast<std::size_t>(rows), 0.0f);
  const float* losses = impl->losses();
  if (losses != nullptr) {
    kernels::Memcpy(losses_host.data(), losses,
                    static_cast<std::size_t>(rows) * sizeof(float),
                    CopyDir::kDeviceToHost);
  }

  // Stage the raw logits once; the argmax and every focused logit read from
  // this host copy.
  const std::int64_t logit_count =
      rows * static_cast<std::int64_t>(padded_vocab);
  std::vector<ComputeType> logits_host(static_cast<std::size_t>(logit_count),
                                       ComputeType{});
  const ComputeType* logits = impl->raw_logits();
  if (logits != nullptr) {
    kernels::Memcpy(logits_host.data(), logits,
                    static_cast<std::size_t>(logit_count) * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
  }

  out->resize(static_cast<std::size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    ScoreResult& result = (*out)[static_cast<std::size_t>(b)];
    result.nll.assign(static_cast<std::size_t>(seq), 0.0f);
    result.argmax.assign(static_cast<std::size_t>(seq), -1);
    int length = lengths != nullptr ? lengths[b] : seq;
    if (length < 0) length = 0;
    if (length > seq) length = seq;
    const std::int64_t base = static_cast<std::int64_t>(b) * seq;
    for (int p = 0; p + 1 < length; ++p) {
      const std::int64_t row = base + p;
      result.nll[static_cast<std::size_t>(p)] =
          losses_host[static_cast<std::size_t>(row)];
      const ComputeType* lrow =
          logits_host.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(padded_vocab);
      int best = 0;
      float best_value = AsF(lrow[0]);
      for (int j = 1; j < vocab; ++j) {
        const float value = AsF(lrow[j]);
        if (value > best_value) {
          best_value = value;
          best = j;
        }
      }
      result.argmax[static_cast<std::size_t>(p)] = best;
    }
    // Masked positions keep the zero loss and the ignore-index argmax.
  }

  if (focus != nullptr) {
    for (int b = 0; b < batch; ++b) {
      const ScoreFocus& request = focus[b];
      ScoreResult& result = (*out)[static_cast<std::size_t>(b)];
      if (request.position < 0 || request.count <= 0 ||
          request.ids == nullptr) {
        continue;
      }
      if (request.position >= seq) continue;
      const std::int64_t row =
          static_cast<std::int64_t>(b) * seq + request.position;
      const ComputeType* lrow =
          logits_host.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(padded_vocab);
      result.focus_logits.reserve(static_cast<std::size_t>(request.count));
      for (int k = 0; k < request.count; ++k) {
        const int id = request.ids[k];
        const float value = (id >= 0 && id < vocab) ? AsF(lrow[id]) : 0.0f;
        result.focus_logits.push_back(value);
      }
    }
  }
}

// Bits-per-byte over the packed validation rows (design decision D6 of
// docs/sft-parity-plan.md). The packed-row counterpart of `EvalBpb`: the rows
// come from an `NCSFT001` shard instead of a document loader, and the byte
// table is the tokenizer's, as an `int` per token id.
//
// `Model::ForwardLoss` returns the mean over the valid targets of one batch, so
// the nats that batch contributes are `mean_loss * valid_targets`. The metric
// then divides by the bytes those targets represent, exactly as the reference
// `evaluate_bpb` does.
bool EvaluatePackedRows(Model* model, const SftShard& shard,
                        const std::vector<int>& token_bytes, int device_batch,
                        int eval_steps, PackedBpb* out, std::string* error) {
  auto fail = [error](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out != nullptr) *out = PackedBpb();
  if (model == nullptr) return fail("the evaluator has no model");
  if (device_batch <= 0) {
    return fail("the evaluation batch size is not positive");
  }
  if (eval_steps <= 0) return fail("the evaluation step count is not positive");
  if (shard.seq <= 0 || shard.rows <= 0) {
    return fail("the validation shard is empty");
  }
  if (shard.tokens.size() < static_cast<std::size_t>(shard.rows) * shard.seq ||
      shard.targets.size() < static_cast<std::size_t>(shard.rows) * shard.seq) {
    return fail("the validation shard holds fewer values than its shape says");
  }

  // One pass, never a restart: the pass reads `eval_steps` batches of
  // `device_batch` rows from the start of the shard (section 5.4 of the plan,
  // and the validation-length risk of section 10).
  const std::int64_t want_rows =
      static_cast<std::int64_t>(eval_steps) * device_batch;
  if (want_rows > shard.rows) {
    return fail("the validation shard holds " + std::to_string(shard.rows) +
                " rows, fewer than the " + std::to_string(want_rows) +
                " that eval_steps * device_batch reads in one pass");
  }

  // Forward-only path (docs/eval.md): this mirrors the reference
  // `@torch.no_grad()`. The guard restores the previous grad mode on return.
  NoGradGuard guard(model);

  const std::size_t rows_per_batch = static_cast<std::size_t>(device_batch) *
                                     static_cast<std::size_t>(shard.seq);
  const std::size_t table_size = token_bytes.size();
  double total_nats = 0.0;
  std::int64_t total_bytes = 0;
  std::int64_t total_valid = 0;

  for (int step = 0; step < eval_steps; ++step) {
    const std::size_t base = static_cast<std::size_t>(step) * rows_per_batch;
    const int* tokens = shard.tokens.data() + base;
    const int* targets = shard.targets.data() + base;

    // The divisor of this batch: every target that is not the ignore index,
    // which is what `ForwardLoss` divides its mean by.
    std::int64_t valid = 0;
    for (std::size_t i = 0; i < rows_per_batch; ++i) {
      if (targets[i] >= 0) ++valid;
    }
    if (valid == 0) continue;

    const float mean_loss =
        model->ForwardLoss(tokens, targets, device_batch, shard.seq);
    if (!std::isfinite(mean_loss)) {
      return fail("the model returned a non-finite loss for evaluation step " +
                  std::to_string(step));
    }
    total_nats += static_cast<double>(mean_loss) * static_cast<double>(valid);
    for (std::size_t i = 0; i < rows_per_batch; ++i) {
      const int target = targets[i];
      if (target < 0) continue;
      if (static_cast<std::size_t>(target) >= table_size) {
        return fail("the shard targets token " + std::to_string(target) +
                    ", which the token-byte table does not cover");
      }
      const int bytes = token_bytes[static_cast<std::size_t>(target)];
      if (bytes > 0) total_bytes += bytes;
    }
    total_valid += valid;
  }

  if (out != nullptr) {
    out->valid_targets = total_valid;
    out->target_bytes = total_bytes;
  }
  if (total_bytes <= 0) {
    return fail("the packed validation rows hold no valid target bytes");
  }
  if (out != nullptr) {
    out->bpb = static_cast<float>(
        total_nats / (std::log(2.0) * static_cast<double>(total_bytes)));
  }
  return true;
}

std::string DecodeGeneratedRow(const Tokenizer& tokenizer,
                               const GeneratedSequence& row) {
  TokenStreamDecoder decoder(tokenizer);
  std::string text;
  // The prompt ids carry mask 0; the sampled ids carry mask 1. Skip the
  // prompt so the caller sees the new text only.
  std::size_t start = 0;
  while (start < row.mask.size() && row.mask[start] == 0) ++start;
  for (std::size_t i = start; i < row.tokens.size(); ++i) {
    decoder.Push(row.tokens[i], &text);
  }
  decoder.Flush(&text);
  return text;
}

void DecodeGeneratedRows(const Tokenizer& tokenizer,
                         const std::vector<GeneratedSequence>& rows,
                         std::vector<std::string>* out) {
  if (out == nullptr) return;
  out->resize(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    (*out)[i] = DecodeGeneratedRow(tokenizer, rows[i]);
  }
}

}  // namespace nanochat
