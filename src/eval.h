#ifndef NANOCHAT_SRC_EVAL_H_
#define NANOCHAT_SRC_EVAL_H_

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/tokenizer.h"
#include "src/sft_shard.h"

// Host-side helpers for the evaluation and generation command lines. They turn
// the token-id rows that `GenerateBatch` produced into text. A single token can
// end in the middle of a UTF-8 sequence, so the decode uses
// `TokenStreamDecoder`, which holds the incomplete suffix for the next token.
//
// `EvaluatePackedRows` is the supervised-fine-tuning validation metric (design
// decision D6 of docs/sft-parity-plan.md): bits-per-byte over the packed
// `NCSFT001` rows, which is what the reference `evaluate_bpb` reports for a
// finetune.

namespace nanochat {

// Decodes the sampled part of one generated row to text. The prompt ids are
// skipped, so the result is the new text only. The terminal token is absent
// because `GenerateBatch` does not append it.
std::string DecodeGeneratedRow(const Tokenizer& tokenizer,
                               const GeneratedSequence& row);

// Decodes every row with `DecodeGeneratedRow`. `out` is resized to `rows`.
void DecodeGeneratedRows(const Tokenizer& tokenizer,
                         const std::vector<GeneratedSequence>& rows,
                         std::vector<std::string>* out);

// The result of one bits-per-byte pass over the packed validation rows.
struct PackedBpb {
  // `total_nats / (ln(2) * total_bytes)`, the reference metric. It stays at
  // infinity when the pass found no target bytes, which the caller only sees
  // when the evaluator returned false.
  float bpb = std::numeric_limits<float>::infinity();
  // How many targets the pass scored: every position whose target is not the
  // ignore index `-1`. It is the divisor `Model::ForwardLoss` uses, so the
  // summed nats are `mean_loss * valid_targets` per batch.
  std::int64_t valid_targets = 0;
  // The source bytes those targets represent, summed from the table.
  std::int64_t target_bytes = 0;
};

// Bits-per-byte over one pass of the packed validation rows (design decision
// D6 of docs/sft-parity-plan.md), forward-only.
//
// `shard` is an `NCSFT001` shard, `token_bytes` holds one entry per token id
// (the number of source bytes the id represents, as `Tokenizer::TokenBytes`
// reports it), and `device_batch` x `eval_steps` is the number of rows the pass
// reads. The rows are taken from the start of the shard, in order, one batch at
// a time.
//
// The metric is the reference `evaluate_bpb`: sum the negative log-probability
// of every valid target, sum the bytes those targets represent, and divide by
// `ln(2) * total_bytes`. `Model::ForwardLoss` returns the mean over the valid
// targets, so the per-batch sum is `mean_loss * valid_targets`; that is exact
// because every target the packer emits is a real piece with at least one byte
// (a masked or padding position is `-1`, and an id the table reports as zero
// bytes would contribute nothing to the byte sum either).
//
// The pass reads the shard once and never restarts it, so the shard must hold
// `eval_steps * device_batch` rows. A shorter shard is an error rather than a
// quiet partial pass, because a restart rule would report a number no reference
// produces (section 5.4 and section 10 of the plan).
bool EvaluatePackedRows(Model* model, const SftShard& shard,
                        const std::vector<int>& token_bytes, int device_batch,
                        int eval_steps, PackedBpb* out, std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_EVAL_H_
