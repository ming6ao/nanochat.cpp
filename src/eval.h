#ifndef NANOCHAT_SRC_EVAL_H_
#define NANOCHAT_SRC_EVAL_H_

#include <string>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/tokenizer.h"

// Host-side helpers for the evaluation and generation command lines. They turn
// the token-id rows that `GenerateBatch` produced into text. A single token can
// end in the middle of a UTF-8 sequence, so the decode uses
// `TokenStreamDecoder`, which holds the incomplete suffix for the next token.

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

}  // namespace nanochat

#endif  // NANOCHAT_SRC_EVAL_H_
