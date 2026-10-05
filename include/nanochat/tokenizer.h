#ifndef NANOCHAT_TOKENIZER_H_
#define NANOCHAT_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Host BPE tokenizer: encode/decode for the CLI and evaluation, plus the
// per-token byte lengths that `EvalBpb` needs. Training and evaluation read
// pre-tokenized shards, so this is not on the training hot path (docs/model.md).

namespace nanochat {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;

  // Token id for a special token name, for example "<|bos|>", or -1.
  virtual int SpecialId(std::string_view name) const = 0;

  // Encodes ordinary text. When `prepend`/`append` are >= 0 the corresponding
  // special id is added at that end.
  virtual std::vector<int> Encode(std::string_view text, int prepend = -1,
                                  int append = -1) const = 0;

  virtual std::string Decode(const int* ids, int count) const = 0;

  // Number of source bytes each token id represents, length `vocab_size`;
  // special tokens and masked ids are 0. Mirrors nanochat's `get_token_bytes`.
  virtual const std::uint8_t* TokenBytes() const = 0;
};

// Loads a saved tokenizer artifact. Returns null when the file is missing or
// malformed.
std::unique_ptr<Tokenizer> LoadTokenizer(const std::string& path);

}  // namespace nanochat

#endif  // NANOCHAT_TOKENIZER_H_
