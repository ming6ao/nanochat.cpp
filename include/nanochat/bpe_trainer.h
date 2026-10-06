#ifndef NANOCHAT_BPE_TRAINER_H_
#define NANOCHAT_BPE_TRAINER_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Byte pair encoding trainer: a native port of `rustbpe`'s
// `train_core_incremental` (docs/tokenizer.md section 9). The trainer is a host
// utility beside `DataLoader`. It uses the C++ standard library and OpenMP. It
// has no kernel dependency, and every test runs on the CPU.

namespace nanochat {

// The fixed nanochat split pattern from docs/tokenizer.md section 2.5. The
// trainer and the encoder share the same scanner.
std::string NanochatSplitPattern();

// Trains a base vocabulary from a text corpus. The base vocabulary holds 256
// byte tokens plus one token per merge. The merge index gives the token id.
class BpeTrainer {
 public:
  explicit BpeTrainer(std::string pattern = NanochatSplitPattern());

  // Splits `text` with the fixed pattern, then adds the count of each unique
  // chunk. The split and count step can use OpenMP, because the reduction is
  // commutative.
  void AddDocument(std::string_view text);

  // Adds one chunk count directly, for tests and for the low-level path.
  void AddChunk(std::string_view chunk, std::int64_t count);

  // Runs the merge loop with `base_vocab_size` base tokens. The loop is
  // single-threaded, so the result does not depend on thread order.
  void Train(int base_vocab_size);

  // Returns the ordered merges. The rank of a merge is 256 plus its index in
  // the returned vector.
  std::vector<std::pair<std::vector<std::uint8_t>, int>> MergeableRanks() const;

  const std::string& pattern() const { return pattern_; }

 private:
  std::string pattern_;

  // Count of each unique chunk, keyed by the raw chunk bytes. The bytes can
  // hold any value, including a zero byte.
  std::map<std::string, std::int64_t> chunk_counts_;

  // The ordered merges after `Train`. Each pair is (left id, right id).
  std::vector<std::pair<int, int>> merges_;
};

}  // namespace nanochat

#endif  // NANOCHAT_BPE_TRAINER_H_
