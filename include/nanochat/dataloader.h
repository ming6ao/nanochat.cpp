#ifndef NANOCHAT_DATALOADER_H_
#define NANOCHAT_DATALOADER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Batched token loader over pre-tokenized shards. Rows are packed BOS-aligned
// (each row starts with BOS; documents are best-fit packed to keep utilization
// at 100%), so `targets` are `tokens` shifted one position to the left
// (docs/data.md).

namespace nanochat {

class DataLoader {
 public:
  // `shards` is an ordered list of token-shard paths; the loader cycles over
  // them. `seed` fixes the row order; when `shuffle` is false the stream is
  // read deterministically in file order.
  DataLoader(std::vector<std::string> shards, int batch, int seq,
             std::uint64_t seed = 42, bool shuffle = true);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Fills `tokens` and `targets` (each `batch * seq` ints). Returns false when
  // the underlying stream is exhausted.
  bool Next(int* tokens, int* targets);

  void Reset();

  int batch() const { return batch_; }
  int seq() const { return seq_; }

  // Per-token byte lengths for bits-per-byte, length `*vocab_size`, or null
  // when unavailable. Not owned by the caller.
  const std::uint8_t* token_bytes(int* vocab_size) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int batch_ = 0;
  int seq_ = 0;
};

}  // namespace nanochat

#endif  // NANOCHAT_DATALOADER_H_
