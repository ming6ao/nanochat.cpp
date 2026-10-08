#ifndef NANOCHAT_DATALOADER_H_
#define NANOCHAT_DATALOADER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Batched token loader over raw documents (docs/parquet-native.md). The loader
// reads documents from a `DocumentSource` and tokenizes them with the reference
// BOS-aligned best-fit packing. Each row starts with BOS; the largest document
// that fits wins; when no document fits, the shortest document fills the row
// exactly. `tokens` and `targets` are the row and its one-position shift.

namespace nanochat {

class Tokenizer;

// A stream of raw text documents, in dataset order.
class DocumentSource {
 public:
  virtual ~DocumentSource() = default;

  // Fetches up to the batch size set at construction. Returns false at the end
  // of the input or on an error. `error` stays empty at the end of the input.
  virtual bool Next(std::vector<std::string>* documents,
                    std::string* error) = 0;
};

// Creates a fresh document source. The loader calls this at construction and
// on every `Reset`, so a training run can cycle epochs. The factory sets
// `*error` and returns null when the source cannot be opened.
using DocumentSourceFactory =
    std::function<std::unique_ptr<DocumentSource>(std::string* error)>;

// Wraps `factory` so that rank `rank` of `world_size` sees document `i` only
// when `i % world_size == rank`, and drops the rest
// (docs/distributed-design.md section 7). The stride is deterministic, so every
// rank rebuilds the same partition on a new epoch. A `world_size` of 1 returns
// `factory` unchanged.
DocumentSourceFactory ShardDocumentSourceFactory(DocumentSourceFactory factory,
                                                 int rank, int world_size);

class DataLoader {
 public:
  // The loader owns a producer thread that reads documents from
  // `source_factory` and tokenizes them with `tokenizer` using the reference
  // BOS-aligned best-fit packing (docs/parquet-native.md). `tokenizer` must
  // outlive the loader. `tokenizer_threads` bounds the encode workers;
  // `document_buffer` bounds the encoded documents held in memory.
  DataLoader(DocumentSourceFactory source_factory, const Tokenizer* tokenizer,
             int batch, int seq, std::uint64_t seed = 42,
             int tokenizer_threads = 4, std::size_t document_buffer = 1000);

  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Fills `tokens` and `targets` (each `batch * seq` ints). Returns false when
  // the underlying stream is exhausted. In document mode a row is always full
  // when this returns true: a document longer than a row is cropped.
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
