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

  // The number of documents this source has read from the underlying stream,
  // counting the documents it filtered out
  // (docs/distributed-design.md section 11). The value is a resumable cursor:
  // a fresh source can skip to it. Returns -1 when the source does not track a
  // position.
  virtual std::int64_t position() const { return -1; }
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
//
// `start` is a global source position (docs/distributed-design.md section 11).
// The wrapper drops every document before `start` and then resumes the stride
// filter. It applies `start` on the creation whose 0-based ordinal is
// `start_on_creation`; every other creation begins at 0. A resume passes the
// stored epoch as `start_on_creation`, so earlier epochs are skipped without a
// full replay. `start` of 0 and `start_on_creation` of 0 keep the
// single-process behavior.
//
// The wrapper always tracks `position()`, even when `world_size` is 1, because
// the data cursor needs the underlying read position to resume exactly.
DocumentSourceFactory ShardDocumentSourceFactory(DocumentSourceFactory factory,
                                                 int rank, int world_size,
                                                 std::int64_t start = 0,
                                                 int start_on_creation = 0);

// The serializable data cursor (docs/distributed-design.md section 11).
// `epoch` is the number of `Reset` calls. `source_position` is the read
// position of the sharded source. `documents` is the encoded, not-yet-packed
// buffer. The reference best-fit packing takes the largest document that fits,
// so it reorders documents within the buffer; the buffer is part of the cursor
// because a document index alone cannot reproduce the packing.
struct DataLoaderState {
  std::int64_t epoch = 0;
  std::int64_t source_position = 0;
  std::vector<std::vector<int>> documents;
};

class DataLoader {
 public:
  // The loader owns a producer thread that reads documents from
  // `source_factory` and tokenizes them with `tokenizer` using the reference
  // BOS-aligned best-fit packing (docs/parquet-native.md). `tokenizer` must
  // outlive the loader. `tokenizer_threads` bounds the encode workers;
  // `document_buffer` bounds the encoded documents held in memory.
  // `restore` primes the buffer and the epoch from a stored cursor. The caller
  // must build `source_factory` so that its first source starts at
  // `restore->source_position` (docs/distributed-design.md section 11). A null
  // `restore` starts fresh.
  DataLoader(DocumentSourceFactory source_factory, const Tokenizer* tokenizer,
             int batch, int seq, std::uint64_t seed = 42,
             int tokenizer_threads = 4, std::size_t document_buffer = 1000,
             const DataLoaderState* restore = nullptr);

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

  // A consistent snapshot of the data cursor for a checkpoint
  // (docs/distributed-design.md section 11). The producer is idle at the
  // snapshot point, so the restored buffer and position replay the packing
  // exactly.
  DataLoaderState State() const;

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
