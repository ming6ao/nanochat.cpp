#ifndef NANOCHAT_SRC_SHARD_WRITER_H_
#define NANOCHAT_SRC_SHARD_WRITER_H_

#include <cstdint>
#include <memory>
#include <string>

// Writes the pre-tokenized `NANO` shard and the `<shard>.bytes` sidecar that
// `DataLoader` reads (docs/model.md, docs/tokenizer.md section 10). The shard
// is a fixed 24-byte little-endian header followed by the token stream. The
// sidecar is a little-endian `uint32` vocabulary size followed by one byte per
// token id.
//
// The class is a host utility. It has no kernel dependency and no DuckDB
// dependency.

namespace nanochat {

class ShardWriter {
 public:
  // Opens `path` for writing. `width` is 2 (uint16) or 4 (uint32).
  static std::unique_ptr<ShardWriter> Open(const std::string& path, int width,
                                           std::string* error);
  ~ShardWriter();

  ShardWriter(const ShardWriter&) = delete;
  ShardWriter& operator=(const ShardWriter&) = delete;

  // Appends `count` token ids. Returns false and sets `error` on a bad token,
  // a bad width, or a write failure.
  bool Write(const int* tokens, std::int64_t count, std::string* error);

  // Writes the final header with the token count and closes the file. Returns
  // false and sets `error` on failure.
  bool Close(std::string* error);

  std::uint64_t num_tokens() const { return num_tokens_; }

 private:
  ShardWriter();

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::uint64_t num_tokens_ = 0;
};

// Writes `<shard_path>.bytes`: a little-endian `uint32` vocabulary size then
// one byte per token id. Returns false and sets `error` on a bad argument or a
// write failure.
bool WriteTokenBytes(const std::string& shard_path,
                     const std::uint8_t* token_bytes, int vocab_size,
                     std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_SHARD_WRITER_H_
