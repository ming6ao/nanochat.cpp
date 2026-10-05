#ifndef NANOCHAT_DATA_H_
#define NANOCHAT_DATA_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nanochat/tensor.h"

// Data formats: the pre-tokenized token shards and the self-describing
// checkpoint container (docs/model.md). No tokenizer, parquet, pyarrow, or
// numpy is needed at runtime; the training loop reads raw little-endian bytes.

namespace nanochat {

// ---------------------------------------------------------------------------
// Pre-tokenized token shards
// ---------------------------------------------------------------------------
//
// Layout: a fixed 24-byte little-endian header followed by the token stream.
// `magic` is "NANO" read as bytes, `token_width` is 2 (uint16) or 4 (uint32).

inline constexpr std::uint32_t kTokenShardMagic = 0x4f4e414eu;
inline constexpr std::uint32_t kTokenShardVersion = 1;

enum class TokenWidth : std::uint32_t {
  kUint16 = 2,
  kUint32 = 4,
};

struct TokenShardHeader {
  std::uint32_t magic = kTokenShardMagic;
  std::uint32_t version = kTokenShardVersion;
  std::uint64_t num_tokens = 0;
  std::uint32_t token_width = static_cast<std::uint32_t>(TokenWidth::kUint16);
  std::uint32_t reserved = 0;
};

// Reads a shard. Tokens are widened to int on the host so kernel signatures can
// stay `const int*`.
class TokenShard {
 public:
  ~TokenShard();

  static std::unique_ptr<TokenShard> Open(const std::string& path,
                                          std::string* error = nullptr);

  const TokenShardHeader& header() const { return header_; }
  std::uint64_t num_tokens() const { return header_.num_tokens; }

  // Copies `count` tokens starting at `offset` into `out` (already widened to
  // int). Returns the number of tokens actually read.
  std::int64_t Read(std::uint64_t offset, std::int64_t count, int* out);

 private:
  TokenShard();

  struct Impl;
  std::unique_ptr<Impl> impl_;
  TokenShardHeader header_;
};

// ---------------------------------------------------------------------------
// Checkpoint container
// ---------------------------------------------------------------------------
//
// A self-describing container: `name`, dtype, shape, and raw payload per
// tensor. Not a framework `state_dict`.

struct TensorRecord {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;

  std::int64_t numel() const {
    std::int64_t total = 1;
    for (std::int64_t dim : shape) total *= dim;
    return total;
  }
};

class Checkpoint {
 public:
  bool Load(const std::string& path, std::string* error = nullptr);
  bool Save(const std::string& path) const;

  const std::vector<TensorRecord>& tensors() const { return tensors_; }
  std::vector<TensorRecord>& tensors() { return tensors_; }

  const TensorRecord* Find(std::string_view name) const;
  TensorRecord* Find(std::string_view name);

  void Add(TensorRecord record);
  void Clear() { tensors_.clear(); }

 private:
  std::vector<TensorRecord> tensors_;
};

}  // namespace nanochat

#endif  // NANOCHAT_DATA_H_
