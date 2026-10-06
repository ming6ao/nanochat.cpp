// Data containers (docs/model.md): the pre-tokenized token shard, the
// self-describing checkpoint container, and the batched `DataLoader`.
//
// Everything here is host-side and vendor-free; it only reads and writes
// little-endian bytes. The shard header is a fixed 24 bytes; the checkpoint is
// a simple tabular container (name, dtype, shape, raw payload) that round-trips
// every `TensorRecord`. It is deliberately not a framework `state_dict`.
//
// `LoadTokenizer` lives in the `tokenizer` library
// (src/tokenizer/tokenizer.cc); this file does not define it.

#include "nanochat/data.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/rand.h"

namespace nanochat {
namespace {

// --- Little-endian stream helpers ------------------------------------------

void WriteU16(std::ostream& out, std::uint16_t value) {
  const unsigned char bytes[2] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteU32(std::ostream& out, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu),
      static_cast<unsigned char>((value >> 16) & 0xffu),
      static_cast<unsigned char>((value >> 24) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteU64(std::ostream& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    const unsigned char byte =
        static_cast<unsigned char>((value >> (8 * i)) & 0xffu);
    out.write(reinterpret_cast<const char*>(&byte), 1);
  }
}

void WriteI64(std::ostream& out, std::int64_t value) {
  WriteU64(out, static_cast<std::uint64_t>(value));
}

bool ReadBytes(std::istream& in, void* out, std::size_t count) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(count));
  return static_cast<std::size_t>(in.gcount()) == count;
}

bool ReadU16(std::istream& in, std::uint16_t* value) {
  unsigned char bytes[2];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  *value = static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
  return true;
}

bool ReadU32(std::istream& in, std::uint32_t* value) {
  unsigned char bytes[4];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  *value = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
  return true;
}

bool ReadU64(std::istream& in, std::uint64_t* value) {
  unsigned char bytes[8];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  std::uint64_t result = 0;
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  *value = result;
  return true;
}

bool ReadI64(std::istream& in, std::int64_t* value) {
  std::uint64_t raw = 0;
  if (!ReadU64(in, &raw)) return false;
  *value = static_cast<std::int64_t>(raw);
  return true;
}

constexpr char kCheckpointMagic[8] = {'N', 'C', 'H', 'K', 'P', 'T', '0', '1'};

}  // namespace

// ---------------------------------------------------------------------------
// TokenShard
// ---------------------------------------------------------------------------

struct TokenShard::Impl {
  std::ifstream in;
  std::uint32_t width = 2;
};

TokenShard::TokenShard() = default;
TokenShard::~TokenShard() = default;

std::unique_ptr<TokenShard> TokenShard::Open(const std::string& path,
                                             std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<TokenShard> {
    if (error != nullptr) *error = message;
    return nullptr;
  };

  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open token shard: " + path);

  unsigned char raw[24];
  if (!ReadBytes(in, raw, sizeof(raw))) {
    return fail("token shard header is truncated: " + path);
  }
  auto u32 = [](const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
  };
  std::uint64_t num_tokens = 0;
  for (int i = 0; i < 8; ++i) {
    num_tokens |= static_cast<std::uint64_t>(raw[8 + i]) << (8 * i);
  }

  TokenShardHeader header;
  header.magic = u32(raw + 0);
  header.version = u32(raw + 4);
  header.num_tokens = num_tokens;
  header.token_width = u32(raw + 16);
  header.reserved = u32(raw + 20);

  if (header.magic != kTokenShardMagic) {
    return fail("bad token shard magic: " + path);
  }
  if (header.version != kTokenShardVersion) {
    return fail("unsupported token shard version: " + path);
  }
  if (header.token_width != static_cast<std::uint32_t>(TokenWidth::kUint16) &&
      header.token_width != static_cast<std::uint32_t>(TokenWidth::kUint32)) {
    return fail("unsupported token width: " + path);
  }

  // Reject a header that claims more tokens than the file can hold.
  in.clear();
  in.seekg(0, std::ios::end);
  const std::streamoff file_size = in.tellg();
  const std::uint64_t expected =
      static_cast<std::uint64_t>(24) +
      header.num_tokens * static_cast<std::uint64_t>(header.token_width);
  if (file_size < 0 || static_cast<std::uint64_t>(file_size) < expected) {
    return fail("token shard is shorter than its header claims: " + path);
  }
  in.seekg(24, std::ios::beg);

  // `TokenShard` has a private constructor, so std::make_unique cannot
  // reach it from this static factory. Wrap the raw allocation directly.
  auto shard = std::unique_ptr<TokenShard>(new TokenShard());
  shard->header_ = header;
  shard->impl_ = std::make_unique<Impl>();
  shard->impl_->in = std::move(in);
  shard->impl_->width = header.token_width;
  return shard;
}

std::int64_t TokenShard::Read(std::uint64_t offset, std::int64_t count,
                              int* out) {
  if (impl_ == nullptr || out == nullptr || count <= 0) return 0;
  if (offset >= header_.num_tokens) return 0;

  const std::uint64_t available = header_.num_tokens - offset;
  if (static_cast<std::uint64_t>(count) > available) {
    count = static_cast<std::int64_t>(available);
  }

  std::ifstream& in = impl_->in;
  in.clear();
  in.seekg(static_cast<std::streamoff>(
               24 + offset * static_cast<std::uint64_t>(impl_->width)),
           std::ios::beg);
  if (!in) return 0;

  const std::size_t bytes = static_cast<std::size_t>(count) * impl_->width;
  std::vector<unsigned char> buffer(bytes);
  in.read(reinterpret_cast<char*>(buffer.data()),
          static_cast<std::streamsize>(bytes));
  const std::int64_t produced = in.gcount() / impl_->width;

  for (std::int64_t i = 0; i < produced; ++i) {
    const unsigned char* p = buffer.data() + i * impl_->width;
    if (impl_->width == 2) {
      out[i] = static_cast<int>(p[0] | (p[1] << 8));
    } else {
      const std::uint32_t value = static_cast<std::uint32_t>(p[0]) |
                                  (static_cast<std::uint32_t>(p[1]) << 8) |
                                  (static_cast<std::uint32_t>(p[2]) << 16) |
                                  (static_cast<std::uint32_t>(p[3]) << 24);
      out[i] = static_cast<int>(value);
    }
  }
  return produced;
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

bool Checkpoint::Load(const std::string& path, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };

  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open checkpoint: " + path);

  char magic[8];
  if (!ReadBytes(in, magic, sizeof(magic)) ||
      std::memcmp(magic, kCheckpointMagic, sizeof(magic)) != 0) {
    return fail("bad checkpoint magic: " + path);
  }
  std::uint32_t version = 0;
  std::uint32_t count = 0;
  if (!ReadU32(in, &version) || !ReadU32(in, &count)) {
    return fail("truncated checkpoint header: " + path);
  }
  if (version != 1) return fail("unsupported checkpoint version: " + path);

  std::vector<TensorRecord> records;
  records.reserve(count);
  for (std::uint32_t r = 0; r < count; ++r) {
    TensorRecord record;
    std::uint16_t name_len = 0;
    if (!ReadU16(in, &name_len)) {
      return fail("truncated checkpoint record name: " + path);
    }
    record.name.resize(name_len);
    if (name_len > 0 && !ReadBytes(in, record.name.data(), name_len)) {
      return fail("truncated checkpoint record name: " + path);
    }
    std::uint32_t dtype = 0;
    std::uint32_t rank = 0;
    if (!ReadU32(in, &dtype) || !ReadU32(in, &rank)) {
      return fail("truncated checkpoint record header: " + path);
    }
    if (dtype > static_cast<std::uint32_t>(DType::kFp16)) {
      return fail("unknown checkpoint dtype: " + path);
    }
    if (rank > 8) return fail("checkpoint rank is too large: " + path);
    record.dtype = static_cast<DType>(dtype);
    record.shape.resize(rank);
    for (std::uint32_t d = 0; d < rank; ++d) {
      if (!ReadI64(in, &record.shape[d])) {
        return fail("truncated checkpoint shape: " + path);
      }
      if (record.shape[d] < 0) {
        return fail("negative checkpoint dimension: " + path);
      }
    }
    std::uint64_t data_bytes = 0;
    if (!ReadU64(in, &data_bytes)) {
      return fail("truncated checkpoint payload size: " + path);
    }
    record.data.resize(static_cast<std::size_t>(data_bytes));
    if (data_bytes > 0 && !ReadBytes(in, record.data.data(), data_bytes)) {
      return fail("truncated checkpoint payload: " + path);
    }
    records.push_back(std::move(record));
  }

  tensors_ = std::move(records);
  return true;
}

bool Checkpoint::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;

  out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
  WriteU32(out, 1);
  WriteU32(out, static_cast<std::uint32_t>(tensors_.size()));
  for (const TensorRecord& record : tensors_) {
    const std::size_t name_len = record.name.size();
    WriteU16(out, static_cast<std::uint16_t>(
                      std::min<std::size_t>(name_len, 0xffffu)));
    out.write(
        record.name.data(),
        static_cast<std::streamsize>(std::min<std::size_t>(name_len, 0xffffu)));
    WriteU32(out, static_cast<std::uint32_t>(record.dtype));
    WriteU32(out, static_cast<std::uint32_t>(record.shape.size()));
    for (std::int64_t dim : record.shape) WriteI64(out, dim);
    WriteU64(out, static_cast<std::uint64_t>(record.data.size()));
    if (!record.data.empty()) {
      out.write(reinterpret_cast<const char*>(record.data.data()),
                static_cast<std::streamsize>(record.data.size()));
    }
  }
  return static_cast<bool>(out);
}

const TensorRecord* Checkpoint::Find(std::string_view name) const {
  for (const TensorRecord& record : tensors_) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

TensorRecord* Checkpoint::Find(std::string_view name) {
  for (TensorRecord& record : tensors_) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

void Checkpoint::Add(TensorRecord record) {
  tensors_.push_back(std::move(record));
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

struct DataLoader::Impl {
  std::vector<std::unique_ptr<TokenShard>> shards;
  std::vector<std::uint64_t> offsets;  // size shards.size() + 1
  std::uint64_t total = 0;
  std::uint64_t cursor = 0;
  std::uint64_t emitted = 0;
  std::uint64_t max_batches = 0;
  std::uint64_t rows = 0;
  std::uint64_t seed = 42;
  bool shuffle = true;
  Rand rng;
  std::vector<std::uint8_t> token_bytes;
  int token_bytes_vocab = 0;

  // Optional sidecar with the per-token byte lengths (for bits-per-byte). The
  // format is a little-endian uint32 `vocab_size` followed by `vocab_size`
  // bytes. It lives next to the first shard as `<shard>.bytes`; when missing,
  // the loader reports no byte table and `EvalBpb` falls back to one byte per
  // token.
  void LoadTokenBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::uint32_t vocab = 0;
    if (!ReadU32(in, &vocab) || vocab == 0) return;
    std::vector<std::uint8_t> table(vocab);
    if (!ReadBytes(in, table.data(), vocab)) return;
    token_bytes = std::move(table);
    token_bytes_vocab = static_cast<int>(vocab);
  }

  bool ReadRange(std::uint64_t start, std::int64_t count, int* out) {
    if (count <= 0) return true;
    if (start + static_cast<std::uint64_t>(count) > total) return false;
    std::int64_t produced = 0;
    std::uint64_t cursor = start;
    for (std::size_t i = 0; i < shards.size() && produced < count; ++i) {
      const std::uint64_t shard_start = offsets[i];
      const std::uint64_t shard_end = offsets[i + 1];
      if (cursor >= shard_end) continue;
      const std::int64_t take = static_cast<std::int64_t>(
          std::min<std::uint64_t>(static_cast<std::uint64_t>(count - produced),
                                  shard_end - cursor));
      const std::int64_t got =
          shards[i]->Read(cursor - shard_start, take, out + produced);
      produced += got;
      cursor += static_cast<std::uint64_t>(got);
      if (got < take) break;
    }
    return produced == count;
  }
};

DataLoader::DataLoader(std::vector<std::string> shards, int batch, int seq,
                       std::uint64_t seed, bool shuffle)
    : impl_(std::make_unique<Impl>()), batch_(batch), seq_(seq) {
  impl_->shuffle = shuffle;
  impl_->seed = seed;
  impl_->rng.Seed(seed);
  impl_->offsets.push_back(0);
  for (const std::string& path : shards) {
    if (path.empty()) continue;
    std::unique_ptr<TokenShard> shard = TokenShard::Open(path);
    if (shard == nullptr) continue;
    impl_->total += shard->num_tokens();
    impl_->offsets.push_back(impl_->total);
    impl_->shards.push_back(std::move(shard));
  }
  if (batch_ > 0 && seq_ > 0) {
    impl_->rows =
        static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(seq_);
    if (impl_->rows > 0) {
      const std::uint64_t need = impl_->rows + 1;
      if (impl_->total >= need) impl_->max_batches = impl_->total / impl_->rows;
    }
  }
  if (!shards.empty()) {
    impl_->LoadTokenBytes(shards.front() + ".bytes");
  }
}

DataLoader::~DataLoader() = default;

bool DataLoader::Next(int* tokens, int* targets) {
  if (tokens == nullptr || targets == nullptr || impl_ == nullptr) return false;
  if (batch_ <= 0 || seq_ <= 0) return false;
  const std::uint64_t rows =
      static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(seq_);
  const std::uint64_t need = rows + 1;
  if (impl_->total < need) return false;

  std::uint64_t start = 0;
  if (impl_->shuffle) {
    if (impl_->emitted >= impl_->max_batches) return false;
    const std::uint64_t span = impl_->total - need + 1;
    start = span > 0 ? impl_->rng.NextU64() % span : 0;
    ++impl_->emitted;
  } else {
    if (impl_->cursor + need > impl_->total) return false;
    start = impl_->cursor;
    impl_->cursor += rows;
  }

  std::vector<int> buffer(static_cast<std::size_t>(need));
  if (!impl_->ReadRange(start, static_cast<std::int64_t>(need),
                        buffer.data())) {
    return false;
  }
  for (std::uint64_t i = 0; i < rows; ++i) {
    tokens[i] = buffer[static_cast<std::size_t>(i)];
    targets[i] = buffer[static_cast<std::size_t>(i + 1)];
  }
  return true;
}

void DataLoader::Reset() {
  if (impl_ == nullptr) return;
  impl_->cursor = 0;
  impl_->emitted = 0;
  impl_->rng.Seed(impl_->seed);
}

const std::uint8_t* DataLoader::token_bytes(int* vocab_size) const {
  if (impl_ == nullptr || impl_->token_bytes.empty()) {
    if (vocab_size != nullptr) *vocab_size = 0;
    return nullptr;
  }
  if (vocab_size != nullptr) *vocab_size = impl_->token_bytes_vocab;
  return impl_->token_bytes.data();
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

// The native tokenizer loader is implemented once, in the `src:tokenizer`
// library (src/tokenizer/tokenizer.cc). A target that calls `LoadTokenizer`
// links that library. The earlier null stub would shadow the real loader at
// link time, so this file deliberately defines no loader.

}  // namespace nanochat
