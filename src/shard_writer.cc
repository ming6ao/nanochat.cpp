// The `NANO` shard writer and the `<shard>.bytes` sidecar writer
// (docs/tokenizer.md section 10). The format matches `TokenShard::Open` and
// `DataLoader::Impl::LoadTokenBytes` in `src/data.cc`.

#include "src/shard_writer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/data.h"

namespace nanochat {
namespace {

constexpr std::size_t kHeaderSize = 24;
constexpr std::size_t kFlushThreshold = 1u << 20;

void AppendU16(std::vector<std::uint8_t>* out, std::uint16_t value) {
  out->push_back(static_cast<std::uint8_t>(value & 0xffu));
  out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

void WriteHeader(std::vector<std::uint8_t>* out, std::uint64_t num_tokens,
                 std::uint32_t width) {
  out->clear();
  AppendU32(out, kTokenShardMagic);
  AppendU32(out, kTokenShardVersion);
  AppendU64(out, num_tokens);
  AppendU32(out, width);
  AppendU32(out, 0);
}

}  // namespace

struct ShardWriter::Impl {
  std::ofstream out;
  int width = 2;
  std::vector<std::uint8_t> buffer;

  bool Flush(std::string* error) {
    if (buffer.empty()) return true;
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    if (!out.good()) {
      if (error != nullptr) *error = "shard writer: write failed";
      return false;
    }
    buffer.clear();
    return true;
  }
};

ShardWriter::ShardWriter() = default;
ShardWriter::~ShardWriter() = default;

std::unique_ptr<ShardWriter> ShardWriter::Open(const std::string& path,
                                               int width, std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<ShardWriter> {
    if (error != nullptr) *error = message;
    return nullptr;
  };
  if (width != 2 && width != 4) {
    return fail("shard writer: width must be 2 or 4");
  }
  // `ShardWriter` has a private constructor, so `std::make_unique` cannot
  // reach it. Wrap the raw allocation directly.
  std::unique_ptr<ShardWriter> writer(new ShardWriter());
  writer->impl_ = std::make_unique<Impl>();
  writer->impl_->width = width;
  writer->impl_->out.open(path, std::ios::binary | std::ios::trunc);
  if (!writer->impl_->out) {
    return fail("shard writer: cannot open " + path);
  }
  // A placeholder header; `Close` rewrites it with the final token count.
  WriteHeader(&writer->impl_->buffer, 0, static_cast<std::uint32_t>(width));
  if (!writer->impl_->Flush(error)) return nullptr;
  return writer;
}

bool ShardWriter::Write(const int* tokens, std::int64_t count,
                        std::string* error) {
  if (impl_ == nullptr) {
    if (error != nullptr) *error = "shard writer: writer is closed";
    return false;
  }
  if (tokens == nullptr || count <= 0) return true;
  const std::uint64_t limit =
      impl_->width == 2 ? 0x10000u
                        : static_cast<std::uint64_t>(
                              std::numeric_limits<std::uint32_t>::max()) +
                              1u;
  for (std::int64_t index = 0; index < count; ++index) {
    const int token = tokens[index];
    if (token < 0 || static_cast<std::uint64_t>(token) >= limit) {
      if (error != nullptr) *error = "shard writer: token id out of range";
      return false;
    }
    if (impl_->width == 2) {
      AppendU16(&impl_->buffer, static_cast<std::uint16_t>(token));
    } else {
      AppendU32(&impl_->buffer, static_cast<std::uint32_t>(token));
    }
    if (impl_->buffer.size() >= kFlushThreshold) {
      if (!impl_->Flush(error)) return false;
    }
  }
  num_tokens_ += static_cast<std::uint64_t>(count);
  return true;
}

bool ShardWriter::Close(std::string* error) {
  if (impl_ == nullptr) return true;
  if (!impl_->Flush(error)) return false;
  std::vector<std::uint8_t> header;
  WriteHeader(&header, num_tokens_, static_cast<std::uint32_t>(impl_->width));
  impl_->out.seekp(0, std::ios::beg);
  impl_->out.write(reinterpret_cast<const char*>(header.data()),
                   static_cast<std::streamsize>(header.size()));
  impl_->out.flush();
  if (!impl_->out.good()) {
    if (error != nullptr) *error = "shard writer: final header write failed";
    return false;
  }
  impl_->out.close();
  return true;
}

bool WriteTokenBytes(const std::string& shard_path,
                     const std::uint8_t* token_bytes, int vocab_size,
                     std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (token_bytes == nullptr || vocab_size <= 0) {
    return fail("shard writer: empty token byte table");
  }
  std::ofstream out(shard_path + ".bytes", std::ios::binary | std::ios::trunc);
  if (!out) return fail("shard writer: cannot open " + shard_path + ".bytes");
  std::vector<std::uint8_t> header;
  AppendU32(&header, static_cast<std::uint32_t>(vocab_size));
  out.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(token_bytes),
            static_cast<std::streamsize>(vocab_size));
  out.flush();
  if (!out.good()) return fail("shard writer: sidecar write failed");
  return true;
}

}  // namespace nanochat
