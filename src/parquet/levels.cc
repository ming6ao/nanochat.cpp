#include "src/parquet/levels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nanochat {
namespace parquet {
namespace {

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

// Reads one unsigned LEB128 varint. Returns false on a short or overlong
// buffer.
bool ReadVarint(const std::uint8_t* data, std::size_t size, std::size_t* pos,
                std::uint64_t* value) {
  std::uint64_t result = 0;
  int shift = 0;
  while (true) {
    if (*pos >= size) return false;
    const std::uint8_t byte = data[(*pos)++];
    result |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
    if ((byte & 0x80u) == 0) break;
    shift += 7;
    if (shift > 63) return false;
  }
  *value = result;
  return true;
}

}  // namespace

bool DecodeRleBitPacked(const std::uint8_t* data, std::size_t size,
                        int bit_width, std::size_t count,
                        std::vector<std::uint32_t>* out, std::string* error) {
  if (out == nullptr) return Fail(error, "levels: null output");
  if (bit_width < 0 || bit_width > 32) {
    return Fail(error, "levels: bad bit width");
  }
  out->reserve(count);
  if (count == 0) return true;
  if (bit_width == 0) {
    out->assign(count, 0u);
    return true;
  }

  const std::uint32_t mask =
      bit_width == 32 ? 0xffffffffu : ((1u << bit_width) - 1u);
  const std::size_t byte_width = static_cast<std::size_t>((bit_width + 7) / 8);
  std::size_t pos = 0;
  std::size_t produced = 0;

  while (produced < count) {
    std::uint64_t header = 0;
    if (!ReadVarint(data, size, &pos, &header)) {
      return Fail(error, "levels: truncated run header");
    }
    if ((header & 1u) == 0) {
      // A repeated run: `header >> 1` copies of one value.
      const std::uint64_t run = header >> 1;
      if (pos + byte_width > size) {
        return Fail(error, "levels: truncated repeated run");
      }
      std::uint32_t value = 0;
      for (std::size_t i = 0; i < byte_width; ++i) {
        value |= static_cast<std::uint32_t>(data[pos + i]) << (8 * i);
      }
      pos += byte_width;
      value &= mask;
      for (std::uint64_t i = 0; i < run && produced < count; ++i) {
        out->push_back(value);
        ++produced;
      }
    } else {
      // A bit-packed run: `header >> 1` groups of eight values.
      const std::uint64_t groups = header >> 1;
      const std::uint64_t values = groups * 8;
      const std::size_t bytes =
          static_cast<std::size_t>((values * bit_width + 7) / 8);
      if (pos + bytes > size) {
        return Fail(error, "levels: truncated bit-packed run");
      }
      for (std::uint64_t i = 0; i < values && produced < count; ++i) {
        std::uint32_t raw = 0;
        for (int b = 0; b < bit_width; ++b) {
          const std::uint64_t bit =
              i * static_cast<std::uint64_t>(bit_width) + b;
          const std::size_t byte = pos + static_cast<std::size_t>(bit / 8);
          const int shift = static_cast<int>(bit % 8);
          if (byte >= size) {
            return Fail(error, "levels: truncated bit-packed value");
          }
          raw |= static_cast<std::uint32_t>((data[byte] >> shift) & 1u) << b;
        }
        out->push_back(raw);
        ++produced;
      }
      pos += bytes;
    }
  }
  return true;
}

}  // namespace parquet
}  // namespace nanochat
