#include "src/parquet/thrift.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace nanochat {
namespace parquet {

CompactReader::CompactReader(const std::uint8_t* data, std::size_t size)
    : data_(data), size_(size) {}

void CompactReader::Fail(const std::string& message) {
  if (ok_) {
    ok_ = false;
    error_ = message;
  }
}

std::uint8_t CompactReader::ReadByte() {
  if (position_ >= size_) {
    Fail("thrift: truncated buffer");
    return 0;
  }
  return data_[position_++];
}

std::uint64_t CompactReader::ReadVarint() {
  std::uint64_t result = 0;
  int shift = 0;
  while (true) {
    if (position_ >= size_) {
      Fail("thrift: truncated varint");
      return 0;
    }
    const std::uint8_t byte = data_[position_++];
    result |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
    if ((byte & 0x80u) == 0) break;
    shift += 7;
    if (shift > 63) {
      Fail("thrift: varint is too long");
      return 0;
    }
  }
  return result;
}

std::int64_t CompactReader::ZigZag(std::uint64_t value) {
  return static_cast<std::int64_t>(value >> 1) ^
         -static_cast<std::int64_t>(value & 1u);
}

bool CompactReader::NextField(int* field_id, CompactType* type) {
  if (!ok_) return false;
  if (position_ >= size_) {
    Fail("thrift: truncated struct");
    return false;
  }
  const std::uint8_t header = data_[position_++];
  if (header == 0) {
    *type = CompactType::kStop;
    return false;
  }
  const int delta = (header >> 4) & 0x0f;
  *type = static_cast<CompactType>(header & 0x0f);
  int id = 0;
  if (delta == 0) {
    id = static_cast<int>(ZigZag(ReadVarint()));
    if (!ok_) return false;
  } else {
    id = last_field_id_ + delta;
  }
  last_field_id_ = id;
  *field_id = id;
  return true;
}

std::int64_t CompactReader::ReadInt() { return ZigZag(ReadVarint()); }

bool CompactReader::ReadBool(CompactType type) {
  if (type == CompactType::kBoolTrue) return true;
  if (type == CompactType::kBoolFalse) return false;
  Fail("thrift: expected a boolean");
  return false;
}

double CompactReader::ReadDouble() {
  if (position_ + 8 > size_) {
    Fail("thrift: truncated double");
    return 0.0;
  }
  std::uint64_t bits = 0;
  for (int i = 0; i < 8; ++i) {
    bits |= static_cast<std::uint64_t>(data_[position_ + i]) << (8 * i);
  }
  position_ += 8;
  double value = 0.0;
  static_assert(sizeof(value) == sizeof(bits), "double must be 8 bytes");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string_view CompactReader::ReadBinary() {
  const std::uint64_t length = ReadVarint();
  if (!ok_) return {};
  if (length > size_ - position_) {
    Fail("thrift: truncated binary");
    return {};
  }
  const char* start = reinterpret_cast<const char*>(data_ + position_);
  position_ += static_cast<std::size_t>(length);
  return std::string_view(start, static_cast<std::size_t>(length));
}

bool CompactReader::ReadListHeader(ListHeader* header) {
  if (!ok_) return false;
  const std::uint8_t byte = ReadByte();
  if (!ok_) return false;
  const int type = byte & 0x0f;
  std::size_t size = (byte >> 4) & 0x0f;
  if (size == 15) {
    size = static_cast<std::size_t>(ReadVarint());
    if (!ok_) return false;
  }
  header->type = static_cast<CompactType>(type);
  header->size = size;
  return true;
}

bool CompactReader::ReadMapHeader(MapHeader* header) {
  if (!ok_) return false;
  const std::uint64_t size = ReadVarint();
  if (!ok_) return false;
  header->size = static_cast<std::size_t>(size);
  if (header->size == 0) {
    header->key_type = CompactType::kStop;
    header->value_type = CompactType::kStop;
    return true;
  }
  const std::uint8_t types = ReadByte();
  if (!ok_) return false;
  header->key_type = static_cast<CompactType>((types >> 4) & 0x0f);
  header->value_type = static_cast<CompactType>(types & 0x0f);
  return true;
}

void CompactReader::Skip(CompactType type) {
  switch (type) {
    case CompactType::kBoolTrue:
    case CompactType::kBoolFalse:
    case CompactType::kStop:
      return;
    case CompactType::kByte:
      ReadByte();
      return;
    case CompactType::kI16:
    case CompactType::kI32:
    case CompactType::kI64:
      ReadVarint();
      return;
    case CompactType::kDouble:
      ReadDouble();
      return;
    case CompactType::kBinary:
      ReadBinary();
      return;
    case CompactType::kList:
    case CompactType::kSet: {
      ListHeader header;
      if (!ReadListHeader(&header)) return;
      for (std::size_t i = 0; i < header.size && ok_; ++i) Skip(header.type);
      return;
    }
    case CompactType::kMap: {
      MapHeader header;
      if (!ReadMapHeader(&header)) return;
      for (std::size_t i = 0; i < header.size && ok_; ++i) {
        Skip(header.key_type);
        Skip(header.value_type);
      }
      return;
    }
    case CompactType::kStruct:
      SkipStruct();
      return;
  }
  Fail("thrift: unknown type code");
}

void CompactReader::SkipStruct() {
  const int saved = last_field_id_;
  last_field_id_ = 0;
  while (ok_) {
    int field_id = 0;
    CompactType type = CompactType::kStop;
    if (!NextField(&field_id, &type)) break;
    Skip(type);
  }
  last_field_id_ = saved;
}

}  // namespace parquet
}  // namespace nanochat
