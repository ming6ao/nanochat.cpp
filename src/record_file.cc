#include "src/record_file.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>

namespace nanochat {
namespace {

constexpr std::uint8_t kMaxRank = 16;

bool CheckMagic(const std::vector<std::uint8_t>& bytes,
                const std::vector<std::string>& magics) {
  if (bytes.size() < 8) return false;
  for (const std::string& magic : magics) {
    if (magic.size() == 8 && std::memcmp(bytes.data(), magic.data(), 8) == 0) {
      return true;
    }
  }
  return false;
}

std::uint16_t ReadU16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t ReadU32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t ReadU64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
  return value;
}

void AppendU16(std::vector<std::uint8_t>* out, std::uint16_t value) {
  for (int shift = 0; shift < 16; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

void AppendU32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

void AppendU64(std::vector<std::uint8_t>* out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

Record MakeRecord(const std::string& name, RecordDType dtype,
                  std::vector<std::uint64_t> shape,
                  const std::vector<std::uint8_t>& payload) {
  Record record;
  record.name = name;
  record.dtype = dtype;
  record.shape = std::move(shape);
  record.data = payload;
  return record;
}

}  // namespace

std::size_t RecordElemSize(RecordDType dtype) {
  switch (dtype) {
    case RecordDType::kFp32:
    case RecordDType::kInt32:
      return 4;
    case RecordDType::kFp64:
    case RecordDType::kInt64:
      return 8;
    case RecordDType::kUInt8:
      return 1;
  }
  return 0;
}

std::int64_t Record::numel() const {
  std::int64_t total = 1;
  for (std::uint64_t extent : shape) {
    total *= static_cast<std::int64_t>(extent);
  }
  return total;
}

bool ParseRecordFile(const std::vector<std::uint8_t>& bytes,
                     const std::vector<std::string>& magics,
                     std::vector<Record>* records, std::string* error) {
  if (bytes.size() < 16) {
    *error = "truncated record file header";
    return false;
  }
  if (!CheckMagic(bytes, magics)) {
    *error = "bad record file magic";
    return false;
  }
  const std::uint32_t version = ReadU32(bytes.data() + 8);
  if (version != kRecordFileVersion) {
    *error = "unsupported record file version";
    return false;
  }
  const std::uint32_t count = ReadU32(bytes.data() + 12);

  records->clear();
  records->reserve(count);
  std::size_t offset = 16;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (offset + 2 > bytes.size()) {
      *error = "truncated record name length";
      return false;
    }
    const std::uint16_t name_len = ReadU16(bytes.data() + offset);
    offset += 2;
    if (offset + name_len + 2 > bytes.size()) {
      *error = "truncated record name";
      return false;
    }
    Record record;
    record.name.assign(reinterpret_cast<const char*>(bytes.data() + offset),
                       name_len);
    offset += name_len;
    record.dtype = static_cast<RecordDType>(bytes[offset]);
    const std::uint8_t rank = bytes[offset + 1];
    offset += 2;
    const std::size_t elem = RecordElemSize(record.dtype);
    if (elem == 0 || rank > kMaxRank) {
      *error = "malformed record " + record.name;
      return false;
    }
    record.shape.reserve(rank);
    for (std::uint8_t d = 0; d < rank; ++d) {
      if (offset + 8 > bytes.size()) {
        *error = "truncated record shape";
        return false;
      }
      record.shape.push_back(ReadU64(bytes.data() + offset));
      offset += 8;
    }
    const std::int64_t numel = record.numel();
    if (numel < 0) {
      *error = "negative record extent";
      return false;
    }
    const std::size_t payload = static_cast<std::size_t>(numel) * elem;
    if (offset + payload > bytes.size()) {
      *error = "truncated record payload for " + record.name;
      return false;
    }
    record.data.assign(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + payload));
    offset += payload;
    records->push_back(std::move(record));
  }
  if (offset != bytes.size()) {
    *error = "trailing bytes after the record file";
    return false;
  }
  return true;
}

bool LoadRecordFile(const std::string& path,
                    const std::vector<std::string>& magics,
                    std::vector<Record>* records, std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot open the record file " + path;
    return false;
  }
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
  return ParseRecordFile(bytes, magics, records, error);
}

bool SaveRecordFile(const std::string& path, const std::string& magic,
                    const std::vector<Record>& records, std::string* error) {
  if (magic.size() != 8) {
    *error = "the record file magic must be 8 bytes";
    return false;
  }
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), magic.begin(), magic.end());
  AppendU32(&bytes, kRecordFileVersion);
  AppendU32(&bytes, static_cast<std::uint32_t>(records.size()));
  for (const Record& record : records) {
    if (record.name.size() > 0xffff || record.shape.size() > kMaxRank) {
      *error = "record too large to serialise";
      return false;
    }
    const std::size_t elem = RecordElemSize(record.dtype);
    if (elem == 0 ||
        record.data.size() != static_cast<std::size_t>(record.numel()) * elem) {
      *error = "record " + record.name + " has a payload of the wrong size";
      return false;
    }
    AppendU16(&bytes, static_cast<std::uint16_t>(record.name.size()));
    bytes.insert(bytes.end(), record.name.begin(), record.name.end());
    bytes.push_back(static_cast<std::uint8_t>(record.dtype));
    bytes.push_back(static_cast<std::uint8_t>(record.shape.size()));
    for (std::uint64_t extent : record.shape) AppendU64(&bytes, extent);
    bytes.insert(bytes.end(), record.data.begin(), record.data.end());
  }
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    *error = "cannot write the record file " + path;
    return false;
  }
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    *error = "write failed for " + path;
    return false;
  }
  return true;
}

const Record* FindRecord(const std::vector<Record>& records,
                         const std::string& name) {
  for (const Record& record : records) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

std::int32_t RecordIntAt(const Record& record, std::size_t index) {
  std::int32_t value = 0;
  std::memcpy(&value, record.data.data() + index * sizeof(std::int32_t),
              sizeof(std::int32_t));
  return value;
}

float RecordFloatAt(const Record& record, std::size_t index) {
  float value = 0.0f;
  std::memcpy(&value, record.data.data() + index * sizeof(float),
              sizeof(float));
  return value;
}

Record MakeInt32Record(const std::string& name,
                       std::vector<std::uint64_t> shape,
                       std::vector<std::int32_t> values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(std::int32_t));
  for (std::int32_t value : values) {
    AppendU32(&payload, static_cast<std::uint32_t>(value));
  }
  return MakeRecord(name, RecordDType::kInt32, std::move(shape), payload);
}

Record MakeFloat32Record(const std::string& name,
                         std::vector<std::uint64_t> shape,
                         const std::vector<float>& values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(float));
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(&payload, bits);
  }
  return MakeRecord(name, RecordDType::kFp32, std::move(shape), payload);
}

bool ReadRecordInt(const std::vector<Record>& records, const std::string& name,
                   std::int32_t* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != RecordDType::kInt32 ||
      record->numel() != 1) {
    *error = "missing or malformed int record '" + name + "'";
    return false;
  }
  *out = RecordIntAt(*record, 0);
  return true;
}

bool ReadRecordFloat(const std::vector<Record>& records,
                     const std::string& name, float* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != RecordDType::kFp32 ||
      record->numel() != 1) {
    *error = "missing or malformed float record '" + name + "'";
    return false;
  }
  *out = RecordFloatAt(*record, 0);
  return true;
}

bool ReadRecordInt32Vector(const std::vector<Record>& records,
                           const std::string& name, std::int64_t expected,
                           std::vector<std::int32_t>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != RecordDType::kInt32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        RecordIntAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

bool ReadRecordFloat32Vector(const std::vector<Record>& records,
                             const std::string& name, std::int64_t expected,
                             std::vector<float>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != RecordDType::kFp32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        RecordFloatAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

}  // namespace nanochat
