// The `NANOORC1` fixture container shared by the RL entry points
// (docs/rl-notebook.md section 4). See `src/rl_fixture.h` for the format.

#include "src/rl_fixture.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>

namespace nanochat {
namespace rl_fixture {

namespace {

constexpr char kMagic[8] = {'N', 'A', 'N', 'O', 'O', 'R', 'C', '1'};

std::size_t ElemSize(std::uint8_t dtype) {
  switch (dtype) {
    case kFp32:
    case kInt32:
      return 4;
    case kFp64:
    case kInt64:
      return 8;
    case kUInt8:
      return 1;
    default:
      return 0;
  }
}

// Little-endian byte cursor over an in-memory file image.
struct Reader {
  const std::vector<std::uint8_t>* bytes = nullptr;
  std::size_t offset = 0;
  bool ok = true;

  bool Need(std::size_t count) {
    if (!ok) return false;
    if (offset + count > bytes->size()) {
      ok = false;
      return false;
    }
    return true;
  }

  template <typename T>
  T Read() {
    T value{};
    if (!Need(sizeof(T))) return value;
    std::memcpy(&value, bytes->data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }
};

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

Record MakeRecord(const std::string& name, std::uint8_t dtype,
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

std::int64_t Record::numel() const {
  std::int64_t total = 1;
  for (std::uint64_t extent : shape) {
    total *= static_cast<std::int64_t>(extent);
  }
  return total;
}

bool LoadRecords(const std::string& path, std::vector<Record>* records,
                 std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot open fixture " + path;
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() < 16 ||
      std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
    *error = "bad fixture magic in " + path;
    return false;
  }
  Reader reader;
  reader.bytes = &bytes;
  reader.offset = 8;
  const std::uint32_t version = reader.Read<std::uint32_t>();
  const std::uint32_t count = reader.Read<std::uint32_t>();
  if (version != kVersion) {
    *error = "unsupported fixture version";
    return false;
  }
  records->clear();
  for (std::uint32_t i = 0; i < count && reader.ok; ++i) {
    Record record;
    const std::uint16_t name_len = reader.Read<std::uint16_t>();
    if (!reader.Need(name_len)) break;
    record.name.assign(
        reinterpret_cast<const char*>(bytes.data() + reader.offset), name_len);
    reader.offset += name_len;
    record.dtype = reader.Read<std::uint8_t>();
    const std::uint8_t ndim = reader.Read<std::uint8_t>();
    const std::size_t elem = ElemSize(record.dtype);
    if (elem == 0 || ndim > 16) {
      *error = "malformed record " + record.name;
      return false;
    }
    record.shape.reserve(ndim);
    for (std::uint8_t d = 0; d < ndim; ++d) {
      record.shape.push_back(reader.Read<std::uint64_t>());
    }
    const std::int64_t numel = record.numel();
    if (numel < 0) {
      *error = "negative record extent";
      return false;
    }
    const std::size_t payload = static_cast<std::size_t>(numel) * elem;
    if (!reader.Need(payload)) break;
    record.data.assign(bytes.begin() + reader.offset,
                       bytes.begin() + reader.offset + payload);
    reader.offset += payload;
    records->push_back(std::move(record));
  }
  if (!reader.ok) {
    *error = "truncated fixture " + path;
    return false;
  }
  return true;
}

bool SaveRecords(const std::string& path, const std::vector<Record>& records,
                 std::string* error) {
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), kMagic, kMagic + sizeof(kMagic));
  AppendU32(&bytes, kVersion);
  AppendU32(&bytes, static_cast<std::uint32_t>(records.size()));
  for (const Record& record : records) {
    if (record.name.size() > 0xffff || record.shape.size() > 16) {
      *error = "record too large to serialise";
      return false;
    }
    AppendU16(&bytes, static_cast<std::uint16_t>(record.name.size()));
    bytes.insert(bytes.end(), record.name.begin(), record.name.end());
    bytes.push_back(record.dtype);
    bytes.push_back(static_cast<std::uint8_t>(record.shape.size()));
    for (std::uint64_t extent : record.shape) AppendU64(&bytes, extent);
    bytes.insert(bytes.end(), record.data.begin(), record.data.end());
  }
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    *error = "cannot write fixture " + path;
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

std::int32_t IntAt(const Record& record, std::size_t index) {
  std::int32_t value = 0;
  std::memcpy(&value, record.data.data() + index * sizeof(std::int32_t),
              sizeof(std::int32_t));
  return value;
}

float FloatAt(const Record& record, std::size_t index) {
  float value = 0.0f;
  std::memcpy(&value, record.data.data() + index * sizeof(float),
              sizeof(float));
  return value;
}

Record MakeInt32(const std::string& name, std::vector<std::uint64_t> shape,
                 std::vector<std::int32_t> values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(std::int32_t));
  for (std::int32_t value : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
      payload.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xff));
    }
  }
  return MakeRecord(name, kInt32, std::move(shape), payload);
}

Record MakeFloat32(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::vector<float>& values) {
  std::vector<std::uint8_t> payload;
  payload.reserve(values.size() * sizeof(float));
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8) {
      payload.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xff));
    }
  }
  return MakeRecord(name, kFp32, std::move(shape), payload);
}

bool ScalarInt(const std::vector<Record>& records, const std::string& name,
               std::int32_t* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kInt32 || record->numel() != 1) {
    *error = "missing or malformed int record '" + name + "'";
    return false;
  }
  *out = IntAt(*record, 0);
  return true;
}

bool ScalarFloat(const std::vector<Record>& records, const std::string& name,
                 float* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kFp32 || record->numel() != 1) {
    *error = "missing or malformed float record '" + name + "'";
    return false;
  }
  *out = FloatAt(*record, 0);
  return true;
}

bool ReadInt32Vector(const std::vector<Record>& records,
                     const std::string& name, std::int64_t expected,
                     std::vector<std::int32_t>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kInt32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        IntAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

bool ReadFloat32Vector(const std::vector<Record>& records,
                       const std::string& name, std::int64_t expected,
                       std::vector<float>* out, std::string* error) {
  const Record* record = FindRecord(records, name);
  if (record == nullptr || record->dtype != kFp32 ||
      record->numel() != expected) {
    *error = "record '" + name + "' has the wrong dtype or shape";
    return false;
  }
  out->resize(static_cast<std::size_t>(expected));
  for (std::int64_t i = 0; i < expected; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        FloatAt(*record, static_cast<std::size_t>(i));
  }
  return true;
}

}  // namespace rl_fixture
}  // namespace nanochat
