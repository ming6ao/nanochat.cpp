#ifndef NANOCHAT_SRC_CONTAINER_H_
#define NANOCHAT_SRC_CONTAINER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The little-endian typed-record file. One layout backs the CPU oracle
// fixture, the evaluation fixture, the tokenizer fixture, the numeric trace,
// and the RL fixture. Python writes each file and this reader owns the layout,
// so a test and a production binary agree by construction.
//
// Layout, little-endian:
//   magic    8 bytes  caller-selected, for example "NANOORC1"
//   version  u32      1
//   count    u32      the record count
//   then `count` records:
//     name_len u16, name bytes
//     dtype    u8
//     rank     u8
//     dims     u64[rank]
//     payload  numel * element_size bytes
//
// The payload size is implicit: `numel` is the product of `dims`, and the
// element size comes from `dtype`. A record has no attribute block.

namespace nanochat {

enum class RecordDType : std::uint8_t {
  kFp32 = 0,
  kInt32 = 1,
  kFp64 = 2,
  kInt64 = 3,
  kUInt8 = 4,
};

// The record file version this library reads and writes.
inline constexpr std::uint32_t kRecordFileVersion = 1;

// The element size of `dtype` in bytes, or 0 for an unknown code.
std::size_t RecordElemSize(RecordDType dtype);

struct Record {
  std::string name;
  RecordDType dtype = RecordDType::kFp32;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> data;

  std::int64_t numel() const;
};

// Parses a record file already in memory. `magics` lists the accepted 8-byte
// magics. Returns false and writes a reason to `*error`.
bool ParseRecordFile(const std::vector<std::uint8_t>& bytes,
                     const std::vector<std::string>& magics,
                     std::vector<Record>* records, std::string* error);

// Reads a record file from `path`. Returns false and writes a reason to
// `*error`.
bool LoadRecordFile(const std::string& path,
                    const std::vector<std::string>& magics,
                    std::vector<Record>* records, std::string* error);

// Writes `records` to `path` under `magic`. Returns false and writes a reason
// to `*error`.
bool SaveRecordFile(const std::string& path, const std::string& magic,
                    const std::vector<Record>& records, std::string* error);

// Finds one record by name, or null.
const Record* FindRecord(const std::vector<Record>& records,
                         const std::string& name);

// Typed element access. The caller has already checked the dtype and the size.
std::int32_t RecordIntAt(const Record& record, std::size_t index);
float RecordFloatAt(const Record& record, std::size_t index);

// Builders for one record.
Record MakeInt32Record(const std::string& name,
                       std::vector<std::uint64_t> shape,
                       std::vector<std::int32_t> values);
Record MakeFloat32Record(const std::string& name,
                         std::vector<std::uint64_t> shape,
                         const std::vector<float>& values);

// Validating readers. Each returns false and writes a reason to `*error` on a
// missing record, a bad dtype, or a bad size.
bool ReadRecordInt(const std::vector<Record>& records, const std::string& name,
                   std::int32_t* out, std::string* error);
bool ReadRecordFloat(const std::vector<Record>& records,
                     const std::string& name, float* out, std::string* error);
bool ReadRecordInt32Vector(const std::vector<Record>& records,
                           const std::string& name, std::int64_t expected,
                           std::vector<std::int32_t>* out, std::string* error);
bool ReadRecordFloat32Vector(const std::vector<Record>& records,
                             const std::string& name, std::int64_t expected,
                             std::vector<float>* out, std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_CONTAINER_H_
