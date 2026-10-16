#ifndef NANOCHAT_SRC_RL_FIXTURE_H_
#define NANOCHAT_SRC_RL_FIXTURE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "src/record_file.h"

// The RL fixture is the typed-record file under the `NANOORC1`
// magic. It holds the model configuration, the optimizer and schedule
// configuration, the initial parameters, and, per step, one recorded rollout
// (docs/rl-notebook.md section 4, docs/training-seam.md section 6.4).
//
// The layout lives in `src/record_file.{h,cc}`, so the file-driven `rl_step`
// binary, the persistent worker, and the oracle tests share one reader. This
// header keeps the RL names as thin wrappers.

namespace nanochat {
namespace rl_fixture {

using Dtype = RecordDType;
using Record = nanochat::Record;

constexpr std::uint32_t kVersion = kRecordFileVersion;
constexpr Dtype kFp32 = RecordDType::kFp32;
constexpr Dtype kInt32 = RecordDType::kInt32;
constexpr Dtype kFp64 = RecordDType::kFp64;
constexpr Dtype kInt64 = RecordDType::kInt64;
constexpr Dtype kUInt8 = RecordDType::kUInt8;

// The 8-byte magic of the RL fixture.
inline constexpr const char* kMagic = "NANOORC1";

// Reads every record of `path`; returns false with a message in `*error` on a
// missing file, a bad magic, or a malformed record.
inline bool LoadRecords(const std::string& path, std::vector<Record>* records,
                        std::string* error) {
  return LoadRecordFile(path, {kMagic}, records, error);
}

// Writes `records` as one record file; used by the worker's optional smoke
// result file.
inline bool SaveRecords(const std::string& path,
                        const std::vector<Record>& records,
                        std::string* error) {
  return SaveRecordFile(path, kMagic, records, error);
}

using nanochat::FindRecord;

// Typed accessors. The caller has already checked the dtype and the count.
inline std::int32_t IntAt(const Record& record, std::size_t index) {
  return RecordIntAt(record, index);
}

inline float FloatAt(const Record& record, std::size_t index) {
  return RecordFloatAt(record, index);
}

// Builders for the result file.
inline Record MakeInt32(const std::string& name,
                        std::vector<std::uint64_t> shape,
                        std::vector<std::int32_t> values) {
  return MakeInt32Record(name, std::move(shape), std::move(values));
}

inline Record MakeFloat32(const std::string& name,
                          std::vector<std::uint64_t> shape,
                          const std::vector<float>& values) {
  return MakeFloat32Record(name, std::move(shape), values);
}

// Scalar and vector readers that validate the dtype and the extent.
inline bool ScalarInt(const std::vector<Record>& records,
                      const std::string& name, std::int32_t* out,
                      std::string* error) {
  return ReadRecordInt(records, name, out, error);
}

inline bool ScalarFloat(const std::vector<Record>& records,
                        const std::string& name, float* out,
                        std::string* error) {
  return ReadRecordFloat(records, name, out, error);
}

inline bool ReadInt32Vector(const std::vector<Record>& records,
                            const std::string& name, std::int64_t expected,
                            std::vector<std::int32_t>* out,
                            std::string* error) {
  return ReadRecordInt32Vector(records, name, expected, out, error);
}

inline bool ReadFloat32Vector(const std::vector<Record>& records,
                              const std::string& name, std::int64_t expected,
                              std::vector<float>* out, std::string* error) {
  return ReadRecordFloat32Vector(records, name, expected, out, error);
}

}  // namespace rl_fixture
}  // namespace nanochat

#endif  // NANOCHAT_SRC_RL_FIXTURE_H_
