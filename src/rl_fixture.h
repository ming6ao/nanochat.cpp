#ifndef NANOCHAT_SRC_RL_FIXTURE_H_
#define NANOCHAT_SRC_RL_FIXTURE_H_

// Reader (and writer) for the little-endian `NANOORC1` container that
// `tools/dump_rl_fixture.py` records and that the RL parity gate consumes
// (docs/rl-notebook.md section 4, docs/training-seam.md section 6.4).
//
// The persistent RL worker loads its model configuration, its optimizer and
// schedule configuration, and its initial parameters from the same fixture the
// file-driven `rl_step` binary reads, so the worker can be gated by the
// fixture-driven smoke run without the Python bridge. This header materialises
// the format once for `src/` so a production binary can link it without
// depending on `//tests` (the test copy is `tests/oracle_fixture.h`).
//
// The container is: magic[8] = "NANOORC1", uint32 version, uint32 record count,
// then per record a uint16 name length, the name bytes, a uint8 dtype, a uint8
// rank, `rank` little-endian uint64 extents, and the row-major payload.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nanochat {
namespace rl_fixture {

constexpr std::uint32_t kVersion = 1;

enum Dtype : std::uint8_t {
  kFp32 = 0,
  kInt32 = 1,
  kFp64 = 2,
  kInt64 = 3,
  kUInt8 = 4,
};

struct Record {
  std::string name;
  std::uint8_t dtype = kFp32;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> data;

  std::int64_t numel() const;
};

// Reads every record of `path`; returns false with a message in `*error` on a
// missing file, a bad magic, or a malformed record.
bool LoadRecords(const std::string& path, std::vector<Record>* records,
                 std::string* error);

// Writes `records` as one container; used by the worker's optional smoke
// result file.
bool SaveRecords(const std::string& path, const std::vector<Record>& records,
                 std::string* error);

const Record* FindRecord(const std::vector<Record>& records,
                         const std::string& name);

// Typed accessors. The caller has already checked the dtype and the count.
std::int32_t IntAt(const Record& record, std::size_t index);
float FloatAt(const Record& record, std::size_t index);

// Builders for the result file.
Record MakeInt32(const std::string& name, std::vector<std::uint64_t> shape,
                 std::vector<std::int32_t> values);
Record MakeFloat32(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::vector<float>& values);

// Scalar and vector readers that validate the dtype and the extent.
bool ScalarInt(const std::vector<Record>& records, const std::string& name,
               std::int32_t* out, std::string* error);
bool ScalarFloat(const std::vector<Record>& records, const std::string& name,
                 float* out, std::string* error);
bool ReadInt32Vector(const std::vector<Record>& records,
                     const std::string& name, std::int64_t expected,
                     std::vector<std::int32_t>* out, std::string* error);
bool ReadFloat32Vector(const std::vector<Record>& records,
                       const std::string& name, std::int64_t expected,
                       std::vector<float>* out, std::string* error);

}  // namespace rl_fixture
}  // namespace nanochat

#endif  // NANOCHAT_SRC_RL_FIXTURE_H_
