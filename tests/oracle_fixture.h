#ifndef NANOCHAT_TESTS_ORACLE_FIXTURE_H_
#define NANOCHAT_TESTS_ORACLE_FIXTURE_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/record_file.h"

// Reader for the CPU oracle fixture, `tests/data/debug_state.bin`, produced
// offline by `tools/dump_fixtures.py oracle`. The fixture is data, so
// this reader has no torch dependency and no dependency on the model: any test
// (CPU, CUDA, or the model skeleton) can parse the same reference tensors.
//
// The same little-endian record encoding backs the evaluation fixture
// (`NANOEVL1`, written by `python/nanochat_cpp/eval_fixture.py` and
// `tools/dump_fixtures.py eval`), the tokenizer fixture (`NANOTOK1`,
// written by `tools/dump_fixtures.py tokenizer`), and the staged numeric
// trace
// (`NANONUM1`, written by `tests/numerics_trace_emit_main.cc` and
// `tools/dump_fixtures.py numerics`); `Parse` accepts all four magics
// so each parity test can reuse this reader. The numeric trace is flat by
// design -- the phase, the step, and the parameter name go into the record
// name, for example `train/step3/param_hash/transformer.h.0.attn.c_q.weight` --
// so it needs no record sequence and no reader change beyond the magic. See the
// module docstring in tools/dump_fixtures.py oracle for the exact
// layout. This header is header-only on purpose: it can be included directly,
// or linked through the
// `//tests:oracle_fixture` library.
//
// Two things travel with the bytes besides the records (`models/<name>/` and
// the bring-up's section 15):
//
//   * the **format version** the bytes were written under. The reader accepts
//     the current version and, once one has been written, the previous one, the
//     `NCHKPT01` policy of `spec/versions.md` sections 2 and 5. Writers write
//     the current version only; there is no compatibility writer and no
//     version-selecting flag. This campaign writes no new fixture byte, so the
//     acceptance set is still `{1}` and the previous number is the `0` sentinel
//     ("no earlier number was ever written"). The first bump makes 1 the
//     previous version and the set `{2, 1}` with no reader change.
//   * the **origin** of a fixture: the model package that owns it and the file
//     inside that package. A model's fixtures live in
//     `models/<name>/fixtures/` and the harness data stays in `tests/data/`;
//     the two coexist. `FixtureSet` below is the lookup that starts from the
//     model package, so no caller can resolve a fixture from a shape, a
//     record count, or a dtype: two models may share every shape and declare
//     different arithmetic, so a shape is not an identity
//     (`spec/versions.md` section 3.3).

namespace nanochat {
namespace oracle {

// ---------------------------------------------------------------------------
// The fixture format version
// ---------------------------------------------------------------------------

// The record-file version this reader accepts as the current one. It is the
// writer's `kRecordFileVersion` from `src/record_file.h`; the two constants are
// separate on purpose, because that header owns the layout and this header owns
// the acceptance set a *reader* applies to it (`spec/versions.md` section 1,
// the fixture-format row).
inline constexpr std::uint32_t kFixtureFormatVersion = 1;

// The previous version accepted beside the current one. `0` is the sentinel
// `spec/versions.md` section 2 records for fixture format 1: version 1 is the
// only number ever written, so there is no earlier version to accept and the
// acceptance set is exactly `{1}`. A bump sets this to the old current value
// and the set becomes `{2, 1}`; a removal one release later moves it again.
inline constexpr std::uint32_t kFixtureFormatPreviousVersion = 0;

// The acceptance set, current first. `{kFixtureFormatVersion}` today.
inline std::vector<std::uint32_t> AcceptedFixtureFormatVersions() {
  std::vector<std::uint32_t> accepted{kFixtureFormatVersion};
  if (kFixtureFormatPreviousVersion != 0 &&
      kFixtureFormatPreviousVersion != kFixtureFormatVersion) {
    accepted.push_back(kFixtureFormatPreviousVersion);
  }
  return accepted;
}

// The current version this reader accepts is the version the one writer in the
// tree writes. A bump that moves one number and not the other is a compile
// error rather than a reader that accepts what nothing writes.
static_assert(kFixtureFormatVersion == nanochat::kRecordFileVersion,
              "the fixture-format reader accepts the record-file writer's "
              "current version (spec/versions.md section 2)");

// True when `version` is in the acceptance set. Writes a refusal that names the
// version and the set this reader is written against, so an unknown number is
// refused by number rather than rounded to the current one
// (`spec/versions.md` section 3.1). `*error` is cleared on success.
inline bool AcceptFixtureVersion(std::uint32_t version,
                                 std::string* error = nullptr) {
  if (error != nullptr) error->clear();
  for (std::uint32_t accepted : AcceptedFixtureFormatVersions()) {
    if (version == accepted) return true;
  }
  if (error != nullptr) {
    std::string set;
    for (std::uint32_t accepted : AcceptedFixtureFormatVersions()) {
      if (!set.empty()) set += ", ";
      set += std::to_string(accepted);
    }
    *error = "fixture format version " + std::to_string(version) +
             " is not in this reader's acceptance set {" + set +
             "}; a reader accepts the current version and the previous one "
             "(spec/versions.md section 2)";
  }
  return false;
}

// Where a fixture came from: the model package that owns the bytes and the file
// inside it.
//
// An empty `model` means "harness data": a file under `tests/data/` that no
// model package owns (a tokenizer, a checkpoint, a plan fixture). The origin is
// printed with every read error, so a failing comparison names the model and
// the file instead of a bare path.
struct FixtureOrigin {
  std::string
      model;         // "nanochat-default", or empty for tests/data harness data
  std::string file;  // "fixtures/debug_state.bin", or the path as given

  std::string Describe() const {
    if (model.empty()) return file.empty() ? std::string("(unkeyed)") : file;
    if (file.empty()) return model;
    return model + "/" + file;
  }
};

enum class DType : std::uint8_t {
  kFp32 = 0,
  kInt32 = 1,
  kFp64 = 2,
  kInt64 = 3,
  kUInt8 = 4,
};

inline std::size_t ElemSize(DType dtype) {
  switch (dtype) {
    case DType::kFp32:
    case DType::kInt32:
      return 4;
    case DType::kFp64:
    case DType::kInt64:
      return 8;
    case DType::kUInt8:
      return 1;
  }
  throw std::runtime_error("oracle: unknown dtype code");
}

// A non-owning view of one record. `data` points into the owning Fixture's
// payload storage, which stays alive for the Fixture's lifetime.
struct Tensor {
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t bytes = 0;

  std::int64_t numel() const {
    std::int64_t count = 1;
    for (std::int64_t extent : shape) count *= extent;
    return count;
  }

  bool is_fp32() const { return dtype == DType::kFp32; }
  bool is_int32() const { return dtype == DType::kInt32; }

  const float* f32() const {
    if (dtype != DType::kFp32)
      throw std::runtime_error("oracle: record is not fp32");
    return reinterpret_cast<const float*>(data);
  }

  const std::int32_t* i32() const {
    if (dtype != DType::kInt32)
      throw std::runtime_error("oracle: record is not int32");
    return reinterpret_cast<const std::int32_t*>(data);
  }

  const std::int64_t* i64() const {
    if (dtype != DType::kInt64)
      throw std::runtime_error("oracle: record is not int64");
    return reinterpret_cast<const std::int64_t*>(data);
  }

  float scalar_f32() const {
    if (numel() != 1)
      throw std::runtime_error("oracle: record is not a scalar");
    return f32()[0];
  }

  std::int64_t scalar_int() const {
    if (numel() != 1)
      throw std::runtime_error("oracle: record is not a scalar");
    if (dtype == DType::kInt32) return i32()[0];
    if (dtype == DType::kInt64) return i64()[0];
    throw std::runtime_error("oracle: integer scalar has an unexpected dtype");
  }
};

namespace detail {

inline std::uint16_t ReadU16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
}

inline std::uint32_t ReadU32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t ReadU64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
  return value;
}

}  // namespace detail

class Fixture {
 public:
  static Fixture Load(const std::string& path) {
    return Load(FixtureOrigin{std::string(), path}, path);
  }

  // The same read, with the fixture's identity attached. A caller that reached
  // the bytes through a model package passes the model name here, so every
  // error names `model/file` rather than a path that means nothing outside this
  // working directory.
  static Fixture Load(const FixtureOrigin& origin, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("oracle: cannot open fixture '" +
                               origin.Describe() + "' at " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    if (bytes.empty())
      throw std::runtime_error("oracle: fixture is empty: " +
                               origin.Describe());
    return Parse(std::move(bytes), origin);
  }

  static Fixture Parse(std::vector<std::uint8_t> bytes,
                       const FixtureOrigin& origin = FixtureOrigin()) {
    Fixture fixture;
    fixture.origin_ = origin;
    // The magic is the container's name and the version word describes the
    // bytes inside it; neither is a shape, and both are carried so a caller can
    // print them.
    if (bytes.size() >= 8) {
      fixture.magic_.assign(reinterpret_cast<const char*>(bytes.data()), 8);
    }
    if (bytes.size() >= 12) {
      const std::uint32_t version = detail::ReadU32(bytes.data() + 8);
      std::string version_error;
      if (!AcceptFixtureVersion(version, &version_error)) {
        throw std::runtime_error("oracle: " + version_error + " (" +
                                 origin.Describe() + ")");
      }
      fixture.format_version_ = version;
    }
    std::vector<Record> records;
    std::string error;
    if (!ParseRecordFile(bytes,
                         {"NANOORC1", "NANOEVL1", "NANOTOK1", "NANONUM1"},
                         &records, &error)) {
      throw std::runtime_error("oracle: " + error + " (" + origin.Describe() +
                               ")");
    }
    fixture.payloads_.reserve(records.size());
    for (Record& record : records) {
      fixture.payloads_.push_back(std::move(record.data));
      Tensor tensor;
      tensor.dtype = static_cast<DType>(record.dtype);
      tensor.shape.reserve(record.shape.size());
      for (std::uint64_t extent : record.shape) {
        tensor.shape.push_back(static_cast<std::int64_t>(extent));
      }
      tensor.data = fixture.payloads_.back().data();
      tensor.bytes = fixture.payloads_.back().size();
      const std::string name = record.name;
      fixture.order_.push_back(name);
      fixture.tensors_.emplace(name, std::move(tensor));
    }
    return fixture;
  }

  bool Has(const std::string& name) const { return tensors_.count(name) != 0; }

  const Tensor& Get(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      throw std::runtime_error("oracle: missing record '" + name + "'");
    }
    return it->second;
  }

  const std::vector<std::string>& names() const { return order_; }
  std::size_t size() const { return tensors_.size(); }

  // The eight-byte container magic the bytes carried ("NANOORC1" and so on),
  // empty for a container too short to hold one.
  const std::string& magic() const { return magic_; }

  // The version word the bytes declared after the acceptance check, or 0 for a
  // header too short to hold one. It is never a number outside
  // `AcceptedFixtureFormatVersions()`, because `Parse` throws first.
  std::uint32_t format_version() const { return format_version_; }

  // The model package that owns the bytes, and the file inside it. Unkeyed for
  // a `Parse` of in-memory bytes; `FixtureSet::Load` always sets it.
  const FixtureOrigin& origin() const { return origin_; }

 private:
  std::vector<std::vector<std::uint8_t>> payloads_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::vector<std::string> order_;
  std::string magic_;
  std::uint32_t format_version_ = 0;
  FixtureOrigin origin_;
};

// ---------------------------------------------------------------------------
// Resolving a model package's fixtures
// ---------------------------------------------------------------------------

inline bool FixtureFileExists(const std::string& path) {
  std::ifstream in(path);
  return in.good();
}

// A path candidate list for a repository-relative path: the Bazel runfiles
// tree first, then the tree relative to the working directory and up to four
// parents above it. The same order `src/device_profile.cc` uses for
// `backends/*/caps.json`, and for the same reason: a test and a simulated run
// read one set of bytes.
inline std::vector<std::string> FixturePathCandidates(
    const std::string& relative) {
  std::vector<std::string> candidates;
  const auto env = [](const char* name) -> const char* {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
  };
  const std::string src_dir = env("TEST_SRCDIR");
  if (!src_dir.empty()) {
    const std::string workspace = env("TEST_WORKSPACE");
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace + "/" + relative);
    }
    candidates.push_back(src_dir + "/_main/" + relative);
  }
  candidates.push_back(relative);
  std::string prefix;
  for (int level = 0; level < 4; ++level) {
    prefix += "../";
    candidates.push_back(prefix + relative);
  }
  return candidates;
}

// The directory `models/<model>`, resolved as a *model package*: a directory
// counts only when it carries a `spec.json`, which is the package's identity.
// Returns an empty string when no candidate is a model package, so a caller
// reports a missing model instead of reading a directory that merely exists.
inline std::string FindModelPackageDirectory(const std::string& model) {
  if (model.empty()) return std::string();
  const std::string relative = "models/" + model;
  for (const std::string& candidate : FixturePathCandidates(relative)) {
    if (FixtureFileExists(candidate + "/spec.json")) return candidate;
  }
  return std::string();
}

// The directory `models/<model>/fixtures`, resolved the same way. The marker
// file is the manifest, because a fixture directory without one cannot prove
// which bytes belong to the model.
inline std::string FindModelFixturesDirectory(const std::string& model) {
  const std::string package = FindModelPackageDirectory(model);
  if (package.empty()) return std::string();
  const std::string fixtures = package + "/fixtures";
  if (!FixtureFileExists(fixtures + "/manifest.json")) return std::string();
  return fixtures;
}

// One model package's fixture set, resolved from the model directory.
//
// The key of a lookup is `(model, file, record)`: the model that owns the
// bytes, the file inside its `fixtures/` directory, and the record name inside
// that file. There is deliberately no overload that takes a shape, a dtype, a
// record count, or a lone record name: a fixture is never keyed by shape alone,
// because two models can share every shape and mean different arithmetic
// (`spec/versions.md` section 3.3). The model name travels into every Fixture
// this set loads, so a failing comparison names the model and the file.
//
// The set caches each file it reads, so a test that compares two records of the
// same fixture reads the bytes once, and a second model's set is a second
// instance: nothing is shared by shape.
class FixtureSet {
 public:
  // The fixture set of `model`. Throws when the package has no fixtures
  // directory, so a typo is a named error rather than an empty comparison that
  // passes.
  static FixtureSet ForModel(const std::string& model) {
    const std::string directory = FindModelFixturesDirectory(model);
    if (directory.empty()) {
      throw std::runtime_error(
          "oracle: model package '" + model +
          "' has no fixtures directory (models/" + model +
          "/fixtures with a manifest.json); a fixture is resolved from its "
          "model package, never from a shape");
    }
    return FixtureSet(model, directory);
  }

  // The fixture set at an explicit directory, for a package a test fabricated
  // in a temporary directory. The model name is still required: it is the key.
  FixtureSet(std::string model, std::string directory)
      : model_(std::move(model)), directory_(std::move(directory)) {}

  const std::string& model() const { return model_; }
  const std::string& directory() const { return directory_; }

  // The path of one file inside the package. A `..` component is refused: the
  // set never reads outside the model it was built for.
  std::string Path(const std::string& file) const {
    if (file.empty()) {
      throw std::runtime_error("oracle: empty fixture file for model '" +
                               model_ + "'");
    }
    if (file == ".." || file.compare(0, 3, "../") == 0 ||
        file.find("/../") != std::string::npos) {
      throw std::runtime_error("oracle: fixture '" + file +
                               "' escapes the model package '" + model_ + "'");
    }
    return directory_ + "/" + file;
  }

  bool Has(const std::string& file) const {
    return FixtureFileExists(Path(file));
  }

  // Reads (or returns the cached) fixture `file`, with the origin set.
  const Fixture& Load(const std::string& file) const {
    const auto found = fixtures_.find(file);
    if (found != fixtures_.end()) return found->second;
    Fixture fixture = Fixture::Load(FixtureOrigin{model_, file}, Path(file));
    const auto inserted = fixtures_.emplace(file, std::move(fixture));
    return inserted.first->second;
  }

  // `(model, file, record)` in one call: the only record lookup this header
  // offers, and it cannot be reached without naming the model's file.
  const Tensor& Record(const std::string& file,
                       const std::string& record) const {
    return Load(file).Get(record);
  }

 private:
  std::string model_;
  std::string directory_;
  mutable std::map<std::string, Fixture> fixtures_;
};

}  // namespace oracle
}  // namespace nanochat

#endif  // NANOCHAT_TESTS_ORACLE_FIXTURE_H_
