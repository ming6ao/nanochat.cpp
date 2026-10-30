// tests/model_isolation_test.cc -- the cross-package non-interference gate of
// docs/model-and-device-bringup.md section 10.4, in code.
//
// The gate answers one question: *did building this model change another
// model's golden bytes?* For the flagship model it has three steps, and all
// three run on the CPU with no GPU, no broker, and no torch.
//
//   1. **Re-derive.** Every committed fixture under `models/nanochat-default/
//      fixtures` is re-read from its bytes, and the metadata the model package
//      declares about it - magic, format version, record count, record names
//      and their order, size, and `fnv1a64` - must be exactly what those bytes
//      say. The manifest is the derivation, not a second opinion.
//   2. **Diff within the model's tolerance contract.** The same fixture is
//      compared record by record against the source it was copied from
//      (`tests/data/<file>`, the generator's home for this release) using the
//      record-name selectors of `models/nanochat-default/tolerances.json`. That
//      is `tools/isolation_gate.py`'s step 1 and 2 with the C++ reader, and the
//      comparison applies the bound the model declares: it never loosens one
//      (section 15) and never compares a record with no selector.
//   3. **Prove isolation.** A *second* fabricated model spec is built in a
//      temporary directory. Section 2 keeps the second model out of scope, so
//      it is not committed: the spec is written to the test's scratch
//      directory, parsed back through the real reader, derived into its own
//      `Config`, and its fixture set is proved unreachable. The default
//      model's fixture bytes are then re-read and compared with the digests
//      taken before the second model existed. Step 3 is the isolation proof,
//      and it needs no GPU.
//
// Two things this gate refuses to do. It never rewrites a committed fixture
// byte: a changed byte is a different fixture and belongs to a model package,
// not to a test run. And it never hand-maintains what the generator derives:
// the record list and the `fnv1a64` digests are re-derived here, so a manifest
// edited to match a mutated file is a failure in step 1 rather than a pass.
//
// A fixture is keyed by its model package (`FixtureSet`): its records are
// reached as `(model, file, record)` and never by shape alone, because two
// models may share every shape and mean different arithmetic
// (`spec/versions.md` section 3.3).
//
// Run with `tools/nanochat test //tests:model_isolation_test`.

#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/model_spec.h"
#include "src/json.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::json::Value;
using nanochat::oracle::DType;
using nanochat::oracle::Fixture;
using nanochat::oracle::FixtureOrigin;
using nanochat::oracle::FixtureSet;
using nanochat::oracle::Tensor;

//: The model whose package this gate owns. Every path below starts here.
constexpr const char* kModel = "nanochat-default";

//: The fabricated second model. It is a *name*, not a package: section 2 keeps
//: the second model out of scope, so no `models/isolation-probe` exists and
//: step 3 asserts that it does not.
constexpr const char* kProbeModel = "isolation-probe";

//: The context length the fabricated spec declares. It differs from the
//: flagship's 2048 so the two models derive different `Config`s, which is what
//: makes "a second model" a fact rather than a rename.
constexpr int kProbeSeqLen = 1024;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Check(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

std::string Format(const char* fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------
// Small file and JSON helpers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
}

bool FileExists(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

// FNV-1a 64, the digest `models/<name>/fixtures/manifest.json` pins beside the
// sha256. It is cheap enough to re-derive in a test and it is order sensitive,
// so a mutated byte always moves it.
std::string Fnv1a64(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t value = 0xCBF29CE484222325ULL;
  for (std::uint8_t byte : bytes) {
    value ^= byte;
    value *= 0x100000001B3ULL;
  }
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "0x%016llx",
                static_cast<unsigned long long>(value));
  return std::string(buffer);
}

std::uint32_t ReadU32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

const Value* Member(const Value& object, const char* key,
                    const std::string& where) {
  const Value* value = object.Find(key);
  if (value == nullptr) {
    Fail(where + ": no \"" + key + "\"");
  }
  return value;
}

std::string WantString(const Value& object, const char* key,
                       const std::string& where) {
  const Value* value = Member(object, key, where);
  if (value == nullptr) return std::string();
  if (!value->is_string()) {
    Fail(where + ": \"" + key + "\" is not a string");
    return std::string();
  }
  return value->string();
}

int WantInt(const Value& object, const char* key, const std::string& where,
            int fallback = 0) {
  const Value* value = Member(object, key, where);
  if (value == nullptr) return fallback;
  int out = fallback;
  if (!value->Integer(&out)) {
    Fail(where + ": \"" + key + "\" is not an integer");
    return fallback;
  }
  return out;
}

bool WantBool(const Value& object, const char* key, const std::string& where,
              bool fallback = false) {
  const Value* value = Member(object, key, where);
  if (value == nullptr) return fallback;
  if (!value->is_bool()) {
    Fail(where + ": \"" + key + "\" is not a boolean");
    return fallback;
  }
  return value->boolean();
}

const std::vector<Value>& WantArray(const Value& object, const char* key,
                                    const std::string& where) {
  static const std::vector<Value> kEmpty;
  const Value* value = Member(object, key, where);
  if (value == nullptr) return kEmpty;
  if (!value->is_array()) {
    Fail(where + ": \"" + key + "\" is not an array");
    return kEmpty;
  }
  return value->array();
}

Value ParseJsonFile(const std::string& path) {
  Value value;
  std::string error;
  if (!nanochat::json::ParseFile(path, &value, &error)) {
    throw std::runtime_error(path + ": " + error);
  }
  return value;
}

// ---------------------------------------------------------------------------
// Record-name selectors (`tolerances.json`)
// ---------------------------------------------------------------------------

// The selector glob: `*` matches any run of characters and `?` one character.
// The vocabulary is deliberately small - `fnmatch`'s character classes would
// let a selector mean something a reader cannot see - and every selector the
// model package carries is expressible in it.
bool GlobMatch(const std::string& pattern, const std::string& name) {
  std::size_t p = 0;
  std::size_t n = 0;
  std::size_t star = std::string::npos;
  std::size_t star_name = 0;
  while (n < name.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
      ++p;
      ++n;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star = p++;
      star_name = n;
    } else if (star != std::string::npos) {
      p = star + 1;
      n = ++star_name;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}

// The first record-name selector of `entry` that matches `name`, or null. The
// first match wins, which is the rule the model package states in its
// `coverage` field.
const Value* SelectBound(const Value& entry, const std::string& name) {
  for (const Value& selector : WantArray(entry, "records", "tolerances")) {
    const std::string match = WantString(selector, "match", "selector");
    if (GlobMatch(match, name)) return &selector;
  }
  return nullptr;
}

std::string SelectorPattern(const Value& selector) {
  return WantString(selector, "match", "selector");
}

// ---------------------------------------------------------------------------
// Payload comparison (`tools/isolation_gate.py::compare_payload`, in C++)
// ---------------------------------------------------------------------------

std::vector<double> Floats(int dtype, const Tensor& tensor) {
  std::vector<double> values;
  if (dtype == static_cast<int>(DType::kFp32)) {
    const std::size_t count = tensor.bytes / sizeof(float);
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      float value = 0.0f;
      std::memcpy(&value, tensor.data + i * sizeof(float), sizeof(float));
      values.push_back(static_cast<double>(value));
    }
  } else if (dtype == static_cast<int>(DType::kFp64)) {
    const std::size_t count = tensor.bytes / sizeof(double);
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      double value = 0.0;
      std::memcpy(&value, tensor.data + i * sizeof(double), sizeof(double));
      values.push_back(value);
    }
  }
  return values;
}

// The first tolerance violation in one record, or an empty string. The modes
// are the ones `tolerances.json` declares and `tools/isolation_gate.py`
// implements; an unknown mode is a failure, never a pass, because "the bound
// did not parse" must not read as "within tolerance".
std::string ComparePayload(const Tensor& committed, const Tensor& generated,
                           const Value& bound, const std::string& where) {
  const std::string mode = WantString(bound, "mode", where);
  if (mode == "report_only") return std::string();
  const int dtype = static_cast<int>(committed.dtype);
  const bool is_float = dtype == static_cast<int>(DType::kFp32) ||
                        dtype == static_cast<int>(DType::kFp64);
  if (!is_float || mode == "exact" || mode == "producer_identity" ||
      mode == "exact_or_skip_device") {
    if (committed.bytes != generated.bytes ||
        std::memcmp(committed.data, generated.data, committed.bytes) != 0) {
      return mode == "exact" || mode == "producer_identity" ||
                     mode == "exact_or_skip_device"
                 ? "record differs (exact)"
                 : "integer/byte record differs (exact)";
    }
    return std::string();
  }
  if (mode != "absolute" && mode != "relative") {
    return "unknown tolerance mode \"" + mode + "\"";
  }
  double atol = 0.0;
  double rtol = 0.0;
  if (const Value* value = bound.Find("atol")) {
    if (value->is_number()) atol = value->number();
  }
  if (const Value* value = bound.Find("rtol")) {
    if (value->is_number()) rtol = value->number();
  }
  const std::vector<double> expected = Floats(dtype, committed);
  const std::vector<double> actual = Floats(dtype, generated);
  if (expected.size() != actual.size()) return "element count differs";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double want = expected[i];
    const double got = actual[i];
    if (want == got) continue;
    if (std::isnan(want) && std::isnan(got)) continue;
    if (std::fabs(got - want) <= atol + rtol * std::fabs(want)) continue;
    return Format("element %zu is %.9g, expected %.9g (atol=%g, rtol=%g)", i,
                  got, want, atol, rtol);
  }
  return std::string();
}

// ---------------------------------------------------------------------------
// Step 1: re-derive each fixture's metadata from its own bytes
// ---------------------------------------------------------------------------

void CheckFixtureIsSelfDescribing(const std::string& model_dir,
                                  const Value& manifest_entry,
                                  const Value& tolerance_entry,
                                  const FixtureOrigin& origin) {
  const std::string file = WantString(manifest_entry, "file", "manifest");
  const std::string where = kModel + std::string("/") + file;
  const std::string path = model_dir + "/" + file;
  Check(FileExists(path), where + ": the committed fixture exists");
  if (!FileExists(path)) return;

  const std::vector<std::uint8_t> bytes = ReadBytes(path);
  const int size = WantInt(manifest_entry, "size_bytes", where,
                           static_cast<int>(bytes.size()));
  Check(static_cast<std::size_t>(size) == bytes.size(),
        Format("%s: size %zu matches the manifest's %d", where.c_str(),
               bytes.size(), size));
  Check(Fnv1a64(bytes) == WantString(manifest_entry, "fnv1a64", where),
        where + ": fnv1a64 re-derives from the bytes");
  Check(WantInt(manifest_entry, "record_count", where) ==
            WantInt(tolerance_entry, "record_count", where),
        where + ": the manifest and tolerances.json agree on the record count");
  Check(WantString(manifest_entry, "sha256", where) ==
            WantString(tolerance_entry, "sha256", where),
        where + ": the manifest and tolerances.json pin the same sha256");
  Check(!WantString(tolerance_entry, "precision", where).empty(),
        where +
            ": the tolerance contract names the precision it was recorded "
            "at");

  // The container's own words: magic and version. A fixture whose version is
  // not in the reader's acceptance set throws here rather than comparing.
  const std::string magic = WantString(manifest_entry, "magic", where);
  Check(
      bytes.size() >= 12 &&
          std::string(reinterpret_cast<const char*>(bytes.data()), 8) == magic,
      where + ": the bytes carry the manifest's magic (" + magic + ")");
  const int declared_version = WantInt(manifest_entry, "format_version", where);
  Check(bytes.size() >= 12 &&
            static_cast<int>(ReadU32(bytes.data() + 8)) == declared_version,
        where + ": the bytes carry the manifest's format version");
  Check(nanochat::oracle::AcceptFixtureVersion(
            static_cast<std::uint32_t>(declared_version)),
        where +
            ": the declared format version is in the reader's acceptance "
            "set");

  const Fixture fixture = Fixture::Load(origin, path);
  Check(fixture.magic() == magic, where + ": the reader reports the magic");
  Check(static_cast<int>(fixture.format_version()) == declared_version,
        where + ": the reader reports the format version");
  Check(static_cast<int>(fixture.size()) ==
            WantInt(tolerance_entry, "record_count", where),
        where + ": the reader counts the records the model declares");

  // Record names and their order are part of the fixture: a reordered file is
  // a different fixture.
  const std::vector<Value>& names =
      WantArray(manifest_entry, "record_names", where);
  Check(names.size() == fixture.size(),
        where + ": the manifest lists one name per record");
  for (std::size_t i = 0; i < names.size() && i < fixture.names().size(); ++i) {
    if (!names[i].is_string() || names[i].string() != fixture.names()[i]) {
      Fail(
          Format("%s: record %zu is \"%s\" in the bytes but \"%s\" in the "
                 "manifest",
                 where.c_str(), i, fixture.names()[i].c_str(),
                 names[i].is_string() ? names[i].string().c_str() : "?"));
      break;
    }
  }

  // The selectors must cover the record set, and every selector must name at
  // least one record: a dead selector is a tolerance nobody reads.
  std::vector<bool> used(WantArray(tolerance_entry, "records", where).size(),
                         false);
  const std::vector<Value>& selectors =
      WantArray(tolerance_entry, "records", where);
  for (const std::string& name : fixture.names()) {
    if (SelectBound(tolerance_entry, name) == nullptr) {
      Fail(where + ": record \"" + name +
           "\" has no tolerance selector in tolerances.json");
      continue;
    }
    for (std::size_t i = 0; i < selectors.size(); ++i) {
      if (GlobMatch(SelectorPattern(selectors[i]), name)) used[i] = true;
    }
  }
  for (std::size_t i = 0; i < selectors.size(); ++i) {
    if (!used[i]) {
      Fail(where + ": selector \"" + SelectorPattern(selectors[i]) +
           "\" matches no record; the coverage claim in tolerances.json is "
           "false");
    }
  }

  // The source the package was copied from, and the copy's identity. This
  // release keeps both directories (`models/<name>/fixtures` and `tests/data`)
  // and pins them equal, so a divergence is a failure rather than a preference.
  const std::string source = WantString(tolerance_entry, "source", where);
  Check(source == WantString(manifest_entry, "source", where),
        where + ": the manifest and tolerances.json name the same source");
  if (WantBool(manifest_entry, "byte_identical_to_source", where)) {
    const std::string source_path = model_dir + "/../../" + source;
    const bool exists = FileExists(source_path);
    Check(exists, where + ": the source " + source + " still exists");
    if (exists) {
      Check(
          ReadBytes(source_path) == bytes,
          where + ": the model package's copy is byte-identical to " + source);
    }
  }
}

// ---------------------------------------------------------------------------
// Step 2: diff the re-derived bytes within the model's tolerance contract
// ---------------------------------------------------------------------------

void CheckFixtureWithinTolerances(const std::string& model_dir,
                                  const Value& tolerance_entry,
                                  const FixtureOrigin& origin) {
  const std::string file = WantString(tolerance_entry, "file", "tolerances");
  const std::string where = kModel + std::string("/") + file;
  const std::string committed_path = model_dir + "/" + file;
  const std::string source = WantString(tolerance_entry, "source", where);
  const std::string generated_path = model_dir + "/../../" + source;
  if (!FileExists(committed_path) || !FileExists(generated_path)) return;

  const Fixture committed = Fixture::Load(origin, committed_path);
  const Fixture generated = Fixture::Load(
      FixtureOrigin{std::string("regenerated"), source}, generated_path);
  Check(committed.magic() == generated.magic(),
        where + ": the regenerated container carries the same magic");
  Check(committed.format_version() == generated.format_version(),
        where + ": the regenerated container carries the same format version");
  if (committed.names() != generated.names()) {
    Fail(where +
         ": the regenerated record names or their order differ from "
         "the committed fixture");
    return;
  }
  for (const std::string& name : committed.names()) {
    const Tensor& want = committed.Get(name);
    const Tensor& got = generated.Get(name);
    if (want.dtype != got.dtype || want.shape != got.shape) {
      Fail(where + ": record \"" + name + "\" changed dtype or shape");
      continue;
    }
    const Value* selector = SelectBound(tolerance_entry, name);
    if (selector == nullptr) {
      Fail(where + ": record \"" + name + "\" has no tolerance selector");
      continue;
    }
    const Value* bound = selector->Find("bound");
    if (bound == nullptr || !bound->is_object()) {
      Fail(where + ": record \"" + name + "\" names a selector with no bound");
      continue;
    }
    const std::string detail =
        ComparePayload(want, got, *bound, where + ": record \"" + name + "\"");
    if (!detail.empty()) {
      Fail(where + ": record \"" + name + "\": " + detail);
    }
  }
}

// ---------------------------------------------------------------------------
// Step 3: the isolation proof
// ---------------------------------------------------------------------------

// The fabricated second model's spec, as the reader reads it: the flagship's
// field set with a different name, a shorter context, and a smaller vocabulary,
// so the two models derive different `Config`s. It is written into the test's
// scratch directory and never into `models/`.
std::string ProbeSpecJson() {
  const char* kBlock =
      R"({"norm": "rms_norm", "attention": "%s", "attention_variant": "mha", )"
      R"("position": "rope", "mlp": "relu_square", "residual": "blend", )"
      R"("dtype": "build_default"})";
  std::string blocks;
  // The flagship's `SSSL` tiling, shortened to two blocks: one windowed block
  // and one full-context block, which is the smallest block list that declares
  // both attention kinds.
  for (const char* kind : {"sliding_window", "causal"}) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), kBlock, kind);
    if (!blocks.empty()) blocks += ",\n    ";
    blocks += buffer;
  }
  return std::string(R"({
  "spec_version": 1,
  "name": ")") +
         kProbeModel + R"(",
  "requires": ["rope", "sliding_window", "value_gate", "primitive_tier_complete"],
  "dimensions": {
    "num_heads": 6,
    "num_kv_heads": 6,
    "hidden_dim": 768,
    "seq_len": )" +
         std::to_string(kProbeSeqLen) + R"(,
    "rope_base": 100000.0
  },
  "blocks": [
    )" + blocks +
         R"(
  ],
  "embeddings": {
    "vocab_size": 4096,
    "padded_vocab_size": 4096,
    "value_embedding": true
  },
  "head": {"kind": "untied_linear", "logit_softcap": 15.0},
  "loss": {"kind": "softcap_cross_entropy", "ignore_index": -1},
  "policy": {
    "dtype": "build_default",
    "allow_fused_overrides": true,
    "require_primitive_tier": true
  }
}
)";
}

// A scratch directory of this test's own. `TEST_TMPDIR` is Bazel's per-test
// scratch tree; `/tmp` keeps a direct run off the repository.
std::string CreateScratchDirectory() {
  const char* base = std::getenv("TEST_TMPDIR");
  const std::string root = (base != nullptr && *base != '\0') ? base : "/tmp";
  std::string pattern = root + "/nanochat-isolation-XXXXXX";
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  if (mkdtemp(buffer.data()) == nullptr) {
    throw std::runtime_error("cannot create a scratch directory under " + root);
  }
  return std::string(buffer.data());
}

void WriteFile(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ParentDirectory(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

// Every model package under `models/`: a directory counts only when it carries
// a `spec.json`, the package's identity.
std::vector<std::string> ModelPackages(const std::string& models_dir) {
  std::vector<std::string> packages;
  DIR* dir = opendir(models_dir.c_str());
  if (dir == nullptr) return packages;
  while (const dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (FileExists(models_dir + "/" + name + "/spec.json")) {
      packages.push_back(name);
    }
  }
  closedir(dir);
  std::sort(packages.begin(), packages.end());
  return packages;
}

void CheckSecondModelIsIsolated(const std::string& model_dir,
                                const Value& manifest) {
  // The digest of every committed fixture, taken *before* the second model
  // exists. Step 3 compares against this snapshot, so "unchanged" is a
  // measurement rather than a claim.
  std::map<std::string, std::string> before;
  std::map<std::string, std::size_t> before_size;
  for (const Value& entry : WantArray(manifest, "fixtures", "manifest")) {
    const std::string file = WantString(entry, "file", "manifest");
    const std::string path = model_dir + "/" + file;
    if (!FileExists(path)) continue;
    const std::vector<std::uint8_t> bytes = ReadBytes(path);
    before[file] = Fnv1a64(bytes);
    before_size[file] = bytes.size();
  }
  Check(!before.empty(),
        "the flagship package has committed fixtures to protect");

  // Build the second model, in a scratch directory that is not the repository.
  const std::string scratch = CreateScratchDirectory();
  const std::string probe_path = scratch + "/spec.json";
  WriteFile(probe_path, ProbeSpecJson());
  Check(Contains(scratch, "nanochat-isolation-") &&
            !Contains(probe_path, model_dir),
        "the fabricated spec is written into the test's scratch directory, not "
        "into a model package: the second model is not committed (section 2)");

  nanochat::ModelSpec probe;
  std::string error;
  Check(nanochat::ParseModelSpec(probe_path, &probe, &error),
        "the fabricated spec parses through the real reader: " + error);
  Check(probe.name == kProbeModel,
        "the fabricated spec names its own model: " + probe.name);
  Check(nanochat::ValidateModelSpec(probe, &error),
        "the fabricated spec passes the closed-field-set validation: " + error);

  // The two models are different models: same mechanisms, different derived
  // dimensions. A rename would prove nothing.
  const nanochat::ModelSpec& flagship = nanochat::DefaultModelSpec();
  Check(probe.requires_features == flagship.requires_features,
        "the fabricated model requires the same mechanisms as the flagship");
  nanochat::Config probe_config;
  nanochat::Config flagship_config;
  Check(nanochat::ModelSpecToConfig(probe, &probe_config, &error),
        "the fabricated spec derives a Config: " + error);
  Check(nanochat::ModelSpecToConfig(flagship, &flagship_config, &error),
        "the flagship spec derives a Config: " + error);
  Check(probe_config.seq_len == kProbeSeqLen,
        Format("the fabricated model keeps its own context (%d)",
               probe_config.seq_len));
  Check(flagship_config.seq_len == 2048,
        Format("the flagship model keeps its own context (%d)",
               flagship_config.seq_len));
  Check(probe_config.seq_len != flagship_config.seq_len,
        "the two models derive different contexts");
  Check(probe.blocks.size() != flagship.blocks.size(),
        "the fabricated model's block list is its own");

  // Nothing about the second model is committed, and its fixtures are
  // unreachable: a fixture is resolved from a model package, so a model with no
  // package can never hand back another model's bytes.
  const std::string probe_models = ParentDirectory(model_dir);
  const std::vector<std::string> packages = ModelPackages(probe_models);
  Check(packages.size() == 1 && packages[0] == kModel,
        "the tree holds exactly one model package, the flagship");
  Check(nanochat::oracle::FindModelPackageDirectory(kProbeModel).empty(),
        "no models/isolation-probe/spec.json was committed");
  Check(nanochat::oracle::FindModelFixturesDirectory(kProbeModel).empty(),
        "no models/isolation-probe/fixtures directory was committed");
  bool threw = false;
  try {
    FixtureSet::ForModel(kProbeModel);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw,
        "FixtureSet::ForModel refuses a model with no package instead of "
        "resolving a fixture by shape");

  // The default model's fixtures are unchanged, byte for byte.
  for (const std::pair<const std::string, std::string>& entry : before) {
    const std::string path = model_dir + "/" + entry.first;
    Check(FileExists(path),
          entry.first + ": the committed fixture is still there");
    if (!FileExists(path)) continue;
    const std::vector<std::uint8_t> bytes = ReadBytes(path);
    Check(bytes.size() == before_size[entry.first],
          Format("%s: size %zu is unchanged after the second model was built",
                 entry.first.c_str(), bytes.size()));
    Check(Fnv1a64(bytes) == entry.second,
          entry.first +
              ": fnv1a64 is unchanged after the second model was "
              "built");
  }

  // And the flagship's own spec still parses as the flagship: building a second
  // model rewrote nothing on the first model's axis.
  nanochat::ModelSpec reread;
  Check(nanochat::ParseModelSpec(model_dir + "/spec.json", &reread, &error),
        "models/nanochat-default/spec.json still parses: " + error);
  Check(reread.name == kModel,
        "models/nanochat-default/spec.json still names the flagship");
  Check(reread.dimensions.seq_len == flagship_config.seq_len,
        "the flagship spec's dimensions are unchanged");
}

// The fixture set of the flagship resolves through the model package and
// carries the model in every fixture's origin, which is what makes a fixture
// keyed by its model rather than by its shape.
void CheckFixtureSetIsKeyedByModel(const std::string& model_dir) {
  FixtureSet set = FixtureSet::ForModel(kModel);
  Check(set.model() == kModel,
        "the fixture set names the model it was built for");
  Check(set.directory() == model_dir || Contains(set.directory(), kModel),
        "the fixture set resolved models/nanochat-default/fixtures: " +
            set.directory());
  Check(set.Has("debug_state.bin"), "the fixture set finds the oracle fixture");
  const Fixture& fixture = set.Load("debug_state.bin");
  Check(fixture.origin().model == kModel,
        "a fixture loaded through the set carries its model");
  Check(fixture.origin().file == "debug_state.bin",
        "a fixture loaded through the set carries its file");
  Check(set.Record("debug_state.bin", "config/layers").scalar_int() > 0,
        "a record is reached as (model, file, record)");
  bool threw = false;
  try {
    set.Path("../debug_state.bin");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "a fixture path may not escape its model package");
}

}  // namespace

int main() {
  try {
    const std::string model_dir =
        nanochat::oracle::FindModelPackageDirectory(kModel);
    if (model_dir.empty()) {
      std::fprintf(stderr,
                   "FAIL: models/%s is not in the runfiles; the gate cannot "
                   "read a model package\n",
                   kModel);
      return 1;
    }
    const Value manifest = ParseJsonFile(model_dir + "/fixtures/manifest.json");
    const Value tolerances = ParseJsonFile(model_dir + "/tolerances.json");
    Check(WantString(manifest, "model", "manifest") == kModel,
          "the manifest names the model it belongs to");
    Check(WantString(tolerances, "model", "tolerances") == kModel,
          "tolerances.json names the model it belongs to");

    // The two files describe the same fixture set. A fixture in one and not the
    // other is a gate that cannot decide.
    std::map<std::string, const Value*> manifest_by_file;
    for (const Value& entry : WantArray(manifest, "fixtures", "manifest")) {
      const std::string file = WantString(entry, "file", "manifest");
      if (manifest_by_file.count(file) != 0) {
        Fail(file + ": the manifest lists the fixture twice");
      }
      manifest_by_file[file] = &entry;
    }
    std::size_t tolerance_entries = 0;
    for (const Value& entry : WantArray(tolerances, "fixtures", "tolerances")) {
      ++tolerance_entries;
      const std::string file = WantString(entry, "file", "tolerances");
      const auto found = manifest_by_file.find(file);
      Check(found != manifest_by_file.end(),
            file + ": tolerances.json names a fixture the manifest does not");
      if (found == manifest_by_file.end()) continue;
      const FixtureOrigin origin{kModel, file};
      CheckFixtureIsSelfDescribing(model_dir, *found->second, entry, origin);
      CheckFixtureWithinTolerances(model_dir, entry, origin);
    }
    Check(tolerance_entries == manifest_by_file.size(),
          Format("every committed fixture has a tolerance entry (%zu of %zu)",
                 tolerance_entries, manifest_by_file.size()));

    CheckFixtureSetIsKeyedByModel(model_dir);
    CheckSecondModelIsIsolated(model_dir, manifest);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    ++g_failures;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "model_isolation_test: %d check(s) failed\n",
                 g_failures);
    return 1;
  }
  std::printf("model_isolation_test: %s is isolated\n", kModel);
  return 0;
}
