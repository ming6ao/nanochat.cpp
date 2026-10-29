// tests/checkpoint_compat_test.cc -- the version 0/1 checkpoint compatibility
// guard (docs/checkpoint-metadata-and-examples-plan.md stage A.4 and section 5;
// the "a checkpoint that already sits on the Hub keeps loading" rule of
// docs/checkpoint-format.md).
//
// What this file proves, in one place, because the release gate reads it:
//
// 1. The committed version 0 fixture, `tests/data/checkpoint_v0.ckpt`, still
//    loads with the exact record names, storage dtypes, shapes, and payload
//    bytes it was pinned with, and its metadata reads as the member
//    initializers, because a version 0 file carries no metadata section. The
//    bytes came from the writer before the metadata work moved it
//    (`tools/dump_fixtures.py checkpoint`), and `tests/data/checkpoint_v0.txt`
//    is the row-by-row note this test cites, so a change that moves a version 0
//    byte fails here instead of on a Hub artifact.
// 2. A version 1 file round-trips: every metadata field and every record comes
//    back equal.
// 3. Both drift directions are accepted and counted: a block with trailing
//    words this build does not know reports `extra_u32`, and a block missing
//    the last fields reports `missing_fields` while the initializers stand
//    (design decisions D5 and D6).
// 4. Every malformed input is a hard error and `Checkpoint::Load` returns
//    false, with no partial load (design decision D7).
// 5. The precision gate of section 8: the fixture reads identically under
//    `--config=bf16` and `--config=fp8`, and the recorded `precision` value
//    never changes a load decision.
//
// Open decision O5 selected version 1 with the existing container magic, so
// the plan's two magic/version mismatch cases (a second magic word beside a
// version 0 word, and the current magic beside an unsupported version word)
// collapse: there is one magic, the magic names the layout family, and the
// `version` word alone selects the layout inside it. The case was considered
// and does not apply, by the O5 decision recorded in `include/nanochat/data.h`.
// What O5 does leave is the release gate of section 11: a version word this
// build does not know must be rejected with a clear message instead of
// misparsed, and the hard-error table below pins that.
//
// The two drift cases and the "version 0 header, then metadata words" case need
// a file whose metadata block is not the one `Checkpoint::Save` writes, so the
// test splices a replacement block into the header of a saved file and keeps
// the record stream of that file byte for byte. The only layout the splice
// names is the container header the format declares (8 magic bytes, the
// `version` word, the metadata word count), so no record framing is copied
// here; the record stream stays single-sourced in `src/data.cc`. The drift
// cases then use the `OldFields`/`NewFields` trick of `gemma.cpp`
// `io/fields_test.cc`: one struct shape writes the word stream and a second,
// longer one reads it.
//
// Runs in the default T0 CPU loop, and in the bf16 and fp8 builds:
//   tools/nanochat test //tests:checkpoint_compat_test
//   tools/nanochat test //tests:checkpoint_compat_test --config=bf16
//   tools/nanochat test //tests:checkpoint_compat_test --config=fp8

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/checkpoint_metadata.h"
#include "nanochat/data.h"
#include "nanochat/fields.h"
#include "nanochat/mfu.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::Checkpoint;
using nanochat::CheckpointFormat;
using nanochat::CheckpointMetadata;
using nanochat::ComputePrecision;
using nanochat::Config;
using nanochat::DType;
using nanochat::IFields;
using nanochat::IFieldsVisitor;
using nanochat::PrecisionName;
using nanochat::Stage;
using nanochat::TensorRecord;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

std::string Hex(std::uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "0x%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

std::string Count(std::uint32_t value) { return std::to_string(value); }

// --- The container header ---------------------------------------------------
//
// The header every container version shares: 8 magic bytes and the `version`
// word, and for version 1 the metadata word count directly after it
// (docs/checkpoint-format.md section 2.1, and section 2.1 of the plan). The
// test needs these three offsets to splice a metadata block into a file
// `Checkpoint::Save` wrote. They are the format's own header, not a copy of the
// record framing: everything after the block is carried through verbatim.
constexpr std::size_t kMagicSize = 8;
constexpr std::size_t kMetadataCountOffset = kMagicSize + 4;
constexpr std::size_t kWholeHeaderSize = kMagicSize + 8;
constexpr std::uint32_t kContainerVersion0 = 0;
constexpr std::uint32_t kContainerVersion1 = 1;
// The magic word itself, eight bytes. A code constant keeps the magic; the
// prose in this file says "the container magic" (stage A.6's sweep rule).
constexpr char kContainerMagic[] = "NCHKPT01";

// The payload bytes of the version 0 fixture, from `tests/data/
// checkpoint_v0.txt`. Pinned as hex so a moved byte names itself.
constexpr char kFixtureWeightsHex[] =
    "3fe1043f4eb34d3f79f693bdd1eff7bea13c34bf62b1c6be";
constexpr char kFixtureBiasHex[] = "34bf933ef23e";
// The FNV-1a 64 checksums the note records: per record, and over both payloads
// in record order.
constexpr std::uint64_t kFixtureWeightsSum = 0x7610708711193cf0ull;
constexpr std::uint64_t kFixtureBiasSum = 0x702bf156049a2035ull;
constexpr std::uint64_t kFixtureConcatSum = 0x41273f0935abaaa8ull;
// The committed file size, one more guard on the pinned bytes.
constexpr std::uint64_t kFixtureSize = 145;

void AppendU32(std::vector<char>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<char>((value >> (8 * i)) & 0xffu));
  }
}

std::uint32_t ReadU32At(const std::vector<char>& bytes, std::size_t offset) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(
                 static_cast<unsigned char>(bytes[offset + i]))
             << (8 * i);
  }
  return value;
}

bool ReadFileBytes(const std::string& path, std::vector<char>* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  out->assign(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
  return true;
}

bool WriteFileBytes(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

std::uint64_t FileSize(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return 0;
  return static_cast<std::uint64_t>(in.tellg());
}

// The version word plus the magic, and nothing else: the start of a container
// the test completes by hand.
std::vector<char> HeaderBytes(std::uint32_t version) {
  // `kMagicSize` and not `sizeof(kContainerMagic)`: the literal keeps its
  // terminator, and the header carries exactly the eight magic bytes.
  std::vector<char> bytes(kContainerMagic, kContainerMagic + kMagicSize);
  AppendU32(&bytes, version);
  return bytes;
}

// A container header, then `words` as its metadata block, then `tail`
// verbatim. The tail comes from a file `Checkpoint::Save` wrote, so the
// parameter count and the record stream stay the ones the writer produced.
bool WriteContainerWithMetadata(const std::string& path,
                                std::vector<std::uint32_t> words,
                                const std::vector<char>& tail) {
  std::vector<char> bytes = HeaderBytes(kContainerVersion1);
  AppendU32(&bytes, static_cast<std::uint32_t>(words.size()));
  for (std::uint32_t word : words) AppendU32(&bytes, word);
  bytes.insert(bytes.end(), tail.begin(), tail.end());
  return WriteFileBytes(path, bytes);
}

// The bytes of `path` from the end of its metadata block on: the parameter
// count and the version 0 record stream, unchanged.
bool ReadTailAfterMetadata(const std::string& path, std::vector<char>* tail) {
  std::vector<char> bytes;
  if (!ReadFileBytes(path, &bytes)) return false;
  if (bytes.size() < kWholeHeaderSize) return false;
  const std::uint32_t count = ReadU32At(bytes, kMetadataCountOffset);
  const std::size_t offset =
      kWholeHeaderSize + 4 * static_cast<std::size_t>(count);
  if (bytes.size() < offset) return false;
  tail->assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
               bytes.end());
  return true;
}

// --- Comparisons ------------------------------------------------------------

std::string HexString(const std::vector<std::byte>& bytes) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::byte byte : bytes) {
    const unsigned value = std::to_integer<unsigned>(byte);
    out.push_back(kDigits[value >> 4]);
    out.push_back(kDigits[value & 0xfu]);
  }
  return out;
}

// FNV-1a, 64 bit, over the payload bytes: the checksum
// `tests/data/checkpoint_v0.txt` records, so the test cites the same table the
// note does.
std::uint64_t Fnv1a64(const std::vector<std::byte>& bytes) {
  std::uint64_t hash = 0xcbf29ce484222325ull;
  for (std::byte byte : bytes) {
    hash ^= static_cast<std::uint64_t>(std::to_integer<unsigned>(byte));
    hash *= 0x100000001b3ull;
  }
  return hash;
}

std::vector<std::byte> Payload(std::initializer_list<int> values) {
  std::vector<std::byte> out;
  out.reserve(values.size());
  for (int value : values) {
    out.push_back(static_cast<std::byte>(value & 0xff));
  }
  return out;
}

bool SameConfig(const Config& left, const Config& right) {
  return left.num_layers == right.num_layers &&
         left.num_heads == right.num_heads &&
         left.num_kv_heads == right.num_kv_heads &&
         left.hidden_dim == right.hidden_dim && left.seq_len == right.seq_len &&
         left.vocab_size == right.vocab_size &&
         left.padded_vocab_size == right.padded_vocab_size &&
         left.rope_base == right.rope_base &&
         left.window_pattern == right.window_pattern &&
         left.value_embedding == right.value_embedding;
}

// Field by field, so a failure names the field that moved. The order is the
// `VisitFields` order of `nanochat/checkpoint_metadata.h`, which is the format.
void ExpectSameMetadata(const CheckpointMetadata& got,
                        const CheckpointMetadata& want,
                        const std::string& what) {
  Expect(got.schema == want.schema, what + ": schema changed");
  Expect(got.stage == want.stage, what + ": stage changed");
  Expect(got.step == want.step, what + ": step changed (" +
                                    std::to_string(got.step) +
                                    " != " + std::to_string(want.step) + ")");
  Expect(got.world_size == want.world_size, what + ": world_size changed");
  Expect(got.precision == want.precision, what + ": precision changed");
  Expect(SameConfig(got.config, want.config),
         what + ": the run config changed");
  Expect(got.tokenizer_id == want.tokenizer_id,
         what + ": tokenizer_id changed");
  Expect(got.git_revision == want.git_revision,
         what + ": git_revision changed");
  Expect(got.created_utc == want.created_utc, what + ": created_utc changed");
  Expect(got.notes == want.notes, what + ": notes changed");
}

void ExpectSameRecords(const Checkpoint& got, const Checkpoint& want,
                       const std::string& what) {
  const std::vector<TensorRecord>& left = got.tensors();
  const std::vector<TensorRecord>& right = want.tensors();
  Expect(left.size() == right.size(), what + ": the record count changed");
  if (left.size() != right.size()) return;
  for (std::size_t i = 0; i < left.size(); ++i) {
    const std::string at = what + ": record " + std::to_string(i);
    Expect(left[i].name == right[i].name, at + " changed its name");
    Expect(left[i].dtype == right[i].dtype, at + " changed its dtype");
    Expect(left[i].shape == right[i].shape, at + " changed its shape");
    Expect(left[i].data == right[i].data, at + " changed its payload");
    Expect(left[i].attributes.size() == right[i].attributes.size(),
           at + " changed its attribute count");
  }
  Expect(got.optimizer_state().size() == want.optimizer_state().size(),
         what + ": the optimizer-state count changed");
}

// --- Containers built by the test -------------------------------------------

// The smallest config `Config::Validate` accepts: 24 hidden dimensions, two
// heads, an even per-head width, one KV head, and a two-entry window pattern.
Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.hidden_dim = 24;
  config.seq_len = 16;
  config.vocab_size = 32;
  config.padded_vocab_size = 32;
  config.rope_base = 10000.0f;
  config.window_pattern = "SL";
  config.value_embedding = true;
  return config;
}

// A version 1 container with every metadata field set away from its default, so
// a field that silently fell back to its initializer fails the round trip. The
// step is above `2^32`, so a stream that wrote or read the two words in the
// wrong order (open decision O6) cannot come back equal.
Checkpoint MakeContainer() {
  Checkpoint checkpoint;
  {
    TensorRecord record;
    record.name = "compat/weights";
    record.dtype = DType::kFp32;
    record.shape = {2, 3};
    record.data = Payload({0x3f, 0x80, 0x00, 0x00, 0xbf, 0x80, 0x00, 0x00,
                           0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x80, 0x00, 0x00, 0x00, 0x7f, 0xc0, 0x00, 0x01});
    checkpoint.Add(std::move(record));
  }
  {
    TensorRecord record;
    record.name = "compat/bias";
    record.dtype = DType::kBf16;
    record.shape = {3};
    record.data = Payload({0x3f, 0x80, 0xbf, 0x80, 0x00, 0x00});
    checkpoint.Add(std::move(record));
  }

  CheckpointMetadata metadata;
  metadata.stage = Stage::kSft;
  metadata.step = 4294967298ull;  // 2^32 + 2
  metadata.world_size = 4;
  metadata.precision = ComputePrecision::kBf16;
  metadata.config = TinyConfig();
  metadata.tokenizer_id = "nanochat-32k";
  metadata.git_revision = "0123456789abcdef0123456789abcdef01234567";
  metadata.created_utc = "2025-06-01T00:00:00Z";
  metadata.notes = {"compat: written by the version 1 writer",
                    "compat: read by the version 1 reader"};
  checkpoint.set_metadata(metadata);
  return checkpoint;
}

// The metadata block as an older release would have written it: everything the
// version 1 format carries except the last two fields, `created_utc` and
// `notes` (rule 1: a new field goes last, so these two are the newest ones).
// This is the `OldFields` half of the `gemma.cpp` trick; the `NewFields` half
// is `CheckpointMetadata` itself. The field order matches
// `CheckpointMetadata::VisitFields` up to `git_revision` and then stops.
struct OldMetadata : public IFields {
  const char* Name() const override { return "OldCheckpointMetadata"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(schema);
    visitor(stage);
    visitor(step);
    visitor(world_size);
    visitor(precision);
    nanochat::ConfigBlock config_block(config);
    visitor(config_block);
    visitor(tokenizer_id);
    visitor(git_revision);
    // `created_utc` and `notes` do not exist in this release.
  }

  std::uint32_t schema = nanochat::kMetadataSchema;
  Stage stage = Stage::kBase;
  std::uint64_t step = 0;
  std::uint32_t world_size = 1;
  ComputePrecision precision = ComputePrecision::kFp32;
  Config config;
  std::string tokenizer_id;
  std::string git_revision;
};

// --- Cases ------------------------------------------------------------------

void ExpectLoadFails(const std::string& path, const std::string& what,
                     const std::string& error_needle = "") {
  Checkpoint checkpoint;
  std::string error;
  if (checkpoint.Load(path, &error)) {
    Fail(what + ": Checkpoint::Load accepted a malformed file");
    return;
  }
  Expect(!error.empty(), what + ": Load returned false with no error string");
  if (!error_needle.empty()) {
    Expect(error.find(error_needle) != std::string::npos,
           what + ": the error does not mention \"" + error_needle +
               "\": " + error);
  }
}

// Case 1 of the section 5 table: the committed version 0 fixture, read with the
// shipped reader. Every expectation below is a row of
// `tests/data/checkpoint_v0.txt`.
void TestVersionZeroFixture(const std::string& path) {
  Expect(nanochat::IsCheckpointContainer(path),
         "fixture: IsCheckpointContainer rejected the fixture");
  Expect(nanochat::DetectFormat(path) == CheckpointFormat::kContainerV0,
         "fixture: DetectFormat did not report a version 0 container");
  Expect(FileSize(path) == kFixtureSize,
         "fixture: the file size changed (" + std::to_string(FileSize(path)) +
             " != " + std::to_string(kFixtureSize) + ")");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("fixture: Checkpoint::Load failed: " + error);
    return;
  }

  const std::vector<TensorRecord>& records = loaded.tensors();
  Expect(records.size() == 2, "fixture: the file holds " +
                                  std::to_string(records.size()) +
                                  " record(s), not 2");
  if (records.size() != 2) return;

  // `fixture/weights`: float32, (2, 3), 24 payload bytes.
  Expect(records[0].name == "fixture/weights", "fixture: record 0 is \"" +
                                                   records[0].name +
                                                   "\", not "
                                                   "\"fixture/weights\"");
  Expect(records[0].dtype == DType::kFp32,
         "fixture: record 0 is not stored as float32");
  Expect(records[0].shape == std::vector<std::int64_t>({2, 3}),
         "fixture: record 0 is not shaped (2, 3)");
  Expect(HexString(records[0].data) == kFixtureWeightsHex,
         "fixture: the weights payload changed (" + HexString(records[0].data) +
             ")");
  Expect(Fnv1a64(records[0].data) == kFixtureWeightsSum,
         "fixture: the weights checksum changed (" +
             Hex(Fnv1a64(records[0].data)) + " != " + Hex(kFixtureWeightsSum) +
             ")");

  // `fixture/bias`: bfloat16, (3,), 6 payload bytes.
  Expect(records[1].name == "fixture/bias", "fixture: record 1 is \"" +
                                                records[1].name +
                                                "\", not "
                                                "\"fixture/bias\"");
  Expect(records[1].dtype == DType::kBf16,
         "fixture: record 1 is not stored as bfloat16");
  Expect(records[1].shape == std::vector<std::int64_t>({3}),
         "fixture: record 1 is not shaped (3,)");
  Expect(
      HexString(records[1].data) == kFixtureBiasHex,
      "fixture: the bias payload changed (" + HexString(records[1].data) + ")");
  Expect(Fnv1a64(records[1].data) == kFixtureBiasSum,
         "fixture: the bias checksum changed (" +
             Hex(Fnv1a64(records[1].data)) + " != " + Hex(kFixtureBiasSum) +
             ")");

  // The note's concatenated checksum, over both payloads in record order.
  std::vector<std::byte> concatenated = records[0].data;
  concatenated.insert(concatenated.end(), records[1].data.begin(),
                      records[1].data.end());
  Expect(Fnv1a64(concatenated) == kFixtureConcatSum,
         "fixture: the concatenated payload checksum changed (" +
             Hex(Fnv1a64(concatenated)) + " != " + Hex(kFixtureConcatSum) +
             ")");

  // Both records carry no attribute block, and the file is parameter-only.
  Expect(records[0].attributes.empty() && records[1].attributes.empty(),
         "fixture: a record grew an attribute block");
  Expect(loaded.optimizer_state().empty(),
         "fixture: the parameter-only fixture carries optimizer state");

  // A version 0 file has no metadata section, so the block reads as the member
  // initializers and every field is still defined (rule 1).
  const CheckpointMetadata metadata = loaded.metadata();
  Expect(metadata.schema == nanochat::kMetadataSchema,
         "fixture: the metadata schema is not the initializer");
  Expect(metadata.stage == Stage::kBase,
         "fixture: the metadata stage is not the initializer (base)");
  Expect(metadata.step == 0,
         "fixture: the metadata step is not the initializer");
  Expect(metadata.world_size == 1,
         "fixture: the metadata world size is not the initializer");
  // The initializer is the fixed `kFp32` and not `BuildComputePrecision()`, so
  // the same file reports the same metadata in the bf16 and fp8 builds
  // (section 8's precision gate).
  Expect(metadata.precision == ComputePrecision::kFp32,
         "fixture: the recorded precision is not the fixed fp32 initializer");
  Expect(SameConfig(metadata.config, Config()),
         "fixture: the metadata config is not the initializer");
  Expect(metadata.tokenizer_id.empty() && metadata.git_revision.empty() &&
             metadata.created_utc.empty() && metadata.notes.empty(),
         "fixture: a metadata string or the notes is not the initializer");
}

// Case 2: write version 1, read version 1, every field equal.
void TestVersionOneRoundTrip() {
  const Checkpoint written = MakeContainer();
  const std::string path = TempPath("compat_v1_roundtrip.ckpt");
  if (!written.Save(path)) {
    Fail("round trip: Checkpoint::Save failed");
    return;
  }
  Expect(nanochat::DetectFormat(path) == CheckpointFormat::kContainerV1,
         "round trip: DetectFormat did not report a version 1 container");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("round trip: Checkpoint::Load failed: " + error);
    return;
  }
  ExpectSameMetadata(loaded.metadata(), written.metadata(), "round trip");
  ExpectSameRecords(loaded, written, "round trip");
  // Named explicitly, because a swapped word pair (O6) would still be a
  // plausible-looking step.
  Expect(loaded.metadata().step == 4294967298ull,
         "round trip: the two-word step did not survive (" +
             std::to_string(loaded.metadata().step) + ")");
  Expect(loaded.metadata().config.window_pattern == "SL",
         "round trip: the config string did not survive");
}

// Case 3: three trailing words this build does not know. The read succeeds, the
// counter counts them, and the records are unchanged (design decision D5).
void TestExtraTrailingWords() {
  const Checkpoint written = MakeContainer();
  const std::string base_path = TempPath("compat_extra_base.ckpt");
  if (!written.Save(base_path)) {
    Fail("extra words: Checkpoint::Save failed");
    return;
  }
  std::vector<char> tail;
  if (!ReadTailAfterMetadata(base_path, &tail)) {
    Fail("extra words: cannot read the record stream of the saved file");
    return;
  }

  // A newer release appended a field: three words this build does not read.
  std::vector<std::uint32_t> words = written.metadata().Write();
  if (words.empty()) {
    Fail("extra words: the metadata block did not encode");
    return;
  }
  words.push_back(0xa5a5a5a5u);
  words.push_back(0x5a5a5a5au);
  words.push_back(0x12345678u);

  // The counter, from the visitor over the same word stream the container
  // reads: `extra_u32` is the number of words beyond what `VisitFields` read.
  CheckpointMetadata read_block;
  const IFields::ReadResult result = read_block.Read(words, 0);
  Expect(result.pos != 0, "extra words: the block did not read");
  Expect(result.extra_u32 == 3,
         "extra words: extra_u32 is " + Count(result.extra_u32) + ", not 3");
  Expect(result.missing_fields == 0,
         "extra words: the read reported missing fields too");
  ExpectSameMetadata(read_block, written.metadata(), "extra words");

  // The same block inside a file: `Load` succeeds (the container warns once on
  // stderr) and the record stream is untouched.
  const std::string path = TempPath("compat_extra_words.ckpt");
  if (!WriteContainerWithMetadata(path, words, tail)) {
    Fail("extra words: cannot write the spliced container");
    return;
  }
  Expect(nanochat::DetectFormat(path) == CheckpointFormat::kContainerV1,
         "extra words: DetectFormat did not report a version 1 container");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("extra words: Checkpoint::Load failed: " + error);
    return;
  }
  ExpectSameMetadata(loaded.metadata(), written.metadata(), "extra words");
  ExpectSameRecords(loaded, written, "extra words");
}

// Case 4: a metadata block that lacks the last two fields. The read succeeds,
// `missing_fields` counts them, and the two members keep their initializers
// (design decision D6).
void TestMissingTrailingFields() {
  const Checkpoint written = MakeContainer();
  const std::string base_path = TempPath("compat_missing_base.ckpt");
  if (!written.Save(base_path)) {
    Fail("missing fields: Checkpoint::Save failed");
    return;
  }
  std::vector<char> tail;
  if (!ReadTailAfterMetadata(base_path, &tail)) {
    Fail("missing fields: cannot read the record stream of the saved file");
    return;
  }

  OldMetadata older;
  older.stage = Stage::kRl;
  older.step = 9000000000ull;  // above 2^32, so both words are meaningful
  older.world_size = 2;
  older.precision = ComputePrecision::kFp8;
  older.config = TinyConfig();
  older.tokenizer_id = "nanochat-8k";
  older.git_revision = "fedcba9876543210fedcba9876543210fedcba98";
  const std::vector<std::uint32_t> words = older.Write();
  if (words.empty()) {
    Fail("missing fields: the older block did not encode");
    return;
  }

  // New code, old data: the fields the block carries are applied, the two it
  // does not carry keep their initializers, and the count is exactly two.
  CheckpointMetadata read;
  const IFields::ReadResult result = read.Read(words, 0);
  Expect(result.pos != 0, "missing fields: the block did not read");
  Expect(result.missing_fields == 2, "missing fields: missing_fields is " +
                                         Count(result.missing_fields) +
                                         ", not 2");
  Expect(result.extra_u32 == 0,
         "missing fields: the read reported trailing words too");
  Expect(read.stage == older.stage,
         "missing fields: the stage was not applied");
  Expect(read.step == older.step, "missing fields: the step was not applied");
  Expect(read.world_size == older.world_size,
         "missing fields: the world size was not applied");
  Expect(read.precision == older.precision,
         "missing fields: the precision was not applied");
  Expect(read.tokenizer_id == older.tokenizer_id,
         "missing fields: the tokenizer id was not applied");
  Expect(read.git_revision == older.git_revision,
         "missing fields: the git revision was not applied");
  Expect(SameConfig(read.config, older.config),
         "missing fields: the run config was not applied");
  Expect(read.created_utc.empty(),
         "missing fields: created_utc is not the initializer");
  Expect(read.notes.empty(), "missing fields: notes is not the initializer");

  // The same block inside a file: the load succeeds and reads the record
  // stream of the trailing bytes.
  const std::string path = TempPath("compat_missing_fields.ckpt");
  if (!WriteContainerWithMetadata(path, words, tail)) {
    Fail("missing fields: cannot write the spliced container");
    return;
  }
  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("missing fields: Checkpoint::Load failed: " + error);
    return;
  }
  Expect(loaded.metadata().step == older.step,
         "missing fields: the loaded step is not the older block's");
  Expect(loaded.metadata().stage == Stage::kRl,
         "missing fields: the loaded stage is not the older block's");
  Expect(
      loaded.metadata().created_utc.empty() && loaded.metadata().notes.empty(),
      "missing fields: a missing field did not keep its initializer");
  ExpectSameRecords(loaded, written, "missing fields");
}

// Cases 5 to 10: every malformed input is a hard error and `Load` returns
// false. Nothing here half-loads (design decision D7).
void TestHardErrors() {
  // A version 1 header that ends before the metadata word count.
  {
    const std::string path = TempPath("compat_no_metadata_count.ckpt");
    if (!WriteFileBytes(path, HeaderBytes(kContainerVersion1))) {
      Fail("hard error: cannot write the header-only fixture");
    } else {
      ExpectLoadFails(path, "a version 1 header with no metadata section",
                      "metadata");
    }
  }

  // A version 1 header whose metadata section is empty. A version 1 writer
  // always writes the block, so a zero count is a corrupted header and not a
  // file from an older writer.
  {
    std::vector<char> bytes = HeaderBytes(kContainerVersion1);
    AppendU32(&bytes, 0);
    const std::string path = TempPath("compat_empty_metadata.ckpt");
    if (!WriteFileBytes(path, bytes)) {
      Fail("hard error: cannot write the empty-metadata fixture");
    } else {
      ExpectLoadFails(path, "an empty metadata section", "metadata");
    }
  }

  // A metadata word count above the 64 KiB cap. The cap is checked before any
  // word is read or allocated.
  {
    std::vector<char> bytes = HeaderBytes(kContainerVersion1);
    AppendU32(&bytes, static_cast<std::uint32_t>(nanochat::kMaxBlockWords) + 1);
    const std::string path = TempPath("compat_huge_metadata.ckpt");
    if (!WriteFileBytes(path, bytes)) {
      Fail("hard error: cannot write the huge-metadata fixture");
    } else {
      ExpectLoadFails(path, "a metadata word count above the cap", "too large");
    }
  }

  // A metadata word count that promises more words than the file holds.
  {
    std::vector<char> bytes = HeaderBytes(kContainerVersion1);
    AppendU32(&bytes, 100);
    AppendU32(&bytes, 1);
    AppendU32(&bytes, 2);
    const std::string path = TempPath("compat_truncated_metadata.ckpt");
    if (!WriteFileBytes(path, bytes)) {
      Fail("hard error: cannot write the truncated-metadata fixture");
    } else {
      ExpectLoadFails(path, "a truncated metadata section", "truncated");
    }
  }

  // A version 0 header followed by metadata words. A version 0 file has no
  // metadata section, so those bytes are the parameter count and the record
  // stream; the read must fail instead of misparsing them into records. The
  // count word is read as the parameter count, and it promises one record per
  // metadata word, which leaves four bytes per record: less than the shortest
  // record framing. The read cannot succeed, whatever the block holds.
  {
    std::vector<char> bytes = HeaderBytes(kContainerVersion0);
    const std::vector<std::uint32_t> words = CheckpointMetadata().Write();
    AppendU32(&bytes, static_cast<std::uint32_t>(words.size()));
    for (std::uint32_t word : words) AppendU32(&bytes, word);
    const std::string path = TempPath("compat_v0_with_metadata.ckpt");
    if (!WriteFileBytes(path, bytes)) {
      Fail("hard error: cannot write the version 0 fixture");
    } else {
      ExpectLoadFails(path, "a version 0 header followed by metadata words");
    }
  }

  // The release gate open decision O5 leaves: the container magic beside a
  // version word this build does not know. There is no second magic, so this is
  // the one magic/version pair that can reach the reader, and it must fail with
  // a message that names the version rather than misparse the file
  // (section 11, definition of done item 2).
  {
    std::vector<char> bytes = HeaderBytes(kContainerVersion1 + 1);
    AppendU32(&bytes, 0);
    const std::string path = TempPath("compat_unsupported_version.ckpt");
    if (!WriteFileBytes(path, bytes)) {
      Fail("hard error: cannot write the unsupported-version fixture");
    } else {
      ExpectLoadFails(path, "an unsupported container version",
                      "unsupported checkpoint version");
    }
  }
}

// The precision gate of section 8. The same fixture must read identically under
// `--config=bf16` and `--config=fp8`, and the recorded precision value must
// never change a load decision. This function runs in all three builds, so the
// build precision is printed for the failure log.
void TestPrecisionGate(const std::string& fixture) {
  std::printf("checkpoint_compat_test: build precision %s\n",
              PrecisionName(nanochat::BuildComputePrecision()));

  // (a) The fixture read does not depend on the build. A version 0 file reports
  // the fixed `kFp32` initializer and not `BuildComputePrecision()`, and the
  // storage dtypes and payloads are the ones the file names.
  Checkpoint fixture_checkpoint;
  std::string error;
  if (!fixture_checkpoint.Load(fixture, &error)) {
    Fail("precision: the fixture did not load: " + error);
    return;
  }
  Expect(fixture_checkpoint.metadata().precision == ComputePrecision::kFp32,
         "precision: the recorded precision follows the build instead of the "
         "fixed fp32 initializer");
  if (fixture_checkpoint.tensors().size() == 2) {
    Expect(fixture_checkpoint.tensors()[0].dtype == DType::kFp32 &&
               fixture_checkpoint.tensors()[1].dtype == DType::kBf16,
           "precision: the fixture's storage dtypes follow the build");
    Expect(
        HexString(fixture_checkpoint.tensors()[0].data) == kFixtureWeightsHex,
        "precision: the weights payload depends on the build");
    Expect(HexString(fixture_checkpoint.tensors()[1].data) == kFixtureBiasHex,
           "precision: the bias payload depends on the build");
  } else {
    Fail("precision: the fixture record count is not 2");
  }

  // (b) The recorded value never changes a load decision: the same container
  // saved with each of the three values loads with the same records and the
  // same metadata apart from the field itself.
  const ComputePrecision all[] = {
      ComputePrecision::kFp32, ComputePrecision::kBf16, ComputePrecision::kFp8};
  for (ComputePrecision precision : all) {
    Checkpoint written = MakeContainer();
    CheckpointMetadata metadata = written.metadata();
    metadata.precision = precision;
    written.set_metadata(metadata);

    const std::string name = PrecisionName(precision);
    const std::string path =
        TempPath(std::string("compat_precision_") + name + ".ckpt");
    if (!written.Save(path)) {
      Fail("precision: Checkpoint::Save failed for " + name);
      continue;
    }
    Checkpoint loaded;
    std::string load_error;
    if (!loaded.Load(path, &load_error)) {
      Fail("precision: the recorded value \"" + name +
           "\" changed the load decision: " + load_error);
      continue;
    }
    Expect(loaded.metadata().precision == precision,
           "precision: the recorded value did not survive the round trip (" +
               name + ")");
    ExpectSameMetadata(loaded.metadata(), written.metadata(),
                       "precision " + name);
    ExpectSameRecords(loaded, written, "precision " + name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <path to tests/data/checkpoint_v0.ckpt>\n",
                 argv[0]);
    return 1;
  }
  const std::string fixture = argv[1];

  TestVersionZeroFixture(fixture);
  TestVersionOneRoundTrip();
  TestExtraTrailingWords();
  TestMissingTrailingFields();
  TestHardErrors();
  TestPrecisionGate(fixture);

  if (g_failures != 0) {
    std::fprintf(stderr, "checkpoint_compat_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("checkpoint_compat_test: ok\n");
  return 0;
}
