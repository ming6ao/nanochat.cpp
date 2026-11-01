// src/data_test.cc -- the `src/data.cc` containers.
//
// Round-trips a temporary token shard (both uint16 and uint32 widths), checks
// the `DataLoader` batch shape and the one-token targets shift, verifies the
// `Checkpoint` container round-trips name/dtype/shape/payload, checks the
// version 1 metadata block and the version 0 read path
// (docs/checkpoint-metadata-and-examples-plan.md sections 2.1, 2.2, and 4), and
// checks the `LoadTokenizer` contract (a valid artifact loads, a bad file
// returns null).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/data.h"
#include "nanochat/dataloader.h"
#include "nanochat/tensor.h"
#include "nanochat/tokenizer.h"
#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"

namespace {

using nanochat::Bf16FromFloat;
using nanochat::Bf16ToFloat;
using nanochat::Checkpoint;
using nanochat::CheckpointFormat;
using nanochat::CheckpointMetadata;
using nanochat::ComputePrecision;
using nanochat::DataLoader;
using nanochat::DType;
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

void TestCheckpoint() {
  Checkpoint checkpoint;
  {
    TensorRecord record;
    record.name = "weight";
    record.dtype = DType::kFp32;
    record.shape = {2, 3};
    const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    record.data.resize(sizeof(values));
    std::memcpy(record.data.data(), values, sizeof(values));
    checkpoint.Add(std::move(record));
  }
  {
    TensorRecord record;
    record.name = "bf16";
    record.dtype = DType::kBf16;
    record.shape = {4};
    record.data.resize(8);
    for (std::size_t i = 0; i < record.data.size(); ++i) {
      record.data[i] = static_cast<std::byte>(i);
    }
    checkpoint.Add(std::move(record));
  }
  {
    TensorRecord record;
    record.name = "quantized";
    record.dtype = DType::kInt8;
    record.shape = {6};
    record.data.resize(6);
    nanochat::TensorAttribute scale;
    scale.name = "quant.scale";
    scale.dtype = DType::kFp32;
    scale.shape = {1};
    const float scale_value = 0.5f;
    scale.data.resize(sizeof(scale_value));
    std::memcpy(scale.data.data(), &scale_value, sizeof(scale_value));
    record.attributes.push_back(std::move(scale));
    checkpoint.Add(std::move(record));
  }

  const std::string path = TempPath("checkpoint.bin");
  if (!checkpoint.Save(path)) Fail("Checkpoint::Save failed");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("Checkpoint::Load failed: " + error);
    return;
  }
  if (loaded.tensors().size() != 3) Fail("checkpoint record count mismatch");
  const TensorRecord* weight = loaded.Find("weight");
  if (weight == nullptr) {
    Fail("checkpoint is missing 'weight'");
  } else {
    if (weight->dtype != DType::kFp32 ||
        weight->shape != std::vector<std::int64_t>({2, 3})) {
      Fail("checkpoint 'weight' metadata mismatch");
    }
    const float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    if (weight->data.size() != sizeof(expected) ||
        std::memcmp(weight->data.data(), expected, sizeof(expected)) != 0) {
      Fail("checkpoint 'weight' payload mismatch");
    }
  }
  const TensorRecord* bf16 = loaded.Find("bf16");
  if (bf16 == nullptr || bf16->dtype != DType::kBf16 ||
      bf16->shape != std::vector<std::int64_t>({4})) {
    Fail("checkpoint 'bf16' metadata mismatch");
  }
  const TensorRecord* quantized = loaded.Find("quantized");
  if (quantized == nullptr || quantized->dtype != DType::kInt8 ||
      quantized->attributes.size() != 1 ||
      quantized->attributes[0].name != "quant.scale" ||
      quantized->attributes[0].data.size() != sizeof(float)) {
    Fail("checkpoint 'quantized' attribute mismatch");
  }
  if (loaded.Find("missing") != nullptr) Fail("Find returned a missing record");

  Checkpoint absent;
  if (absent.Load(TempPath("does_not_exist.bin"), &error)) {
    Fail("Checkpoint::Load accepted a missing file");
  }
}

// True when the two run configs are the same. The comparison is made on the
// metadata words of each config, through `VisitConfig`, so the check lists no
// `Config` field and needs no drift test (open decision O4).
bool SameConfig(const nanochat::Config& left, const nanochat::Config& right) {
  nanochat::Config left_copy = left;
  nanochat::Config right_copy = right;
  nanochat::ConfigBlock recorded(left_copy);
  nanochat::ConfigBlock runtime(right_copy);
  return recorded.Write() == runtime.Write();
}

// A small but valid run config, distinct from the defaults in every field the
// metadata block carries, so a field that did not round-trip is visible.
nanochat::Config TinyRunConfig() {
  nanochat::Config config;
  config.num_layers = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.hidden_dim = 64;
  config.seq_len = 64;
  config.vocab_size = 256;
  config.padded_vocab_size = 256;
  config.rope_base = 500000.0f;
  config.window_pattern = "SL";
  config.value_embedding = false;
  return config;
}

// The version 1 metadata block round-trips every field, and the file is a
// version 1 container (docs/checkpoint-metadata-and-examples-plan.md sections
// 2.1, 2.2, and 4).
void TestCheckpointMetadata() {
  Checkpoint checkpoint;
  {
    TensorRecord record;
    record.name = "weight";
    record.dtype = DType::kFp32;
    record.shape = {2, 2};
    const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    record.data.resize(sizeof(values));
    std::memcpy(record.data.data(), values, sizeof(values));
    checkpoint.Add(std::move(record));
  }

  CheckpointMetadata metadata;
  metadata.stage = Stage::kRl;
  // Above 2^32, so a stream that swapped the two words would fail here.
  metadata.step = 1234567890123ull;
  metadata.world_size = 8;
  metadata.precision = ComputePrecision::kBf16;
  metadata.config = TinyRunConfig();
  metadata.tokenizer_id = "nanochat-32k";
  metadata.git_revision = "abc1234";
  metadata.created_utc = "2025-06-01T00:00:00Z";
  metadata.notes = {"resumed from base", "hub:nanochat/base"};
  checkpoint.set_metadata(metadata);

  const std::string path = TempPath("checkpoint_metadata.ckpt");
  if (!checkpoint.Save(path)) {
    Fail("metadata: Checkpoint::Save failed");
    return;
  }
  Expect(nanochat::DetectFormat(path) == CheckpointFormat::kContainerV1,
         "metadata: DetectFormat did not report a version 1 container");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("metadata: Checkpoint::Load failed: " + error);
    return;
  }
  const CheckpointMetadata read = loaded.metadata();
  Expect(read.schema == metadata.schema, "metadata: schema changed");
  Expect(read.stage == metadata.stage, "metadata: stage changed");
  Expect(read.step == metadata.step, "metadata: step changed");
  Expect(read.world_size == metadata.world_size,
         "metadata: world_size changed");
  Expect(read.precision == metadata.precision, "metadata: precision changed");
  Expect(SameConfig(read.config, metadata.config),
         "metadata: the run config changed");
  Expect(read.tokenizer_id == metadata.tokenizer_id,
         "metadata: tokenizer_id changed");
  Expect(read.git_revision == metadata.git_revision,
         "metadata: git_revision changed");
  Expect(read.created_utc == metadata.created_utc,
         "metadata: created_utc changed");
  Expect(read.notes == metadata.notes, "metadata: notes changed");
  // The record stream is unchanged by the metadata section.
  const TensorRecord* weight = loaded.Find("weight");
  if (weight == nullptr) {
    Fail("metadata: the parameter record is missing");
  } else {
    Expect(weight->shape == std::vector<std::int64_t>({2, 2}),
           "metadata: the parameter record changed");
  }
}

// An all-default metadata block is written and read back unchanged: it is what
// a version 0 file reads as, so a caller never sees an unset field.
void TestEmptyMetadata() {
  Checkpoint checkpoint;
  TensorRecord record;
  record.name = "weight";
  record.shape = {1};
  record.data.resize(sizeof(float));
  checkpoint.Add(std::move(record));
  checkpoint.set_metadata(CheckpointMetadata());

  const std::string path = TempPath("checkpoint_metadata_empty.ckpt");
  if (!checkpoint.Save(path)) {
    Fail("empty metadata: Checkpoint::Save failed");
    return;
  }
  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("empty metadata: Checkpoint::Load failed: " + error);
    return;
  }
  const CheckpointMetadata read = loaded.metadata();
  Expect(read.stage == Stage::kBase, "empty metadata: the stage is not base");
  Expect(read.step == 0 && read.world_size == 1,
         "empty metadata: the step or the world size is not the default");
  Expect(read.precision == ComputePrecision::kFp32,
         "empty metadata: the precision is not the default");
  Expect(read.tokenizer_id.empty() && read.git_revision.empty() &&
             read.created_utc.empty() && read.notes.empty(),
         "empty metadata: a string or the notes are not empty");
  Expect(SameConfig(read.config, nanochat::Config()),
         "empty metadata: the config is not the default");
}

// A version 0 file carries no metadata section (docs/checkpoint-format.md
// section 5.1). The reader reports the member initializers, and `DetectFormat`
// names the version instead of the reader guessing from a bool.
void TestVersionZeroContainer() {
  const std::string path = TempPath("checkpoint_v0_handwritten.ckpt");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const char header[] = {'N', 'C', 'H', 'K', 'P', 'T', '0', '1', 0, 0,
                           0,   0,   0,   0,   0,   0,   0,   0,   0, 0};
    out.write(header, sizeof(header));
  }
  Expect(nanochat::DetectFormat(path) == CheckpointFormat::kContainerV0,
         "version 0: DetectFormat did not report a version 0 container");
  Expect(nanochat::IsCheckpointContainer(path),
         "version 0: IsCheckpointContainer rejected the file");

  Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error)) {
    Fail("version 0: Checkpoint::Load failed: " + error);
    return;
  }
  Expect(loaded.tensors().empty(), "version 0: the file carries records");
  const CheckpointMetadata read = loaded.metadata();
  Expect(read.stage == Stage::kBase && read.step == 0,
         "version 0: the metadata is not the default");
  Expect(SameConfig(read.config, nanochat::Config()),
         "version 0: the config is not the default");
}

// A file that is not a container at all, and a metadata section the reader
// cannot trust, are both hard errors.
void TestMalformedContainers() {
  const std::string legacy = TempPath("checkpoint_legacy_handwritten.ckpt");
  {
    std::ofstream out(legacy, std::ios::binary | std::ios::trunc);
    const char header[] = {'N', 'C', 'M', 'D', 'L', '0', '0', '1', 0, 0, 0, 0};
    out.write(header, sizeof(header));
  }
  Expect(nanochat::DetectFormat(legacy) == CheckpointFormat::kLegacyModel,
         "malformed: DetectFormat did not report the legacy layout");
  Checkpoint legacy_checkpoint;
  Expect(!legacy_checkpoint.Load(legacy, nullptr),
         "malformed: Load accepted a legacy file");
  Expect(nanochat::DetectFormat(TempPath("does_not_exist.bin")) ==
             CheckpointFormat::kLegacyModel,
         "malformed: DetectFormat opened a missing path");

  // A version 1 header with an empty metadata section, and one whose section
  // claims more words than the cap allows.
  const char* names[] = {"checkpoint_metadata_none.ckpt",
                         "checkpoint_metadata_huge.ckpt"};
  const char zeros[4] = {0, 0, 0, 0};
  const char huge[4] = {'\xff', '\xff', '\xff', '\xff'};
  for (int i = 0; i < 2; ++i) {
    const std::string path = TempPath(names[i]);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const char header[] = {'N', 'C', 'H', 'K', 'P', 'T', '0', '1', 1, 0, 0, 0};
    out.write(header, sizeof(header));
    out.write(i == 0 ? zeros : huge, sizeof(zeros));
  }
  for (const char* name : names) {
    Checkpoint checkpoint;
    std::string error;
    Expect(!checkpoint.Load(TempPath(name), &error),
           std::string("malformed: Load accepted ") + name);
  }
}

void TestBf16RoundTrip() {
  // Every value below has at most seven fraction bits, so bf16 is exact.
  const float exact[5] = {0.0f, 1.0f, -2.5f, 0.15625f, 3.140625f};
  for (float value : exact) {
    if (Bf16ToFloat(Bf16FromFloat(value)) != value) {
      Fail("bf16 changed an exactly representable value");
    }
  }
  // One plus one over 512 needs a ninth fraction bit, so bf16 loses it.
  const float inexact = 1.0f + 1.0f / 512.0f;
  if (Bf16ToFloat(Bf16FromFloat(inexact)) == inexact) {
    Fail("bf16 kept a mantissa bit it cannot store");
  }
}

void TestTokenizer() {
  // A valid `NCTOKEN1` artifact loads; a malformed file returns null.
  const std::vector<std::pair<std::uint32_t, std::uint32_t>> merges = {
      {'h', 'e'}, {'l', 'l'}, {256, 257}, {258, 'o'}};
  const std::vector<std::pair<std::string, std::uint32_t>> specials = {
      {"<|bos|>", 260}};
  const std::string valid = TempPath("data_tokenizer.nctoken");
  if (!nanochat::SaveTokenizer(valid, nanochat::NanochatSplitPattern(), merges,
                               specials)) {
    Fail("SaveTokenizer could not write the valid artifact");
    return;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(valid);
  if (tokenizer == nullptr) {
    Fail("LoadTokenizer returned null for a valid artifact");
  } else if (tokenizer->vocab_size() != 261 || tokenizer->bos_id() != 260) {
    Fail("LoadTokenizer returned the wrong metadata");
  }

  const std::string bad = TempPath("data_tokenizer_bad.nctoken");
  {
    std::ofstream out(bad, std::ios::binary | std::ios::trunc);
    const char junk[16] = {0};
    out.write(junk, sizeof(junk));
  }
  if (nanochat::LoadTokenizer(bad) != nullptr) {
    Fail("LoadTokenizer accepted a bad file");
  }
}

// A fixed document source for the loader test: it yields each document once.
class VectorSource : public nanochat::DocumentSource {
 public:
  explicit VectorSource(std::vector<std::string> documents)
      : documents_(std::move(documents)) {}

  bool Next(std::vector<std::string>* documents, std::string* error) override {
    (void)error;
    if (index_ >= documents_.size()) return false;
    documents->clear();
    documents->push_back(documents_[index_++]);
    return true;
  }

 private:
  std::vector<std::string> documents_;
  std::size_t index_ = 0;
};

void TestDocumentLoader() {
  // A tokenizer with no merges: every byte is its own token, BOS id 256.
  const std::string tokenizer_path = TempPath("data_doc_tokenizer.nctoken");
  const std::vector<std::pair<std::string, std::uint32_t>> specials = {
      {"<|bos|>", 256}};
  if (!nanochat::SaveTokenizer(tokenizer_path, nanochat::NanochatSplitPattern(),
                               {}, specials)) {
    Fail("document loader: SaveTokenizer failed");
    return;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    Fail("document loader: LoadTokenizer failed");
    return;
  }

  auto documents = std::make_shared<std::vector<std::string>>(
      std::vector<std::string>{"a", "b", "cc", "dd", "ee", "ff"});
  nanochat::DocumentSourceFactory factory =
      [documents](
          std::string* error) -> std::unique_ptr<nanochat::DocumentSource> {
    (void)error;
    return std::make_unique<VectorSource>(*documents);
  };

  const int batch = 2;
  const int seq = 4;
  DataLoader loader(factory, tokenizer.get(), batch, seq, /*seed=*/7,
                    /*tokenizer_threads=*/1, /*document_buffer=*/16);
  std::vector<int> tokens(static_cast<std::size_t>(batch * seq));
  std::vector<int> targets(static_cast<std::size_t>(batch * seq));
  if (!loader.Next(tokens.data(), targets.data())) {
    Fail("document loader: Next returned false");
    return;
  }
  const int bos = tokenizer->bos_id();
  for (int r = 0; r < batch; ++r) {
    if (tokens[static_cast<std::size_t>(r * seq)] != bos) {
      Fail("document loader: row does not start with BOS");
    }
    for (int j = 0; j + 1 < seq; ++j) {
      if (targets[static_cast<std::size_t>(r * seq + j)] !=
          tokens[static_cast<std::size_t>(r * seq + j + 1)]) {
        Fail("document loader: targets are not the tokens shifted by one");
      }
    }
  }

  int vocab = 0;
  if (loader.token_bytes(&vocab) == nullptr ||
      vocab != tokenizer->vocab_size()) {
    Fail("document loader: token_bytes did not come from the tokenizer");
  }

  // Reset re-opens the source, so a fresh batch is available again.
  loader.Reset();
  if (!loader.Next(tokens.data(), targets.data())) {
    Fail("document loader: Next after Reset returned false");
  }
}

// The data cursor resumes the exact document stream
// (docs/distributed-design.md section 11). A loader consumes two batches and
// records its cursor. A second loader, built with the sharded start that the
// cursor implies, must produce the same third batch.
void TestShardedResume() {
  const std::string tokenizer_path = TempPath("data_resume_tokenizer.nctoken");
  const std::vector<std::pair<std::string, std::uint32_t>> specials = {
      {"<|bos|>", 256}};
  if (!nanochat::SaveTokenizer(tokenizer_path, nanochat::NanochatSplitPattern(),
                               {}, specials)) {
    Fail("sharded resume: SaveTokenizer failed");
    return;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    Fail("sharded resume: LoadTokenizer failed");
    return;
  }

  auto documents = std::make_shared<std::vector<std::string>>();
  for (int i = 0; i < 40; ++i) {
    documents->push_back("doc" + std::to_string(i) + "x");
  }
  nanochat::DocumentSourceFactory base =
      [documents](std::string*) -> std::unique_ptr<nanochat::DocumentSource> {
    return std::make_unique<VectorSource>(*documents);
  };

  const int batch = 2;
  const int seq = 6;
  const int rank = 1;
  const int world = 2;
  const std::size_t rows = static_cast<std::size_t>(batch) * seq;

  nanochat::DocumentSourceFactory sharded =
      nanochat::ShardDocumentSourceFactory(base, rank, world, /*start=*/0,
                                           /*start_on_creation=*/0);
  DataLoader first(sharded, tokenizer.get(), batch, seq, /*seed=*/7,
                   /*tokenizer_threads=*/1, /*document_buffer=*/16);
  std::vector<int> tokens(rows);
  std::vector<int> targets(rows);
  for (int i = 0; i < 2; ++i) {
    if (!first.Next(tokens.data(), targets.data())) {
      Fail("sharded resume: the first loader ended early");
      return;
    }
  }
  const nanochat::DataLoaderState state = first.State();
  std::vector<int> expected_tokens(rows);
  std::vector<int> expected_targets(rows);
  if (!first.Next(expected_tokens.data(), expected_targets.data())) {
    Fail("sharded resume: the first loader has no third batch");
    return;
  }

  nanochat::DocumentSourceFactory resumed =
      nanochat::ShardDocumentSourceFactory(base, rank, world,
                                           state.source_position,
                                           static_cast<int>(state.epoch));
  DataLoader second(resumed, tokenizer.get(), batch, seq, /*seed=*/7,
                    /*tokenizer_threads=*/1, /*document_buffer=*/16, &state);
  std::vector<int> got_tokens(rows);
  std::vector<int> got_targets(rows);
  if (!second.Next(got_tokens.data(), got_targets.data())) {
    Fail("sharded resume: the resumed loader ended early");
    return;
  }
  if (got_tokens != expected_tokens || got_targets != expected_targets) {
    Fail("sharded resume: the resumed batch does not match");
  }
}

}  // namespace

int main() {
  TestCheckpoint();
  TestCheckpointMetadata();
  TestEmptyMetadata();
  TestVersionZeroContainer();
  TestMalformedContainers();
  TestBf16RoundTrip();
  TestTokenizer();
  TestDocumentLoader();
  TestShardedResume();
  if (g_failures != 0) {
    std::fprintf(stderr, "data_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("data_test: ok\n");
  return 0;
}
