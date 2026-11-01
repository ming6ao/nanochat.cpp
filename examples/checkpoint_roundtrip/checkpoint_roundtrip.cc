// The smallest container example: build two records, fill the run metadata,
// save a version 1 checkpoint, load it back, and print the metadata block and
// the record table. It proves the container works outside src/ through the
// public headers alone, and it is executable documentation for the metadata
// work, because it fills `CheckpointMetadata` through the same struct `Save`
// takes rather than a shortcut (docs/checkpoint-metadata-and-examples-plan.md
// section 6, decisions E1-E6, stage B.2).
//
// Run it with:
//   tools/nanochat run t0-cpu --
//   bazel-bin/examples/checkpoint_roundtrip/checkpoint_roundtrip
//
// To try a larger config, change this one line in TinyConfig() (keep head_dim
// even and num_kv_heads a divisor of num_heads):
//   config.hidden_dim = 48;
//
// The same source is also `//examples/checkpoint_roundtrip:roundtrip_test`. For
// the two fixed records it pins the record count and the presence of both the
// metadata block and the record table; the exact text stays unpinned, because
// a formatting change may move it (stage B.4).

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/checkpoint_metadata.h"
#include "nanochat/data.h"
#include "nanochat/mfu.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

// The fixed seed of the example: no model file, and no arbitrary payload.
constexpr std::uint64_t kSeed = 20240517;
constexpr std::size_t kRecords = 2;

// A tiny architecture that passes Config::Validate: hidden_dim is at least 24,
// head_dim is even (32 / 4), num_kv_heads divides num_heads (4 / 2), and
// padded_vocab_size equals vocab_size.
nanochat::Config TinyConfig() {
  nanochat::Config config;
  config.num_layers = 2;
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 16;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

// A reproducible payload value in [0, 1) from the fixed seed.
float Value(std::int64_t index) {
  return static_cast<float>((kSeed * static_cast<std::uint64_t>(index + 1)) %
                            97) /
         97.0f;
}

// Two records: six fp32 values in two rows of three, and four bf16 values, so
// the table covers both storage dtypes the container writes today.
std::vector<nanochat::TensorRecord> BuildRecords() {
  nanochat::TensorRecord fp32;
  fp32.name = "weight";
  fp32.dtype = nanochat::DType::kFp32;
  fp32.shape = {2, 3};
  fp32.data.resize(fp32.numel() * sizeof(float));
  for (std::int64_t i = 0; i < fp32.numel(); ++i) {
    const float value = Value(i);
    std::memcpy(fp32.data.data() + i * sizeof(value), &value, sizeof(value));
  }

  nanochat::TensorRecord bf16;
  bf16.name = "norm";
  bf16.dtype = nanochat::DType::kBf16;
  bf16.shape = {4};
  bf16.data.resize(bf16.numel() * sizeof(std::uint16_t));
  for (std::int64_t i = 0; i < bf16.numel(); ++i) {
    const std::uint16_t bits = nanochat::Bf16FromFloat(Value(i + 6)).bits;
    std::memcpy(bf16.data.data() + i * sizeof(bits), &bits, sizeof(bits));
  }
  return {std::move(fp32), std::move(bf16)};
}

const char* StageName(nanochat::Stage stage) {
  switch (stage) {
    case nanochat::Stage::kBase:
      return "base";
    case nanochat::Stage::kSft:
      return "sft";
    case nanochat::Stage::kRl:
      return "rl";
  }
  return "unknown";
}

const char* FormatName(nanochat::CheckpointFormat format) {
  switch (format) {
    case nanochat::CheckpointFormat::kLegacyModel:
      return "legacy-model";
    case nanochat::CheckpointFormat::kContainerV0:
      return "container-v0";
    case nanochat::CheckpointFormat::kContainerV1:
      return "container-v1";
  }
  return "unknown";
}

// The three container storage dtypes plus the two GEMM operand tags.
const char* DTypeName(nanochat::DType dtype) {
  switch (dtype) {
    case nanochat::DType::kFp32:
      return "fp32";
    case nanochat::DType::kBf16:
      return "bf16";
    case nanochat::DType::kInt8:
      return "int8";
    case nanochat::DType::kFp8E4M3:
      return "fp8e4m3";
    case nanochat::DType::kFp8E5M2:
      return "fp8e5m2";
  }
  return "unknown";
}

// "2x3", so the table prints a shape as one field.
std::string ShapeText(const std::vector<std::int64_t>& shape) {
  std::string text;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) text += "x";
    text += std::to_string(shape[i]);
  }
  return text;
}

// One line of the report.
void AppendLine(std::string* out, const char* format, ...) {
  char buffer[512];
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (written > 0) {
    const std::size_t length = static_cast<std::size_t>(written);
    out->append(buffer, length < sizeof(buffer) ? length : sizeof(buffer) - 1);
  }
  out->push_back('\n');
}

// `TEST_TMPDIR` is set under `tools/nanochat test`; `/tmp` keeps a `run` from
// writing a checkpoint into the checkout.
std::string TempPath(const char* name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

// The metadata block of the report, field by field.
void AppendMetadata(std::string* report,
                    const nanochat::CheckpointMetadata& metadata) {
  const nanochat::Config& config = metadata.config;
  AppendLine(report, "metadata:");
  AppendLine(report, "  schema: %u", static_cast<unsigned>(metadata.schema));
  AppendLine(report, "  stage: %s", StageName(metadata.stage));
  AppendLine(report, "  step: %llu",
             static_cast<unsigned long long>(metadata.step));
  AppendLine(report, "  world_size: %u",
             static_cast<unsigned>(metadata.world_size));
  AppendLine(report, "  precision: %s",
             nanochat::PrecisionName(metadata.precision));
  AppendLine(report,
             "  config: layers=%d heads=%d kv_heads=%d hidden=%d seq_len=%d "
             "vocab=%d padded=%d window=%s",
             config.num_layers, config.num_heads, config.num_kv_heads,
             config.hidden_dim, config.seq_len, config.vocab_size,
             config.padded_vocab_size, config.window_pattern.c_str());
  AppendLine(report, "  tokenizer_id: %s", metadata.tokenizer_id.c_str());
  AppendLine(report, "  git_revision: %s", metadata.git_revision.c_str());
  AppendLine(report, "  created_utc: %s", metadata.created_utc.c_str());
  AppendLine(report, "  notes: %zu", metadata.notes.size());
  for (const std::string& note : metadata.notes) {
    AppendLine(report, "    %s", note.c_str());
  }
}

// The record table of the report: one row per record.
void AppendRecords(std::string* report,
                   const std::vector<nanochat::TensorRecord>& records) {
  AppendLine(report, "records:");
  for (const nanochat::TensorRecord& record : records) {
    AppendLine(report, "  %-8s %-5s %-6s %3zu bytes", record.name.c_str(),
               DTypeName(record.dtype), ShapeText(record.shape).c_str(),
               record.data.size());
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("example");

  const std::string path = TempPath("checkpoint_roundtrip.ckpt");

  // The save side: the two records and the run metadata, through the public
  // API.
  nanochat::Checkpoint saved;
  for (nanochat::TensorRecord& record : BuildRecords()) {
    saved.Add(std::move(record));
  }
  nanochat::CheckpointMetadata metadata;
  metadata.stage = nanochat::Stage::kSft;
  metadata.step = kSeed;
  metadata.world_size = 1;
  // The recorded precision is informational: it never changes a load decision.
  metadata.precision = nanochat::BuildComputePrecision();
  metadata.config = TinyConfig();
  metadata.tokenizer_id = "tiny-64";
  metadata.git_revision = "informational";
  metadata.created_utc = "1970-01-01T00:00:00Z";
  metadata.notes = {"checkpoint_roundtrip example"};
  saved.set_metadata(metadata);

  if (!saved.Save(path)) {
    std::fprintf(stderr, "checkpoint_roundtrip: save failed\n");
    return 1;
  }

  // The load side: a parameter-only read, then the metadata and the records.
  nanochat::Checkpoint loaded;
  std::string error;
  if (!loaded.Load(path, &error, /*with_optimizer_state=*/false)) {
    std::fprintf(stderr, "checkpoint_roundtrip: load failed: %s\n",
                 error.c_str());
    std::remove(path.c_str());
    return 1;
  }

  std::string report;
  AppendLine(&report, "file: %s", path.c_str());
  AppendLine(&report, "format: %s", FormatName(nanochat::DetectFormat(path)));
  AppendMetadata(&report, loaded.metadata());
  AppendRecords(&report, loaded.tensors());
  std::fputs(report.c_str(), stdout);
  std::remove(path.c_str());

  // The stage B.4 guard: a fixed record count for the fixed records, both
  // sections present, and the values that came back are the ones written.
  const nanochat::CheckpointMetadata read_back = loaded.metadata();
  const nanochat::TensorRecord* weight = loaded.Find("weight");
  int failures = loaded.tensors().size() != kRecords ? 1 : 0;
  failures += report.find("metadata:") == std::string::npos ? 1 : 0;
  failures += report.find("records:") == std::string::npos ? 1 : 0;
  failures += read_back.stage != metadata.stage ? 1 : 0;
  failures += read_back.step != metadata.step ? 1 : 0;
  failures += read_back.config.hidden_dim != metadata.config.hidden_dim ? 1 : 0;
  failures += read_back.tokenizer_id != metadata.tokenizer_id ? 1 : 0;
  failures += weight == nullptr || weight->data != saved.Find("weight")->data;
  if (failures != 0) {
    std::fprintf(stderr, "checkpoint_roundtrip: %d check(s) failed\n",
                 failures);
    return 1;
  }
  return 0;
}
