// src/data_test.cc — the `src/data.cc` containers.
//
// Round-trips a temporary token shard (both uint16 and uint32 widths), checks
// the `DataLoader` batch shape and the one-token targets shift, verifies the
// `Checkpoint` container round-trips name/dtype/shape/payload, and checks the
// out-of-scope `LoadTokenizer` contract (null, no artifact).

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

namespace {

using nanochat::Checkpoint;
using nanochat::DataLoader;
using nanochat::DType;
using nanochat::TensorRecord;
using nanochat::TokenShard;
using nanochat::TokenWidth;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

void WriteU16(std::ofstream& out, std::uint16_t value) {
  const unsigned char bytes[2] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), 2);
}

void WriteU32(std::ofstream& out, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu),
      static_cast<unsigned char>((value >> 16) & 0xffu),
      static_cast<unsigned char>((value >> 24) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), 4);
}

void WriteU64(std::ofstream& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    const unsigned char byte =
        static_cast<unsigned char>((value >> (8 * i)) & 0xffu);
    out.write(reinterpret_cast<const char*>(&byte), 1);
  }
}

// Writes a shard with the given token stream. `width` is 2 or 4.
std::string WriteShard(const std::string& name,
                       const std::vector<int>& tokens, int width) {
  const std::string path = TempPath(name);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  WriteU32(out, nanochat::kTokenShardMagic);
  WriteU32(out, nanochat::kTokenShardVersion);
  WriteU64(out, static_cast<std::uint64_t>(tokens.size()));
  WriteU32(out, static_cast<std::uint32_t>(width));
  WriteU32(out, 0);
  for (int token : tokens) {
    if (width == 2) {
      WriteU16(out, static_cast<std::uint16_t>(token));
    } else {
      WriteU32(out, static_cast<std::uint32_t>(token));
    }
  }
  return path;
}

void TestTokenShardRoundTrip() {
  const std::vector<int> tokens = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  for (int width : {2, 4}) {
    const std::string path =
        WriteShard(width == 2 ? "shard16.bin" : "shard32.bin", tokens, width);
    std::string error;
    std::unique_ptr<TokenShard> shard = TokenShard::Open(path, &error);
    if (shard == nullptr) {
      Fail("TokenShard::Open failed: " + error);
      continue;
    }
    if (shard->header().magic != nanochat::kTokenShardMagic) {
      Fail("shard magic mismatch");
    }
    if (shard->header().token_width != static_cast<std::uint32_t>(width)) {
      Fail("shard width mismatch");
    }
    if (shard->num_tokens() != tokens.size()) {
      Fail("shard token count mismatch");
    }
    std::vector<int> got(tokens.size(), -1);
    const std::int64_t read = shard->Read(0, 10, got.data());
    if (read != 10 || got != tokens) {
      Fail("shard full read mismatch");
    }
    std::vector<int> partial(4, -1);
    if (shard->Read(3, 4, partial.data()) != 4 ||
        partial != std::vector<int>({8, 9, 10, 11})) {
      Fail("shard partial read mismatch");
    }
    std::vector<int> tail(8, -1);
    if (shard->Read(8, 8, tail.data()) != 2 || tail[0] != 13 ||
        tail[1] != 14) {
      Fail("shard clamped tail read mismatch");
    }
  }

  // Bad magic must be rejected.
  const std::string bad = TempPath("shard_bad.bin");
  {
    std::ofstream out(bad, std::ios::binary | std::ios::trunc);
    const char junk[24] = {0};
    out.write(junk, 24);
  }
  std::string error;
  if (TokenShard::Open(bad, &error) != nullptr) {
    Fail("TokenShard accepted a bad magic");
  }
}

void TestDataLoader() {
  std::vector<int> tokens(32);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int>(i * 3 + 1);
  }
  const std::string path = WriteShard("loader.bin", tokens, 2);

  DataLoader loader({path}, 2, 4, 7, false);
  if (loader.batch() != 2 || loader.seq() != 4) Fail("loader shape mismatch");

  std::vector<int> batch(8);
  std::vector<int> targets(8);
  if (!loader.Next(batch.data(), targets.data())) Fail("first batch missing");
  for (int i = 0; i < 8; ++i) {
    if (batch[static_cast<std::size_t>(i)] != tokens[i]) {
      Fail("loader token mismatch at index " + std::to_string(i));
    }
    if (targets[static_cast<std::size_t>(i)] != tokens[i + 1]) {
      Fail("loader target shift mismatch at index " + std::to_string(i));
    }
  }

  // Second batch starts at offset 8.
  if (!loader.Next(batch.data(), targets.data())) Fail("second batch missing");
  if (batch[0] != tokens[8]) Fail("second batch offset mismatch");

  // Exhaustion: batches start at 0, 8, 16; a fourth window starting at 24
  // would need 33 tokens, so the stream is exhausted after three batches.
  if (!loader.Next(batch.data(), targets.data())) Fail("third batch missing");
  if (loader.Next(batch.data(), targets.data())) Fail("loader did not exhaust");

  // Reset rewinds to the first batch.
  loader.Reset();
  std::vector<int> again(8);
  std::vector<int> again_targets(8);
  if (!loader.Next(again.data(), again_targets.data()) ||
      again != std::vector<int>(tokens.begin(), tokens.begin() + 8)) {
    Fail("Reset did not rewind the loader");
  }

  // Shuffle with a fixed seed is deterministic, and Reset reproduces it.
  DataLoader shuffled({path}, 2, 4, 123, true);
  std::vector<int> first(8);
  std::vector<int> first_targets(8);
  if (!shuffled.Next(first.data(), first_targets.data())) {
    Fail("shuffled loader produced no batch");
  }
  shuffled.Reset();
  std::vector<int> second(8);
  std::vector<int> second_targets(8);
  if (!shuffled.Next(second.data(), second_targets.data()) || first != second) {
    Fail("shuffled loader is not reproducible");
  }

  // The optional byte table.
  const std::string bytes_path = path + ".bytes";
  {
    std::ofstream out(bytes_path, std::ios::binary | std::ios::trunc);
    WriteU32(out, 4);
    const unsigned char table[4] = {0, 1, 2, 3};
    out.write(reinterpret_cast<const char*>(table), 4);
  }
  DataLoader with_bytes({path}, 2, 4, 7, false);
  int vocab = 0;
  const std::uint8_t* table = with_bytes.token_bytes(&vocab);
  if (table == nullptr || vocab != 4 || table[0] != 0 || table[3] != 3) {
    Fail("token_bytes sidecar did not load");
  }
  std::remove(bytes_path.c_str());

  DataLoader without_bytes({path}, 2, 4, 7, false);
  vocab = 123;
  if (without_bytes.token_bytes(&vocab) != nullptr || vocab != 0) {
    Fail("token_bytes without a sidecar should be null");
  }
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
    record.name = "half";
    record.dtype = DType::kFp16;
    record.shape = {4};
    record.data.resize(8);
    for (std::size_t i = 0; i < record.data.size(); ++i) {
      record.data[i] = static_cast<std::byte>(i);
    }
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
  if (loaded.tensors().size() != 2) Fail("checkpoint record count mismatch");
  const TensorRecord* weight = loaded.Find("weight");
  if (weight == nullptr) {
    Fail("checkpoint is missing 'weight'");
  } else {
    if (weight->dtype != DType::kFp32 || weight->shape !=
                                               std::vector<std::int64_t>({2, 3})) {
      Fail("checkpoint 'weight' metadata mismatch");
    }
    const float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    if (weight->data.size() != sizeof(expected) ||
        std::memcmp(weight->data.data(), expected, sizeof(expected)) != 0) {
      Fail("checkpoint 'weight' payload mismatch");
    }
  }
  const TensorRecord* half = loaded.Find("half");
  if (half == nullptr || half->dtype != DType::kFp16 ||
      half->shape != std::vector<std::int64_t>({4})) {
    Fail("checkpoint 'half' metadata mismatch");
  }
  if (loaded.Find("missing") != nullptr) Fail("Find returned a missing record");

  Checkpoint absent;
  if (absent.Load(TempPath("does_not_exist.bin"), &error)) {
    Fail("Checkpoint::Load accepted a missing file");
  }
}

void TestTokenizer() {
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(TempPath("tokenizer.bin"));
  if (tokenizer != nullptr) {
    Fail("LoadTokenizer should return null (no artifact in tree)");
  }
}

}  // namespace

int main() {
  TestTokenShardRoundTrip();
  TestDataLoader();
  TestCheckpoint();
  TestTokenizer();
  if (g_failures != 0) {
    std::fprintf(stderr, "data_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("data_test: ok\n");
  return 0;
}
