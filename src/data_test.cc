// src/data_test.cc -- the `src/data.cc` containers.
//
// Round-trips a temporary token shard (both uint16 and uint32 widths), checks
// the `DataLoader` batch shape and the one-token targets shift, verifies the
// `Checkpoint` container round-trips name/dtype/shape/payload, and checks the
// `LoadTokenizer` contract (a valid artifact loads, a bad file returns null).

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

using nanochat::Checkpoint;
using nanochat::DataLoader;
using nanochat::DType;
using nanochat::TensorRecord;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
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
