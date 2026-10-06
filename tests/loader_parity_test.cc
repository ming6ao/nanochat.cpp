// tests/loader_parity_test.cc -- the document-mode loader parity gate
// (docs/parquet-native.md phase 8).
//
// Reads the committed fixture that `tools/dump_loader_fixture.py` produced from
// the reference `tokenizing_distributed_data_loader_bos_bestfit`
// (`nanochat/dataloader.py`). The fixture holds the documents, the loader
// shape, and the reference rows. The test rebuilds the loader over a fixed
// document source and checks the rows token for token. It never imports
// Python.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/tokenizer.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

bool ReadBytes(std::istream& in, void* out, std::size_t count) {
  in.read(static_cast<char*>(out), static_cast<std::streamsize>(count));
  return static_cast<std::size_t>(in.gcount()) == count;
}

bool ReadU32(std::istream& in, std::uint32_t* value) {
  unsigned char bytes[4];
  if (!ReadBytes(in, bytes, sizeof(bytes))) return false;
  *value = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
  return true;
}

struct Fixture {
  int batch = 0;
  int seq = 0;
  int buffer = 0;
  int source_batch = 0;
  std::vector<std::string> documents;
  std::vector<std::vector<int>> tokens;
  std::vector<std::vector<int>> targets;
};

bool ReadFixture(const std::string& path, Fixture* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open fixture: " + path;
    return false;
  }
  char magic[8];
  if (!ReadBytes(in, magic, sizeof(magic)) ||
      std::string(magic, sizeof(magic)) != "NCLDRP01") {
    *error = "bad fixture magic";
    return false;
  }
  std::uint32_t version = 0;
  std::uint32_t batch = 0;
  std::uint32_t seq = 0;
  std::uint32_t buffer = 0;
  std::uint32_t source_batch = 0;
  if (!ReadU32(in, &version) || !ReadU32(in, &batch) || !ReadU32(in, &seq) ||
      !ReadU32(in, &buffer) || !ReadU32(in, &source_batch)) {
    *error = "truncated fixture header";
    return false;
  }
  if (version != 1) {
    *error = "unsupported fixture version";
    return false;
  }
  out->batch = static_cast<int>(batch);
  out->seq = static_cast<int>(seq);
  out->buffer = static_cast<int>(buffer);
  out->source_batch = static_cast<int>(source_batch);

  std::uint32_t doc_count = 0;
  if (!ReadU32(in, &doc_count)) {
    *error = "truncated document count";
    return false;
  }
  out->documents.resize(doc_count);
  for (std::uint32_t i = 0; i < doc_count; ++i) {
    std::uint32_t length = 0;
    if (!ReadU32(in, &length)) {
      *error = "truncated document length";
      return false;
    }
    out->documents[i].resize(length);
    if (length > 0 && !ReadBytes(in, out->documents[i].data(), length)) {
      *error = "truncated document";
      return false;
    }
  }

  std::uint32_t steps = 0;
  if (!ReadU32(in, &steps)) {
    *error = "truncated step count";
    return false;
  }
  const std::size_t row =
      static_cast<std::size_t>(out->batch) * static_cast<std::size_t>(out->seq);
  out->tokens.resize(steps);
  out->targets.resize(steps);
  for (std::uint32_t s = 0; s < steps; ++s) {
    out->tokens[s].resize(row);
    out->targets[s].resize(row);
    for (std::size_t i = 0; i < row; ++i) {
      std::uint32_t value = 0;
      if (!ReadU32(in, &value)) {
        *error = "truncated tokens";
        return false;
      }
      out->tokens[s][i] = static_cast<int>(value);
    }
    for (std::size_t i = 0; i < row; ++i) {
      std::uint32_t value = 0;
      if (!ReadU32(in, &value)) {
        *error = "truncated targets";
        return false;
      }
      out->targets[s][i] = static_cast<int>(value);
    }
  }
  return true;
}

// A fixed document source that yields `source_batch` documents per call, the
// same granularity as the reference `tokenizer_batch_size`.
class VectorSource : public nanochat::DocumentSource {
 public:
  VectorSource(std::vector<std::string> documents, int source_batch)
      : documents_(std::move(documents)), source_batch_(source_batch) {}

  bool Next(std::vector<std::string>* documents, std::string* error) override {
    (void)error;
    if (index_ >= documents_.size()) return false;
    documents->clear();
    const std::size_t end = index_ + static_cast<std::size_t>(source_batch_);
    for (; index_ < end && index_ < documents_.size(); ++index_) {
      documents->push_back(documents_[index_]);
    }
    return true;
  }

 private:
  std::vector<std::string> documents_;
  std::size_t source_batch_ = 1;
  std::size_t index_ = 0;
};

void CheckRows(const Fixture& fixture,
               const std::vector<std::string>& documents,
               nanochat::Tokenizer* tokenizer) {
  auto shared = std::make_shared<std::vector<std::string>>(documents);
  const int source_batch = fixture.source_batch;
  nanochat::DocumentSourceFactory factory =
      [shared, source_batch](
          std::string* error) -> std::unique_ptr<nanochat::DocumentSource> {
    (void)error;
    return std::make_unique<VectorSource>(*shared, source_batch);
  };

  nanochat::DataLoader loader(factory, tokenizer, fixture.batch, fixture.seq,
                              /*seed=*/7, /*tokenizer_threads=*/1,
                              static_cast<std::size_t>(fixture.buffer));
  const std::size_t expected = static_cast<std::size_t>(fixture.batch) *
                               static_cast<std::size_t>(fixture.seq);
  for (std::size_t step = 0; step < fixture.tokens.size(); ++step) {
    std::vector<int> tokens(expected);
    std::vector<int> targets(expected);
    if (!loader.Next(tokens.data(), targets.data())) {
      Fail("Next returned false at step " + std::to_string(step));
      return;
    }
    if (tokens != fixture.tokens[step]) {
      for (std::size_t i = 0; i < expected; ++i) {
        if (tokens[i] != fixture.tokens[step][i]) {
          Fail("token mismatch at step " + std::to_string(step) + " index " +
               std::to_string(i) + ": got " + std::to_string(tokens[i]) +
               ", want " + std::to_string(fixture.tokens[step][i]));
          break;
        }
      }
    }
    if (targets != fixture.targets[step]) {
      for (std::size_t i = 0; i < expected; ++i) {
        if (targets[i] != fixture.targets[step][i]) {
          Fail("target mismatch at step " + std::to_string(step) + " index " +
               std::to_string(i) + ": got " + std::to_string(targets[i]) +
               ", want " + std::to_string(fixture.targets[step][i]));
          break;
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "loader_parity_test: usage: <fixture> <tokenizer>\n");
    return 2;
  }
  Fixture fixture;
  std::string error;
  if (!ReadFixture(argv[1], &fixture, &error)) {
    std::fprintf(stderr, "loader_parity_test: %s\n", error.c_str());
    return 2;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(argv[2]);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "loader_parity_test: cannot load the tokenizer\n");
    return 2;
  }
  CheckRows(fixture, fixture.documents, tokenizer.get());
  if (g_failures != 0) {
    std::fprintf(stderr, "loader_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("loader_parity_test: ok (%zu steps)\n", fixture.tokens.size());
  return 0;
}
