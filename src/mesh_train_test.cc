// src/mesh_train_test.cc -- the mesh training driver on the CPU backend
// (docs/client-runtime-design.md section 10.1, docs/training-seam.md
// section 6.6).
//
// The gates: the driver writes one record per step and a summary; a resume
// continues the counter and appends the metrics file; and the driver loss
// sequence equals a hand-run `MeshRuntime` on the same batches.
//
// CPU only; no GPU and no broker.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/dataloader.h"
#include "nanochat/mesh.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tokenizer.h"
#include "src/mesh_runtime.h"
#include "src/mesh_train.h"
#include "src/record_file.h"
#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"
#include "src/train.h"

namespace {

using nanochat::Config;
using nanochat::DocumentSource;
using nanochat::DocumentSourceFactory;
using nanochat::MeshRuntime;
using nanochat::MeshRuntimeConfig;
using nanochat::MeshSpec;
using nanochat::MeshTrainConfig;
using nanochat::MeshTrainDriver;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 512;
  config.padded_vocab_size = 512;
  config.window_pattern = "SL";
  return config;
}

// A fixed in-memory document source. It yields every document once, then
// reports the end so `DataLoader::Reset` starts a new epoch.
class VectorSource : public DocumentSource {
 public:
  VectorSource(std::vector<std::string> documents, std::size_t source_batch)
      : documents_(std::move(documents)), source_batch_(source_batch) {}

  bool Next(std::vector<std::string>* documents, std::string* error) override {
    (void)error;
    if (index_ >= documents_.size()) return false;
    documents->clear();
    const std::size_t end = index_ + source_batch_;
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

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) return "";
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

int CountLines(const std::string& text) {
  int lines = 0;
  for (const char c : text) {
    if (c == '\n') ++lines;
  }
  return lines;
}

std::string UniqueToken() {
  static int counter = 0;
  return std::to_string(++counter);
}

struct Fixture {
  std::filesystem::path root;
  std::string tokenizer_path;
  DocumentSourceFactory factory;
};

Fixture MakeFixture() {
  Fixture fixture;
  fixture.root =
      std::filesystem::temp_directory_path() / ("mesh_train_" + UniqueToken());
  std::filesystem::create_directories(fixture.root);
  fixture.tokenizer_path = (fixture.root / "tiny.nctoken").string();
  if (!nanochat::SaveTokenizer(fixture.tokenizer_path,
                               nanochat::NanochatSplitPattern(), {},
                               {{"<|bos|>", 256}})) {
    Fail("SaveTokenizer failed");
  }
  auto documents = std::make_shared<std::vector<std::string>>(
      std::vector<std::string>{"alpha", "beta", "gamma", "delta", "epsilon",
                               "zeta", "eta", "theta", "iota", "kappa"});
  fixture.factory =
      [documents](std::string* error) -> std::unique_ptr<DocumentSource> {
    (void)error;
    return std::make_unique<VectorSource>(*documents, 4);
  };
  return fixture;
}

MeshTrainConfig TinyDriverConfig(const Fixture& fixture, const std::string& out,
                                 int steps) {
  MeshTrainConfig config;
  config.mesh = MeshSpec::Data(2);
  config.model = TinyConfig();
  config.preset = "track3";
  config.batch = 4;
  config.grad_accum = 2;
  config.seq = 8;
  config.seed = 20241001;
  config.num_iterations = steps;
  config.log_every = 0;
  config.save_every = 0;
  config.out_dir = out;
  config.tokenizer_path = fixture.tokenizer_path;
  config.train_source = fixture.factory;
  config.tokenizer_threads = 1;
  config.document_buffer = 16;
  config.flops_per_token = 2.0;
  config.peak_flops = 1000.0;
  return config;
}

// The driver writes one record per step and a summary.
void TestDriverWritesRecordAndSummary() {
  const Fixture fixture = MakeFixture();
  const std::string out = (fixture.root / "run").string();
  MeshTrainConfig config = TinyDriverConfig(fixture, out, 3);
  MeshTrainDriver driver(std::move(config));
  driver.Run();

  if (driver.records().size() != 3) {
    Fail("the driver must write three records");
  }
  const std::string metrics = ReadFile(out + "/metrics.jsonl");
  if (CountLines(metrics) != 3) {
    Fail("metrics.jsonl must hold three lines, got " +
         std::to_string(CountLines(metrics)));
  }
  if (metrics.find("\"version\":1") == std::string::npos) {
    Fail("the record must carry a version");
  }
  if (metrics.find("\"device_loss\":[") == std::string::npos) {
    Fail("the record must carry a per-device loss");
  }
  const std::string summary = ReadFile(out + "/summary.json");
  if (summary.find("\"steps\":3") == std::string::npos) {
    Fail("summary.json must report three steps");
  }
  if (summary.find("\"replica_digests\":[") == std::string::npos) {
    Fail("summary.json must carry the replica digests");
  }
  if (driver.last_step() != 3) Fail("last_step must be 3");
}

// A resume continues the counter and appends to the metrics file.
void TestResumeContinues() {
  const Fixture fixture = MakeFixture();
  const std::string out = (fixture.root / "resume").string();
  {
    MeshTrainConfig config = TinyDriverConfig(fixture, out, 2);
    config.save_every = 1;
    MeshTrainDriver driver(std::move(config));
    driver.Run();
  }
  MeshTrainConfig second = TinyDriverConfig(fixture, out, 4);
  second.resume = true;
  second.save_every = 1;
  MeshTrainDriver resumed(std::move(second));
  resumed.Run();
  if (resumed.records().size() != 2) {
    Fail("the resume must run two more steps");
  }
  if (resumed.records().front().step != 3) {
    Fail("the resume must start at step 3");
  }
  if (CountLines(ReadFile(out + "/metrics.jsonl")) != 4) {
    Fail("the resume must append to metrics.jsonl");
  }
}

// The driver loss sequence equals a hand-run runtime on the same batches.
void TestDriverMatchesHandRun() {
  const Fixture fixture = MakeFixture();
  const std::string out = (fixture.root / "parity").string();
  MeshTrainConfig config = TinyDriverConfig(fixture, out, 3);
  MeshTrainDriver driver(std::move(config));
  driver.Run();

  // The hand-run uses the same preset, the same shape, and the same documents.
  const nanochat::TrainConfig baseline = nanochat::TrackThreeBaseline();
  MeshRuntimeConfig runtime_config;
  runtime_config.model = TinyConfig();
  runtime_config.optimizer = baseline.optimizer;
  runtime_config.scheduler = baseline.scheduler;
  runtime_config.scheduler.num_iterations = 3;
  runtime_config.batch = 4;
  runtime_config.grad_accum = 2;
  runtime_config.seq = 8;
  runtime_config.seed = 20241001;
  MeshRuntime runtime(MeshSpec::Data(2), runtime_config);

  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(fixture.tokenizer_path);
  if (tokenizer == nullptr) {
    Fail("LoadTokenizer failed");
    return;
  }
  nanochat::DataLoader loader(fixture.factory, tokenizer.get(), 4, 8, 20241001,
                              1, 16);
  std::vector<int> tokens(4 * 8);
  std::vector<int> targets(4 * 8);
  for (int step = 0; step < 3; ++step) {
    if (!loader.Next(tokens.data(), targets.data())) {
      loader.Reset();
      loader.Next(tokens.data(), targets.data());
    }
    const float expected = runtime.TrainStep(tokens.data(), targets.data());
    const float got = driver.records()[static_cast<std::size_t>(step)].loss;
    if (std::abs(expected - got) > 1e-5f) {
      Fail("step " + std::to_string(step + 1) + ": driver " +
           std::to_string(got) + " vs hand-run " + std::to_string(expected));
    }
  }
}

}  // namespace

// The packed `NCSFT001` shard: the driver cycles the rows and writes one record
// per step (docs/training-seam.md section 6.2).
void TestDriverRunsPackedRows() {
  const Fixture fixture = MakeFixture();
  const std::string out = (fixture.root / "rows").string();
  const std::string shard_path = (fixture.root / "rows.nchsft").string();
  const int seq = 8;
  const int rows = 3;
  std::vector<std::int32_t> flat_tokens;
  std::vector<std::int32_t> flat_targets;
  for (int row = 0; row < rows; ++row) {
    for (int i = 0; i < seq; ++i) flat_tokens.push_back((i + row) % 64);
    for (int i = 0; i < seq; ++i) flat_targets.push_back(i % 64);
  }
  const std::vector<std::uint64_t> shape = {static_cast<std::uint64_t>(rows),
                                            static_cast<std::uint64_t>(seq)};
  std::vector<nanochat::Record> records;
  records.push_back(nanochat::MakeInt32Record("tokens", shape, flat_tokens));
  records.push_back(nanochat::MakeInt32Record("targets", shape, flat_targets));
  std::string error;
  if (!nanochat::SaveRecordFile(shard_path, "NCSFT001", records, &error)) {
    Fail("SaveRecordFile failed: " + error);
    return;
  }

  MeshTrainConfig config = TinyDriverConfig(fixture, out, 2);
  config.packed_rows_path = shard_path;
  config.tokenizer_path.clear();
  config.train_source = nullptr;
  config.seq = seq;
  MeshTrainDriver driver(std::move(config));
  driver.Run();
  if (driver.records().size() != 2) Fail("the rows run must write two records");
  if (CountLines(ReadFile(out + "/metrics.jsonl")) != 2) {
    Fail("the rows run must write two metric lines");
  }
}

int main() {
  TestDriverWritesRecordAndSummary();
  TestResumeContinues();
  TestDriverMatchesHandRun();
  TestDriverRunsPackedRows();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("mesh_train_test: all checks passed\n");
  return 0;
}
