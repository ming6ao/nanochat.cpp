// src/harness_test.cc -- the training harness utilities and driver.
//
// Uses the same tiny architecture as `src/generate_test.cc` and
// `src/optim_test.cc` and checks, in order:
//   1. the `mfu.h` FLOP numbers against hand-computed values;
//   2. `Logger` writes a structured record to the run file;
//   3. the `Checkpointer` container round-trips every parameter;
//   4. one `TrainLoop` step matches a hand-run `Model::TrainStep`;
//   5. `EvalBpb` produces a finite bits-per-byte.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/data.h"
#include "nanochat/dataloader.h"
#include "nanochat/logger.h"
#include "nanochat/mfu.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "nanochat/tokenizer.h"
#include "src/cli.h"
#include "src/ops.h"
#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"
#include "src/train.h"

namespace {

using nanochat::Checkpointer;
using nanochat::ComputeType;
using nanochat::Config;
using nanochat::DataLoader;
using nanochat::Logger;
using nanochat::LogRecord;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::TrainConfig;
using nanochat::TrainLoop;

constexpr std::uint64_t kSeed = 20241001;
int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& what) {
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(stderr, "FAIL: %s: got %.10g want %.10g\n", what.c_str(),
                 actual, expected);
    ++g_failures;
  }
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

void TestMfu() {
  const Config config = TinyConfig();
  // Hand-computed for the tiny architecture. See the node spec:
  // 23600 matmul parameters; 6 * 23600 + 12 * 8 * 4 * 8 * 2 = 147744.
  ExpectNear(nanochat::EstimateFlopsPerToken(config), 147744.0, 1e-6,
             "EstimateFlopsPerToken");
  ExpectNear(nanochat::EstimatePrefillFlops(config, 4), 191360.0, 1e-6,
             "EstimatePrefillFlops");
  ExpectNear(nanochat::EstimateDecodeFlops(config, 8), 49248.0, 1e-6,
             "EstimateDecodeFlops");

  ExpectNear(nanochat::PeakFlopsForDevice("NVIDIA GeForce GTX 1080 Ti"),
             11.34e12, 1.0, "PeakFlops 1080 Ti");
  ExpectNear(nanochat::PeakFlopsForDevice("NVIDIA H100 NVL"), 835e12, 1.0,
             "PeakFlops H100 NVL");
  ExpectNear(nanochat::PeakFlopsForDevice("unknown device"), 0.0, 1e-9,
             "PeakFlops unknown");
  ExpectNear(nanochat::PeakFlopsForDevice(""), 0.0, 1e-9, "PeakFlops empty");

  const double expected_mfu = 1000.0 * 147744.0 / 11.34e12;
  ExpectNear(nanochat::ComputeMfu(config, 1000.0, 11.34e12), expected_mfu,
             1e-12, "ComputeMfu");
  ExpectNear(nanochat::ComputeMfu(config, 1000.0, 0.0), 0.0, 1e-12,
             "ComputeMfu unknown device");
}

void TestLogger() {
  const std::string path = TempPath("harness.log");
  {
    Logger logger(path);
    LogRecord record;
    record.step = 7;
    record.loss = 1.5f;
    record.lr = 0.25f;
    record.grad_norm = 3.0f;
    record.tokens_per_second = 100.0f;
    record.mfu = 0.5f;
    logger.Log(record);
    logger.Info("hello");
    logger.Close();
  }
  std::ifstream in(path);
  std::string contents((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  if (contents.find("step 000007") == std::string::npos ||
      contents.find("loss 1.500000") == std::string::npos ||
      contents.find("mfu 50.00%") == std::string::npos ||
      contents.find("[nanochat] hello") == std::string::npos) {
    Fail("Logger output missing an expected field: " + contents);
  }
}

// A fixed in-memory document source for the harness test: it yields each
// document once, `source_batch` at a time.
class VectorSource : public nanochat::DocumentSource {
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

void TestTrainLoopMatchesHandRun() {
  Config config = TinyConfig();
  // The native tokenizer starts at 256 byte tokens, so the model vocab must be
  // larger than the tiny default (docs/tokenizer.md section 2).
  config.vocab_size = 512;
  config.padded_vocab_size = 512;
  const int batch = 2;
  const int seq = 8;

  const std::string tokenizer_path = TempPath("harness.nctoken");
  if (!nanochat::SaveTokenizer(tokenizer_path, nanochat::NanochatSplitPattern(),
                               {}, {{"<|bos|>", 256}})) {
    Fail("harness: SaveTokenizer failed");
    return;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    Fail("harness: LoadTokenizer failed");
    return;
  }
  auto documents = std::make_shared<std::vector<std::string>>(
      std::vector<std::string>{"alpha", "beta", "gamma", "delta", "epsilon",
                               "zeta", "eta", "theta", "iota", "kappa"});
  nanochat::DocumentSourceFactory factory =
      [documents](
          std::string* error) -> std::unique_ptr<nanochat::DocumentSource> {
    (void)error;
    return std::make_unique<VectorSource>(*documents, 4);
  };

  // The probe loader produces the hand-run input with the same settings as
  // `TrainLoop`: same documents, tokenizer, threads, and buffer.
  nanochat::DataLoader probe(factory, tokenizer.get(), batch, seq, kSeed, 1,
                             16);
  std::vector<int> tokens(static_cast<std::size_t>(batch) * seq);
  std::vector<int> targets(static_cast<std::size_t>(batch) * seq);
  if (!probe.Next(tokens.data(), targets.data())) {
    Fail("harness: the probe loader produced no batch");
    return;
  }

  // Hand-run: one TrainStep with the same weights and the same first batch.
  std::unique_ptr<Model> hand = Model::Create(config);
  hand->InitWeights(kSeed);
  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = 1;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(hand.get(), optimizer_config, scheduler);
  const float expected = hand->TrainStep(tokens.data(), targets.data(), batch,
                                         seq, optimizer.get());

  // Driver: the first training step must reproduce it.
  TrainConfig train_config;
  train_config.model = config;
  train_config.batch = batch;
  train_config.seq = seq;
  train_config.num_iterations = 1;
  train_config.log_every = 0;
  train_config.seed = kSeed;
  train_config.tokenizer_path = tokenizer_path;
  train_config.train_source = factory;
  train_config.tokenizer_threads = 1;
  train_config.document_buffer = 16;
  TrainLoop loop(std::move(train_config));
  const float got = loop.Run();
  ExpectNear(static_cast<double>(got), static_cast<double>(expected), 1e-5,
             "TrainLoop vs hand-run TrainStep");

  // Checkpointer round-trip through the `Checkpoint` container.
  const std::string checkpoint = TempPath("harness.ckpt");
  if (!Checkpointer::SaveModel(*hand, checkpoint)) {
    Fail("Checkpointer::SaveModel failed");
  }
  std::unique_ptr<Model> reloaded = Model::Create(config);
  reloaded->InitWeights(kSeed + 99);
  if (!Checkpointer::LoadModel(reloaded.get(), checkpoint)) {
    Fail("Checkpointer::LoadModel failed");
  }
  const std::vector<nanochat::ParamView> a = hand->params();
  const std::vector<nanochat::ParamView> b = reloaded->params();
  if (a.size() != b.size()) {
    Fail("Checkpointer parameter count mismatch");
  } else {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      for (std::int64_t j = 0; j < a[i].count; ++j) {
        worst = std::max(
            worst,
            std::fabs(static_cast<double>(nanochat::AsF(a[i].value[j])) -
                      static_cast<double>(nanochat::AsF(b[i].value[j]))));
      }
    }
    if (worst > 0.0) Fail("Checkpointer round-trip changed a parameter");
  }

  // EvalBpb on the validation loader.
  nanochat::DataLoader loader(factory, tokenizer.get(), batch, seq, kSeed, 1,
                              16);
  const float bpb = nanochat::EvalBpb(reloaded.get(), &loader, 1);
  if (!std::isfinite(bpb) || bpb <= 0.0f) {
    Fail("EvalBpb did not return a finite positive value");
  }
}

void TestGradientAccumulation() {
  const Config config = TinyConfig();
  const int micro_batch = 2;
  const int seq = 8;
  const int rows = micro_batch * seq;
  std::vector<int> tokens_a(static_cast<std::size_t>(rows));
  std::vector<int> targets_a(static_cast<std::size_t>(rows));
  std::vector<int> tokens_b(static_cast<std::size_t>(rows));
  std::vector<int> targets_b(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    tokens_a[static_cast<std::size_t>(i)] = (i * 7 + 3) % config.vocab_size;
    targets_a[static_cast<std::size_t>(i)] = (i * 11 + 5) % config.vocab_size;
    tokens_b[static_cast<std::size_t>(i)] = (i * 13 + 1) % config.vocab_size;
    targets_b[static_cast<std::size_t>(i)] = (i * 17 + 2) % config.vocab_size;
  }

  // Reference: one backward over the concatenated batch.
  std::unique_ptr<Model> reference = Model::Create(config);
  reference->InitWeights(kSeed);
  std::vector<int> tokens(tokens_a);
  tokens.insert(tokens.end(), tokens_b.begin(), tokens_b.end());
  std::vector<int> targets(targets_a);
  targets.insert(targets.end(), targets_b.begin(), targets_b.end());
  reference->ForwardLoss(tokens.data(), targets.data(), 2 * micro_batch, seq);
  reference->Backward();

  // Accumulated: two micro-batches, each scaled by 1/2, summed.
  std::unique_ptr<Model> accumulated = Model::Create(config);
  accumulated->InitWeights(kSeed);
  accumulated->ZeroGrad();
  accumulated->ForwardLoss(tokens_a.data(), targets_a.data(), micro_batch, seq);
  accumulated->BackwardAccumulate(0.5f);
  accumulated->ForwardLoss(tokens_b.data(), targets_b.data(), micro_batch, seq);
  accumulated->BackwardAccumulate(0.5f);

  const std::vector<nanochat::ParamView> ra = reference->params();
  const std::vector<nanochat::ParamView> aa = accumulated->params();
  if (ra.size() != aa.size()) {
    Fail("gradient accumulation: parameter count mismatch");
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    for (std::int64_t j = 0; j < ra[i].count; ++j) {
      worst = std::max(
          worst, std::fabs(static_cast<double>(nanochat::AsF(ra[i].grad[j])) -
                           static_cast<double>(nanochat::AsF(aa[i].grad[j]))));
    }
  }
  if (worst > 1e-4) {
    Fail("gradient accumulation gradient mismatch: worst = " +
         std::to_string(worst));
  }
}

void TestModelFlagParsing() {
  Config config;
  if (!nanochat::cli::ApplyModelFlag("--layers", "3", &config) ||
      config.num_layers != 3) {
    Fail("ApplyModelFlag --layers failed");
  }
  if (!nanochat::cli::ApplyModelFlag("--window-pattern", "SSSL", &config) ||
      config.window_pattern != "SSSL") {
    Fail("ApplyModelFlag --window-pattern failed");
  }
  if (nanochat::cli::ApplyModelFlag("--layers", "not-a-number", &config)) {
    Fail("ApplyModelFlag accepted a bad integer");
  }
  const std::vector<std::string> parts = nanochat::cli::SplitCsv("a,b,,c");
  if (parts != std::vector<std::string>({"a", "b", "c"})) {
    Fail("SplitCsv mismatch");
  }
}

}  // namespace

int main() {
  TestMfu();
  TestLogger();
  TestModelFlagParsing();
  TestTrainLoopMatchesHandRun();
  TestGradientAccumulation();
  if (g_failures != 0) {
    std::fprintf(stderr, "harness_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("harness_test: ok\n");
  return 0;
}
