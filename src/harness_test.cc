// src/harness_test.cc — the training harness utilities and driver.
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

#include "cli.h"
#include "nanochat/data.h"
#include "nanochat/dataloader.h"
#include "nanochat/logger.h"
#include "nanochat/mfu.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "ops.h"
#include "train.h"

namespace {

using nanochat::Checkpointer;
using nanochat::ComputeType;
using nanochat::Config;
using nanochat::DataLoader;
using nanochat::LogRecord;
using nanochat::Logger;
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

std::string WriteShard(const std::string& name,
                       const std::vector<int>& tokens) {
  const std::string path = TempPath(name);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  WriteU32(out, nanochat::kTokenShardMagic);
  WriteU32(out, nanochat::kTokenShardVersion);
  WriteU64(out, static_cast<std::uint64_t>(tokens.size()));
  WriteU32(out, 2);
  WriteU32(out, 0);
  for (int token : tokens) WriteU16(out, static_cast<std::uint16_t>(token));
  return path;
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
  ExpectNear(nanochat::PeakFlopsForDevice(""), 0.0, 1e-9,
             "PeakFlops empty");

  const double expected_mfu =
      1000.0 * 147744.0 / 11.34e12;
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

void TestTrainLoopMatchesHandRun() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int total = 32;
  std::vector<int> stream(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i) {
    stream[static_cast<std::size_t>(i)] =
        (i * 7 + 3) % config.vocab_size;
  }
  const std::string shard = WriteShard("harness.bin", stream);

  // Hand-run: one TrainStep with the same weights and the same first batch.
  std::unique_ptr<Model> hand = Model::Create(config);
  hand->InitWeights(kSeed);
  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = 1;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(hand.get(), optimizer_config, scheduler);
  std::vector<int> tokens(static_cast<std::size_t>(batch) * seq);
  std::vector<int> targets(static_cast<std::size_t>(batch) * seq);
  for (int i = 0; i < batch * seq; ++i) {
    tokens[static_cast<std::size_t>(i)] = stream[static_cast<std::size_t>(i)];
    targets[static_cast<std::size_t>(i)] =
        stream[static_cast<std::size_t>(i + 1)];
  }
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
  train_config.shuffle = false;
  train_config.train_shards = {shard};
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
            worst, std::fabs(static_cast<double>(nanochat::AsF(a[i].value[j])) -
                             static_cast<double>(nanochat::AsF(b[i].value[j]))));
      }
    }
    if (worst > 0.0) Fail("Checkpointer round-trip changed a parameter");
  }

  // EvalBpb on the validation loader.
  DataLoader loader({shard}, batch, seq, kSeed, false);
  const float bpb = nanochat::EvalBpb(reloaded.get(), &loader, 1);
  if (!std::isfinite(bpb) || bpb <= 0.0f) {
    Fail("EvalBpb did not return a finite positive value");
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
  if (g_failures != 0) {
    std::fprintf(stderr, "harness_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("harness_test: ok\n");
  return 0;
}
