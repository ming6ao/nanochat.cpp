// src/train_parallel_test.cc -- data parallel training on the CPU backend
// (docs/distributed-design.md section 12).
//
// Two checks:
//   1. Two ranks that split one batch and reduce with a sum equal one rank on
//      the whole batch. Each rank computes the mean of its half, scales by
//      1/world_size, and the host reference sums the two gradients, so the
//      summed gradient is the global mean.
//   2. Two `TrainLoop`s at rank 0 and rank 1 end with identical parameters: the
//      reduction gives both the same gradient, so the steps agree.

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "nanochat/tokenizer.h"
#include "src/distributed.h"
#include "src/ops.h"
#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"
#include "src/train.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::DistributedConfig;
using nanochat::DocumentSource;
using nanochat::DocumentSourceFactory;
using nanochat::GradientSync;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
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

int FindFreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    return -1;
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    ::close(fd);
    return -1;
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

void CompareParameters(const Model& a, const Model& b, const std::string& what,
                       double tolerance) {
  const std::vector<ParamView> pa = a.params();
  const std::vector<ParamView> pb = b.params();
  if (pa.size() != pb.size()) {
    Fail(what + ": parameter count mismatch");
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    for (std::int64_t j = 0; j < pa[i].count; ++j) {
      worst = std::max(
          worst, std::fabs(static_cast<double>(nanochat::AsF(pa[i].value[j])) -
                           static_cast<double>(nanochat::AsF(pb[i].value[j]))));
    }
  }
  if (worst > tolerance) {
    Fail(what + ": parameter mismatch, worst = " + std::to_string(worst));
  }
}

// Runs one rank of a fixed split batch: the forward mean over the rank's rows,
// scaled by 1/world_size, reduced with a sum, then one optimizer step.
std::unique_ptr<Model> RunFixedBatchRank(const Config& config, int rank,
                                         int port, const int* tokens,
                                         const int* targets, int rank_batch,
                                         int seq, int row_offset) {
  DistributedConfig distributed;
  distributed.rank = rank;
  distributed.world_size = 2;
  distributed.master = "127.0.0.1";
  distributed.port = port;
  std::unique_ptr<GradientSync> sync =
      nanochat::CreateGradientSync(distributed);
  if (sync == nullptr) return nullptr;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(kSeed);
  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = 1;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
  model->ZeroGrad();
  model->ForwardLoss(tokens + static_cast<std::size_t>(row_offset) * seq,
                     targets + static_cast<std::size_t>(row_offset) * seq,
                     rank_batch, seq);
  model->BackwardAccumulate(1.0f / 2.0f);
  for (const ParamView& parameter : model->params()) {
    if (parameter.grad == nullptr || parameter.count <= 0) continue;
    sync->AllReduceSum(parameter.grad, parameter.count);
  }
  optimizer->Step(1);
  return model;
}

void TestTwoRanksEqualOneRank() {
  const Config config = TinyConfig();
  const int batch = 4;
  const int seq = 8;
  const int rows = batch * seq;
  std::vector<int> tokens(rows);
  std::vector<int> targets(rows);
  for (int i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] = (i * 7 + 3) % config.vocab_size;
    targets[static_cast<std::size_t>(i)] = (i * 11 + 5) % config.vocab_size;
  }

  // Reference: one rank over the full batch.
  std::unique_ptr<Model> reference = Model::Create(config);
  reference->InitWeights(kSeed);
  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = 1;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(reference.get(), optimizer_config, scheduler);
  reference->ZeroGrad();
  reference->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  reference->Backward();
  optimizer->Step(1);

  const int port = FindFreePort();
  if (port <= 0) {
    Fail("cannot find a free port");
    return;
  }
  const int half = batch / 2;

  std::unique_ptr<Model> rank1_model;
  std::thread worker([&] {
    rank1_model = RunFixedBatchRank(config, 1, port, tokens.data(),
                                    targets.data(), half, seq, half);
  });
  std::unique_ptr<Model> rank0_model = RunFixedBatchRank(
      config, 0, port, tokens.data(), targets.data(), half, seq, 0);
  worker.join();

  if (rank0_model == nullptr || rank1_model == nullptr) {
    Fail("a rank did not join the gradient-sync group");
    return;
  }
  CompareParameters(*rank0_model, *reference, "rank 0 vs one rank", 1e-4);
  CompareParameters(*rank1_model, *reference, "rank 1 vs one rank", 1e-4);
}

// A fixed in-memory document source, as in `src/harness_test.cc`.
class VectorSource final : public DocumentSource {
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

// Two `TrainLoop`s agree on the parameters after one step. The reduction gives
// both the same summed gradient, so the optimizer steps agree even though the
// two ranks read different document shards.
void TestTrainLoopRanksAgree() {
  Config config = TinyConfig();
  // The native tokenizer starts at 256 byte tokens, so the model vocab must be
  // larger than the tiny default (docs/tokenizer.md section 2).
  config.vocab_size = 512;
  config.padded_vocab_size = 512;
  const int batch = 2;
  const int seq = 8;
  const std::string tokenizer_path = TempPath("train_parallel.nctoken");
  if (!nanochat::SaveTokenizer(tokenizer_path, nanochat::NanochatSplitPattern(),
                               {}, {{"<|bos|>", 256}})) {
    Fail("train_parallel: SaveTokenizer failed");
    return;
  }
  auto documents = std::make_shared<std::vector<std::string>>(
      std::vector<std::string>{"alpha", "beta", "gamma", "delta", "epsilon",
                               "zeta", "eta", "theta", "iota", "kappa"});
  DocumentSourceFactory factory =
      [documents](std::string*) -> std::unique_ptr<DocumentSource> {
    return std::make_unique<VectorSource>(*documents, 4);
  };

  const int port = FindFreePort();
  if (port <= 0) {
    Fail("cannot find a free port");
    return;
  }

  TrainConfig base;
  base.model = config;
  base.batch = batch;
  base.seq = seq;
  base.num_iterations = 1;
  base.log_every = 0;
  base.seed = kSeed;
  base.tokenizer_path = tokenizer_path;
  base.train_source = factory;
  base.tokenizer_threads = 1;
  base.document_buffer = 16;
  base.master = "127.0.0.1";
  base.port = port;

  TrainConfig rank0_config = base;
  rank0_config.rank = 0;
  rank0_config.world_size = 2;
  TrainConfig rank1_config = base;
  rank1_config.rank = 1;
  rank1_config.world_size = 2;

  // Construct rank 0 first: it binds and listens, so rank 1 can connect.
  TrainLoop loop0(std::move(rank0_config));
  TrainLoop loop1(std::move(rank1_config));

  std::thread worker([&] { loop0.Run(); });
  loop1.Run();
  worker.join();

  CompareParameters(*loop0.model(), *loop1.model(), "two-rank TrainLoop", 1e-5);
}

void TestBackwardScale() {
  // The micro-batch sum and the all-reduce sum give the global mean when the
  // backward carries 1/(grad_accum * world_size)
  // (docs/distributed-design.md section 6).
  if (std::fabs(nanochat::DistributedBackwardScale(4, 1) - 0.25f) > 1e-7f) {
    Fail("backward scale: world size 1 changed the accumulation scale");
  }
  if (std::fabs(nanochat::DistributedBackwardScale(4, 2) - 0.125f) > 1e-7f) {
    Fail("backward scale: world size 2 did not average");
  }
  if (std::fabs(nanochat::DistributedBackwardScale(1, 2) - 0.5f) > 1e-7f) {
    Fail("backward scale: one micro-batch did not average");
  }
  if (std::fabs(nanochat::DistributedBackwardScale(0, 0) - 1.0f) > 1e-7f) {
    Fail("backward scale: the defaults are not a no-op");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestBackwardScale();
  TestTwoRanksEqualOneRank();
  TestTrainLoopRanksAgree();
  if (g_failures != 0) {
    std::fprintf(stderr, "train_parallel_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("train_parallel_test: ok\n");
  return 0;
}
