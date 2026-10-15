// src/mesh_runtime_test.cc -- the single-client multi-device runtime on the CPU
// backend (docs/client-runtime-design.md sections 10 and 16).
//
// The gate: two replicas that split one global batch reduce with the in-process
// collective and equal one replica on the whole batch. The two runs differ only
// in the mesh size, so the loss and every parameter must agree.
//
// CPU only; no GPU and no broker.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/mesh.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/mesh_runtime.h"
#include "src/ops.h"
#include "src/train.h"

namespace {

using nanochat::Config;
using nanochat::MeshRuntime;
using nanochat::MeshRuntimeConfig;
using nanochat::MeshSpec;
using nanochat::Model;
using nanochat::ParamView;

constexpr std::uint64_t kSeed = 20241001;
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
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

MeshRuntimeConfig TinyRuntimeConfig(const Config& model) {
  MeshRuntimeConfig config;
  config.model = model;
  config.batch = 4;
  config.grad_accum = 2;
  config.seq = 8;
  config.seed = kSeed;
  config.scheduler.num_iterations = 2;
  return config;
}

std::vector<int> MakeIds(int rows, int vocab, int a, int b) {
  std::vector<int> ids(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    ids[static_cast<std::size_t>(i)] = (i * a + b) % vocab;
  }
  return ids;
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

// Two replicas equal one replica on the same global batch.
void TestTwoReplicasEqualOne() {
  const Config model = TinyConfig();
  const MeshRuntimeConfig config = TinyRuntimeConfig(model);
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);

  MeshRuntime single(MeshSpec::Data(1), config);
  MeshRuntime pair(MeshSpec::Data(2), config);

  float loss_single = 0.0f;
  float loss_pair = 0.0f;
  for (int step = 0; step < 2; ++step) {
    loss_single = single.TrainStep(tokens.data(), targets.data());
    loss_pair = pair.TrainStep(tokens.data(), targets.data());
  }

  if (std::fabs(loss_single - loss_pair) > 1e-4f) {
    Fail("the global loss differs: one replica " + std::to_string(loss_single) +
         ", two replicas " + std::to_string(loss_pair));
  }
  CompareParameters(*single.model(0), *pair.model(0), "replica 0 vs one", 1e-4);
  CompareParameters(*single.model(0), *pair.model(1), "replica 1 vs one", 1e-4);
}

// Four replicas equal one replica: the collective scales past a pair.
void TestFourReplicasEqualOne() {
  const Config model = TinyConfig();
  MeshRuntimeConfig config = TinyRuntimeConfig(model);
  config.batch = 8;
  config.grad_accum = 1;
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);

  MeshRuntime single(MeshSpec::Data(1), config);
  MeshRuntime quad(MeshSpec::Data(4), config);
  single.TrainStep(tokens.data(), targets.data());
  quad.TrainStep(tokens.data(), targets.data());

  for (int rank = 0; rank < quad.world_size(); ++rank) {
    CompareParameters(*single.model(0), *quad.model(rank),
                      "replica " + std::to_string(rank) + " vs one", 1e-4);
  }
}

// The replicas of one runtime stay byte-identical after several steps.
void TestReplicasStayEqual() {
  const Config model = TinyConfig();
  const MeshRuntimeConfig config = TinyRuntimeConfig(model);
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 13, 1);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 17, 9);

  MeshRuntime pair(MeshSpec::Data(2), config);
  for (int step = 0; step < 3; ++step) {
    pair.TrainStep(tokens.data(), targets.data());
  }
  CompareParameters(*pair.model(0), *pair.model(1), "replica 0 vs replica 1",
                    0.0);
}

// The accessors and the configured shape.
void TestRuntimeShape() {
  const Config model = TinyConfig();
  MeshRuntimeConfig config = TinyRuntimeConfig(model);
  MeshRuntime pair(MeshSpec::Data(2), config);
  if (pair.world_size() != 2) Fail("world_size is not 2");
  if (pair.global_batch() != config.batch) Fail("global_batch is wrong");
  if (pair.slice_batch() != config.batch / 2) Fail("slice_batch is wrong");
  if (pair.effective_seq() != config.seq) Fail("effective_seq is wrong");
  if (pair.model(2) != nullptr) Fail("model(2) must be null");
  if (pair.param_count(0) <= 0) Fail("replica 0 has no parameters");
  if (pair.param_count(0) != pair.param_count(1)) {
    Fail("the replicas have different parameter counts");
  }
}

// The step metrics: one loss per replica, the reduced gradient norm, and the
// step time (docs/client-runtime-design.md section 20.4).
void TestMetrics() {
  const Config model = TinyConfig();
  const MeshRuntimeConfig config = TinyRuntimeConfig(model);
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);

  MeshRuntime pair(MeshSpec::Data(2), config);
  const std::uint64_t before = pair.ParamDigest(0);
  if (before != pair.ParamDigest(1)) {
    Fail("the replicas start unequal");
  }
  const float loss = pair.TrainStep(tokens.data(), targets.data());
  if (pair.ParamDigest(0) != pair.ParamDigest(1)) {
    Fail("the replica digest differs after a step");
  }
  if (pair.ParamDigest(0) == before) {
    Fail("a step must change the parameters");
  }
  const nanochat::MeshMetrics& metrics = pair.metrics();
  if (static_cast<int>(metrics.loss.size()) != pair.world_size()) {
    Fail("metrics.loss must hold one entry per replica");
  }
  if (metrics.step_seconds <= 0.0f) Fail("step_seconds must be positive");
  if (metrics.grad_norm < 0.0f) Fail("grad_norm must not be negative");
  float mean = 0.0f;
  for (float value : metrics.loss) mean += value;
  mean /= static_cast<float>(metrics.loss.size());
  if (std::fabs(mean - loss) > 1e-4f) {
    Fail("the mean per-replica loss differs from TrainStep");
  }
}

// A save carries the optimizer state and the step, so a resumed run continues
// the schedule and equals one uninterrupted run.
void TestResume() {
  const Config model = TinyConfig();
  const MeshRuntimeConfig config = TinyRuntimeConfig(model);
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);
  const std::string path = "mesh_runtime_resume.nchkpt01";

  MeshRuntime uninterrupted(MeshSpec::Data(1), config);
  uninterrupted.TrainStep(tokens.data(), targets.data());
  uninterrupted.TrainStep(tokens.data(), targets.data());
  if (uninterrupted.step() != 2) Fail("the step is not 2 after two steps");

  MeshRuntime resumed(MeshSpec::Data(1), config);
  resumed.TrainStep(tokens.data(), targets.data());
  resumed.Save(path);
  if (resumed.step() != 1) Fail("the step is not 1 after one step");

  MeshRuntime restarted(MeshSpec::Data(1), config);
  restarted.Load(path);
  if (restarted.step() != 1) Fail("the loaded step is not 1");
  restarted.TrainStep(tokens.data(), targets.data());

  CompareParameters(*uninterrupted.model(0), *restarted.model(0),
                    "resume vs uninterrupted", 0.0);
  std::remove(path.c_str());
}

// A parameter-only load moves the parameters and leaves the optimizer state and
// the step alone (notebooks/checkpoint.md section 9.1). Two runtimes that start
// from the same parameters and the same fresh optimizer state must stay equal
// after one step: one loaded the parameters through `LoadWeights`, and the
// other through the parameter-only `Checkpointer::LoadModel`.
void TestLoadWeightsKeepsOptimizerState() {
  const Config model = TinyConfig();
  const MeshRuntimeConfig config = TinyRuntimeConfig(model);
  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);
  const std::string full_path = "mesh_runtime_full.nchkpt01";
  const std::string params_path = "mesh_runtime_params_only.nchkpt01";

  // The source run trains one step, so its optimizer state is not zero.
  MeshRuntime source(MeshSpec::Data(1), config);
  source.TrainStep(tokens.data(), targets.data());
  source.Save(full_path);
  nanochat::Checkpointer::SaveModel(*source.model(0), params_path);

  // `LoadWeights` reads the full file but stops before the optimizer
  // section. The step stays at zero and the parameters arrive.
  MeshRuntime loaded(MeshSpec::Data(1), config);
  loaded.LoadWeights(full_path);
  if (loaded.step() != 0) Fail("load_weights: the step moved");

  // The reference runtime gets the same parameters through the parameter-only
  // loader, so it also keeps the fresh optimizer state.
  MeshRuntime reference(MeshSpec::Data(1), config);
  if (!nanochat::Checkpointer::LoadModel(reference.model(0), params_path)) {
    Fail("load_weights: the reference parameter-only load failed");
    return;
  }

  // One step from the same parameters and the same fresh optimizer state must
  // agree exactly. A loaded optimizer state would diverge.
  loaded.TrainStep(tokens.data(), targets.data());
  reference.TrainStep(tokens.data(), targets.data());
  CompareParameters(*loaded.model(0), *reference.model(0),
                    "load_weights vs a fresh optimizer", 0.0);
  std::remove(full_path.c_str());
  std::remove(params_path.c_str());
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestRuntimeShape();
  TestMetrics();
  TestTwoReplicasEqualOne();
  TestFourReplicasEqualOne();
  TestReplicasStayEqual();
  TestResume();
  TestLoadWeightsKeepsOptimizerState();
  if (g_failures != 0) {
    std::fprintf(stderr, "mesh_runtime_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("mesh_runtime_test: ok\n");
  return 0;
}
