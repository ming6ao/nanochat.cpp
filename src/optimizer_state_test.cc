// src/optimizer_state_test.cc -- the optimizer-state records in NCHKPT01
// (docs/post-training.md section 7).
//
// The test proves three things on the tiny architecture used by
// `src/harness_test.cc` and `src/optim_test.cc`:
//   1. a checkpoint that carries the optimizer state reloads the AdamW
//      moments and Muon buffers exactly: a run that saves at step K and
//      resumes from the file reproduces the reference loss curve for steps
//      K+1..N;
//   2. the resumed final optimizer-state records equal the reference final
//      records byte for byte, so the moments match;
//   3. a parameter-only checkpoint stays loadable, carries no optimizer-state
//      records, and leaves the moments at zero.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/data.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "src/train.h"

namespace {

using nanochat::Checkpoint;
using nanochat::Checkpointer;
using nanochat::Config;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::TensorRecord;

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

struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed) {}
  std::uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};

void MakeBatch(Rng* rng, int rows, int vocab, std::vector<int>* tokens,
               std::vector<int>* targets) {
  tokens->resize(static_cast<std::size_t>(rows));
  targets->resize(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    (*tokens)[static_cast<std::size_t>(i)] =
        static_cast<int>(rng->Next() % static_cast<std::uint64_t>(vocab));
    (*targets)[static_cast<std::size_t>(i)] =
        static_cast<int>(rng->Next() % static_cast<std::uint64_t>(vocab));
  }
}

// One training step, driven by an explicit 1-based step number so a resumed
// run reproduces the schedules exactly (mirrors TrainLoop::Run).
float RunStep(Model* model, Optimizer* optimizer, int step,
              const std::vector<int>& tokens, const std::vector<int>& targets,
              int batch, int seq) {
  optimizer->ZeroGrad();
  const float loss =
      model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
  model->Backward();
  optimizer->Step(step);
  return loss;
}

// Loads two checkpoints and compares every optimizer-state record. The
// comparison is byte for byte, so it also catches a shape or dtype change.
void ExpectSameOptimizerState(const std::string& a, const std::string& b) {
  Checkpoint ca;
  Checkpoint cb;
  std::string error;
  if (!ca.Load(a, &error)) {
    Fail("optimizer state: cannot load " + a + ": " + error);
    return;
  }
  if (!cb.Load(b, &error)) {
    Fail("optimizer state: cannot load " + b + ": " + error);
    return;
  }
  const std::vector<TensorRecord>& sa = ca.optimizer_state();
  const std::vector<TensorRecord>& sb = cb.optimizer_state();
  if (sa.empty()) {
    Fail("optimizer state: no records were written");
    return;
  }
  if (sa.size() != sb.size()) {
    Fail("optimizer state: record count mismatch");
    return;
  }
  for (const TensorRecord& ra : sa) {
    const TensorRecord* rb = cb.FindOptimizerState(ra.name);
    if (rb == nullptr) {
      Fail("optimizer state: missing record " + ra.name);
      continue;
    }
    if (ra.dtype != rb->dtype || ra.shape != rb->shape || ra.data != rb->data) {
      Fail("optimizer state: record mismatch " + ra.name);
    }
  }
}

// A resumed run continues the reference loss curve exactly. The optimizer
// moments must come back from the file; a zero-moment restart would diverge.
void TestResumeContinuesRun() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const int total_steps = 6;
  const int save_step = 3;

  std::vector<std::vector<int>> tokens(static_cast<std::size_t>(total_steps));
  std::vector<std::vector<int>> targets(static_cast<std::size_t>(total_steps));
  Rng batch_rng(987654321);
  for (int step = 0; step < total_steps; ++step) {
    MakeBatch(&batch_rng, rows, config.vocab_size,
              &tokens[static_cast<std::size_t>(step)],
              &targets[static_cast<std::size_t>(step)]);
  }

  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = total_steps;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;

  const std::string resume_path = TempPath("optimizer_state_resume.ckpt");
  const std::string reference_path = TempPath("optimizer_state_reference.ckpt");
  std::vector<float> reference_losses(static_cast<std::size_t>(total_steps),
                                      0.0f);
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    for (int step = 1; step <= total_steps; ++step) {
      reference_losses[static_cast<std::size_t>(step - 1)] =
          RunStep(model.get(), optimizer.get(), step,
                  tokens[static_cast<std::size_t>(step - 1)],
                  targets[static_cast<std::size_t>(step - 1)], batch, seq);
      if (step == save_step &&
          !Checkpointer::SaveModel(*model, *optimizer, resume_path)) {
        Fail("resume: save with optimizer state failed");
        return;
      }
    }
    if (!Checkpointer::SaveModel(*model, *optimizer, reference_path)) {
      Fail("resume: final save failed");
      return;
    }
  }

  const std::string resumed_path = TempPath("optimizer_state_resumed.ckpt");
  std::vector<float> resumed_losses(static_cast<std::size_t>(total_steps),
                                    0.0f);
  {
    std::unique_ptr<Model> model = Model::Create(config);
    // A different seed proves the parameter load overwrites the weights.
    model->InitWeights(kSeed + 777);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    if (!Checkpointer::LoadModel(model.get(), optimizer.get(), resume_path)) {
      Fail("resume: load with optimizer state failed");
      return;
    }
    for (int step = save_step + 1; step <= total_steps; ++step) {
      resumed_losses[static_cast<std::size_t>(step - 1)] =
          RunStep(model.get(), optimizer.get(), step,
                  tokens[static_cast<std::size_t>(step - 1)],
                  targets[static_cast<std::size_t>(step - 1)], batch, seq);
    }
    if (!Checkpointer::SaveModel(*model, *optimizer, resumed_path)) {
      Fail("resume: resumed final save failed");
      return;
    }
  }

  for (int step = save_step + 1; step <= total_steps; ++step) {
    ExpectNear(resumed_losses[static_cast<std::size_t>(step - 1)],
               reference_losses[static_cast<std::size_t>(step - 1)], 1e-5,
               "resumed loss at step " + std::to_string(step));
  }
  ExpectSameOptimizerState(reference_path, resumed_path);
}

// A parameter-only checkpoint stays loadable. It carries no optimizer-state
// records, and loading it leaves the moments at zero.
void TestParameterOnlyStaysLoadable() {
  const Config config = TinyConfig();
  const std::string path = TempPath("optimizer_state_params_only.ckpt");
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(kSeed);
  if (!Checkpointer::SaveModel(*model, path)) {
    Fail("parameter-only: save failed");
    return;
  }

  Checkpoint checkpoint;
  std::string error;
  if (!checkpoint.Load(path, &error)) {
    Fail("parameter-only: load failed: " + error);
    return;
  }
  if (!checkpoint.optimizer_state().empty()) {
    Fail("parameter-only: file carries optimizer-state records");
  }
  if (checkpoint.tensors().empty()) {
    Fail("parameter-only: file carries no parameter records");
  }

  // Load with an optimizer; the moments must stay zero. A fresh optimizer on
  // the same weights is the zero-state reference.
  SchedulerConfig scheduler_config;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  std::unique_ptr<Model> reloaded = Model::Create(config);
  reloaded->InitWeights(kSeed + 31);
  std::unique_ptr<Optimizer> restored =
      nanochat::CreateOptimizer(reloaded.get(), optimizer_config, scheduler);
  if (!Checkpointer::LoadModel(reloaded.get(), restored.get(), path)) {
    Fail("parameter-only: LoadModel with optimizer failed");
    return;
  }
  std::unique_ptr<Optimizer> reference =
      nanochat::CreateOptimizer(reloaded.get(), optimizer_config, scheduler);
  const std::string restored_path = TempPath("optimizer_state_zero_a.ckpt");
  const std::string reference_zero_path =
      TempPath("optimizer_state_zero_b.ckpt");
  if (!Checkpointer::SaveModel(*reloaded, *restored, restored_path) ||
      !Checkpointer::SaveModel(*reloaded, *reference, reference_zero_path)) {
    Fail("parameter-only: zero-state save failed");
    return;
  }
  ExpectSameOptimizerState(restored_path, reference_zero_path);
}

// The ANVIL optimizer state (the twin-rail velocity and the lane energy)
// round-trips through NCHKPT01: a run that saves at step K and resumes from the
// file reproduces the reference final records byte for byte, and the records
// carry the ANVIL names rather than the Muon buffers. `adam_step_period` is
// exercised at 1 and 2 so the AdamW bias correction stays consistent across a
// resume when the cadence skips steps.
void TestAnvilStateRoundTrip(int adam_step_period) {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const int total_steps = 5;
  const int save_step = 2;

  std::vector<std::vector<int>> tokens(static_cast<std::size_t>(total_steps));
  std::vector<std::vector<int>> targets(static_cast<std::size_t>(total_steps));
  Rng batch_rng(1357911);
  for (int step = 0; step < total_steps; ++step) {
    MakeBatch(&batch_rng, rows, config.vocab_size,
              &tokens[static_cast<std::size_t>(step)],
              &targets[static_cast<std::size_t>(step)]);
  }

  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = total_steps;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;
  optimizer_config.matrix_optimizer = 1;
  optimizer_config.adam_step_period = adam_step_period;

  const std::string tag = "_p" + std::to_string(adam_step_period);
  const std::string resume_path =
      TempPath("anvil_state_resume" + tag + ".ckpt");
  const std::string reference_path =
      TempPath("anvil_state_reference" + tag + ".ckpt");
  const std::string resumed_path =
      TempPath("anvil_state_resumed" + tag + ".ckpt");
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    for (int step = 1; step <= total_steps; ++step) {
      RunStep(model.get(), optimizer.get(), step,
              tokens[static_cast<std::size_t>(step - 1)],
              targets[static_cast<std::size_t>(step - 1)], batch, seq);
      if (step == save_step &&
          !Checkpointer::SaveModel(*model, *optimizer, resume_path)) {
        Fail("anvil state: save with optimizer state failed");
        return;
      }
    }
    if (!Checkpointer::SaveModel(*model, *optimizer, reference_path)) {
      Fail("anvil state: final save failed");
      return;
    }
  }
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed + 99);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    if (!Checkpointer::LoadModel(model.get(), optimizer.get(), resume_path)) {
      Fail("anvil state: load with optimizer state failed");
      return;
    }
    for (int step = save_step + 1; step <= total_steps; ++step) {
      RunStep(model.get(), optimizer.get(), step,
              tokens[static_cast<std::size_t>(step - 1)],
              targets[static_cast<std::size_t>(step - 1)], batch, seq);
    }
    if (!Checkpointer::SaveModel(*model, *optimizer, resumed_path)) {
      Fail("anvil state: resumed final save failed");
      return;
    }
  }
  ExpectSameOptimizerState(reference_path, resumed_path);

  Checkpoint checkpoint;
  std::string error;
  if (!checkpoint.Load(reference_path, &error)) {
    Fail("anvil state: cannot load the reference: " + error);
    return;
  }
  bool saw_velocity = false;
  bool saw_lane_energy = false;
  for (const TensorRecord& record : checkpoint.optimizer_state()) {
    const bool anvil = record.name.rfind("anvil.", 0) == 0;
    if (anvil && record.name.find(".velocity") != std::string::npos) {
      saw_velocity = true;
    }
    if (anvil && record.name.find(".lane_energy") != std::string::npos) {
      saw_lane_energy = true;
    }
  }
  if (!saw_velocity) Fail("anvil state: no anvil.<shape>.velocity record");
  if (!saw_lane_energy) {
    Fail("anvil state: no anvil.<shape>.lane_energy record");
  }
}

// The optimizer-step record round-trips and drives a resumed schedule
// (docs/training-seam.md section 10). The 4-arg `SaveModel` writes the step
// beside the optimizer state; the 4-arg `LoadModel` reads it back, so a resume
// continues from the stored step instead of restarting the warmup. A
// parameter-only file carries no step record and leaves the caller at zero.
void TestOptimizerStepRoundTrip() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const int total_steps = 6;
  const int save_step = 3;

  std::vector<std::vector<int>> tokens(static_cast<std::size_t>(total_steps));
  std::vector<std::vector<int>> targets(static_cast<std::size_t>(total_steps));
  Rng batch_rng(246801357);
  for (int step = 0; step < total_steps; ++step) {
    MakeBatch(&batch_rng, rows, config.vocab_size,
              &tokens[static_cast<std::size_t>(step)],
              &targets[static_cast<std::size_t>(step)]);
  }

  SchedulerConfig scheduler_config;
  scheduler_config.num_iterations = total_steps;
  Scheduler scheduler(scheduler_config);
  OptimizerConfig optimizer_config;

  const std::string resume_path = TempPath("optimizer_step_resume.ckpt");
  std::vector<float> reference_losses(static_cast<std::size_t>(total_steps),
                                      0.0f);
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    for (int step = 1; step <= total_steps; ++step) {
      reference_losses[static_cast<std::size_t>(step - 1)] =
          RunStep(model.get(), optimizer.get(), step,
                  tokens[static_cast<std::size_t>(step - 1)],
                  targets[static_cast<std::size_t>(step - 1)], batch, seq);
      if (step == save_step &&
          !Checkpointer::SaveModel(*model, *optimizer, step, resume_path)) {
        Fail("step record: save failed");
        return;
      }
    }
  }

  // The record carries the 1-based step under its stable name.
  Checkpoint checkpoint;
  std::string error;
  if (!checkpoint.Load(resume_path, &error)) {
    Fail("step record: cannot load the checkpoint: " + error);
    return;
  }
  const TensorRecord* record = checkpoint.FindOptimizerState("optimizer/step");
  if (record == nullptr) {
    Fail("step record: no optimizer/step record");
    return;
  }
  if (record->data.size() != sizeof(std::int32_t)) {
    Fail("step record: wrong payload size");
    return;
  }
  std::int32_t stored = 0;
  std::memcpy(&stored, record->data.data(), sizeof(stored));
  if (stored != save_step) Fail("step record: wrong stored step");

  // A load reads the step back, and a run from that step continues the curve.
  std::vector<float> resumed_losses(static_cast<std::size_t>(total_steps),
                                    0.0f);
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed + 5);
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    int resume_step = 0;
    if (!Checkpointer::LoadModel(model.get(), optimizer.get(), &resume_step,
                                 resume_path)) {
      Fail("step record: load failed");
      return;
    }
    if (resume_step != save_step) {
      Fail("step record: LoadModel did not restore the step");
      return;
    }
    for (int step = resume_step + 1; step <= total_steps; ++step) {
      resumed_losses[static_cast<std::size_t>(step - 1)] =
          RunStep(model.get(), optimizer.get(), step,
                  tokens[static_cast<std::size_t>(step - 1)],
                  targets[static_cast<std::size_t>(step - 1)], batch, seq);
    }
  }
  for (int step = save_step + 1; step <= total_steps; ++step) {
    ExpectNear(resumed_losses[static_cast<std::size_t>(step - 1)],
               reference_losses[static_cast<std::size_t>(step - 1)], 1e-5,
               "step record: resumed loss at step " + std::to_string(step));
  }

  // A parameter-only file carries no step record and leaves the step alone.
  const std::string params_only = TempPath("optimizer_step_params_only.ckpt");
  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed);
    if (!Checkpointer::SaveModel(*model, params_only)) {
      Fail("step record: parameter-only save failed");
      return;
    }
  }
  {
    std::unique_ptr<Model> model = Model::Create(config);
    int resume_step = 99;
    if (!Checkpointer::LoadModel(model.get(), nullptr, &resume_step,
                                 params_only)) {
      Fail("step record: parameter-only load failed");
      return;
    }
    if (resume_step != 99) {
      Fail("step record: a parameter-only load changed the step");
    }
  }
}

}  // namespace

int main() {
  TestResumeContinuesRun();
  TestParameterOnlyStaysLoadable();
  TestAnvilStateRoundTrip(1);
  TestAnvilStateRoundTrip(2);
  TestOptimizerStepRoundTrip();
  if (g_failures != 0) {
    std::fprintf(stderr, "optimizer_state_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("optimizer_state_test: ok\n");
  return 0;
}
