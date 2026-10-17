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
#include <fstream>
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
  model->ZeroGrad();
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

// The parameter-only read skips the optimizer-state section
// (docs/checkpoint-format.md sections 9 and 22). A full file reads the same
// parameters in both modes, and a truncated optimizer section does not stop a
// parameter-only read.
void TestParameterOnlyReadSkipsOptimizerState() {
  const Config config = TinyConfig();
  const int batch = 2;
  const int seq = 8;
  const int rows = batch * seq;
  const std::string full_path = TempPath("parameter_only_full.ckpt");
  const std::string param_path = TempPath("parameter_only_params.ckpt");
  const std::string trunc_path = TempPath("parameter_only_truncated.ckpt");

  Rng rng(987654321);
  std::vector<int> tokens;
  std::vector<int> targets;
  MakeBatch(&rng, rows, config.vocab_size, &tokens, &targets);

  {
    std::unique_ptr<Model> model = Model::Create(config);
    model->InitWeights(kSeed);
    SchedulerConfig scheduler_config;
    Scheduler scheduler(scheduler_config);
    OptimizerConfig optimizer_config;
    std::unique_ptr<Optimizer> optimizer =
        nanochat::CreateOptimizer(model.get(), optimizer_config, scheduler);
    RunStep(model.get(), optimizer.get(), 1, tokens, targets, batch, seq);
    if (!Checkpointer::SaveModel(*model, *optimizer, 1, full_path)) {
      Fail("parameter-only read: full save failed");
      return;
    }
    if (!Checkpointer::SaveModel(*model, param_path)) {
      Fail("parameter-only read: parameter-only save failed");
      return;
    }
  }

  // The full read and the parameter-only read return the same parameters, and
  // only the full read carries the optimizer-state records.
  Checkpoint full;
  Checkpoint params_only;
  std::string error;
  if (!full.Load(full_path, &error)) {
    Fail("parameter-only read: full load failed: " + error);
    return;
  }
  if (!params_only.Load(full_path, &error, /*with_optimizer_state=*/false)) {
    Fail("parameter-only read: parameter-only load failed: " + error);
    return;
  }
  if (full.optimizer_state().empty()) {
    Fail("parameter-only read: the full read carries no optimizer state");
  }
  if (!params_only.optimizer_state().empty()) {
    Fail("parameter-only read: the parameter-only read kept optimizer state");
  }
  if (full.tensors().size() != params_only.tensors().size()) {
    Fail("parameter-only read: parameter count mismatch");
    return;
  }
  for (const TensorRecord& record : full.tensors()) {
    const TensorRecord* other = params_only.Find(record.name);
    if (other == nullptr) {
      Fail("parameter-only read: missing parameter " + record.name);
      continue;
    }
    if (record.dtype != other->dtype || record.shape != other->shape ||
        record.data != other->data) {
      Fail("parameter-only read: parameter mismatch " + record.name);
    }
  }

  // Cut the full file in the middle of the optimizer section. The parameter
  // section ends four bytes before the end of a parameter-only file, because
  // that file carries a trailing zero optimizer count. The full read fails on
  // the truncated record; the parameter-only read stops before the section and
  // succeeds.
  std::streamoff parameter_end = 0;
  {
    std::ifstream in(param_path, std::ios::binary | std::ios::ate);
    parameter_end = in.tellg();
  }
  if (parameter_end < 4) {
    Fail("parameter-only read: the parameter-only file is too short");
    return;
  }
  parameter_end -= 4;
  {
    std::ifstream in(full_path, std::ios::binary | std::ios::ate);
    const std::streamoff size = in.tellg();
    if (size < parameter_end + 5) {
      Fail("parameter-only read: the full file is too short");
      return;
    }
    in.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(parameter_end) + 5);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    std::ofstream out(trunc_path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  Checkpoint truncated_params;
  if (!truncated_params.Load(trunc_path, &error,
                             /*with_optimizer_state=*/false)) {
    Fail("parameter-only read: a truncated section stopped the read: " + error);
  }
  if (truncated_params.tensors().empty()) {
    Fail("parameter-only read: the truncated file carries no parameters");
  }
  Checkpoint truncated_full;
  if (truncated_full.Load(trunc_path, &error)) {
    Fail("parameter-only read: the full read accepted a truncated section");
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
  TestParameterOnlyReadSkipsOptimizerState();
  TestOptimizerStepRoundTrip();
  if (g_failures != 0) {
    std::fprintf(stderr, "optimizer_state_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("optimizer_state_test: ok\n");
  return 0;
}
