// `rl_worker` -- the persistent reinforcement-learning worker
// (docs/rl-notebook.md section 4, docs/training-seam.md section 6.4).
//
// The process holds the model and the optimizer across steps and speaks the
// pipe protocol documented in `src/rl_worker.h`. On a workstation the facade
// launches it through `tools/nanochat train -- <worker>`; on Kaggle the facade
// starts it directly under `NANOCHAT_SANDBOX_BACKEND=none`
// (docs/rl-notebook.md section 6).
//
// It also offers a fixture-driven smoke mode so the worker can be gated
// without the Python bridge:
//
//   tools/nanochat verify -- ./bazel-bin/src/rl_worker
//       --fixture tests/data/rl_parity.bin --steps 2
//
// The smoke run replays the recorded rollouts through the same shared
// `nanochat::RlStep` the bridge would reach, so a passing run proves the worker
// end to end without generation.
//
// Every run must go through `tools/nanochat`; the entry point refuses to start
// outside the sandbox (`RequireSandboxOrDie`).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "src/cli.h"
#include "src/rl_fixture.h"
#include "src/rl_worker.h"
#include "src/train.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
using nanochat::RlWorker;
using nanochat::SchedulerConfig;
namespace rf = nanochat::rl_fixture;

template <typename T = ComputeType>
float AsFloat32(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return nanochat::Fp16ToFloat(value);
  }
}

template <typename T = ComputeType>
void StoreFloat(T* dst, float value) {
  if constexpr (std::is_same_v<T, float>) {
    *dst = value;
  } else {
    *dst = nanochat::Fp16FromFloat(value);
  }
}

std::vector<float> ReadCompute(const ComputeType* src, std::int64_t count) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src, static_cast<std::size_t>(count) * sizeof(ComputeType),
        CopyDir::kDeviceToHost);
  }
  std::vector<float> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] =
        AsFloat32(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

double L2Norm(const ComputeType* data, std::int64_t count) {
  const std::vector<float> host = ReadCompute(data, count);
  double sum = 0.0;
  for (float value : host) sum += static_cast<double>(value) * value;
  return std::sqrt(sum);
}

Config ConfigFromFixture(const std::vector<rf::Record>& records,
                         std::string* error) {
  Config config;
  std::int32_t value = 0;
  auto int_field = [&](const char* name, int* out) -> bool {
    if (!rf::ScalarInt(records, name, &value, error)) return false;
    *out = static_cast<int>(value);
    return true;
  };
  if (!int_field("config/layers", &config.num_layers) ||
      !int_field("config/heads", &config.num_heads) ||
      !int_field("config/kv_heads", &config.num_kv_heads) ||
      !int_field("config/embd", &config.hidden_dim) ||
      !int_field("config/seq", &config.seq_len) ||
      !int_field("config/vocab", &config.vocab_size) ||
      !int_field("config/padded_vocab", &config.padded_vocab_size)) {
    return config;
  }
  config.rope_base = 100000.0f;
  const rf::Record* pattern = rf::FindRecord(records, "config/window_pattern");
  if (pattern != nullptr) {
    config.window_pattern =
        std::string(pattern->data.begin(), pattern->data.end());
  }
  return config;
}

OptimizerConfig OptimizerFromFixture(const std::vector<rf::Record>& records,
                                     std::string* error) {
  OptimizerConfig opt;
  if (!rf::ScalarFloat(records, "config/opt/unembedding_lr",
                       &opt.unembedding_lr, error) ||
      !rf::ScalarFloat(records, "config/opt/embedding_lr", &opt.embedding_lr,
                       error) ||
      !rf::ScalarFloat(records, "config/opt/matrix_lr", &opt.matrix_lr,
                       error) ||
      !rf::ScalarFloat(records, "config/opt/scalar_lr", &opt.scalar_lr,
                       error) ||
      !rf::ScalarFloat(records, "config/opt/weight_decay", &opt.weight_decay,
                       error) ||
      !rf::ScalarFloat(records, "config/opt/clip", &opt.clip, error) ||
      !rf::ScalarFloat(records, "config/opt/adam_eps", &opt.adam_eps, error) ||
      !rf::ScalarFloat(records, "config/opt/muon_beta2", &opt.muon_beta2,
                       error)) {
    return opt;
  }
  std::int32_t ns_steps = 0;
  if (!rf::ScalarInt(records, "config/opt/muon_ns_steps", &ns_steps, error)) {
    return opt;
  }
  opt.muon_ns_steps = static_cast<int>(ns_steps);
  return opt;
}

SchedulerConfig SchedulerFromFixture(const std::vector<rf::Record>& records,
                                     std::string* error) {
  SchedulerConfig sched;
  std::int32_t int_value = 0;
  if (!rf::ScalarInt(records, "config/sched/num_iterations", &int_value,
                     error)) {
    return sched;
  }
  sched.num_iterations = static_cast<int>(int_value);
  if (!rf::ScalarInt(records, "config/sched/warmup_steps", &int_value, error)) {
    return sched;
  }
  sched.warmup_steps = static_cast<int>(int_value);
  if (!rf::ScalarFloat(records, "config/sched/warmdown_ratio",
                       &sched.warmdown_ratio, error) ||
      !rf::ScalarFloat(records, "config/sched/final_lr_frac",
                       &sched.final_lr_frac, error) ||
      !rf::ScalarFloat(records, "config/sched/weight_decay_base",
                       &sched.weight_decay_base, error) ||
      !rf::ScalarFloat(records, "config/sched/muon_momentum_warmup_steps",
                       &sched.muon_momentum_warmup_steps, error) ||
      !rf::ScalarFloat(records, "config/sched/muon_momentum_start",
                       &sched.muon_momentum_start, error) ||
      !rf::ScalarFloat(records, "config/sched/muon_momentum_peak",
                       &sched.muon_momentum_peak, error) ||
      !rf::ScalarFloat(records, "config/sched/muon_momentum_final",
                       &sched.muon_momentum_final, error)) {
    return sched;
  }
  return sched;
}

// Loads `param/<name>` into every model parameter. The names match PyTorch's
// `named_parameters()` (proven by the oracle test).
bool LoadParameters(const std::vector<rf::Record>& records, Model* model,
                    std::string* error) {
  for (const ParamView& view : model->params()) {
    const std::string name = std::string("param/") + view.name;
    const rf::Record* source = rf::FindRecord(records, name);
    if (source == nullptr || source->dtype != rf::kFp32 ||
        source->numel() != view.count) {
      *error = "missing or malformed parameter record '" + name + "'";
      return false;
    }
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i,
                 rf::FloatAt(*source, static_cast<std::size_t>(i)));
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        CopyDir::kHostToDevice);
  }
  return true;
}

void Usage() {
  std::fprintf(
      stderr,
      "usage: rl_worker [options]\n"
      "  --fixture PATH        load config, optimizer, schedule, and initial\n"
      "                        parameters from an NANOORC1 RL fixture\n"
      "  --checkpoint PATH     load parameters and optimizer state from an\n"
      "                        NCHKPT01 checkpoint\n"
      "  --steps N             fixture smoke mode: replay N recorded rollouts\n"
      "                        and exit (requires --fixture)\n"
      "  --out PATH            smoke mode: write the result fixture here\n"
      "  --save PATH           write a checkpoint on exit\n"
      "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
      "   --vocab --padded-vocab --window-pattern --rope-base]\n"
      "  [optimizer flags: --embedding-lr --unembedding-lr --matrix-lr\n"
      "   --scalar-lr --weight-decay --clip --adam-eps --muon-ns-steps\n"
      "   --muon-beta2 --matrix-optimizer --adam-step-period]\n"
      "  [schedule flags: --num-iterations --warmup-steps --warmdown-ratio\n"
      "   --final-lr-frac --weight-decay-base --muon-momentum-warmup-steps\n"
      "   --muon-momentum-start --muon-momentum-peak --muon-momentum-final]\n"
      "\n"
      "Without --steps the worker serves the pipe protocol on stdin/stdout\n"
      "(src/rl_worker.h).\n");
}

// Runs the recorded rollouts of `records` through the shared step.
int SmokeRun(RlWorker* worker, const std::vector<rf::Record>& records,
             int batch, int seq, int num_passes, int examples_per_rank,
             int requested_steps, const std::string& out_path) {
  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<rf::Record> output;
  if (!out_path.empty()) {
    output.push_back(rf::MakeInt32("result/version", {1},
                                   {static_cast<std::int32_t>(rf::kVersion)}));
    output.push_back(rf::MakeInt32("result/batch", {1}, {batch}));
    output.push_back(rf::MakeInt32("result/seq", {1}, {seq}));
    output.push_back(rf::MakeInt32("result/steps", {1}, {requested_steps}));
  }

  std::string error;
  for (int step = 1; step <= requested_steps; ++step) {
    const std::string prefix = "rollout/" + std::to_string(step - 1) + "/";
    std::vector<std::int32_t> tokens_i32;
    std::vector<std::int32_t> targets_i32;
    std::vector<float> advantages;
    if (!rf::ReadInt32Vector(records, prefix + "tokens", rows, &tokens_i32,
                             &error) ||
        !rf::ReadInt32Vector(records, prefix + "targets", rows, &targets_i32,
                             &error) ||
        !rf::ReadFloat32Vector(records, prefix + "advantages", rows,
                               &advantages, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    const std::vector<int> tokens(tokens_i32.begin(), tokens_i32.end());
    const std::vector<int> targets(targets_i32.begin(), targets_i32.end());

    const nanochat::RlWorkerResult result =
        worker->Step(tokens.data(), targets.data(), advantages.data(), batch,
                     seq, num_passes, examples_per_rank);

    if (!out_path.empty()) {
      const std::string step_prefix = "result/step/" + std::to_string(step);
      output.push_back(
          rf::MakeFloat32(step_prefix + "/loss", {1}, {result.loss}));
      output.push_back(
          rf::MakeFloat32(step_prefix + "/grad_norm", {1}, {result.grad_norm}));
      output.push_back(rf::MakeInt32(step_prefix + "/valid_targets", {1},
                                     {result.valid_targets}));
      for (const ParamView& view : worker->model()->params()) {
        const double l2 = L2Norm(view.value, view.count);
        output.push_back(rf::MakeFloat32(step_prefix + "/param_l2/" + view.name,
                                         {1}, {static_cast<float>(l2)}));
      }
    }
    std::printf("rl_worker: step %d loss %.6g grad_norm %.6g valid %d\n", step,
                static_cast<double>(result.loss),
                static_cast<double>(result.grad_norm), result.valid_targets);
  }

  if (!out_path.empty() && !rf::SaveRecords(out_path, output, &error)) {
    std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("rl_worker");

  Config model_config;
  OptimizerConfig optimizer_config;
  SchedulerConfig scheduler_config;
  std::string fixture_path;
  std::string checkpoint_path;
  std::string save_path;
  std::string out_path;
  int requested_steps = 0;
  bool have_steps = false;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "rl_worker: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "rl_worker: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    float parsed_float = 0.0f;
    if (flag == "--fixture") {
      fixture_path = value;
    } else if (flag == "--checkpoint") {
      checkpoint_path = value;
    } else if (flag == "--save") {
      save_path = value;
    } else if (flag == "--out") {
      out_path = value;
    } else if (flag == "--steps") {
      if (!nanochat::cli::ParseInt(value, &parsed_int) || parsed_int <= 0) {
        std::fprintf(stderr, "rl_worker: bad --steps value\n");
        return 2;
      }
      requested_steps = parsed_int;
      have_steps = true;
    } else if (flag == "--embedding-lr") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.embedding_lr = parsed_float;
    } else if (flag == "--unembedding-lr") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.unembedding_lr = parsed_float;
    } else if (flag == "--matrix-lr") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.matrix_lr = parsed_float;
    } else if (flag == "--scalar-lr") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.scalar_lr = parsed_float;
    } else if (flag == "--weight-decay") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.weight_decay = parsed_float;
    } else if (flag == "--clip") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.clip = parsed_float;
    } else if (flag == "--adam-eps") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.adam_eps = parsed_float;
    } else if (flag == "--muon-ns-steps") {
      optimizer_config.muon_ns_steps =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 5;
    } else if (flag == "--muon-beta2") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      optimizer_config.muon_beta2 = parsed_float;
    } else if (flag == "--matrix-optimizer") {
      optimizer_config.matrix_optimizer =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--adam-step-period") {
      optimizer_config.adam_step_period =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 1;
    } else if (flag == "--num-iterations") {
      scheduler_config.num_iterations =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--warmup-steps") {
      scheduler_config.warmup_steps =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--warmdown-ratio") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.warmdown_ratio = parsed_float;
    } else if (flag == "--final-lr-frac") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.final_lr_frac = parsed_float;
    } else if (flag == "--weight-decay-base") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.weight_decay_base = parsed_float;
    } else if (flag == "--muon-momentum-warmup-steps") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.muon_momentum_warmup_steps = parsed_float;
    } else if (flag == "--muon-momentum-start") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.muon_momentum_start = parsed_float;
    } else if (flag == "--muon-momentum-peak") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.muon_momentum_peak = parsed_float;
    } else if (flag == "--muon-momentum-final") {
      if (!nanochat::cli::ParseFloat(value, &parsed_float)) parsed_float = 0.0f;
      scheduler_config.muon_momentum_final = parsed_float;
    } else {
      std::fprintf(stderr, "rl_worker: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  // --- load the fixture configuration and parameters ---------------------
  std::vector<rf::Record> records;
  int batch = 1;
  int seq = 0;
  int fixture_steps = 0;
  int num_passes = 1;
  int examples_per_rank = 1;
  if (!fixture_path.empty()) {
    std::string error;
    if (!rf::LoadRecords(fixture_path, &records, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    std::int32_t version = 0;
    if (!rf::ScalarInt(records, "config/version", &version, &error) ||
        version != 1) {
      std::fprintf(stderr, "rl_worker: unsupported fixture version\n");
      return 1;
    }
    model_config = ConfigFromFixture(records, &error);
    if (!error.empty()) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    optimizer_config = OptimizerFromFixture(records, &error);
    if (!error.empty()) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    scheduler_config = SchedulerFromFixture(records, &error);
    if (!error.empty()) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    std::int32_t value = 0;
    if (!rf::ScalarInt(records, "config/batch", &value, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    batch = static_cast<int>(value);
    if (!rf::ScalarInt(records, "config/seq", &value, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    seq = static_cast<int>(value);
    if (!rf::ScalarInt(records, "config/steps", &value, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
    fixture_steps = static_cast<int>(value);
    if (!rf::ScalarInt(records, "config/num_passes", &value, &error) ||
        value <= 0) {
      std::fprintf(stderr, "rl_worker: bad config/num_passes\n");
      return 1;
    }
    num_passes = static_cast<int>(value);
    if (!rf::ScalarInt(records, "config/examples_per_rank", &value, &error) ||
        value <= 0) {
      std::fprintf(stderr, "rl_worker: bad config/examples_per_rank\n");
      return 1;
    }
    examples_per_rank = static_cast<int>(value);
  }

  if (have_steps && fixture_path.empty()) {
    std::fprintf(stderr, "rl_worker: --steps needs --fixture\n");
    return 2;
  }
  if (fixture_path.empty() && checkpoint_path.empty()) {
    std::fprintf(stderr, "rl_worker: need --fixture or --checkpoint\n");
    return 2;
  }

  RlWorker worker(model_config, optimizer_config, scheduler_config);
  if (worker.model() == nullptr || worker.optimizer() == nullptr) {
    std::fprintf(stderr, "rl_worker: cannot create the model or optimizer\n");
    return 1;
  }

  if (!fixture_path.empty()) {
    std::string error;
    if (!LoadParameters(records, worker.model(), &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
  }
  if (!checkpoint_path.empty()) {
    int step = 0;
    if (!nanochat::Checkpointer::LoadModel(worker.model(), worker.optimizer(),
                                           &step, checkpoint_path)) {
      std::fprintf(stderr, "rl_worker: cannot load checkpoint %s\n",
                   checkpoint_path.c_str());
      return 1;
    }
    worker.set_step(step);
  }

  int status = 0;
  if (have_steps) {
    if (requested_steps > fixture_steps) {
      std::fprintf(stderr,
                   "rl_worker: --steps %d exceeds the fixture's %d steps\n",
                   requested_steps, fixture_steps);
      return 2;
    }
    status = SmokeRun(&worker, records, batch, seq, num_passes,
                      examples_per_rank, requested_steps, out_path);
  } else {
    std::fprintf(stderr, "rl_worker: ready (step %d)\n", worker.step());
    status = worker.Serve(std::cin, std::cout);
  }

  if (status == 0 && !save_path.empty()) {
    std::string error;
    if (!worker.SaveCheckpoint(save_path, &error)) {
      std::fprintf(stderr, "rl_worker: %s\n", error.c_str());
      return 1;
    }
  }
  return status;
}
