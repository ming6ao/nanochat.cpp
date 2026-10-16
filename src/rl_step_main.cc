// `rl_step` -- the file-driven reinforcement-learning step.
//
// It is the RL analog of the training-parity harness (docs/rl-notebook.md
// section 4, docs/training-seam.md section 6.4, docs/post-training.md section
// 12). It reads a fixture that holds the model configuration, the optimizer and
// schedule configuration, the initial parameters, and, per step, a recorded
// rollout (tokens and targets with `-1` at every ignored position) and the
// per-position advantage weights. It runs the *shared* RL step
// (`nanochat::RlStep`, the same code the C ABI and the persistent worker call),
// and reports the loss, the global gradient norm, and the L2 norm of every
// parameter after every step. It writes those as a result fixture and prints a
// summary.
//
// The step is shared, not copied: the divisor `num_valid * num_passes *
// examples_per_rank` is computed in C++ by `RlStep` (docs/training-seam.md
// section 5.7), so this binary never computes it.
//
// The fixture wire format is the little-endian `NANOORC1` container of
// `src/container.h`; the fixture is produced by
// `tools/dump_training_fixtures.py rl`. The result file carries only `result/`
// records, so a reader can parse it on its own.
//
//   tools/nanochat run t0-cpu -- <rl_step> --fixture in.bin --out out.bin

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "src/rl.h"
#include "src/rl_fixture.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::rl_fixture::FindRecord;
using nanochat::rl_fixture::FloatAt;
using nanochat::rl_fixture::IntAt;
using nanochat::rl_fixture::kFp32;
using nanochat::rl_fixture::kFp64;
using nanochat::rl_fixture::kInt32;
using nanochat::rl_fixture::kInt64;
using nanochat::rl_fixture::kUInt8;
using nanochat::rl_fixture::kVersion;
using nanochat::rl_fixture::LoadRecords;
using nanochat::rl_fixture::MakeFloat32;
using nanochat::rl_fixture::MakeInt32;
using nanochat::rl_fixture::ReadFloat32Vector;
using nanochat::rl_fixture::ReadInt32Vector;
using nanochat::rl_fixture::Record;
using nanochat::rl_fixture::SaveRecords;
using nanochat::rl_fixture::ScalarFloat;
using nanochat::rl_fixture::ScalarInt;

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

// Copies a (possibly device-resident) buffer to host floats.
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

Config ConfigFromFixture(const std::vector<Record>& records,
                         std::string* error) {
  Config config;
  std::int32_t value = 0;
  auto int_field = [&](const char* name, int* out) -> bool {
    if (!ScalarInt(records, name, &value, error)) return false;
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
  const Record* pattern = FindRecord(records, "config/window_pattern");
  if (pattern != nullptr) {
    config.window_pattern =
        std::string(pattern->data.begin(), pattern->data.end());
  }
  return config;
}

OptimizerConfig OptimizerFromFixture(const std::vector<Record>& records,
                                     std::string* error) {
  OptimizerConfig opt;
  if (!ScalarFloat(records, "config/opt/unembedding_lr", &opt.unembedding_lr,
                   error) ||
      !ScalarFloat(records, "config/opt/embedding_lr", &opt.embedding_lr,
                   error) ||
      !ScalarFloat(records, "config/opt/matrix_lr", &opt.matrix_lr, error) ||
      !ScalarFloat(records, "config/opt/scalar_lr", &opt.scalar_lr, error) ||
      !ScalarFloat(records, "config/opt/weight_decay", &opt.weight_decay,
                   error) ||
      !ScalarFloat(records, "config/opt/clip", &opt.clip, error) ||
      !ScalarFloat(records, "config/opt/adam_eps", &opt.adam_eps, error) ||
      !ScalarFloat(records, "config/opt/muon_beta2", &opt.muon_beta2, error)) {
    return opt;
  }
  std::int32_t ns_steps = 0;
  if (!ScalarInt(records, "config/opt/muon_ns_steps", &ns_steps, error)) {
    return opt;
  }
  opt.muon_ns_steps = static_cast<int>(ns_steps);
  return opt;
}

SchedulerConfig SchedulerFromFixture(const std::vector<Record>& records,
                                     std::string* error) {
  SchedulerConfig sched;
  std::int32_t int_value = 0;
  if (!ScalarInt(records, "config/sched/num_iterations", &int_value, error)) {
    return sched;
  }
  sched.num_iterations = static_cast<int>(int_value);
  if (!ScalarInt(records, "config/sched/warmup_steps", &int_value, error)) {
    return sched;
  }
  sched.warmup_steps = static_cast<int>(int_value);
  if (!ScalarFloat(records, "config/sched/warmdown_ratio",
                   &sched.warmdown_ratio, error) ||
      !ScalarFloat(records, "config/sched/final_lr_frac", &sched.final_lr_frac,
                   error) ||
      !ScalarFloat(records, "config/sched/weight_decay_base",
                   &sched.weight_decay_base, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_warmup_steps",
                   &sched.muon_momentum_warmup_steps, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_start",
                   &sched.muon_momentum_start, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_peak",
                   &sched.muon_momentum_peak, error) ||
      !ScalarFloat(records, "config/sched/muon_momentum_final",
                   &sched.muon_momentum_final, error)) {
    return sched;
  }
  return sched;
}

// Loads `param/<name>` into every model parameter. The names match PyTorch's
// `named_parameters()` (proven by the oracle test), so the mapping is direct.
bool LoadParameters(const std::vector<Record>& records, Model* model,
                    std::string* error) {
  for (const ParamView& view : model->params()) {
    const std::string name = std::string("param/") + view.name;
    const Record* source = FindRecord(records, name);
    if (source == nullptr || source->dtype != kFp32 ||
        source->numel() != view.count) {
      *error = "missing or malformed parameter record '" + name + "'";
      return false;
    }
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i,
                 FloatAt(*source, static_cast<std::size_t>(i)));
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
      "usage: rl_step --fixture PATH --out PATH\n"
      "  --fixture PATH   input RL fixture with parameters and "
      "rollouts\n"
      "  --out PATH       output fixture with the reported trajectory\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("rl_step");

  std::string fixture_path;
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "rl_step: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    if (flag == "--fixture" || flag == "--in" || flag == "--input") {
      fixture_path = value;
    } else if (flag == "--out" || flag == "--output") {
      out_path = value;
    } else {
      std::fprintf(stderr, "rl_step: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }
  if (fixture_path.empty() || out_path.empty()) {
    std::fprintf(stderr, "rl_step: --fixture and --out are required\n");
    Usage();
    return 2;
  }

  std::vector<Record> records;
  std::string error;
  if (!LoadRecords(fixture_path, &records, &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }

  std::int32_t version = 0;
  if (!ScalarInt(records, "config/version", &version, &error) || version != 1) {
    std::fprintf(stderr, "rl_step: unsupported fixture version\n");
    return 1;
  }

  const Config config = ConfigFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const OptimizerConfig opt = OptimizerFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const SchedulerConfig sched = SchedulerFromFixture(records, &error);
  if (!error.empty()) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  const Scheduler scheduler(sched);

  std::int32_t batch = 0;
  std::int32_t seq = 0;
  std::int32_t steps = 0;
  std::int32_t num_passes = 0;
  std::int32_t examples_per_rank = 0;
  if (!ScalarInt(records, "config/batch", &batch, &error) ||
      !ScalarInt(records, "config/seq", &seq, &error) ||
      !ScalarInt(records, "config/steps", &steps, &error) ||
      !ScalarInt(records, "config/num_passes", &num_passes, &error) ||
      !ScalarInt(records, "config/examples_per_rank", &examples_per_rank,
                 &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  if (batch <= 0 || seq <= 0 || steps <= 0 || num_passes <= 0 ||
      examples_per_rank <= 0) {
    std::fprintf(stderr, "rl_step: the fixture has a nonpositive dimension\n");
    return 1;
  }

  std::unique_ptr<Model> model = Model::Create(config);
  if (model == nullptr) {
    std::fprintf(stderr, "rl_step: cannot create the model\n");
    return 1;
  }
  if (!LoadParameters(records, model.get(), &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), opt, scheduler);
  if (optimizer == nullptr) {
    std::fprintf(stderr, "rl_step: cannot create the optimizer\n");
    return 1;
  }

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<Record> output;
  output.push_back(
      MakeInt32("result/version", {1}, {static_cast<std::int32_t>(kVersion)}));
  output.push_back(MakeInt32("result/batch", {1}, {batch}));
  output.push_back(MakeInt32("result/seq", {1}, {seq}));
  output.push_back(MakeInt32("result/steps", {1}, {steps}));

  for (int step = 1; step <= steps; ++step) {
    const std::string prefix = "rollout/" + std::to_string(step - 1) + "/";
    std::vector<std::int32_t> tokens_i32;
    std::vector<std::int32_t> targets_i32;
    std::vector<float> advantages;
    if (!ReadInt32Vector(records, prefix + "tokens", rows, &tokens_i32,
                         &error) ||
        !ReadInt32Vector(records, prefix + "targets", rows, &targets_i32,
                         &error) ||
        !ReadFloat32Vector(records, prefix + "advantages", rows, &advantages,
                           &error)) {
      std::fprintf(stderr, "rl_step: %s\n", error.c_str());
      return 1;
    }
    const std::vector<int> tokens(tokens_i32.begin(), tokens_i32.end());
    const std::vector<int> targets(targets_i32.begin(), targets_i32.end());

    const float loss = nanochat::RlStep(
        model.get(), optimizer.get(), tokens.data(), targets.data(),
        advantages.data(), batch, seq, num_passes, examples_per_rank, step);

    std::int32_t valid = 0;
    for (int index = 0; index < static_cast<int>(rows); ++index) {
      if (targets[static_cast<std::size_t>(index)] != -1) ++valid;
    }
    const std::string step_prefix = "result/step/" + std::to_string(step);
    output.push_back(MakeFloat32(step_prefix + "/loss", {1}, {loss}));
    output.push_back(
        MakeFloat32(step_prefix + "/grad_norm", {1}, {optimizer->GradNorm()}));
    output.push_back(MakeInt32(step_prefix + "/valid_targets", {1}, {valid}));
    for (const ParamView& view : model->params()) {
      const double l2 = L2Norm(view.value, view.count);
      output.push_back(MakeFloat32(step_prefix + "/param_l2/" + view.name, {1},
                                   {static_cast<float>(l2)}));
    }
    std::printf("rl_step: step %d loss %.6g grad_norm %.6g valid %d\n", step,
                static_cast<double>(loss),
                static_cast<double>(optimizer->GradNorm()), valid);
  }

  if (!SaveRecords(out_path, output, &error)) {
    std::fprintf(stderr, "rl_step: %s\n", error.c_str());
    return 1;
  }
  std::printf("rl_step: ran %d step(s) over %d x %d, wrote %s\n", steps, batch,
              seq, out_path.c_str());
  return 0;
}
