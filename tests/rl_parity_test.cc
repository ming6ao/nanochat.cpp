// tests/rl_parity_test.cc -- the reinforcement-learning parity gate.
//
// Reads the fixture produced by `tools/dump_rl_fixture.py` and drives the RL
// step over the recorded rollouts from the recorded initial parameters. The
// step is the shared `nanochat::RlStep` (docs/training-seam.md section 5.7), so
// the test and the file-driven `rl_step` binary cannot drift apart.
//
// The gate runs the same step two ways and compares them:
//
//   * in process, over the fixture's parameters and rollouts, and
//   * through the `//src:rl_step_main` binary, launched as a subprocess on the
//     same fixture.
//
// It compares the loss, the global gradient norm, and the L2 norm of every
// parameter after every step. It also checks the model-independent
// normalization the fixture records: the valid-target count and the divisor
// `num_valid * num_passes * examples_per_rank` (docs/post-training.md section
// 5.2). The fixture carries only the model-independent reference, because the
// host that generated it has no torch; the model trajectory parity is pinned by
// the two runs above, and the objective arithmetic is pinned by
// `//tests:numerics_trace_test`.
//
// The fixture is data, so the same source links the reference CPU backend by
// default and the CUDA backend under `--config=cuda`, and every device read
// goes through `kernels::Memcpy`.

#include <sys/wait.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "src/rl.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::CreateOptimizer;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
using nanochat::RlStep;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// The two runs share the same fp32 (or fp16) kernels and the same inputs, so
// the difference is roundoff from separate processes. These bounds match the
// training-parity gate: they leave headroom for a device backend's atomic
// reductions without hiding a real divergence.
constexpr double kLossAtol = 1e-4;
constexpr double kLossRtol = 5e-3;
constexpr double kGradAtol = 1e-4;
constexpr double kGradRtol = 1e-2;
constexpr double kParamL2Atol = 1e-4;
constexpr double kParamL2Rtol = 5e-3;

int g_failures = 0;
double g_max_loss_error = 0.0;
double g_max_grad_error = 0.0;
double g_max_param_error = 0.0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

bool Within(double got, double want, double atol, double rtol) {
  return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

std::string Getenv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : std::string();
}

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

Config ConfigFromFixture(const Fixture& fixture) {
  Config config;
  config.num_layers =
      static_cast<int>(fixture.Get("config/layers").scalar_int());
  config.num_heads = static_cast<int>(fixture.Get("config/heads").scalar_int());
  config.num_kv_heads =
      static_cast<int>(fixture.Get("config/kv_heads").scalar_int());
  config.hidden_dim = static_cast<int>(fixture.Get("config/embd").scalar_int());
  config.seq_len = static_cast<int>(fixture.Get("config/seq").scalar_int());
  config.vocab_size =
      static_cast<int>(fixture.Get("config/vocab").scalar_int());
  config.padded_vocab_size =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  config.rope_base = 100000.0f;
  const Tensor& pattern = fixture.Get("config/window_pattern");
  config.window_pattern =
      std::string(pattern.data, pattern.data + pattern.numel());
  return config;
}

OptimizerConfig OptimizerFromFixture(const Fixture& fixture) {
  OptimizerConfig opt;
  opt.unembedding_lr = fixture.Get("config/opt/unembedding_lr").scalar_f32();
  opt.embedding_lr = fixture.Get("config/opt/embedding_lr").scalar_f32();
  opt.matrix_lr = fixture.Get("config/opt/matrix_lr").scalar_f32();
  opt.scalar_lr = fixture.Get("config/opt/scalar_lr").scalar_f32();
  opt.weight_decay = fixture.Get("config/opt/weight_decay").scalar_f32();
  opt.clip = fixture.Get("config/opt/clip").scalar_f32();
  opt.adam_eps = fixture.Get("config/opt/adam_eps").scalar_f32();
  opt.muon_ns_steps =
      static_cast<int>(fixture.Get("config/opt/muon_ns_steps").scalar_int());
  opt.muon_beta2 = fixture.Get("config/opt/muon_beta2").scalar_f32();
  return opt;
}

SchedulerConfig SchedulerFromFixture(const Fixture& fixture) {
  SchedulerConfig sched;
  sched.num_iterations =
      static_cast<int>(fixture.Get("config/sched/num_iterations").scalar_int());
  sched.warmup_steps =
      static_cast<int>(fixture.Get("config/sched/warmup_steps").scalar_int());
  sched.warmdown_ratio =
      fixture.Get("config/sched/warmdown_ratio").scalar_f32();
  sched.final_lr_frac = fixture.Get("config/sched/final_lr_frac").scalar_f32();
  sched.weight_decay_base =
      fixture.Get("config/sched/weight_decay_base").scalar_f32();
  sched.muon_momentum_warmup_steps =
      fixture.Get("config/sched/muon_momentum_warmup_steps").scalar_f32();
  sched.muon_momentum_start =
      fixture.Get("config/sched/muon_momentum_start").scalar_f32();
  sched.muon_momentum_peak =
      fixture.Get("config/sched/muon_momentum_peak").scalar_f32();
  sched.muon_momentum_final =
      fixture.Get("config/sched/muon_momentum_final").scalar_f32();
  return sched;
}

// Loads `param/<name>` into every model parameter. The names match PyTorch's
// `named_parameters()` (proven by the oracle test), so the mapping is direct.
void LoadParameters(const Fixture& fixture, Model* model) {
  for (const ParamView& view : model->params()) {
    const std::string record = std::string("param/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing parameter record '" + record + "'");
      continue;
    }
    const Tensor& source = fixture.Get(record);
    if (source.numel() != view.count) {
      Fail(Format("parameter '%s' count mismatch: model %lld fixture %lld",
                  view.name, static_cast<long long>(view.count),
                  static_cast<long long>(source.numel())));
      continue;
    }
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i, source.f32()[i]);
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        CopyDir::kHostToDevice);
  }
}

std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  const std::string src_dir = Getenv("TEST_SRCDIR");
  const std::string workspace = Getenv("TEST_WORKSPACE");
  if (!src_dir.empty()) {
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace +
                           "/tests/data/rl_parity.bin");
    }
    candidates.push_back(src_dir + "/_main/tests/data/rl_parity.bin");
  }
  candidates.emplace_back("tests/data/rl_parity.bin");
  candidates.emplace_back("../tests/data/rl_parity.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "RL parity fixture not found; pass tests/data/rl_parity.bin as argv[1]");
}

// Locates the `rl_step` binary next to the test in the runfiles tree. The
// `//src:rl_step_main` data dependency is what puts it there.
std::string LocateRlStep() {
  std::vector<std::string> candidates;
  const std::string override = Getenv("NANOCHAT_RL_STEP");
  if (!override.empty()) candidates.push_back(override);
  const std::string src_dir = Getenv("TEST_SRCDIR");
  const std::string workspace = Getenv("TEST_WORKSPACE");
  if (!src_dir.empty()) {
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace + "/src/rl_step_main");
    }
    candidates.push_back(src_dir + "/_main/src/rl_step_main");
  }
  candidates.emplace_back("bazel-bin/src/rl_step_main");
  candidates.emplace_back("../bazel-bin/src/rl_step_main");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "rl_step_main not found in the runfiles tree; add //src:rl_step_main to "
      "the test data or set NANOCHAT_RL_STEP");
}

std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::string TempDir() {
  const std::string dir = Getenv("TEST_TMPDIR");
  return dir.empty() ? std::string("/tmp") : dir;
}

void CheckLoss(int step, double got, double want) {
  const double diff = std::fabs(got - want);
  g_max_loss_error = std::max(g_max_loss_error, diff);
  if (!Within(got, want, kLossAtol, kLossRtol)) {
    Fail(Format("step %d loss: got %.9g want %.9g (|diff|=%.3g)", step, got,
                want, diff));
  }
}

void CheckGradNorm(int step, double got, double want) {
  const double diff = std::fabs(got - want);
  g_max_grad_error = std::max(g_max_grad_error, diff);
  if (!Within(got, want, kGradAtol, kGradRtol)) {
    Fail(Format("step %d grad_norm: got %.9g want %.9g (|diff|=%.3g)", step,
                got, want, diff));
  }
}

void CheckParamL2(int step, const std::string& name, double got, double want) {
  const double diff = std::fabs(got - want);
  g_max_param_error = std::max(g_max_param_error, diff);
  if (!Within(got, want, kParamL2Atol, kParamL2Rtol)) {
    Fail(Format("step %d param_l2/%s: got %.9g want %.9g (|diff|=%.3g)", step,
                name.c_str(), got, want, diff));
  }
}

void Run(const std::string& fixture_path) {
  const Fixture fixture = Fixture::Load(fixture_path);
  if (fixture.Get("config/version").scalar_int() != 1) {
    throw std::runtime_error("unsupported RL-parity fixture version");
  }
  const Config config = ConfigFromFixture(fixture);
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int steps = static_cast<int>(fixture.Get("config/steps").scalar_int());
  const int num_passes =
      static_cast<int>(fixture.Get("config/num_passes").scalar_int());
  const int examples_per_rank =
      static_cast<int>(fixture.Get("config/examples_per_rank").scalar_int());
  if (batch <= 0 || seq <= 0 || steps <= 0 || num_passes <= 0 ||
      examples_per_rank <= 0) {
    throw std::runtime_error("the fixture has a nonpositive dimension");
  }
  const OptimizerConfig opt = OptimizerFromFixture(fixture);
  const SchedulerConfig sched = SchedulerFromFixture(fixture);
  const Scheduler scheduler(sched);

  // --- the expected trajectory, in process -------------------------------
  std::unique_ptr<Model> model = Model::Create(config);
  if (model == nullptr) throw std::runtime_error("cannot create the model");
  LoadParameters(fixture, model.get());
  std::unique_ptr<Optimizer> optimizer =
      CreateOptimizer(model.get(), opt, scheduler);
  if (optimizer == nullptr) {
    throw std::runtime_error("cannot create the optimizer");
  }

  std::vector<double> expected_loss(static_cast<std::size_t>(steps) + 1, 0.0);
  std::vector<double> expected_grad(static_cast<std::size_t>(steps) + 1, 0.0);
  std::vector<std::vector<double>> expected_param(
      static_cast<std::size_t>(steps) + 1,
      std::vector<double>(model->params().size(), 0.0));
  std::vector<int> expected_valid(static_cast<std::size_t>(steps) + 1, 0);

  for (int step = 1; step <= steps; ++step) {
    const std::string prefix = "rollout/" + std::to_string(step - 1) + "/";
    const Tensor& tokens = fixture.Get(prefix + "tokens");
    const Tensor& targets = fixture.Get(prefix + "targets");
    const Tensor& advantages = fixture.Get(prefix + "advantages");
    const int* token_data = reinterpret_cast<const int*>(tokens.i32());
    const int* target_data = reinterpret_cast<const int*>(targets.i32());
    const float* advantage_data = advantages.f32();

    int valid = 0;
    for (std::int64_t i = 0; i < targets.numel(); ++i) {
      if (target_data[i] != -1) ++valid;
    }
    expected_valid[static_cast<std::size_t>(step)] = valid;

    const float loss =
        RlStep(model.get(), optimizer.get(), token_data, target_data,
               advantage_data, batch, seq, num_passes, examples_per_rank, step);
    expected_loss[static_cast<std::size_t>(step)] = loss;
    expected_grad[static_cast<std::size_t>(step)] = optimizer->GradNorm();
    const std::vector<ParamView> params = model->params();
    for (std::size_t index = 0; index < params.size(); ++index) {
      expected_param[static_cast<std::size_t>(step)][index] =
          L2Norm(params[index].value, params[index].count);
    }
  }

  // --- the same trajectory, through the file-driven binary ---------------
  const std::string out_path = TempDir() + "/rl_parity_result.bin";
  const std::string binary = LocateRlStep();
  const std::string command = ShellQuote(binary) + " --fixture " +
                              ShellQuote(fixture_path) + " --out " +
                              ShellQuote(out_path);
  const int status = std::system(command.c_str());
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    Fail("rl_step_main did not run successfully (exit " +
         std::to_string(status) + ")");
    return;
  }
  const Fixture actual = Fixture::Load(out_path);
  if (actual.Get("result/version").scalar_int() != 1) {
    Fail("rl_step_main wrote an unsupported result version");
    return;
  }
  if (actual.Get("result/batch").scalar_int() != batch ||
      actual.Get("result/seq").scalar_int() != seq ||
      actual.Get("result/steps").scalar_int() != steps) {
    Fail("rl_step_main wrote the wrong result shape");
    return;
  }

  for (int step = 1; step <= steps; ++step) {
    const std::string prefix = "result/step/" + std::to_string(step);
    CheckLoss(step, actual.Get(prefix + "/loss").scalar_f32(),
              expected_loss[static_cast<std::size_t>(step)]);
    CheckGradNorm(step, actual.Get(prefix + "/grad_norm").scalar_f32(),
                  expected_grad[static_cast<std::size_t>(step)]);
    if (static_cast<int>(actual.Get(prefix + "/valid_targets").scalar_int()) !=
        expected_valid[static_cast<std::size_t>(step)]) {
      Fail(Format(
          "step %d valid_targets: got %d want %d", step,
          static_cast<int>(actual.Get(prefix + "/valid_targets").scalar_int()),
          expected_valid[static_cast<std::size_t>(step)]));
    }
    const std::vector<ParamView> params = model->params();
    for (std::size_t index = 0; index < params.size(); ++index) {
      const std::string record = prefix + "/param_l2/" + params[index].name;
      CheckParamL2(step, params[index].name, actual.Get(record).scalar_f32(),
                   expected_param[static_cast<std::size_t>(step)][index]);
    }
  }

  // --- the model-independent normalization reference ---------------------
  for (int step = 0; step < steps; ++step) {
    const std::string prefix = "ref/" + std::to_string(step) + "/";
    const int want_valid =
        static_cast<int>(fixture.Get(prefix + "valid_targets").scalar_int());
    const std::int64_t want_divisor =
        fixture.Get(prefix + "divisor").scalar_int();
    const std::int64_t expected_divisor =
        static_cast<std::int64_t>(want_valid) * num_passes * examples_per_rank;
    if (want_valid != expected_valid[static_cast<std::size_t>(step + 1)]) {
      Fail(Format("ref/%d/valid_targets: got %d want %d", step, want_valid,
                  expected_valid[static_cast<std::size_t>(step + 1)]));
    }
    if (want_divisor != expected_divisor) {
      Fail(Format("ref/%d/divisor: got %lld want %lld", step,
                  static_cast<long long>(want_divisor),
                  static_cast<long long>(expected_divisor)));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(LocateFixture(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rl_parity: %s\n", error.what());
    return 1;
  }
  std::printf(
      "rl_parity: max errors (loss %.3g, grad_norm %.3g, param_l2 %.3g)%s\n",
      g_max_loss_error, g_max_grad_error, g_max_param_error,
      g_failures == 0 ? "" : " [FAILED]");
  return g_failures == 0 ? 0 : 1;
}
