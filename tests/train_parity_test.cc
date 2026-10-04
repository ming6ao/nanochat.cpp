// tests/train_parity_test.cc -- the multi-step training-parity gate.
//
// Reads a fixture produced by `tools/dump_train_fixture.py` and drives
// `nanochat.cpp`'s `Model::TrainStep` over the recorded batches from the
// recorded initial parameters, with the recorded optimizer and learning-rate
// schedule. After every step it compares, against the PyTorch reference:
//
//   * the post-step loss,
//   * the global gradient norm of that step, and
//   * the L2 norm of every parameter tensor (a compact trajectory checksum),
//
// and, when present, the final parameter tensors.
//
// This is the scaled-up version of `tests/oracle_test.cc`: the oracle pins one
// forward, one backward, and a couple of optimizer steps on tiny shapes; this
// test pins a full training trajectory. The fixture is data, so the same source
// links the reference CPU backend by default and the CUDA backend under
// `--config=cuda`, and every device read goes through `kernels::Memcpy`.
//
// The source also builds as the standalone `//tests:train_parity` binary so a
// longer production fixture can be run under the sandbox and the GPU broker:
//
//   tools/nanochat gpu --profile t2-parity -- ./bazel-bin/tests/train_parity
//       /tmp/train_parity_d8_s512.bin

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
#include "src/model_impl.h"
#include "tests/oracle_fixture.h"

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
using nanochat::TrainModel;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// Tolerances. Both sides run fp32 from the same initial parameters over the
// same batches, so the difference is roundoff that can accumulate over the
// trajectory. These bounds are a relative-plus-absolute match; the measured
// maxima are always printed so a slow drift is visible even when it passes.
constexpr double kLossAtol = 2e-3;
constexpr double kLossRtol = 2e-3;
constexpr double kParamL2Atol = 1e-3;
constexpr double kParamL2Rtol = 5e-3;
constexpr double kGradNormAtol = 1e-3;
constexpr double kGradNormRtol = 1e-2;

int g_failures = 0;
bool g_stop_on_first_failure = false;
double g_max_loss_error = 0.0;
double g_max_param_l2_error = 0.0;
double g_max_grad_l2_error = 0.0;
double g_max_grad_norm_error = 0.0;
double g_max_final_param_error = 0.0;

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
  if (fixture.Has("config/opt/muon_ns_steps")) {
    opt.muon_ns_steps =
        static_cast<int>(fixture.Get("config/opt/muon_ns_steps").scalar_int());
  }
  if (fixture.Has("config/opt/muon_beta2")) {
    opt.muon_beta2 = fixture.Get("config/opt/muon_beta2").scalar_f32();
  }
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

void CheckLoss(const Fixture& fixture, int step, double loss) {
  const std::string record = "step/" + std::to_string(step) + "/loss";
  const double want = fixture.Get(record).scalar_f32();
  const double diff = std::fabs(loss - want);
  g_max_loss_error = std::max(g_max_loss_error, diff);
  if (!Within(loss, want, kLossAtol, kLossRtol)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", record.c_str(), loss,
                want, diff));
  }
}

void CheckParamL2(const Fixture& fixture, int step, const ParamView& view,
                  double l2) {
  const std::string record =
      "step/" + std::to_string(step) + "/param_l2/" + view.name;
  if (!fixture.Has(record)) {
    Fail("fixture is missing record '" + record + "'");
    return;
  }
  const double want = fixture.Get(record).scalar_f32();
  const double diff = std::fabs(l2 - want);
  g_max_param_l2_error = std::max(g_max_param_l2_error, diff);
  if (!Within(l2, want, kParamL2Atol, kParamL2Rtol)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", record.c_str(), l2,
                want, diff));
  }
}

void CheckGradL2(const Fixture& fixture, int step, const ParamView& view,
                 double l2) {
  const std::string record =
      "step/" + std::to_string(step) + "/grad_l2/" + view.name;
  if (!fixture.Has(record)) return;  // older fixtures omit per-parameter grads
  const double want = fixture.Get(record).scalar_f32();
  const double diff = std::fabs(l2 - want);
  g_max_grad_l2_error = std::max(g_max_grad_l2_error, diff);
  if (!Within(l2, want, kParamL2Atol, kParamL2Rtol)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", record.c_str(), l2,
                want, diff));
  }
}

void Run(const std::string& path) {
  const Fixture fixture = Fixture::Load(path);
  if (fixture.Get("config/version").scalar_int() != 1) {
    throw std::runtime_error("unsupported training-parity fixture version");
  }

  const Config config = ConfigFromFixture(fixture);
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int steps = static_cast<int>(fixture.Get("config/steps").scalar_int());
  const OptimizerConfig opt = OptimizerFromFixture(fixture);
  const SchedulerConfig sched = SchedulerFromFixture(fixture);
  const Scheduler scheduler(sched);

  std::unique_ptr<Model> model = Model::Create(config);
  LoadParameters(fixture, model.get());
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), opt, scheduler);

  auto batch_tokens = [&](int step) -> const int* {
    return reinterpret_cast<const int*>(
        fixture.Get("batch/" + std::to_string(step) + "/tokens").i32());
  };
  auto batch_targets = [&](int step) -> const int* {
    return reinterpret_cast<const int*>(
        fixture.Get("batch/" + std::to_string(step) + "/targets").i32());
  };

  // Step 0: initial loss and initial parameter trajectory point.
  const float loss0 =
      model->ForwardLoss(batch_tokens(0), batch_targets(0), batch, seq);
  CheckLoss(fixture, 0, loss0);
  for (const ParamView& view : model->params()) {
    CheckParamL2(fixture, 0, view, L2Norm(view.value, view.count));
  }

  for (int step = 1; step <= steps; ++step) {
    const int failures_before = g_failures;
    model->TrainStep(batch_tokens(step - 1), batch_targets(step - 1), batch,
                     seq, optimizer.get());

    // Recompute the loss on the updated parameters: `TrainStep` returns the
    // pre-update loss, the fixture records the post-update loss.
    const float step_loss = model->ForwardLoss(
        batch_tokens(step - 1), batch_targets(step - 1), batch, seq);
    CheckLoss(fixture, step, step_loss);

    {
      const std::string record = "step/" + std::to_string(step) + "/grad_norm";
      const double got = optimizer->GradNorm();
      const double want = fixture.Get(record).scalar_f32();
      const double diff = std::fabs(got - want);
      g_max_grad_norm_error = std::max(g_max_grad_norm_error, diff);
      if (!Within(got, want, kGradNormAtol, kGradNormRtol)) {
        Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", record.c_str(), got,
                    want, diff));
      }
    }

    for (const ParamView& view : model->params()) {
      CheckParamL2(fixture, step, view, L2Norm(view.value, view.count));
      CheckGradL2(fixture, step, view, L2Norm(view.grad, view.count));
    }

    if (g_stop_on_first_failure && g_failures > failures_before) {
      std::printf("train_parity: first divergence at step %d\n", step);
      break;
    }
  }

  // Optional exact final-parameter comparison.
  if (fixture.Has(std::string("final/param/") + model->params().front().name)) {
    for (const ParamView& view : model->params()) {
      const std::string record = std::string("final/param/") + view.name;
      const Tensor& want = fixture.Get(record);
      if (want.numel() != view.count) {
        Fail(record + ": count mismatch");
        continue;
      }
      const std::vector<float> got = ReadCompute(view.value, view.count);
      double max_diff = 0.0;
      for (std::int64_t i = 0; i < view.count; ++i) {
        max_diff = std::max(
            max_diff,
            std::fabs(static_cast<double>(got[static_cast<std::size_t>(i)]) -
                      static_cast<double>(want.f32()[i])));
      }
      g_max_final_param_error = std::max(g_max_final_param_error, max_diff);
      if (max_diff > 1e-3) {
        Fail(Format("%s: max |diff| = %.3g", record.c_str(), max_diff));
      }
    }
  }
}

std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/train_parity.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/train_parity.bin");
  }
  candidates.emplace_back("tests/data/train_parity.bin");
  candidates.emplace_back("../tests/data/train_parity.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "training-parity fixture not found; pass it as argv[1]");
}

}  // namespace

int main(int argc, char** argv) {
  g_stop_on_first_failure = std::getenv("TRAIN_PARITY_STOP") != nullptr;
  try {
    Run(LocateFixture(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_parity: %s\n", error.what());
    return 1;
  }
  std::printf(
      "train_parity: max errors (loss %.3g, param_l2 %.3g, grad_l2 %.3g, "
      "grad_norm %.3g, final_param %.3g)%s\n",
      g_max_loss_error, g_max_param_l2_error, g_max_grad_l2_error,
      g_max_grad_norm_error, g_max_final_param_error,
      g_failures == 0 ? "" : " [FAILED]");
  return g_failures == 0 ? 0 : 1;
}
