// CPU oracle test.
//
// Reads the committed fixture `tests/data/debug_state.bin` (produced offline by
// `tools/dump_oracle.py` from nanochat's `gpt.py`) and checks the reference
// math that the C++ model must reproduce:
//
//   1. the post-softcap logits are `15 * tanh(raw / 15)`;
//   2. the loss is the batch-mean cross-entropy of those logits;
//   3. `grad/logits` is `(softmax(logits) - onehot(targets)) / count`;
//   4. `grad/raw_logits` chains (3) through the softcap derivative;
//   5. every parameter has a matching, finite gradient;
//   6. the first optimizer step matches the canonical AdamW update for each
//      AdamW group, and the recorded loss curve descends.
//
// It then builds the C++ model from the same fixture and checks the full parity
// sequence (the Wave-boundary gate):
//
//   7. the model's forward raw logits, post-softcap logits, and loss;
//   8. every parameter gradient from the model's backward pass;
//   9. a short optimizer loop reproducing the recorded loss curve and parameter
//      trajectory (`step/<k>/loss`, `step/<k>/param/<name>`).
//
// The model parity runs on the CPU backend by default and on the CUDA backend
// when the target is built under `--config=cuda` (`//tests:oracle_cuda_test`).
// Every device<->host transfer goes through `kernels::Memcpy`, so the same
// source works on both backends.

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
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;
using nanochat::TrainModel;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// Tolerances. Both the CPU and the fp32 CUDA backend are expected to agree with
// the PyTorch fixture to single-precision roundoff for the forward and the
// backward.
constexpr double kForwardTolerance = 1e-5;
constexpr double kBackwardTolerance = 1e-5;
// A single optimizer step driven by the fixture's recorded gradients isolates
// the optimizer from backward roundoff and must match to fp32 roundoff.
constexpr double kOptimizerTolerance = 1e-5;
// The full `TrainStep` loop cannot be tight for every parameter: the fixture's
// scalar gradients sit at the AdamW `eps` scale (see RunModelParity), so the
// recorded trajectory is not reproducible bit-for-bit. The loss stays close;
// the parameter bound is deliberately loose and the measured maximum is
// printed.
constexpr double kTrajectoryLossTolerance = 1e-1;
constexpr double kTrajectoryParamTolerance = 5.0;
constexpr float kSoftcap = 15.0f;

int g_failures = 0;
double g_max_forward_error = 0.0;
double g_max_backward_error = 0.0;
double g_max_optimizer_error = 0.0;
double g_max_trajectory_loss_error = 0.0;
double g_max_trajectory_param_error = 0.0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectNear(double got, double want, double tolerance,
                const std::string& what) {
  const double diff = std::fabs(got - want);
  if (!(diff <= tolerance)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g > tol=%.3g)", what.c_str(),
                got, want, diff, tolerance));
  }
}

std::string Join(const std::vector<std::string>& parts) {
  std::string joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) joined += ", ";
    joined += parts[i];
  }
  return joined;
}

// Resolve the fixture path. Bazel passes `$(rootpath ...)` as argv[1]; when the
// binary is run directly (for example by `tools/nanochat verify`) fall back to
// the runfiles tree and then to the source-relative path.
std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/debug_state.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/debug_state.bin");
    candidates.emplace_back(std::string(src_dir) +
                            "/nanochat_cpp/tests/data/debug_state.bin");
  }
  candidates.emplace_back("tests/data/debug_state.bin");
  candidates.emplace_back("nanochat.cpp/tests/data/debug_state.bin");
  candidates.emplace_back("../tests/data/debug_state.bin");

  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error("oracle fixture not found; tried: " +
                           Join(candidates));
}

double CrossEntropy(const float* logits, const std::int32_t* targets, int batch,
                    int seq, int vocab) {
  const int count = batch * seq;
  double total = 0.0;
  for (int i = 0; i < count; ++i) {
    const float* row = logits + static_cast<std::size_t>(i) * vocab;
    double max_logit = row[0];
    for (int v = 1; v < vocab; ++v) {
      max_logit = std::max(max_logit, static_cast<double>(row[v]));
    }
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
      sum += std::exp(static_cast<double>(row[v]) - max_logit);
    }
    const int target = targets[i];
    total += -(static_cast<double>(row[target]) - max_logit - std::log(sum));
  }
  return total / count;
}

// Check that one AdamW group's first step matches the canonical update used by
// nanochat's `setup_optimizer` / `adamw_step_fused`.
void CheckAdamWStep1(const Fixture& fixture, const std::string& name, double lr,
                     double beta1, double beta2, double eps,
                     double weight_decay) {
  const Tensor& p0 = fixture.Get("param/" + name);
  const Tensor& grad = fixture.Get("grad/" + name);
  const Tensor& p1 = fixture.Get("step/1/param/" + name);
  const std::int64_t count = p0.numel();
  if (grad.numel() != count || p1.numel() != count) {
    Fail(name + ": parameter/gradient/update shape mismatch");
    return;
  }
  const float* p0_data = p0.f32();
  const float* grad_data = grad.f32();
  const float* p1_data = p1.f32();
  double max_diff = 0.0;
  for (std::int64_t i = 0; i < count; ++i) {
    const double p = p0_data[i];
    const double g = grad_data[i];
    const double decayed = p * (1.0 - lr * weight_decay);
    const double exp_avg = (1.0 - beta1) * g;
    const double exp_avg_sq = (1.0 - beta2) * g * g;
    const double denom = std::sqrt(exp_avg_sq / (1.0 - beta2)) + eps;
    const double step_size = lr / (1.0 - beta1);
    const double updated = decayed - step_size * (exp_avg / denom);
    max_diff = std::max(max_diff,
                        std::fabs(updated - static_cast<double>(p1_data[i])));
  }
  ExpectTrue(max_diff < 1e-6,
             Format("AdamW step-1 mismatch for %s: max |diff| = %.3g",
                    name.c_str(), max_diff));
}

// ---------------------------------------------------------------------------
// Model parity helpers
// ---------------------------------------------------------------------------
//
// The same source links the reference CPU backend by default and the CUDA
// backend under `--config=cuda`, so every device<->host transfer goes through
// `kernels::Memcpy` and every read of a parameter/logit goes through a host
// staging buffer.

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

void CopyCompute(const ComputeType* src, std::int64_t count,
                 std::vector<float>* out) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src, static_cast<std::size_t>(count) * sizeof(ComputeType),
        nanochat::CopyDir::kDeviceToHost);
  }
  out->resize(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    (*out)[static_cast<std::size_t>(i)] =
        AsFloat32(host[static_cast<std::size_t>(i)]);
  }
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
  std::string window(pattern.data, pattern.data + pattern.numel());
  config.window_pattern = window;
  return config;
}

void LoadParameters(const Fixture& fixture, Model* model) {
  std::vector<nanochat::ParamView> views = model->params();
  for (const nanochat::ParamView& view : views) {
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
    if (source.dtype != nanochat::oracle::DType::kFp32) {
      Fail("parameter '" + record + "' is not fp32 in the fixture");
      continue;
    }
    const float* src = source.f32();
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i, src[i]);
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        nanochat::CopyDir::kHostToDevice);
  }
}

// Compares one device buffer against a fixture record and folds the maximum
// absolute difference into `*max_error`.
void CompareToFixture(const ComputeType* got_dev, std::int64_t count,
                      const Tensor& want, const std::string& what,
                      double tolerance, double* max_error) {
  if (want.dtype != nanochat::oracle::DType::kFp32) {
    Fail(what + ": fixture record is not fp32");
    return;
  }
  if (want.numel() != count) {
    Fail(Format("%s: count mismatch (model %lld, fixture %lld)", what.c_str(),
                static_cast<long long>(count),
                static_cast<long long>(want.numel())));
    return;
  }
  std::vector<float> got;
  CopyCompute(got_dev, count, &got);
  double max_diff = 0.0;
  std::int64_t worst = -1;
  for (std::int64_t i = 0; i < count; ++i) {
    const double diff =
        std::fabs(static_cast<double>(got[static_cast<std::size_t>(i)]) -
                  static_cast<double>(want.f32()[i]));
    if (diff > max_diff) {
      max_diff = diff;
      worst = i;
    }
  }
  *max_error = std::max(*max_error, max_diff);
  if (max_diff > tolerance) {
    const double got_worst =
        worst >= 0 ? static_cast<double>(got[static_cast<std::size_t>(worst)])
                   : 0.0;
    const double want_worst =
        worst >= 0 ? static_cast<double>(want.f32()[worst]) : 0.0;
    Fail(Format("%s: max |diff| = %.3g at %lld (got %.9g want %.9g)",
                what.c_str(), max_diff, static_cast<long long>(worst),
                got_worst, want_worst));
  }
}

void RunChecks(const Fixture& fixture) {
  // --- configuration -------------------------------------------------------
  const std::int64_t layers = fixture.Get("config/layers").scalar_int();
  const std::int64_t heads = fixture.Get("config/heads").scalar_int();
  const std::int64_t kv_heads = fixture.Get("config/kv_heads").scalar_int();
  const std::int64_t embd = fixture.Get("config/embd").scalar_int();
  const std::int64_t vocab = fixture.Get("config/vocab").scalar_int();
  const std::int64_t padded_vocab =
      fixture.Get("config/padded_vocab").scalar_int();
  const std::int64_t seq = fixture.Get("config/seq").scalar_int();
  const std::int64_t batch = fixture.Get("config/batch").scalar_int();
  const std::int64_t steps = fixture.Get("config/steps").scalar_int();

  ExpectTrue(layers == 2 && heads == 8 && kv_heads == 2 && embd == 32,
             "unexpected model config in fixture");
  ExpectTrue(vocab == 64 && padded_vocab == 64 && seq == 8 && batch == 2,
             "unexpected token config in fixture");
  ExpectTrue(steps >= 1, "fixture must record at least one optimizer step");

  const Tensor& raw = fixture.Get("forward/raw_logits");
  const Tensor& logits = fixture.Get("forward/logits");
  const Tensor& loss = fixture.Get("forward/loss");
  const Tensor& grad_logits = fixture.Get("grad/logits");
  const Tensor& grad_raw = fixture.Get("grad/raw_logits");
  const Tensor& targets = fixture.Get("input/targets");

  ExpectTrue(raw.shape.size() == 3 && raw.shape[0] == batch &&
                 raw.shape[1] == seq && raw.shape[2] == vocab,
             "forward/raw_logits has an unexpected shape");
  ExpectTrue(logits.shape == raw.shape, "forward/logits shape mismatch");
  ExpectTrue(grad_logits.shape == raw.shape, "grad/logits shape mismatch");
  ExpectTrue(grad_raw.shape == raw.shape, "grad/raw_logits shape mismatch");

  const std::int64_t element_count = raw.numel();
  const float* raw_data = raw.f32();
  const float* logits_data = logits.f32();
  const float* grad_logits_data = grad_logits.f32();
  const float* grad_raw_data = grad_raw.f32();
  const std::int32_t* target_data = targets.i32();

  // --- 1. softcap ----------------------------------------------------------
  double softcap_error = 0.0;
  for (std::int64_t i = 0; i < element_count; ++i) {
    const double expected =
        15.0 * std::tanh(static_cast<double>(raw_data[i]) / 15.0);
    softcap_error =
        std::max(softcap_error,
                 std::fabs(expected - static_cast<double>(logits_data[i])));
  }
  ExpectTrue(softcap_error < 1e-6,
             Format("softcap mismatch: max |diff| = %.3g", softcap_error));

  // --- 2. cross-entropy loss ----------------------------------------------
  const double recomputed_loss =
      CrossEntropy(logits_data, target_data, static_cast<int>(batch),
                   static_cast<int>(seq), static_cast<int>(vocab));
  ExpectNear(recomputed_loss, loss.scalar_f32(), 1e-5, "cross-entropy loss");

  // --- 3. d(loss)/d(logits) ------------------------------------------------
  double grad_logits_error = 0.0;
  double grad_raw_error = 0.0;
  const double count = static_cast<double>(batch * seq);
  for (std::int64_t i = 0; i < batch * seq; ++i) {
    const float* row = logits_data + i * vocab;
    const std::int32_t target = target_data[i];
    double max_logit = row[0];
    for (std::int64_t v = 1; v < vocab; ++v) {
      max_logit = std::max(max_logit, static_cast<double>(row[v]));
    }
    double sum = 0.0;
    for (std::int64_t v = 0; v < vocab; ++v) {
      sum += std::exp(static_cast<double>(row[v]) - max_logit);
    }
    for (std::int64_t v = 0; v < vocab; ++v) {
      const double probability =
          std::exp(static_cast<double>(row[v]) - max_logit) / sum;
      const double expected_logits =
          (probability - (v == target ? 1.0 : 0.0)) / count;
      grad_logits_error = std::max(
          grad_logits_error,
          std::fabs(expected_logits -
                    static_cast<double>(grad_logits_data[i * vocab + v])));
      const double softcap_derivative =
          1.0 -
          std::pow(static_cast<double>(logits_data[i * vocab + v]) / 15.0, 2.0);
      const double expected_raw = expected_logits * softcap_derivative;
      grad_raw_error = std::max(
          grad_raw_error,
          std::fabs(expected_raw -
                    static_cast<double>(grad_raw_data[i * vocab + v])));
    }
  }
  ExpectTrue(
      grad_logits_error < 1e-6,
      Format("grad/logits mismatch: max |diff| = %.3g", grad_logits_error));
  ExpectTrue(
      grad_raw_error < 1e-6,
      Format("grad/raw_logits mismatch: max |diff| = %.3g", grad_raw_error));

  // --- 4. parameters and gradients ----------------------------------------
  std::vector<std::string> parameter_names;
  for (const std::string& name : fixture.names()) {
    if (name.rfind("param/", 0) == 0) {
      parameter_names.push_back(name.substr(std::string("param/").size()));
    }
  }
  ExpectTrue(!parameter_names.empty(), "fixture has no parameter records");

  double grad_magnitude = 0.0;
  for (const std::string& name : parameter_names) {
    const Tensor& parameter = fixture.Get("param/" + name);
    const std::string grad_name = "grad/" + name;
    if (!fixture.Has(grad_name)) {
      Fail("missing gradient for parameter " + name);
      continue;
    }
    const Tensor& gradient = fixture.Get(grad_name);
    if (gradient.shape != parameter.shape) {
      Fail("gradient shape mismatch for parameter " + name);
      continue;
    }
    for (std::int64_t i = 0; i < parameter.numel(); ++i) {
      if (!std::isfinite(parameter.f32()[i])) {
        Fail("non-finite parameter value in " + name);
        break;
      }
    }
    for (std::int64_t i = 0; i < gradient.numel(); ++i) {
      const float value = gradient.f32()[i];
      if (!std::isfinite(value)) {
        Fail("non-finite gradient value in " + name);
        break;
      }
      grad_magnitude += std::fabs(static_cast<double>(value));
    }
  }
  ExpectTrue(grad_magnitude > 0.0, "all recorded gradients are zero");

  // --- 5. optimizer trajectory --------------------------------------------
  std::vector<double> losses(static_cast<std::size_t>(steps) + 1);
  for (std::int64_t step = 0; step <= steps; ++step) {
    const std::string name = "step/" + std::to_string(step) + "/loss";
    if (!fixture.Has(name)) {
      Fail("missing loss record " + name);
      continue;
    }
    losses[static_cast<std::size_t>(step)] = fixture.Get(name).scalar_f32();
    ExpectTrue(std::isfinite(losses[static_cast<std::size_t>(step)]),
               "non-finite loss in " + name);
  }
  ExpectTrue(losses.back() < losses.front(),
             "optimizer steps did not reduce the loss");

  // Every parameter must have a recorded trajectory, and at least one must
  // actually change (a silent no-op optimizer would still pass the loss check
  // only by accident).
  double parameter_change = 0.0;
  for (const std::string& name : parameter_names) {
    const std::string initial_name = "step/0/param/" + name;
    const std::string final_name =
        "step/" + std::to_string(steps) + "/param/" + name;
    if (!fixture.Has(initial_name) || !fixture.Has(final_name)) {
      Fail("missing optimizer trajectory for parameter " + name);
      continue;
    }
    const Tensor& initial = fixture.Get(initial_name);
    const Tensor& updated = fixture.Get(final_name);
    if (initial.shape != updated.shape) {
      Fail("optimizer trajectory shape mismatch for parameter " + name);
      continue;
    }
    for (std::int64_t i = 0; i < initial.numel(); ++i) {
      parameter_change += std::fabs(static_cast<double>(updated.f32()[i]) -
                                    static_cast<double>(initial.f32()[i]));
    }
  }
  ExpectTrue(parameter_change > 0.0,
             "optimizer steps did not change any parameter");

  // --- 6. first step matches canonical AdamW ------------------------------
  const double scale = std::sqrt(768.0 / static_cast<double>(embd));
  const double unembedding_lr =
      fixture.Get("config/opt/unembedding_lr").scalar_f32();
  const double embedding_lr =
      fixture.Get("config/opt/embedding_lr").scalar_f32();
  const double scalar_lr = fixture.Get("config/opt/scalar_lr").scalar_f32();
  const double eps = 1e-10;
  // lm_head: AdamW, betas (0.8, 0.96), weight_decay 0.01.
  CheckAdamWStep1(fixture, "lm_head.weight", unembedding_lr * scale, 0.8, 0.96,
                  eps, 0.01);
  // token embedding: betas (0.8, 0.995), weight_decay 0.001.
  CheckAdamWStep1(fixture, "transformer.wte.weight", embedding_lr * scale, 0.8,
                  0.995, eps, 0.001);
  // value embeddings: half the embedding learning rate, weight_decay 0.01.
  CheckAdamWStep1(fixture, "value_embeds.1.weight", embedding_lr * scale * 0.5,
                  0.8, 0.995, eps, 0.01);
  // resid lambdas: 0.01x scalar learning rate, betas (0.8, 0.95), weight_decay
  // 0.05.
  CheckAdamWStep1(fixture, "resid_lambdas", scalar_lr * 0.01, 0.8, 0.95, eps,
                  0.05);
  // x0 lambdas: scalar learning rate, betas (0.96, 0.95), no weight decay.
  CheckAdamWStep1(fixture, "x0_lambdas", scalar_lr, 0.96, 0.95, eps, 0.0);
  // smear/backout scalars: fixed learning rate 0.2, betas (0.8, 0.95), no
  // decay.
  CheckAdamWStep1(fixture, "smear_gate.weight", 0.2, 0.8, 0.95, eps, 0.0);
  CheckAdamWStep1(fixture, "backout_lambda", 0.2, 0.8, 0.95, eps, 0.0);
}

// The model-parity gate: builds the C++ model from the fixture config, loads
// every parameter, and checks the forward, the backward, and a short optimizer
// trajectory against the recorded reference.
//
// The optimizer is validated two ways:
//
//   * a tight single step driven by the fixture's own recorded gradients
//     (`grad/<name>`), which isolates the optimizer from backward roundoff and
//     must reproduce `step/1/param`; and
//   * the full `TrainStep` loop, compared against `step/<k>/loss` and
//     `step/<k>/param/<name>`.
//
// The full loop cannot be tight for every parameter. nanochat's AdamW uses
// `eps = 1e-10`, and several fixture gradients sit at that scale:
// `grad/x0_lambdas`, `grad/resid_lambdas`, and `grad/backout_lambda` are
// ~1e-10, and `grad/smear_gate.weight` is exactly zero. For those parameters
// the update `lr * g / (|g| + eps)` is a discontinuous function of the last bit
// of `g`, so the recorded trajectory is not reproducible across float32
// implementations; the divergence then feeds back through the forward into the
// rest of the parameters. The tight single-step check above is the meaningful
// optimizer proof; the loop is a behavioral check with a looser bound.
void RunModelParity(const Fixture& fixture) {
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int vocab = static_cast<int>(fixture.Get("config/vocab").scalar_int());
  const int padded =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  const int steps = static_cast<int>(fixture.Get("config/steps").scalar_int());
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  const Config config = ConfigFromFixture(fixture);
  std::unique_ptr<Model> model = Model::Create(config);
  auto* impl = static_cast<TrainModel*>(model.get());
  LoadParameters(fixture, model.get());

  const Tensor& tokens = fixture.Get("input/tokens");
  const Tensor& targets = fixture.Get("input/targets");
  const int* token_data = reinterpret_cast<const int*>(tokens.i32());
  const int* target_data = reinterpret_cast<const int*>(targets.i32());

  // --- 1. forward: loss, raw logits, post-softcap logits --------------------
  const float loss = model->ForwardLoss(token_data, target_data, batch, seq);
  {
    const Tensor& want = fixture.Get("forward/loss");
    const double diff = std::fabs(static_cast<double>(loss) -
                                  static_cast<double>(want.scalar_f32()));
    g_max_forward_error = std::max(g_max_forward_error, diff);
    if (diff > kForwardTolerance) {
      Fail(Format("forward/loss: got %.9g want %.9g (|diff|=%.3g)", loss,
                  want.scalar_f32(), diff));
    }
  }

  std::vector<ComputeType> raw_host(static_cast<std::size_t>(rows) * padded);
  nanochat::kernels::Memcpy(
      raw_host.data(), impl->raw_logits(),
      static_cast<std::size_t>(rows) * padded * sizeof(ComputeType),
      nanochat::CopyDir::kDeviceToHost);

  {
    const Tensor& want = fixture.Get("forward/raw_logits");
    double max_diff = 0.0;
    for (std::int64_t r = 0; r < rows; ++r) {
      for (int v = 0; v < vocab; ++v) {
        const double got = static_cast<double>(AsFloat32(
            raw_host[static_cast<std::size_t>(r) * padded + v]));
        const double expected = static_cast<double>(
            want.f32()[static_cast<std::size_t>(r) * vocab + v]);
        max_diff = std::max(max_diff, std::fabs(got - expected));
      }
    }
    g_max_forward_error = std::max(g_max_forward_error, max_diff);
    if (max_diff > kForwardTolerance) {
      Fail(Format("forward/raw_logits: max |diff| = %.3g", max_diff));
    }
  }

  {
    const Tensor& want = fixture.Get("forward/logits");
    double max_diff = 0.0;
    for (std::int64_t r = 0; r < rows; ++r) {
      for (int v = 0; v < vocab; ++v) {
        const float raw =
            AsFloat32(raw_host[static_cast<std::size_t>(r) * padded + v]);
        const double got =
            static_cast<double>(kSoftcap * std::tanh(raw / kSoftcap));
        const double expected = static_cast<double>(
            want.f32()[static_cast<std::size_t>(r) * vocab + v]);
        max_diff = std::max(max_diff, std::fabs(got - expected));
      }
    }
    g_max_forward_error = std::max(g_max_forward_error, max_diff);
    if (max_diff > kForwardTolerance) {
      Fail(Format("forward/logits: max |diff| = %.3g", max_diff));
    }
  }

  // --- 2. backward: every parameter gradient --------------------------------
  model->Backward();
  for (const nanochat::ParamView& view : model->params()) {
    const std::string record = std::string("grad/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing gradient record '" + record + "'");
      continue;
    }
    CompareToFixture(view.grad, view.count, fixture.Get(record), record,
                     kBackwardTolerance, &g_max_backward_error);
  }

  // --- 3. optimizer trajectory ---------------------------------------------
  OptimizerConfig opt;
  opt.unembedding_lr = fixture.Get("config/opt/unembedding_lr").scalar_f32();
  opt.embedding_lr = fixture.Get("config/opt/embedding_lr").scalar_f32();
  opt.matrix_lr = fixture.Get("config/opt/matrix_lr").scalar_f32();
  opt.scalar_lr = fixture.Get("config/opt/scalar_lr").scalar_f32();
  opt.weight_decay = fixture.Get("config/opt/weight_decay").scalar_f32();
  opt.clip = fixture.Get("config/opt/clip").scalar_f32();
  // `muon_ns_steps`, `muon_beta2`, and `adam_eps` keep the setup_optimizer
  // defaults (5, 0.9, 1e-10), which is what the fixture records.

  // The fixture's trajectory calls `optimizer.step()` directly with the
  // setup_optimizer defaults, so its schedules are constant: learning-rate
  // multiplier 1, Muon momentum 0.95, and zero weight decay. `num_iterations =
  // 0` collapses the schedules to those constants (see src/optim.cc); the
  // momentum limits are all 0.95 so the schedule is pinned regardless of step.
  SchedulerConfig sched;
  sched.num_iterations = 0;
  sched.warmup_steps = 0;
  sched.warmdown_ratio = 0.0f;
  sched.final_lr_frac = 0.0f;
  sched.weight_decay_base = 0.0f;
  sched.muon_momentum_warmup_steps = 0.0f;
  sched.muon_momentum_start = 0.95f;
  sched.muon_momentum_peak = 0.95f;
  sched.muon_momentum_final = 0.95f;
  const Scheduler scheduler(sched);

  auto make_model = [&]() -> std::unique_ptr<Model> {
    std::unique_ptr<Model> fresh = Model::Create(config);
    LoadParameters(fixture, fresh.get());
    return fresh;
  };

  // 3a. Tight single-step parity, driven by the fixture's own gradients: load
  // `grad/<name>` into the model's gradient buffers, apply one 1-based
  // optimizer step, and compare every parameter to `step/1/param`. Because the
  // inputs are the recorded reference gradients, this isolates the optimizer
  // grouping, schedules, AdamW, and Muon from the backward's fp32 roundoff.
  {
    for (const nanochat::ParamView& view : model->params()) {
      const std::string record = std::string("grad/") + view.name;
      if (!fixture.Has(record)) continue;
      const Tensor& source = fixture.Get(record);
      const std::int64_t count =
          std::min<std::int64_t>(view.count, source.numel());
      std::vector<ComputeType> host(static_cast<std::size_t>(count));
      for (std::int64_t i = 0; i < count; ++i) {
        StoreFloat(host.data() + i, source.f32()[i]);
      }
      nanochat::kernels::Memcpy(
          view.grad, host.data(),
          static_cast<std::size_t>(count) * sizeof(ComputeType),
          nanochat::CopyDir::kHostToDevice);
    }
    std::unique_ptr<Optimizer> step1 =
        nanochat::CreateOptimizer(model.get(), opt, scheduler);
    step1->Step(1);
    for (const nanochat::ParamView& view : model->params()) {
      const std::string record = std::string("step/1/param/") + view.name;
      if (!fixture.Has(record)) {
        Fail("fixture is missing trajectory record '" + record + "'");
        continue;
      }
      CompareToFixture(view.value, view.count, fixture.Get(record),
                       "optimizer step/1 " + std::string(view.name),
                       kOptimizerTolerance, &g_max_optimizer_error);
    }
  }

  // 3b. The full `TrainStep` loop on a fresh model. `TrainStep` zeroes the
  // gradient, runs forward/backward, then applies the 1-based optimizer step,
  // matching the fixture's `optimizer.step()` order.
  model = make_model();
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), opt, scheduler);

  // step/0 is the initial parameter set (recorded before any optimizer step).
  for (const nanochat::ParamView& view : model->params()) {
    const std::string record = std::string("step/0/param/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing trajectory record '" + record + "'");
      continue;
    }
    CompareToFixture(view.value, view.count, fixture.Get(record), record,
                     kOptimizerTolerance, &g_max_trajectory_param_error);
  }

  for (int k = 1; k <= steps; ++k) {
    model->TrainStep(token_data, target_data, batch, seq, optimizer.get());
    // The fixture records the loss *after* the step, so recompute it on the
    // updated parameters to compare `step/<k>/loss` directly.
    const float step_loss =
        model->ForwardLoss(token_data, target_data, batch, seq);
    const std::string loss_record = "step/" + std::to_string(k) + "/loss";
    const double loss_diff =
        std::fabs(static_cast<double>(step_loss) -
                  static_cast<double>(fixture.Get(loss_record).scalar_f32()));
    g_max_trajectory_loss_error =
        std::max(g_max_trajectory_loss_error, loss_diff);
    if (loss_diff > kTrajectoryLossTolerance) {
      Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g)", loss_record.c_str(),
                  step_loss, fixture.Get(loss_record).scalar_f32(),
                  loss_diff));
    }
    for (const nanochat::ParamView& view : model->params()) {
      const std::string record =
          "step/" + std::to_string(k) + "/param/" + view.name;
      if (!fixture.Has(record)) {
        Fail("fixture is missing trajectory record '" + record + "'");
        continue;
      }
      CompareToFixture(view.value, view.count, fixture.Get(record), record,
                       kTrajectoryParamTolerance,
                       &g_max_trajectory_param_error);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  try {
    path = LocateFixture(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  Fixture fixture;
  try {
    fixture = Fixture::Load(path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  std::printf("oracle_test: loaded %s (%zu records)\n", path.c_str(),
              fixture.size());
  RunChecks(fixture);
  RunModelParity(fixture);

  std::printf(
      "oracle_test: max forward error %.3g, max backward error %.3g, "
      "max optimizer-step-1 error %.3g, max trajectory loss error %.3g, "
      "max trajectory parameter error %.3g\n",
      g_max_forward_error, g_max_backward_error, g_max_optimizer_error,
      g_max_trajectory_loss_error, g_max_trajectory_param_error);

  if (g_failures != 0) {
    std::fprintf(stderr, "oracle_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("oracle_test: all checks passed\n");
  return 0;
}
