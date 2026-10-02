// src/optim_test.cc — the optimizer grouping, schedules, and update rules.
//
// The test builds a tiny model and verifies, in order:
//   1. the Scheduler values (learning-rate multiplier, Muon momentum, and
//      cosine weight decay) against hand-computed numbers;
//   2. the setup_optimizer grouping (which parameter belongs to AdamW and
//      which to Muon) and every group's nominal learning rate;
//   3. a few synthetic-gradient steps against the canonical AdamW update and
//      against the Muon kernel driven one parameter at a time, including the
//      global gradient-norm clip;
//   4. an independent closed form for a zero gradient, where both optimizers
//      reduce to `p <- p * (1 - lr * weight_decay)`.
//
// The schedules are checked with `step - 1` as nanochat's 0-based loop counter,
// because Optimizer::Step is 1-based.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"

namespace nanochat {
// Declared (not frozen) by src/optim.cc. Reports kind 0 = AdamW, 1 = Muon, the
// matrix extents, and the nominal (pre-schedule) learning rate.
bool OptimizerParamGroupForTest(const Optimizer* optimizer, const char* name,
                                int* kind, int* rows, int* cols, float* lr);
}  // namespace nanochat

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::Model;
using nanochat::Optimizer;
using nanochat::OptimizerConfig;
using nanochat::ParamView;
using nanochat::Scheduler;
using nanochat::SchedulerConfig;

int g_failures = 0;

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

void ExpectNear(float actual, float expected, float tolerance,
                const std::string& what) {
  if (std::fabs(actual - expected) > tolerance) {
    Fail(Format("%s: got %.8g want %.8g (tol %.3g)", what.c_str(), actual,
                expected, tolerance));
  }
}

template <typename T = ComputeType>
float AsF32(T value) {
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

struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed) {}
  std::uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  float Uniform() {
    return static_cast<float>((Next() >> 11) * (1.0 / 9007199254740992.0)) *
               2.0f -
           1.0f;
  }
};

Config MakeConfig() {
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

OptimizerConfig MakeOptimizerConfig() {
  OptimizerConfig config;
  // The nanochat defaults, reproduced explicitly.
  config.unembedding_lr = 0.004f;
  config.embedding_lr = 0.2f;
  config.matrix_lr = 0.02f;
  config.scalar_lr = 0.5f;
  config.weight_decay = 0.0f;
  config.muon_ns_steps = 5;
  config.muon_beta2 = 0.9f;
  config.adam_eps = 1e-10f;
  config.clip = 1e9f;  // disabled unless a case overrides it
  return config;
}

SchedulerConfig MakeSchedulerConfig() {
  SchedulerConfig config;
  config.num_iterations = 10;
  config.warmup_steps = 2;
  config.warmdown_ratio = 0.5f;
  config.final_lr_frac = 0.1f;
  config.weight_decay_base = 0.02f;
  config.muon_momentum_warmup_steps = 4.0f;
  config.muon_momentum_start = 0.85f;
  config.muon_momentum_peak = 0.97f;
  config.muon_momentum_final = 0.90f;
  return config;
}

// ---------------------------------------------------------------------------
// 1. Schedules
// ---------------------------------------------------------------------------

void TestSchedules() {
  const Scheduler scheduler(MakeSchedulerConfig());

  const struct {
    int step;
    float lr;
  } kLr[] = {
      {1, 0.5f}, {2, 1.0f}, {3, 1.0f}, {6, 1.0f}, {7, 0.82f}, {11, 0.1f},
  };
  for (const auto& tc : kLr) {
    ExpectNear(scheduler.LrMultiplier(tc.step), tc.lr, 1e-6f,
               Format("LrMultiplier(%d)", tc.step));
  }

  const struct {
    int step;
    float momentum;
  } kMomentum[] = {
      {1, 0.85f}, {2, 0.88f}, {3, 0.91f}, {4, 0.94f},
      {5, 0.97f}, {7, 0.956f}, {11, 0.90f},
  };
  for (const auto& tc : kMomentum) {
    ExpectNear(scheduler.MuonMomentum(tc.step), tc.momentum, 1e-6f,
               Format("MuonMomentum(%d)", tc.step));
  }

  const struct {
    int step;
    float weight_decay;
  } kWeightDecay[] = {
      {1, 0.02f}, {6, 0.01f}, {11, 0.0f},
  };
  for (const auto& tc : kWeightDecay) {
    ExpectNear(scheduler.WeightDecay(tc.step), tc.weight_decay, 1e-6f,
               Format("WeightDecay(%d)", tc.step));
  }
  std::printf("optim_test: schedules ok\n");
}

// ---------------------------------------------------------------------------
// 2/3. Grouping and a few update steps
// ---------------------------------------------------------------------------

// The expected group for a parameter: kind, nominal learning rate, and (for
// AdamW) the betas and weight decay from `setup_optimizer`.
struct GroupSpec {
  int kind = 0;  // 0 = AdamW, 1 = Muon
  float lr = 0.0f;
  float beta1 = 0.0f;
  float beta2 = 0.0f;
  float weight_decay = 0.0f;
};

[[maybe_unused]] GroupSpec SpecFor(const std::string& name,
                                  const Config& config,
                                  const OptimizerConfig& oc, int rows,
                                  int cols) {
  const float scale =
      std::sqrt(768.0f / static_cast<float>(config.hidden_dim));
  if (name == "lm_head.weight") {
    return {0, oc.unembedding_lr * scale, 0.8f, 0.96f, 0.01f};
  }
  if (name == "transformer.wte.weight") {
    return {0, oc.embedding_lr * scale, 0.8f, 0.995f, 0.001f};
  }
  if (name.rfind("value_embeds.", 0) == 0) {
    return {0, oc.embedding_lr * scale * 0.5f, 0.8f, 0.995f, 0.01f};
  }
  if (name == "resid_lambdas") {
    return {0, oc.scalar_lr * 0.01f, 0.8f, 0.95f, 0.05f};
  }
  if (name == "x0_lambdas") {
    return {0, oc.scalar_lr, 0.96f, 0.95f, 0.0f};
  }
  if (name == "smear_gate.weight" || name == "smear_lambda" ||
      name == "backout_lambda") {
    // The 1 x 24 smear gate is deliberately an AdamW parameter.
    return {0, 0.2f, 0.8f, 0.95f, 0.0f};
  }
  const float lr =
      oc.matrix_lr *
      std::sqrt(std::max(1.0f, static_cast<float>(rows) /
                                   static_cast<float>(cols)));
  return {1, lr, 0.0f, 0.0f, 0.0f};
}

struct ParamState {
  const ParamView* view = nullptr;
  GroupSpec spec;
  int rows = 0;
  int cols = 0;
  std::vector<ComputeType> grad;
  std::vector<ComputeType> ref_value;  // expected current parameter
  std::vector<float> m;                // AdamW first moment
  std::vector<float> v;                // AdamW second moment
  std::vector<float> buf1;             // Muon momentum
  std::vector<float> buf2;             // Muon factored second moment
};

[[maybe_unused]] std::vector<ParamState> MakeStates(
    const std::vector<ParamView>& views, const Config& config,
    const OptimizerConfig& oc) {
  std::vector<ParamState> states;
  states.reserve(views.size());
  for (const ParamView& view : views) {
    ParamState state;
    state.view = &view;
    state.rows = view.rows;
    state.cols = view.cols;
    state.spec =
        SpecFor(std::string(view.name), config, oc, view.rows, view.cols);
    const std::size_t count = static_cast<std::size_t>(view.count);
    state.grad.assign(count, ComputeType{});
    state.ref_value.resize(count);
    for (std::size_t i = 0; i < count; ++i) state.ref_value[i] = view.value[i];
    if (state.spec.kind == 0) {
      state.m.assign(count, 0.0f);
      state.v.assign(count, 0.0f);
    } else {
      state.buf1.assign(count, 0.0f);
      const int red = view.rows >= view.cols ? view.rows : view.cols;
      state.buf2.assign(static_cast<std::size_t>(red), 0.0f);
    }
    states.push_back(std::move(state));
  }
  return states;
}

// Applies one reference step to every parameter. AdamW uses the canonical
// closed form; Muon is driven through the kernel one parameter at a time, which
// is exactly what the optimizer does for a stacked group.
[[maybe_unused]] void ApplyReference(std::vector<ParamState>& states,
                                     const Scheduler& scheduler,
                                     const OptimizerConfig& oc,
                                     float grad_scale, int step) {
  const float lrm = scheduler.LrMultiplier(step);
  for (ParamState& state : states) {
    if (state.spec.kind == 0) {
      const float lr = state.spec.lr * lrm;
      const float bias1 =
          1.0f - std::pow(state.spec.beta1, static_cast<float>(step));
      const float bias2 =
          1.0f - std::pow(state.spec.beta2, static_cast<float>(step));
      const float step_size = lr / bias1;
      for (std::size_t i = 0; i < state.grad.size(); ++i) {
        const float g = AsF32(state.grad[i]) * grad_scale;
        float p = AsF32(state.ref_value[i]);
        p *= (1.0f - lr * state.spec.weight_decay);
        state.m[i] += (1.0f - state.spec.beta1) * (g - state.m[i]);
        state.v[i] +=
            (1.0f - state.spec.beta2) * (g * g - state.v[i]);
        const float denom = std::sqrt(state.v[i] / bias2) + oc.adam_eps;
        p -= step_size * (state.m[i] / denom);
        StoreFloat(&state.ref_value[i], p);
      }
    } else {
      std::vector<ComputeType> scaled(state.grad.size());
      for (std::size_t i = 0; i < state.grad.size(); ++i) {
        StoreFloat(&scaled[i], AsF32(state.grad[i]) * grad_scale);
      }
      nanochat::MuonParams params;
      params.num_params = 1;
      params.rows = state.rows;
      params.cols = state.cols;
      params.lr = state.spec.lr * lrm;
      params.momentum = scheduler.MuonMomentum(step);
      params.beta2 = oc.muon_beta2;
      params.weight_decay = scheduler.WeightDecay(step);
      params.ns_steps = oc.muon_ns_steps;
      params.red_dim = state.rows >= state.cols ? -1 : -2;
      params.nesterov = true;
      nanochat::kernels::MuonUpdate(params, scaled.data(),
                                    state.ref_value.data(), state.buf1.data(),
                                    state.buf2.data());
    }
  }
}

void RunStepsTest(float clip) {
#if defined(NANOCHAT_PRECISION_FP16)
  (void)clip;
  std::printf("optim_test: update checks skipped (fp16 build)\n");
#else
  const Config config = MakeConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(0x5eed1234ull);

  OptimizerConfig oc = MakeOptimizerConfig();
  oc.clip = clip;
  const Scheduler scheduler(MakeSchedulerConfig());
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), oc, scheduler);

  const std::vector<ParamView> views = model->params();
  std::vector<ParamState> states = MakeStates(views, config, oc);

  // Grouping: every parameter must land in the expected group with the
  // expected nominal learning rate, and the matrix extents must survive.
  for (const ParamState& state : states) {
    int kind = -1;
    int rows = -1;
    int cols = -1;
    float lr = 0.0f;
    if (!nanochat::OptimizerParamGroupForTest(optimizer.get(), state.view->name,
                                              &kind, &rows, &cols, &lr)) {
      Fail("unclassified parameter " + std::string(state.view->name));
      continue;
    }
    if (kind != state.spec.kind || rows != state.rows || cols != state.cols) {
      Fail("group mismatch for " + std::string(state.view->name));
    }
    ExpectNear(lr, state.spec.lr, 1e-6f,
               "nominal lr " + std::string(state.view->name));
  }

  Rng rng(0x123456789abcdefull);
  constexpr float kTolerance = 2e-5f;
  for (int step = 1; step <= 3; ++step) {
    double sum_sq = 0.0;
    for (ParamState& state : states) {
      for (std::size_t i = 0; i < state.grad.size(); ++i) {
        const float value = 0.05f * rng.Uniform();
        StoreFloat(&state.grad[i], value);
        sum_sq += static_cast<double>(value) * value;
      }
      std::memcpy(state.view->grad, state.grad.data(),
                  state.grad.size() * sizeof(ComputeType));
    }
    const float expected_norm = static_cast<float>(std::sqrt(sum_sq));
    const float scale =
        (clip > 0.0f && expected_norm > clip) ? clip / expected_norm : 1.0f;

    ApplyReference(states, scheduler, oc, scale, step);
    optimizer->Step(step);

    if (std::fabs(optimizer->GradNorm() - expected_norm) > 1e-3f) {
      Fail(Format("step %d grad norm: got %.8g want %.8g", step,
                  optimizer->GradNorm(), expected_norm));
    }

    for (const ParamState& state : states) {
      bool reported = false;
      for (std::size_t i = 0; i < state.ref_value.size(); ++i) {
        const float got = AsF32(state.view->value[i]);
        const float want = AsF32(state.ref_value[i]);
        if (std::fabs(got - want) > kTolerance) {
          Fail(Format("%s[%zu] step %d: got %.8g want %.8g",
                      state.view->name, i, step, got, want));
          reported = true;
          break;
        }
      }
      (void)reported;
    }
  }
  std::printf("optim_test: grouping and updates ok (clip=%.6g)\n", clip);
#endif
}

// ---------------------------------------------------------------------------
// 4. Zero gradient: both optimizers reduce to `p <- p * (1 - lr * wd)`.
// ---------------------------------------------------------------------------

void RunZeroGradTest() {
#if defined(NANOCHAT_PRECISION_FP16)
  std::printf("optim_test: zero-gradient check skipped (fp16 build)\n");
#else
  const Config config = MakeConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(0xfeedfaceull);

  OptimizerConfig oc = MakeOptimizerConfig();
  const Scheduler scheduler(MakeSchedulerConfig());
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), oc, scheduler);

  const std::vector<ParamView> views = model->params();
  std::vector<ParamState> states = MakeStates(views, config, oc);
  std::vector<std::vector<float>> initial(views.size());
  for (std::size_t k = 0; k < views.size(); ++k) {
    initial[k].resize(static_cast<std::size_t>(views[k].count));
    for (std::int64_t i = 0; i < views[k].count; ++i) {
      initial[k][static_cast<std::size_t>(i)] = AsF32(views[k].value[i]);
    }
    nanochat::kernels::Memset(views[k].grad, 0,
                              static_cast<std::size_t>(views[k].count) *
                                  sizeof(ComputeType));
  }

  optimizer->Step(1);

  const float lrm = scheduler.LrMultiplier(1);
  const float muon_decay = scheduler.WeightDecay(1);
  for (std::size_t k = 0; k < states.size(); ++k) {
    const ParamState& state = states[k];
    const float wd =
        state.spec.kind == 0 ? state.spec.weight_decay : muon_decay;
    const float factor = 1.0f - state.spec.lr * lrm * wd;
    for (std::int64_t i = 0; i < state.view->count; ++i) {
      const float got = AsF32(state.view->value[i]);
      const float want = initial[k][static_cast<std::size_t>(i)] * factor;
      if (std::fabs(got - want) > 2e-5f) {
        Fail(Format("zero-grad %s[%lld]: got %.8g want %.8g", state.view->name,
                    static_cast<long long>(i), got, want));
        break;
      }
    }
  }
  std::printf("optim_test: zero-gradient decay ok\n");
#endif
}

// ---------------------------------------------------------------------------
// 5. TrainStep wiring: the model's convenience entry point drives the
// optimizer and advances its own 1-based step counter.
// ---------------------------------------------------------------------------

void RunTrainStepTest() {
#if defined(NANOCHAT_PRECISION_FP16)
  std::printf("optim_test: TrainStep check skipped (fp16 build)\n");
#else
  const Config config = MakeConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(0xabcdef01ull);

  OptimizerConfig oc = MakeOptimizerConfig();
  const Scheduler scheduler(MakeSchedulerConfig());
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), oc, scheduler);

  const int batch = 2;
  const int seq = 8;
  std::vector<int> tokens(static_cast<std::size_t>(batch * seq));
  std::vector<int> targets(static_cast<std::size_t>(batch * seq));
  for (int i = 0; i < batch * seq; ++i) {
    tokens[static_cast<std::size_t>(i)] = (i * 7 + 3) % config.vocab_size;
    targets[static_cast<std::size_t>(i)] = (i * 11 + 5) % config.vocab_size;
  }

  std::vector<ParamView> views = model->params();
  std::vector<float> before(views.size());
  for (std::size_t k = 0; k < views.size(); ++k) {
    before[k] = AsF32(views[k].value[0]);
  }

  const float loss =
      model->TrainStep(tokens.data(), targets.data(), batch, seq,
                       optimizer.get());
  if (!std::isfinite(loss)) Fail("TrainStep returned a non-finite loss");

  double change = 0.0;
  for (std::size_t k = 0; k < views.size(); ++k) {
    change += std::fabs(static_cast<double>(AsF32(views[k].value[0])) -
                        static_cast<double>(before[k]));
  }
  if (change == 0.0) Fail("TrainStep did not update any parameter");
  std::printf("optim_test: TrainStep wiring ok\n");
#endif
}

}  // namespace

int main() {
  TestSchedules();
  RunStepsTest(1e9f);
  RunStepsTest(0.05f);
  RunZeroGradTest();
  RunTrainStepTest();

  if (g_failures != 0) {
    std::fprintf(stderr, "optim_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("optim_test: all checks passed\n");
  return 0;
}
