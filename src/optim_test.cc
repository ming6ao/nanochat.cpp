// src/optim_test.cc -- the optimizer grouping, schedules, and update rules.
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
// Also declared by src/optim_state.h and defined by src/optim.cc: the SFT
// driver's progress-multiplier hand-off (design decision D1).
void SetOptimizerProgressMultiplier(Optimizer* optimizer, float multiplier);
void ClearOptimizerProgressMultiplier(Optimizer* optimizer);
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

inline float AsF32(ComputeType value) { return value; }

inline void StoreFloat(ComputeType* dst, float value) { *dst = value; }

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
      {1, 0.85f}, {2, 0.88f},  {3, 0.91f},  {4, 0.94f},
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
      {1, 0.02f},
      {6, 0.01f},
      {11, 0.0f},
  };
  for (const auto& tc : kWeightDecay) {
    ExpectNear(scheduler.WeightDecay(tc.step), tc.weight_decay, 1e-6f,
               Format("WeightDecay(%d)", tc.step));
  }
  std::printf("optim_test: schedules ok\n");
}

// The SFT progress schedule (design decision D1 of docs/sft-parity-plan.md): the
// three branches of `scripts/chat_sft.py:310-317`, the inclusive right endpoint
// of the constant segment, the `progress == 1.0` endpoint, and the two guard
// cases (`warmup_ratio == 0`, `warmdown_ratio == 0`) that must not divide by
// zero.
void TestLrMultiplierForProgress() {
  SchedulerConfig config = MakeSchedulerConfig();
  config.warmup_ratio = 0.1f;
  config.warmdown_ratio = 0.5f;
  config.final_lr_frac = 0.1f;
  const Scheduler scheduler(config);

  // Branch 1 (warmup): (progress + 1e-8) / warmup_ratio.
  ExpectNear(scheduler.LrMultiplierForProgress(0.0f), 1e-7f, 1e-9f,
             "LrMultiplierForProgress(0) is not the warmup floor");
  ExpectNear(scheduler.LrMultiplierForProgress(0.05f), 0.5f, 1e-6f,
             "LrMultiplierForProgress(0.05)");
  // `progress == warmup_ratio` is not inside the warmup: the comparison is
  // strict, so the value is the constant segment's 1.0.
  ExpectNear(scheduler.LrMultiplierForProgress(0.1f), 1.0f, 1e-6f,
             "LrMultiplierForProgress(0.1) is not at the warmup endpoint");
  // Branch 2 (constant): the right endpoint `1 - warmdown_ratio` is inclusive.
  ExpectNear(scheduler.LrMultiplierForProgress(0.2f), 1.0f, 1e-6f,
             "LrMultiplierForProgress(0.2)");
  ExpectNear(scheduler.LrMultiplierForProgress(0.5f), 1.0f, 1e-6f,
             "LrMultiplierForProgress(0.5) is not inclusive");
  // Branch 3 (warmdown): 1 + decay * (final_lr_frac - 1).
  ExpectNear(scheduler.LrMultiplierForProgress(0.75f), 0.55f, 1e-6f,
             "LrMultiplierForProgress(0.75)");
  // The endpoint returns `final_lr_frac` exactly.
  ExpectNear(scheduler.LrMultiplierForProgress(1.0f), config.final_lr_frac,
             1e-6f, "LrMultiplierForProgress(1.0) is not final_lr_frac");

  // The pinned SFT run: no warmup (section 5.2), so the first branch is off and
  // no input may divide by zero.
  SchedulerConfig pinned = MakeSchedulerConfig();
  pinned.warmup_ratio = 0.0f;
  pinned.warmdown_ratio = 0.5f;
  pinned.final_lr_frac = 0.0f;
  const Scheduler sft(pinned);
  const struct {
    float progress;
    float lr;
  } kPinned[] = {
      {0.0f, 1.0f}, {0.5f, 1.0f}, {0.625f, 0.75f}, {0.75f, 0.5f}, {1.0f, 0.0f},
  };
  for (const auto& tc : kPinned) {
    const float got = sft.LrMultiplierForProgress(tc.progress);
    if (!std::isfinite(got)) {
      Fail(Format("pinned warmup_ratio 0: progress %.3f is not finite",
                  static_cast<double>(tc.progress)));
      continue;
    }
    ExpectNear(got, tc.lr, 1e-6f,
               Format("pinned LrMultiplierForProgress(%.3f)",
                      static_cast<double>(tc.progress)));
  }

  // A zero warmdown leaves no segment to walk. `progress == 1.0` is still the
  // inclusive right endpoint of the constant segment, so it returns 1.0, and a
  // progress past the end returns the final fraction rather than dividing by
  // zero.
  SchedulerConfig no_warmdown = pinned;
  no_warmdown.warmdown_ratio = 0.0f;
  no_warmdown.final_lr_frac = 0.25f;
  const Scheduler flat(no_warmdown);
  ExpectNear(flat.LrMultiplierForProgress(0.5f), 1.0f, 1e-6f,
             "zero warmdown: progress 0.5");
  ExpectNear(flat.LrMultiplierForProgress(1.0f), 1.0f, 1e-6f,
             "zero warmdown: progress 1.0 is the inclusive endpoint");
  const float endpoint = flat.LrMultiplierForProgress(1.5f);
  if (!std::isfinite(endpoint)) {
    Fail("zero warmdown: a progress past the end is not finite");
  }
  ExpectNear(endpoint, 0.25f, 1e-6f, "zero warmdown: progress 1.5");

  // The pretraining entry point is untouched: `warmup_ratio` is not one of its
  // inputs, so setting it changes no step-based value (design decision D1 keeps
  // `LrMultiplier(int step)` for pretraining).
  SchedulerConfig with_ratio = MakeSchedulerConfig();
  with_ratio.warmup_ratio = 0.4f;
  const Scheduler pretrain_ratio(with_ratio);
  const Scheduler pretrain(MakeSchedulerConfig());
  for (int step = 1; step <= 11; ++step) {
    ExpectNear(pretrain_ratio.LrMultiplier(step), pretrain.LrMultiplier(step),
               0.0f,
               Format("warmup_ratio changed LrMultiplier(%d)", step));
  }
  std::printf("optim_test: progress schedule ok\n");
}

// The warmdown count must match Python's `round`, so a half-integer product
// rounds to even. A half-away-from-zero count starts the warmdown one step
// early. See docs/parity.md D9.
void TestWarmdownRounding() {
  SchedulerConfig config = MakeSchedulerConfig();
  config.num_iterations = 5;
  config.warmdown_ratio = 0.5f;  // 0.5 * 5 = 2.5 -> half to even -> 2
  const Scheduler scheduler(config);

  const struct {
    int step;
    float lr;
  } kLr[] = {
      {4, 1.0f},
      {5, 0.55f},
  };
  for (const auto& tc : kLr) {
    ExpectNear(scheduler.LrMultiplier(tc.step), tc.lr, 1e-6f,
               Format("warmdown rounding: LrMultiplier(%d)", tc.step));
  }

  const struct {
    int step;
    float momentum;
  } kMomentum[] = {
      {5, 0.935f},
  };
  for (const auto& tc : kMomentum) {
    ExpectNear(scheduler.MuonMomentum(tc.step), tc.momentum, 1e-6f,
               Format("warmdown rounding: MuonMomentum(%d)", tc.step));
  }
  std::printf("optim_test: warmdown rounding ok\n");
}

// ---------------------------------------------------------------------------
// 2/3. Grouping and a few update steps
// ---------------------------------------------------------------------------

// The expected group for a parameter: kind, nominal learning rate, and (for
// AdamW) the betas and weight decay from `setup_optimizer`.
//
// The nominal rate is the section 5.2 rate after the `(hidden_dim / 768) **
// -0.5` scale and after the single `init_lr_frac` multiplication of design
// decision D3, which `Build` applies to every group. A caller that leaves
// `init_lr_frac` at its default of 1.0 keeps the unscaled rates.
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
  const float scale = std::sqrt(768.0f / static_cast<float>(config.hidden_dim));
  const float frac = oc.init_lr_frac;
  if (name == "lm_head.weight") {
    return {0, oc.unembedding_lr * scale * frac, 0.8f, 0.96f, 0.01f};
  }
  if (name == "transformer.wte.weight") {
    return {0, oc.embedding_lr * scale * frac, 0.8f, 0.995f, 0.001f};
  }
  if (name.rfind("value_embeds.", 0) == 0) {
    return {0, oc.embedding_lr * scale * 0.5f * frac, 0.8f, 0.995f, 0.01f};
  }
  if (name == "resid_lambdas") {
    return {0, oc.scalar_lr * 0.01f * frac, 0.8f, 0.95f, 0.05f};
  }
  if (name == "x0_lambdas") {
    return {0, oc.scalar_lr * frac, 0.96f, 0.95f, 0.0f};
  }
  if (name == "smear_gate.weight" || name == "smear_lambda" ||
      name == "backout_lambda") {
    // The 1 x 24 smear gate is deliberately an AdamW parameter, and its rate is
    // the recipe's constant 0.2 rather than a configured rate. D3 scales it
    // like every other group, because the reference scales the smear group too.
    return {0, 0.2f * frac, 0.8f, 0.95f, 0.0f};
  }
  const float lr = oc.matrix_lr * frac *
                   std::sqrt(std::max(1.0f, static_cast<float>(rows) /
                                                static_cast<float>(cols)));
  return {1, lr, 0.0f, 0.0f, 0.0f};
}

struct ParamState {
  const ParamView* view = nullptr;
  GroupSpec spec;
  int rows = 0;
  int cols = 0;
  std::vector<float> grad;
  std::vector<ComputeType> ref_value;  // expected current compute copy
  std::vector<float> ref_master;       // expected current fp32 master
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
    state.grad.assign(count, 0.0f);
    state.ref_value.resize(count);
    state.ref_master.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      state.ref_value[i] = view.value[i];
      state.ref_master[i] = view.master[i];
    }
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
        const float g = state.grad[i] * grad_scale;
        float p = state.ref_master[i];
        p *= (1.0f - lr * state.spec.weight_decay);
        state.m[i] += (1.0f - state.spec.beta1) * (g - state.m[i]);
        state.v[i] += (1.0f - state.spec.beta2) * (g * g - state.v[i]);
        const float denom = std::sqrt(state.v[i] / bias2) + oc.adam_eps;
        p -= step_size * (state.m[i] / denom);
        state.ref_master[i] = p;
        StoreFloat(&state.ref_value[i], p);
      }
    } else {
      std::vector<float> scaled(state.grad.size());
      for (std::size_t i = 0; i < state.grad.size(); ++i) {
        scaled[i] = state.grad[i] * grad_scale;
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
      nanochat::kernels::MuonUpdate(
          params, scaled.data(), state.ref_master.data(),
          state.ref_value.data(), state.buf1.data(), state.buf2.data());
    }
  }
}

void RunStepsTest(float clip, float init_lr_frac = 1.0f) {
  const Config config = MakeConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(0x5eed1234ull);

  OptimizerConfig oc = MakeOptimizerConfig();
  oc.clip = clip;
  oc.init_lr_frac = init_lr_frac;
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
        state.grad[i] = value;
        sum_sq += static_cast<double>(value) * value;
      }
      std::memcpy(state.view->grad, state.grad.data(),
                  state.grad.size() * sizeof(float));
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
      for (std::size_t i = 0; i < state.ref_master.size(); ++i) {
        const float got = state.view->master[i];
        const float want = state.ref_master[i];
        if (std::fabs(got - want) > kTolerance) {
          Fail(Format("%s[%zu] step %d: got %.8g want %.8g", state.view->name,
                      i, step, got, want));
          reported = true;
          break;
        }
      }
      (void)reported;
    }
  }
  std::printf("optim_test: grouping and updates ok (clip=%.6g, "
              "init_lr_frac=%.6g)\n",
              clip, static_cast<double>(init_lr_frac));
}

// ---------------------------------------------------------------------------
// 4. Zero gradient: both optimizers reduce to `p <- p * (1 - lr * wd)`.
// ---------------------------------------------------------------------------

void RunZeroGradTest() {
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
      initial[k][static_cast<std::size_t>(i)] = views[k].master[i];
    }
    nanochat::kernels::Memset(
        views[k].grad, 0,
        static_cast<std::size_t>(views[k].count) * sizeof(float));
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
      const float want = nanochat::ComputeToFloat(nanochat::ComputeFromFloat(
          initial[k][static_cast<std::size_t>(i)] * factor));
      if (std::fabs(got - want) > 2e-5f) {
        Fail(Format("zero-grad %s[%lld]: got %.8g want %.8g", state.view->name,
                    static_cast<long long>(i), got, want));
        break;
      }
    }
  }
  std::printf("optim_test: zero-gradient decay ok\n");
}

// Design decision D3: `init_lr_frac` scales every group rate inside `Build`,
// exactly once. Two optimizers over the same model shape, one at 1.0 and one at
// 0.8, must report group rates in that exact ratio -- for the unembedding, the
// embedding, the value embeddings, the residual lambdas, the x0 lambdas, the
// smear and backout group, and the Muon matrices. A rate scaled twice would show
// up as 0.64 of the base rate and a group that missed the scale as 1.0 of it, so
// the ratio check fails on both mistakes.
void TestInitLrFrac() {
  constexpr float kFrac = 0.8f;
  const Config config = MakeConfig();
  std::unique_ptr<Model> full_model = Model::Create(config);
  std::unique_ptr<Model> scaled_model = Model::Create(config);

  OptimizerConfig full = MakeOptimizerConfig();
  OptimizerConfig scaled = MakeOptimizerConfig();
  scaled.init_lr_frac = kFrac;
  const Scheduler scheduler(MakeSchedulerConfig());
  std::unique_ptr<Optimizer> full_optimizer =
      nanochat::CreateOptimizer(full_model.get(), full, scheduler);
  std::unique_ptr<Optimizer> scaled_optimizer =
      nanochat::CreateOptimizer(scaled_model.get(), scaled, scheduler);

  int adamw_params = 0;
  int muon_params = 0;
  int smear_params = 0;
  for (const ParamView& view : full_model->params()) {
    int full_kind = -1;
    int scaled_kind = -1;
    float full_lr = 0.0f;
    float scaled_lr = 0.0f;
    if (!nanochat::OptimizerParamGroupForTest(full_optimizer.get(), view.name,
                                              &full_kind, nullptr, nullptr,
                                              &full_lr) ||
        !nanochat::OptimizerParamGroupForTest(scaled_optimizer.get(), view.name,
                                              &scaled_kind, nullptr, nullptr,
                                              &scaled_lr)) {
      Fail("init_lr_frac: unclassified parameter " + std::string(view.name));
      continue;
    }
    if (full_kind != scaled_kind) {
      Fail("init_lr_frac: the grouping changed with the scale: " +
           std::string(view.name));
    }
    ExpectNear(scaled_lr, kFrac * full_lr, 1e-6f,
               "init_lr_frac rate " + std::string(view.name));
    const std::string name(view.name);
    if (full_kind == 0) {
      ++adamw_params;
    } else {
      ++muon_params;
    }
    if (name == "smear_gate.weight" || name == "smear_lambda" ||
        name == "backout_lambda") {
      ++smear_params;
      // The smear group's rate is the recipe's constant 0.2, and the reference
      // scales it too, even though it is not a configured rate.
      ExpectNear(full_lr, 0.2f, 1e-6f, "init_lr_frac smear rate " + name);
      ExpectNear(scaled_lr, 0.2f * kFrac, 1e-6f,
                 "init_lr_frac scaled smear rate " + name);
    }
  }
  if (adamw_params == 0 || muon_params == 0) {
    Fail("init_lr_frac: the model has no AdamW or no Muon parameters");
  }
  if (smear_params == 0) Fail("init_lr_frac: the smear group was not seen");

  // The scaled rate is also the rate the first step uses. At step 1 with
  // `beta1 = 0.8` and `beta2 = 0.96` the bias correction turns the AdamW update
  // into `- lr * sign(g)`, so the unembedding's closed form is
  // `p * (1 - lr * 0.01) - lr * sign(g)` and a rate scaled twice would move an
  // element about 0.64/0.8 as far as it should.
  std::unique_ptr<Model> step_full_model = Model::Create(config);
  std::unique_ptr<Model> step_scaled_model = Model::Create(config);
  step_full_model->InitWeights(0x1313ull);
  step_scaled_model->InitWeights(0x1313ull);
  std::unique_ptr<Optimizer> step_full =
      nanochat::CreateOptimizer(step_full_model.get(), full, scheduler);
  std::unique_ptr<Optimizer> step_scaled =
      nanochat::CreateOptimizer(step_scaled_model.get(), scaled, scheduler);

  std::vector<float> grad;
  std::vector<float> initial_full;
  std::vector<float> initial_scaled;
  Rng rng(0x1f2e3d4cull);
  for (const ParamView& view : step_full_model->params()) {
    nanochat::kernels::Memset(
        view.grad, 0, static_cast<std::size_t>(view.count) * sizeof(float));
  }
  for (const ParamView& view : step_scaled_model->params()) {
    nanochat::kernels::Memset(
        view.grad, 0, static_cast<std::size_t>(view.count) * sizeof(float));
  }
  for (const ParamView& view : step_full_model->params()) {
    if (std::string(view.name) != "lm_head.weight") continue;
    grad.resize(static_cast<std::size_t>(view.count));
    initial_full.resize(grad.size());
    for (std::size_t i = 0; i < grad.size(); ++i) {
      grad[i] = 0.05f * rng.Uniform();
      initial_full[i] = view.master[i];
    }
    std::memcpy(view.grad, grad.data(), grad.size() * sizeof(float));
    for (const ParamView& other : step_scaled_model->params()) {
      if (std::string(other.name) != view.name) continue;
      std::memcpy(other.grad, grad.data(), grad.size() * sizeof(float));
      initial_scaled.assign(grad.size(), 0.0f);
      for (std::size_t i = 0; i < grad.size(); ++i) {
        initial_scaled[i] = other.master[i];
      }
    }
  }
  step_full->Step(1);
  step_scaled->Step(1);

  const float scale = std::sqrt(768.0f / static_cast<float>(config.hidden_dim));
  const float lrm = scheduler.LrMultiplier(1);
  for (const ParamView& view : step_full_model->params()) {
    if (std::string(view.name) != "lm_head.weight") continue;
    const float lr = full.unembedding_lr * scale * full.init_lr_frac * lrm;
    for (std::int64_t i = 0; i < view.count; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      const float g = grad[index];
      const float sign = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
      const float want = initial_full[index] * (1.0f - lr * 0.01f) - lr * sign;
      const float got = view.master[i];
      if (std::fabs(got - want) > 2e-5f) {
        Fail(Format("init_lr_frac step[%zu]: got %.8g want %.8g", index, got,
                    want));
        break;
      }
    }
  }
  for (const ParamView& view : step_scaled_model->params()) {
    if (std::string(view.name) != "lm_head.weight") continue;
    const float lr = scaled.unembedding_lr * scale * scaled.init_lr_frac * lrm;
    for (std::int64_t i = 0; i < view.count; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      const float g = grad[index];
      const float sign = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
      const float want =
          initial_scaled[index] * (1.0f - lr * 0.01f) - lr * sign;
      const float got = view.master[i];
      if (std::fabs(got - want) > 2e-5f) {
        Fail(Format("init_lr_frac scaled step[%zu]: got %.8g want %.8g",
                    index, got, want));
        break;
      }
    }
  }
  std::printf("optim_test: init_lr_frac ok\n");
}

// ---------------------------------------------------------------------------
// 5. TrainStep wiring: the model's convenience entry point drives the
// optimizer and advances its own 1-based step counter.
// ---------------------------------------------------------------------------

void RunTrainStepTest() {
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

  const float loss = model->TrainStep(tokens.data(), targets.data(), batch, seq,
                                      optimizer.get());
  if (!std::isfinite(loss)) Fail("TrainStep returned a non-finite loss");

  double change = 0.0;
  for (std::size_t k = 0; k < views.size(); ++k) {
    change += std::fabs(static_cast<double>(AsF32(views[k].value[0])) -
                        static_cast<double>(before[k]));
  }
  if (change == 0.0) Fail("TrainStep did not update any parameter");
  std::printf("optim_test: TrainStep wiring ok\n");
}

// The AdamW wiring: two steps must match a canonical AdamW with bias
// correction at `t` = 1 and 2, driven through `kernels::AdamWUpdate`.
void TestAdamWiring() {
  const Config config = MakeConfig();
  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(0xcad4e001ull);

  OptimizerConfig oc = MakeOptimizerConfig();
  const Scheduler scheduler(MakeSchedulerConfig());
  std::unique_ptr<Optimizer> optimizer =
      nanochat::CreateOptimizer(model.get(), oc, scheduler);

  const std::vector<ParamView> views = model->params();
  const ParamView* target = nullptr;
  for (const ParamView& view : views) {
    if (std::string(view.name) == "lm_head.weight") target = &view;
  }
  if (target == nullptr) {
    Fail("AdamW wiring: lm_head.weight is missing");
    return;
  }

  for (const ParamView& view : views) {
    nanochat::kernels::Memset(
        view.grad, 0, static_cast<std::size_t>(view.count) * sizeof(float));
  }
  const std::size_t count = static_cast<std::size_t>(target->count);
  Rng rng(0xca4e5eedull);
  std::vector<float> grad(count);
  for (std::size_t i = 0; i < count; ++i) {
    grad[i] = 0.05f * rng.Uniform();
  }
  std::memcpy(target->grad, grad.data(), count * sizeof(float));

  std::vector<float> initial(count);
  for (std::size_t i = 0; i < count; ++i) initial[i] = target->master[i];

  optimizer->Step(1);
  optimizer->Step(2);

  const float scale = std::sqrt(768.0f / static_cast<float>(config.hidden_dim));
  const float group_lr = oc.unembedding_lr * scale;
  const float beta1 = 0.8f;
  const float beta2 = 0.96f;
  const float wd = 0.01f;
  std::vector<float> want = initial;
  std::vector<float> m(count, 0.0f);
  std::vector<float> v(count, 0.0f);
  for (int t = 1; t <= 2; ++t) {
    const float lr = group_lr * scheduler.LrMultiplier(t);
    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    const float step_size = lr / bias1;
    for (std::size_t i = 0; i < count; ++i) {
      const float g = grad[i];
      float p = want[i] * (1.0f - lr * wd);
      m[i] += (1.0f - beta1) * (g - m[i]);
      v[i] += (1.0f - beta2) * (g * g - v[i]);
      const float denom = std::sqrt(v[i] / bias2) + oc.adam_eps;
      p -= step_size * (m[i] / denom);
      want[i] = p;
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    const float got = AsF32(target->value[i]);
    const float expected =
        nanochat::ComputeToFloat(nanochat::ComputeFromFloat(want[i]));
    if (std::fabs(got - expected) > 2e-5f) {
      Fail(Format("AdamW wiring[%zu]: got %.8g want %.8g", i, got, expected));
      break;
    }
  }
  std::printf("optim_test: AdamW wiring ok\n");
}

// The progress schedule reaches the optimizer through
// `SetOptimizerProgressMultiplier` (design decision D1). The multiplier
// replaces the step-based `LrMultiplier(step)` output for the step, it does not
// multiply it, so the resolved learning rate is `group.lr * multiplier` with the
// `init_lr_frac` scale in `group.lr` applied once and nothing squared.
void TestProgressMultiplierOverride() {
  const Config config = MakeConfig();
  SchedulerConfig scheduler_config = MakeSchedulerConfig();
  scheduler_config.num_iterations = 8;
  scheduler_config.warmup_steps = 0;
  scheduler_config.warmup_ratio = 0.0f;
  scheduler_config.warmdown_ratio = 0.5f;
  scheduler_config.final_lr_frac = 0.0f;
  const Scheduler scheduler(scheduler_config);
  // The step-based curve is 1.0 at step 1, so a multiplier of 0.5 can only come
  // from the override; a value of 0.25 would mean the multiplier was applied
  // twice.
  ExpectNear(scheduler.LrMultiplier(1), 1.0f, 1e-6f,
             "override: the step-based curve is not 1.0 at step 1");
  ExpectNear(scheduler.LrMultiplierForProgress(0.75f), 0.5f, 1e-6f,
             "override: the progress curve is not 0.5 at 0.75");

  OptimizerConfig oc = MakeOptimizerConfig();
  oc.init_lr_frac = 0.8f;

  std::vector<std::unique_ptr<Model>> models;
  std::vector<std::unique_ptr<Optimizer>> optimizers;
  for (int k = 0; k < 3; ++k) {
    models.push_back(Model::Create(config));
    models.back()->InitWeights(0x9977ull);
    optimizers.push_back(
        nanochat::CreateOptimizer(models.back().get(), oc, scheduler));
  }
  // The three optimizers share one gradient table, so every difference in the
  // result comes from the multiplier.
  Rng rng(0x5a5a1234ull);
  for (const ParamView& view : models[0]->params()) {
    nanochat::kernels::Memset(
        view.grad, 0, static_cast<std::size_t>(view.count) * sizeof(float));
  }
  const ParamView* head0 = nullptr;
  for (const ParamView& view : models[0]->params()) {
    if (std::string(view.name) == "lm_head.weight") head0 = &view;
  }
  if (head0 == nullptr) {
    Fail("override: lm_head.weight is missing");
    return;
  }
  const std::size_t count = static_cast<std::size_t>(head0->count);
  std::vector<float> grad(count);
  std::vector<float> initial(count);
  for (std::size_t i = 0; i < count; ++i) {
    grad[i] = 0.05f * rng.Uniform();
    initial[i] = head0->master[i];
  }
  for (std::size_t k = 0; k < models.size(); ++k) {
    for (const ParamView& view : models[k]->params()) {
      nanochat::kernels::Memset(
          view.grad, 0, static_cast<std::size_t>(view.count) * sizeof(float));
      if (std::string(view.name) != "lm_head.weight") continue;
      std::memcpy(view.grad, grad.data(), count * sizeof(float));
    }
  }

  // Optimizer 0 takes the SFT progress multiplier, optimizer 1 the step-based
  // one, and optimizer 2 has its override cleared before the step.
  nanochat::SetOptimizerProgressMultiplier(optimizers[0].get(), 0.5f);
  nanochat::SetOptimizerProgressMultiplier(optimizers[1].get(),
                                           scheduler.LrMultiplier(1));
  nanochat::SetOptimizerProgressMultiplier(optimizers[2].get(), 0.5f);
  nanochat::ClearOptimizerProgressMultiplier(optimizers[2].get());
  for (const auto& optimizer : optimizers) optimizer->Step(1);

  const float scale = std::sqrt(768.0f / static_cast<float>(config.hidden_dim));
  const float group_lr = oc.unembedding_lr * scale * oc.init_lr_frac;
  for (std::size_t k = 0; k < models.size(); ++k) {
    const float multiplier = k == 0 ? 0.5f : scheduler.LrMultiplier(1);
    const float lr = group_lr * multiplier;
    for (const ParamView& view : models[k]->params()) {
      if (std::string(view.name) != "lm_head.weight") continue;
      for (std::size_t i = 0; i < count; ++i) {
        const float sign =
            grad[i] > 0.0f ? 1.0f : (grad[i] < 0.0f ? -1.0f : 0.0f);
        const float want = initial[i] * (1.0f - lr * 0.01f) - lr * sign;
        const float got = view.master[static_cast<std::int64_t>(i)];
        if (std::fabs(got - want) > 2e-5f) {
          Fail(Format("override optimizer %zu value %zu: got %.8g want %.8g",
                      k, i, got, want));
          break;
        }
      }
    }
  }
  std::printf("optim_test: progress multiplier override ok\n");
}

}  // namespace

int main() {
  TestSchedules();
  TestLrMultiplierForProgress();
  TestWarmdownRounding();
  RunStepsTest(1e9f);
  RunStepsTest(0.05f);
  // The pinned SFT run: every group rate carries the `init_lr_frac` scale, and
  // the three steps must still match a reference that applies it once
  // (design decision D3).
  RunStepsTest(1e9f, 0.8f);
  TestInitLrFrac();
  RunZeroGradTest();
  RunTrainStepTest();
  TestAdamWiring();
  TestProgressMultiplierOverride();

  if (g_failures != 0) {
    std::fprintf(stderr, "optim_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("optim_test: all checks passed\n");
  return 0;
}
