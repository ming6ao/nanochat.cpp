// The C application binary interface test (docs/python.md section 11,
// phase 1). It calls every function in `nanochat/capi.h` and checks the
// contract: the sandbox rule, the one owned handle per create, the flat struct
// copies, the thread-local error, and the round trip of the tokenizer, the
// loader, the optimizer, the model, and the two inference entry points.
//
// The static assertions below compare each flat C mirror struct to the C++
// struct in `nanochat/model.h` that it copies. A type change in the C++ struct
// stops this test at compile time, so an ABI drift cannot pass silently
// (docs/python.md section 13).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/capi.h"
#include "nanochat/config.h"
#include "nanochat/model.h"
#include "nanochat/tensor.h"
#include "src/train.h"

namespace {

// --- Drift checks against the C++ structs --------------------------------

static_assert(sizeof(nanochat_config::num_layers) ==
                  sizeof(nanochat::Config::num_layers),
              "nanochat_config.num_layers drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::num_heads) ==
                  sizeof(nanochat::Config::num_heads),
              "nanochat_config.num_heads drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::num_kv_heads) ==
                  sizeof(nanochat::Config::num_kv_heads),
              "nanochat_config.num_kv_heads drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::hidden_dim) ==
                  sizeof(nanochat::Config::hidden_dim),
              "nanochat_config.hidden_dim drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::seq_len) ==
                  sizeof(nanochat::Config::seq_len),
              "nanochat_config.seq_len drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::vocab_size) ==
                  sizeof(nanochat::Config::vocab_size),
              "nanochat_config.vocab_size drifted from nanochat::Config");
static_assert(
    sizeof(nanochat_config::padded_vocab_size) ==
        sizeof(nanochat::Config::padded_vocab_size),
    "nanochat_config.padded_vocab_size drifted from nanochat::Config");
static_assert(sizeof(nanochat_config::rope_base) ==
                  sizeof(nanochat::Config::rope_base),
              "nanochat_config.rope_base drifted from nanochat::Config");
// The C mirror widens the C++ `bool` to an `int`, so this is a size floor
// rather than an equality, the same rule the other scalar mirrors use.
static_assert(sizeof(nanochat_config::value_embedding) >=
                  sizeof(nanochat::Config::value_embedding),
              "nanochat_config.value_embedding drifted from nanochat::Config");

static_assert(sizeof(nanochat_param::count) ==
                  sizeof(nanochat::ParamView::count),
              "nanochat_param.count drifted from nanochat::ParamView");
static_assert(sizeof(nanochat_param::rows) == sizeof(nanochat::ParamView::rows),
              "nanochat_param.rows drifted from nanochat::ParamView");
static_assert(sizeof(nanochat_param::cols) == sizeof(nanochat::ParamView::cols),
              "nanochat_param.cols drifted from nanochat::ParamView");

static_assert(sizeof(nanochat_device::total_memory_bytes) ==
                  sizeof(nanochat::Caps::total_memory_bytes),
              "nanochat_device.total_memory_bytes drifted from nanochat::Caps");
static_assert(std::is_same<decltype(nanochat_device::device_name),
                           decltype(nanochat::Caps::device_name)>::value,
              "nanochat_device.device_name drifted from nanochat::Caps");

static_assert(std::is_same<decltype(nanochat_focus::position),
                           decltype(nanochat::ScoreFocus::position)>::value,
              "nanochat_focus.position drifted from nanochat::ScoreFocus");
static_assert(std::is_same<decltype(nanochat_focus::ids),
                           decltype(nanochat::ScoreFocus::ids)>::value,
              "nanochat_focus.ids drifted from nanochat::ScoreFocus");
static_assert(std::is_same<decltype(nanochat_focus::count),
                           decltype(nanochat::ScoreFocus::count)>::value,
              "nanochat_focus.count drifted from nanochat::ScoreFocus");

static_assert(
    std::is_same<decltype(nanochat_generate_params::num_samples),
                 decltype(nanochat::GenerateParams::num_samples)>::value,
    "nanochat_generate_params.num_samples drifted");
static_assert(
    std::is_same<decltype(nanochat_generate_params::max_tokens),
                 decltype(nanochat::GenerateParams::max_tokens)>::value,
    "nanochat_generate_params.max_tokens drifted");
static_assert(
    std::is_same<decltype(nanochat_generate_params::temperature),
                 decltype(nanochat::GenerateParams::temperature)>::value,
    "nanochat_generate_params.temperature drifted");
static_assert(std::is_same<decltype(nanochat_generate_params::top_k),
                           decltype(nanochat::GenerateParams::top_k)>::value,
              "nanochat_generate_params.top_k drifted");
static_assert(std::is_same<decltype(nanochat_generate_params::seed),
                           decltype(nanochat::GenerateParams::seed)>::value,
              "nanochat_generate_params.seed drifted");
static_assert(std::is_same<decltype(nanochat_generate_params::stop_id),
                           decltype(nanochat::GenerateParams::stop_id)>::value,
              "nanochat_generate_params.stop_id drifted");
static_assert(std::is_same<decltype(nanochat_generate_params::bos_id),
                           decltype(nanochat::GenerateParams::bos_id)>::value,
              "nanochat_generate_params.bos_id drifted");
static_assert(std::is_same<decltype(nanochat_generate_params::stop_ids),
                           decltype(nanochat::GenerateParams::stop_ids)>::value,
              "nanochat_generate_params.stop_ids drifted");

// --- Test harness ---------------------------------------------------------

// The weighted and uniform backward paths reach the same value through
// different arithmetic. fp32 agrees to a tight tolerance; the fp16 storage
// rounds the per-row scale, so it needs a wider one.
#if defined(NANOCHAT_PRECISION_FP16)
constexpr double kGradientTolerance = 5e-2;
#else
constexpr double kGradientTolerance = 1e-4;
#endif

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Check(const char* what, bool ok) {
  if (!ok) Fail(what);
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

void ClearSandboxEnv() {
  unsetenv("NANOCHAT_SANDBOX");
  unsetenv("NANOCHAT_SANDBOX_BACKEND");
  unsetenv("NANOCHAT_ALLOW_UNSANDBOXED");
}

// The sandbox rule of docs/python.md section 2.1, in each branch.
void CheckSandboxRule() {
  ClearSandboxEnv();
  Check("sandbox: rejects an unsandboxed process",
        nanochat_init() != NANOCHAT_STATUS_OK);
  Check("sandbox: sets the corrective message",
        std::strlen(nanochat_last_error()) > 0);

  ClearSandboxEnv();
  setenv("NANOCHAT_SANDBOX", "t0-cpu", 1);
  Check("sandbox: accepts NANOCHAT_SANDBOX",
        nanochat_init() == NANOCHAT_STATUS_OK);

  ClearSandboxEnv();
  setenv("NANOCHAT_SANDBOX_BACKEND", "none", 1);
  Check("sandbox: accepts the none backend",
        nanochat_init() == NANOCHAT_STATUS_OK);

  ClearSandboxEnv();
  setenv("NANOCHAT_ALLOW_UNSANDBOXED", "1", 1);
  Check("sandbox: accepts the explicit override",
        nanochat_init() == NANOCHAT_STATUS_OK);

  // Leave the process in the accepted state for the remaining calls.
  ClearSandboxEnv();
  setenv("NANOCHAT_SANDBOX", "t0-cpu", 1);
  Check("init: ready", nanochat_init() == NANOCHAT_STATUS_OK);
}

// The device enumeration of docs/client-runtime-design.md section 5.1. The
// reference backend reports one CPU device, and `nanochat_device_at` rejects an
// out-of-range ordinal.
void CheckDeviceEnumeration() {
  const int count = nanochat_device_count();
  Check("device: the count is positive", count >= 1);
  nanochat_device first;
  std::memset(&first, 0, sizeof(first));
  Check("device: ordinal 0 resolves", nanochat_device_at(0, &first) == 0);
  Check("device: ordinal 0 has a name",
        first.device_name != nullptr && first.device_name[0] != '\0');
  Check("device: ordinal 0 is the first index", first.device_index == 0);
  nanochat_device outside;
  std::memset(&outside, 0, sizeof(outside));
  Check("device: an out-of-range ordinal is rejected",
        nanochat_device_at(count, &outside) != 0);
  Check("device: a negative ordinal is rejected",
        nanochat_device_at(-1, &outside) != 0);
  Check("device: a null out pointer is rejected",
        nanochat_device_at(0, nullptr) != 0);
}

nanochat_config TinyConfig() {
  nanochat_config config;
  std::memset(&config, 0, sizeof(config));
  config.num_layers = 2;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  // The value gate reads the first 12 channels, so the width must cover them;
  // a narrower model would skip the gate (ops::ValueResidualForward).
  config.hidden_dim = 32;
  config.seq_len = 8;
  // The loader fixture tokenizer has a vocabulary of 487 (ids up to 486), so
  // the model must cover it for `nanochat_eval_bpb` to index safely.
  config.vocab_size = 512;
  config.padded_vocab_size = 512;
  config.rope_base = 10000.0f;
  config.window_pattern = "S";
  // The expected parameter counts include the value embeddings and the
  // odd-layer value gate, so the tiny config keeps them on.
  config.value_embedding = 1;
  return config;
}

// The plan inputs of docs/python.md section 4.1. The expected counts below
// are hand-derived from the reference `GPT.num_scaling_params` grouping for
// `TinyConfig`: query/key/value/proj and MLP per layer, the value gate on the
// odd layer, `lm_head`, the token and value embeddings, and the scalars.
void CheckParams() {
  const nanochat_config config = TinyConfig();
  nanochat_params counts;
  std::memset(&counts, 0, sizeof(counts));
  Check("params: get returns ok",
        nanochat_params_get(&config, &counts) == NANOCHAT_STATUS_OK);
  Check("params: total is the group sum",
        counts.total == counts.transformer_matrices + counts.lm_head +
                            counts.embeddings + counts.scalars);
  Check("params: transformer_matrices", counts.transformer_matrices == 22540);
  Check("params: lm_head", counts.lm_head == 16384);
  Check("params: embeddings", counts.embeddings == 24576);
  Check("params: scalars", counts.scalars == 30);
  Check("params: total", counts.total == 63530);
  Check("params: flops_per_token is positive", counts.flops_per_token > 0.0);

  Check("params: a null config fails",
        nanochat_params_get(nullptr, &counts) != NANOCHAT_STATUS_OK);
  Check("params: a null out fails",
        nanochat_params_get(&config, nullptr) != NANOCHAT_STATUS_OK);
}

std::vector<int> MakeTokens(int batch, int seq, int vocab) {
  std::vector<int> tokens(static_cast<std::size_t>(batch) * seq);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int>((i * 7 + 3) % static_cast<std::size_t>(vocab));
  }
  return tokens;
}

std::vector<int> MakeTargets(int batch, int seq, int vocab) {
  std::vector<int> targets(static_cast<std::size_t>(batch) * seq);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    targets[i] =
        static_cast<int>((i * 11 + 5) % static_cast<std::size_t>(vocab));
  }
  return targets;
}

// Reads one parameter gradient into a float vector. It handles both build
// precisions: a four-byte element is float32, a two-byte element is float16.
std::vector<float> ReadGradient(nanochat_model* model, int index) {
  nanochat_param param;
  std::memset(&param, 0, sizeof(param));
  if (nanochat_param_info(model, index, &param) != 0) return {};
  const int element_size = nanochat_compute_type_size();
  const std::size_t bytes = static_cast<std::size_t>(param.count) *
                            static_cast<std::size_t>(element_size);
  std::vector<unsigned char> raw(bytes);
  if (nanochat_param_read(model, index, /*grad=*/1, 0, param.count,
                          raw.data()) != param.count) {
    return {};
  }
  std::vector<float> values(static_cast<std::size_t>(param.count));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (element_size == 4) {
      float value = 0.0f;
      std::memcpy(&value, raw.data() + i * 4, sizeof(value));
      values[i] = value;
    } else {
      nanochat::Fp16 value;
      std::memcpy(&value.bits, raw.data() + i * 2, sizeof(value.bits));
      values[i] = nanochat::Fp16ToFloat(value);
    }
  }
  return values;
}

// Checks that the gradient of `actual` equals `expected_scale` times the
// gradient of `expected`, element by element, with a relative tolerance. A
// near-zero element falls back to an absolute comparison.
void CheckScaledGradients(nanochat_model* expected, nanochat_model* actual,
                          double expected_scale, const char* what,
                          double tolerance) {
  const int count = nanochat_param_count(expected);
  if (count != nanochat_param_count(actual)) {
    Fail(std::string(what) + ": parameter counts differ");
    return;
  }
  for (int i = 0; i < count; ++i) {
    const std::vector<float> a = ReadGradient(expected, i);
    const std::vector<float> b = ReadGradient(actual, i);
    if (a.size() != b.size()) {
      Fail(std::string(what) + ": gradient sizes differ");
      return;
    }
    for (std::size_t j = 0; j < a.size(); ++j) {
      const double want = expected_scale * static_cast<double>(a[j]);
      const double got = static_cast<double>(b[j]);
      const double magnitude =
          std::fmax(1.0, std::fmax(std::fabs(want), std::fabs(got)));
      if (std::fabs(got - want) > tolerance * magnitude) {
        Fail(std::string(what) + ": gradient element differs");
        return;
      }
    }
  }
}

nanochat_optim_config TinyOptimizerConfig() {
  nanochat_optim_config config;
  std::memset(&config, 0, sizeof(config));
  config.unembedding_lr = 0.004f;
  config.embedding_lr = 0.2f;
  config.matrix_lr = 0.02f;
  config.scalar_lr = 0.5f;
  config.weight_decay = 0.0f;
  config.muon_ns_steps = 5;
  config.muon_beta2 = 0.9f;
  config.adam_eps = 1e-10f;
  config.clip = 1.0f;
  config.num_iterations = 10;
  config.warmup_steps = 1;
  config.warmdown_ratio = 0.65f;
  config.final_lr_frac = 0.0f;
  config.weight_decay_base = 0.0f;
  config.muon_momentum_warmup_steps = 4.0f;
  config.muon_momentum_start = 0.85f;
  config.muon_momentum_peak = 0.97f;
  config.muon_momentum_final = 0.90f;
  return config;
}

// Model, parameter views, the forward/backward graph, the score and generate
// copies, and the checkpoint round trip.
nanochat_model* CheckModel() {
  // A width below the gate channels is rejected instead of reading out of
  // bounds in the smear and value gates.
  nanochat_config narrow = TinyConfig();
  narrow.hidden_dim = 8;
  Check("model: a narrow hidden_dim is rejected",
        nanochat_model_create(&narrow, /*seed=*/0) == nullptr);

  const nanochat_config config = TinyConfig();
  nanochat_model* model = nanochat_model_create(&config, /*seed=*/2024);
  Check("model: create returns a handle", model != nullptr);
  if (model == nullptr) return nullptr;

  Check("model: version is set", std::strlen(nanochat_version()) > 0);

  const int count = nanochat_param_count(model);
  Check("model: parameter count is positive", count > 0);

  nanochat_param param;
  Check("model: param_info reads the first view",
        nanochat_param_info(model, 0, &param) == 0);
  Check("model: param_info names the view",
        param.name != nullptr && param.name[0] != '\0');
  Check("model: param_info gives a value pointer", param.value != nullptr);
  Check("model: param_info rejects a bad index",
        nanochat_param_info(model, count, &param) != 0);

  // The backend and device description. The backend names the linked kernels,
  // and a CUDA backend must report a device.
  const char* backend = nanochat_backend();
  Check("device: backend is cpu or cuda",
        std::strcmp(backend, "cpu") == 0 || std::strcmp(backend, "cuda") == 0);
  const int element_size = nanochat_compute_type_size();
  Check("device: compute type size is 2 or 4",
        element_size == 2 || element_size == 4);
  nanochat_device device;
  std::memset(&device, 0, sizeof(device));
  nanochat_device_info(&device);
  Check("device: name is set",
        device.device_name != nullptr && device.device_name[0] != '\0');
  Check("device: cuda names a device",
        std::strcmp(backend, "cuda") != 0 ||
            std::strcmp(device.device_name, "cpu") != 0);

  // The host-safe parameter copy. It is the only correct way to read or write
  // a parameter buffer on a device backend, because `param.value` points into
  // device memory there.
  const std::int64_t param_count = param.count;
  const std::size_t param_bytes = static_cast<std::size_t>(param_count) *
                                  static_cast<std::size_t>(element_size);
  std::vector<unsigned char> value(param_bytes);
  std::vector<unsigned char> value_copy(param_bytes);
  Check("param: read returns the element count",
        nanochat_param_read(model, 0, /*grad=*/0, 0, param_count,
                            value.data()) == param_count);
  Check("param: write returns the element count",
        nanochat_param_write(model, 0, /*grad=*/0, 0, param_count,
                             value.data()) == param_count);
  Check("param: read back returns the element count",
        nanochat_param_read(model, 0, /*grad=*/0, 0, param_count,
                            value_copy.data()) == param_count);
  Check("param: the copy round trip preserves the bytes", value == value_copy);
  Check("param: read rejects a bad range",
        nanochat_param_read(model, 0, /*grad=*/0, 0, param_count + 1,
                            value.data()) < 0);
  Check("param: read rejects a bad index",
        nanochat_param_read(model, count, /*grad=*/0, 0, 1, value.data()) < 0);

  const int batch = 2;
  const int seq = 4;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  const std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);

  const float loss =
      nanochat_forward_loss(model, tokens.data(), targets.data(), batch, seq);
  Check("forward_loss: finite", std::isfinite(loss));

  nanochat_backward(model);

  // The optimizer consumes the gradients from the backward pass.
  const nanochat_optim_config optim_config = TinyOptimizerConfig();
  nanochat_optim* optimizer = nanochat_optim_create(model, &optim_config);
  Check("optim: create returns a handle", optimizer != nullptr);
  if (optimizer != nullptr) {
    nanochat_optim_step(optimizer, 1);
    Check("optim: grad norm is finite",
          std::isfinite(nanochat_optim_grad_norm(optimizer)));
    nanochat_optim_free(optimizer);
  }

  nanochat_zero_grad(model);

  // The named optimizer preset and the runtime metrics. The preset returns the
  // same fields as `TrackThreeBaseline` (docs/client-runtime-design.md section
  // 20.4).
  nanochat_optim_config preset;
  std::memset(&preset, 0, sizeof(preset));
  Check("preset: track3 returns ok",
        nanochat_optim_config_preset("track3", &preset) == NANOCHAT_STATUS_OK);
  Check("preset: track3 sets the embedding rate",
        std::fabs(preset.embedding_lr - 0.7f) < 1e-6f);
  Check("preset: track3 sets the warmdown ratio",
        std::fabs(preset.warmdown_ratio - 0.7f) < 1e-6f);
  Check("preset: an unknown name fails",
        nanochat_optim_config_preset("nope", &preset) != NANOCHAT_STATUS_OK);

  nanochat_mesh* mesh = nanochat_mesh_create(nullptr, 1);
  Check("mesh: create returns a handle", mesh != nullptr);
  if (mesh != nullptr) {
    nanochat_runtime* runtime =
        nanochat_runtime_create(mesh, &config, &optim_config, batch, 1, seq, 7);
    Check("runtime: create returns a handle", runtime != nullptr);
    if (runtime != nullptr) {
      const float step_loss =
          nanochat_runtime_train_step(runtime, tokens.data(), targets.data());
      Check("runtime: train_step is finite", std::isfinite(step_loss));
      nanochat_metrics metrics;
      std::memset(&metrics, 0, sizeof(metrics));
      Check("runtime: metrics returns ok",
            nanochat_runtime_metrics(runtime, &metrics) == NANOCHAT_STATUS_OK);
      Check("runtime: one loss per device", metrics.device_count == 1);
      Check("runtime: the loss buffer is set", metrics.loss != nullptr);
      Check("runtime: the step time is positive", metrics.step_seconds > 0.0f);
      std::uint64_t digest = 0;
      Check("runtime: digest returns ok",
            nanochat_runtime_param_digest(runtime, 0, &digest) ==
                NANOCHAT_STATUS_OK);
      Check("runtime: digest is set", digest != 0);
      Check("runtime: the step advances", nanochat_runtime_step(runtime) == 1);
      nanochat_runtime_free(runtime);
    }
    nanochat_mesh_free(mesh);
  }

  // ScoreBatch copies each ScoreResult to the flat C struct.
  std::vector<int> lengths = {seq, seq};
  std::vector<nanochat_focus> focus(static_cast<std::size_t>(batch));
  std::memset(focus.data(), 0, focus.size() * sizeof(nanochat_focus));
  const int focus_ids[3] = {1, 2, 3};
  focus[0].position = 1;
  focus[0].ids = focus_ids;
  focus[0].count = 3;
  focus[1].position = -1;
  std::vector<nanochat_score_result> score(static_cast<std::size_t>(batch));
  nanochat_score_batch(model, tokens.data(), batch, seq, lengths.data(),
                       focus.data(), score.data());
  Check("score_batch: nll buffer", score[0].nll != nullptr);
  Check("score_batch: argmax buffer", score[0].argmax != nullptr);
  Check("score_batch: focus buffer", score[0].focus_logits != nullptr);

  // GenerateBatch copies each GeneratedSequence to the flat C struct.
  const int prompt[2] = {1, 2};
  nanochat_generate_params params;
  std::memset(&params, 0, sizeof(params));
  params.num_samples = 2;
  params.max_tokens = 3;
  params.temperature = 0.0f;
  params.top_k = 0;
  params.seed = 42;
  params.stop_id = -1;
  params.bos_id = -1;
  params.stop_ids = nullptr;
  nanochat_sequences sequences;
  std::memset(&sequences, 0, sizeof(sequences));
  nanochat_generate(model, prompt, 2, &params, &sequences);
  Check("generate: row count", sequences.count == 2);
  Check("generate: token buffer", sequences.tokens != nullptr);
  Check("generate: mask buffer", sequences.mask != nullptr);
  Check("generate: lengths buffer", sequences.lengths != nullptr);
  Check("generate: offsets buffer", sequences.offsets != nullptr);
  Check("generate: rows are nonempty",
        sequences.lengths != nullptr && sequences.lengths[0] > 0);

  // The multi-prompt entry reuses the same staging buffers. Two prompts and
  // two samples produce four rows in prompt-major order.
  const int prompts[4] = {1, 2, 3, 4};
  nanochat_generate_multi_params multi_params;
  std::memset(&multi_params, 0, sizeof(multi_params));
  multi_params.num_prompts = 2;
  multi_params.prompt_len = 2;
  multi_params.row_stops = nullptr;
  multi_params.params = params;
  nanochat_sequences multi_sequences;
  std::memset(&multi_sequences, 0, sizeof(multi_sequences));
  nanochat_generate_multi(model, prompts, &multi_params, &multi_sequences);
  Check("generate_multi: row count", multi_sequences.count == 4);
  Check("generate_multi: token buffer", multi_sequences.tokens != nullptr);
  Check("generate_multi: lengths buffer", multi_sequences.lengths != nullptr);
  Check("generate_multi: offsets buffer", multi_sequences.offsets != nullptr);
  Check("generate_multi: rows are nonempty",
        multi_sequences.lengths != nullptr && multi_sequences.lengths[0] > 0);

  // Checkpoint round trip through the C surface.
  const std::string path = TempPath("bindings_capi.nchkpt");
  nanochat_save(model, path.c_str());
  nanochat_load(model, path.c_str());

  return model;
}

// The harness `NCHKPT01` container loads through `nanochat_load`
// (docs/model.md). A writer model saves the name-keyed container; a reader
// model with a different seed loads it and must match a fresh model that uses
// the writer's seed. The config turns the value embeddings off, so the test
// also covers the Track 3 shape (docs/distributed-design.md section 8).
void CheckHarnessCheckpoint() {
  nanochat_config c_config;
  std::memset(&c_config, 0, sizeof(c_config));
  c_config.num_layers = 2;
  c_config.num_heads = 2;
  c_config.num_kv_heads = 1;
  c_config.hidden_dim = 32;
  c_config.seq_len = 8;
  c_config.vocab_size = 512;
  c_config.padded_vocab_size = 512;
  c_config.rope_base = 10000.0f;
  c_config.window_pattern = "S";
  c_config.value_embedding = 0;

  nanochat::Config cpp_config;
  cpp_config.num_layers = c_config.num_layers;
  cpp_config.num_heads = c_config.num_heads;
  cpp_config.num_kv_heads = c_config.num_kv_heads;
  cpp_config.hidden_dim = c_config.hidden_dim;
  cpp_config.seq_len = c_config.seq_len;
  cpp_config.vocab_size = c_config.vocab_size;
  cpp_config.padded_vocab_size = c_config.padded_vocab_size;
  cpp_config.rope_base = c_config.rope_base;
  cpp_config.window_pattern = c_config.window_pattern;
  cpp_config.value_embedding = false;
  std::unique_ptr<nanochat::Model> writer = nanochat::Model::Create(cpp_config);
  Check("harness: writer creates", writer != nullptr);
  if (writer == nullptr) return;
  writer->InitWeights(101);

  const std::string path = TempPath("bindings_harness.nchkpt01");
  Check("harness: save writes the container",
        nanochat::Checkpointer::SaveModel(*writer, path));

  nanochat_model* same = nanochat_model_create(&c_config, /*seed=*/101);
  nanochat_model* reader = nanochat_model_create(&c_config, /*seed=*/202);
  Check("harness: both handles create", same != nullptr && reader != nullptr);
  if (same != nullptr && reader != nullptr) {
    nanochat_load(reader, path.c_str());
    const int count = nanochat_param_count(same);
    bool equal = count > 0 && count == nanochat_param_count(reader);
    for (int i = 0; i < count && equal; ++i) {
      nanochat_param param;
      if (nanochat_param_info(same, i, &param) != 0) {
        equal = false;
        break;
      }
      std::vector<float> writer_values(static_cast<std::size_t>(param.count));
      std::vector<float> reader_values(static_cast<std::size_t>(param.count));
      const int64_t read_a = nanochat_param_read(
          same, i, /*grad=*/0, 0, param.count, writer_values.data());
      const int64_t read_b = nanochat_param_read(
          reader, i, /*grad=*/0, 0, param.count, reader_values.data());
      equal = read_a == param.count && read_b == param.count &&
              writer_values == reader_values;
    }
    Check("harness: loaded reader matches the writer weights", equal);
  }
  nanochat_model_free(reader);
  nanochat_model_free(same);
}

// The weighted-backward entry point. On a masked batch with non-uniform row
// weights the gradient must match the uniform-scale reference. The backward
// pass is linear in the row scale, so a weight vector plus its complement
// (`1 - w`) accumulates the same gradient as the plain uniform backward. The
// scale argument is a uniform multiplier, so doubling it doubles every
// gradient.
void CheckWeightedBackward() {
  const nanochat_config config = TinyConfig();
  const int batch = 2;
  const int seq = 4;
  const int rows = batch * seq;

  nanochat_model* reference = nanochat_model_create(&config, /*seed=*/77);
  nanochat_model* weighted = nanochat_model_create(&config, /*seed=*/77);
  Check("weighted: create returns two handles",
        reference != nullptr && weighted != nullptr);
  if (reference == nullptr || weighted == nullptr) {
    nanochat_model_free(reference);
    nanochat_model_free(weighted);
    return;
  }

  // A tiny masked batch: rows 1 and 6 are ignored targets, so valid == 6.
  std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);
  targets[1] = -1;
  targets[6] = -1;

  // Non-uniform row weights. The ignored rows carry weight 0, which the
  // classifier also enforces on its own.
  const float pattern[8] = {1.0f, 0.0f, 0.5f, 2.0f, 0.75f, 1.25f, 0.0f, 3.0f};
  std::vector<float> weights(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    weights[static_cast<std::size_t>(i)] = pattern[i];
  }

  // The reference is the uniform backward at scale 1.
  nanochat_forward_loss(reference, tokens.data(), targets.data(), batch, seq);
  nanochat_zero_grad(reference);
  nanochat_backward(reference);

  // The weighted run accumulates `w` and then its complement `1 - w`. The
  // effective row scale is `w + (1 - w) == 1`, the uniform scale.
  std::vector<float> complement(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    complement[static_cast<std::size_t>(i)] =
        1.0f - weights[static_cast<std::size_t>(i)];
  }
  nanochat_forward_loss(weighted, tokens.data(), targets.data(), batch, seq);
  nanochat_zero_grad(weighted);
  nanochat_backward_weighted(weighted, weights.data(), 1.0f);
  nanochat_backward_weighted(weighted, complement.data(), 1.0f);
  CheckScaledGradients(reference, weighted, 1.0,
                       "weighted: non-uniform weights match the uniform "
                       "reference",
                       kGradientTolerance);

  // The scale argument is a uniform multiplier: doubling the scale doubles
  // every gradient.
  nanochat_zero_grad(weighted);
  nanochat_backward_weighted(weighted, weights.data(), 2.0f);
  nanochat_zero_grad(reference);
  nanochat_backward_weighted(reference, weights.data(), 1.0f);
  CheckScaledGradients(reference, weighted, 2.0,
                       "weighted: the scale doubles the gradient",
                       kGradientTolerance);

  // Error handling: a null model and a null weight buffer are errors, not
  // crashes, and each reports the function name. The plain backward first
  // seeds the thread-local error with a different name, so the check cannot
  // pass on a stale message.
  nanochat_backward(nullptr);
  Check("weighted: the error starts as the plain backward",
        std::strstr(nanochat_last_error(), "nanochat_backward_weighted") ==
            nullptr);
  nanochat_backward_weighted(nullptr, weights.data(), 1.0f);
  Check("weighted: a null model sets the error",
        std::strstr(nanochat_last_error(), "nanochat_backward_weighted") !=
            nullptr);
  nanochat_backward(nullptr);
  nanochat_backward_weighted(weighted, nullptr, 1.0f);
  Check("weighted: a null weight buffer sets the error",
        std::strstr(nanochat_last_error(), "nanochat_backward_weighted") !=
            nullptr);

  nanochat_model_free(reference);
  nanochat_model_free(weighted);
}

// The gradient-accumulation and combined-step entry points
// (docs/training-seam.md section 6.1). `nanochat_backward_accumulate` matches
// the plain backward on identical models, and `nanochat_train_step` runs
// forward + backward + one optimizer step; a null optimizer skips the update.
void CheckTrainStep() {
  const nanochat_config config = TinyConfig();
  const int batch = 2;
  const int seq = 4;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  const std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);

  nanochat_model* plain = nanochat_model_create(&config, /*seed=*/91);
  nanochat_model* accum = nanochat_model_create(&config, /*seed=*/91);
  Check("accumulate: create returns two handles",
        plain != nullptr && accum != nullptr);
  if (plain != nullptr && accum != nullptr) {
    nanochat_forward_loss(plain, tokens.data(), targets.data(), batch, seq);
    nanochat_zero_grad(plain);
    nanochat_backward(plain);
    nanochat_forward_loss(accum, tokens.data(), targets.data(), batch, seq);
    nanochat_zero_grad(accum);
    nanochat_backward_accumulate(accum, 1.0f);
    CheckScaledGradients(plain, accum, 1.0,
                         "accumulate: matches the plain backward",
                         kGradientTolerance);
  }
  nanochat_model_free(plain);
  nanochat_model_free(accum);

  // A null model is an error, not a crash, and reports the function name.
  nanochat_backward(nullptr);
  nanochat_backward_accumulate(nullptr, 1.0f);
  Check("accumulate: a null model sets the error",
        std::strstr(nanochat_last_error(), "nanochat_backward_accumulate") !=
            nullptr);

  nanochat_model* trained = nanochat_model_create(&config, /*seed=*/13);
  Check("train_step: create returns a handle", trained != nullptr);
  if (trained != nullptr) {
    // A null optimizer runs the forward and backward without an update.
    const float bare = nanochat_train_step(trained, nullptr, tokens.data(),
                                           targets.data(), batch, seq);
    Check("train_step: a null optimizer returns a finite loss",
          std::isfinite(bare));

    const nanochat_optim_config optim_config = TinyOptimizerConfig();
    nanochat_optim* optimizer = nanochat_optim_create(trained, &optim_config);
    Check("train_step: create returns a handle", optimizer != nullptr);
    if (optimizer != nullptr) {
      const float loss = nanochat_train_step(trained, optimizer, tokens.data(),
                                             targets.data(), batch, seq);
      Check("train_step: returns a finite loss", std::isfinite(loss));
      nanochat_optim_free(optimizer);
    }

    // A null model is an error, not a crash, and reports the function name.
    nanochat_train_step(nullptr, nullptr, tokens.data(), targets.data(), batch,
                        seq);
    Check("train_step: a null model sets the error",
          std::strstr(nanochat_last_error(), "nanochat_train_step") != nullptr);
    nanochat_model_free(trained);
  }
}

// The reinforcement-learning step (docs/training-seam.md section 5.7,
// docs/post-training.md section 5.2). `nanochat_rl_step` runs `ForwardLoss`,
// then `BackwardWeighted`, then `Optimizer::Step`, and the divisor
// `num_valid * num_passes * examples_per_rank` is computed in C++. These
// checks prove the divisor is applied, that the two normalization inputs
// scale the gradient, that the step starts from a fresh gradient, and that the
// optimizer step runs.
void CheckRlStep() {
  const nanochat_config config = TinyConfig();
  const int batch = 2;
  const int seq = 4;
  const int rows = batch * seq;
  const std::vector<int> tokens = MakeTokens(batch, seq, config.vocab_size);
  std::vector<int> targets = MakeTargets(batch, seq, config.vocab_size);
  targets[1] = -1;  // valid == rows - 2 == 6
  targets[6] = -1;

  // A non-uniform per-row advantage. The ignored rows carry weight 0, which
  // the classifier also enforces on its own.
  const float pattern[8] = {0.5f, 0.0f,   1.5f, -0.75f,
                            2.0f, -1.25f, 0.0f, 0.25f};
  std::vector<float> advantages(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    advantages[static_cast<std::size_t>(i)] = pattern[i];
  }

  // The reference gradient: the weighted backward at scale 1, which is the
  // RL step with num_passes == examples_per_rank == 1. At those values the
  // divisor is num_valid, so the RL scale is 1 and the two must agree.
  nanochat_model* reference = nanochat_model_create(&config, /*seed=*/55);
  Check("rl_step: create returns a reference handle", reference != nullptr);
  if (reference != nullptr) {
    nanochat_forward_loss(reference, tokens.data(), targets.data(), batch, seq);
    nanochat_zero_grad(reference);
    nanochat_backward_weighted(reference, advantages.data(), 1.0f);
  }

  nanochat_model* actual = nanochat_model_create(&config, /*seed=*/55);
  Check("rl_step: create returns a handle", actual != nullptr);
  if (reference != nullptr && actual != nullptr) {
    const float loss =
        nanochat_rl_step(actual, nullptr, tokens.data(), targets.data(),
                         advantages.data(), batch, seq, /*num_passes=*/1,
                         /*examples_per_rank=*/1, /*step=*/1);
    Check("rl_step: returns a finite loss", std::isfinite(loss));
    CheckScaledGradients(reference, actual, 1.0,
                         "rl_step: the singular divisor matches the weighted "
                         "backward",
                         kGradientTolerance);

    // `examples_per_rank` enters the divisor: a value of 4 quarters every
    // gradient. Python never computes this factor.
    nanochat_rl_step(actual, nullptr, tokens.data(), targets.data(),
                     advantages.data(), batch, seq, /*num_passes=*/1,
                     /*examples_per_rank=*/4, /*step=*/1);
    CheckScaledGradients(reference, actual, 0.25,
                         "rl_step: examples_per_rank divides the gradient",
                         kGradientTolerance);

    // `num_passes` enters the divisor too: a value of 2 halves every gradient.
    nanochat_rl_step(actual, nullptr, tokens.data(), targets.data(),
                     advantages.data(), batch, seq, /*num_passes=*/2,
                     /*examples_per_rank=*/1, /*step=*/1);
    CheckScaledGradients(reference, actual, 0.5,
                         "rl_step: num_passes divides the gradient",
                         kGradientTolerance);
  }

  // The optimizer step runs: a parameter changes with a non-null optimizer and
  // stays put with a null one.
  const nanochat_optim_config optim_config = TinyOptimizerConfig();
  nanochat_model* held = nanochat_model_create(&config, /*seed=*/55);
  Check("rl_step: create returns a held handle", held != nullptr);
  if (held != nullptr) {
    nanochat_param param;
    std::memset(&param, 0, sizeof(param));
    Check("rl_step: param_info reads the first view",
          nanochat_param_info(held, 0, &param) == 0);
    const int element_size = nanochat_compute_type_size();
    const std::size_t bytes = static_cast<std::size_t>(param.count) *
                              static_cast<std::size_t>(element_size);
    std::vector<unsigned char> before(bytes);
    std::vector<unsigned char> after(bytes);
    Check("rl_step: read the initial value",
          nanochat_param_read(held, 0, /*grad=*/0, 0, param.count,
                              before.data()) == param.count);

    // A null optimizer runs the step without an update.
    nanochat_rl_step(held, nullptr, tokens.data(), targets.data(),
                     advantages.data(), batch, seq, 1, 1, 1);
    Check("rl_step: read the unchanged value",
          nanochat_param_read(held, 0, /*grad=*/0, 0, param.count,
                              after.data()) == param.count);
    Check("rl_step: a null optimizer leaves the parameters unchanged",
          before == after);

    nanochat_optim* optimizer = nanochat_optim_create(held, &optim_config);
    Check("rl_step: create returns an optimizer", optimizer != nullptr);
    if (optimizer != nullptr) {
      const float loss =
          nanochat_rl_step(held, optimizer, tokens.data(), targets.data(),
                           advantages.data(), batch, seq, 1, 1, /*step=*/1);
      Check("rl_step: a non-null optimizer returns a finite loss",
            std::isfinite(loss));
      Check("rl_step: a non-null optimizer reports a finite grad norm",
            std::isfinite(nanochat_optim_grad_norm(optimizer)));
      Check("rl_step: read the stepped value",
            nanochat_param_read(held, 0, /*grad=*/0, 0, param.count,
                                after.data()) == param.count);
      Check("rl_step: the optimizer updates a parameter", before != after);
      nanochat_optim_free(optimizer);
    }
    nanochat_model_free(held);
  }

  // Error handling: a null pointer and a bad normalization input are errors,
  // not crashes, and each reports the function name. The plain backward first
  // seeds the thread-local error with a different name, so the check cannot
  // pass on a stale message.
  nanochat_backward(nullptr);
  nanochat_rl_step(nullptr, nullptr, tokens.data(), targets.data(),
                   advantages.data(), batch, seq, 1, 1, 1);
  Check("rl_step: a null model sets the error",
        std::strstr(nanochat_last_error(), "nanochat_rl_step") != nullptr);
  nanochat_rl_step(actual, nullptr, tokens.data(), targets.data(), nullptr,
                   batch, seq, 1, 1, 1);
  Check("rl_step: a null advantage buffer sets the error",
        std::strstr(nanochat_last_error(), "nanochat_rl_step") != nullptr);
  nanochat_rl_step(actual, nullptr, tokens.data(), targets.data(),
                   advantages.data(), batch, seq, /*num_passes=*/0, 1, 1);
  Check("rl_step: a zero num_passes sets the error",
        std::strstr(nanochat_last_error(), "nanochat_rl_step") != nullptr);

  nanochat_model_free(reference);
  nanochat_model_free(actual);
}

// Tokenizer, encode/decode, the document loader, and EvalBpb.
void CheckTokenizerAndLoader(const char* parquet_path,
                             const char* tokenizer_path,
                             nanochat_model* model) {
  nanochat_tokenizer* tokenizer = nanochat_tokenizer_load(tokenizer_path);
  Check("tokenizer: load returns a handle", tokenizer != nullptr);
  if (tokenizer == nullptr) return;

  const int needed = nanochat_encode(tokenizer, "hello", nullptr, 0);
  Check("encode: reports a positive length", needed > 0);
  std::vector<int> ids(static_cast<std::size_t>(needed));
  Check("encode: fills the buffer",
        nanochat_encode(tokenizer, "hello", ids.data(), needed) == needed);

  const int text_bytes =
      nanochat_decode(tokenizer, ids.data(), needed, nullptr, 0);
  Check("decode: reports a positive length", text_bytes > 0);
  std::vector<char> text(static_cast<std::size_t>(text_bytes) + 1, '\0');
  Check("decode: fills the buffer",
        nanochat_decode(tokenizer, ids.data(), needed, text.data(),
                        text_bytes + 1) == text_bytes);

  const char* parquet[1] = {parquet_path};
  nanochat_loader* loader =
      nanochat_loader_create(parquet, 1, "text", tokenizer, /*batch=*/1,
                             /*seq=*/4, /*seed=*/7, /*tokenizer_threads=*/1,
                             /*document_buffer=*/16);
  Check("loader: create returns a handle", loader != nullptr);
  if (loader != nullptr) {
    if (model != nullptr) {
      const float bpb = nanochat_eval_bpb(model, loader, /*steps=*/1);
      Check("eval_bpb: finite", std::isfinite(bpb));
    }
    std::vector<int> tokens(static_cast<std::size_t>(1 * 4));
    std::vector<int> targets(tokens.size());
    const int next =
        nanochat_loader_next(loader, tokens.data(), targets.data());
    Check("loader: next reports a batch or the end", next >= 0);
    Check("loader: next shifts the targets",
          next <= 0 || targets[0] == tokens[1]);
    int vocab = 0;
    const std::uint8_t* bytes = nanochat_loader_token_bytes(loader, &vocab);
    Check("loader: token bytes are available", bytes != nullptr);
    Check("loader: vocab is positive", vocab > 0);
    nanochat_loader_free(loader);
  }

  nanochat_tokenizer_free(tokenizer);
}

}  // namespace

// The checkpoint container surface (docs/model.md). One write call and one
// read call round-trip every storage dtype, the attribute block, and both
// sections, so a host tool can reach the layout without repeating it.
void CheckCheckpointContainer() {
  const char* name = nullptr;
  int32_t code = 0;
  int32_t size = 0;
  Check(
      "checkpoint dtype fp32",
      nanochat_checkpoint_dtype(0, &name, &code, &size) == NANOCHAT_STATUS_OK &&
          name != nullptr && std::strcmp(name, "float32") == 0 && code == 0 &&
          size == 4);
  Check(
      "checkpoint dtype bf16",
      nanochat_checkpoint_dtype(1, &name, &code, &size) == NANOCHAT_STATUS_OK &&
          name != nullptr && std::strcmp(name, "bfloat16") == 0 && code == 1 &&
          size == 2);
  Check(
      "checkpoint dtype int8",
      nanochat_checkpoint_dtype(2, &name, &code, &size) == NANOCHAT_STATUS_OK &&
          name != nullptr && std::strcmp(name, "int8") == 0 && code == 2 &&
          size == 1);
  Check(
      "checkpoint dtype ends at the last entry",
      nanochat_checkpoint_dtype(3, &name, &code, &size) == NANOCHAT_STATUS_OK &&
          name == nullptr);
  Check("checkpoint dtype rejects a null out pointer",
        nanochat_checkpoint_dtype(0, nullptr, &code, &size) ==
            NANOCHAT_STATUS_ERROR);

  const float weights[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::int64_t weight_shape[2] = {2, 3};
  const std::uint16_t bias_bits[2] = {0x3f80u, 0x4000u};  // 1.0 and 2.0
  const std::int64_t bias_shape[1] = {2};
  const std::uint8_t quantized[6] = {1, 2, 3, 4, 5, 6};
  const std::int64_t quantized_shape[2] = {2, 3};
  const float scale = 0.5f;
  const std::int64_t scale_shape[1] = {1};
  const float momentum[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  const std::int64_t momentum_shape[1] = {4};

  const nanochat_record scale_record = {
      "quant.scale", 0, 1, scale_shape, &scale, sizeof(scale), 0, nullptr};
  const nanochat_record parameters[3] = {
      {"weight", 0, 2, weight_shape, weights, sizeof(weights), 0, nullptr},
      {"bias", 1, 1, bias_shape, bias_bits, sizeof(bias_bits), 0, nullptr},
      {"quantized", 2, 2, quantized_shape, quantized, sizeof(quantized), 1,
       &scale_record},
  };
  const nanochat_record optimizer[1] = {
      {"adamw.lm_head.m", 0, 1, momentum_shape, momentum, sizeof(momentum), 0,
       nullptr},
  };

  const std::string path = TempPath("capi_container.ckpt");
  Check("checkpoint write",
        nanochat_checkpoint_write(path.c_str(), parameters, 3, optimizer, 1) ==
            NANOCHAT_STATUS_OK);

  nanochat_checkpoint* owner = nullptr;
  const nanochat_record* records = nullptr;
  int64_t record_count = 0;
  const nanochat_record* state = nullptr;
  int64_t state_count = 0;
  Check("checkpoint read",
        nanochat_checkpoint_read(path.c_str(), &owner, &records, &record_count,
                                 &state, &state_count) == NANOCHAT_STATUS_OK);
  Check("checkpoint owner", owner != nullptr);
  if (owner == nullptr) return;

  Check("checkpoint parameter count", record_count == 3);
  Check("checkpoint optimizer count", state_count == 1);

  const nanochat_record& weight = records[0];
  Check("checkpoint record name",
        weight.name != nullptr && std::strcmp(weight.name, "weight") == 0);
  Check("checkpoint record dtype", weight.dtype == 0);
  Check("checkpoint record shape", weight.ndim == 2 && weight.dims != nullptr &&
                                       weight.dims[0] == 2 &&
                                       weight.dims[1] == 3);
  Check("checkpoint record payload",
        weight.bytes == static_cast<std::int64_t>(sizeof(weights)) &&
            std::memcmp(weight.data, weights, sizeof(weights)) == 0);
  Check("checkpoint record has no attribute", weight.attribute_count == 0);

  Check("checkpoint bf16 dtype",
        records[1].dtype == 1 && records[1].bytes == 4);

  const nanochat_record& quantized_record = records[2];
  Check("checkpoint int8 attribute count",
        quantized_record.attribute_count == 1);
  Check("checkpoint attribute pointer", quantized_record.attributes != nullptr);
  if (quantized_record.attributes != nullptr) {
    const nanochat_record& attribute = quantized_record.attributes[0];
    Check("checkpoint attribute name",
          attribute.name != nullptr &&
              std::strcmp(attribute.name, "quant.scale") == 0);
    Check("checkpoint attribute payload",
          attribute.bytes == static_cast<std::int64_t>(sizeof(scale)) &&
              std::memcmp(attribute.data, &scale, sizeof(scale)) == 0);
  }
  Check("checkpoint optimizer record",
        state[0].name != nullptr &&
            std::strcmp(state[0].name, "adamw.lm_head.m") == 0);
  nanochat_checkpoint_free(owner);
  nanochat_checkpoint_free(nullptr);
}

// A version 1 header loads in no build, because the container is version 0.
void CheckCheckpointRejectsVersionOne() {
  const std::string path = TempPath("capi_container_bad.ckpt");
  std::FILE* file = std::fopen(path.c_str(), "wb");
  Check("checkpoint bad file opens", file != nullptr);
  if (file != nullptr) {
    const char bytes[16] = {'N', 'C', 'H', 'K', 'P', 'T', '0', '1',
                            1,   0,   0,   0,   0,   0,   0,   0};
    std::fwrite(bytes, 1, sizeof(bytes), file);
    std::fclose(file);
  }
  nanochat_checkpoint* owner = nullptr;
  const nanochat_record* records = nullptr;
  int64_t record_count = 0;
  const nanochat_record* state = nullptr;
  int64_t state_count = 0;
  Check(
      "checkpoint rejects version 1",
      nanochat_checkpoint_read(path.c_str(), &owner, &records, &record_count,
                               &state, &state_count) == NANOCHAT_STATUS_ERROR);
  Check("checkpoint version 1 leaves no owner", owner == nullptr);
  nanochat_checkpoint_free(owner);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <text.parquet> <tokenizer.nctoken>\n",
                 argv[0]);
    return 1;
  }

  CheckSandboxRule();
  CheckCheckpointContainer();
  CheckCheckpointRejectsVersionOne();
  CheckDeviceEnumeration();
  CheckParams();
  CheckHarnessCheckpoint();
  CheckWeightedBackward();
  CheckTrainStep();
  CheckRlStep();

  nanochat_model* model = CheckModel();
  if (model != nullptr) {
    CheckTokenizerAndLoader(argv[1], argv[2], model);
    nanochat_model_free(model);
  } else {
    CheckTokenizerAndLoader(argv[1], argv[2], nullptr);
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "nanochat_capi_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("nanochat_capi_test: ok\n");
  return 0;
}
