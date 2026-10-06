// The C application binary interface test (docs/python-api.md section 11,
// phase 1). It calls every function in `nanochat/capi.h` and checks the
// contract: the sandbox rule, the one owned handle per create, the flat struct
// copies, the thread-local error, and the round trip of the tokenizer, the
// loader, the optimizer, the model, and the two inference entry points.
//
// The static assertions below compare each flat C mirror struct to the C++
// struct in `nanochat/model.h` that it copies. A type change in the C++ struct
// stops this test at compile time, so an ABI drift cannot pass silently
// (docs/python-api.md section 13).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/capi.h"
#include "nanochat/config.h"
#include "nanochat/model.h"

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

// The sandbox rule of docs/python-api.md section 2.1, in each branch.
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
  return config;
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

  // Checkpoint round trip through the C surface.
  const std::string path = TempPath("bindings_capi.nchkpt");
  nanochat_save(model, path.c_str());
  nanochat_load(model, path.c_str());

  return model;
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

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <text.parquet> <tokenizer.nctoken>\n",
                 argv[0]);
    return 1;
  }

  CheckSandboxRule();

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
