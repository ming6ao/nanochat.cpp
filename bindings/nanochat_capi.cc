// The C application binary interface for the in-process Python API
// (docs/python.md section 4). This translation unit is the whole C
// surface: every function in `nanochat/capi.h` is defined here in C++.
//
// Three rules shape the code:
//
//  1. No C++ type crosses the boundary. Opaque handles hide `Model`,
//     `Optimizer`, `DataLoader`, and `Tokenizer`. The flat C structs copy the
//     C++ structs in exactly one place each.
//  2. No C++ exception crosses the boundary. Every entry point catches
//     `std::exception` and any other exception, stores the message for the
//     current thread, and returns a safe value. `nanochat_last_error` reports
//     that message.
//  3. Every `create` function returns one owned handle. The matching free
//     function releases it.
//
// The library links the workflow and the public headers only. It adds no CUDA
// dependency, so the same shim serves the reference backend and the CUDA
// backend (chosen by the build configuration of `//src:model`).

#include "nanochat/capi.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/mfu.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "nanochat/tokenizer.h"
#include "src/parquet/reader.h"

namespace {

// The C ABI version. Bump this string on every change to the surface or to a
// mirrored struct; docs/python.md section 13 names that rule.
constexpr char kNanochatVersion[] = "0.2.0";

// The message for the current thread. A `thread_local` string keeps the
// pointer from `nanochat_last_error` valid until the next failure on this
// thread.
thread_local std::string g_last_error;

void SetLastError(const char* function, const std::string& detail) {
  g_last_error = std::string(function) + ": " + detail;
}

// Runs `body` and returns its value. On an exception the function records the
// message and returns `fallback`. Use this form for every value-returning
// entry point.
template <typename T, typename F>
T Guard(const char* function, T fallback, F&& body) {
  try {
    return body();
  } catch (const std::exception& error) {
    SetLastError(function, error.what());
    return fallback;
  } catch (...) {
    SetLastError(function, "unknown C++ exception");
    return fallback;
  }
}

// Runs `body` for a void entry point. On an exception the function records the
// message and returns.
template <typename F>
void GuardVoid(const char* function, F&& body) {
  try {
    body();
  } catch (const std::exception& error) {
    SetLastError(function, error.what());
  } catch (...) {
    SetLastError(function, "unknown C++ exception");
  }
}

// True when the environment satisfies the sandbox rule of
// docs/python.md section 2.1: a defined `NANOCHAT_SANDBOX`, a
// `NANOCHAT_SANDBOX_BACKEND` of `none`, or a true `NANOCHAT_ALLOW_UNSANDBOXED`.
bool SandboxAllowed() {
  if (*nanochat::SandboxProfile() != '\0') return true;
  const char* backend = std::getenv("NANOCHAT_SANDBOX_BACKEND");
  if (backend != nullptr && std::strcmp(backend, "none") == 0) return true;
  return nanochat::SandboxOverrideSet();
}

// Converts the flat C config to `nanochat::Config`. This is the one place the
// conversion happens (docs/python.md section 4.3).
nanochat::Config ToConfig(const nanochat_config& source) {
  nanochat::Config config;
  config.num_layers = source.num_layers;
  config.num_heads = source.num_heads;
  config.num_kv_heads = source.num_kv_heads;
  config.hidden_dim = source.hidden_dim;
  config.seq_len = source.seq_len;
  config.vocab_size = source.vocab_size;
  config.padded_vocab_size = source.padded_vocab_size;
  config.rope_base = source.rope_base;
  config.window_pattern =
      source.window_pattern != nullptr ? source.window_pattern : "";
  return config;
}

// Rejects a config that cannot build a model graph.
void ValidateConfig(const nanochat::Config& config) {
  if (config.num_layers <= 0 || config.num_heads <= 0 ||
      config.num_kv_heads <= 0 || config.hidden_dim <= 0 ||
      config.seq_len <= 0 || config.vocab_size <= 0 ||
      config.padded_vocab_size <= 0) {
    throw std::invalid_argument("config fields must be positive");
  }
  // The ResFormer smear and value gates read the first 24 and 12 channels of a
  // hidden row (ops::kSmearChannels and ops::kVeGateChannels). A narrower row
  // would read out of bounds, so reject it instead of computing NaN.
  if (config.hidden_dim < 24) {
    throw std::invalid_argument(
        "hidden_dim must be at least 24 for the "
        "smear and value gates");
  }
  if (config.hidden_dim % config.num_heads != 0) {
    throw std::invalid_argument("hidden_dim must be divisible by num_heads");
  }
  if (config.padded_vocab_size < config.vocab_size) {
    throw std::invalid_argument("padded_vocab_size must be >= vocab_size");
  }
}

// Converts the flat C optimizer config to the `OptimizerConfig` and
// `SchedulerConfig` pair. The `nanochat_optim_config` struct holds both groups
// in that order (docs/python.md section 4).
void ToOptimizerConfig(const nanochat_optim_config& source,
                       nanochat::OptimizerConfig* optimizer,
                       nanochat::SchedulerConfig* scheduler) {
  optimizer->unembedding_lr = source.unembedding_lr;
  optimizer->embedding_lr = source.embedding_lr;
  optimizer->matrix_lr = source.matrix_lr;
  optimizer->scalar_lr = source.scalar_lr;
  optimizer->weight_decay = source.weight_decay;
  optimizer->muon_ns_steps = source.muon_ns_steps;
  optimizer->muon_beta2 = source.muon_beta2;
  optimizer->adam_eps = source.adam_eps;
  optimizer->clip = source.clip;

  scheduler->num_iterations = source.num_iterations;
  scheduler->warmup_steps = source.warmup_steps;
  scheduler->warmdown_ratio = source.warmdown_ratio;
  scheduler->final_lr_frac = source.final_lr_frac;
  scheduler->weight_decay_base = source.weight_decay_base;
  scheduler->muon_momentum_warmup_steps = source.muon_momentum_warmup_steps;
  scheduler->muon_momentum_start = source.muon_momentum_start;
  scheduler->muon_momentum_peak = source.muon_momentum_peak;
  scheduler->muon_momentum_final = source.muon_momentum_final;
}

}  // namespace

// The opaque handles live at global scope, because `nanochat/capi.h` declares
// them as global struct tags. Each one owns exactly one C++ object.

// A model handle. The model owns the weights, the gradients, and the staging
// buffers of the last `nanochat_score_batch` and `nanochat_generate` call, so
// the returned C pointers stay valid until the next such call on this handle.
struct nanochat_model {
  std::unique_ptr<nanochat::Model> model;
  std::vector<float> score_nll;
  std::vector<int> score_argmax;
  std::vector<float> score_focus_logits;
  std::vector<int> gen_tokens;
  std::vector<std::uint8_t> gen_mask;
  std::vector<int> gen_lengths;
  std::vector<int> gen_offsets;
};

struct nanochat_optim {
  std::unique_ptr<nanochat::Optimizer> optimizer;
};

struct nanochat_loader {
  std::unique_ptr<nanochat::DataLoader> loader;
};

struct nanochat_tokenizer {
  std::unique_ptr<nanochat::Tokenizer> tokenizer;
};

extern "C" {

nanochat_status nanochat_init(void) {
  return Guard("nanochat_init", NANOCHAT_STATUS_ERROR, []() -> nanochat_status {
    if (!SandboxAllowed()) {
      SetLastError(
          "nanochat_init",
          "the library must run inside the sandbox; set "
          "NANOCHAT_SANDBOX_BACKEND=none, or NANOCHAT_SANDBOX=<profile>, or "
          "NANOCHAT_ALLOW_UNSANDBOXED=1 (docs/python.md section 2.1)");
      return NANOCHAT_STATUS_ERROR;
    }
    // The build fixes the precision. Fail fast when the device cannot run it,
    // instead of failing inside the first kernel launch (docs/build.md).
    const nanochat::Caps caps = nanochat::kernels::GetCaps();
    if (!caps.Supports(nanochat::kComputeDType)) {
      const char* precision =
          nanochat::kComputeDType == nanochat::DType::kFp16 ? "fp16" : "fp32";
      SetLastError(
          "nanochat_init",
          std::string("the device does not support the build ") +
              "precision (" + precision + "): " +
              (caps.device_name != nullptr ? caps.device_name : "unknown"));
      return NANOCHAT_STATUS_ERROR;
    }
    return NANOCHAT_STATUS_OK;
  });
}

const char* nanochat_last_error(void) { return g_last_error.c_str(); }

const char* nanochat_version(void) { return kNanochatVersion; }

const char* nanochat_backend(void) {
  return nanochat::kernels::GetCaps().is_device ? "cuda" : "cpu";
}

int nanochat_compute_type_size(void) {
  return static_cast<int>(sizeof(nanochat::ComputeType));
}

void nanochat_device_info(nanochat_device* out) {
  GuardVoid("nanochat_device_info", [&]() {
    if (out == nullptr) throw std::invalid_argument("out is null");
    const nanochat::Caps caps = nanochat::kernels::GetCaps();
    out->device_index = caps.device_index;
    out->compute_major = caps.compute_major;
    out->compute_minor = caps.compute_minor;
    out->total_memory_bytes =
        static_cast<std::int64_t>(caps.total_memory_bytes);
    out->device_name = caps.device_name;
  });
}

nanochat_status nanochat_params_get(const nanochat_config* config,
                                    nanochat_params* out) {
  return Guard(
      "nanochat_params_get", NANOCHAT_STATUS_ERROR, [&]() -> nanochat_status {
        if (config == nullptr || out == nullptr) {
          throw std::invalid_argument(
              "config and out must not be "
              "null");
        }
        const nanochat::Config cpp_config = ToConfig(*config);
        const nanochat::ParamBreakdown counts =
            nanochat::CountParams(cpp_config);
        out->total = counts.total;
        out->transformer_matrices = counts.transformer_matrices;
        out->lm_head = counts.lm_head;
        out->embeddings = counts.embeddings;
        out->scalars = counts.scalars;
        out->flops_per_token = nanochat::EstimateFlopsPerToken(cpp_config);
        return NANOCHAT_STATUS_OK;
      });
}

nanochat_model* nanochat_model_create(const nanochat_config* config,
                                      std::uint64_t seed) {
  return Guard("nanochat_model_create", static_cast<nanochat_model*>(nullptr),
               [&]() -> nanochat_model* {
                 if (config == nullptr) {
                   throw std::invalid_argument("config is null");
                 }
                 const nanochat::Config cpp_config = ToConfig(*config);
                 ValidateConfig(cpp_config);
                 std::unique_ptr<nanochat_model> handle =
                     std::make_unique<nanochat_model>();
                 handle->model = nanochat::Model::Create(cpp_config);
                 if (handle->model == nullptr) {
                   throw std::runtime_error("Model::Create returned null");
                 }
                 handle->model->InitWeights(seed);
                 return handle.release();
               });
}

void nanochat_model_free(nanochat_model* model) { delete model; }

float nanochat_forward_loss(nanochat_model* model, const int* tokens,
                            const int* targets, int batch, int seq) {
  return Guard("nanochat_forward_loss", 0.0f, [&]() -> float {
    if (model == nullptr || tokens == nullptr || targets == nullptr) {
      throw std::invalid_argument("model or token buffer is null");
    }
    if (batch <= 0 || seq <= 0) {
      throw std::invalid_argument("batch and seq must be positive");
    }
    return model->model->ForwardLoss(tokens, targets, batch, seq);
  });
}

void nanochat_backward(nanochat_model* model) {
  GuardVoid("nanochat_backward", [&]() {
    if (model == nullptr) throw std::invalid_argument("model is null");
    model->model->Backward();
  });
}

void nanochat_zero_grad(nanochat_model* model) {
  GuardVoid("nanochat_zero_grad", [&]() {
    if (model == nullptr) throw std::invalid_argument("model is null");
    model->model->ZeroGrad();
  });
}

int nanochat_param_count(nanochat_model* model) {
  return Guard("nanochat_param_count", 0, [&]() -> int {
    if (model == nullptr) throw std::invalid_argument("model is null");
    return static_cast<int>(model->model->params().size());
  });
}

int nanochat_param_info(nanochat_model* model, int index, nanochat_param* out) {
  return Guard("nanochat_param_info", -1, [&]() -> int {
    if (model == nullptr || out == nullptr) {
      throw std::invalid_argument("model or out is null");
    }
    const std::vector<nanochat::ParamView> params = model->model->params();
    if (index < 0 || index >= static_cast<int>(params.size())) return -1;
    const nanochat::ParamView& view = params[static_cast<std::size_t>(index)];
    out->name = view.name;
    out->value = view.value;
    out->grad = view.grad;
    out->count = view.count;
    out->rows = view.rows;
    out->cols = view.cols;
    return 0;
  });
}

// Resolves one parameter buffer and validates the requested element range.
// `grad` selects the gradient when nonzero. The returned pointer addresses
// the model buffer, which is host memory on the CPU backend and device memory
// on the CUDA backend. `host` is the caller buffer; it may be null only for an
// empty range.
nanochat::ComputeType* ResolveParamRange(nanochat_model* model, int index,
                                         int grad, int64_t offset,
                                         int64_t count, const void* host) {
  if (model == nullptr) throw std::invalid_argument("model is null");
  const std::vector<nanochat::ParamView> params = model->model->params();
  if (index < 0 || index >= static_cast<int>(params.size())) {
    throw std::out_of_range("parameter index out of range");
  }
  const nanochat::ParamView& view = params[static_cast<std::size_t>(index)];
  nanochat::ComputeType* buffer = grad != 0 ? view.grad : view.value;
  if (buffer == nullptr) throw std::runtime_error("parameter buffer is null");
  if (host == nullptr && count > 0) {
    throw std::invalid_argument("data is null");
  }
  if (offset < 0 || count < 0 || offset + count > view.count) {
    throw std::out_of_range("parameter range out of bounds");
  }
  return buffer + offset;
}

int64_t nanochat_param_read(nanochat_model* model, int index, int grad,
                            int64_t offset, int64_t count, void* data) {
  return Guard("nanochat_param_read", static_cast<std::int64_t>(-1),
               [&]() -> std::int64_t {
                 nanochat::ComputeType* source =
                     ResolveParamRange(model, index, grad, offset, count, data);
                 if (count == 0) return 0;
                 nanochat::kernels::Memcpy(data, source,
                                           static_cast<std::size_t>(count) *
                                               sizeof(nanochat::ComputeType),
                                           nanochat::CopyDir::kDeviceToHost);
                 return count;
               });
}

int64_t nanochat_param_write(nanochat_model* model, int index, int grad,
                             int64_t offset, int64_t count, const void* data) {
  return Guard("nanochat_param_write", static_cast<std::int64_t>(-1),
               [&]() -> std::int64_t {
                 nanochat::ComputeType* target =
                     ResolveParamRange(model, index, grad, offset, count, data);
                 if (count == 0) return 0;
                 nanochat::kernels::Memcpy(target, data,
                                           static_cast<std::size_t>(count) *
                                               sizeof(nanochat::ComputeType),
                                           nanochat::CopyDir::kHostToDevice);
                 return count;
               });
}

void nanochat_save(nanochat_model* model, const char* path) {
  GuardVoid("nanochat_save", [&]() {
    if (model == nullptr || path == nullptr) {
      throw std::invalid_argument("model or path is null");
    }
    model->model->Save(path);
  });
}

void nanochat_load(nanochat_model* model, const char* path) {
  GuardVoid("nanochat_load", [&]() {
    if (model == nullptr || path == nullptr) {
      throw std::invalid_argument("model or path is null");
    }
    model->model->Load(path);
  });
}

nanochat_optim* nanochat_optim_create(nanochat_model* model,
                                      const nanochat_optim_config* config) {
  return Guard("nanochat_optim_create", static_cast<nanochat_optim*>(nullptr),
               [&]() -> nanochat_optim* {
                 if (model == nullptr || config == nullptr) {
                   throw std::invalid_argument("model or config is null");
                 }
                 nanochat::OptimizerConfig optimizer_config;
                 nanochat::SchedulerConfig scheduler_config;
                 ToOptimizerConfig(*config, &optimizer_config,
                                   &scheduler_config);
                 const nanochat::Scheduler scheduler(scheduler_config);
                 std::unique_ptr<nanochat_optim> handle =
                     std::make_unique<nanochat_optim>();
                 handle->optimizer = nanochat::CreateOptimizer(
                     model->model.get(), optimizer_config, scheduler);
                 if (handle->optimizer == nullptr) {
                   throw std::runtime_error("CreateOptimizer returned null");
                 }
                 return handle.release();
               });
}

void nanochat_optim_step(nanochat_optim* optimizer, int step) {
  GuardVoid("nanochat_optim_step", [&]() {
    if (optimizer == nullptr) {
      throw std::invalid_argument("optimizer is null");
    }
    optimizer->optimizer->Step(step);
  });
}

float nanochat_optim_grad_norm(nanochat_optim* optimizer) {
  return Guard("nanochat_optim_grad_norm", 0.0f, [&]() -> float {
    if (optimizer == nullptr) {
      throw std::invalid_argument("optimizer is null");
    }
    return optimizer->optimizer->GradNorm();
  });
}

void nanochat_optim_free(nanochat_optim* optimizer) { delete optimizer; }

nanochat_loader* nanochat_loader_create(const char** parquet, int count,
                                        const char* text_column,
                                        nanochat_tokenizer* tokenizer,
                                        int batch, int seq, std::uint64_t seed,
                                        int tokenizer_threads,
                                        std::size_t document_buffer) {
  return Guard(
      "nanochat_loader_create", static_cast<nanochat_loader*>(nullptr),
      [&]() -> nanochat_loader* {
        if (parquet == nullptr || count <= 0 || tokenizer == nullptr) {
          throw std::invalid_argument(
              "parquet list, count, or tokenizer is invalid");
        }
        if (batch <= 0 || seq <= 0) {
          throw std::invalid_argument("batch and seq must be positive");
        }
        std::vector<std::string> files;
        files.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
          if (parquet[i] == nullptr) {
            throw std::invalid_argument("parquet path is null");
          }
          files.emplace_back(parquet[i]);
        }
        const std::string column =
            text_column != nullptr ? text_column : "text";
        // The factory opens the dataset again on every call, so the
        // loader can cycle epochs with `Reset`.
        nanochat::DocumentSourceFactory factory = [files,
                                                   column](std::string* error)
            -> std::unique_ptr<nanochat::DocumentSource> {
          return nanochat::OpenParquetSource(files, column, 256, error);
        };
        std::unique_ptr<nanochat_loader> handle =
            std::make_unique<nanochat_loader>();
        handle->loader = std::make_unique<nanochat::DataLoader>(
            std::move(factory), tokenizer->tokenizer.get(), batch, seq, seed,
            tokenizer_threads, document_buffer);
        return handle.release();
      });
}

int nanochat_loader_next(nanochat_loader* loader, int* tokens, int* targets) {
  // A positive value means one batch was written, 0 means the end of the
  // input, and a negative value means an error (the message is in
  // `nanochat_last_error`).
  return Guard("nanochat_loader_next", -1, [&]() -> int {
    if (loader == nullptr || tokens == nullptr || targets == nullptr) {
      throw std::invalid_argument("loader or output buffer is null");
    }
    return loader->loader->Next(tokens, targets) ? 1 : 0;
  });
}

const std::uint8_t* nanochat_loader_token_bytes(nanochat_loader* loader,
                                                int* vocab) {
  return Guard("nanochat_loader_token_bytes",
               static_cast<const std::uint8_t*>(nullptr),
               [&]() -> const std::uint8_t* {
                 if (loader == nullptr || vocab == nullptr) {
                   throw std::invalid_argument("loader or vocab is null");
                 }
                 return loader->loader->token_bytes(vocab);
               });
}

void nanochat_loader_free(nanochat_loader* loader) { delete loader; }

void nanochat_score_batch(nanochat_model* model, const int* tokens, int batch,
                          int seq, const int* lengths,
                          const nanochat_focus* focus,
                          nanochat_score_result* out) {
  GuardVoid("nanochat_score_batch", [&]() {
    if (model == nullptr || tokens == nullptr || out == nullptr) {
      throw std::invalid_argument("model, tokens, or out is null");
    }
    if (batch <= 0 || seq <= 0) {
      throw std::invalid_argument("batch and seq must be positive");
    }
    std::vector<nanochat::ScoreFocus> focus_requests;
    const nanochat::ScoreFocus* focus_ptr = nullptr;
    if (focus != nullptr) {
      focus_requests.reserve(static_cast<std::size_t>(batch));
      for (int i = 0; i < batch; ++i) {
        nanochat::ScoreFocus request;
        request.position = focus[i].position;
        request.ids = focus[i].ids;
        request.count = focus[i].count;
        focus_requests.push_back(request);
      }
      focus_ptr = focus_requests.data();
    }

    std::vector<nanochat::ScoreResult> results;
    nanochat::ScoreBatch(model->model.get(), tokens, batch, seq, lengths,
                         focus_ptr, &results);

    // Flatten every per-row vector into the model-owned staging buffers, then
    // fill the caller array with pointers into those buffers. The two passes
    // keep the pointers valid after the final reallocation.
    model->score_nll.clear();
    model->score_argmax.clear();
    model->score_focus_logits.clear();
    std::vector<std::size_t> nll_offsets(static_cast<std::size_t>(batch));
    std::vector<std::size_t> argmax_offsets(static_cast<std::size_t>(batch));
    std::vector<std::size_t> focus_offsets(static_cast<std::size_t>(batch));
    for (int i = 0; i < batch; ++i) {
      const nanochat::ScoreResult& result =
          results[static_cast<std::size_t>(i)];
      const std::size_t slot = static_cast<std::size_t>(i);
      nll_offsets[slot] = model->score_nll.size();
      model->score_nll.insert(model->score_nll.end(), result.nll.begin(),
                              result.nll.end());
      argmax_offsets[slot] = model->score_argmax.size();
      model->score_argmax.insert(model->score_argmax.end(),
                                 result.argmax.begin(), result.argmax.end());
      focus_offsets[slot] = model->score_focus_logits.size();
      model->score_focus_logits.insert(model->score_focus_logits.end(),
                                       result.focus_logits.begin(),
                                       result.focus_logits.end());
    }
    for (int i = 0; i < batch; ++i) {
      const std::size_t slot = static_cast<std::size_t>(i);
      out[slot].nll = model->score_nll.data() + nll_offsets[slot];
      out[slot].argmax = model->score_argmax.data() + argmax_offsets[slot];
      out[slot].focus_logits =
          model->score_focus_logits.data() + focus_offsets[slot];
    }
  });
}

void nanochat_generate(nanochat_model* model, const int* prompt, int length,
                       const nanochat_generate_params* params,
                       nanochat_sequences* out) {
  GuardVoid("nanochat_generate", [&]() {
    if (model == nullptr || prompt == nullptr || out == nullptr) {
      throw std::invalid_argument("model, prompt, or out is null");
    }
    if (length < 0) throw std::invalid_argument("length must be >= 0");

    nanochat::GenerateParams cpp_params;
    if (params != nullptr) {
      cpp_params.num_samples = params->num_samples;
      cpp_params.max_tokens = params->max_tokens;
      cpp_params.temperature = params->temperature;
      cpp_params.top_k = params->top_k;
      cpp_params.seed = params->seed;
      cpp_params.stop_id = params->stop_id;
      cpp_params.bos_id = params->bos_id;
      cpp_params.stop_ids = params->stop_ids;
    }

    std::vector<nanochat::GeneratedSequence> results;
    nanochat::GenerateBatch(model->model.get(), prompt, length, cpp_params,
                            &results);

    model->gen_tokens.clear();
    model->gen_mask.clear();
    model->gen_lengths.clear();
    model->gen_offsets.clear();
    for (const nanochat::GeneratedSequence& row : results) {
      model->gen_offsets.push_back(static_cast<int>(model->gen_tokens.size()));
      model->gen_lengths.push_back(static_cast<int>(row.tokens.size()));
      model->gen_tokens.insert(model->gen_tokens.end(), row.tokens.begin(),
                               row.tokens.end());
      model->gen_mask.insert(model->gen_mask.end(), row.mask.begin(),
                             row.mask.end());
    }
    out->count = static_cast<int>(results.size());
    out->tokens = model->gen_tokens.data();
    out->mask = model->gen_mask.data();
    out->lengths = model->gen_lengths.data();
    out->offsets = model->gen_offsets.data();
  });
}

float nanochat_eval_bpb(nanochat_model* model, nanochat_loader* loader,
                        int steps) {
  return Guard("nanochat_eval_bpb", 0.0f, [&]() -> float {
    if (model == nullptr || loader == nullptr) {
      throw std::invalid_argument("model or loader is null");
    }
    return nanochat::EvalBpb(model->model.get(), loader->loader.get(), steps);
  });
}

nanochat_tokenizer* nanochat_tokenizer_load(const char* path) {
  return Guard("nanochat_tokenizer_load",
               static_cast<nanochat_tokenizer*>(nullptr),
               [&]() -> nanochat_tokenizer* {
                 if (path == nullptr) {
                   throw std::invalid_argument("path is null");
                 }
                 std::unique_ptr<nanochat_tokenizer> handle =
                     std::make_unique<nanochat_tokenizer>();
                 handle->tokenizer = nanochat::LoadTokenizer(path);
                 if (handle->tokenizer == nullptr) {
                   throw std::runtime_error("LoadTokenizer returned null");
                 }
                 return handle.release();
               });
}

int nanochat_encode(nanochat_tokenizer* tokenizer, const char* text, int* out,
                    int capacity) {
  // Returns the number of ids the encoder produced. When `out` is not null the
  // function copies `min(needed, capacity)` ids into it.
  return Guard("nanochat_encode", -1, [&]() -> int {
    if (tokenizer == nullptr || text == nullptr) {
      throw std::invalid_argument("tokenizer or text is null");
    }
    const std::vector<int> ids = tokenizer->tokenizer->Encode(text);
    const int needed = static_cast<int>(ids.size());
    if (out != nullptr && capacity > 0) {
      const int copied = needed < capacity ? needed : capacity;
      std::memcpy(out, ids.data(),
                  static_cast<std::size_t>(copied) * sizeof(int));
    }
    return needed;
  });
}

int nanochat_decode(nanochat_tokenizer* tokenizer, const int* ids, int count,
                    char* out, int capacity) {
  // Returns the number of bytes the decoder produced. When `out` is not null
  // the function copies `min(needed, capacity)` bytes into it and appends a
  // null terminator when room remains.
  return Guard("nanochat_decode", -1, [&]() -> int {
    if (tokenizer == nullptr || (ids == nullptr && count > 0)) {
      throw std::invalid_argument("tokenizer or ids is null");
    }
    const std::string text = tokenizer->tokenizer->Decode(ids, count);
    const int needed = static_cast<int>(text.size());
    if (out != nullptr && capacity > 0) {
      const int copied = needed < capacity ? needed : capacity;
      std::memcpy(out, text.data(), static_cast<std::size_t>(copied));
      if (copied < capacity) out[copied] = '\0';
    }
    return needed;
  });
}

void nanochat_tokenizer_free(nanochat_tokenizer* tokenizer) {
  delete tokenizer;
}

}  // extern "C"
