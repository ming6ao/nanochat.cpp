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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/data.h"
#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/mesh.h"
#include "nanochat/mfu.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "nanochat/tokenizer.h"
#include "src/mesh_runtime.h"
#include "src/mesh_train.h"
#include "src/parquet/reader.h"
#include "src/rl.h"
#include "src/train.h"

namespace {

// The C ABI version. Bump this string on every change to the surface or to a
// mirrored struct; docs/python.md section 13 names that rule.
constexpr char kNanochatVersion[] = "0.12.0";

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
  config.value_embedding = source.value_embedding != 0;
  return config;
}

// Rejects a config that cannot build a model graph.
void ValidateConfig(const nanochat::Config& config) {
  std::string error;
  if (!config.Validate(&error)) throw std::invalid_argument(error);
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

  optimizer->matrix_optimizer = source.matrix_optimizer;
  optimizer->adam_step_period = source.adam_step_period;
  optimizer->anvil_lr = source.anvil_lr;
  optimizer->anvil_weight_decay = source.anvil_weight_decay;
  optimizer->anvil_momentum = source.anvil_momentum;
  optimizer->anvil_beta2 = source.anvil_beta2;
  optimizer->anvil_fast_beta = source.anvil_fast_beta;
  optimizer->anvil_slow_beta = source.anvil_slow_beta;
  optimizer->anvil_fast_weight = source.anvil_fast_weight;
  optimizer->anvil_engage_step = source.anvil_engage_step;
  optimizer->anvil_num_maps = source.anvil_num_maps;

  scheduler->rail_beta_warmup_steps = source.rail_beta_warmup_steps;
  scheduler->rail_beta_cooldown_steps = source.rail_beta_cooldown_steps;
  scheduler->rail_beta_min = source.rail_beta_min;
  scheduler->rail_beta_max = source.rail_beta_max;
}

// The reverse of `ToOptimizerConfig`: copies a C++ optimizer and schedule into
// the flat C struct. One place owns the field list, so the preset and any later
// reader agree.
void FromOptimizerConfig(const nanochat::OptimizerConfig& optimizer,
                         const nanochat::SchedulerConfig& scheduler,
                         nanochat_optim_config* out) {
  out->unembedding_lr = optimizer.unembedding_lr;
  out->embedding_lr = optimizer.embedding_lr;
  out->matrix_lr = optimizer.matrix_lr;
  out->scalar_lr = optimizer.scalar_lr;
  out->weight_decay = optimizer.weight_decay;
  out->muon_ns_steps = optimizer.muon_ns_steps;
  out->muon_beta2 = optimizer.muon_beta2;
  out->adam_eps = optimizer.adam_eps;
  out->clip = optimizer.clip;

  out->num_iterations = scheduler.num_iterations;
  out->warmup_steps = scheduler.warmup_steps;
  out->warmdown_ratio = scheduler.warmdown_ratio;
  out->final_lr_frac = scheduler.final_lr_frac;
  out->weight_decay_base = scheduler.weight_decay_base;
  out->muon_momentum_warmup_steps = scheduler.muon_momentum_warmup_steps;
  out->muon_momentum_start = scheduler.muon_momentum_start;
  out->muon_momentum_peak = scheduler.muon_momentum_peak;
  out->muon_momentum_final = scheduler.muon_momentum_final;

  out->matrix_optimizer = optimizer.matrix_optimizer;
  out->adam_step_period = optimizer.adam_step_period;
  out->anvil_lr = optimizer.anvil_lr;
  out->anvil_weight_decay = optimizer.anvil_weight_decay;
  out->anvil_momentum = optimizer.anvil_momentum;
  out->anvil_beta2 = optimizer.anvil_beta2;
  out->anvil_fast_beta = optimizer.anvil_fast_beta;
  out->anvil_slow_beta = optimizer.anvil_slow_beta;
  out->anvil_fast_weight = optimizer.anvil_fast_weight;
  out->anvil_engage_step = optimizer.anvil_engage_step;
  out->anvil_num_maps = optimizer.anvil_num_maps;

  out->rail_beta_warmup_steps = scheduler.rail_beta_warmup_steps;
  out->rail_beta_cooldown_steps = scheduler.rail_beta_cooldown_steps;
  out->rail_beta_min = scheduler.rail_beta_min;
  out->rail_beta_max = scheduler.rail_beta_max;
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

struct nanochat_mesh {
  nanochat::MeshSpec mesh;
};

struct nanochat_runtime {
  std::unique_ptr<nanochat::MeshRuntime> runtime;
};

// A checkpoint handle. `nanochat_checkpoint_read` fills the arrays and returns
// the handle, so the caller can release everything with one free. The flat
// arrays point into `checkpoint`, so every field lives as long as the handle.
struct nanochat_checkpoint {
  nanochat::Checkpoint checkpoint;
  std::vector<nanochat_record> records;
  std::vector<nanochat_record> optimizer_state;
  std::vector<nanochat_record> attributes;
};

namespace {

// The replica model of a runtime handle. Throws on a null handle or a bad
// replica index, so the runtime entry points stay free of repeated checks.
nanochat::Model* RuntimeModel(nanochat_runtime* runtime, int replica) {
  if (runtime == nullptr || runtime->runtime == nullptr) {
    throw std::invalid_argument("runtime is null");
  }
  nanochat::Model* model = runtime->runtime->model(replica);
  if (model == nullptr) {
    throw std::out_of_range("the replica index is out of range");
  }
  return model;
}

// The parameter view at `index` on `replica`.
nanochat::ParamView RuntimeParam(nanochat_runtime* runtime, int replica,
                                 int index) {
  const std::vector<nanochat::ParamView> params =
      RuntimeModel(runtime, replica)->params();
  if (index < 0 || index >= static_cast<int>(params.size())) {
    throw std::out_of_range("the parameter index is out of range");
  }
  return params[static_cast<std::size_t>(index)];
}

// Converts the C driver config into the C++ value. The caller keeps owning
// every pointer and the mesh handle, so the value copies each string and the
// mesh.
nanochat::MeshTrainConfig ToMeshTrainConfig(
    const nanochat_mesh_train_config& source) {
  nanochat::MeshTrainConfig config;
  config.model = ToConfig(source.model);
  ValidateConfig(config.model);
  if (source.mesh == nullptr) {
    throw std::invalid_argument("mesh must not be null");
  }
  config.mesh = source.mesh->mesh;
  config.preset = source.preset != nullptr ? source.preset : "track3";
  if (source.optimizer != nullptr) {
    ToOptimizerConfig(*source.optimizer, &config.optimizer, &config.scheduler);
    config.use_preset = false;
  }
  config.warm_start_path =
      source.warm_start_path != nullptr ? source.warm_start_path : "";
  config.batch = source.run.batch;
  config.grad_accum = source.run.grad_accum;
  config.seq = source.run.seq;
  config.seed = source.run.seed;
  config.data_seed = source.data.data_seed;
  config.num_iterations = source.run.num_iterations;
  config.log_every = source.run.log_every;
  config.save_every = source.run.save_every;
  config.resume = source.run.resume != 0;
  config.flops_per_token = source.flops_per_token;
  config.peak_flops = source.peak_flops;
  config.out_dir =
      source.output.out_dir != nullptr ? source.output.out_dir : "";
  config.checkpoint_path = source.output.checkpoint_path != nullptr
                               ? source.output.checkpoint_path
                               : "";
  config.tokenizer_path =
      source.data.tokenizer_path != nullptr ? source.data.tokenizer_path : "";
  config.packed_rows_path = source.data.packed_rows_path != nullptr
                                ? source.data.packed_rows_path
                                : "";
  for (int i = 0; i < source.data.parquet_count; ++i) {
    if (source.data.parquet != nullptr && source.data.parquet[i] != nullptr) {
      config.train_parquet.emplace_back(source.data.parquet[i]);
    }
  }
  return config;
}

}  // namespace

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
      SetLastError(
          "nanochat_init",
          std::string("the device does not support the build ") +
              "precision (fp32): " +
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

int nanochat_device_count(void) {
  return Guard("nanochat_device_count", 0, []() -> int {
    return static_cast<int>(nanochat::EnumerateDevices().size());
  });
}

int nanochat_device_at(int ordinal, nanochat_device* out) {
  return Guard("nanochat_device_at", 1, [&]() -> int {
    if (out == nullptr) throw std::invalid_argument("out is null");
    const std::vector<nanochat::DeviceInfo> devices =
        nanochat::EnumerateDevices();
    if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
      throw std::out_of_range("the device ordinal is out of range");
    }
    const nanochat::DeviceInfo& info =
        devices[static_cast<std::size_t>(ordinal)];
    out->device_index = info.index;
    out->compute_major = info.compute_major;
    out->compute_minor = info.compute_minor;
    out->total_memory_bytes =
        static_cast<std::int64_t>(info.total_memory_bytes);
    out->device_name = info.name;
    return 0;
  });
}

nanochat_mesh* nanochat_mesh_create(const int* indices, int count) {
  return Guard("nanochat_mesh_create", static_cast<nanochat_mesh*>(nullptr),
               [&]() -> nanochat_mesh* {
                 const int n = count < 1 ? 1 : count;
                 std::vector<int> device_indices(static_cast<std::size_t>(n));
                 for (int i = 0; i < n; ++i) {
                   device_indices[static_cast<std::size_t>(i)] =
                       indices != nullptr ? indices[i] : i;
                 }
                 auto handle = std::make_unique<nanochat_mesh>();
                 handle->mesh =
                     nanochat::MeshSpec::Data(std::move(device_indices));
                 return handle.release();
               });
}

void nanochat_mesh_free(nanochat_mesh* mesh) {
  GuardVoid("nanochat_mesh_free", [&]() { delete mesh; });
}

int nanochat_mesh_size(nanochat_mesh* mesh) {
  return Guard("nanochat_mesh_size", 0, [&]() -> int {
    if (mesh == nullptr) throw std::invalid_argument("mesh is null");
    return mesh->mesh.size();
  });
}

nanochat_runtime* nanochat_runtime_create(const nanochat_mesh* mesh,
                                          const nanochat_config* config,
                                          const nanochat_optim_config* optim,
                                          int batch, int grad_accum, int seq,
                                          uint64_t seed) {
  return Guard(
      "nanochat_runtime_create", static_cast<nanochat_runtime*>(nullptr),
      [&]() -> nanochat_runtime* {
        if (mesh == nullptr || config == nullptr) {
          throw std::invalid_argument("mesh and config must not be null");
        }
        nanochat::MeshRuntimeConfig runtime_config;
        runtime_config.model = ToConfig(*config);
        ValidateConfig(runtime_config.model);
        if (optim != nullptr) {
          ToOptimizerConfig(*optim, &runtime_config.optimizer,
                            &runtime_config.scheduler);
        }
        runtime_config.batch = batch;
        runtime_config.grad_accum = grad_accum;
        runtime_config.seq = seq;
        runtime_config.seed = seed;
        auto handle = std::make_unique<nanochat_runtime>();
        handle->runtime = std::make_unique<nanochat::MeshRuntime>(
            mesh->mesh, std::move(runtime_config));
        return handle.release();
      });
}

void nanochat_runtime_free(nanochat_runtime* runtime) {
  GuardVoid("nanochat_runtime_free", [&]() { delete runtime; });
}

int nanochat_runtime_world_size(nanochat_runtime* runtime) {
  return Guard("nanochat_runtime_world_size", 0, [&]() -> int {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    return runtime->runtime->world_size();
  });
}

int nanochat_runtime_step(nanochat_runtime* runtime) {
  return Guard("nanochat_runtime_step", -1, [&]() -> int {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    return runtime->runtime->step();
  });
}

float nanochat_mesh_train_run(const nanochat_mesh_train_config* config,
                              nanochat_mesh_train_summary* out) {
  return Guard("nanochat_mesh_train_run", 0.0f, [&]() -> float {
    if (config == nullptr) throw std::invalid_argument("config is null");
    nanochat::MeshTrainDriver driver(ToMeshTrainConfig(*config));
    const float loss = driver.Run();
    if (out != nullptr) {
      const nanochat::MeshTrainSummary& summary = driver.summary();
      out->steps = summary.steps;
      out->interrupted = summary.interrupted ? 1 : 0;
      out->mean_loss = summary.mean_loss;
      out->steady_mfu = summary.steady_mfu;
      std::snprintf(out->checkpoint, sizeof(out->checkpoint), "%s",
                    summary.checkpoint.c_str());
    }
    return loss;
  });
}

float nanochat_runtime_train_step(nanochat_runtime* runtime, const int* tokens,
                                  const int* targets) {
  return Guard("nanochat_runtime_train_step", 0.0f, [&]() -> float {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (tokens == nullptr || targets == nullptr) {
      throw std::invalid_argument("tokens and targets must not be null");
    }
    return runtime->runtime->TrainStep(tokens, targets);
  });
}

nanochat_status nanochat_runtime_metrics(nanochat_runtime* runtime,
                                         nanochat_metrics* out) {
  return Guard("nanochat_runtime_metrics", NANOCHAT_STATUS_ERROR, [&]() {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (out == nullptr) throw std::invalid_argument("out is null");
    const nanochat::MeshMetrics& metrics = runtime->runtime->metrics();
    out->device_count = static_cast<int>(metrics.loss.size());
    out->loss = metrics.loss.empty() ? nullptr : metrics.loss.data();
    out->grad_norm = metrics.grad_norm;
    out->step_seconds = metrics.step_seconds;
    return NANOCHAT_STATUS_OK;
  });
}

int nanochat_runtime_param_count(nanochat_runtime* runtime, int replica) {
  return Guard("nanochat_runtime_param_count", 0, [&]() -> int {
    return static_cast<int>(RuntimeModel(runtime, replica)->params().size());
  });
}

int nanochat_runtime_param_info(nanochat_runtime* runtime, int replica,
                                int index, nanochat_param* out) {
  return Guard("nanochat_runtime_param_info", -1, [&]() -> int {
    if (out == nullptr) throw std::invalid_argument("out is null");
    const nanochat::ParamView view = RuntimeParam(runtime, replica, index);
    out->name = view.name;
    out->value = view.value;
    out->grad = view.grad;
    out->count = view.count;
    out->rows = view.rows;
    out->cols = view.cols;
    return 0;
  });
}

int64_t nanochat_runtime_param_read(nanochat_runtime* runtime, int replica,
                                    int index, int grad, int64_t offset,
                                    int64_t count, void* data) {
  return Guard(
      "nanochat_runtime_param_read", static_cast<std::int64_t>(-1),
      [&]() -> std::int64_t {
        const nanochat::ParamView view = RuntimeParam(runtime, replica, index);
        const nanochat::ComputeType* source = grad ? view.grad : view.value;
        if (source == nullptr) {
          throw std::invalid_argument("the parameter has no buffer");
        }
        if (offset < 0 || count < 0 || offset + count > view.count) {
          throw std::out_of_range("the parameter range is out of bounds");
        }
        if (count == 0) return 0;
        if (data == nullptr) throw std::invalid_argument("data is null");
        nanochat::kernels::Memcpy(
            data, source + offset,
            static_cast<std::size_t>(count) * sizeof(nanochat::ComputeType),
            nanochat::CopyDir::kDeviceToHost);
        return count;
      });
}

nanochat_status nanochat_runtime_param_digest(nanochat_runtime* runtime,
                                              int replica, uint64_t* out) {
  return Guard("nanochat_runtime_param_digest", NANOCHAT_STATUS_ERROR, [&]() {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (out == nullptr) throw std::invalid_argument("out is null");
    *out = runtime->runtime->ParamDigest(replica);
    return NANOCHAT_STATUS_OK;
  });
}

void nanochat_runtime_save(nanochat_runtime* runtime, const char* path) {
  GuardVoid("nanochat_runtime_save", [&]() {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (path == nullptr) throw std::invalid_argument("path is null");
    runtime->runtime->Save(path);
  });
}

void nanochat_runtime_load(nanochat_runtime* runtime, const char* path) {
  GuardVoid("nanochat_runtime_load", [&]() {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (path == nullptr) throw std::invalid_argument("path is null");
    runtime->runtime->Load(path);
  });
}

void nanochat_runtime_load_weights(nanochat_runtime* runtime,
                                   const char* path) {
  GuardVoid("nanochat_runtime_load_weights", [&]() {
    if (runtime == nullptr || runtime->runtime == nullptr) {
      throw std::invalid_argument("runtime is null");
    }
    if (path == nullptr) throw std::invalid_argument("path is null");
    runtime->runtime->LoadWeights(path);
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

void nanochat_backward_weighted(nanochat_model* model, const float* row_weights,
                                float scale) {
  GuardVoid("nanochat_backward_weighted", [&]() {
    if (model == nullptr || row_weights == nullptr) {
      throw std::invalid_argument("model or row_weights is null");
    }
    model->model->BackwardWeighted(row_weights, scale);
  });
}

void nanochat_backward_accumulate(nanochat_model* model, float scale) {
  GuardVoid("nanochat_backward_accumulate", [&]() {
    if (model == nullptr) throw std::invalid_argument("model is null");
    model->model->BackwardAccumulate(scale);
  });
}

float nanochat_train_step(nanochat_model* model, nanochat_optim* optimizer,
                          const int* tokens, const int* targets, int batch,
                          int seq) {
  return Guard("nanochat_train_step", 0.0f, [&]() -> float {
    if (model == nullptr || tokens == nullptr || targets == nullptr) {
      throw std::invalid_argument("model or token buffer is null");
    }
    if (batch <= 0 || seq <= 0) {
      throw std::invalid_argument("batch and seq must be positive");
    }
    // A null optimizer runs the forward and backward without an update
    // (docs/training-seam.md section 6.1).
    nanochat::Optimizer* raw =
        optimizer != nullptr ? optimizer->optimizer.get() : nullptr;
    return model->model->TrainStep(tokens, targets, batch, seq, raw);
  });
}

void nanochat_zero_grad(nanochat_model* model) {
  GuardVoid("nanochat_zero_grad", [&]() {
    if (model == nullptr) throw std::invalid_argument("model is null");
    model->model->ZeroGrad();
  });
}

float nanochat_rl_step(nanochat_model* model, nanochat_optim* optimizer,
                       const int* tokens, const int* targets,
                       const float* advantages, int batch, int seq,
                       int num_passes, int examples_per_rank, int step) {
  return Guard("nanochat_rl_step", 0.0f, [&]() -> float {
    if (model == nullptr || tokens == nullptr || targets == nullptr ||
        advantages == nullptr) {
      throw std::invalid_argument("model or input buffer is null");
    }
    if (batch <= 0 || seq <= 0) {
      throw std::invalid_argument("batch and seq must be positive");
    }
    // A null optimizer runs the forward and the weighted backward without an
    // update. The divisor `num_valid * num_passes * examples_per_rank` and the
    // ForwardLoss/BackwardWeighted/Optimizer::Step sequence live in `RlStep`
    // (docs/training-seam.md section 5.7).
    nanochat::Optimizer* raw =
        optimizer != nullptr ? optimizer->optimizer.get() : nullptr;
    return nanochat::RlStep(model->model.get(), raw, tokens, targets,
                            advantages, batch, seq, num_passes,
                            examples_per_rank, step);
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
    // Write the harness `NCHKPT01` container: one format for every writer
    // (docs/model.md). This writes the parameters only, with no optimizer
    // state and no step.
    if (!nanochat::Checkpointer::SaveModel(*model->model, path)) {
      throw std::runtime_error("cannot save checkpoint: " + std::string(path));
    }
  });
}

void nanochat_load(nanochat_model* model, const char* path) {
  GuardVoid("nanochat_load", [&]() {
    if (model == nullptr || path == nullptr) {
      throw std::invalid_argument("model or path is null");
    }
    // `Checkpointer::LoadModel` routes a `NCHKPT01` container or a legacy
    // `NCMDL001` file, so a distributed run's checkpoint loads in process.
    if (!nanochat::Checkpointer::LoadModel(model->model.get(), path)) {
      throw std::runtime_error("cannot load checkpoint: " + std::string(path));
    }
  });
}

nanochat_status nanochat_optim_config_preset(const char* name,
                                             nanochat_optim_config* out) {
  return Guard("nanochat_optim_config_preset", NANOCHAT_STATUS_ERROR, [&]() {
    if (name == nullptr) throw std::invalid_argument("name is null");
    if (out == nullptr) throw std::invalid_argument("out is null");
    if (std::strcmp(name, "track3") != 0) {
      throw std::invalid_argument(std::string("unknown preset '") + name + "'");
    }
    const nanochat::TrainConfig preset = nanochat::TrackThreeBaseline();
    FromOptimizerConfig(preset.optimizer, preset.scheduler, out);
    return NANOCHAT_STATUS_OK;
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

void nanochat_generate_multi(nanochat_model* model, const int* prompts,
                             const nanochat_generate_multi_params* params,
                             nanochat_sequences* out) {
  GuardVoid("nanochat_generate_multi", [&]() {
    if (model == nullptr || prompts == nullptr || out == nullptr) {
      throw std::invalid_argument("model, prompts, or out is null");
    }
    if (params == nullptr) {
      throw std::invalid_argument("params is null");
    }

    nanochat::GenerateParams cpp_params;
    cpp_params.num_samples = params->params.num_samples;
    cpp_params.max_tokens = params->params.max_tokens;
    cpp_params.temperature = params->params.temperature;
    cpp_params.top_k = params->params.top_k;
    cpp_params.seed = params->params.seed;
    cpp_params.stop_id = params->params.stop_id;
    cpp_params.bos_id = params->params.bos_id;
    cpp_params.stop_ids = params->params.stop_ids;

    std::vector<nanochat::GeneratedSequence> results;
    nanochat::GenerateMultiPrompt(model->model.get(), prompts,
                                  params->num_prompts, params->prompt_len,
                                  cpp_params, params->row_stops, &results);

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

// --- Checkpoint container -------------------------------------------------

namespace {

nanochat::DType ReadStorageDType(int32_t dtype) {
  const nanochat::DType storage = static_cast<nanochat::DType>(dtype);
  if (!nanochat::IsStorageDType(storage)) {
    throw std::invalid_argument("unknown storage dtype code");
  }
  return storage;
}

void ReadShape(int32_t ndim, const int64_t* dims,
               std::vector<std::int64_t>* shape) {
  if (ndim < 0 || ndim > 8) {
    throw std::invalid_argument("ndim is out of range");
  }
  if (ndim > 0 && dims == nullptr) throw std::invalid_argument("dims is null");
  shape->clear();
  for (int32_t axis = 0; axis < ndim; ++axis) shape->push_back(dims[axis]);
}

std::vector<std::byte> ReadPayload(const void* data, int64_t bytes) {
  if (bytes < 0) throw std::invalid_argument("byte count is negative");
  if (bytes > 0 && data == nullptr) throw std::invalid_argument("data is null");
  std::vector<std::byte> payload(static_cast<std::size_t>(bytes));
  if (bytes > 0) {
    std::memcpy(payload.data(), data, static_cast<std::size_t>(bytes));
  }
  return payload;
}

// Copies one `nanochat_record` and its attributes into a C++ record.
nanochat::TensorRecord ToRecord(const nanochat_record& view) {
  if (view.name == nullptr) throw std::invalid_argument("record name is null");
  if (view.attribute_count < 0) {
    throw std::invalid_argument("negative attribute count");
  }
  if (view.attribute_count > 0 && view.attributes == nullptr) {
    throw std::invalid_argument("attributes is null");
  }
  nanochat::TensorRecord record;
  record.name = view.name;
  record.dtype = ReadStorageDType(view.dtype);
  ReadShape(view.ndim, view.dims, &record.shape);
  record.data = ReadPayload(view.data, view.bytes);
  for (int64_t index = 0; index < view.attribute_count; ++index) {
    const nanochat_record& source = view.attributes[index];
    if (source.name == nullptr) {
      throw std::invalid_argument("attribute name is null");
    }
    nanochat::TensorAttribute attribute;
    attribute.name = source.name;
    attribute.dtype = ReadStorageDType(source.dtype);
    ReadShape(source.ndim, source.dims, &attribute.shape);
    attribute.data = ReadPayload(source.data, source.bytes);
    record.attributes.push_back(std::move(attribute));
  }
  return record;
}

// Converts a caller's list into C++ records.
void AppendRecords(const nanochat_record* records, int64_t count,
                   std::vector<nanochat::TensorRecord>* out) {
  if (count < 0) throw std::invalid_argument("negative record count");
  if (count > 0 && records == nullptr) {
    throw std::invalid_argument("records is null");
  }
  for (int64_t index = 0; index < count; ++index) {
    out->push_back(ToRecord(records[index]));
  }
}

// Copies one C++ record into a `nanochat_record`. The pointers point into the
// source record, which must outlive the view. `attributes` points into
// `attribute_storage`, which the caller sizes before the fill.
void FillRecord(const nanochat::TensorRecord& record, nanochat_record* out,
                std::vector<nanochat_record>* attribute_storage,
                std::size_t* cursor) {
  out->name = record.name.c_str();
  out->dtype = static_cast<int32_t>(record.dtype);
  out->ndim = static_cast<int32_t>(record.shape.size());
  out->dims = record.shape.empty() ? nullptr : record.shape.data();
  out->data = reinterpret_cast<const void*>(record.data.data());
  out->bytes = static_cast<int64_t>(record.data.size());
  out->attribute_count = static_cast<int64_t>(record.attributes.size());
  if (record.attributes.empty()) {
    out->attributes = nullptr;
    return;
  }
  const std::size_t start = *cursor;
  for (const nanochat::TensorAttribute& attribute : record.attributes) {
    nanochat_record& entry = (*attribute_storage)[(*cursor)++];
    entry.name = attribute.name.c_str();
    entry.dtype = static_cast<int32_t>(attribute.dtype);
    entry.ndim = static_cast<int32_t>(attribute.shape.size());
    entry.dims = attribute.shape.empty() ? nullptr : attribute.shape.data();
    entry.data = reinterpret_cast<const void*>(attribute.data.data());
    entry.bytes = static_cast<int64_t>(attribute.data.size());
    entry.attribute_count = 0;
    entry.attributes = nullptr;
  }
  out->attributes = attribute_storage->data() + start;
}

// Copies a whole C++ list into `out`. `attribute_storage` must already hold
// room for every attribute of the list, and `cursor` advances across calls.
void FillRecords(const std::vector<nanochat::TensorRecord>& source,
                 std::vector<nanochat_record>* out,
                 std::vector<nanochat_record>* attribute_storage,
                 std::size_t* cursor) {
  out->resize(source.size());
  for (std::size_t index = 0; index < source.size(); ++index) {
    FillRecord(source[index], &(*out)[index], attribute_storage, cursor);
  }
}

// The attribute count of a whole list.
std::size_t CountAttributes(const std::vector<nanochat::TensorRecord>& source) {
  std::size_t total = 0;
  for (const nanochat::TensorRecord& record : source) {
    total += record.attributes.size();
  }
  return total;
}

// The stable name of a storage dtype.
const char* StorageDTypeName(nanochat::DType dtype) {
  switch (dtype) {
    case nanochat::DType::kBf16:
      return "bfloat16";
    case nanochat::DType::kInt8:
      return "int8";
    case nanochat::DType::kFp32:
    default:
      return "float32";
  }
}

// The storage dtypes, in code order.
constexpr nanochat::DType kStorageDTypes[] = {
    nanochat::DType::kFp32, nanochat::DType::kBf16, nanochat::DType::kInt8};

}  // namespace

nanochat_status nanochat_checkpoint_dtype(int32_t index, const char** name,
                                          int32_t* code, int32_t* size) {
  return Guard("nanochat_checkpoint_dtype", NANOCHAT_STATUS_ERROR, [&]() {
    if (name == nullptr || code == nullptr || size == nullptr) {
      throw std::invalid_argument("name, code, or size is null");
    }
    constexpr int32_t kCount = static_cast<int32_t>(sizeof(kStorageDTypes) /
                                                    sizeof(kStorageDTypes[0]));
    if (index < 0 || index >= kCount) {
      *name = nullptr;
      *code = 0;
      *size = 0;
      return NANOCHAT_STATUS_OK;
    }
    const nanochat::DType dtype = kStorageDTypes[index];
    *name = StorageDTypeName(dtype);
    *code = static_cast<int32_t>(dtype);
    *size = static_cast<int32_t>(nanochat::SizeOf(dtype));
    return NANOCHAT_STATUS_OK;
  });
}

nanochat_status nanochat_checkpoint_write(const char* path,
                                          const nanochat_record* records,
                                          int64_t record_count,
                                          const nanochat_record* optimizer,
                                          int64_t optimizer_count) {
  return Guard("nanochat_checkpoint_write", NANOCHAT_STATUS_ERROR, [&]() {
    if (path == nullptr) throw std::invalid_argument("path is null");
    nanochat::Checkpoint checkpoint;
    AppendRecords(records, record_count, &checkpoint.tensors());
    AppendRecords(optimizer, optimizer_count, &checkpoint.optimizer_state());
    if (!checkpoint.Save(path)) {
      throw std::runtime_error("cannot write checkpoint: " + std::string(path));
    }
    return NANOCHAT_STATUS_OK;
  });
}

nanochat_status nanochat_checkpoint_read(const char* path,
                                         nanochat_checkpoint** owner,
                                         const nanochat_record** records,
                                         int64_t* record_count,
                                         const nanochat_record** optimizer,
                                         int64_t* optimizer_count) {
  return Guard("nanochat_checkpoint_read", NANOCHAT_STATUS_ERROR, [&]() {
    if (path == nullptr || owner == nullptr || records == nullptr ||
        record_count == nullptr || optimizer == nullptr ||
        optimizer_count == nullptr) {
      throw std::invalid_argument("a read argument is null");
    }
    std::unique_ptr<nanochat_checkpoint> handle(new nanochat_checkpoint());
    std::string error;
    if (!handle->checkpoint.Load(path, &error)) {
      throw std::runtime_error("cannot read checkpoint: " + error);
    }
    const std::vector<nanochat::TensorRecord>& parameters =
        handle->checkpoint.tensors();
    const std::vector<nanochat::TensorRecord>& state =
        handle->checkpoint.optimizer_state();
    handle->attributes.resize(CountAttributes(parameters) +
                              CountAttributes(state));
    std::size_t cursor = 0;
    FillRecords(parameters, &handle->records, &handle->attributes, &cursor);
    FillRecords(state, &handle->optimizer_state, &handle->attributes, &cursor);
    nanochat_checkpoint* result = handle.release();
    *owner = result;
    *records = result->records.data();
    *record_count = static_cast<int64_t>(result->records.size());
    *optimizer = result->optimizer_state.data();
    *optimizer_count = static_cast<int64_t>(result->optimizer_state.size());
    return NANOCHAT_STATUS_OK;
  });
}

void nanochat_checkpoint_free(nanochat_checkpoint* owner) { delete owner; }

}  // extern "C"
