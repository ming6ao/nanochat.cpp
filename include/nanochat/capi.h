#ifndef NANOCHAT_CAPI_H_
#define NANOCHAT_CAPI_H_

#include <stddef.h>
#include <stdint.h>

// The C application binary interface for the in-process Python API
// (docs/python.md section 4). A Python binding loads the shared library
// with the standard `ctypes` module and calls these functions.
//
// Every type here is a C type. No C++ type crosses the boundary. Opaque handles
// hide the C++ objects. This header is additive; it changes no frozen header.

#ifdef __cplusplus
extern "C" {
#endif

// Status code. Zero means success and a nonzero value means failure. The
// message for the current thread comes from nanochat_last_error.
typedef enum {
  NANOCHAT_STATUS_OK = 0,
  NANOCHAT_STATUS_ERROR = 1,
} nanochat_status;

// Opaque handles. The matching free function releases each one.
typedef struct nanochat_model nanochat_model;
typedef struct nanochat_optim nanochat_optim;
typedef struct nanochat_loader nanochat_loader;
typedef struct nanochat_tokenizer nanochat_tokenizer;

// Model hyperparameters, a mirror of nanochat::Config.
typedef struct {
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int hidden_dim;
  int seq_len;
  int vocab_size;
  int padded_vocab_size;
  float rope_base;
  const char* window_pattern;
} nanochat_config;

// Model-derived planning inputs (docs/python.md section 4.1). `total` counts
// every allocated parameter, including the token and value embeddings.
// `transformer_matrices` and `lm_head` mirror the reference
// `GPT.num_scaling_params()`. The shim fills this from `CountParams` and
// `EstimateFlopsPerToken`.
typedef struct {
  int64_t total;
  int64_t transformer_matrices;
  int64_t lm_head;
  int64_t embeddings;
  int64_t scalars;
  double flops_per_token;
} nanochat_params;

// A vendor-free description of the active device, a partial mirror of
// nanochat::Caps. The backend itself comes from nanochat_backend: `cpu` or
// `cuda`. `device_name` is a static string that the caller must not free.
typedef struct {
  int device_index;
  int compute_major;
  int compute_minor;
  int64_t total_memory_bytes;
  const char* device_name;
} nanochat_device;

// One parameter view, a mirror of nanochat::ParamView. `value` and `grad`
// point to the build compute type, float32 or float16. `count` is the number
// of elements. `rows` and `cols` are the matrix extents for Muon; a zero
// `rows` marks a vector or a scalar.
typedef struct {
  const char* name;
  void* value;
  void* grad;
  int64_t count;
  int rows;
  int cols;
} nanochat_param;

// One sequence focus request for nanochat_score_batch, a mirror of
// nanochat::ScoreFocus. A negative `position` or a nonpositive `count`
// disables the focus.
typedef struct {
  int position;
  const int* ids;
  int count;
} nanochat_focus;

// One sequence score result, a mirror of nanochat::ScoreResult. The buffers
// belong to the model and stay valid until the next call on that model.
typedef struct {
  const float* nll;
  const int* argmax;
  const float* focus_logits;
} nanochat_score_result;

// Parameters for nanochat_generate, a mirror of nanochat::GenerateParams.
typedef struct {
  int num_samples;
  int max_tokens;
  float temperature;
  int top_k;
  uint64_t seed;
  int stop_id;
  int bos_id;
  const int* stop_ids;
} nanochat_generate_params;

// Generated rows from nanochat_generate. The `tokens` and `mask` arrays hold
// the rows end to end; `offsets` and `lengths` give the extent of each row.
// The buffers belong to the model and stay valid until the next call.
typedef struct {
  int count;
  const int* tokens;
  const uint8_t* mask;
  const int* lengths;
  const int* offsets;
} nanochat_sequences;

// Optimizer and schedule hyperparameters. The two groups mirror
// nanochat::OptimizerConfig and nanochat::SchedulerConfig.
typedef struct {
  // nanochat::OptimizerConfig.
  float unembedding_lr;
  float embedding_lr;
  float matrix_lr;
  float scalar_lr;
  float weight_decay;
  int muon_ns_steps;
  float muon_beta2;
  float adam_eps;
  float clip;
  // nanochat::SchedulerConfig.
  int num_iterations;
  int warmup_steps;
  float warmdown_ratio;
  float final_lr_frac;
  float weight_decay_base;
  float muon_momentum_warmup_steps;
  float muon_momentum_start;
  float muon_momentum_peak;
  float muon_momentum_final;
} nanochat_optim_config;

// Checks the sandbox rule and prepares the library. Call before any other
// function. `nanochat_last_error` and `nanochat_version` return thread-local
// and static strings that the caller must not free.
nanochat_status nanochat_init(void);
const char* nanochat_last_error(void);
const char* nanochat_version(void);

// The active backend: "cpu" for the reference backend, or "cuda" when the
// library links the CUDA kernels. The string is static. Call after
// nanochat_init. `nanochat_compute_type_size` reports the parameter element
// size in bytes, four for fp32 and two for fp16.
const char* nanochat_backend(void);
int nanochat_compute_type_size(void);

// Fills `out` with a description of the active device.
void nanochat_device_info(nanochat_device* out);

// Fills `out` from `config`. Allocates no model. Returns a status.
nanochat_status nanochat_params_get(const nanochat_config* config,
                                    nanochat_params* out);

// Model. `nanochat_model_create` returns an owned handle and null on failure.
nanochat_model* nanochat_model_create(const nanochat_config* config,
                                      uint64_t seed);
void nanochat_model_free(nanochat_model* model);
float nanochat_forward_loss(nanochat_model* model, const int* tokens,
                            const int* targets, int batch, int seq);
void nanochat_backward(nanochat_model* model);
// Weighted backward for SFT and RL (docs/post-training.md section 2.2).
// `row_weights` holds one weight per row, that is `batch * seq` entries. A null
// model or a null weight buffer is an error. The call accumulates into the
// parameter gradients, so call nanochat_zero_grad first for a fresh gradient.
void nanochat_backward_weighted(nanochat_model* model, const float* row_weights,
                                float scale);
void nanochat_zero_grad(nanochat_model* model);
int nanochat_param_count(nanochat_model* model);
int nanochat_param_info(nanochat_model* model, int index, nanochat_param* out);

// Copies `count` elements at `offset` of parameter `index` between the model
// buffer and the host buffer `data`. `grad` selects the gradient when nonzero
// and the value otherwise. Each function returns the number of elements
// copied, or a negative value on error. Use these instead of dereferencing the
// `nanochat_param` pointers, because a device backend keeps the buffers in
// device memory. The host buffer holds the build compute type: four-byte
// float32, or two-byte float16 under a half-precision build.
int64_t nanochat_param_read(nanochat_model* model, int index, int grad,
                            int64_t offset, int64_t count, void* data);
int64_t nanochat_param_write(nanochat_model* model, int index, int grad,
                             int64_t offset, int64_t count, const void* data);
void nanochat_save(nanochat_model* model, const char* path);
void nanochat_load(nanochat_model* model, const char* path);

// Optimizer. `nanochat_optim_create` returns an owned handle and null on
// failure.
nanochat_optim* nanochat_optim_create(nanochat_model* model,
                                      const nanochat_optim_config* config);
void nanochat_optim_step(nanochat_optim* optimizer, int step);
float nanochat_optim_grad_norm(nanochat_optim* optimizer);
void nanochat_optim_free(nanochat_optim* optimizer);

// Data loader over parquet documents. `nanochat_loader_create` returns an
// owned handle and null on failure. The tokenizer must outlive the loader.
nanochat_loader* nanochat_loader_create(const char** parquet, int count,
                                        const char* text_column,
                                        nanochat_tokenizer* tokenizer,
                                        int batch, int seq, uint64_t seed,
                                        int tokenizer_threads,
                                        size_t document_buffer);
int nanochat_loader_next(nanochat_loader* loader, int* tokens, int* targets);
const uint8_t* nanochat_loader_token_bytes(nanochat_loader* loader, int* vocab);
void nanochat_loader_free(nanochat_loader* loader);

// Batch score. `focus` points to `batch` requests, or null. `out` points to
// `batch` results with model-owned buffers.
void nanochat_score_batch(nanochat_model* model, const int* tokens, int batch,
                          int seq, const int* lengths,
                          const nanochat_focus* focus,
                          nanochat_score_result* out);

// Batched generation. `out` receives model-owned buffers.
void nanochat_generate(nanochat_model* model, const int* prompt, int length,
                       const nanochat_generate_params* params,
                       nanochat_sequences* out);

// Forward-only bits-per-byte over `steps` batches from the loader.
float nanochat_eval_bpb(nanochat_model* model, nanochat_loader* loader,
                        int steps);

// Tokenizer. `nanochat_tokenizer_load` returns an owned handle and null on
// failure. `nanochat_encode` and `nanochat_decode` return the needed length.
nanochat_tokenizer* nanochat_tokenizer_load(const char* path);
int nanochat_encode(nanochat_tokenizer* tokenizer, const char* text, int* out,
                    int capacity);
int nanochat_decode(nanochat_tokenizer* tokenizer, const int* ids, int count,
                    char* out, int capacity);
void nanochat_tokenizer_free(nanochat_tokenizer* tokenizer);

#ifdef __cplusplus
}
#endif

#endif  // NANOCHAT_CAPI_H_
