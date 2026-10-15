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

// Model hyperparameters, a mirror of nanochat::Config. `value_embedding` is
// nonzero when the ResFormer value embeddings are present. The Track 3 preset
// sets it to zero (docs/distributed-design.md section 8).
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
  int value_embedding;
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

// Parameters for nanochat_generate_multi, a mirror of
// nanochat::GenerateMultiPrompt. `num_prompts` prompts of `prompt_len` ids each
// produce `params.num_samples` rows, so the output holds `num_prompts *
// params.num_samples` rows in prompt-major order. `row_stops` optionally holds
// one terminal id per output row (`num_prompts * params.num_samples` entries in
// the same order); a negative entry disables stopping for that row, and a null
// pointer reuses the `stop_id`/`stop_ids` fields of `params` for every prompt.
typedef struct {
  int num_prompts;
  int prompt_len;
  const int* row_stops;
  nanochat_generate_params params;
} nanochat_generate_multi_params;

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
  // nanochat::OptimizerConfig (ANVIL, docs/optimizer-anvil-design.md).
  int matrix_optimizer;
  int adam_step_period;
  float anvil_lr;
  float anvil_weight_decay;
  float anvil_momentum;
  float anvil_beta2;
  float anvil_fast_beta;
  float anvil_slow_beta;
  float anvil_fast_weight;
  int anvil_engage_step;
  int anvil_num_maps;
  // nanochat::SchedulerConfig (ANVIL fast-rail beta).
  float rail_beta_warmup_steps;
  float rail_beta_cooldown_steps;
  float rail_beta_min;
  float rail_beta_max;
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

// Enumerates the local accelerators for the single-client multi-device runtime
// (docs/client-runtime-design.md section 5.1). `nanochat_device_count` returns
// the number of local devices: one on the reference backend, or the CUDA device
// count. `nanochat_device_at` fills `out` with the device at `ordinal` and
// returns 0 on success, or a nonzero value when `ordinal` is out of range. The
// device name points at a static string; the caller must not free it.
int nanochat_device_count(void);
int nanochat_device_at(int ordinal, nanochat_device* out);

// The single-client multi-device runtime (docs/client-runtime-design.md). A
// mesh handle owns one `MeshSpec`; a runtime handle owns one `MeshRuntime`,
// with one model replica per device. `nanochat_runtime_train_step` splits one
// global batch across the mesh, reduces the gradients in process, and returns
// the global mean loss. `nanochat_runtime_param_read` copies a replica
// parameter to the host through the kernel seam, so it is safe on a device
// backend. Create the mesh first, then the runtime, and free each handle once.
typedef struct nanochat_mesh nanochat_mesh;
typedef struct nanochat_runtime nanochat_runtime;

// Builds a rank-1 data mesh over `count` devices. `indices` gives the backend
// device index at each mesh position; a null pointer means `0, 1, ...`. A count
// below 1 becomes 1.
nanochat_mesh* nanochat_mesh_create(const int* indices, int count);
void nanochat_mesh_free(nanochat_mesh* mesh);
int nanochat_mesh_size(nanochat_mesh* mesh);

// Builds the runtime over `mesh`. `config` is the model shape, and `optim` may
// be null for the defaults. `batch` is the global batch in sequences,
// `grad_accum` is the micro-batch count per replica, and `seq` of 0 selects
// `config.seq_len`. Returns null on failure.
nanochat_runtime* nanochat_runtime_create(const nanochat_mesh* mesh,
                                          const nanochat_config* config,
                                          const nanochat_optim_config* optim,
                                          int batch, int grad_accum, int seq,
                                          uint64_t seed);
void nanochat_runtime_free(nanochat_runtime* runtime);
int nanochat_runtime_world_size(nanochat_runtime* runtime);

// Runs one step over `batch * seq` row-major token and target ids. Returns the
// global mean loss, or 0 on error.
float nanochat_runtime_train_step(nanochat_runtime* runtime, const int* tokens,
                                  const int* targets);

// One step report, a mirror of `nanochat::MeshMetrics`
// (docs/client-runtime-design.md section 20.4). `device_count` is the mesh
// size. `loss` points to `device_count` per-replica losses. The buffer belongs
// to the runtime and stays valid until the next step or the next call to
// nanochat_runtime_metrics. `grad_norm` is the global gradient norm after the
// reduce, so every replica reports the same value. `step_seconds` is the
// wall-clock time of the last step.
typedef struct {
  int device_count;
  const float* loss;
  float grad_norm;
  float step_seconds;
} nanochat_metrics;

// Fills `out` with the metrics of the last step. Call after
// nanochat_runtime_train_step. Returns a status.
nanochat_status nanochat_runtime_metrics(nanochat_runtime* runtime,
                                         nanochat_metrics* out);

// The parameter count of `replica`, or 0 on error. `grad` selects the gradient
// when nonzero and the value otherwise. The read returns the number of
// elements copied, or a negative value on error.
int nanochat_runtime_param_count(nanochat_runtime* runtime, int replica);
int nanochat_runtime_param_info(nanochat_runtime* runtime, int replica,
                                int index, nanochat_param* out);
int64_t nanochat_runtime_param_read(nanochat_runtime* runtime, int replica,
                                    int index, int grad, int64_t offset,
                                    int64_t count, void* data);

// A 64-bit FNV-1a digest of every parameter of `replica`. The two replicas
// return the same digest after a step, so a client can check the gradient
// reduce without copying the whole model. A bad replica index or a null `out`
// is an error. Returns a status.
nanochat_status nanochat_runtime_param_digest(nanochat_runtime* runtime,
                                              int replica, uint64_t* out);

// Writes the runtime's canonical checkpoint from replica 0, or loads one into
// every replica. The file is the harness `NCHKPT01` container, so it carries
// the optimizer state and the step beside the parameters. A load restores the
// state and moves the step, so a resume continues the schedule. A load that
// does not find the file sets the thread-local error.
void nanochat_runtime_save(nanochat_runtime* runtime, const char* path);
void nanochat_runtime_load(nanochat_runtime* runtime, const char* path);

// Loads the parameter records of a checkpoint into every replica and leaves the
// optimizer state and the step unchanged (notebooks/checkpoint.md section 9.1).
// A fine-tuning warm start uses this entry point, so it does not inherit the
// pretraining moments. The optimizer section is not read. A load that does not
// find the file sets the thread-local error.
void nanochat_runtime_load_weights(nanochat_runtime* runtime, const char* path);

// The 1-based step of the last step, or the loaded step after a load. Returns
// -1 on a bad handle.
int nanochat_runtime_step(nanochat_runtime* runtime);

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
// Gradient accumulation (docs/training-seam.md section 6.1). Accumulates the
// backward pass over the most recent forward into the parameter gradients
// without zeroing them. `scale` multiplies the loss gradient; use
// `1 / micro_batches`. A null model is an error.
void nanochat_backward_accumulate(nanochat_model* model, float scale);
// Forward + backward + one optimizer step (docs/training-seam.md section 6.1).
// Returns the batch-mean loss over the valid targets. A null `optimizer` runs
// the forward and backward without an update.
float nanochat_train_step(nanochat_model* model, nanochat_optim* optimizer,
                          const int* tokens, const int* targets, int batch,
                          int seq);
// One reinforcement-learning optimizer step (docs/training-seam.md section
// 5.7, docs/post-training.md section 5.2). It runs ForwardLoss, then
// BackwardWeighted with `advantages` as the per-row weight, then
// Optimizer::Step(step). The divisor `num_valid * num_passes *
// examples_per_rank` is computed in C++, so the caller never computes it;
// `num_valid` is the number of `targets` that are not -1. `tokens` and
// `targets` hold `batch * seq` entries and `advantages` holds one weight per
// row, that is `batch * seq` entries. The `-1` target is the only mask
// channel. `step` is 1-based and drives the optimizer schedules. Returns the
// forward mean loss over the valid targets. A null `optimizer` runs the step
// without the parameter update. A null model or input buffer, a nonpositive
// `batch`/`seq`, or a nonpositive `num_passes`/`examples_per_rank` is an error.
float nanochat_rl_step(nanochat_model* model, nanochat_optim* optimizer,
                       const int* tokens, const int* targets,
                       const float* advantages, int batch, int seq,
                       int num_passes, int examples_per_rank, int step);
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
// `nanochat_save` writes the self-describing `NCHKPT01` container with the
// parameters only. `nanochat_load` accepts that container and the legacy
// `NCMDL001` file, so an older checkpoint still loads in process
// (docs/model.md).
void nanochat_save(nanochat_model* model, const char* path);
void nanochat_load(nanochat_model* model, const char* path);
// The checkpoint container surface (docs/model.md). The byte layout lives in
// `src/data.cc` only; this surface lets a host tool build and inspect a
// container without repeating the layout. The C++ training runtime does not
// need it.
//
// One record and its attributes. A writer fills the struct, and a reader
// returns it. `attributes` points to `attribute_count` entries, or to null.
// Every pointer is owned by the caller of `write`, or by `*owner` after a
// `read`.
typedef struct nanochat_record {
  const char* name;
  int32_t dtype;
  int32_t ndim;
  const int64_t* dims;
  const void* data;
  int64_t bytes;
  int64_t attribute_count;
  const struct nanochat_record* attributes;
} nanochat_record;

// Opaque owner of the arrays that `nanochat_checkpoint_read` returns.
typedef struct nanochat_checkpoint nanochat_checkpoint;

// The storage dtype at `index`, in a stable order. Fills the name, the code,
// and the element size. `name` points to a static string. The call fills
// `*name` with null when `index` is past the last dtype, so a caller
// enumerates from 0 until the name is null. A null out pointer is an error.
nanochat_status nanochat_checkpoint_dtype(int32_t index, const char** name,
                                          int32_t* code, int32_t* size);

// Writes both record lists to `path` in the version 0 container layout. The
// caller owns every buffer, and it keeps them alive for the call.
nanochat_status nanochat_checkpoint_write(const char* path,
                                          const nanochat_record* records,
                                          int64_t record_count,
                                          const nanochat_record* optimizer,
                                          int64_t optimizer_count);

// Reads both record lists from `path`. Every out parameter is required. The
// arrays belong to `*owner`, and they stay valid until
// `nanochat_checkpoint_free`. The file must be a version 0 container.
nanochat_status nanochat_checkpoint_read(const char* path,
                                         nanochat_checkpoint** owner,
                                         const nanochat_record** records,
                                         int64_t* record_count,
                                         const nanochat_record** optimizer,
                                         int64_t* optimizer_count);

// Releases a handle and the arrays it owns. A null handle is accepted.
void nanochat_checkpoint_free(nanochat_checkpoint* owner);

// Fills `out` with a named preset of the optimizer and scheduler
// hyperparameters, so a client gets the same values as the `train_main`
// `--preset` flag without copying them. `name` is `track3` today. A null
// `out`, a null `name`, or an unknown name is an error. See
// docs/client-runtime-design.md section 20.4.
nanochat_status nanochat_optim_config_preset(const char* name,
                                             nanochat_optim_config* out);

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

// Multi-prompt batched generation: a whole rollout in one call, so a rollout
// does not cross the ABI once per prompt. `prompts` holds `num_prompts *
// prompt_len` row-major ids and `out` receives
// `num_prompts * params.num_samples` rows in prompt-major order, with
// model-owned buffers that stay valid until the next call. Additive; the
// single-prompt nanochat_generate is unchanged.
void nanochat_generate_multi(nanochat_model* model, const int* prompts,
                             const nanochat_generate_multi_params* params,
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
