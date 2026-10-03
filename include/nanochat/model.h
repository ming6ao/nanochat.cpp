#ifndef NANOCHAT_MODEL_H_
#define NANOCHAT_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/sampler.h"
#include "nanochat/tensor.h"

// The public model API: the training graph, the two inference graphs, and the
// evaluation entry point and compute primitives (docs/model.md, docs/eval.md).
// This header is frozen; treat it as stable by default and prefer additive
// changes (DESIGN.md section 7).
//
// The topology lives in model.cc (train forward + backward) and generate.cc
// (prefill + decode). No other translation unit knows the architecture.

namespace nanochat {

class Optimizer;
class DataLoader;

// A non-owning view of one model parameter and its gradient, as exposed for the
// optimizer. `rows` and `cols` are the trailing matrix extents for parameters
// that Muon orthogonalizes; `rows == 0` marks a vector/scalar that AdamW owns.
// Moment buffers belong to the optimizer, not the model.
struct ParamView {
  const char* name = "";
  ComputeType* value = nullptr;
  ComputeType* grad = nullptr;
  std::int64_t count = 0;
  int rows = 0;
  int cols = 0;

  bool is_matrix() const { return rows > 0 && cols > 0; }
};

class Model {
 public:
  virtual ~Model() = default;

  // Builds a model with the given architecture. The returned model owns all of
  // its weights and per-graph workspace.
  static std::unique_ptr<Model> Create(const Config& config);

  // Deterministically initializes every weight (docs/model.md).
  virtual void InitWeights(std::uint64_t seed) = 0;

  const Config& config() const { return config_; }

  // Training graph: forward + softcap + cross-entropy. Returns the batch-mean
  // loss in nats. Saves the activations Backward() needs.
  virtual float ForwardLoss(const int* tokens, const int* targets, int batch,
                            int seq) = 0;

  // Runs the hand-written backward pass over the saved activations.
  virtual void Backward() = 0;

  // Zeroes every parameter gradient. Call before a sequence of
  // `BackwardAccumulate` passes so the passes sum into an empty gradient.
  virtual void ZeroGrad() = 0;

  // Runs the backward pass and *accumulates* into the parameter gradients
  // without zeroing them first. `scale` multiplies the loss gradient; use
  // `1 / micro_batches` for gradient accumulation. `Backward()` is equivalent
  // to `ZeroGrad()` followed by `BackwardAccumulate(1.0f)`.
  virtual void BackwardAccumulate(float scale) = 0;

  // Forward + backward + one optimizer step. Returns the batch-mean loss.
  virtual float TrainStep(const int* tokens, const int* targets, int batch,
                          int seq, Optimizer* optimizer) = 0;

  // Grad mode (docs/grad-mode.md), the analogue of `torch.set_grad_enabled()`.
  // The default state is on. When the caller turns it off, `ForwardLoss` runs
  // the forward-only path: it builds a smaller workspace, reuses one block's
  // activations across layers, and does not save anything `Backward()` reads.
  // `Backward`, `BackwardAccumulate`, and `TrainStep` then stop with an error.
  // The defaults keep the methods additive: an implementation that does not
  // support grad mode stays in the always-on state.
  virtual void SetGradEnabled(bool /*enabled*/) {}
  virtual bool grad_enabled() const { return true; }

  // Parameter views in the optimizer's grouping order.
  virtual std::vector<ParamView> params() const = 0;

  // Self-describing checkpoint container (docs/data.md).
  virtual void Save(const std::string& path) const = 0;
  virtual void Load(const std::string& path) = 0;

 protected:
  explicit Model(Config config) : config_(std::move(config)) {}

  Config config_;
};

// RAII guard for the grad-mode state (docs/grad-mode.md). It is the analogue of
// `torch.no_grad()`: the constructor turns grad mode off, and the destructor
// restores the previous state. The model must outlive the guard.
class NoGradGuard {
 public:
  explicit NoGradGuard(Model* model)
      : model_(model), previous_(model->grad_enabled()) {
    model_->SetGradEnabled(false);
  }
  ~NoGradGuard() { model_->SetGradEnabled(previous_); }
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  Model* model_;
  bool previous_;
};

// Host-side key/value cache for the inference graphs. The concrete storage
// layout is backend-visible and lives in generate.cc; callers only create it,
// hand it to Prefill/Decode, and read its position.
class KvCache {
 public:
  virtual ~KvCache() = default;

  // Number of tokens written so far (the next decode position).
  virtual int pos() const = 0;

  // Maximum sequence length the cache was sized for.
  virtual int capacity() const = 0;
};

std::unique_ptr<KvCache> CreateKvCache(const Config& config, int max_seq);

// Inference graph 1: runs the trunk over `num_tokens` and appends every layer's
// key/value rows to `kv`.
void Prefill(Model* model, const int* tokens, int num_tokens, KvCache* kv);

// Inference graph 2: attends one token against `kv`, appends its row, and
// samples the next token id.
int Decode(Model* model, int token, KvCache* kv, const SampleParams& params);

// Forward-only bits-per-byte over `steps` batches from `loader` (mirrors
// nanochat's `evaluate_bpb`).
float EvalBpb(Model* model, DataLoader* loader, int steps);

// ---------------------------------------------------------------------------
// Evaluation compute primitives (docs/eval.md section 3). Both are
// forward-only, host-orchestrated entry points over the training graph; they
// never step an optimizer. They stay tokenizer-agnostic: ids in, ids and
// logits out.
// ---------------------------------------------------------------------------

// One sequence's optional focus request for ScoreBatch: read the logits that
// the model produces at prediction position `position` (those logits predict
// the token at `position + 1`) for the `count` token ids in `ids`, in order.
// `position < 0` or `count <= 0` disables the focus for this sequence.
struct ScoreFocus {
  int position = -1;
  const int* ids = nullptr;
  int count = 0;
};

// Per-sequence result of ScoreBatch, staged to the host. `nll` and `argmax`
// hold one entry per prediction position in `[0, seq)`.
struct ScoreResult {
  // Negative log-likelihood in nats of the target token at each prediction
  // position. Masked positions (padding and each row's final position) carry a
  // zero and must be skipped by the caller.
  std::vector<float> nll;
  // Argmax token id of the raw (pre-softcap) logits at each prediction
  // position, restricted to `[0, vocab_size)`. Masked positions repeat the
  // ignore index (-1).
  std::vector<int> argmax;
  // Raw (pre-softcap) logits at the focus position for each focus id, in the
  // request order; empty when the sequence carries no focus. The softcap is
  // monotonic, so an argmax over these equals an argmax over the soft-capped
  // logits.
  std::vector<float> focus_logits;
};

// Forward-only scoring over `batch` independent sequences of length `seq`,
// using the same shifted-target convention as `ForwardLoss`: the logits at
// position `i` predict the token at `i + 1`. `tokens` is a row-major
// `batch * seq` host buffer; `lengths` gives the number of valid ids per row,
// and every position at or past a row's length is masked with the ignore index
// (-1) so it contributes no loss. `lengths` may be null to treat every row as
// full length. `focus` may be null for no focused logits. `out` is resized to
// `batch` entries. Runs one forward; never touches gradients.
void ScoreBatch(Model* model, const int* tokens, int batch, int seq,
                const int* lengths, const ScoreFocus* focus,
                std::vector<ScoreResult>* out);

// Parameters for batched, tokenizer-agnostic generation.
struct GenerateParams {
  // Number of rows to sample from the prefilled prompt.
  int num_samples = 1;
  // Maximum new tokens per row.
  int max_tokens = 256;
  // Temperature; <= 0 selects greedy (argmax) decoding.
  float temperature = 1.0f;
  // Keep only the `top_k` highest logits; 0 disables the filter.
  int top_k = 0;
  // Seed for the internal generator of the sampled rows.
  std::uint64_t seed = 42;
  // Terminal token id for every row; -1 disables stopping on a token.
  int stop_id = -1;
  // Prepended to the prompt when >= 0 (for example <|bos|>).
  int bos_id = -1;
  // Optional per-row terminal ids (length `num_samples`) that override
  // `stop_id`. A negative entry disables stopping for that row.
  const int* stop_ids = nullptr;
};

// One generated row. `tokens` is the effective prompt (including a prepended
// `bos_id` when set) followed by the generated ids, and excludes the terminal
// token when one was hit; `mask` is aligned with it and holds 1 for a sampled
// id and 0 for a prompt (or prepended) id.
struct GeneratedSequence {
  std::vector<int> tokens;
  std::vector<std::uint8_t> mask;
};

// Batched sampling: prefills `prompt` once, clones the cache per row, decodes
// the rows in lockstep, and stops each row on its own terminal id (or after
// `params.max_tokens`). Writes `params.num_samples` rows to `out`, resized as
// needed. Greedy decoding is deterministic; sampled decoding uses
// `params.seed`. The prompt is not modified, and the call is forward-only.
void GenerateBatch(Model* model, const int* prompt, int prompt_len,
                   const GenerateParams& params,
                   std::vector<GeneratedSequence>* out);

}  // namespace nanochat

#endif  // NANOCHAT_MODEL_H_
