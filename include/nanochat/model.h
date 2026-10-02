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
// evaluation entry point (docs/model.md). This header is frozen; treat it as
// stable by default and prefer additive changes (DESIGN.md section 7).
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

  // Parameter views in the optimizer's grouping order.
  virtual std::vector<ParamView> params() const = 0;

  // Self-describing checkpoint container (docs/data.md).
  virtual void Save(const std::string& path) const = 0;
  virtual void Load(const std::string& path) = 0;

 protected:
  explicit Model(Config config) : config_(std::move(config)) {}

  Config config_;
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

}  // namespace nanochat

#endif  // NANOCHAT_MODEL_H_
