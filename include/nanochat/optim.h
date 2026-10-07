#ifndef NANOCHAT_OPTIM_H_
#define NANOCHAT_OPTIM_H_

#include <memory>

#include "nanochat/model.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"

// Combined AdamW + Muon optimizer (docs/model.md). The grouping mirrors
// nanochat's `setup_optimizer`: AdamW owns embeddings, the unembedding, value
// embeddings, and scalars; Muon owns the stacked matrix parameters. The state
// lives in the optimizer, not the model.

namespace nanochat {

struct OptimizerConfig {
  float unembedding_lr = 0.004f;
  float embedding_lr = 0.2f;
  float matrix_lr = 0.02f;
  float scalar_lr = 0.5f;
  float weight_decay = 0.0f;
  int muon_ns_steps = 5;
  float muon_beta2 = 0.9f;
  float adam_eps = 1e-10f;
  // Global gradient-norm clip; <= 0 disables clipping.
  float clip = 1.0f;

  // ANVIL (docs/optimizer-anvil-design.md). `matrix_optimizer` selects the
  // stacked-matrix update: 0 = Muon (default), 1 = ANVIL. ANVIL is opt-in and
  // the two group sets are never populated together. `adam_step_period` gates
  // the AdamW groups: 1 updates every step, 2 updates on odd steps only.
  int matrix_optimizer = 0;
  int adam_step_period = 1;
  float anvil_lr = 0.023f;
  float anvil_weight_decay = 2.25f;
  float anvil_momentum = 0.95f;
  float anvil_beta2 = 0.9f;
  float anvil_fast_beta = 0.85f;
  float anvil_slow_beta = 0.98f;
  float anvil_fast_weight = 0.4385f;
  int anvil_engage_step = 514;
  int anvil_num_maps = 6;
};

class Optimizer {
 public:
  virtual ~Optimizer() = default;

  // Zeroes every parameter gradient before a backward pass.
  virtual void ZeroGrad() = 0;

  // Applies one update step; `step` is 1-based and drives the schedules.
  virtual void Step(int step) = 0;

  // Global gradient norm from the most recent step.
  virtual float GradNorm() const = 0;
};

// Builds the nanochat parameter grouping from `model->params()`.
std::unique_ptr<Optimizer> CreateOptimizer(Model* model,
                                           const OptimizerConfig& config,
                                           const Scheduler& scheduler);

}  // namespace nanochat

#endif  // NANOCHAT_OPTIM_H_
