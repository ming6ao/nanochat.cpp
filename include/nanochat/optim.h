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
  // The fraction of every group rate that a run starts from (design decision D3
  // of docs/sft-parity-plan.md): the reference SFT run multiplies each group
  // rate by `init_lr_frac = 0.8` once, before the first step, and lets the
  // schedule take it from there.
  //
  // `src/optim.cc` applies it once, to every group, inside `Build`, which is
  // where the per-group rates are set: the unembedding, the embedding, the
  // value embeddings, the residual lambdas, the x0 lambdas, the smear and
  // backout group (the reference scales the smear group too, even though its
  // rate is the one constant of section 5.2's table rather than a configured
  // rate), and the Muon matrices. No call site scales a rate: `Step` multiplies
  // `group.lr` by the schedule only, so a second scaling at a call site would
  // square the factor.
  //
  // The default of 1.0 is today's behavior, so a run that does not set it (and
  // every caller of the C surface `nanochat/capi.h`, whose flat
  // `nanochat_optim_config` does not carry the field) is unchanged.
  float init_lr_frac = 1.0f;
};

class Optimizer {
 public:
  virtual ~Optimizer() = default;

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
