// The reinforcement-learning optimizer step (docs/rl-notebook.md section 4,
// docs/training-seam.md sections 5.7 and 6.4, docs/post-training.md section
// 5.2).
//
// This translation unit is the single source of truth for the RL
// normalization. The divisor `num_valid * num_passes * examples_per_rank` is
// computed here, in C++, so no caller computes it (docs/training-seam.md
// section 5.7). The step itself is `ForwardLoss`, then `BackwardWeighted`,
// then `Optimizer::Step`.
//
// The step is shared, not copied: the file-driven `rl_step` parity binary, the
// persistent worker, and the `nanochat_rl_step` C ABI call all reach it through
// `src/rl.h`.
//
// The compute stays behind `Model` and `Optimizer`, so this file is CPU- and
// CUDA-safe and adds no vendor dependency.

#include "src/rl.h"

#include <cstdint>
#include <stdexcept>

#include "nanochat/optim.h"

namespace nanochat {

namespace {

// Number of targets that are not the ignore index (-1). This is the RL
// objective's `num_valid` (docs/post-training.md section 2.1). The classifier
// writes a zero loss for an ignored row and the weighted backward zeroes its
// gradient, so these rows do not enter the mean.
std::int64_t CountValidTargets(const int* targets, std::int64_t rows) {
  std::int64_t valid = 0;
  for (std::int64_t row = 0; row < rows; ++row) {
    if (targets[row] != -1) ++valid;
  }
  return valid;
}

}  // namespace

// The RL objective divides the policy-gradient loss by
// `num_valid * num_passes * examples_per_rank` (docs/post-training.md section
// 5.2). `BackwardWeighted` already divides by `num_valid` on its own, so the
// scale that reaches the full divisor is `num_valid / divisor`. Forming the
// full divisor here, in C++, is the point of docs/training-seam.md section 5.7:
// the normalization has a single source of truth and Python never computes it.
float RlStep(Model* model, Optimizer* optimizer, const int* tokens,
             const int* targets, const float* advantages, int batch, int seq,
             int num_passes, int examples_per_rank, int step) {
  if (model == nullptr || tokens == nullptr || targets == nullptr ||
      advantages == nullptr) {
    throw std::invalid_argument("RlStep: a required pointer is null");
  }
  if (batch <= 0 || seq <= 0) {
    throw std::invalid_argument("RlStep: batch and seq must be positive");
  }
  if (num_passes <= 0 || examples_per_rank <= 0) {
    throw std::invalid_argument(
        "RlStep: num_passes and examples_per_rank must be positive");
  }

  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;
  const std::int64_t valid = CountValidTargets(targets, rows);

  // Start from a fresh gradient; the step is self-contained.
  model->ZeroGrad();
  const float loss = model->ForwardLoss(tokens, targets, batch, seq);

  if (valid == 0) {
    // Every target is ignored, so the divisor is zero and there is no gradient
    // to take. The forward still reports the (zero) loss; do not touch the
    // optimizer state.
    return loss;
  }

  const std::int64_t divisor = valid * static_cast<std::int64_t>(num_passes) *
                               static_cast<std::int64_t>(examples_per_rank);
  const float scale = static_cast<float>(valid) / static_cast<float>(divisor);

  model->BackwardWeighted(advantages, scale);
  if (optimizer != nullptr) optimizer->Step(step);
  return loss;
}

}  // namespace nanochat
