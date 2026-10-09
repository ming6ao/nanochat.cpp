#ifndef NANOCHAT_SRC_RL_H_
#define NANOCHAT_SRC_RL_H_

#include "nanochat/model.h"

// The reinforcement-learning optimizer step (docs/rl-notebook.md section 4,
// docs/training-seam.md sections 5.7 and 6.4, docs/post-training.md section
// 5.2). It is the one place the RL normalization lives, so a Python driver
// never computes the divisor.
//
// The step is shared, not copied: the file-driven `rl_step` parity binary, the
// persistent worker, and the C ABI all call `RlStep`.

namespace nanochat {

// Runs one RL optimizer step. It is `ForwardLoss`, then `BackwardWeighted`
// with `advantages` as the per-row weight, then `Optimizer::Step(step)` when
// `optimizer` is non-null. The parameter gradients are zeroed first, so the
// step starts from a fresh gradient.
//
// The loss is normalized by the divisor of docs/post-training.md section 5.2,
// `num_valid * num_passes * examples_per_rank`, where `num_valid` is the number
// of targets that are not `-1`. The whole divisor is computed here in C++
// (docs/training-seam.md section 5.7) and never by the caller.
//
// `tokens` and `targets` hold `batch * seq` entries; `-1` in `targets` is the
// one and only mask channel (an ignored position). `advantages` holds one
// weight per row, that is `batch * seq` entries. `step` is 1-based and drives
// the optimizer schedules. Returns the forward mean loss over the valid
// targets. A null model or input buffer, a nonpositive `batch`/`seq`, or a
// nonpositive `num_passes`/`examples_per_rank` throws `std::invalid_argument`.
float RlStep(Model* model, Optimizer* optimizer, const int* tokens,
             const int* targets, const float* advantages, int batch, int seq,
             int num_passes, int examples_per_rank, int step);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_RL_H_
