#ifndef NANOCHAT_SRC_OPTIM_STATE_H_
#define NANOCHAT_SRC_OPTIM_STATE_H_

#include "nanochat/data.h"

// Optimizer-state checkpoint records (docs/post-training.md section 7). The
// AdamW first and second moments and the Muon momentum and second-moment
// buffers live in this separate list beside the model parameters. The record
// names are stable and unique, so a later load can match them to the optimizer
// groups.
//
// This header is private to `src/`. The public `Optimizer` class has no state
// surface, so `src/optim.cc` owns the records and `src/train.cc` only moves
// them between the optimizer and the `Checkpoint` container.

namespace nanochat {

class Optimizer;

// The stable name of the optimizer-step record (docs/training-seam.md
// section 10). The record sits in the optimizer-state section beside the
// moments, so a resume can continue the schedule from the stored step.
inline constexpr char kOptimizerStepRecordName[] = "optimizer/step";

// Adds the 1-based optimizer step to the optimizer-state section as one
// four-byte record. `DType` has no integer type (docs/post-training.md
// section 7), so the payload is the little-endian int32 value under an fp32
// tag. This is the driver's counter, not a per-group buffer.
void SaveOptimizerStep(int step, Checkpoint* checkpoint);

// Reads the stored optimizer step. Returns 0 when the file carries no step
// record, so a parameter-only checkpoint leaves the caller at the start of the
// schedule.
int LoadOptimizerStep(const Checkpoint& checkpoint);

// Appends the optimizer's AdamW moments and Muon buffers to `checkpoint` as
// optimizer-state records. A no-op when `optimizer` is not the native
// optimizer.
void SaveOptimizerState(const Optimizer& optimizer, Checkpoint* checkpoint);

// Copies every matching optimizer-state record from `checkpoint` into the
// optimizer. Returns true when at least one record was restored. A
// parameter-only checkpoint returns false and leaves the optimizer unchanged.
bool LoadOptimizerState(const Checkpoint& checkpoint, Optimizer* optimizer);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_OPTIM_STATE_H_
