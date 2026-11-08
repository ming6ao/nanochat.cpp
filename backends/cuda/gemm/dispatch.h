#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_

// The closed shape list of the hand tensor-core GEMM, and the one verdict that
// decides whether a call runs on it (docs/tensor-core-gemm-plan.md section
// B.6).
//
// The list holds one row per shape of the closed list: thirteen Muon rows and
// fifteen Linear rows. Every row is inert, because the sweep measured each
// shape against cuBLAS and no shape reached the promotion gate
// (`backends/cuda/gemm/README.md` section 8). A shape takes the tile only after
// a committed measurement turns its bit on, so no run can change behaviour
// without that measurement. cuBLAS serves every other shape, and a shape the
// tile does not hold exactly stays on cuBLAS.
//
// `GemmTileMode` is the measurement switch. A process reads it from
// `NANOCHAT_GEMM_LIB`, and a gate passes it in directly.
//
// `NANOCHAT_GEMM_SCHEDULE` names one schedule of `gemm_plan.h` and overrides
// the row's own, which is how a card run sweeps one shape over the sweep.

#include <string>

#include "backends/cuda/gemm/gemm_shape.h"

namespace nanochat {
namespace kernels {

enum class GemmTileMode {
  // The shape table decides. The shipped value.
  kAuto,
  // The Hopper mainloop of `cutlass_bf16.cu` takes every shape its own rule
  // accepts. An A/B measurement sets this, and it is the value a promotion
  // would leave on.
  kCutlass,
  // The hand tile takes every callable shape, for an A/B measurement. Nothing
  // but a benchmark sets this.
  kHand,
  // cuBLAS takes every call, for the other half of the same measurement.
  kCublas,
};

// `mode` as a report or a log line spells it.
const char* GemmTileModeName(GemmTileMode mode);

// The mode `NANOCHAT_GEMM_LIB` names: "cutlass", "hand", "cublas", or "auto"
// for anything else, including an unset variable.
GemmTileMode GemmTileModeFromEnv();

// The schedule this call takes, or -1 when cuBLAS serves it.
int GemmTileScheduleFor(const ResolvedGemm& gemm, GemmTileMode mode);

// True when `GemmTileScheduleFor` names a schedule.
bool GemmTileSelected(const ResolvedGemm& gemm, GemmTileMode mode);

// The sweep of `gemm_plan.h`, by name.
int GemmScheduleCount();
const char* GemmScheduleName(int index);

// The index `NANOCHAT_GEMM_SCHEDULE` names, or -1 when the variable is unset,
// empty, or names no schedule.
int GemmScheduleIndexFromEnv();

// The rows of the closed list, in table order. One row per shape, each with the
// name the benchmark reports and the key it matches.
int GemmTileShapeCount();
const char* GemmTileShapeName(int index);
const ResolvedGemm& GemmTileShapeGemm(int index);
bool GemmTileShapeEnabled(int index);
int GemmTileShapeSchedule(int index);

// Turns one row on or off. The shipped table is inert, so a campaign flips a
// bit here after a measurement and a gate flips one to prove the path. The
// row's schedule is already the one its measurement chose.
void SetGemmTileShapeEnabled(int index, bool enabled);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
