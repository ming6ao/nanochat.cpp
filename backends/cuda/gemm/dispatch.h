#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_

// The closed shape list of the hand tensor-core GEMM, and the one verdict that
// decides whether a call runs on it (docs/tensor-core-gemm-plan.md section
// B.6).
//
// The list holds one row per shape, and every row starts inert. A shape takes
// the tile only after a committed measurement turns its bit on, so no run can
// change behaviour without that measurement. cuBLAS serves every other shape,
// and a shape the tile does not hold exactly stays on cuBLAS.
//
// `GemmTileMode` is the measurement switch. A process reads it from
// `NANOCHAT_GEMM_LIB`, and a gate passes it in directly.

#include <string>

#include "backends/cuda/gemm/gemm_shape.h"

namespace nanochat {
namespace kernels {

enum class GemmTileMode {
  // The shape table decides. The shipped value.
  kAuto,
  // The tile takes every callable shape, for an A/B measurement. Nothing but a
  // benchmark sets this.
  kHand,
  // cuBLAS takes every call, for the other half of the same measurement.
  kCublas,
};

// `mode` as a report or a log line spells it.
const char* GemmTileModeName(GemmTileMode mode);

// The mode `NANOCHAT_GEMM_LIB` names: "hand", "cublas", or "auto" for anything
// else, including an unset variable.
GemmTileMode GemmTileModeFromEnv();

// True when this call should run on the tile.
bool GemmTileSelected(const ResolvedGemm& gemm, GemmTileMode mode);

// The rows of the closed list, in table order. One row per shape, each with the
// name the benchmark reports and the key it matches.
int GemmTileShapeCount();
const char* GemmTileShapeName(int index);
const ResolvedGemm& GemmTileShapeGemm(int index);
bool GemmTileShapeEnabled(int index);

// Turns one row on or off. The shipped table is inert, so a campaign flips a
// bit here after a measurement and a gate flips one to prove the path.
void SetGemmTileShapeEnabled(int index, bool enabled);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
