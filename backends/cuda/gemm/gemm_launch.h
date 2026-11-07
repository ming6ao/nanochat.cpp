#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_LAUNCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_LAUNCH_H_

// The launcher of the hand-written tensor-core GEMM
// (docs/kernels.md Part 3, docs/tensor-core-gemm-plan.md Part B).
//
// Backend-internal: `backends/cuda/gemm.cu` keeps the seam entry and calls one
// of these when its shape table selects the tile. The frozen seam gains no
// symbol.
//
// Two entries, because `GemmMode` is the caller's label and not an operand:
// `Gemm` and `GemmWgrad` differ only in the type they store.
//
// Each returns false when the tile did not take the call, which is not an
// error: the caller keeps cuBLAS.
//
// The declarations are unconditional and name no vendor type, so a host pass
// reads this header. Only a bf16 or fp8 build defines them: an fp32 build links
// no tile, and the `if constexpr` in `gemm.cu` discards the only call.

#include "backends/cuda/gemm/gemm_plan.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// `C = alpha * op(A) * op(B) + beta * C`, with C stored as `ComputeType`.
// `schedule` indexes the sweep of `gemm_plan.h`; the default is the base
// schedule, and an index outside the sweep takes the default.
bool LaunchGemmTile(const GemmParams& params, const ComputeType* a,
                    const ComputeType* b, ComputeType* c,
                    int schedule = gemm_tile::kDefaultSchedule);

// The same call with C stored as fp32, which is the weight-gradient form. The
// accumulator is fp32 in both, so this changes the store and nothing else.
bool LaunchGemmTileToFp32(const GemmParams& params, const ComputeType* a,
                          const ComputeType* b, float* c,
                          int schedule = gemm_tile::kDefaultSchedule);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_LAUNCH_H_
