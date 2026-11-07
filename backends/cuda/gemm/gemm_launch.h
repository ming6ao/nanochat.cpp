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

#if defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// `C = alpha * op(A) * op(B) + beta * C`, with C stored as `ComputeType`.
bool LaunchGemmTile(const GemmParams& params, const ComputeType* a,
                    const ComputeType* b, ComputeType* c);

// The same call with C stored as fp32, which is the weight-gradient form. The
// accumulator is fp32 in both, so this changes the store and nothing else.
bool LaunchGemmTileToFp32(const GemmParams& params, const ComputeType* a,
                          const ComputeType* b, float* c);

}  // namespace kernels
}  // namespace nanochat

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_LAUNCH_H_
