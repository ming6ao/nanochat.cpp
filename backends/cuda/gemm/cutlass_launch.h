#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_

// The launcher of the Hopper mainloop that NVIDIA CUTLASS supplies
// (docs/kernels.md Part 3 section 5, dev/cutlass/README.md).
//
// Backend-internal: `backends/cuda/gemm.cu` keeps the seam entry and calls one
// of these when `JudgeCutlass` accepts the call. The frozen seam gains no
// symbol.
//
// Two entries, because `GemmMode` is the caller's label and not an operand:
// `Gemm` and `GemmWgrad` differ only in the type they store. That difference is
// not cosmetic here: the store type selects one template instantiation and one
// epilogue element, so eight kernels exist rather than four.
//
// Each returns false when the kernel did not take the call, which is not an
// error. A false happens on a device that cannot run the instruction, and the
// caller keeps the hand tile and then cuBLAS.
//
// The declarations are unconditional, name no vendor type, and include no CUDA
// header, so a host pass reads this file. Only a bf16 or fp8 build defines
// them. `gemm.cu` discards the call through `if constexpr (kComputeIsBf16)`,
// exactly as it does for the hand tile.
//
// `ResolvedGemm` is the argument and not `GemmParams`, because the caller has
// already resolved every default to decide the verdict. Two resolutions of one
// call could disagree, and the disagreement would be silent.

#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// `C = alpha * op(A) * op(B) + beta * C`, with C stored as `ComputeType`.
bool LaunchGemmCutlass(const ResolvedGemm& gemm, const ComputeType* a,
                       const ComputeType* b, ComputeType* c);

// The same call with C stored as fp32, which is the weight-gradient form. The
// accumulator is fp32 in both, so this changes the store and nothing else.
bool LaunchGemmCutlassToFp32(const ResolvedGemm& gemm, const ComputeType* a,
                             const ComputeType* b, float* c);

// How many kernel instantiations the adapter holds: four operand orientations
// times two store types. A host test reads it to state the matrix is complete,
// and the answer is a constant of the translation unit and not of the card.
int GemmCutlassKernelCount();

// The tile the adapter runs, as a name a report can print.
const char* GemmCutlassTileName();

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_
