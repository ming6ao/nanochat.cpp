#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_LAUNCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_LAUNCH_H_

// The launcher of the fp32 shared-memory GEMM
// (docs/kernels.md Part 3, section 6 phase 3).
//
// Backend-internal: `backends/cuda/gemm.cu` keeps the seam entry and calls this
// when the fp32 build takes the tile. The frozen seam gains no symbol.
//
// One entry, because an fp32 build stores fp32 for both the forward and the
// weight gradient: `ComputeType` is `float`, so the two seam calls name the
// same store type and the same kernel.
//
// It answers false when the tile did not take the call, which is not an error:
// the caller keeps cuBLAS.
//
// The declaration names no vendor type, so a host pass reads this header. Only
// an fp32 build defines it: a bf16 or fp8 build selects no source for
// `gemm_fp32.cu`, and the `if constexpr` in `gemm.cu` discards the only call.

#include "backends/cuda/gemm/gemm_fp32_plan.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

// `C = alpha * op(A) * op(B) + beta * C`, in fp32 throughout. `schedule`
// indexes the sweep of `gemm_fp32_plan.h`; an index outside the sweep takes the
// default.
bool LaunchGemmFp32(const GemmParams& params, const float* a, const float* b,
                    float* c, int schedule = gemm_fp32::kDefaultSchedule);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_LAUNCH_H_
