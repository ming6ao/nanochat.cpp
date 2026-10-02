#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_RMS_NORM_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_RMS_NORM_H_

// Backend-internal, host-callable entry points for the fused residual RMSNorm
// variant. The frozen seam (nanochat/kernels.h) has only the plain
// RmsNormForward/RmsNormBackward pair; the residual fusion is a private helper
// the dev/kernels tests exercise, so its declarations live here rather than in
// the frozen header. See docs/kernels.md.

#include "nanochat/kernels.h"

namespace nanochat {
namespace cuda_kernels {

// Fused residual add + RMSNorm. On entry `residual` holds the incoming
// residual; on return it holds the summed residual `x + residual`, and `out`
// holds `rmsnorm(x + residual)`. `rstd` receives the per-row reciprocal
// standard deviation, exactly like RmsNormForward.
void RmsNormForwardFusedResidual(const RmsNormParams& params,
                                 const ComputeType* x, ComputeType* residual,
                                 ComputeType* out, float* rstd);

// Backward of the fused variant. `residual` is the saved summed residual from
// the forward pass. The gradient with respect to `x` and to the incoming
// residual are identical, so both `dx` and `dresidual` receive it.
void RmsNormBackwardFusedResidual(const RmsNormParams& params,
                                  const ComputeType* residual,
                                  const float* rstd, const ComputeType* dy,
                                  ComputeType* dx, ComputeType* dresidual);

}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_RMS_NORM_H_
