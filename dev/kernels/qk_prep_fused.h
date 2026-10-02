#ifndef NANOCHAT_DEV_KERNELS_QK_PREP_FUSED_H_
#define NANOCHAT_DEV_KERNELS_QK_PREP_FUSED_H_

// Wave-3 fusion prototype: the QkPrep boundary (RMSNorm -> RoPE -> scale on q
// and k). The seam already exposes a fused `kernels::QkPrepForward`/`Backward`
// pair that launches once per tensor; this header adds
//
//   * a *combined* fused prototype that launches a single grid over q and k
//     together, and
//   * an explicit *decomposed* path (`RmsNormForward` + a standalone
//     RoPE-then-scale kernel) for the benchmark, so the fusion can be judged
//     against a real decomposition rather than against nothing.
//
// Everything here is dev-only scaffolding: nothing in this header is part of
// the frozen seam, and the CUDA backend is unchanged. See DESIGN.md section 3
// and dev/kernels/README.md.

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// Combined fused prototype. On entry `q`/`k` hold the pre-norm projections; on
// return they hold the normalized, rotated, and scaled activations, exactly
// like `kernels::QkPrepForward` but with q and k handled by one kernel launch.
void QkPrepFusedForward(const QkPrepParams& params, const float* cos,
                        const float* sin, ComputeType* q, ComputeType* k);

// Backward of the combined fused prototype. `dq`/`dk` are the upstream
// gradients with respect to the final activation. `q`/`k` hold the saved
// pre-norm rows on entry and are overwritten with the input gradients.
void QkPrepFusedBackward(const QkPrepParams& params, const float* cos,
                         const float* sin, const ComputeType* dq,
                         const ComputeType* dk, ComputeType* q, ComputeType* k);

// Decomposed forward: RMSNorm (the seam entry point) then a standalone
// RoPE + scale kernel. `q_in`/`k_in` are the pre-norm projections and are left
// untouched; `q_out`/`k_out` receive the final activation, while
// `q_normed`/`k_normed` are the RMSNorm outputs (scratch for the benchmark)
// and `q_rstd`/`k_rstd` are the saved statistics the backward needs.
void QkPrepDecomposedForward(const QkPrepParams& params, const float* cos,
                             const float* sin, const ComputeType* q_in,
                             const ComputeType* k_in, ComputeType* q_normed,
                             ComputeType* k_normed, ComputeType* q_out,
                             ComputeType* k_out, float* q_rstd, float* k_rstd);

// Decomposed backward: a standalone RoPE + scale backward followed by the seam
// RMSNorm backward. `dq`/`dk` are the upstream gradients, `q_in`/`k_in` the
// saved pre-norm projections, and `q_rstd`/`k_rstd` the saved statistics.
// `dq_normed`/`dk_normed` are scratch; `q_grad`/`k_grad` receive the result.
void QkPrepDecomposedBackward(const QkPrepParams& params, const float* cos,
                              const float* sin, const ComputeType* dq,
                              const ComputeType* dk, const ComputeType* q_in,
                              const ComputeType* k_in, const float* q_rstd,
                              const float* k_rstd, ComputeType* dq_normed,
                              ComputeType* dk_normed, ComputeType* q_grad,
                              ComputeType* k_grad);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_QK_PREP_FUSED_H_
