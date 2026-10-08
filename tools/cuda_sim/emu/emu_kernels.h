#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_KERNELS_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_KERNELS_H_

// The emulated kernel entry points (docs/simulator.md section 9).
//
// `emu_prelude.h` prefixes the seam symbols with `Emu` when it compiles a real
// `backends/cuda/kernels/*.cu` source, so this header is the adapter: it names
// what the emulated translation units export, and it lets a test link the
// emulated device code and the CPU reference side by side and compare them.
//
// The signatures are exactly the seam's. Only the symbol name differs.

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

void EmuRmsNormForward(const RmsNormParams& params, const ComputeType* x,
                       ComputeType* out, float* rstd);

void EmuRmsNormBackward(const RmsNormParams& params, const ComputeType* x,
                        const ComputeType* dy, const float* rstd,
                        ComputeType* dx);

void EmuPointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                         const ComputeType* b, float alpha, float beta,
                         ComputeType* out);

void EmuPointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                          const ComputeType* b, const ComputeType* dy,
                          float alpha, float beta, ComputeType* da,
                          ComputeType* db);

}  // namespace kernels

namespace cuda_kernels {

void EmuRmsNormForwardFusedResidual(const RmsNormParams& params,
                                    const ComputeType* x, ComputeType* residual,
                                    ComputeType* out, float* rstd);

void EmuRmsNormBackwardFusedResidual(const RmsNormParams& params,
                                     const ComputeType* residual,
                                     const float* rstd, const ComputeType* dy,
                                     ComputeType* dx, ComputeType* dresidual);

}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_KERNELS_H_
