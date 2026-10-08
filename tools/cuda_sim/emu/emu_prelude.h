#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_PRELUDE_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_PRELUDE_H_

// The force-include every emulated translation unit compiles with
// (docs/simulator.md section 9.1). Two things happen here, and nothing else:
//
//   1. The CUDA runtime stub is installed before the source's own includes, so
//      `backends/cuda/device.h`'s `<cuda_runtime.h>` resolves to the emulation.
//   2. `cuda_backend::Launch` -- the one place a `.cu` file writes the
//      `<<<...>>>` launch syntax, which only nvcc parses -- is replaced by the
//      emulator's launcher. The real `device.h` guards its own definition on
//      `__CUDACC__`, which is not defined for a host compile, so there is no
//      redefinition.
//
// The seam entry points are renamed with an `Emu` prefix. The emulated library
// and the CPU reference therefore define different symbols for the same
// interface, which is what lets `//tests:kernel_emu_test` link both and compare
// the device code against the reference. A `#define` is used rather than a
// namespace rename because the kernel sources name `nanochat::kernels`
// themselves; renaming the namespace would also rename the types in
// `nanochat/tensor.h`, and the two sides would no longer be comparable.

#include "tools/cuda_sim/emu/cuda_emu.h"

#define RmsNormForward EmuRmsNormForward
#define RmsNormBackward EmuRmsNormBackward
#define RmsNormForwardFusedResidual EmuRmsNormForwardFusedResidual
#define RmsNormBackwardFusedResidual EmuRmsNormBackwardFusedResidual
#define PointwiseForward EmuPointwiseForward
#define PointwiseBackward EmuPointwiseBackward

namespace nanochat {
namespace cuda_backend {

// `backends/cuda/device.h` declares these too. The declarations must match
// exactly; the definitions live in `emu_runtime.cc`.
cudaStream_t Stream();
void CheckCuda(cudaError_t status, const char* what);
void CheckLastError(const char* what);

// The replacement for the real `cuda_backend::Launch`.
template <typename Kernel, typename... Args>
void Launch(Kernel kernel, dim3 grid, dim3 block, std::size_t shared,
            Args... args) {
  ::nanochat::emu::LaunchEmulatedKernel(kernel, grid, block, shared, args...);
}

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_EMU_PRELUDE_H_
