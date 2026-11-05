// The device half of the launch probe: a kernel with the fused Muon kernel's
// launch resources, and one cooperative launch of it for the host test. See
// launch_probe.h for why this translation unit exists and why it is test-only.
//
// The probe kernel is the smallest body that still spends the resources the
// query is about: its launch bounds are the fused kernel's, and it reads back
// what it wrote to dynamic shared memory so the write cannot be optimised away
// and every block must really have run before its entry is visible.

#include "backends/cuda/launch_probe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "backends/cuda/device.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace cuda_backend {
namespace testing {
namespace {

// `ran[block]` receives `block` once that block has run. The host side
// initializes every slot to -1, so a slot that still holds -1 is a block that
// did not run.
__global__ void __launch_bounds__(kProbeBlockThreads, 3)
    ProbeKernelImpl(int* __restrict__ ran) {
  extern __shared__ float shared[];
  shared[threadIdx.x] = static_cast<float>(blockIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) ran[blockIdx.x] = static_cast<int>(shared[0]);
}

// The sentinel the host side writes before a launch, one four-byte word per
// block.
constexpr int kNotRun = -1;

}  // namespace

const void* ProbeKernel() {
  return reinterpret_cast<const void*>(&ProbeKernelImpl);
}

CooperativeLaunchOutcome ProbeCooperativeLaunch(int grid_blocks,
                                                int* blocks_ran) {
  if (blocks_ran != nullptr) *blocks_ran = 0;
  if (grid_blocks <= 0) {
    // A grid with no blocks is not a launch this probe has an outcome for, and
    // `cudaLaunchCooperativeKernel` rejects it with an error about the
    // configuration rather than about the grid being too large.
    return CooperativeLaunchOutcome::kFailed;
  }

  const std::size_t bytes = static_cast<std::size_t>(grid_blocks) * sizeof(int);
  int* device_ran = static_cast<int*>(kernels::Alloc(bytes));
  kernels::Memset(device_ran, 0xFF, bytes);

  const CooperativeLaunchOutcome outcome = LaunchCooperative(
      ProbeKernelImpl, dim3(grid_blocks), dim3(kProbeBlockThreads),
      kProbeDynamicSharedBytes, device_ran);

  // The launch is asynchronous on the backend stream, so the copy back is what
  // makes the writes visible. It runs for an accepted launch only: a refused
  // one wrote nothing, which is exactly what the caller is told.
  if (outcome == CooperativeLaunchOutcome::kLaunched) {
    kernels::Synchronize();
    std::vector<int> host(static_cast<std::size_t>(grid_blocks), kNotRun);
    kernels::Memcpy(host.data(), device_ran, bytes, CopyDir::kDeviceToHost);
    int count = 0;
    for (const int block : host) {
      if (block != kNotRun) ++count;
    }
    if (blocks_ran != nullptr) *blocks_ran = count;
  }

  kernels::Free(device_ran);
  return outcome;
}

}  // namespace testing
}  // namespace cuda_backend
}  // namespace nanochat
