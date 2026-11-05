#ifndef NANOCHAT_BACKENDS_CUDA_LAUNCH_PROBE_H_
#define NANOCHAT_BACKENDS_CUDA_LAUNCH_PROBE_H_

// Test-only: a device translation unit that carries the fused Muon kernel's
// launch resources, so the launch surface of `device.h` can be exercised
// against a real kernel symbol.
//
// `ActiveBlocksPerMultiprocessor` and `ResidentBlocks` are questions about one
// kernel's registers and static shared memory, and a cooperative launch needs a
// kernel to launch. A host-only test can supply neither, because `__global__`
// is not available to it -- `cuda_runtime_test.cc` stays free of vendor
// constructs on purpose. This library is the one device translation unit the
// test links, and it exposes the two things a host caller cannot build: the
// symbol, and one cooperative launch of it. It is `testonly` in
// `BUILD.bazel` and nothing in `backends/cuda` links it.
//
// The quantities it carries are the design's, not this file's:
// docs/muon-fused-kernel.md section 4 asks for 256 threads, a
// `__launch_bounds__(256, 3)` register budget, and dynamic shared memory of
// about 12 KB, and section 8 asks the device how many such blocks fit.

#include <cstddef>

#include "backends/cuda/device.h"

namespace nanochat {
namespace cuda_backend {
namespace testing {

// The fused kernel's block size.
constexpr int kProbeBlockThreads = 256;

// The dynamic shared-memory request that reproduces the fused kernel's
// three-blocks-per-SM budget on the host card.
//
// The fused kernel reaches three blocks per SM out of registers (85 per thread
// at 256 threads and `__launch_bounds__(256, 3)`), and a probe kernel cannot
// ask ptxas for a register count. It spends the same budget out of shared
// memory instead: the host card's sweeping multiprocessor has 96 KB, so 32 KB
// is the third that leaves three blocks resident. 32 KB is also under the 48 KB
// a block may take, so the launch needs no `cudaFuncSetAttribute`, and the
// answer is the same 3 per SM the design sizes the persistent grid from.
constexpr std::size_t kProbeDynamicSharedBytes = 32u * 1024u;

// The address of the probe kernel, for
// `ActiveBlocksPerMultiprocessor`/`ResidentBlocks`.
const void* ProbeKernel();

// One cooperative launch of the probe kernel over `grid_blocks` blocks at
// `kProbeBlockThreads` threads and `kProbeDynamicSharedBytes` of dynamic shared
// memory, on the backend stream. Returns the typed outcome `LaunchCooperative`
// produced. When the launch was accepted, `*blocks_ran` receives how many
// blocks actually wrote, which is what tells a launch that ran from one the
// driver merely accepted; it is 0 for every refused launch.
//
// A refused launch is not an error here: the test asks for one, and
// `cudaErrorCooperativeLaunchTooLarge` is the answer it expects.
CooperativeLaunchOutcome ProbeCooperativeLaunch(int grid_blocks,
                                                int* blocks_ran);

}  // namespace testing
}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_LAUNCH_PROBE_H_
