#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_PIPELINE_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_PIPELINE_H_

// The emulator's `<cuda_pipeline.h>`: the asynchronous shared-memory copy
// (docs/simulator.md section 9.2).
//
// The copy completes at the call, and the commit and the wait are empty. A real
// wait only delays visibility, so this is one schedule the device allows. The
// engine therefore checks the staging arithmetic and not the overlap.
//
// Raw `asm volatile("cp.async...")` has no host mapping, so a kernel that needs
// the simulator leg uses these intrinsics.

#include <cstddef>
#include <cstring>

// Copies `size_and_align` bytes, and zero-fills the last `zfill` of them.
inline void __pipeline_memcpy_async(void* dst_shared, const void* src_global,
                                    std::size_t size_and_align,
                                    std::size_t zfill = 0) {
  const std::size_t copied =
      size_and_align > zfill ? size_and_align - zfill : 0;
  if (copied > 0) std::memcpy(dst_shared, src_global, copied);
  if (zfill > 0) {
    std::memset(static_cast<char*>(dst_shared) + copied, 0, zfill);
  }
}

inline void __pipeline_commit() {}

inline void __pipeline_wait_prior(std::size_t /*prior*/) {}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_PIPELINE_H_
