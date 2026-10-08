// The emulation engine's host runtime (docs/simulator.md section 9.1).
//
// `cuda_emu.h` carries the mapping table, the fiber scheduler, and the launcher
// template, because all of them must be visible where a kernel launch is
// written. This file defines the execution geometry, the fiber entry point and
// barrier release, the launch counters, and the three symbols
// `backends/cuda/device.h` declares.

#include "tools/cuda_sim/emu/cuda_emu.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace nanochat {
namespace emu {

thread_local dim3 t_thread_idx;
thread_local dim3 t_block_idx;
thread_local dim3 t_block_dim;
thread_local dim3 t_grid_dim;
thread_local BlockRuntime* g_block = nullptr;

LaunchStats& Stats() {
  static LaunchStats* stats = new LaunchStats();
  return *stats;
}

void YieldToScheduler() {
  BlockRuntime* runtime = g_block;
  Fiber* fiber =
      runtime->fibers[static_cast<std::size_t>(runtime->current)].get();
  swapcontext(&fiber->context, &runtime->scheduler);
}

namespace detail {

void FiberEntry() {
  BlockRuntime* runtime = g_block;
  Fiber* fiber =
      runtime->fibers[static_cast<std::size_t>(runtime->current)].get();
  fiber->entry();
  fiber->finished = true;
  ++runtime->finished;
  // Back to the scheduler, which never resumes a finished fiber.
  swapcontext(&fiber->context, &runtime->scheduler);
}

bool ReleaseBarriers(BlockRuntime& runtime) {
  const int count = static_cast<int>(runtime.fibers.size());
  const int live = count - runtime.finished;
  if (live == 0) return true;
  bool released = false;

  // Warp barriers first: their arrivals are per-warp, so a shuffle inside a
  // phase can complete before the block barrier does.
  int live_lanes[kMaxWarps] = {};
  for (int t = 0; t < count; ++t) {
    const Fiber* fiber = runtime.fibers[static_cast<std::size_t>(t)].get();
    if (fiber->finished) continue;
    live_lanes[fiber->index.x / kWarpSize] += 1;
  }
  for (int warp = 0; warp < kMaxWarps; ++warp) {
    if (runtime.warp_waiting[warp] == 0) continue;
    if (runtime.warp_waiting[warp] != live_lanes[warp]) continue;
    for (int t = 0; t < count; ++t) {
      Fiber* fiber = runtime.fibers[static_cast<std::size_t>(t)].get();
      if (fiber->waiting == WaitKind::kWarp && fiber->warp == warp) {
        fiber->runnable = true;
      }
    }
    runtime.warp_waiting[warp] = 0;
    released = true;
  }

  if (runtime.block_waiting == live) {
    for (int t = 0; t < count; ++t) {
      Fiber* fiber = runtime.fibers[static_cast<std::size_t>(t)].get();
      if (fiber->waiting == WaitKind::kBlock) fiber->runnable = true;
    }
    runtime.block_waiting = 0;
    released = true;
  }
  return released;
}

}  // namespace detail

}  // namespace emu

namespace cuda_backend {

cudaStream_t Stream() { return nullptr; }

void CheckCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "emu: %s failed with status %d\n", what, status);
    std::abort();
  }
}

void CheckLastError(const char* what) {
  // A launch runs to completion inside `Launch`, so there is never a pending
  // asynchronous error to collect.
  (void)what;
}

}  // namespace cuda_backend
}  // namespace nanochat
