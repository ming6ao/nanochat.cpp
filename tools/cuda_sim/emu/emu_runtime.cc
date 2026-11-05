// The emulation engine's host runtime (docs/simulator.md section 9.1).
//
// `cuda_emu.h` carries the mapping table, the fiber scheduler, and the launcher
// template, because all of them must be visible where a kernel launch is
// written. This file defines the execution geometry, the fiber entry point and
// barrier release, the launch counters, and the runtime entry points and host
// helpers `backends/cuda/device.h` declares: the emulator has no device
// translation unit to define the latter, so the emulated library would not link
// without them, and the answers they give are the emulator's.

#include "tools/cuda_sim/emu/cuda_emu.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "backends/cuda/device.h"

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

std::size_t RecordLaunchStats(const dim3& grid, const dim3& block,
                              std::size_t shared_bytes) {
  const std::size_t threads =
      static_cast<std::size_t>(block.x) * block.y * block.z;
  LaunchStats& stats = Stats();
  ++stats.launches;
  stats.threads +=
      static_cast<std::int64_t>(threads) * grid.x * grid.y * grid.z;
  if (threads > stats.max_block_threads) stats.max_block_threads = threads;
  if (shared_bytes > stats.max_dynamic_shared_bytes) {
    stats.max_dynamic_shared_bytes = shared_bytes;
  }
  return threads;
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

// The simulated residency, in the one place the three answers below read it, so
// the occupancy query, the multiprocessor count, and the capacity a cooperative
// grid is measured against cannot disagree.
constexpr int kMultiprocessors = 1;
constexpr int kActiveBlocksPerMultiprocessor = 1;

}  // namespace emu
}  // namespace nanochat

// ---------------------------------------------------------------------------
// The three runtime entry points a fused build asks for
// ---------------------------------------------------------------------------
//
// `backends/cuda/device.cu` reaches all three as soon as the fused Muon path is
// linked, whether or not the launch it finally makes is the phase driver's
// (docs/muon-fused-kernel.md section 8), so the emulator answers them. The
// answers are the design's: 0 for `cudaDevAttrCooperativeLaunch`, which is what
// sends the emulated build to the phase driver, and 1 block per multiprocessor
// on 1 multiprocessor for the occupancy query, so the persistent grid the
// cooperative driver would size is one block. Each call is recorded in
// `Stats()` beside the launch counters, for the same reason a launch is: a run
// reports what the driver was told, not only what it dispatched.

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attribute,
                                   int device) {
  // The emulator models one device, and it is device 0, so the ordinal carries
  // no information and is not checked.
  (void)device;
  if (value == nullptr) return cudaErrorInvalidValue;
  nanochat::emu::LaunchStats& stats = nanochat::emu::Stats();
  ++stats.device_attribute_queries;
  switch (attribute) {
    case cudaDevAttrCooperativeLaunch:
      // No cooperative launch: a grid barrier over blocks the engine runs one
      // at a time is not something it can execute.
      *value = 0;
      stats.cooperative_launch_attribute = *value;
      return cudaSuccess;
    case cudaDevAttrMultiProcessorCount:
      *value = nanochat::emu::kMultiprocessors;
      stats.multiprocessors = *value;
      return cudaSuccess;
  }
  // An attribute no emulated launch path asks about. The emulation has no
  // answer, so it reports one rather than inventing a device fact. The API
  // interposer answers the same question by recording "attribute not
  // modelled" and returning success (it has a log to say it in); this engine
  // has none, so the error is the record.
  *value = 0;
  return cudaErrorInvalidValue;
}

// One block per multiprocessor is the answer the design needs: the persistence
// test in `backends/cuda/device.h` fails only on 0, and a cooperative grid is
// never sized from it here because the attribute above answers 0. A block size
// no device can run answers 0, which is the documented "this configuration
// cannot run at all" rather than a rounded-up one.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* num_blocks, const void* kernel, int block_threads,
    std::size_t dynamic_shared_bytes) {
  if (num_blocks == nullptr) return cudaErrorInvalidValue;
  // The emulation has neither a register file nor a shared-memory budget to
  // weigh against, so the kernel's own resources cannot narrow the answer.
  (void)kernel;
  (void)dynamic_shared_bytes;
  nanochat::emu::LaunchStats& stats = nanochat::emu::Stats();
  ++stats.occupancy_queries;
  const int blocks =
      block_threads > 0 && block_threads <= nanochat::emu::kMaxBlockThreads
          ? nanochat::emu::kActiveBlocksPerMultiprocessor
          : 0;
  *num_blocks = blocks;
  stats.active_blocks_per_multiprocessor = blocks;
  return cudaSuccess;
}

// A cooperative launch is recorded as a launch, not silently accepted: it
// counts in `launches` and `threads` like any other, and in
// `cooperative_launches`, so a run can see that one was asked for. No block
// runs here, for two reasons the emulator has to state -- the argument array is
// `void*[]`, which the typed launcher cannot replay, and the engine runs one
// block at a time, so a grid barrier could not be expressed. No emulated build
// reaches this call: the attribute above answers 0, so `MuonFusedUpdate` takes
// the phase driver, which is why the record is the whole effect.
cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 grid, dim3 block,
                                        void** args, std::size_t shared,
                                        cudaStream_t stream) {
  (void)func;
  (void)args;
  (void)stream;
  const std::size_t threads =
      nanochat::emu::RecordLaunchStats(grid, block, shared);
  ++nanochat::emu::Stats().cooperative_launches;
  if (threads == 0 || threads > nanochat::emu::kMaxBlockThreads) {
    std::fprintf(stderr,
                 "emu: cooperative launch geometry invalid (block of %zu "
                 "threads, max %d)\n",
                 threads, nanochat::emu::kMaxBlockThreads);
    std::abort();
  }
  return cudaSuccess;
}

// The host helpers `backends/cuda/device.h` declares, in the namespace it
// declares them in. A real build defines them in `device.cu`.
namespace nanochat {
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

void ReportCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "emu: %s failed with status %d\n", what, status);
  }
}

// The typed launch outcome, the classification, and the attribute reading are
// the same functions `device.cu` defines for a real device: they carry no
// device knowledge, and the emulated build needs them because the emulated
// kernel sources call them (`muon_fused.cu`'s driver selection) and the
// emulator compiles no device translation unit that would define them.

const char* CooperativeLaunchOutcomeName(CooperativeLaunchOutcome outcome) {
  switch (outcome) {
    case CooperativeLaunchOutcome::kLaunched:
      return "launched";
    case CooperativeLaunchOutcome::kTooLarge:
      return "cooperative-launch-too-large";
    case CooperativeLaunchOutcome::kFailed:
      return "failed";
  }
  return "unknown";
}

CooperativeLaunchOutcome ClassifyCooperativeLaunch(cudaError_t status) {
  if (status == cudaSuccess) return CooperativeLaunchOutcome::kLaunched;
  if (status == cudaErrorCooperativeLaunchTooLarge) {
    return CooperativeLaunchOutcome::kTooLarge;
  }
  return CooperativeLaunchOutcome::kFailed;
}

bool CooperativeLaunchSupported(int cooperative_launch_attribute) {
  return cooperative_launch_attribute != 0;
}

// The three queries a device answers from its driver, answered here from the
// emulator's runtime stubs. `cudaDevAttrCooperativeLaunch` answers 0, so the
// emulated fused build takes the phase loop and never reaches the grid barrier
// the block-at-a-time engine could not execute (docs/muon-fused-kernel.md
// section 8).

bool SupportsCooperativeLaunch() {
  int value = 0;
  CheckCuda(cudaDeviceGetAttribute(&value, cudaDevAttrCooperativeLaunch, 0),
            "cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch)");
  return CooperativeLaunchSupported(value);
}

int MultiprocessorCount() {
  int value = 0;
  CheckCuda(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0),
            "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)");
  return value;
}

int ActiveBlocksPerMultiprocessor(const void* kernel, int block_threads,
                                  std::size_t dynamic_shared_bytes) {
  int blocks = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, kernel, block_threads, dynamic_shared_bytes),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  return blocks < 0 ? 0 : blocks;
}

int ResidentBlocks(const void* kernel, int block_threads,
                   std::size_t dynamic_shared_bytes) {
  return ActiveBlocksPerMultiprocessor(kernel, block_threads,
                                       dynamic_shared_bytes) *
         MultiprocessorCount();
}

}  // namespace cuda_backend
}  // namespace nanochat
