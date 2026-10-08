#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_EMU_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_EMU_H_

// Device-code emulation: the CUDA subset the kernel families use, mapped onto
// the host (docs/simulator.md section 9).
//
// The engine compiles the *real* `backends/cuda/kernels/*.cu` sources with the
// host compiler. This header supplies the device-language vocabulary they need
// -- `__global__`, `threadIdx`, `__syncthreads`, `__shared__`, `atomicAdd`, and
// the shuffles -- and a launcher that walks the grid, the block, and the
// threads of the block. The kernel bodies are unchanged: only the launch
// mechanism differs.
//
// Execution model. A CUDA block is a set of threads that cooperate through
// shared memory and `__syncthreads`, so the threads must be *concurrently
// live*, but they never need to run in parallel: every ordering the device
// allows is an ordering of a barrier-synchronized program, and a
// barrier-synchronized program is correctly executed by running its threads to
// the next barrier one at a time. The emulator therefore runs the block as a
// set of cooperative fibers on the launching host thread (`ucontext`), one per
// CUDA thread:
//
//   * The scheduler runs the runnable fibers in thread order until each either
//     finishes or blocks at a barrier.
//   * A barrier is released when every live fiber has arrived. A cooperative
//     pass that releases nothing while fibers remain blocked is a deadlock, and
//     the emulator aborts with a diagnostic rather than spinning. The abort
//     only covers a block whose fibers yield; a kernel that spins on another
//     thread's flag without yielding never returns to the scheduler and hangs
//     (see the divergence note below).
//   * Blocks run one at a time, in `(z, y, x)` order.
//   * `__shared__` maps to a function-static array. Only one block is live at a
//     time, and only the running host thread touches it, so the array is
//     exactly the block's shared window.
//   * `atomicAdd`/`atomicCAS` are host atomics. Running the block
//     cooperatively serialises them, which is one of the orderings the device
//     permits.
//
// Running the block on the launching thread -- rather than one host thread per
// CUDA thread -- is also what lets the engine run inside the test sandbox: the
// `t0-cpu` profile caps the process at 128 tasks, and a 256-thread Pointwise
// block would exceed it. A cooperative block needs exactly one.
//
// Known divergences from a device, stated plainly: the relative order of the
// threads inside a barrier phase is fixed (thread order) instead of
// unspecified; there is no memory-model ordering beyond the barriers; and a
// kernel that spins on a flag written by another thread without a barrier
// would hang, because such a spin never reaches the scheduler that detects a
// deadlock. No family in the tree does that.
//
// Not covered by the first cut: `extern __shared__` dynamic shared memory,
// texture and surface objects, and `__constant__` arrays. Attention and the
// ResFormer value gate use `extern __shared__`, so they are outside it.
//
// This header is emulator-only: it is reachable from `tools/cuda_sim/emu/**`
// and from the emulated-kernel build, never from a production target.

#include <ucontext.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// Types the kernel sources and `backends/cuda/device.h` name
// ---------------------------------------------------------------------------

// The launch geometry. Same shape and converting constructor as the CUDA type,
// so `dim3(rows)`, `dim3(block)`, and the default-constructed value all work.
struct dim3 {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
  constexpr dim3() = default;
  constexpr dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

// The runtime types `backends/cuda/device.h` references. Only the declarations
// matter: the emulator never reaches the real runtime.
using cudaError_t = int;
using cudaStream_t = void*;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};
constexpr cudaError_t cudaSuccess = 0;

// ---------------------------------------------------------------------------
// Execution geometry, read by the kernel bodies
// ---------------------------------------------------------------------------
//
// The scheduler writes these before resuming a fiber and the fiber reads them
// while it runs, so one set of variables serves every logical thread. The
// kernel sources spell them exactly as they do on a device.

namespace nanochat {
namespace emu {

extern thread_local dim3 t_thread_idx;
extern thread_local dim3 t_block_idx;
extern thread_local dim3 t_block_dim;
extern thread_local dim3 t_grid_dim;

}  // namespace emu
}  // namespace nanochat

#define threadIdx (::nanochat::emu::t_thread_idx)
#define blockIdx (::nanochat::emu::t_block_idx)
#define blockDim (::nanochat::emu::t_block_dim)
#define gridDim (::nanochat::emu::t_grid_dim)

// The device math builtins that the C library also has (`expf`, `tanhf`,
// `fmaxf`, ...) come from `<cmath>`. `rsqrtf` has no C counterpart, so it is
// defined here with the same meaning.
inline float rsqrtf(float value) { return 1.0f / std::sqrt(value); }

// ---------------------------------------------------------------------------
// Function qualifiers
// ---------------------------------------------------------------------------

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __constant__
#define __managed__
#define __launch_bounds__(...)

// `__shared__` maps to a function-static array; see the execution model above.
// `extern __shared__` is a separate form and is not supported.
#define __shared__ static

// ---------------------------------------------------------------------------
// The block runtime
// ---------------------------------------------------------------------------

namespace nanochat {
namespace emu {

// The largest block the emulator runs, matching the device's block limit.
constexpr int kMaxBlockThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kMaxBlockThreads / kWarpSize;
// The bytes one lane owns in the shuffle window. It covers every type the
// families shuffle (float and double).
constexpr int kShuffleLaneBytes = 8;
// One fiber per CUDA thread. The kernel bodies use small stacks (the
// reductions keep their arrays in `__shared__`), so this is generous.
constexpr std::size_t kFiberStackBytes = 256 * 1024;

// What a blocked fiber is waiting on.
enum class WaitKind {
  kNone = 0,
  kBlock,
  kWarp,
};

struct Fiber {
  ucontext_t context;             // resume point while blocked
  std::unique_ptr<char[]> stack;  // the fiber's own stack
  std::function<void()> entry;    // the kernel call
  dim3 index;                     // this logical thread's id
  WaitKind waiting = WaitKind::kNone;
  int warp = 0;
  bool runnable = false;
  bool finished = false;
};

// Everything one running block needs.
struct BlockRuntime {
  std::vector<std::unique_ptr<Fiber>> fibers;
  ucontext_t scheduler;
  char* shuffle = nullptr;
  int current = -1;  // the fiber that is running, or -1
  int cursor = 0;    // the scheduler's place in the thread list
  int finished = 0;
  int block_waiting = 0;
  int warp_waiting[kMaxWarps] = {};
};

extern thread_local BlockRuntime* g_block;

// Resumes `g_block->scheduler`; the scheduler resumes this fiber later.
void YieldToScheduler();

// `__syncthreads()`: blocks the calling fiber until every live fiber of the
// block has arrived at a block barrier.
inline void Syncthreads() {
  if (g_block == nullptr) return;
  Fiber* fiber =
      g_block->fibers[static_cast<std::size_t>(g_block->current)].get();
  fiber->waiting = WaitKind::kBlock;
  ++g_block->block_waiting;
  YieldToScheduler();
  fiber->waiting = WaitKind::kNone;
}

// The lane's slot base in the running block's shuffle window.
template <typename T>
inline T* WarpSlots() {
  const unsigned int warp = t_thread_idx.x / kWarpSize;
  char* base = g_block->shuffle +
               static_cast<std::size_t>(warp) * kWarpSize * kShuffleLaneBytes;
  return reinterpret_cast<T*>(base);
}

}  // namespace emu
}  // namespace nanochat

#define __syncthreads() (::nanochat::emu::Syncthreads())

// ---------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------

namespace nanochat {
namespace emu {

// Device atomics are host atomics. A cooperative block serialises them anyway;
// the lock keeps the operation correct if a host ever launches from several
// threads at once.
inline std::mutex& AtomicMutex() {
  static std::mutex* mutex = new std::mutex();
  return *mutex;
}

inline float AtomicAdd(float* address, float value) {
  std::lock_guard<std::mutex> lock(AtomicMutex());
  const float previous = *address;
  *address = previous + value;
  return previous;
}

inline unsigned int AtomicCas(unsigned int* address, unsigned int compare,
                              unsigned int value) {
  unsigned int expected = compare;
  __atomic_compare_exchange_n(address, &expected, value, false,
                              __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return expected;
}

}  // namespace emu
}  // namespace nanochat

inline float atomicAdd(float* address, float value) {
  return ::nanochat::emu::AtomicAdd(address, value);
}

inline unsigned int atomicCAS(unsigned int* address, unsigned int compare,
                              unsigned int value) {
  return ::nanochat::emu::AtomicCas(address, compare, value);
}

// ---------------------------------------------------------------------------
// Warp shuffles
// ---------------------------------------------------------------------------

namespace nanochat {
namespace emu {

// Reads the lane `lane_of` names. The write, the warp barrier, and the read
// make up one warp-wide exchange; a full mask behaves exactly as on a device.
// A second barrier follows the read: without it a released lane could overwrite
// its slot for the *next* exchange before a slower peer has read this one.
// `WarpMax`/`WarpSum` in attention.cu run five sequential `__shfl_xor_sync`
// calls and hit exactly that read-ahead hazard.
template <typename T, typename LaneFn>
inline T Shfl(unsigned int /*mask*/, T value, LaneFn lane_of) {
  Fiber* fiber =
      g_block->fibers[static_cast<std::size_t>(g_block->current)].get();
  T* slots = WarpSlots<T>();
  slots[t_thread_idx.x % kWarpSize] = value;
  const int warp = static_cast<int>(t_thread_idx.x / kWarpSize);
  fiber->waiting = WaitKind::kWarp;
  fiber->warp = warp;
  ++g_block->warp_waiting[warp];
  YieldToScheduler();
  fiber->waiting = WaitKind::kNone;
  T result = value;
  const int from = lane_of(static_cast<int>(t_thread_idx.x % kWarpSize));
  if (from >= 0 && from < kWarpSize)
    result = slots[static_cast<std::size_t>(from)];
  // The read-ahead barrier: every lane has now consumed this exchange, so the
  // next exchange may safely write the slots again.
  fiber->waiting = WaitKind::kWarp;
  fiber->warp = warp;
  ++g_block->warp_waiting[warp];
  YieldToScheduler();
  fiber->waiting = WaitKind::kNone;
  return result;
}

}  // namespace emu
}  // namespace nanochat

// XOR, down, and broadcast shuffles. The lane each device form names.
#define NANOCHAT_EMU_SHUFFLE_OVERLOADS(NAME, LANE_OF)                    \
  inline float NAME(unsigned int mask, float value, int lane) {          \
    return ::nanochat::emu::Shfl<float>(mask, value, [lane](int self) {  \
      (void)self;                                                        \
      return LANE_OF;                                                    \
    });                                                                  \
  }                                                                      \
  inline double NAME(unsigned int mask, double value, int lane) {        \
    return ::nanochat::emu::Shfl<double>(mask, value, [lane](int self) { \
      (void)self;                                                        \
      return LANE_OF;                                                    \
    });                                                                  \
  }                                                                      \
  inline int NAME(unsigned int mask, int value, int lane) {              \
    return ::nanochat::emu::Shfl<int>(mask, value, [lane](int self) {    \
      (void)self;                                                        \
      return LANE_OF;                                                    \
    });                                                                  \
  }

NANOCHAT_EMU_SHUFFLE_OVERLOADS(__shfl_xor_sync, self ^ lane)
NANOCHAT_EMU_SHUFFLE_OVERLOADS(__shfl_down_sync, self + lane)
NANOCHAT_EMU_SHUFFLE_OVERLOADS(__shfl_sync, lane)

#undef NANOCHAT_EMU_SHUFFLE_OVERLOADS

// ---------------------------------------------------------------------------
// The launcher
// ---------------------------------------------------------------------------
//
// The device `.cu` files launch through `cuda_backend::Launch` in
// `backends/cuda/device.h`, whose `<<<...>>>` form only nvcc parses. The
// emulator's prelude (`emu_prelude.h`) installs the template below before that
// header is seen, so a kernel launch in an emulated translation unit becomes a
// host call. Nothing in the kernel body changes.

namespace nanochat {
namespace emu {

// One launch record, so a run can report what it dispatched. Counters only: the
// emulation engine checks values, the interposer checks the API.
struct LaunchStats {
  std::int64_t launches = 0;
  std::int64_t threads = 0;
  std::size_t max_block_threads = 0;
  std::size_t max_dynamic_shared_bytes = 0;
};

LaunchStats& Stats();

namespace detail {

// The entry point `makecontext` calls. It reads the fiber to start from the
// runtime's `current`, so its signature is the no-argument one `makecontext`
// wants and no function-pointer cast is needed.
void FiberEntry();

// Releases every barrier whose arrivals are complete. Returns true when at
// least one fiber became runnable again.
bool ReleaseBarriers(BlockRuntime& runtime);

}  // namespace detail

// Runs one block: builds one fiber per CUDA thread and schedules them until all
// of them finish, or aborts when the block deadlocks.
template <typename Kernel, typename... Args>
void RunBlock(Kernel kernel, const dim3& block_idx, const dim3& block_dim,
              const dim3& grid_dim, char* shuffle, int threads, Args... args) {
  BlockRuntime runtime;
  runtime.shuffle = shuffle;
  runtime.fibers.reserve(static_cast<std::size_t>(threads));
  const unsigned int bx = block_dim.x > 0 ? block_dim.x : 1;
  const unsigned int by = block_dim.y > 0 ? block_dim.y : 1;
  // `makecontext` is a `setjmp` to the compiler, so the loop index lives in
  // `runtime` for the same reason the scheduler's does; see below.
  for (runtime.cursor = 0; runtime.cursor < threads; ++runtime.cursor) {
    const unsigned int index = static_cast<unsigned int>(runtime.cursor);
    auto fiber = std::make_unique<Fiber>();
    fiber->stack.reset(new char[kFiberStackBytes]);
    fiber->index = dim3(index % bx, (index / bx) % by, index / (bx * by));
    fiber->entry = [kernel, args...] { kernel(args...); };
    fiber->runnable = true;
    getcontext(&fiber->context);
    fiber->context.uc_stack.ss_sp = fiber->stack.get();
    fiber->context.uc_stack.ss_size = kFiberStackBytes;
    fiber->context.uc_link = nullptr;
    makecontext(&fiber->context, &detail::FiberEntry, 0);
    runtime.fibers.push_back(std::move(fiber));
  }

  g_block = &runtime;
  // The loop position lives in `runtime` rather than in a local: `swapcontext`
  // is a `setjmp`, and a local modified across it can be indeterminate when a
  // fiber yields (GCC's `-Wclobbered`). `runtime` is address-taken, so it stays
  // in memory.
  while (runtime.finished < threads) {
    for (runtime.cursor = 0; runtime.cursor < threads; ++runtime.cursor) {
      Fiber* fiber =
          runtime.fibers[static_cast<std::size_t>(runtime.cursor)].get();
      if (!fiber->runnable) continue;
      fiber->runnable = false;
      runtime.current = runtime.cursor;
      t_thread_idx = fiber->index;
      t_block_idx = block_idx;
      t_block_dim = block_dim;
      t_grid_dim = grid_dim;
      swapcontext(&runtime.scheduler, &fiber->context);
      runtime.current = -1;
    }
    if (runtime.finished == threads) break;
    if (!detail::ReleaseBarriers(runtime)) {
      std::fprintf(stderr,
                   "emu: block of %d threads deadlocked (block %u,%u,%u)\n",
                   threads, block_idx.x, block_idx.y, block_idx.z);
      std::abort();
    }
  }
  g_block = nullptr;
}

// The emulator's replacement for `cuda_backend::Launch`. It records the shape,
// then dispatches block by block over the same function the device would call.
template <typename Kernel, typename... Args>
void LaunchEmulatedKernel(Kernel kernel, dim3 grid, dim3 block,
                          std::size_t shared_bytes, Args... args) {
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
  if (threads == 0 || threads > kMaxBlockThreads) {
    std::fprintf(
        stderr, "emu: launch geometry invalid (block of %zu threads, max %d)\n",
        static_cast<std::size_t>(threads), kMaxBlockThreads);
    std::abort();
  }

  std::vector<char> shuffle(static_cast<std::size_t>(kMaxWarps) * kWarpSize *
                            kShuffleLaneBytes);
  for (unsigned int bz = 0; bz < grid.z; ++bz) {
    for (unsigned int by_ = 0; by_ < grid.y; ++by_) {
      for (unsigned int bx_ = 0; bx_ < grid.x; ++bx_) {
        RunBlock(kernel, dim3(bx_, by_, bz), block, grid, shuffle.data(),
                 static_cast<int>(threads), args...);
      }
    }
  }
}

}  // namespace emu
}  // namespace nanochat

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_EMU_H_
