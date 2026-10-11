// The native gradient-sync adapter (docs/distributed-native-plan.md sections
// 5.4-5.6). It implements `GradientSync` with a registered window pool and the
// ring schedule. The CUDA separable compilation unit is the only place the
// collective's add kernel lives.
//
// The adapter owns the control plane (`Rendezvous`), the two ring neighbors
// (`RingTransports`), and the transfer-window pool. `AllReduceSumAsync` copies
// the caller's bucket into a free window, runs the ring reduce-scatter +
// all-gather, and copies the sum back. The host transport stages through host
// memory, so the adapter is correct on a host without NVLink peer access, which
// is the 2x T4 case. The NVLink and RDMA transports are selected when the
// hardware supports them (ring_schedule.cc SelectTransportKind).

#include "src/distributed.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/collective/host_transport.h"
#include "backends/cuda/collective/ring_schedule.h"
#include "backends/cuda/collective/transport.h"
#include "backends/cuda/device.h"
#include "nanochat/kernels.h"
#include "src/rendezvous.h"

namespace nanochat {
namespace {

void Die(const std::string& message) {
  std::fprintf(stderr, "native_sync: %s\n", message.c_str());
  std::abort();
}

std::size_t NextPowerOfTwo(std::size_t value) {
  std::size_t result = 1;
  while (result < value) result <<= 1;
  return result;
}

// Adds `src` into `dst` elementwise on the backend stream, then synchronizes so
// the host-staged transport can safely read `dst`.
__global__ void AddKernel(ComputeType* dst, const ComputeType* src,
                          long long count) {
  const long long index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) dst[index] = dst[index] + src[index];
}

void DeviceAdd(ComputeType* dst, const ComputeType* src, std::int64_t count) {
  if (count <= 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  AddKernel<<<blocks, threads, 0, cuda_backend::Stream()>>>(dst, src, count);
  cuda_backend::CheckLastError("native collective add kernel");
  kernels::Synchronize();
}

// The single-process object. Every reduction is a no-op.
class NoopGradientSync final : public GradientSync {
 public:
  int rank() const override { return 0; }
  int world_size() const override { return 1; }
  void AllReduceSum(ComputeType*, std::int64_t) override {}
};

class NativeGradientSync final : public GradientSync {
 public:
  NativeGradientSync(const DistributedConfig& config, int rank, int world_size)
      : config_(config), rank_(rank), world_size_(world_size) {}

  ~NativeGradientSync() override {
    for (Window& window : windows_) {
      if (window.ptr != nullptr) kernels::Free(window.ptr);
    }
    if (scratch_ != nullptr) kernels::Free(scratch_);
  }

  NativeGradientSync(const NativeGradientSync&) = delete;
  NativeGradientSync& operator=(const NativeGradientSync&) = delete;

  int rank() const override { return rank_; }
  int world_size() const override { return world_size_; }

  bool Setup() {
    rendezvous_ = Rendezvous::Create(config_);
    if (rendezvous_ == nullptr) return false;
    host_fabric_ =
        std::make_shared<cuda_backend::HostFabric>(rank_, world_size_);
    if (!host_fabric_->Setup(config_, rendezvous_.get())) return false;

    // Probe peer access for each ring neighbor. The host has no RDMA adapter,
    // so the rule falls back to host staging. `cudaDeviceCanAccessPeer` needs
    // the current device set; the launch sets CUDA_VISIBLE_DEVICES per rank.
    const int next_rank = (rank_ + 1) % world_size_;
    const int prev_rank = (rank_ - 1 + world_size_) % world_size_;
    std::array<cuda_backend::PeerCapabilities, 2> caps{};
    caps[0].peer_access = PeerAccessible(prev_rank);
    caps[1].peer_access = PeerAccessible(next_rank);
    caps[0].rdma_reachable = false;
    caps[1].rdma_reachable = false;

    const int pool = config_.max_inflight_reductions > 0
                         ? config_.max_inflight_reductions
                         : 1;
    windows_.resize(static_cast<std::size_t>(pool));

    std::shared_ptr<cuda_backend::HostFabric> fabric = host_fabric_;
    cuda_backend::TransportFactory factory =
        [fabric](
            cuda_backend::TransportKind kind, int self, int peer,
            bool send_direction) -> std::unique_ptr<cuda_backend::Transport> {
      // The NVLink and RDMA kinds have no tested transport on this host yet, so
      // every kind stages through host memory. The selection rule still decides
      // the kind; the implementation follows the fallback.
      (void)kind;
      return std::make_unique<cuda_backend::HostTransport>(fabric, self, peer,
                                                           send_direction);
    };
    return ring_.Connect(config_, rendezvous_.get(), caps, factory);
  }

  void AllReduceSum(ComputeType* buffer, std::int64_t count) override {
    AllReduceSumAsync(buffer, count);
    Wait();
  }

  // The synchronous window path: copy in, run the ring, copy out. The seam's
  // contract (the caller must not touch `buffer` until `Wait`) is honored
  // because the call completes before it returns.
  void AllReduceSumAsync(ComputeType* buffer, std::int64_t count) override {
    if (buffer == nullptr || count <= 0) return;
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(ComputeType);
    Window* window = AcquireWindow(bytes);
    if (window == nullptr) Die("cannot allocate a transfer window");
    kernels::Memcpy(window->ptr, buffer, bytes, CopyDir::kDeviceToDevice);
    EnsureScratch(bytes);

    const cuda_backend::TimePoint deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config_.timeout_ms > 0 ? config_.timeout_ms
                                                         : 600000);
    if (!cuda_backend::RingAllReduceSum(
            ring_.Next(), ring_.Prev(), rank_, world_size_, window->ptr, count,
            scratch_, &DeviceAdd, deadline, nullptr)) {
      Die("watchdog: rank " + std::to_string(rank_) +
          " timed out waiting for a peer during the all-reduce");
    }
    kernels::Memcpy(buffer, window->ptr, bytes, CopyDir::kDeviceToDevice);
    window->in_use = false;
  }

  void Wait() override {}

 private:
  struct Window {
    ComputeType* ptr = nullptr;
    std::size_t bytes = 0;
    bool in_use = false;
  };

  bool PeerAccessible(int peer) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, device, peer) != cudaSuccess)
      return false;
    return can != 0;
  }

  Window* AcquireWindow(std::size_t bytes) {
    for (Window& window : windows_) {
      if (!window.in_use && window.bytes >= bytes) {
        window.in_use = true;
        return &window;
      }
    }
    for (Window& window : windows_) {
      if (window.in_use) continue;
      if (window.ptr != nullptr) kernels::Free(window.ptr);
      const std::size_t capacity = NextPowerOfTwo(bytes);
      window.ptr = static_cast<ComputeType*>(kernels::Alloc(capacity));
      if (window.ptr == nullptr) return nullptr;
      window.bytes = capacity;
      window.in_use = true;
      return &window;
    }
    return nullptr;
  }

  void EnsureScratch(std::size_t bytes) {
    // Two chunks of at most `count` elements each.
    const std::size_t needed = 2 * bytes + 64;
    if (scratch_bytes_ >= needed) return;
    if (scratch_ != nullptr) kernels::Free(scratch_);
    scratch_bytes_ = NextPowerOfTwo(needed);
    scratch_ = static_cast<ComputeType*>(kernels::Alloc(scratch_bytes_));
    if (scratch_ == nullptr) Die("cannot allocate the schedule scratch");
  }

  DistributedConfig config_;
  int rank_ = 0;
  int world_size_ = 1;
  std::unique_ptr<Rendezvous> rendezvous_;
  std::shared_ptr<cuda_backend::HostFabric> host_fabric_;
  cuda_backend::RingTransports ring_;
  std::vector<Window> windows_;
  ComputeType* scratch_ = nullptr;
  std::size_t scratch_bytes_ = 0;
};

}  // namespace

std::unique_ptr<GradientSync> CreateGradientSync(
    const DistributedConfig& config) {
  if (config.world_size <= 1) return std::make_unique<NoopGradientSync>();
  if (config.rank < 0 || config.rank >= config.world_size) return nullptr;
  auto sync = std::make_unique<NativeGradientSync>(config, config.rank,
                                                   config.world_size);
  if (!sync->Setup()) return nullptr;
  return sync;
}

}  // namespace nanochat
