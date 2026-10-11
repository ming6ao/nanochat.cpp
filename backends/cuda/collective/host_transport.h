#ifndef NANOCHAT_BACKENDS_CUDA_COLLECTIVE_HOST_TRANSPORT_H_
#define NANOCHAT_BACKENDS_CUDA_COLLECTIVE_HOST_TRANSPORT_H_

// The host-staged transport for the native collective
// (docs/distributed-native-plan.md sections 3.2 and 5.4). It is the third
// selection choice: when neither NVLink peer access nor RDMA reaches a peer,
// the ring stages through host memory and moves bytes over a TCP connection
// between the two ring neighbors.
//
// One `HostFabric` per rank owns the listening socket and the exchanged ports.
// `RingTransports` builds one `HostTransport` per direction. The send direction
// connects to the peer's listener; the receive direction accepts.
//
// The transfer is completion-based: `Send` stages the bytes into host memory
// and hands them to a sender thread, so it returns at once as the `Transport`
// contract requires. `Recv` posts and returns; `Wait` performs the blocking
// read and stages the bytes into the caller's buffer. The sender thread is what
// keeps the ring from deadlocking: a blocking write inside `Send` would park
// both ranks before either issued its receive. This is the fallback path, so it
// favors correctness over bandwidth.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "backends/cuda/collective/transport.h"

namespace nanochat {
namespace cuda_backend {

// The host staging allocator. The CUDA backend installs a pinned allocator so
// the device-to-host copies run at PCIe speed instead of the pageable bounce
// path. The default uses the C++ allocator. Install it once before a transport
// connects; the transport keeps its staging buffers for reuse.
using HostStagingAlloc = void* (*)(std::size_t);
using HostStagingFree = void (*)(void*);
void SetHostStagingAllocator(HostStagingAlloc alloc, HostStagingFree release);

// The per-rank rendezvous and listening socket shared by the two ring
// transports.
class HostFabric {
 public:
  HostFabric(int rank, int world_size) : rank_(rank), world_size_(world_size) {}
  ~HostFabric();
  HostFabric(const HostFabric&) = delete;
  HostFabric& operator=(const HostFabric&) = delete;

  // Creates the listening socket and exchanges the per-rank ports.
  bool Setup(const DistributedConfig& config, Rendezvous* rendezvous);

  int listen_fd() const { return listen_fd_; }
  int peer_port(int peer_rank) const {
    if (peer_rank < 0 || peer_rank >= world_size_) return -1;
    return peer_ports_[static_cast<std::size_t>(peer_rank)];
  }

 private:
  int rank_ = 0;
  int world_size_ = 1;
  int listen_fd_ = -1;
  std::vector<int> peer_ports_;
};

class HostTransport final : public Transport {
 public:
  HostTransport(std::shared_ptr<HostFabric> fabric, int self_rank,
                int peer_rank, bool send_direction);
  ~HostTransport() override;

  int self_rank() const override { return self_rank_; }
  int peer_rank() const override { return peer_rank_; }
  TransportKind kind() const override { return TransportKind::kHost; }

  bool Connect(const DistributedConfig& config,
               Rendezvous* rendezvous) override;
  bool RegisterWindow(void* window, std::size_t bytes) override {
    window_ = window;
    window_bytes_ = bytes;
    return true;
  }
  int SlotCount() const override { return 2; }

  Token Send(int slot, const void* local, std::size_t bytes) override;
  Token Recv(int slot, void* local, std::size_t bytes) override;
  bool Done(Token token) override;
  WaitResult Wait(Token token, TimePoint deadline) override;
  void Release(Token token) override;

 private:
  struct Pending {
    bool is_send = false;
    bool in_use = false;
    bool completed = false;
    bool failed = false;
    int slot = 0;
    std::size_t bytes = 0;
    char* staging = nullptr;  // the send payload, staged on the host
    std::size_t staging_capacity = 0;
    void* dst = nullptr;  // the receive destination
  };

  // Returns the pending entry for `token`, or null. The caller holds the lock.
  Pending* LookupLocked(Token token);
  // The sender thread: drains `send_queue_` in order. Runs only on the send
  // direction.
  void SenderLoop();
  // Reserves a pending slot, reusing a released one when possible.
  std::size_t ReserveLocked();

  std::shared_ptr<HostFabric> fabric_;
  int self_rank_ = 0;
  int peer_rank_ = 0;
  bool send_direction_ = true;
  int fd_ = -1;
  void* window_ = nullptr;
  std::size_t window_bytes_ = 0;
  // The receive staging buffer, reused across chunks.
  char* recv_staging_ = nullptr;
  std::size_t recv_capacity_ = 0;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Pending> pending_;
  std::vector<std::size_t> free_slots_;
  std::deque<std::size_t> send_queue_;
  std::thread sender_;
  bool stop_ = false;
};

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_COLLECTIVE_HOST_TRANSPORT_H_
