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
// The transfer is completion-based: `Recv` posts and returns, `Wait` performs
// the blocking read and stages the bytes into the caller's buffer. This is the
// fallback path, so it favors correctness over bandwidth.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "backends/cuda/collective/transport.h"

namespace nanochat {
namespace cuda_backend {

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
                int peer_rank, bool send_direction)
      : fabric_(std::move(fabric)),
        self_rank_(self_rank),
        peer_rank_(peer_rank),
        send_direction_(send_direction) {}
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
    int slot = 0;
    const void* src = nullptr;
    void* dst = nullptr;
    std::size_t bytes = 0;
    bool completed = false;
  };
  Pending* Lookup(Token token);

  std::shared_ptr<HostFabric> fabric_;
  int self_rank_ = 0;
  int peer_rank_ = 0;
  bool send_direction_ = true;
  int fd_ = -1;
  void* window_ = nullptr;
  std::size_t window_bytes_ = 0;
  std::vector<Pending> pending_;
};

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_COLLECTIVE_HOST_TRANSPORT_H_
