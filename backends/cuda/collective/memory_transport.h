#ifndef NANOCHAT_BACKENDS_CUDA_COLLECTIVE_MEMORY_TRANSPORT_H_
#define NANOCHAT_BACKENDS_CUDA_COLLECTIVE_MEMORY_TRANSPORT_H_

// An in-process memory transport for the CPU schedule test
// (docs/distributed-native-plan.md section 3). It is not a device transport:
// it carries chunks between logical ranks that share one process, so the ring
// schedule can be exercised exactly (including the missing-peer timeout) with
// no GPU and no broker.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "backends/cuda/collective/transport.h"

namespace nanochat {
namespace cuda_backend {

// One shared mailbox per (from, to, slot). The sender writes; the receiver
// reads and clears it.
struct MemorySlot {
  std::mutex mutex;
  std::condition_variable ready_cv;
  std::condition_variable free_cv;
  std::vector<char> data;
  bool ready = false;
};

// The shared conduit every `MemoryTransport` in one process uses.
class MemoryBus {
 public:
  std::shared_ptr<MemorySlot> Slot(int from, int to, int slot);
  // Drops every mailbox. Called between test cases so a repeat run is clean.
  void Clear();

 private:
  std::mutex mutex_;
  std::map<std::tuple<int, int, int>, std::shared_ptr<MemorySlot>> slots_;
};

// One direction of one pair, backed by the shared bus.
class MemoryTransport final : public Transport {
 public:
  MemoryTransport(std::shared_ptr<MemoryBus> bus, int self_rank, int peer_rank)
      : bus_(std::move(bus)), self_rank_(self_rank), peer_rank_(peer_rank) {}

  int self_rank() const override { return self_rank_; }
  int peer_rank() const override { return peer_rank_; }
  TransportKind kind() const override { return TransportKind::kHost; }

  bool Connect(const DistributedConfig&, Rendezvous*) override { return true; }
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

  // The number of posted operations. Exposed so a test can assert the
  // lifecycle.
  int posted() const { return static_cast<int>(pending_.size()); }

 private:
  struct Pending {
    bool is_send = false;
    int slot = 0;
    std::shared_ptr<MemorySlot> mailbox;
    const void* src = nullptr;
    void* dst = nullptr;
    std::size_t bytes = 0;
    bool completed = false;
  };

  Pending* Lookup(Token token);

  std::shared_ptr<MemoryBus> bus_;
  int self_rank_ = 0;
  int peer_rank_ = 0;
  void* window_ = nullptr;
  std::size_t window_bytes_ = 0;
  std::mutex mutex_;
  std::deque<Pending> pending_;
};

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_COLLECTIVE_MEMORY_TRANSPORT_H_
