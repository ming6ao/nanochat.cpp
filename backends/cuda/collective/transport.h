#ifndef NANOCHAT_BACKENDS_CUDA_COLLECTIVE_TRANSPORT_H_
#define NANOCHAT_BACKENDS_CUDA_COLLECTIVE_TRANSPORT_H_

// The backend-private transport interface for the native collective
// (docs/distributed-native-plan.md section 3). It is not part of the frozen
// seam, so the CUDA implementations may include CUDA headers. This header
// itself stays vendor-free so the CPU schedule test can include it.
//
// A `Transport` connects this rank to one peer. A group with mixed links holds
// one instance for each ring neighbor. The transport is completion-based: the
// NVLink implementation backs a token with stream events, the RDMA
// implementation with a work completion.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "nanochat/tensor.h"
#include "src/distributed.h"
#include "src/rendezvous.h"

namespace nanochat {
namespace cuda_backend {

using TimePoint = std::chrono::steady_clock::time_point;

// A completion handle. The transport that issued the token defines the
// encoding. A token names one posted operation: one direction, one slot, and
// one stream or queue pair. The value 0 is invalid. The token is single-use
// and bound to the issuing transport.
struct Token {
  std::uint64_t value = 0;
};

enum class WaitResult { kCompleted, kTimeout, kError };

enum class TransportKind { kNone, kNvlink, kRdma, kHost };

const char* TransportKindName(TransportKind kind);

// Moves chunks between this rank and one peer. One implementation per kind.
class Transport {
 public:
  virtual ~Transport() = default;

  virtual int self_rank() const = 0;
  virtual int peer_rank() const = 0;

  // The transport implementation family. The schedule trace records it.
  virtual TransportKind kind() const = 0;

  // Binds the control plane and opens the data plane to the one peer. Returns
  // false on failure.
  virtual bool Connect(const DistributedConfig& config,
                       Rendezvous* rendezvous) = 0;

  // Publishes the local transfer window to the peer and imports the peer's
  // window. `window` is owned by the adapter. Call once, and again after a
  // growth (section 5.4).
  virtual bool RegisterWindow(void* window, std::size_t bytes) = 0;

  // The number of receive slots for this direction. A ring needs two.
  virtual int SlotCount() const = 0;

  // Posts a send of `bytes` from `local` into the peer's receive slot `slot`.
  // The call returns at once.
  virtual Token Send(int slot, const void* local, std::size_t bytes) = 0;

  // Posts a receive of `bytes` into `local`, the local receive slot `slot`.
  virtual Token Recv(int slot, void* local, std::size_t bytes) = 0;

  // Returns true when the transfer completed. Never blocks.
  virtual bool Done(Token token) = 0;

  // Waits for the transfer until `deadline`. It returns kCompleted, kTimeout,
  // or kError. It never hangs past the deadline.
  virtual WaitResult Wait(Token token, TimePoint deadline) = 0;

  // Frees the receive slot. The transfer must be complete. The call
  // invalidates the token.
  virtual void Release(Token token) = 0;
};

// The per-edge capabilities the selection rule consults
// (docs/distributed-native-plan.md section 3.2).
struct PeerCapabilities {
  bool peer_access = false;     // cudaDeviceCanAccessPeer for the pair
  bool rdma_reachable = false;  // an RDMA adapter reaches the peer
};

// The pure selection rule. The order is NVLink, then RDMA, then host staging.
TransportKind SelectTransportKind(const PeerCapabilities& caps);

// Builds one transport of `kind`. The production factory is installed by the
// CUDA backend; the selection test injects a fake.
using TransportFactory = std::function<std::unique_ptr<Transport>(
    TransportKind kind, int self_rank, int peer_rank, bool send_direction)>;

// Holds the two ring neighbors. It builds the neighbor table at Connect.
class RingTransports {
 public:
  RingTransports() = default;
  RingTransports(const RingTransports&) = delete;
  RingTransports& operator=(const RingTransports&) = delete;

  // Connects the two ring neighbors. `caps[0]` describes the previous rank
  // (the receive direction) and `caps[1]` the next rank (the send direction).
  // `factory` builds the transport of the chosen kind. A `world_size` of 1
  // leaves both neighbors unset and returns true.
  bool Connect(const DistributedConfig& config, Rendezvous* rendezvous,
               const std::array<PeerCapabilities, 2>& caps,
               const TransportFactory& factory);

  Transport* Next() { return next_.get(); }
  Transport* Prev() { return prev_.get(); }
  TransportKind next_kind() const { return next_kind_; }
  TransportKind prev_kind() const { return prev_kind_; }

 private:
  std::unique_ptr<Transport> next_;
  std::unique_ptr<Transport> prev_;
  TransportKind next_kind_ = TransportKind::kNone;
  TransportKind prev_kind_ = TransportKind::kNone;
};

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_COLLECTIVE_TRANSPORT_H_
