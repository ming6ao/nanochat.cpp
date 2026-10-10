// The host-staged transport for the native collective
// (docs/distributed-native-plan.md sections 3.2 and 5.4). See
// host_transport.h.

#include "backends/cuda/collective/host_transport.h"

#include <unistd.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "nanochat/kernels.h"
#include "src/rendezvous.h"

namespace nanochat {
namespace cuda_backend {
namespace {

constexpr int kAcceptTimeoutMs = 30000;

// The fixed part of a transfer: the slot index and the payload length. Written
// as two fixed-width fields so the wire format has no padding.
struct TransferHeader {
  std::uint32_t slot = 0;
  std::uint64_t bytes = 0;
};

}  // namespace

HostFabric::~HostFabric() {
  if (listen_fd_ >= 0) ::close(listen_fd_);
}

bool HostFabric::Setup(const DistributedConfig& config,
                       Rendezvous* rendezvous) {
  (void)config;
  if (rendezvous == nullptr) return false;
  listen_fd_ = rendezvous::ListenOnPort(0);
  if (listen_fd_ < 0) return false;
  const int port = rendezvous::LocalPort(listen_fd_);
  if (port <= 0) return false;
  peer_ports_.assign(static_cast<std::size_t>(world_size_), -1);
  peer_ports_[static_cast<std::size_t>(rank_)] = port;
  return rendezvous->ExchangeBytes(peer_ports_.data(), sizeof(peer_ports_[0]));
}

HostTransport::~HostTransport() {
  if (fd_ >= 0) ::close(fd_);
}

bool HostTransport::Connect(const DistributedConfig&, Rendezvous* rendezvous) {
  if (fabric_ == nullptr) return false;
  if (send_direction_) {
    // The send direction connects to the peer's listener.
    const int peer_port = fabric_->peer_port(peer_rank_);
    if (peer_port <= 0) return false;
    fd_ = rendezvous::ConnectToMaster("127.0.0.1", peer_port);
    return fd_ >= 0;
  }
  // The receive direction accepts. The send side connected first, so the
  // connection is already queued in the listener backlog.
  (void)rendezvous;
  fd_ = rendezvous::AcceptPeer(fabric_->listen_fd(), kAcceptTimeoutMs);
  return fd_ >= 0;
}

HostTransport::Pending* HostTransport::Lookup(Token token) {
  if (token.value == 0 || token.value > pending_.size()) return nullptr;
  return &pending_[static_cast<std::size_t>(token.value - 1)];
}

Token HostTransport::Send(int slot, const void* local, std::size_t bytes) {
  Pending pending;
  pending.is_send = true;
  pending.slot = slot;
  pending.src = local;
  pending.bytes = bytes;

  if (bytes > 0) {
    std::vector<char> staging(bytes);
    kernels::Memcpy(staging.data(), local, bytes, CopyDir::kDeviceToHost);
    TransferHeader header;
    header.slot = static_cast<std::uint32_t>(slot);
    header.bytes = static_cast<std::uint64_t>(bytes);
    if (fd_ < 0 || !rendezvous::WriteAll(fd_, &header, sizeof(header)) ||
        !rendezvous::WriteAll(fd_, staging.data(), bytes)) {
      pending.completed = false;
    } else {
      pending.completed = true;
    }
  } else {
    pending.completed = true;
  }
  pending_.push_back(std::move(pending));
  return Token{static_cast<std::uint64_t>(pending_.size())};
}

Token HostTransport::Recv(int slot, void* local, std::size_t bytes) {
  Pending pending;
  pending.is_send = false;
  pending.slot = slot;
  pending.dst = local;
  pending.bytes = bytes;
  pending_.push_back(std::move(pending));
  return Token{static_cast<std::uint64_t>(pending_.size())};
}

bool HostTransport::Done(Token token) {
  Pending* pending = Lookup(token);
  if (pending == nullptr) return false;
  return pending->completed;
}

WaitResult HostTransport::Wait(Token token, TimePoint deadline) {
  (void)deadline;
  Pending* pending = Lookup(token);
  if (pending == nullptr) return WaitResult::kError;
  if (pending->is_send) {
    return pending->completed ? WaitResult::kCompleted : WaitResult::kError;
  }
  if (pending->completed) return WaitResult::kCompleted;
  if (fd_ < 0) return WaitResult::kError;
  TransferHeader header;
  if (!rendezvous::ReadAll(fd_, &header, sizeof(header))) {
    return WaitResult::kError;
  }
  if (header.bytes > pending->bytes) return WaitResult::kError;
  if (header.bytes > 0) {
    std::vector<char> staging(static_cast<std::size_t>(header.bytes));
    if (!rendezvous::ReadAll(fd_, staging.data(),
                             static_cast<std::size_t>(header.bytes))) {
      return WaitResult::kError;
    }
    kernels::Memcpy(pending->dst, staging.data(),
                    static_cast<std::size_t>(header.bytes),
                    CopyDir::kHostToDevice);
  }
  pending->completed = true;
  return WaitResult::kCompleted;
}

void HostTransport::Release(Token) {
  // The token stays valid; a repeated Done or Wait returns the stored result.
}

}  // namespace cuda_backend
}  // namespace nanochat
