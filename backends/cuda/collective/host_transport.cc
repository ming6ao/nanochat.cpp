// The host-staged transport for the native collective
// (docs/distributed-native-plan.md sections 3.2 and 5.4). See
// host_transport.h.

#include "backends/cuda/collective/host_transport.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
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
// A ring chunk can be tens of megabytes. A large socket buffer keeps the
// sender from blocking on the kernel's default window and cuts the syscall
// count. `TCP_NODELAY` keeps the 16-byte transfer header from waiting on
// Nagle's algorithm before the payload follows.
constexpr int kSocketBufferBytes = 4 * 1024 * 1024;

void TuneSocket(int fd) {
  int yes = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  int buffer = kSocketBufferBytes;
  ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
}

// The fixed part of a transfer: the slot index and the payload length. Written
// as two fixed-width fields so the wire format has no padding.
struct TransferHeader {
  std::uint32_t slot = 0;
  std::uint64_t bytes = 0;
};

// The installed host staging allocator. Null means the C++ allocator.
HostStagingAlloc g_host_alloc = nullptr;
HostStagingFree g_host_free = nullptr;

void* StageAlloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  if (g_host_alloc != nullptr) return g_host_alloc(bytes);
  return new char[bytes];
}

void StageFree(void* ptr) {
  if (ptr == nullptr) return;
  if (g_host_free != nullptr) {
    g_host_free(ptr);
    return;
  }
  delete[] static_cast<char*>(ptr);
}

}  // namespace

void SetHostStagingAllocator(HostStagingAlloc alloc, HostStagingFree release) {
  g_host_alloc = alloc;
  g_host_free = release;
}

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

HostTransport::HostTransport(std::shared_ptr<HostFabric> fabric, int self_rank,
                             int peer_rank, bool send_direction)
    : fabric_(std::move(fabric)),
      self_rank_(self_rank),
      peer_rank_(peer_rank),
      send_direction_(send_direction) {
  if (send_direction_) sender_ = std::thread([this] { SenderLoop(); });
}

HostTransport::~HostTransport() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (sender_.joinable()) sender_.join();
  for (Pending& pending : pending_) StageFree(pending.staging);
  StageFree(recv_staging_);
  if (fd_ >= 0) ::close(fd_);
}

bool HostTransport::Connect(const DistributedConfig&, Rendezvous* rendezvous) {
  if (fabric_ == nullptr) return false;
  if (send_direction_) {
    // The send direction connects to the peer's listener.
    const int peer_port = fabric_->peer_port(peer_rank_);
    if (peer_port <= 0) return false;
    fd_ = rendezvous::ConnectToMaster("127.0.0.1", peer_port);
    if (fd_ >= 0) TuneSocket(fd_);
    return fd_ >= 0;
  }
  // The receive direction accepts. The send side connected first, so the
  // connection is already queued in the listener backlog.
  (void)rendezvous;
  fd_ = rendezvous::AcceptPeer(fabric_->listen_fd(), kAcceptTimeoutMs);
  if (fd_ >= 0) TuneSocket(fd_);
  return fd_ >= 0;
}

std::size_t HostTransport::ReserveLocked() {
  if (!free_slots_.empty()) {
    const std::size_t index = free_slots_.back();
    free_slots_.pop_back();
    Pending& pending = pending_[index];
    // Keep the staging buffer for reuse; only reset the bookkeeping.
    char* staging = pending.staging;
    const std::size_t capacity = pending.staging_capacity;
    pending = Pending{};
    pending.staging = staging;
    pending.staging_capacity = capacity;
    return index;
  }
  pending_.emplace_back();
  return pending_.size() - 1;
}

HostTransport::Pending* HostTransport::LookupLocked(Token token) {
  if (token.value == 0 || token.value > pending_.size()) return nullptr;
  Pending& pending = pending_[static_cast<std::size_t>(token.value - 1)];
  if (!pending.in_use) return nullptr;
  return &pending;
}

Token HostTransport::Send(int slot, const void* local, std::size_t bytes) {
  std::size_t index = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    index = ReserveLocked();
    Pending& pending = pending_[index];
    pending.is_send = true;
    pending.in_use = true;
    pending.slot = slot;
    pending.bytes = bytes;
    if (bytes == 0) pending.completed = true;
  }
  if (bytes > 0) {
    // Stage the payload on the host, then hand it to the sender thread. The
    // device-to-host copy is synchronous, so the bytes are visible to the
    // sender before it is woken.
    Pending& pending = pending_[index];
    if (bytes > pending.staging_capacity) {
      StageFree(pending.staging);
      pending.staging = static_cast<char*>(StageAlloc(bytes));
      pending.staging_capacity = pending.staging != nullptr ? bytes : 0;
    }
    if (pending.staging == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      pending.completed = true;
      pending.failed = true;
      return Token{static_cast<std::uint64_t>(index + 1)};
    }
    kernels::Memcpy(pending.staging, local, bytes, CopyDir::kDeviceToHost);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      send_queue_.push_back(index);
    }
    cv_.notify_all();
  }
  return Token{static_cast<std::uint64_t>(index + 1)};
}

Token HostTransport::Recv(int slot, void* local, std::size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::size_t index = ReserveLocked();
  Pending& pending = pending_[index];
  pending.is_send = false;
  pending.in_use = true;
  pending.slot = slot;
  pending.dst = local;
  pending.bytes = bytes;
  return Token{static_cast<std::uint64_t>(index + 1)};
}

void HostTransport::SenderLoop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    cv_.wait(lock, [this] { return stop_ || !send_queue_.empty(); });
    if (send_queue_.empty()) {
      if (stop_) return;
      continue;
    }
    const std::size_t index = send_queue_.front();
    send_queue_.pop_front();
    Pending& pending = pending_[index];
    const TransferHeader header{static_cast<std::uint32_t>(pending.slot),
                                static_cast<std::uint64_t>(pending.bytes)};
    const int fd = fd_;
    // Release the lock during the write so `Done` stays non-blocking. The
    // entry cannot be released before this write completes: the caller only
    // releases after `Wait` observes completion.
    lock.unlock();
    bool ok = fd >= 0 && rendezvous::WriteAll(fd, &header, sizeof(header)) &&
              rendezvous::WriteAll(fd, pending.staging, pending.bytes);
    lock.lock();
    pending.completed = true;
    pending.failed = !ok;
    cv_.notify_all();
  }
}

bool HostTransport::Done(Token token) {
  std::lock_guard<std::mutex> lock(mutex_);
  Pending* pending = LookupLocked(token);
  if (pending == nullptr) return false;
  return pending->completed;
}

WaitResult HostTransport::Wait(Token token, TimePoint deadline) {
  std::unique_lock<std::mutex> lock(mutex_);
  Pending* pending = LookupLocked(token);
  if (pending == nullptr) return WaitResult::kError;
  if (pending->is_send) {
    if (!cv_.wait_until(lock, deadline,
                        [pending] { return pending->completed; })) {
      return WaitResult::kTimeout;
    }
    return pending->failed ? WaitResult::kError : WaitResult::kCompleted;
  }
  if (pending->completed) return WaitResult::kCompleted;
  const int fd = fd_;
  const std::size_t capacity = pending->bytes;
  void* dst = pending->dst;
  if (fd < 0) return WaitResult::kError;
  // The receive direction has no sender thread, so it is safe to hold the lock
  // across the blocking read.
  TransferHeader header;
  if (!rendezvous::ReadAll(fd, &header, sizeof(header))) {
    return WaitResult::kError;
  }
  if (header.bytes > capacity) return WaitResult::kError;
  if (header.bytes > 0) {
    const std::size_t payload = static_cast<std::size_t>(header.bytes);
    if (payload > recv_capacity_) {
      StageFree(recv_staging_);
      recv_staging_ = static_cast<char*>(StageAlloc(payload));
      recv_capacity_ = recv_staging_ != nullptr ? payload : 0;
    }
    if (recv_staging_ == nullptr) return WaitResult::kError;
    if (!rendezvous::ReadAll(fd, recv_staging_, payload)) {
      return WaitResult::kError;
    }
    kernels::Memcpy(dst, recv_staging_, payload, CopyDir::kHostToDevice);
  }
  pending->completed = true;
  return WaitResult::kCompleted;
}

void HostTransport::Release(Token token) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (token.value == 0 || token.value > pending_.size()) return;
  Pending& pending = pending_[static_cast<std::size_t>(token.value - 1)];
  if (!pending.in_use) return;
  // Keep the staging buffer; `ReserveLocked` reuses it.
  char* staging = pending.staging;
  const std::size_t capacity = pending.staging_capacity;
  pending = Pending{};
  pending.staging = staging;
  pending.staging_capacity = capacity;
  free_slots_.push_back(static_cast<std::size_t>(token.value - 1));
}

}  // namespace cuda_backend
}  // namespace nanochat
