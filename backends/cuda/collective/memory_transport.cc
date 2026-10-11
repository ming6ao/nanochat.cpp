// An in-process memory transport for the CPU schedule test
// (docs/distributed-native-plan.md section 3). See memory_transport.h.

#include "backends/cuda/collective/memory_transport.h"

#include <cstring>
#include <utility>

namespace nanochat {
namespace cuda_backend {

std::shared_ptr<MemorySlot> MemoryBus::Slot(int from, int to, int slot) {
  const auto key = std::make_tuple(from, to, slot);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = slots_.find(key);
  if (it == slots_.end()) {
    auto created = std::make_shared<MemorySlot>();
    it = slots_.emplace(key, created).first;
  }
  return it->second;
}

void MemoryBus::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  slots_.clear();
}

MemoryTransport::Pending* MemoryTransport::Lookup(Token token) {
  if (token.value == 0 || token.value > pending_.size()) return nullptr;
  return &pending_[static_cast<std::size_t>(token.value - 1)];
}

Token MemoryTransport::Send(int slot, const void* local, std::size_t bytes) {
  Pending pending;
  pending.is_send = true;
  pending.slot = slot;
  pending.src = local;
  pending.bytes = bytes;
  pending.mailbox = bus_->Slot(self_rank_, peer_rank_, slot);
  {
    std::unique_lock<std::mutex> lock(pending.mailbox->mutex);
    // Wait for the peer to consume the previous contents of this slot. With
    // two slots and the ring's step order the wait is normally zero, but it
    // removes the overwrite race at the reduce-scatter/all-gather boundary.
    pending.mailbox->free_cv.wait(lock,
                                  [&] { return !pending.mailbox->ready; });
    pending.mailbox->data.assign(static_cast<const char*>(local),
                                 static_cast<const char*>(local) + bytes);
    pending.mailbox->ready = true;
  }
  pending.mailbox->ready_cv.notify_all();
  pending.completed = true;
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(std::move(pending));
  return Token{static_cast<std::uint64_t>(pending_.size())};
}

Token MemoryTransport::Recv(int slot, void* local, std::size_t bytes) {
  Pending pending;
  pending.is_send = false;
  pending.slot = slot;
  pending.dst = local;
  pending.bytes = bytes;
  pending.mailbox = bus_->Slot(peer_rank_, self_rank_, slot);
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(std::move(pending));
  return Token{static_cast<std::uint64_t>(pending_.size())};
}

bool MemoryTransport::Done(Token token) {
  Pending* pending = Lookup(token);
  if (pending == nullptr) return false;
  if (pending->is_send) return true;
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending->completed) return true;
  std::lock_guard<std::mutex> mailbox_lock(pending->mailbox->mutex);
  return pending->mailbox->ready;
}

WaitResult MemoryTransport::Wait(Token token, TimePoint deadline) {
  Pending* pending = Lookup(token);
  if (pending == nullptr) return WaitResult::kError;
  if (pending->is_send) return WaitResult::kCompleted;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending->completed) return WaitResult::kCompleted;
  }
  std::unique_lock<std::mutex> lock(pending->mailbox->mutex);
  const bool ready = pending->mailbox->ready_cv.wait_until(
      lock, deadline, [&] { return pending->mailbox->ready; });
  if (!ready) return WaitResult::kTimeout;
  if (pending->bytes > 0) {
    std::memcpy(pending->dst, pending->mailbox->data.data(), pending->bytes);
  }
  pending->mailbox->ready = false;
  pending->completed = true;
  pending->mailbox->free_cv.notify_all();
  return WaitResult::kCompleted;
}

void MemoryTransport::Release(Token) {
  // The mailbox is per (from, to, slot); completion already cleared it. The
  // token stays valid so a second `Done` or `Wait` still reads the result.
}

}  // namespace cuda_backend
}  // namespace nanochat
