// The ring reduce-scatter + all-gather schedule for the native collective
// (docs/distributed-native-plan.md section 5). Pure host logic; see
// ring_schedule.h.

#include "backends/cuda/collective/ring_schedule.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace nanochat {
namespace cuda_backend {

const char* TransportKindName(TransportKind kind) {
  switch (kind) {
    case TransportKind::kNone:
      return "none";
    case TransportKind::kNvlink:
      return "nvlink";
    case TransportKind::kRdma:
      return "rdma";
    case TransportKind::kHost:
      return "host";
  }
  return "?";
}

const char* ScheduleOpName(ScheduleOp op) {
  switch (op) {
    case ScheduleOp::kSend:
      return "send";
    case ScheduleOp::kRecv:
      return "recv";
    case ScheduleOp::kAdd:
      return "add";
  }
  return "?";
}

TransportKind SelectTransportKind(const PeerCapabilities& caps) {
  if (caps.peer_access) return TransportKind::kNvlink;
  if (caps.rdma_reachable) return TransportKind::kRdma;
  return TransportKind::kHost;
}

bool RingTransports::Connect(const DistributedConfig& config,
                             Rendezvous* rendezvous,
                             const std::array<PeerCapabilities, 2>& caps,
                             const TransportFactory& factory) {
  if (config.world_size <= 1) return true;
  if (config.rank < 0 || config.rank >= config.world_size) return false;
  if (!factory) return false;
  const int next_rank = (config.rank + 1) % config.world_size;
  const int prev_rank =
      (config.rank - 1 + config.world_size) % config.world_size;

  prev_kind_ = SelectTransportKind(caps[0]);
  next_kind_ = SelectTransportKind(caps[1]);
  prev_ = factory(prev_kind_, config.rank, prev_rank, /*send_direction=*/false);
  next_ = factory(next_kind_, config.rank, next_rank, /*send_direction=*/true);
  if (prev_ == nullptr || next_ == nullptr) return false;
  // The send direction connects first; the receive direction accepts. Connect
  // completes through the listener's backlog, so this order works even though
  // the accept is posted second.
  if (!next_->Connect(config, rendezvous)) return false;
  if (!prev_->Connect(config, rendezvous)) return false;
  return true;
}

std::int64_t RingChunkSize(std::int64_t count, int world_size) {
  if (world_size <= 0) return 0;
  return (count + world_size - 1) / world_size;
}

namespace {

bool WaitAndRelease(Transport* transport, Token token, TimePoint deadline) {
  const WaitResult result = transport->Wait(token, deadline);
  if (result != WaitResult::kCompleted) return false;
  transport->Release(token);
  return true;
}

int ChunkLength(std::int64_t count, std::int64_t chunk, int index) {
  const std::int64_t offset = static_cast<std::int64_t>(index) * chunk;
  if (offset >= count) return 0;
  return static_cast<int>(std::min(chunk, count - offset));
}

}  // namespace

bool RingAllReduceSum(Transport* next, Transport* prev, int rank,
                      int world_size, ComputeType* buffer, std::int64_t count,
                      ComputeType* scratch, AddFn add, TimePoint deadline,
                      ScheduleTrace* trace) {
  if (world_size <= 1 || count <= 0) return true;
  if (next == nullptr || prev == nullptr || add == nullptr) return false;
  const int n = world_size;
  const std::int64_t chunk = RingChunkSize(count, n);
  const std::size_t element = sizeof(ComputeType);
  const int next_rank = next->peer_rank();
  const int prev_rank = prev->peer_rank();
  auto offset = [chunk](int c) { return static_cast<std::int64_t>(c) * chunk; };
  auto emit = [&](ScheduleOp op, int step, int c, int slot, TransportKind kind,
                  int peer) {
    if (trace == nullptr) return;
    ScheduleEvent event;
    event.rank = rank;
    event.step = step;
    event.chunk = c;
    event.slot = slot;
    event.op = op;
    event.kind = kind;
    event.peer = peer;
    trace->push_back(event);
  };

  // Reduce-scatter: N-1 steps. Each rank ends holding the fully reduced chunk
  // `(rank + 1) % N`. The slot index alternates over the whole schedule, not
  // per phase, so the reduce-scatter/all-gather boundary does not reuse a slot
  // before the peer consumed it.
  int step = 0;
  for (int s = 0; s < n - 1; ++s, ++step) {
    const int recv_chunk = ((rank - s - 1) % n + n) % n;
    const int send_chunk = ((rank - s) % n + n) % n;
    const int slot = step % 2;
    const int recv_len = ChunkLength(count, chunk, recv_chunk);
    const int send_len = ChunkLength(count, chunk, send_chunk);
    ComputeType* recv_dst = scratch + static_cast<std::int64_t>(slot) * chunk;
    Token recv_token = prev->Recv(slot, recv_dst,
                                  static_cast<std::size_t>(recv_len) * element);
    Token send_token = next->Send(slot, buffer + offset(send_chunk),
                                  static_cast<std::size_t>(send_len) * element);
    emit(ScheduleOp::kRecv, step, recv_chunk, slot, prev->kind(), prev_rank);
    emit(ScheduleOp::kSend, step, send_chunk, slot, next->kind(), next_rank);
    if (!WaitAndRelease(prev, recv_token, deadline)) return false;
    if (!WaitAndRelease(next, send_token, deadline)) return false;
    if (recv_len > 0) add(buffer + offset(recv_chunk), recv_dst, recv_len);
    emit(ScheduleOp::kAdd, step, recv_chunk, slot, next->kind(), rank);
  }

  // All-gather: N-1 steps. The receive replaces the local chunk directly.
  for (int s = 0; s < n - 1; ++s, ++step) {
    const int recv_chunk = ((rank - s) % n + n) % n;
    const int send_chunk = ((rank - s + 1) % n + n) % n;
    const int slot = step % 2;
    const int recv_len = ChunkLength(count, chunk, recv_chunk);
    const int send_len = ChunkLength(count, chunk, send_chunk);
    Token recv_token = prev->Recv(slot, buffer + offset(recv_chunk),
                                  static_cast<std::size_t>(recv_len) * element);
    Token send_token = next->Send(slot, buffer + offset(send_chunk),
                                  static_cast<std::size_t>(send_len) * element);
    emit(ScheduleOp::kRecv, step, recv_chunk, slot, prev->kind(), prev_rank);
    emit(ScheduleOp::kSend, step, send_chunk, slot, next->kind(), next_rank);
    if (!WaitAndRelease(prev, recv_token, deadline)) return false;
    if (!WaitAndRelease(next, send_token, deadline)) return false;
  }
  return true;
}

bool JudgeScheduleTrace(const ScheduleTrace& trace, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  int last_step[4096];
  int last_recv_slot[4096];
  int last_send_slot[4096];
  for (int& v : last_step) v = -1;
  for (int& v : last_recv_slot) v = -2;
  for (int& v : last_send_slot) v = -2;
  for (const ScheduleEvent& event : trace) {
    if (event.rank < 0 || event.rank >= 4096) {
      return fail("trace: rank out of range");
    }
    if (event.op != ScheduleOp::kAdd && (event.slot < 0 || event.slot > 1)) {
      return fail("trace: invalid slot");
    }
    if (event.step < last_step[event.rank]) {
      return fail("trace: step order regressed");
    }
    last_step[event.rank] = event.step;
    if (event.op == ScheduleOp::kRecv) {
      const int expected = (event.step % 2);
      if (event.slot != expected) {
        return fail("trace: receive slot does not alternate");
      }
      last_recv_slot[event.rank] = event.slot;
    } else if (event.op == ScheduleOp::kSend) {
      const int expected = (event.step % 2);
      if (event.slot != expected) {
        return fail("trace: send slot does not alternate");
      }
      last_send_slot[event.rank] = event.slot;
    }
  }
  // Every send must have a matching receive at the peer for the same step and
  // slot. The chunk differs because the sender sends and the receiver receives
  // different ring chunks.
  for (const ScheduleEvent& send : trace) {
    if (send.op != ScheduleOp::kSend) continue;
    bool matched = false;
    for (const ScheduleEvent& recv : trace) {
      if (recv.op != ScheduleOp::kRecv) continue;
      if (recv.rank == send.peer && recv.peer == send.rank &&
          recv.step == send.step && recv.slot == send.slot) {
        matched = true;
        break;
      }
    }
    if (!matched) return fail("trace: a send has no matching receive");
  }
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace cuda_backend
}  // namespace nanochat
