#ifndef NANOCHAT_BACKENDS_CUDA_COLLECTIVE_RING_SCHEDULE_H_
#define NANOCHAT_BACKENDS_CUDA_COLLECTIVE_RING_SCHEDULE_H_

// The ring reduce-scatter + all-gather schedule for the native collective
// (docs/distributed-native-plan.md section 5). It is pure host logic: it calls
// only the `Transport` interface (`Send`, `Recv`, `Done`, `Wait`, `Release`)
// and the caller's elementwise add. A memory transport exercises it exactly,
// so the algorithm is testable on the CPU.
//
// The schedule operates on a caller-owned `scratch` region of two chunks. The
// caller also supplies `add`, which must add `src` into `dst` elementwise. The
// add runs on the buffer's device (the CUDA adapter installs a kernel; the CPU
// test installs a host loop).

#include <cstdint>
#include <string>
#include <vector>

#include "backends/cuda/collective/transport.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace cuda_backend {

// Adds `src` into `dst` elementwise. The schedule's only compute primitive.
using AddFn = void (*)(ComputeType* dst, const ComputeType* src,
                       std::int64_t count);

// One recorded schedule operation. The trace is the correctness artifact for
// the two-stream design (docs/distributed-native-plan.md section 10).
enum class ScheduleOp { kSend, kRecv, kAdd };

const char* ScheduleOpName(ScheduleOp op);

struct ScheduleEvent {
  int rank = 0;
  int step = 0;
  int chunk = 0;
  int slot = 0;
  ScheduleOp op = ScheduleOp::kSend;
  TransportKind kind = TransportKind::kNone;
  int peer = -1;
};

using ScheduleTrace = std::vector<ScheduleEvent>;

// Checks the trace against the four rules of section 10:
//   1. every slot is used by one operation at a time;
//   2. the step order is monotonic (the dependency graph has no cycle);
//   3. every receive has a matching send of the same chunk;
//   4. no event names an invalid slot.
// Returns true and leaves `error` empty when the trace is consistent.
bool JudgeScheduleTrace(const ScheduleTrace& trace, std::string* error);

// The ring all-reduce of `buffer` (length `count`) across `world_size` ranks.
// `next` is the send direction and `prev` the receive direction. `rank` is this
// rank's index. `scratch` holds at least `2 * chunk` elements, where `chunk` is
// the schedule's chunk size. `trace` is optional.
//
// Returns true when every rank is the elementwise sum, false on timeout or
// error. A `world_size` of 1 or a `count` of 0 is a no-op.
bool RingAllReduceSum(Transport* next, Transport* prev, int rank,
                      int world_size, ComputeType* buffer, std::int64_t count,
                      ComputeType* scratch, AddFn add, TimePoint deadline,
                      ScheduleTrace* trace);

// The chunk size the schedule uses for `count` and `world_size`. Exposed so the
// adapter can size its scratch region.
std::int64_t RingChunkSize(std::int64_t count, int world_size);

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_COLLECTIVE_RING_SCHEDULE_H_
