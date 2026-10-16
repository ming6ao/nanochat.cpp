#ifndef NANOCHAT_SRC_MESH_COLLECTIVE_H_
#define NANOCHAT_SRC_MESH_COLLECTIVE_H_

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "src/distributed.h"

// The in-process collective for the single-client multi-device runtime
// (docs/client-runtime-design.md section 9). One `InProcessCollective` holds
// every device thread. Each thread gets one `GradientSync` view through
// `Participant`, so the existing gradient-sync seam is unchanged.
//
// The reduction stages every participant's bucket in one host buffer through
// the kernel-seam copy, uses a reusable barrier to publish the writes, and sums
// every slot in rank order. The rank order makes the sum deterministic, so
// every replica sees the same bytes. The staging copy is the correctness path;
// the peer-to-peer fast path is a later phase (docs/client-runtime-design.md
// section 9.4).

namespace nanochat {

class InProcessCollective {
 public:
  explicit InProcessCollective(int world_size);

  InProcessCollective(const InProcessCollective&) = delete;
  InProcessCollective& operator=(const InProcessCollective&) = delete;

  int world_size() const { return world_size_; }

  // The `GradientSync` view for `rank`. The collective must outlive the view.
  std::unique_ptr<GradientSync> Participant(int rank);

  // Replaces `buffer` with the elementwise sum across every participant. Every
  // participant must call this the same number of times with the same `count`.
  // `buffer` may point to device memory; the collective stages it through the
  // kernel seam. `local` is a caller-owned host scratch buffer, one per
  // participant, so no rank writes a slot another rank reads.
  void AllReduce(int rank, GradientType* buffer, std::int64_t count,
                 std::vector<GradientType>* local);

 private:
  // One reusable barrier phase. The last participant to arrive advances the
  // generation and wakes the rest.
  void WaitPhase();

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<GradientType> staging_;
  int world_size_ = 1;
  int arrived_ = 0;
  int generation_ = 0;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MESH_COLLECTIVE_H_
