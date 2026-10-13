// src/mesh_collective.cc -- the host in-process all-reduce
// (docs/client-runtime-design.md section 9). See the header for the contract.

#include "src/mesh_collective.h"

#include <algorithm>
#include <cstddef>

#include "src/ops.h"

namespace nanochat {
namespace {

// The `GradientSync` view one device thread holds. It forwards to the shared
// collective, so the gradient-sync seam needs no change.
class InProcessGradientSync final : public GradientSync {
 public:
  InProcessGradientSync(InProcessCollective* group, int rank)
      : group_(group), rank_(rank) {}

  int rank() const override { return rank_; }
  int world_size() const override { return group_->world_size(); }

  void AllReduceSum(ComputeType* buffer, std::int64_t count) override {
    group_->AllReduce(rank_, buffer, count);
  }

 private:
  InProcessCollective* group_ = nullptr;
  int rank_ = 0;
};

}  // namespace

InProcessCollective::InProcessCollective(int world_size)
    : world_size_(world_size < 1 ? 1 : world_size) {}

std::unique_ptr<GradientSync> InProcessCollective::Participant(int rank) {
  return std::unique_ptr<GradientSync>(new InProcessGradientSync(this, rank));
}

void InProcessCollective::WaitPhase() {
  std::unique_lock<std::mutex> lock(mutex_);
  const int generation = generation_;
  if (++arrived_ == world_size_) {
    arrived_ = 0;
    ++generation_;
    lock.unlock();
    cv_.notify_all();
    return;
  }
  cv_.wait(lock, [this, generation] { return generation_ != generation; });
}

void InProcessCollective::AllReduce(int rank, ComputeType* buffer,
                                    std::int64_t count) {
  if (world_size_ <= 1 || buffer == nullptr || count <= 0) return;
  const std::size_t elements = static_cast<std::size_t>(count);
  const std::size_t world = static_cast<std::size_t>(world_size_);

  // Publish this rank's bucket into its staging slot. The resize happens under
  // the lock; every participant passes the same `count` for one reduction, so
  // the buffer settles after the first writer.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (staging_.size() < elements * world) {
      staging_.resize(elements * world);
    }
    std::copy(buffer, buffer + elements,
              staging_.begin() + static_cast<std::ptrdiff_t>(rank) *
                                     static_cast<std::ptrdiff_t>(elements));
  }
  WaitPhase();  // every slot is written

  // Sum every other rank's slot into this rank's buffer, in rank order.
  for (int other = 0; other < world_size_; ++other) {
    if (other == rank) continue;
    const ComputeType* slot =
        staging_.data() + static_cast<std::size_t>(other) * elements;
    for (std::size_t i = 0; i < elements; ++i) {
      buffer[i] = ToC(AsF(buffer[i]) + AsF(slot[i]));
    }
  }
  WaitPhase();  // every read is done before the next reduction
}

}  // namespace nanochat
