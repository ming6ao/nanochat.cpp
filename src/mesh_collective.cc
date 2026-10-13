// src/mesh_collective.cc -- the host in-process all-reduce
// (docs/client-runtime-design.md section 9). See the header for the contract.

#include "src/mesh_collective.h"

#include <algorithm>
#include <cstddef>

#include "nanochat/kernels.h"
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
    group_->AllReduce(rank_, buffer, count, &local_);
  }

 private:
  InProcessCollective* group_ = nullptr;
  int rank_ = 0;
  // A private host scratch buffer, so the sum never writes a slot another rank
  // reads.
  std::vector<ComputeType> local_;
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
                                    std::int64_t count,
                                    std::vector<ComputeType>* local) {
  if (world_size_ <= 1 || buffer == nullptr || count <= 0) return;
  const std::size_t elements = static_cast<std::size_t>(count);
  const std::size_t world = static_cast<std::size_t>(world_size_);

  // Publish this rank's bucket into its staging slot. The copy goes through the
  // kernel seam, so `buffer` may point to device memory. The resize and the
  // copy share the lock, so a reallocation never races a copy.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (staging_.size() < elements * world) {
      staging_.resize(elements * world);
    }
    kernels::Memcpy(staging_.data() + static_cast<std::ptrdiff_t>(rank) *
                                          static_cast<std::ptrdiff_t>(elements),
                    buffer, elements * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
  }
  WaitPhase();  // every slot is written

  // Sum every slot into a private host buffer, in rank order, then copy the sum
  // back. The rank order makes the result bit-identical on every replica.
  if (local == nullptr) {
    WaitPhase();  // keep the barrier balanced
    return;
  }
  if (local->size() < elements) local->resize(elements);
  std::copy(staging_.begin(),
            staging_.begin() + static_cast<std::ptrdiff_t>(elements),
            local->begin());
  for (int other = 1; other < world_size_; ++other) {
    const ComputeType* slot =
        staging_.data() + static_cast<std::size_t>(other) * elements;
    for (std::size_t i = 0; i < elements; ++i) {
      (*local)[i] = ToC(AsF((*local)[i]) + AsF(slot[i]));
    }
  }
  kernels::Memcpy(buffer, local->data(), elements * sizeof(ComputeType),
                  CopyDir::kHostToDevice);
  WaitPhase();  // every read is done before the next reduction
}

}  // namespace nanochat
