#ifndef NANOCHAT_SRC_DISTRIBUTED_H_
#define NANOCHAT_SRC_DISTRIBUTED_H_

#include <cstdint>
#include <memory>
#include <string>

#include "nanochat/tensor.h"

// The gradient-sync seam for data parallel training
// (docs/distributed-design.md section 4). The model exposes every gradient
// through `ParamView`; the sync only needs the `grad` pointer and the `count`.
//
// This header includes only `nanochat/tensor.h` and the standard library. The
// factory lives in a backend (the host reference in `src/distributed.cc` and,
// later, NCCL in `backends/cuda`), so no vendor header is reachable here.

namespace nanochat {

struct DistributedConfig {
  int rank = 0;
  int world_size = 1;
  std::string master = "127.0.0.1";
  int port = 29500;
};

class GradientSync {
 public:
  virtual ~GradientSync() = default;

  virtual int rank() const = 0;
  virtual int world_size() const = 0;

  // Replaces `buffer` (length `count`) with the elementwise sum across every
  // rank. The sum, not the mean, keeps the global batch constant on any rank
  // count.
  virtual void AllReduceSum(ComputeType* buffer, std::int64_t count) = 0;
};

// Builds the sync for `config`. A `world_size` of 1 returns a no-op object, so
// a single-process run is unchanged. Returns null when the process cannot join
// the group.
std::unique_ptr<GradientSync> CreateGradientSync(
    const DistributedConfig& config);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_DISTRIBUTED_H_
