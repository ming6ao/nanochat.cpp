// The NCCL gradient-sync backend (docs/distributed-design.md section 5).
//
// This is the only translation unit that includes `nccl.h`. The factory
// `CreateGradientSync` here replaces the host reference on a CUDA build,
// because `//src:model` selects this library for `//:backend_cuda`
// (docs/distributed-plan.md phase 0).
//
// The ranks rendezvous over the `master`:`port` pair that the command line
// supplies. Rank 0 generates the `ncclUniqueId` and broadcasts it to the peers
// through the vendor-free `Rendezvous` all-gather
// (docs/distributed-native-plan.md section 4). Every rank then calls
// `ncclCommInitRank`.
//
// The launch sets `CUDA_VISIBLE_DEVICES` to the rank, so each process sees one
// card as device 0. The code therefore selects device 0.

#include "src/distributed.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "backends/cuda/device.h"
#include "nanochat/kernels.h"
#include "src/rendezvous.h"

namespace nanochat {
namespace {

#if defined(NANOCHAT_PRECISION_FP16)
constexpr ncclDataType_t kNcclType = ncclHalf;
#else
constexpr ncclDataType_t kNcclType = ncclFloat;
#endif

void Die(const std::string& message) {
  std::fprintf(stderr, "nccl_sync: %s\n", message.c_str());
  std::abort();
}

// Broadcasts the rank 0 unique id through the vendor-free control plane. Every
// rank calls `ExchangeBytes` with the same length, so the call is collective;
// only slot 0 (rank 0's id) is meaningful.
bool BroadcastUniqueId(const DistributedConfig& config, ncclUniqueId* id) {
  std::unique_ptr<Rendezvous> rendezvous = Rendezvous::Create(config);
  if (rendezvous == nullptr) return false;
  std::vector<char> table(sizeof(ncclUniqueId) *
                          static_cast<std::size_t>(config.world_size));
  std::memcpy(table.data() +
                  static_cast<std::size_t>(config.rank) * sizeof(ncclUniqueId),
              id, sizeof(ncclUniqueId));
  if (!rendezvous->ExchangeBytes(table.data(), sizeof(ncclUniqueId))) {
    return false;
  }
  std::memcpy(id, table.data(), sizeof(ncclUniqueId));
  return true;
}

// The single-process object. Every reduction is a no-op.
class NoopGradientSync final : public GradientSync {
 public:
  int rank() const override { return 0; }
  int world_size() const override { return 1; }
  void AllReduceSum(ComputeType*, std::int64_t) override {}
};

class NcclGradientSync final : public GradientSync {
 public:
  NcclGradientSync(int rank, int world_size)
      : rank_(rank), world_size_(world_size) {}

  ~NcclGradientSync() override {
    if (comm_ != nullptr) ncclCommDestroy(comm_);
    if (ready_event_ != nullptr) cudaEventDestroy(ready_event_);
    if (side_stream_ != nullptr) cudaStreamDestroy(side_stream_);
  }

  NcclGradientSync(const NcclGradientSync&) = delete;
  NcclGradientSync& operator=(const NcclGradientSync&) = delete;

  int rank() const override { return rank_; }
  int world_size() const override { return world_size_; }

  bool Setup(const ncclUniqueId& id) {
    // The launch sets `CUDA_VISIBLE_DEVICES` to the rank, so device 0 is the
    // rank's card.
    const cudaError_t device = cudaSetDevice(0);
    if (device != cudaSuccess) return false;
    if (cudaStreamCreateWithFlags(&side_stream_, cudaStreamNonBlocking) !=
        cudaSuccess) {
      return false;
    }
    if (cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming) !=
        cudaSuccess) {
      return false;
    }
    return ncclCommInitRank(&comm_, world_size_, id, rank_) == ncclSuccess;
  }

  void AllReduceSum(ComputeType* buffer, std::int64_t count) override {
    AllReduceSumAsync(buffer, count);
    Wait();
  }

  // The reduction runs on a side stream, so the pack copy of the next bucket
  // on the main stream overlaps it (docs/distributed-plan.md phase 2). The
  // side stream waits on an event recorded after the caller produced `buffer`
  // on the main stream.
  void AllReduceSumAsync(ComputeType* buffer, std::int64_t count) override {
    if (buffer == nullptr || count <= 0) return;
    cuda_backend::CheckCuda(
        cudaEventRecord(ready_event_, cuda_backend::Stream()),
        "record the gradient-ready event");
    cuda_backend::CheckCuda(cudaStreamWaitEvent(side_stream_, ready_event_, 0),
                            "make the side stream wait for the gradient");
    const ncclResult_t status =
        ncclAllReduce(buffer, buffer, static_cast<std::size_t>(count),
                      kNcclType, ncclSum, comm_, side_stream_);
    if (status != ncclSuccess) Die("ncclAllReduce failed");
  }

  void Wait() override {
    if (side_stream_ == nullptr) return;
    cuda_backend::CheckCuda(cudaStreamSynchronize(side_stream_),
                            "wait for the reduction");
  }

 private:
  int rank_ = 0;
  int world_size_ = 1;
  ncclComm_t comm_ = nullptr;
  cudaStream_t side_stream_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
};

}  // namespace

std::unique_ptr<GradientSync> CreateGradientSync(
    const DistributedConfig& config) {
  if (config.world_size <= 1) return std::make_unique<NoopGradientSync>();
  if (config.rank < 0 || config.rank >= config.world_size) return nullptr;

  ncclUniqueId id{};
  if (config.rank == 0 && ncclGetUniqueId(&id) != ncclSuccess) return nullptr;
  if (!BroadcastUniqueId(config, &id)) return nullptr;

  auto sync =
      std::make_unique<NcclGradientSync>(config.rank, config.world_size);
  if (!sync->Setup(id)) return nullptr;
  return sync;
}

}  // namespace nanochat
