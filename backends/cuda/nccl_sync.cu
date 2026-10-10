// The NCCL gradient-sync backend (docs/distributed-design.md section 5).
//
// This is the only translation unit that includes `nccl.h`. The factory
// `CreateGradientSync` here replaces the host reference on a CUDA build,
// because
// `//src:model` selects this library for `//:backend_cuda`
// (docs/distributed-plan.md phase 0).
//
// The ranks rendezvous over the `master`:`port` pair that the command line
// supplies. Rank 0 generates the `ncclUniqueId` and broadcasts it to the peers
// with a one-shot TCP exchange. Every rank then calls `ncclCommInitRank`.
//
// The launch sets `CUDA_VISIBLE_DEVICES` to the rank, so each process sees one
// card as device 0. The code therefore selects device 0.

#include "src/distributed.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>

#include "backends/cuda/device.h"
#include "nanochat/kernels.h"

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

bool WriteAll(int fd, const void* data, std::size_t bytes) {
  const char* cursor = static_cast<const char*>(data);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t written = ::send(fd, cursor, remaining, 0);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
  return true;
}

bool ReadAll(int fd, void* data, std::size_t bytes) {
  char* cursor = static_cast<char*>(data);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t got = ::recv(fd, cursor, remaining, 0);
    if (got == 0) return false;
    if (got < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += got;
    remaining -= static_cast<std::size_t>(got);
  }
  return true;
}

int ListenOnPort(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(fd, 8) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int ConnectToMaster(const std::string& master, int port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, master.c_str(), &address.sin_addr) != 1) return -1;
  for (int attempt = 0; attempt < 100; ++attempt) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
        0) {
      return fd;
    }
    ::close(fd);
    ::usleep(50000);
  }
  return -1;
}

int AcceptPeer(int listen_fd) {
  pollfd descriptor{};
  descriptor.fd = listen_fd;
  descriptor.events = POLLIN;
  const int ready = ::poll(&descriptor, 1, 30000);
  if (ready <= 0) return -1;
  return ::accept(listen_fd, nullptr, nullptr);
}

// Broadcasts `id` from rank 0 to every peer through a one-shot rendezvous.
void BroadcastUniqueId(int rank, int world_size, const std::string& master,
                       int port, ncclUniqueId* id) {
  if (rank == 0) {
    const int listen_fd = ListenOnPort(port);
    if (listen_fd < 0) Die("rank 0 cannot bind the rendezvous port");
    for (int peer = 1; peer < world_size; ++peer) {
      const int fd = AcceptPeer(listen_fd);
      if (fd < 0) Die("rank 0 cannot accept a peer");
      if (!WriteAll(fd, id, sizeof(*id))) Die("rank 0 cannot send the id");
      ::close(fd);
    }
    ::close(listen_fd);
    return;
  }
  const int fd = ConnectToMaster(master, port);
  if (fd < 0) Die("a rank cannot connect to the rendezvous port");
  if (!ReadAll(fd, id, sizeof(*id))) Die("a rank cannot read the id");
  ::close(fd);
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
  BroadcastUniqueId(config.rank, config.world_size, config.master, config.port,
                    &id);

  auto sync =
      std::make_unique<NcclGradientSync>(config.rank, config.world_size);
  if (!sync->Setup(id)) return nullptr;
  return sync;
}

}  // namespace nanochat
