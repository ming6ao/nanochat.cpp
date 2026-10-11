// The backend-free host reference for the gradient-sync seam
// (docs/distributed-design.md section 5).
//
// Rank 0 is a parameter server. It accepts one connection per other rank,
// receives each rank's buffer, adds them into its own buffer, and broadcasts
// the result. The other ranks send their buffer and read the sum back. The code
// uses POSIX sockets only and includes no vendor header, so it is the portable
// fallback for a host without NCCL.

#include "src/distributed.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/ops.h"
#include "src/rendezvous.h"

namespace nanochat {
namespace {

// The accept waits long enough for every peer to connect (or to report why it
// did not). It keeps a broken peer from hanging the server forever.
constexpr int kAcceptTimeoutMs = 30000;

void Die(const std::string& message) {
  std::fprintf(stderr, "distributed: %s\n", message.c_str());
  std::abort();
}

// The single-process object. Every reduction is a no-op.
class NoopGradientSync final : public GradientSync {
 public:
  int rank() const override { return 0; }
  int world_size() const override { return 1; }
  void AllReduceSum(ComputeType*, std::int64_t) override {}
};

// The TCP parameter server (rank 0) and its clients (every other rank). Up to
// one staging buffer is needed when the gradients live in device memory.
class HostGradientSync final : public GradientSync {
 public:
  HostGradientSync(int rank, int world_size, std::string master, int port)
      : rank_(rank),
        world_size_(world_size),
        master_(std::move(master)),
        port_(port) {}

  ~HostGradientSync() override {
    for (int fd : peers_) ::close(fd);
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (client_fd_ >= 0) ::close(client_fd_);
  }

  HostGradientSync(const HostGradientSync&) = delete;
  HostGradientSync& operator=(const HostGradientSync&) = delete;

  int rank() const override { return rank_; }
  int world_size() const override { return world_size_; }

  void AllReduceSum(ComputeType* buffer, std::int64_t count) override {
    if (buffer == nullptr || count <= 0) return;
    const std::size_t elements = static_cast<std::size_t>(count);
    const std::size_t bytes = elements * sizeof(ComputeType);

    if (rank_ == 0) {
      if (!AcceptPeers()) Die("cannot accept a peer");
      if (scratch_.size() < elements) scratch_.resize(elements);
      for (int fd : peers_) {
        if (!rendezvous::ReadAll(fd, scratch_.data(), bytes)) {
          Die("cannot receive a gradient");
        }
        for (std::size_t i = 0; i < elements; ++i) {
          buffer[i] = ToC(AsF(buffer[i]) + AsF(scratch_[i]));
        }
      }
      for (int fd : peers_) {
        if (!rendezvous::WriteAll(fd, buffer, bytes)) {
          Die("cannot broadcast the sum");
        }
      }
    } else {
      if (!rendezvous::WriteAll(client_fd_, buffer, bytes)) {
        Die("cannot send a gradient");
      }
      if (!rendezvous::ReadAll(client_fd_, buffer, bytes)) {
        Die("cannot receive the sum");
      }
    }
  }

  // Binds (rank 0) or connects (every other rank). Called once by the factory;
  // the accept is deferred to the first reduction so a forked rank 0 does not
  // block before its peers exist.
  bool Setup() {
    if (rank_ == 0) {
      listen_fd_ = rendezvous::ListenOnPort(port_);
      return listen_fd_ >= 0;
    }
    client_fd_ = rendezvous::ConnectToMaster(master_, port_);
    return client_fd_ >= 0;
  }

 private:
  bool AcceptPeers() {
    if (accepted_) return true;
    for (int i = 0; i < world_size_ - 1; ++i) {
      const int fd = rendezvous::AcceptPeer(listen_fd_, kAcceptTimeoutMs);
      if (fd < 0) return false;
      peers_.push_back(fd);
    }
    accepted_ = true;
    return true;
  }

  int rank_ = 0;
  int world_size_ = 1;
  std::string master_;
  int port_ = 0;
  int listen_fd_ = -1;
  int client_fd_ = -1;
  bool accepted_ = false;
  std::vector<int> peers_;
  std::vector<ComputeType> scratch_;
};

}  // namespace

std::unique_ptr<GradientSync> CreateGradientSync(
    const DistributedConfig& config) {
  if (config.world_size <= 1) return std::make_unique<NoopGradientSync>();
  if (config.rank < 0 || config.rank >= config.world_size) return nullptr;
  auto sync = std::make_unique<HostGradientSync>(config.rank, config.world_size,
                                                 config.master, config.port);
  if (!sync->Setup()) return nullptr;
  return sync;
}

}  // namespace nanochat
