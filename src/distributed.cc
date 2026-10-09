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

#include "nanochat/kernels.h"
#include "src/ops.h"

namespace nanochat {
namespace {

// The client retries the connect while rank 0 binds and listens: the two ranks
// race at startup. 100 attempts of 50 ms is about 5 s.
constexpr int kConnectAttempts = 100;
constexpr int kConnectDelayMicros = 50000;
// The accept waits long enough for every peer to connect (or to report why it
// did not). It keeps a broken peer from hanging the server forever.
constexpr int kAcceptTimeoutMs = 30000;

void Die(const std::string& message) {
  std::fprintf(stderr, "distributed: %s\n", message.c_str());
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
    if (got == 0) return false;  // the peer closed the connection
    if (got < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += got;
    remaining -= static_cast<std::size_t>(got);
  }
  return true;
}

// Binds a listening socket on `port`. Returns the descriptor, or -1.
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

// Connects to `master`:`port`, retrying while rank 0 starts. Returns the
// descriptor, or -1 after the attempts run out or the address is invalid.
int ConnectToMaster(const std::string& master, int port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, master.c_str(), &address.sin_addr) != 1) return -1;
  for (int attempt = 0; attempt < kConnectAttempts; ++attempt) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
        0) {
      return fd;
    }
    ::close(fd);
    ::usleep(kConnectDelayMicros);
  }
  return -1;
}

// Accepts one peer, waiting at most `kAcceptTimeoutMs`. Returns -1 on timeout
// or error.
int AcceptPeer(int listen_fd) {
  pollfd descriptor{};
  descriptor.fd = listen_fd;
  descriptor.events = POLLIN;
  const int ready = ::poll(&descriptor, 1, kAcceptTimeoutMs);
  if (ready <= 0) return -1;
  return ::accept(listen_fd, nullptr, nullptr);
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
  HostGradientSync(int rank, int world_size, bool device_buffers,
                   std::string master, int port)
      : rank_(rank),
        world_size_(world_size),
        device_buffers_(device_buffers),
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

    // The CUDA backend keeps the model in device memory, so the host reference
    // stages the buffer through host memory first. On the CPU reference
    // backend `device_buffers_` is false and this is a passthrough. The
    // `kernels::Memcpy` seam is backend-owned, so this file still includes no
    // vendor header.
    ComputeType* working = buffer;
    if (device_buffers_) {
      if (device_staging_.size() < elements) device_staging_.resize(elements);
      kernels::Memcpy(device_staging_.data(), buffer, bytes,
                      CopyDir::kDeviceToHost);
      working = device_staging_.data();
    }

    if (rank_ == 0) {
      if (!AcceptPeers()) Die("cannot accept a peer");
      if (scratch_.size() < elements) scratch_.resize(elements);
      for (int fd : peers_) {
        if (!ReadAll(fd, scratch_.data(), bytes)) {
          Die("cannot receive a gradient");
        }
        for (std::size_t i = 0; i < elements; ++i) {
          working[i] = ToC(AsF(working[i]) + AsF(scratch_[i]));
        }
      }
      for (int fd : peers_) {
        if (!WriteAll(fd, working, bytes)) Die("cannot broadcast the sum");
      }
    } else {
      if (!WriteAll(client_fd_, working, bytes)) Die("cannot send a gradient");
      if (!ReadAll(client_fd_, working, bytes)) Die("cannot receive the sum");
    }

    if (device_buffers_) {
      kernels::Memcpy(buffer, working, bytes, CopyDir::kHostToDevice);
    }
  }

  // Binds (rank 0) or connects (every other rank). Called once by the factory;
  // the accept is deferred to the first reduction so a forked rank 0 does not
  // block before its peers exist.
  bool Setup() {
    if (rank_ == 0) {
      listen_fd_ = ListenOnPort(port_);
      return listen_fd_ >= 0;
    }
    client_fd_ = ConnectToMaster(master_, port_);
    return client_fd_ >= 0;
  }

 private:
  bool AcceptPeers() {
    if (accepted_) return true;
    for (int i = 0; i < world_size_ - 1; ++i) {
      const int fd = AcceptPeer(listen_fd_);
      if (fd < 0) return false;
      peers_.push_back(fd);
    }
    accepted_ = true;
    return true;
  }

  int rank_ = 0;
  int world_size_ = 1;
  bool device_buffers_ = false;
  std::string master_;
  int port_ = 0;
  int listen_fd_ = -1;
  int client_fd_ = -1;
  bool accepted_ = false;
  std::vector<int> peers_;
  std::vector<ComputeType> scratch_;
  // The host-side copy of one device gradient, reused across parameters.
  std::vector<ComputeType> device_staging_;
};

}  // namespace

std::unique_ptr<GradientSync> CreateGradientSync(
    const DistributedConfig& config) {
  if (config.world_size <= 1) return std::make_unique<NoopGradientSync>();
  if (config.rank < 0 || config.rank >= config.world_size) return nullptr;
  auto sync = std::make_unique<HostGradientSync>(config.rank, config.world_size,
                                                 config.device_buffers,
                                                 config.master, config.port);
  if (!sync->Setup()) return nullptr;
  return sync;
}

}  // namespace nanochat
