// The vendor-free control plane for the gradient-sync backends
// (docs/distributed-native-plan.md section 4). POSIX sockets only; no vendor
// header is reachable from here.

#include "src/rendezvous.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace nanochat {

namespace rendezvous {

namespace {
// The client retries while rank 0 binds and listens: the ranks race at
// startup. 100 attempts of 50 ms is about 5 s.
constexpr int kConnectAttempts = 100;
constexpr int kConnectDelayMicros = 50000;
}  // namespace

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
      ::listen(fd, 16) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int LocalPort(int listen_fd) {
  sockaddr_in address{};
  socklen_t length = sizeof(address);
  if (::getsockname(listen_fd, reinterpret_cast<sockaddr*>(&address),
                    &length) != 0) {
    return -1;
  }
  return ntohs(address.sin_port);
}

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

int AcceptPeer(int listen_fd, int timeout_ms) {
  pollfd descriptor{};
  descriptor.fd = listen_fd;
  descriptor.events = POLLIN;
  const int ready = ::poll(&descriptor, 1, timeout_ms);
  if (ready <= 0) return -1;
  return ::accept(listen_fd, nullptr, nullptr);
}

}  // namespace rendezvous

Rendezvous::~Rendezvous() {
  for (int fd : peers_) ::close(fd);
  if (listen_fd_ >= 0) ::close(listen_fd_);
  if (client_fd_ >= 0) ::close(client_fd_);
}

std::unique_ptr<Rendezvous> Rendezvous::Create(
    const DistributedConfig& config) {
  if (config.world_size <= 0) return nullptr;
  if (config.rank < 0 || config.rank >= config.world_size) return nullptr;
  auto rendezvous = std::unique_ptr<Rendezvous>(
      new Rendezvous(config.rank, config.world_size));
  if (config.rank == 0) {
    rendezvous->listen_fd_ = rendezvous::ListenOnPort(config.port);
    if (rendezvous->listen_fd_ < 0) return nullptr;
    for (int peer = 1; peer < config.world_size; ++peer) {
      const int fd =
          rendezvous::AcceptPeer(rendezvous->listen_fd_, /*timeout_ms=*/30000);
      if (fd < 0) return nullptr;
      rendezvous->peers_.push_back(fd);
    }
  } else {
    rendezvous->client_fd_ =
        rendezvous::ConnectToMaster(config.master, config.port);
    if (rendezvous->client_fd_ < 0) return nullptr;
  }
  return rendezvous;
}

bool Rendezvous::ExchangeBytes(void* data, std::size_t bytes) {
  if (data == nullptr || bytes == 0) return bytes == 0;
  char* buffer = static_cast<char*>(data);
  if (rank_ == 0) {
    // Gather every peer's payload into its slot.
    for (int peer = 1; peer < world_size_; ++peer) {
      if (!rendezvous::ReadAll(peers_[static_cast<std::size_t>(peer - 1)],
                               buffer + static_cast<std::size_t>(peer) * bytes,
                               bytes)) {
        return false;
      }
    }
    // Broadcast the whole table to every peer.
    for (int fd : peers_) {
      if (!rendezvous::WriteAll(
              fd, buffer, static_cast<std::size_t>(world_size_) * bytes)) {
        return false;
      }
    }
    return true;
  }
  // Send this rank's payload, then read the broadcast table.
  if (!rendezvous::WriteAll(client_fd_,
                            buffer + static_cast<std::size_t>(rank_) * bytes,
                            bytes)) {
    return false;
  }
  return rendezvous::ReadAll(client_fd_, buffer,
                             static_cast<std::size_t>(world_size_) * bytes);
}

}  // namespace nanochat
