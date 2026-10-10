#ifndef NANOCHAT_SRC_RENDEZVOUS_H_
#define NANOCHAT_SRC_RENDEZVOUS_H_

// The vendor-free control plane for the gradient-sync backends
// (docs/distributed-native-plan.md section 4).
//
// One instance per process. The class carries the opaque control-plane
// payloads: the CUDA IPC memory handles, the device identifiers, or the RDMA
// queue-pair numbers. It includes POSIX sockets only, so the `src` layer stays
// vendor-free (DESIGN.md section 2.1) and the above-seam code never sees a
// vendor header.
//
// `ExchangeBytes` is an all-gather: every rank contributes `bytes` and every
// rank receives every rank's bytes. Rank 0 gathers and broadcasts. Every call
// is collective, so all ranks call it with the same length and in the same
// order.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/distributed.h"

namespace nanochat {

// The low-level socket plumbing, shared by the host reference and the
// rendezvous. These are POSIX-only and carry no vendor dependency.
namespace rendezvous {

// Writes every byte or returns false. Retries on EINTR.
bool WriteAll(int fd, const void* data, std::size_t bytes);

// Reads exactly `bytes` or returns false. Returns false when the peer closes.
bool ReadAll(int fd, void* data, std::size_t bytes);

// Binds and listens on `port` (0 asks the kernel for an ephemeral port).
// Returns the descriptor, or -1. The caller can recover the chosen port with
// `LocalPort`.
int ListenOnPort(int port);

// The local port a listening socket is bound to, or -1.
int LocalPort(int listen_fd);

// Connects to `master`:`port`, retrying while rank 0 starts. Returns the
// descriptor, or -1 after the attempts run out or the address is invalid.
int ConnectToMaster(const std::string& master, int port);

// Accepts one peer, waiting at most `timeout_ms`. Returns -1 on timeout.
int AcceptPeer(int listen_fd, int timeout_ms);

}  // namespace rendezvous

// The control plane. One instance per process.
class Rendezvous {
 public:
  ~Rendezvous();
  Rendezvous(const Rendezvous&) = delete;
  Rendezvous& operator=(const Rendezvous&) = delete;

  // Binds (rank 0) or connects to rank 0 (every other rank). Returns null on
  // failure. The bind and the connects are collective; rank 0 accepts every
  // peer before it returns.
  static std::unique_ptr<Rendezvous> Create(const DistributedConfig& config);

  // Exchanges `bytes` of opaque data across every rank. `data` points to
  // `world_size * bytes`: slot `rank` holds this rank's payload on entry, and
  // slot `i` holds rank `i`'s payload on exit. Returns false on failure.
  bool ExchangeBytes(void* data, std::size_t bytes);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }

 private:
  Rendezvous(int rank, int world_size) : rank_(rank), world_size_(world_size) {}

  int rank_ = 0;
  int world_size_ = 1;
  int listen_fd_ = -1;
  int client_fd_ = -1;
  // Rank 0 holds one descriptor per peer (ranks 1..world_size-1).
  std::vector<int> peers_;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_RENDEZVOUS_H_
