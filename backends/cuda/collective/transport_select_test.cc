// backends/cuda/collective/transport_select_test.cc -- the per-neighbor
// transport selection rule (docs/distributed-native-plan.md section 3.2 and
// section 13, T0 row 2). It checks the pure rule and the mixed group that
// `RingTransports::Connect` builds.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/collective/transport.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::DistributedConfig;
using nanochat::Rendezvous;
using nanochat::cuda_backend::PeerCapabilities;
using nanochat::cuda_backend::RingTransports;
using nanochat::cuda_backend::SelectTransportKind;
using nanochat::cuda_backend::Token;
using nanochat::cuda_backend::Transport;
using nanochat::cuda_backend::TransportFactory;
using nanochat::cuda_backend::TransportKind;
using nanochat::cuda_backend::WaitResult;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void CheckKind(TransportKind got, TransportKind want, const char* what) {
  if (got != want) {
    Fail(std::string(what) + ": got " +
         nanochat::cuda_backend::TransportKindName(got) + " want " +
         nanochat::cuda_backend::TransportKindName(want));
  }
}

// A transport that records the pair it was built for and does nothing else.
class FakeTransport final : public Transport {
 public:
  FakeTransport(TransportKind kind, int self, int peer)
      : kind_(kind), self_(self), peer_(peer) {}
  int self_rank() const override { return self_; }
  int peer_rank() const override { return peer_; }
  TransportKind kind() const override { return kind_; }
  bool Connect(const DistributedConfig&, Rendezvous*) override {
    connected_ = true;
    return connect_ok_;
  }
  bool RegisterWindow(void*, std::size_t) override { return true; }
  int SlotCount() const override { return 2; }
  Token Send(int, const void*, std::size_t) override { return Token{0}; }
  Token Recv(int, void*, std::size_t) override { return Token{0}; }
  bool Done(Token) override { return true; }
  WaitResult Wait(Token, nanochat::cuda_backend::TimePoint) override {
    return WaitResult::kCompleted;
  }
  void Release(Token) override {}

  void set_connect_ok(bool ok) { connect_ok_ = ok; }
  bool connected() const { return connected_; }

 private:
  TransportKind kind_;
  int self_;
  int peer_;
  bool connected_ = false;
  bool connect_ok_ = true;
};

void CheckPureRule() {
  CheckKind(SelectTransportKind({true, false}), TransportKind::kNvlink,
            "peer access wins");
  CheckKind(SelectTransportKind({false, true}), TransportKind::kRdma,
            "rdma when no peer access");
  CheckKind(SelectTransportKind({false, false}), TransportKind::kHost,
            "host staging is the fallback");
  CheckKind(SelectTransportKind({true, true}), TransportKind::kNvlink,
            "nvlink beats rdma");
}

// Connects `world` ranks with the given capabilities and checks the kinds the
// factory saw. A mixed group is the normal case at a node boundary.
void CheckMixedGroup() {
  struct Case {
    int world;
    PeerCapabilities prev;
    PeerCapabilities next;
    TransportKind want_prev;
    TransportKind want_next;
    const char* label;
  };
  const std::vector<Case> cases = {
      {2,
       {true, false},
       {true, false},
       TransportKind::kNvlink,
       TransportKind::kNvlink,
       "two nvlink"},
      {4,
       {false, true},
       {false, true},
       TransportKind::kRdma,
       TransportKind::kRdma,
       "two rdma"},
      {4,
       {false, false},
       {false, false},
       TransportKind::kHost,
       TransportKind::kHost,
       "two host"},
      {8,
       {true, false},
       {false, true},
       TransportKind::kNvlink,
       TransportKind::kRdma,
       "mixed nvlink/rdma"},
      {3,
       {false, false},
       {true, false},
       TransportKind::kHost,
       TransportKind::kNvlink,
       "mixed host/nvlink"},
  };
  for (const Case& test : cases) {
    for (int rank = 0; rank < test.world; ++rank) {
      DistributedConfig config;
      config.rank = rank;
      config.world_size = test.world;
      std::vector<TransportKind> built;
      TransportFactory factory =
          [&](TransportKind kind, int self, int peer,
              bool send_direction) -> std::unique_ptr<Transport> {
        (void)send_direction;
        built.push_back(kind);
        return std::make_unique<FakeTransport>(kind, self, peer);
      };
      RingTransports ring;
      if (!ring.Connect(config, nullptr, {test.prev, test.next}, factory)) {
        Fail(std::string(test.label) + ": connect failed");
        continue;
      }
      CheckKind(ring.prev_kind(), test.want_prev, test.label);
      CheckKind(ring.next_kind(), test.want_next, test.label);
      // The prev transport is built first, then the next.
      if (built.size() != 2 || built[0] != test.want_prev ||
          built[1] != test.want_next) {
        Fail(std::string(test.label) + ": factory saw the wrong kinds");
      }
      const int expected_next = (rank + 1) % test.world;
      const int expected_prev = (rank - 1 + test.world) % test.world;
      if (ring.Next()->peer_rank() != expected_next ||
          ring.Prev()->peer_rank() != expected_prev) {
        Fail(std::string(test.label) + ": wrong ring neighbors");
      }
    }
  }
}

void CheckWorldOne() {
  DistributedConfig config;
  config.rank = 0;
  config.world_size = 1;
  bool built = false;
  TransportFactory factory =
      [&](TransportKind kind, int self, int peer,
          bool send_direction) -> std::unique_ptr<Transport> {
    (void)send_direction;
    built = true;
    return std::make_unique<FakeTransport>(kind, self, peer);
  };
  RingTransports ring;
  if (!ring.Connect(config, nullptr, {PeerCapabilities{}, PeerCapabilities{}},
                    factory)) {
    Fail("world 1: connect failed");
  }
  if (built || ring.Next() != nullptr || ring.Prev() != nullptr) {
    Fail("world 1: a transport was built");
  }
}

void CheckConnectFailure() {
  DistributedConfig config;
  config.rank = 0;
  config.world_size = 2;
  bool first = true;
  TransportFactory factory =
      [&](TransportKind kind, int self, int peer,
          bool send_direction) -> std::unique_ptr<Transport> {
    (void)send_direction;
    auto transport = std::make_unique<FakeTransport>(kind, self, peer);
    if (first) {
      transport->set_connect_ok(false);
      first = false;
    }
    return transport;
  };
  RingTransports ring;
  if (ring.Connect(config, nullptr, {PeerCapabilities{}, PeerCapabilities{}},
                   factory)) {
    Fail("connect failure: Connect returned true");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  CheckPureRule();
  CheckMixedGroup();
  CheckWorldOne();
  CheckConnectFailure();
  if (g_failures != 0) {
    std::fprintf(stderr, "transport_select_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("transport_select_test: ok\n");
  return 0;
}
