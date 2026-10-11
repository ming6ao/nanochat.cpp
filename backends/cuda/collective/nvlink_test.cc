// backends/cuda/collective/nvlink_test.cc -- the native collective on two
// cards (docs/distributed-native-plan.md section 13, T1 row). Two processes
// reduce a device buffer and check the sum against an independently computed
// reference.
//
// The T4 pair has no NVLink peer access, so `SelectTransportKind` falls back
// to the host-staged transport. The test is therefore the T1 gate for the
// adapter and the ring schedule on real device memory; on an H100 node the
// same test exercises the NVLink transport.

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/distributed.h"

namespace {

using nanochat::ComputeType;
using nanochat::DistributedConfig;
using nanochat::GradientSync;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

int FindFreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    return -1;
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    ::close(fd);
    return -1;
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

float RankValue(int rank, int index) {
  return static_cast<float>(rank) * 10.0f + static_cast<float>(index) * 0.25f;
}

// Counts devices in a throw-away child. A CUDA call in the parent would leave
// the runtime initialized, and forking after that breaks `cudaSetDevice` in the
// child ("initialization error"), so the count must come from a process that
// shares no CUDA state with the rank processes.
int CountDevices() {
  int fds[2];
  if (::pipe(fds) != 0) return 0;
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return 0;
  }
  if (pid == 0) {
    ::close(fds[0]);
    int count = 0;
    cudaGetDeviceCount(&count);
    const ssize_t written = ::write(fds[1], &count, sizeof(count));
    (void)written;
    ::close(fds[1]);
    std::_Exit(0);
  }
  ::close(fds[1]);
  int count = 0;
  const ssize_t got = ::read(fds[0], &count, sizeof(count));
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);
  return got == static_cast<ssize_t>(sizeof(count)) ? count : 0;
}

bool RunRank(int rank, int port, int count) {
  // Each process owns one card. The launch sets CUDA_VISIBLE_DEVICES in
  // production; here the test uses the device index directly.
  if (cudaSetDevice(rank) != cudaSuccess) {
    std::fprintf(stderr, "rank %d: cudaSetDevice failed\n", rank);
    return false;
  }
  DistributedConfig config;
  config.rank = rank;
  config.world_size = 2;
  config.master = "127.0.0.1";
  config.port = port;
  config.timeout_ms = 30000;
  std::unique_ptr<GradientSync> sync = nanochat::CreateGradientSync(config);
  if (sync == nullptr) {
    std::fprintf(stderr, "rank %d: CreateGradientSync failed\n", rank);
    return false;
  }
  if (sync->rank() != rank || sync->world_size() != 2) return false;

  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    host[static_cast<std::size_t>(i)] =
        static_cast<ComputeType>(RankValue(rank, i));
  }
  ComputeType* device = static_cast<ComputeType*>(nanochat::kernels::Alloc(
      static_cast<std::size_t>(count) * sizeof(ComputeType)));
  if (device == nullptr) return false;
  nanochat::kernels::Memcpy(
      device, host.data(),
      static_cast<std::size_t>(count) * sizeof(ComputeType),
      nanochat::CopyDir::kHostToDevice);
  sync->AllReduceSum(device, count);
  nanochat::kernels::Memcpy(
      host.data(), device,
      static_cast<std::size_t>(count) * sizeof(ComputeType),
      nanochat::CopyDir::kDeviceToHost);
  nanochat::kernels::Free(device);

  for (int i = 0; i < count; ++i) {
    const float expected = RankValue(0, i) + RankValue(1, i);
    const float got = static_cast<float>(host[static_cast<std::size_t>(i)]);
    if (std::fabs(got - expected) > 1e-4f * (1.0f + std::fabs(expected))) {
      std::fprintf(stderr, "rank %d element %d: got %f want %f\n", rank, i, got,
                   expected);
      return false;
    }
  }
  return true;
}

void TestTwoRankAllReduce() {
  const int port = FindFreePort();
  if (port <= 0) {
    Fail("cannot find a free port");
    return;
  }
  constexpr int kCount = 64;
  const pid_t pid = ::fork();
  if (pid < 0) {
    Fail("fork failed");
    return;
  }
  if (pid == 0) {
    const bool ok = RunRank(1, port, kCount);
    std::_Exit(ok ? 0 : 1);
  }
  if (!RunRank(0, port, kCount)) {
    Fail("rank 0 native all-reduce mismatch");
  }
  int status = 0;
  ::waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    Fail("rank 1 native all-reduce mismatch");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  const int devices = CountDevices();
  if (devices < 2) {
    std::printf("nvlink_test: skipped (needs two CUDA devices, found %d)\n",
                devices);
    return 0;
  }
  TestTwoRankAllReduce();
  if (g_failures != 0) {
    std::fprintf(stderr, "nvlink_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("nvlink_test: ok\n");
  return 0;
}
