// backends/cuda/collective/schedule_test.cc -- the ring schedule on the CPU.
//
// docs/distributed-native-plan.md section 13, T0 row 1. Exercises the ring
// reduce-scatter + all-gather at world sizes 1, 2, 4, and 8 with the in-process
// memory transport, judges the schedule trace (section 10), and checks that a
// peer that never completes returns a timeout instead of hanging.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "backends/cuda/collective/memory_transport.h"
#include "backends/cuda/collective/ring_schedule.h"
#include "backends/cuda/collective/transport.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::cuda_backend::AddFn;
using nanochat::cuda_backend::JudgeScheduleTrace;
using nanochat::cuda_backend::MemoryBus;
using nanochat::cuda_backend::MemoryTransport;
using nanochat::cuda_backend::RingAllReduceSum;
using nanochat::cuda_backend::RingChunkSize;
using nanochat::cuda_backend::ScheduleTrace;
using nanochat::cuda_backend::TimePoint;
using nanochat::cuda_backend::WaitResult;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void HostAdd(ComputeType* dst, const ComputeType* src, std::int64_t count) {
  for (std::int64_t i = 0; i < count; ++i) dst[i] += src[i];
}

float RankValue(int rank, std::int64_t index) {
  return static_cast<float>(rank) * 1000.0f + static_cast<float>(index) * 0.5f;
}

// Runs one ring all-reduce across `world` logical ranks in one process.
void RunWorld(int world, int count) {
  auto bus = std::make_shared<MemoryBus>();
  std::vector<std::vector<ComputeType>> buffers(
      static_cast<std::size_t>(world));
  std::vector<std::vector<ComputeType>> scratch(
      static_cast<std::size_t>(world));
  std::vector<std::unique_ptr<MemoryTransport>> next(
      static_cast<std::size_t>(world));
  std::vector<std::unique_ptr<MemoryTransport>> prev(
      static_cast<std::size_t>(world));
  std::vector<ScheduleTrace> traces(static_cast<std::size_t>(world));
  std::vector<char> ok(static_cast<std::size_t>(world), 0);

  for (int r = 0; r < world; ++r) {
    buffers[static_cast<std::size_t>(r)].resize(
        static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      buffers[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)] =
          static_cast<ComputeType>(RankValue(r, i));
    }
    const std::int64_t chunk = RingChunkSize(count, world);
    scratch[static_cast<std::size_t>(r)].assign(
        static_cast<std::size_t>(2 * chunk), 0.0f);
    next[static_cast<std::size_t>(r)] =
        std::make_unique<MemoryTransport>(bus, r, (r + 1) % world);
    prev[static_cast<std::size_t>(r)] =
        std::make_unique<MemoryTransport>(bus, r, (r - 1 + world) % world);
    next[static_cast<std::size_t>(r)]->RegisterWindow(nullptr, 0);
    prev[static_cast<std::size_t>(r)]->RegisterWindow(nullptr, 0);
  }

  const TimePoint deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::vector<std::thread> threads;
  for (int r = 0; r < world; ++r) {
    threads.emplace_back([&, r] {
      ok[static_cast<std::size_t>(r)] =
          RingAllReduceSum(next[static_cast<std::size_t>(r)].get(),
                           prev[static_cast<std::size_t>(r)].get(), r, world,
                           buffers[static_cast<std::size_t>(r)].data(), count,
                           scratch[static_cast<std::size_t>(r)].data(), HostAdd,
                           deadline, &traces[static_cast<std::size_t>(r)]);
    });
  }
  for (std::thread& thread : threads) thread.join();

  for (int r = 0; r < world; ++r) {
    if (!ok[static_cast<std::size_t>(r)]) {
      Fail("world " + std::to_string(world) + " rank " + std::to_string(r) +
           ": ring returned failure");
      continue;
    }
    for (int i = 0; i < count; ++i) {
      float expected = 0.0f;
      for (int k = 0; k < world; ++k) expected += RankValue(k, i);
      const float got =
          buffers[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)];
      if (std::fabs(got - expected) > 1e-4f * (1.0f + std::fabs(expected))) {
        Fail("world " + std::to_string(world) + " rank " + std::to_string(r) +
             " element " + std::to_string(i) + ": got " + std::to_string(got) +
             " want " + std::to_string(expected));
        break;
      }
    }
  }

  if (world > 1) {
    ScheduleTrace merged;
    for (const ScheduleTrace& trace : traces) {
      merged.insert(merged.end(), trace.begin(), trace.end());
    }
    std::string error;
    if (!JudgeScheduleTrace(merged, &error)) {
      Fail("world " + std::to_string(world) + ": trace rejected: " + error);
    }
  }
}

// A peer that never sends must return kTimeout, not hang.
void RunTimeout() {
  auto bus = std::make_shared<MemoryBus>();
  MemoryTransport next(bus, 0, 1);
  MemoryTransport prev(bus, 0, 1);
  next.RegisterWindow(nullptr, 0);
  prev.RegisterWindow(nullptr, 0);
  constexpr int kCount = 8;
  std::vector<ComputeType> buffer(kCount, 1.0f);
  std::vector<ComputeType> scratch(2 * kCount, 0.0f);
  const TimePoint deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  const bool ok = RingAllReduceSum(&next, &prev, 0, 2, buffer.data(), kCount,
                                   scratch.data(), HostAdd, deadline, nullptr);
  if (ok) {
    Fail("timeout: the ring did not detect the missing peer");
  }
}

// The token lifecycle: Done and Wait do not consume a completed token.
void RunTokenLifetime() {
  auto bus = std::make_shared<MemoryBus>();
  MemoryTransport sender(bus, 0, 1);
  MemoryTransport receiver(bus, 1, 0);
  std::vector<ComputeType> payload = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<ComputeType> got(payload.size(), 0.0f);
  const nanochat::cuda_backend::Token send_token =
      sender.Send(0, payload.data(), payload.size() * sizeof(ComputeType));
  const nanochat::cuda_backend::Token recv_token =
      receiver.Recv(0, got.data(), got.size() * sizeof(ComputeType));
  const TimePoint deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  sender.Wait(send_token, deadline);
  if (receiver.Wait(recv_token, deadline) != WaitResult::kCompleted) {
    Fail("token: the receive did not complete");
  }
  if (!receiver.Done(recv_token)) {
    Fail("token: Done after completion returned false");
  }
  if (receiver.Wait(recv_token, deadline) != WaitResult::kCompleted) {
    Fail("token: a second Wait did not return completed");
  }
  for (std::size_t i = 0; i < payload.size(); ++i) {
    if (got[i] != payload[i]) {
      Fail("token: payload mismatch");
      break;
    }
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  RunWorld(1, 16);
  RunWorld(2, 16);
  RunWorld(4, 16);
  RunWorld(8, 16);
  // An odd count exercises uneven chunks.
  RunWorld(4, 10);
  RunWorld(8, 10);
  RunTimeout();
  RunTokenLifetime();
  if (g_failures != 0) {
    std::fprintf(stderr, "schedule_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("schedule_test: ok\n");
  return 0;
}
