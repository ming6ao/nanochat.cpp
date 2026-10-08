// tests/collective_sim_test.cc -- the NCCL mock and its coordinator
// (docs/simulator.md sections 6.3, 6.4, and 12.3).
//
// Four checks, all CPU only:
//
//   1. Four ranks run the NCCL mock in separate processes. Every rank must end
//      with the elementwise sum of all four buffers, and the *merged* run log
//      (the real `NANOCHAT_SIM_LOG` path each rank appends to) must satisfy the
//      coordinator's cross-rank rules.
//   2. One rank never issues the all-reduce. The peers must fail with a
//      timeout rather than hang, and the coordinator must reject the merged
//      timeline (the deadlock check of section 12.3).
//   3. Group nesting and buffer-range rules, exercised on a hand-built
//      timeline and by the real `ncclGroupStart`/`ncclGroupEnd` surface.
//   4. The destroyed-handle lifecycle.
//
// The mock rendezvouses through POSIX shared memory, so the processes need no
// daemon. Each case names a fresh segment and log, so a repeat run cannot see
// the leftovers of the last one. These are the collective *mock's* checks; the
// gradient-sync seam's numerics are owned by `//src:distributed_test`.

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/sandbox.h"
#include "tools/cuda_sim/collective/nccl_mock.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

constexpr int kWorld = 4;
constexpr int kCount = 8;

std::string SegmentName(const std::string& suffix) {
  return "collective-" + std::to_string(static_cast<long>(getpid())) + "-" +
         suffix;
}

std::string LogPath(const std::string& suffix) {
  return "/tmp/nanochat-collective-" +
         std::to_string(static_cast<long>(getpid())) + "-" + suffix + ".jsonl";
}

// A fresh rendezvous segment and log for one case.
void SetSegment(const std::string& suffix) {
  const std::string id = SegmentName(suffix);
  setenv("NANOCHAT_SIM_COMM_ID", id.c_str(), 1);
  setenv("NANOCHAT_SIM_TIMEOUT_MS", "1500", 1);
  const std::string log = LogPath(suffix);
  std::remove(log.c_str());
  setenv("NANOCHAT_SIM_LOG", log.c_str(), 1);
}

void UnlinkSegment(const std::string& suffix) {
  shm_unlink(("/nanochat_sim_" + SegmentName(suffix)).c_str());
  std::remove(LogPath(suffix).c_str());
}

// Rank `rank`'s payload: a distinct, well-conditioned ramp.
std::vector<float> RankPayload(int rank) {
  std::vector<float> out(kCount);
  for (int i = 0; i < kCount; ++i) {
    out[static_cast<std::size_t>(i)] =
        static_cast<float>((rank + 1) * 0.25f + i * 0.5f);
  }
  return out;
}

struct RankOutcome {
  int status = 0;
  std::vector<float> result;
};

RankOutcome RunRank(int rank, int missing_rank, int* write_fd) {
  RankOutcome outcome;
  if (rank == missing_rank) {
    outcome.status = ncclTimeout;
    return outcome;
  }
  ncclUniqueId unique_id;
  std::memset(&unique_id, 0, sizeof(unique_id));
  ncclGetUniqueId(&unique_id);
  ncclComm_t comm = nullptr;
  const ncclResult_t init = ncclCommInitRank(&comm, kWorld, unique_id, rank);
  if (init != ncclSuccess) {
    outcome.status = init;
    return outcome;
  }

  const std::vector<float> payload = RankPayload(rank);
  std::vector<float> result(kCount, 0.0f);
  ncclResult_t status = ncclSuccess;
  if (rank != missing_rank) {
    status = ncclAllReduce(payload.data(), result.data(), kCount,
                           ncclDataType_t{7} /* ncclFloat32 */,
                           ncclRedOp_t{0} /* ncclSum */, comm, nullptr);
  }
  outcome.status = status;
  outcome.result = result;
  if (write_fd != nullptr) {
    const ssize_t written =
        write(*write_fd, result.data(), result.size() * sizeof(float));
    (void)written;
  }
  ncclCommDestroy(comm);
  return outcome;
}

// Reads the merged multi-rank log and runs the coordinator over it.
void JudgeMergedLog(const std::string& suffix, bool expect_consistent) {
  std::vector<nanochat::sim::CollectiveRecord> timeline;
  std::string error;
  ExpectTrue(
      nanochat::sim::ReadCollectiveLog(LogPath(suffix), &timeline, &error),
      "the merged log is readable: " + error);
  ExpectTrue(!timeline.empty(), "the merged log holds records");
  const bool consistent =
      nanochat::sim::JudgeCollectiveRecords(timeline, &error);
  ExpectTrue(
      consistent == expect_consistent,
      Format("the merged timeline is %s: %s",
             expect_consistent ? "consistent" : "rejected", error.c_str()));
  if (!expect_consistent) {
    ExpectTrue(error.find("deadlock") != std::string::npos,
               "the rejection names the deadlock: " + error);
  }
}

void CheckFourRankSum() {
  SetSegment("sum");
  int fds[kWorld][2];
  for (int rank = 0; rank < kWorld; ++rank) {
    if (pipe(fds[rank]) != 0) {
      Fail("pipe");
      return;
    }
  }
  std::fflush(nullptr);
  std::vector<pid_t> children;
  for (int rank = 0; rank < kWorld; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) {
      close(fds[rank][0]);
      RunRank(rank, -1, &fds[rank][1]);
      close(fds[rank][1]);
      _exit(0);
    }
    close(fds[rank][1]);
    children.push_back(pid);
  }
  std::vector<float> reference(kCount, 0.0f);
  for (int rank = 0; rank < kWorld; ++rank) {
    const std::vector<float> payload = RankPayload(rank);
    for (int i = 0; i < kCount; ++i) {
      reference[static_cast<std::size_t>(i)] +=
          payload[static_cast<std::size_t>(i)];
    }
  }
  for (int rank = 0; rank < kWorld; ++rank) {
    std::vector<float> got(kCount, 0.0f);
    const ssize_t bytes =
        read(fds[rank][0], got.data(), got.size() * sizeof(float));
    close(fds[rank][0]);
    int status = 0;
    waitpid(children[static_cast<std::size_t>(rank)], &status, 0);
    ExpectTrue(bytes == static_cast<ssize_t>(got.size() * sizeof(float)),
               Format("rank %d reported a result", rank));
    for (int i = 0; i < kCount; ++i) {
      ExpectTrue(std::fabs(got[static_cast<std::size_t>(i)] -
                           reference[static_cast<std::size_t>(i)]) < 1e-5f,
                 Format("rank %d element %d is the elementwise sum", rank, i));
    }
  }
  JudgeMergedLog("sum", /*expect_consistent=*/true);
  UnlinkSegment("sum");
}

void CheckMissingPeer() {
  SetSegment("missing");
  const int missing_rank = kWorld - 1;
  int fds[kWorld][2];
  for (int rank = 0; rank < kWorld; ++rank) {
    if (pipe(fds[rank]) != 0) {
      Fail("pipe");
      return;
    }
  }
  std::fflush(nullptr);
  std::vector<pid_t> children;
  for (int rank = 0; rank < kWorld; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) {
      close(fds[rank][0]);
      const RankOutcome outcome = RunRank(rank, missing_rank, nullptr);
      const int status = outcome.status;
      write(fds[rank][1], &status, sizeof(status));
      close(fds[rank][1]);
      _exit(0);
    }
    close(fds[rank][1]);
    children.push_back(pid);
  }
  for (int rank = 0; rank < kWorld; ++rank) {
    int status = 0;
    const ssize_t bytes = read(fds[rank][0], &status, sizeof(status));
    close(fds[rank][0]);
    int wait_status = 0;
    waitpid(children[static_cast<std::size_t>(rank)], &wait_status, 0);
    ExpectTrue(bytes == static_cast<ssize_t>(sizeof(status)),
               Format("rank %d reported a status", rank));
    if (rank == missing_rank) continue;
    ExpectTrue(status == ncclTimeout,
               Format("rank %d reports the deadlock as a timeout (got %d)",
                      rank, status));
  }
  JudgeMergedLog("missing", /*expect_consistent=*/false);
  UnlinkSegment("missing");
}

void CheckGroupBalance() {
  SetSegment("group");
  ncclUniqueId unique_id;
  std::memset(&unique_id, 0, sizeof(unique_id));
  ncclGetUniqueId(&unique_id);
  ncclComm_t comm = nullptr;
  ExpectTrue(ncclCommInitRank(&comm, kWorld, unique_id, 0) == ncclSuccess,
             "ncclCommInitRank succeeds");
  ExpectTrue(nanochat::sim::GroupDepth() == 0,
             "the process starts outside a group");
  ExpectTrue(ncclGroupStart() == ncclSuccess, "ncclGroupStart succeeds");
  ExpectTrue(nanochat::sim::GroupDepth() == 1, "the group depth is one");
  ExpectTrue(ncclGroupEnd() == ncclSuccess, "ncclGroupEnd succeeds");
  ExpectTrue(nanochat::sim::GroupDepth() == 0, "the group depth is balanced");
  ExpectTrue(ncclGroupEnd() != ncclSuccess,
             "an unmatched ncclGroupEnd is refused");

  // Reuse after destroy is reported, not undefined.
  int count = 0;
  ExpectTrue(ncclCommDestroy(comm) == ncclSuccess, "ncclCommDestroy succeeds");
  ExpectTrue(ncclCommCount(comm, &count) == ncclInvalidUsage,
             "a destroyed communicator is not reused");
  ExpectTrue(ncclCommDestroy(comm) == ncclInvalidUsage,
             "double destroy is refused");
  UnlinkSegment("group");
}

void CheckCoordinator() {
  std::vector<nanochat::sim::CollectiveRecord> timeline;
  for (int rank = 0; rank < kWorld; ++rank) {
    nanochat::sim::CollectiveRecord record;
    record.rank = rank;
    record.world_size = kWorld;
    record.communicator = "collective-test";
    record.call = "ncclAllReduce";
    record.count = kCount;
    record.data_type = "ncclFloat32";
    record.red_op = "ncclSum";
    record.verdict = "ok";
    timeline.push_back(record);
  }
  std::string error;
  ExpectTrue(nanochat::sim::JudgeCollectiveRecords(timeline, &error),
             "a consistent timeline is accepted: " + error);

  // Drop one rank: the coordinator must report the deadlock.
  std::vector<nanochat::sim::CollectiveRecord> missing = timeline;
  missing.pop_back();
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(missing, &error),
             "a missing rank is rejected");
  ExpectTrue(error.find("deadlock") != std::string::npos,
             "the rejection names the deadlock");

  // A rank outside the world is rejected.
  std::vector<nanochat::sim::CollectiveRecord> bad_rank = timeline;
  bad_rank[0].rank = kWorld;
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(bad_rank, &error),
             "a rank outside world_size is rejected");

  // A world_size that disagrees with the others is rejected before any
  // per-rank indexing.
  std::vector<nanochat::sim::CollectiveRecord> bad_world = timeline;
  bad_world[1].world_size = kWorld + 1;
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(bad_world, &error),
             "a disagreeing world_size is rejected");
  ExpectTrue(error.find("world_size") != std::string::npos,
             "the rejection names world_size");

  // A signature mismatch is rejected.
  std::vector<nanochat::sim::CollectiveRecord> mismatch = timeline;
  mismatch[2].count = kCount + 1;
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(mismatch, &error),
             "a count mismatch is rejected");

  // A count that overruns the rendezvous buffer is rejected.
  std::vector<nanochat::sim::CollectiveRecord> oversized = timeline;
  for (auto& record : oversized) record.count = (1ull << 40);
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(oversized, &error),
             "a count above the collective buffer is rejected");
  ExpectTrue(error.find("buffer") != std::string::npos,
             "the rejection names the buffer");

  // Group nesting: an unclosed group on one rank is rejected.
  std::vector<nanochat::sim::CollectiveRecord> unclosed = timeline;
  nanochat::sim::CollectiveRecord start;
  start.rank = 0;
  start.world_size = kWorld;
  start.communicator = "collective-test";
  start.call = "ncclGroupStart";
  start.verdict = "ok";
  unclosed.push_back(start);
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(unclosed, &error),
             "an unclosed group is rejected");
  ExpectTrue(error.find("unclosed") != std::string::npos,
             "the rejection names the unclosed group");

  // A balanced group is accepted.
  std::vector<nanochat::sim::CollectiveRecord> grouped = unclosed;
  nanochat::sim::CollectiveRecord end = start;
  end.call = "ncclGroupEnd";
  grouped.push_back(end);
  ExpectTrue(nanochat::sim::JudgeCollectiveRecords(grouped, &error),
             "a balanced group is accepted: " + error);

  // An empty timeline is rejected.
  std::vector<nanochat::sim::CollectiveRecord> empty;
  ExpectTrue(!nanochat::sim::JudgeCollectiveRecords(empty, &error),
             "an empty timeline is rejected");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("sim profile: %s\n", nanochat::sim::SimProfile().c_str());
  CheckFourRankSum();
  CheckMissingPeer();
  CheckGroupBalance();
  CheckCoordinator();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("collective sim: all checks passed\n");
  return 0;
}
