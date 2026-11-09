// The host gate of the fp32 tile's schedules
// (docs/kernels.md Part 3, section 6 phase 3).
//
// The arithmetic of a schedule decides two things before any card sees it: the
// staged window, which decides how many blocks a multiprocessor holds, and the
// register tile, which decides how many registers a thread asks for. Both are
// compile-time facts, so this test states them and needs no device.

#include <cstdio>

#include "backends/cuda/gemm/gemm_fp32_plan.h"

namespace {

int g_failures = 0;

void Expect(bool condition, const char* what) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", what);
  ++g_failures;
}

template <typename Plan>
void CheckPlan(const char* name) {
  // The register tile is eight by eight, which the kernel reads as four
  // sixteen-byte loads. The launcher and the kernel both assume it.
  Expect(Plan::kThreadTileM == 8 && Plan::kThreadTileN == 8,
         "the thread tile is eight by eight");
  // A staged row is a whole number of sixteen-byte chunks, so every vector load
  // from the window is aligned.
  Expect(Plan::kLeadM % 4 == 0 && Plan::kLeadN % 4 == 0,
         "a staged row is a whole number of vectors");
  Expect(Plan::kTileK % 4 == 0, "a staged slice is a whole number of vectors");
  // The static window of a launch is 48 KiB. A schedule past it would need the
  // opt-in limit, which the launcher does not ask for.
  Expect(Plan::kSmemBytes <= 48 * 1024, "the window fits the static limit");
  // Two blocks of 256 threads at 128 registers is the register file of one
  // multiprocessor, which is the residency the tile is built for.
  Expect(Plan::kThreads * 128 <= 64 * 1024, "two blocks fit the register file");
  std::printf("%s: %d bytes, %d blocks on sm_75, %d threads\n", name,
              Plan::kSmemBytes, Plan::kSm75Blocks, Plan::kThreads);
}

}  // namespace

int main() {
  using nanochat::kernels::gemm_fp32::PlanHalf;
  using nanochat::kernels::gemm_fp32::PlanWide;

  CheckPlan<PlanWide>("wide");
  CheckPlan<PlanHalf>("half");

  // The two schedules differ in what they stage, and the difference is the
  // point: the wide one asks for 33 KiB, so the T4 holds one block of it, and
  // the half one asks for 25 KiB, so the T4 holds two.
  Expect(PlanWide::kSmemBytes == 33792, "the wide window is 33 KiB");
  Expect(PlanHalf::kSmemBytes == 25600, "the half window is 25 KiB");
  Expect(PlanWide::kSm75Blocks == 1, "the wide schedule holds one block");
  Expect(PlanHalf::kSm75Blocks == 2, "the half schedule holds two blocks");

  // The sweep is what a card run walks, so its count and its default have to
  // agree with the thunk table in `gemm_fp32.cu`.
  Expect(nanochat::kernels::gemm_fp32::kScheduleCount == 2,
         "the sweep holds two schedules");
  Expect(nanochat::kernels::gemm_fp32::kDefaultSchedule >= 0 &&
             nanochat::kernels::gemm_fp32::kDefaultSchedule <
                 nanochat::kernels::gemm_fp32::kScheduleCount,
         "the default schedule is inside the sweep");

  if (g_failures != 0) {
    std::fprintf(stderr, "gemm fp32 plan test: %d check(s) failed\n",
                 g_failures);
    return 1;
  }
  std::printf("gemm fp32 plan test: ok\n");
  return 0;
}
