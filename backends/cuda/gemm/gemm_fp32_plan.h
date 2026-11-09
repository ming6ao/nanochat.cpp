#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_PLAN_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_PLAN_H_

// The schedules of the fp32 shared-memory GEMM
// (docs/kernels.md Part 3, section 6 phase 3).
//
// Phase 3 is the chain leg on Turing. The T4 declares no bf16
// (`backends/cuda/caps.json`), so the post-training chain runs there in fp32,
// and `sm_75` has no warpgroup instruction and no Tensor Memory Accelerator.
// The third mainloop of section 5 is therefore a plain fp32 tile: operands in
// shared memory, accumulators in registers, and no tensor core.
//
// **The window is what separates the two schedules.** `sm_75` holds 64 KiB of
// shared memory for each multiprocessor, so one block of 32 KiB or less leaves
// room for a second. `kPlanWide` asks for 33 KiB and holds one block there.
// `kPlanHalf` asks for 25 KiB and holds two. A card measurement decides which
// one wins, and no argument decides it here.
//
// Precision free: ints and names only, so a host test reads a schedule with no
// device and no CUDA header.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace nanochat {
namespace kernels {
namespace gemm_fp32 {

// One schedule of the fp32 tile.
//
// `ThreadsM` and `ThreadsN` are the thread grid of the block, so
// `TileM / ThreadsM` is the register tile of one thread. That tile is eight by
// eight, which is sixty-four accumulators: with the eight A values and the
// eight B values of one k step it is eighty registers, and two blocks of 256
// threads then fit the register file of one multiprocessor.
//
// Both operands stage k-major, `As[k][m]` and `Bs[k][n]`, so one thread reads
// its whole register tile with four sixteen-byte loads. `Pad` breaks the bank
// that a row stride would repeat, and it is a multiple of four so a
// sixteen-byte load stays aligned.
template <int TileM_, int TileN_, int TileK_, int ThreadsM_, int ThreadsN_,
          int Pad_, int Stages_>
struct Plan {
  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = TileK_;
  static constexpr int kThreadsM = ThreadsM_;
  static constexpr int kThreadsN = ThreadsN_;
  static constexpr int kPad = Pad_;
  static constexpr int kStages = Stages_;

  static constexpr int kThreads = ThreadsM_ * ThreadsN_;
  static constexpr int kThreadTileM = TileM_ / ThreadsM_;
  static constexpr int kThreadTileN = TileN_ / ThreadsN_;

  //: Elements of one staged row of each operand, pad included.
  static constexpr int kLeadM = TileM_ + Pad_;
  static constexpr int kLeadN = TileN_ + Pad_;

  //: Elements of one staged k slice of each operand.
  static constexpr int kStageAElements = TileK_ * kLeadM;
  static constexpr int kStageBElements = TileK_ * kLeadN;

  //: The shared window: the staged slices and nothing else. The epilogue reads
  //: and writes global memory, so it needs no scratch.
  static constexpr int kSmemElements =
      Stages_ * (kStageAElements + kStageBElements);
  static constexpr int kSmemBytes = kSmemElements * 4;

  //: The window of one multiprocessor on `sm_75`, in bytes, and the number of
  //: blocks of this schedule it holds.
  static constexpr int kSm75SmemBytes = 64 * 1024;
  static constexpr int kSm75Blocks =
      kSm75SmemBytes / (kSmemBytes > 0 ? kSmemBytes : 1);

  static_assert(TileM_ % ThreadsM_ == 0 && TileN_ % ThreadsN_ == 0,
                "the thread grid divides the block tile");
  static_assert(kThreadTileM == 8 && kThreadTileN == 8,
                "the kernel reads a thread tile of eight by eight as four "
                "sixteen-byte loads");
  static_assert(kThreads % 32 == 0 && kThreads <= 1024,
                "a block is whole warps and at most 1024 threads");
  static_assert(TileK_ % 4 == 0 && Pad_ % 4 == 0,
                "a staged row and a staged column are whole sixteen-byte "
                "chunks");
  static_assert(kStages >= 2, "one buffer in flight hides no latency");
};

//: The wide schedule: a 128 by 128 tile, two stages, 33 KiB, one block on the
//: T4 and two on the H100.
using PlanWide = Plan<128, 128, 16, 16, 16, 4, 2>;

//: The half schedule: 128 by 64, two stages, 25 KiB, two blocks on the T4. It
//: stages half the bytes of `PlanWide` for each unit of arithmetic, which is
//: the trade a card without spare shared memory has to make.
using PlanHalf = Plan<128, 64, 16, 16, 8, 4, 2>;

//: The default schedule. `PlanWide` is the one the first card run times; the
//: measurement of `docs/kernels.md` section 6 phase 3 replaces this answer.
inline constexpr int kDefaultSchedule = 0;

//: The blocks a schedule asks one multiprocessor to hold. The kernel states it
//: as the second `__launch_bounds__` argument, so ptxas keeps the register
//: count inside the budget that residency needs: two blocks of 256 threads
//: share 64 Ki registers, which is 128 for each thread. Without the argument
//: ptxas is free to spend registers the tile does not have.
inline constexpr int kMinBlocksPerSm = 2;

//: The sweep, by index, for `NANOCHAT_GEMM_FP32_SCHEDULE` and for a card run.
inline constexpr int kScheduleCount = 2;

// The name of one schedule of the sweep, in the spelling a benchmark column and
// `NANOCHAT_GEMM_FP32_SCHEDULE` use. An index outside the sweep names the
// default, so a report never holds an empty column.
inline const char* ScheduleName(int index) {
  switch (index) {
    case 0:
      return "wide";
    case 1:
      return "half";
    default:
      return "wide";
  }
}

//: True when a committed card measurement promoted the tile to the shipped
//: fp32 path. The tile is measured before it is promoted, so this bit is off
//: until a run of `gemm_benchmark` at fp32 reads within 1.05 of cuBLAS on the
//: chain rows (`docs/kernels.md` section 6 phase 3). A run can therefore never
//: change behaviour without that measurement.
inline constexpr bool kPromoted = false;

// The index `NANOCHAT_GEMM_FP32_SCHEDULE` names, or -1 when the variable is
// unset, empty, or names no schedule. A name and not an index, so a card run
// spells the schedule it measures and a reordered sweep cannot silently change
// what a recorded command means.
inline int ScheduleIndexFromEnv() {
  const char* setting = std::getenv("NANOCHAT_GEMM_FP32_SCHEDULE");
  if (setting == nullptr || setting[0] == '\0') return -1;
  for (int index = 0; index < kScheduleCount; ++index) {
    if (std::strcmp(setting, ScheduleName(index)) == 0) return index;
  }
  return -1;
}

}  // namespace gemm_fp32
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_FP32_PLAN_H_
