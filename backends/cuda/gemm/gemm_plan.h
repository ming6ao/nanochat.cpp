#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_

// The schedules of the hand tensor-core GEMM, and the sweep that chooses one
// (docs/tensor-core-gemm-plan.md sections B.4 and B.7).
//
// A schedule is the block tile, the warp grid, the k depth, the pad, and the
// number of staged buffers. All of it is compile time, because the fragment
// arrays in the kernel are sized by it.
//
// Every schedule here fits the 48 KiB recurring limit of the shared window, so
// none of them needs `cudaFuncSetAttribute` and the emulator needs no dynamic
// shared memory. The window is the larger of the staged operands and the fp32
// epilogue scratch.
//
// Precision free: ints and names only, so the shape table and a host test can
// name a schedule without a device.

#include <cstdint>

namespace nanochat {
namespace kernels {
namespace gemm_tile {

// One schedule. `Stages` is the number of staged operand buffers, and it is two
// for every schedule that ships: the copy of slice `n + 1` overlaps the
// arithmetic of slice `n`.
template <int TileM, int TileN, int TileK, int WarpsM, int WarpsN, int Pad,
          int Stages>
struct Plan {
  static constexpr int kTileM = TileM;
  static constexpr int kTileN = TileN;
  static constexpr int kTileK = TileK;
  static constexpr int kWarpsM = WarpsM;
  static constexpr int kWarpsN = WarpsN;
  static constexpr int kStages = Stages;
  static constexpr int kThreads = WarpsM * WarpsN * 32;

  static constexpr int kWarpM = TileM / WarpsM;
  static constexpr int kWarpN = TileN / WarpsN;
  static constexpr int kFragsM = kWarpM / 16;
  static constexpr int kFragsN = kWarpN / 16;

  static constexpr int kLeadA = TileK + Pad;
  static constexpr int kLeadB = TileN + Pad;

  //: Elements of one staged A buffer and one staged B buffer.
  static constexpr int kStageAElements = TileM * kLeadA;
  static constexpr int kStageBElements = TileK * kLeadB;
  static constexpr int kStageElements =
      Stages * (kStageAElements + kStageBElements);
  //: Floats of the epilogue scratch, one per element of C, held as two halves.
  static constexpr int kScratchElements = TileM * TileN * 2;
  static constexpr int kSmemElements =
      kStageElements > kScratchElements ? kStageElements : kScratchElements;

  static_assert(TileM % (WarpsM * 16) == 0 && TileN % (WarpsN * 16) == 0,
                "the warp grid must cover the tile in whole fragments");
  static_assert(TileK % 16 == 0, "one k slice must split into whole fragments");
  static_assert(Pad % 8 == 0 && kLeadA % 8 == 0 && kLeadB % 8 == 0,
                "a staged row must keep every sixteen-byte chunk aligned");
  static_assert(TileK % 8 == 0, "a staged row must be whole chunks");
  static_assert(Stages == 2, "the k loop is the two-stage pipeline");
};

//: The base schedule: a whole 128 by 64 tile, four warps, a 32-deep slice.
using PlanBase = Plan<128, 64, 32, 2, 2, 8, 2>;
//: A quarter of the tile and half the threads, for a small or a ragged shape.
using PlanSmall = Plan<64, 64, 32, 1, 2, 8, 2>;
//: The same tile with a 64-deep slice, which halves the barrier count over k.
using PlanDeep = Plan<64, 64, 64, 1, 2, 8, 2>;

//: The sweep. The launcher and the benchmark index it by position, and the
//: registry of thunks in gemm_bf16.cu holds the same three in this order.
inline constexpr int kScheduleCount = 3;
inline constexpr const char* kScheduleNames[kScheduleCount] = {
    "128x64x32", "64x64x32", "64x64x64"};
inline constexpr int kDefaultSchedule = 0;

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
