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

// One schedule. Every schedule is two buffered: the copy of slice `n + 1`
// overlaps the arithmetic of slice `n`. A third buffer measured no faster, so
// the pipeline stays two deep and needs no stage count.
template <int TileM, int TileN, int TileK, int WarpsM, int WarpsN, int Pad>
struct Plan {
  static constexpr int kTileM = TileM;
  static constexpr int kTileN = TileN;
  static constexpr int kTileK = TileK;
  static constexpr int kWarpsM = WarpsM;
  static constexpr int kWarpsN = WarpsN;
  static constexpr int kStages = 2;
  static constexpr int kThreads = WarpsM * WarpsN * 32;

  static constexpr int kWarpM = TileM / WarpsM;
  static constexpr int kWarpN = TileN / WarpsN;
  static constexpr int kFragsM = kWarpM / 16;
  static constexpr int kFragsN = kWarpN / 16;

  // The two staged layouts of one operand. A plain operand stages `TileM` rows
  // of `TileK` elements and a transposed one stages `TileK` rows of `TileM`, so
  // the two leads differ and the buffer covers the larger.
  static constexpr int kLeadARow = TileK + Pad;
  static constexpr int kLeadACol = TileM + Pad;
  static constexpr int kLeadBRow = TileN + Pad;
  static constexpr int kLeadBCol = TileK + Pad;

  //: Elements of one staged A buffer and one staged B buffer, whichever
  //: transpose form the call takes.
  static constexpr int kStageAElements =
      (TileM * kLeadARow) > (TileK * kLeadACol) ? (TileM * kLeadARow)
                                                : (TileK * kLeadACol);
  static constexpr int kStageBElements =
      (TileK * kLeadBRow) > (TileN * kLeadBCol) ? (TileK * kLeadBRow)
                                                : (TileN * kLeadBCol);
  static constexpr int kStageElements =
      kStages * (kStageAElements + kStageBElements);
  //: Floats of the epilogue scratch: one warp tile, held as two halves, which
  //: every warp writes in turn. The whole tile would cost a fifth plan of
  //: shared memory, and the tile store is once per block.
  static constexpr int kScratchElements = kWarpM * kWarpN * 2;
  static constexpr int kSmemElements =
      kStageElements > kScratchElements ? kStageElements : kScratchElements;

  static_assert(TileM % (WarpsM * 16) == 0 && TileN % (WarpsN * 16) == 0,
                "the warp grid must cover the tile in whole fragments");
  static_assert(TileK % 16 == 0, "one k slice must split into whole fragments");
  static_assert(Pad % 8 == 0 && kLeadARow % 8 == 0 && kLeadBRow % 8 == 0,
                "a staged row must keep every sixteen-byte chunk aligned");
  static_assert(kLeadACol % 8 == 0 && kLeadBCol % 8 == 0,
                "a staged column must keep every sixteen-byte chunk aligned");
  static_assert(TileK % 8 == 0, "a staged row must be whole chunks");
};

//: The base schedule: a whole 128 by 64 tile, four warps, a 32-deep slice.
using PlanBase = Plan<128, 64, 32, 2, 2, 8>;
//: A quarter of the tile and half the threads, for a small or a ragged shape.
using PlanSmall = Plan<64, 64, 32, 1, 2, 8>;
//: The same tile with a 64-deep slice, which halves the barrier count over k.
using PlanDeep = Plan<64, 64, 64, 1, 2, 8>;
//: A square tile and eight warps, for a card of the Ampere class and above.
//: Twice the arithmetic of the base schedule for one and a third times its
//: staged bytes, and sixteen warps per SM instead of eight.
using PlanWide = Plan<128, 128, 32, 4, 2, 8>;

//: The sweep. The launcher and the benchmark index it by position, and the
//: registry of thunks in gemm_bf16.cu holds the same four in this order.
inline constexpr int kScheduleCount = 4;
inline constexpr const char* kScheduleNames[kScheduleCount] = {
    "128x64x32", "64x64x32", "64x64x64", "128x128x32"};
//: The position of each schedule in the sweep, so a row names a schedule and
//: not a number.
inline constexpr int kScheduleBase = 0;
inline constexpr int kScheduleSmall = 1;
inline constexpr int kScheduleDeep = 2;
inline constexpr int kScheduleWide = 3;
//: The schedule a call takes when neither the row nor the environment names
//: one. It is the widest that fits the static shared window of every card the
//: tile supports.
inline constexpr int kDefaultSchedule = kScheduleWide;

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
