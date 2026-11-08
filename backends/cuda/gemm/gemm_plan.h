#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_

// The schedules of the hand tensor-core GEMM, and the sweep that chooses one
// (docs/tensor-core-gemm-plan.md sections B.4 and B.7).
//
// A schedule is the block tile, the warp grid, the k depth, the pad, and the
// depth of the copy pipeline. All of it is compile time, because the fragment
// arrays in the kernel are sized by it.
//
// **The staged window is the dynamic one, and its size is a residency
// decision.** A pipeline of four stages reaches past the 48 KiB static window,
// so `gemm_tile.cuh` states the size and the launcher asks for the opt-in limit
// with `cudaFuncSetAttribute`. It is a trade and not a free gain: every extra
// stage costs a share of the multiprocessor, and a tile that is short of
// resident warps loses more than the deeper copy returns
// (docs/kernels.md Part 3 section 10 step 3). The window holds the staged
// operands and no epilogue scratch: the accumulator reaches C through one small
// exchange tile per warp, which lies over the staged buffers once the k loop is
// done.
//
// Precision free: ints and names only, so the shape table and a host test can
// name a schedule without a device.

#include <cstdint>

namespace nanochat {
namespace kernels {
namespace gemm_tile {

// One schedule. `Stages` is the depth of the copy pipeline: the copy of slice
// `n + Stages - 1` overlaps the arithmetic of slice `n`, so `Stages - 1` copies
// may be in flight. The window a schedule asks for decides how many blocks a
// multiprocessor holds, so a deeper pipeline is a trade and not a gain.
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
  //: The shared window: the staged buffers and nothing else. `gemm_tile.cuh`
  //: turns this into the byte count a launch asks for.
  static constexpr int kSmemElements = kStageElements;

  static_assert(TileM % (WarpsM * 16) == 0 && TileN % (WarpsN * 16) == 0,
                "the warp grid must cover the tile in whole fragments");
  static_assert(TileK % 16 == 0, "one k slice must split into whole fragments");
  static_assert(Pad % 8 == 0 && kLeadARow % 8 == 0 && kLeadBRow % 8 == 0,
                "a staged row must keep every sixteen-byte chunk aligned");
  static_assert(kLeadACol % 8 == 0 && kLeadBCol % 8 == 0,
                "a staged column must keep every sixteen-byte chunk aligned");
  static_assert(TileK % 8 == 0, "a staged row must be whole chunks");
  static_assert(Stages >= 2 && Stages <= 8,
                "a pipeline of two to eight stages");
};

//: The base schedule: a whole 128 by 64 tile, four warps, a 32-deep slice.
using PlanBase = Plan<128, 64, 32, 2, 2, 8, 2>;
//: A quarter of the tile and half the threads, for a small or a ragged shape.
using PlanSmall = Plan<64, 64, 32, 1, 2, 8, 2>;
//: A square tile and eight warps, for a card of the Ampere class and above.
//: Twice the arithmetic of the base schedule for one and a third times its
//: staged bytes, and sixteen warps per SM instead of eight.
using PlanWide = Plan<128, 128, 32, 4, 2, 8, 4>;
//: The same tile with two stages. The sweep keeps it as the record of what the
//: deeper pipeline is worth on this tile: four stages gained 3 to 5 percent on
//: five rows, because two blocks of twice the window still fit a Hopper
//: multiprocessor. A small tile takes the shallower pipeline, where every extra
//: stage costs a block.
using PlanShallow = Plan<128, 128, 32, 4, 2, 8, 2>;

//: The sweep. The launcher and the benchmark index it by position, and the
//: registry of thunks in gemm_bf16.cu holds the same four in this order. The
//: stage count is part of a name, because two of them differ only in it.
inline constexpr int kScheduleCount = 4;
inline constexpr const char* kScheduleNames[kScheduleCount] = {
    "128x64x32x2", "64x64x32x2", "128x128x32x2", "128x128x32x4"};
//: The position of each schedule in the sweep, so a row names a schedule and
//: not a number.
inline constexpr int kScheduleBase = 0;
inline constexpr int kScheduleSmall = 1;
inline constexpr int kScheduleShallow = 2;
inline constexpr int kScheduleWide = 3;
//: The schedule a call takes when neither the row nor the environment names
//: one. It is the widest that keeps two blocks on a multiprocessor of the
//: card the sweep measured.
inline constexpr int kDefaultSchedule = kScheduleWide;

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
