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
//
// `SwizzleM` is the order the blocks walk the tile grid in, and it is the one
// parameter that moves no byte of the tile: `SwizzleM` m-tiles form a group
// that walks n first, so the blocks resident together share the A panels of the
// group and a run of B panels that fits the second-level cache. Zero walks the
// grid in the plain row-major order.
template <int TileM, int TileN, int TileK, int WarpsM, int WarpsN, int Pad,
          int Stages, int SwizzleM>
struct Plan {
  static constexpr int kTileM = TileM;
  static constexpr int kTileN = TileN;
  static constexpr int kTileK = TileK;
  static constexpr int kWarpsM = WarpsM;
  static constexpr int kWarpsN = WarpsN;
  static constexpr int kStages = Stages;
  static constexpr int kSwizzleM = SwizzleM;
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
  static_assert(SwizzleM >= 0, "a swizzle of zero m-tiles is the plain order");
};

//: The base schedule: a whole 128 by 64 tile, four warps, a 32-deep slice. It
//: is the narrow tile the sweep keeps for a shape no wider schedule serves.
using PlanBase = Plan<128, 64, 32, 2, 2, 8, 2, 0>;
//: A quarter of the tile and half the threads, for a small or a ragged shape.
//: The five by twelve group of the Muon step is forty calls of a few
//: microseconds each, and this is the schedule that measured fastest on it.
using PlanSmall = Plan<64, 64, 32, 1, 2, 8, 2, 0>;
//: The square tile and eight warps that step 3 shipped, with four stages.
using PlanWide = Plan<128, 128, 32, 4, 2, 8, 4, 0>;
//: The square tile with a 64-deep slice: the same 64 flops a byte, half the
//: barriers over k, and two blocks of 72 KiB still fit a Hopper multiprocessor.
//: Step 4's winner, kept without the swizzle as the order's control.
using PlanKDeep = Plan<128, 128, 64, 2, 4, 8, 2, 0>;
//: The same tile and depth, walked in groups of eight m-tiles.
using PlanKDeepSwizzle = Plan<128, 128, 64, 2, 4, 8, 2, 8>;
//: The same tile with a 96-deep slice and the swizzle. Its window holds two
//: blocks of a Hopper multiprocessor, which is where the deeper slice stops
//: paying.
using PlanKDeeper = Plan<128, 128, 96, 2, 4, 8, 2, 8>;
//: A tall tile: twice the columns of the square one, four warps along each
//: axis, and the same sixty-four accumulator registers a thread. It raises the
//: arithmetic for each staged byte from 64 flops to 87, and it lost to the
//: square tile on every row because a block of 512 threads holds one block per
//: multiprocessor. The sweep keeps it as the record of that.
using PlanTall = Plan<128, 256, 32, 4, 4, 8, 2, 8>;

//: The sweep. The launcher and the benchmark index it by position, and the
//: registry of thunks in gemm_bf16.cu holds the same seven in this order. The
//: stage count is part of a name, because two of them differ only in it.
inline constexpr int kScheduleCount = 7;
inline constexpr const char* kScheduleNames[kScheduleCount] = {
    "64x64x32x2",     "128x128x32x4", "128x128x64x2",  "128x128x64x2s8",
    "128x128x96x2s8", "128x64x32x2",  "128x256x32x2s8"};
//: The position of each schedule in the sweep, so a row names a schedule and
//: not a number.
inline constexpr int kScheduleSmall = 0;
inline constexpr int kScheduleWide = 1;
inline constexpr int kScheduleKDeep = 2;
inline constexpr int kScheduleKDeepSwizzle = 3;
inline constexpr int kScheduleKDeeper = 4;
inline constexpr int kScheduleBase = 5;
inline constexpr int kScheduleTall = 6;
//: The schedule a call takes when neither the row nor the environment names
//: one. It is the widest that keeps two blocks on a multiprocessor of the
//: card the sweep measured.
inline constexpr int kDefaultSchedule = kScheduleKDeepSwizzle;

// The tile one block owns, from its place in the order the plan names.
//
// The plain order walks n within a row of m-tiles. The swizzled one forms a
// group of `kSwizzleM` m-tiles and walks n inside that group first, so the
// blocks that are resident together read the A panels of the group and a run of
// B panels that is `kSwizzleM` panels wide instead of one block-row of them.
// The second-level cache is what the order is for: on the classifier row the
// plain order has each block read an A panel that no other resident block
// shares.
//
// A group of the last row of tiles is shorter than `kSwizzleM`, and the mapping
// stays a bijection because the group then spans that shorter count of m-tiles:
// the tiles of the grid divide into groups of `m_span * grid_n`.
//
// The qualifier is the one function here that a device and a host both call:
// the kernel calls it, and `dispatch_test` checks it is a bijection without a
// card. A compiler that describes itself with `__CUDACC__` gets the device
// qualifier, and every other one gets nothing.
#if defined(__CUDACC__)
#define NANOCHAT_GEMM_HOST_DEVICE __host__ __device__
#else
#define NANOCHAT_GEMM_HOST_DEVICE
#endif

template <typename PlanT>
NANOCHAT_GEMM_HOST_DEVICE inline void TileOf(int index, int grid_m, int grid_n,
                                             int* m_tile, int* n_tile) {
  if constexpr (PlanT::kSwizzleM == 0) {
    *m_tile = index / grid_n;
    *n_tile = index - *m_tile * grid_n;
  } else {
    const int span = PlanT::kSwizzleM * grid_n;
    const int group = index / span;
    const int base = group * PlanT::kSwizzleM;
    const int left = grid_m - base;
    const int m_span = left < PlanT::kSwizzleM ? left : PlanT::kSwizzleM;
    const int within = index - group * span;
    *m_tile = base + within % m_span;
    *n_tile = within / m_span;
  }
}

#undef NANOCHAT_GEMM_HOST_DEVICE

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_PLAN_H_
