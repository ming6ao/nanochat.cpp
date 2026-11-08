// The hand-written tensor-core GEMM of docs/kernels.md Part 3
// (docs/tensor-core-gemm-plan.md Part B).
//
// One kernel per schedule over the universal call
// C = alpha * op(A) * op(B) + beta * C. The transpose flags reach the device as
// two stride pairs plus the fragment layout of the operand, so the four operand
// forms take one body each and share the staging code. The accumulator goes
// straight to C through the lane map of the instruction, so the epilogue holds
// the bounds, the alpha and beta terms, and the fp32 weight-gradient form with
// no shared scratch and no serialized round of warps.
//
// The k loop is the two-stage copy of gemm_tile.cuh: slice `n + 1` is copied
// while slice `n` is used, and a barrier after the arithmetic holds the buffer
// until every warp is done with it.
//
// The arithmetic is the `ldmatrix` and `mma.sync` pair of mma_bf16.cuh and not
// the vendor `wmma` pair, which is mechanism 1 of docs/kernels.md Part 3
// section
// 4. The micro-tile is therefore 16 by 8 and not 16 by 16, so a warp tile holds
// twice as many fragments along n and each one is half as wide.
//
// See backends/cuda/gemm/gemm_tile.cuh for the stage, gemm_plan.h for the
// schedules, and gemm_launch.h for the entry points.
//
// This source exists only in a bf16 or fp8 build, so it holds no guard: the
// build selects it (`backends/cuda/gemm/BUILD.bazel`).

#include "backends/cuda/gemm/gemm_launch.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/gemm_tile.cuh"
#include "backends/cuda/gemm/mma_bf16.cuh"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

namespace {

using gemm_tile::TileElement;

//: The micro-tile of the instruction pair. The k edge is the loop step of the
//: staged slice, and the other two are the fragment's edges.
constexpr int kMicroM = gemm_bf16::kMmaM;
constexpr int kMicroN = gemm_bf16::kMmaN;
constexpr int kMicro = gemm_tile::kMicro;
static_assert(kMicro == gemm_bf16::kMmaK,
              "the k step of the slice must be the k edge of the micro-tile");

// C is `COut` and the accumulator is fp32, so one epilogue serves the bf16
// store of `Gemm` and the fp32 store of `GemmWgrad`.
template <typename PlanT, typename COut, bool TransA, bool TransB>
__global__ void GemmTileKernel(const ComputeType* a, const ComputeType* b,
                               COut* c, ResolvedGemm args) {
  using Plan = PlanT;
  constexpr int kMicro = gemm_tile::kMicro;

  // `alignas` leads: the emulator maps `__shared__` to `static`, and a standard
  // attribute may not follow a storage class.
  alignas(16) __shared__ TileElement smem[Plan::kSmemElements];
  TileElement* stage_a = smem;
  TileElement* stage_b = smem + Plan::kStages * Plan::kStageAElements;

  const int n0 = static_cast<int>(blockIdx.x) * Plan::kTileN;
  const int m0 = static_cast<int>(blockIdx.y) * Plan::kTileM;
  const long long batch = static_cast<long long>(blockIdx.z);

  // The two stride pairs of the operand forms.
  const TileElement* a_tile = reinterpret_cast<const TileElement*>(a);
  const TileElement* b_tile = reinterpret_cast<const TileElement*>(b);
  const TileElement* a_base = a_tile + batch * args.stride_a;
  const TileElement* b_base = b_tile + batch * args.stride_b;

  // The staged row width of each operand, which is the fragment's leading
  // dimension in both the staged layouts.
  const int a_lead = TransA ? Plan::kLeadACol : Plan::kLeadARow;
  const int b_lead = TransB ? Plan::kLeadBCol : Plan::kLeadBRow;

  // The region of C this warp owns.
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int warp_m = (warp / Plan::kWarpsN) * Plan::kWarpM;
  const int warp_n = (warp % Plan::kWarpsN) * Plan::kWarpN;

  //: One row of the warp tile is a whole number of micro-tiles, and so is one
  //: column. The row count is the one `gemm_plan.h` records; the column count
  //: is twice it, because the instruction is half as wide as the vendor
  //: fragment.
  constexpr int kFragM = Plan::kWarpM / kMicroM;
  constexpr int kFragN = Plan::kWarpN / kMicroN;

  gemm_bf16::FragmentC acc[kFragM][kFragN];
  for (int i = 0; i < kFragM; ++i) {
    for (int j = 0; j < kFragN; ++j) {
      gemm_bf16::Fill(acc[i][j], 0.0f);
    }
  }

  // Stages one k slice into one buffer. Every thread calls it, so the two
  // ordering calls that follow are uniform across the block.
  auto stage_slice = [&](int k0, int buffer) {
    gemm_tile::StageSlice<Plan>(a_base, args.lda, TransA, b_base, args.ldb,
                                TransB, m0, n0, args.m, args.n, k0, args.k,
                                stage_a + buffer * Plan::kStageAElements,
                                stage_b + buffer * Plan::kStageBElements);
  };

  stage_slice(0, 0);
  gemm_tile::CommitStage();

  for (int slice = 0, k0 = 0; k0 < args.k; ++slice, k0 += Plan::kTileK) {
    const int buffer = slice % Plan::kStages;
    if ((k0 + Plan::kTileK) < args.k) {
      stage_slice(k0 + Plan::kTileK, (slice + 1) % Plan::kStages);
    }
    gemm_tile::CommitStage();
    // Every iteration commits one group, so waiting for all but the newest
    // completes this slice's copy. The last iteration commits an empty group.
    gemm_tile::WaitStage(1);
    __syncthreads();

    const TileElement* a_slice = stage_a + buffer * Plan::kStageAElements;
    const TileElement* b_slice = stage_b + buffer * Plan::kStageBElements;
    for (int kk = 0; kk < Plan::kTileK; kk += kMicro) {
      gemm_bf16::FragmentA fa[kFragM];
      gemm_bf16::FragmentB fb[kFragN];
      for (int i = 0; i < kFragM; ++i) {
        const int row = warp_m + i * kMicroM;
        // A plain operand stages m by k, so the micro-tile starts at (row, kk).
        // A transposed one stages k by m, so it starts at (kk, row).
        const TileElement* tile =
            TransA ? a_slice + row + kk * a_lead : a_slice + row * a_lead + kk;
        gemm_bf16::LoadA<TransA>(fa[i], tile, a_lead);
      }
      for (int j = 0; j < kFragN; ++j) {
        const int column = warp_n + j * kMicroN;
        // A plain operand stages k by n, so the micro-tile starts at
        // (kk, column). A transposed one stages n by k, so it starts at
        // (column, kk).
        const TileElement* tile = TransB ? b_slice + kk + column * b_lead
                                         : b_slice + kk * b_lead + column;
        gemm_bf16::LoadB<TransB>(fb[j], tile, b_lead);
      }
      for (int i = 0; i < kFragM; ++i) {
        for (int j = 0; j < kFragN; ++j) {
          gemm_bf16::Mma(acc[i][j], fa[i], fb[j]);
        }
      }
    }
    // Every warp is done with this buffer, so the next slice may overwrite it.
    __syncthreads();
  }

  // The epilogue. Each warp's accumulator passes through a small exchange tile
  // in shared memory and leaves it as sixteen-byte chunks of C, which is the
  // one bulk store the state of the art uses for this subtile. A lane's own
  // registers hold eight columns of two rows, so a store taken straight from
  // the lane map would put one warp instruction on eight rows of sixteen bytes;
  // the exchange tile puts the run of columns back together first.
  //
  // The tile is per warp, so the exchange of one warp never waits for another,
  // and it lies over the staged window, which the finished k loop releases.
  constexpr int kWarps = Plan::kWarpsM * Plan::kWarpsN;
  constexpr int kExchangeN = 2 * kMicroN;  // two neighbouring fragments along n
  constexpr int kExchangeElements = kMicroM * kExchangeN;
  //: Sixteen bytes, the widest store a thread makes. A plain aligned structure
  //: of four words states that width for the device compiler and for the host
  //: pass the emulation engine runs, which holds no `float4`.
  struct alignas(16) SixteenBytes {
    std::uint32_t word[4];
  };
  constexpr int kExchangeBytes = 16;
  constexpr int kChunkElements =
      kExchangeBytes / static_cast<int>(sizeof(COut));
  constexpr int kChunksPerRow = kExchangeN / kChunkElements;
  static_assert(kFragN % 2 == 0, "an exchange covers two fragments along n");
  static_assert(
      kChunkElements * static_cast<int>(sizeof(COut)) == kExchangeBytes,
      "a chunk is one store wide");
  static_assert(kWarps * kExchangeElements * static_cast<int>(sizeof(COut)) <=
                    Plan::kSmemElements * static_cast<int>(sizeof(TileElement)),
                "the exchange tile lies over the staged window");

  // The k loop is over, so the staged window may hold the exchange tiles.
  gemm_tile::WaitStage(0);
  __syncthreads();
  COut* exchange = reinterpret_cast<COut*>(smem) + warp * kExchangeElements;

  const int lane = static_cast<int>(threadIdx.x) % 32;
  COut* c_base = c + batch * args.stride_c;
  const int warp_rows = m0 + warp_m;
  const int warp_columns = n0 + warp_n;
  for (int i = 0; i < kFragM; ++i) {
    for (int pair = 0; pair < kFragN; pair += 2) {
      // The accumulator lands in the exchange tile, with alpha, beta, the
      // bounds and the store type applied there: one rounding, whatever beta
      // asks for. An element outside C is dropped, and the write below skips
      // the chunk it belongs to.
      const int fragment_row = warp_rows + i * kMicroM;
      for (int half = 0; half < 2; ++half) {
        const int j = pair + half;
        const int fragment_column = warp_columns + j * kMicroN;
        for (int reg = 0; reg < gemm_bf16::kLaneRegisters; ++reg) {
          int row = 0;
          int column = 0;
          gemm_bf16::ElementOf(lane, reg, &row, &column);
          float value =
              args.alpha * gemm_bf16::ElementValue(acc[i][j], lane, reg);
          const int out_row = fragment_row + row;
          const int out_column = fragment_column + column;
          if (args.beta != 0.0f && out_row < args.m && out_column < args.n) {
            const long long offset =
                static_cast<long long>(out_row) * args.ldc + out_column;
            value += args.beta * static_cast<float>(c_base[offset]);
          }
          exchange[row * kExchangeN + half * kMicroN + column] =
              static_cast<COut>(value);
        }
      }
      __syncwarp();

      // One chunk a lane, so a warp covers the sixteen columns of sixteen rows
      // in as few stores as the tile allows, and each is contiguous.
      const int base_column = warp_columns + pair * kMicroN;
      for (int chunk = lane; chunk < kMicroM * kChunksPerRow; chunk += 32) {
        const int row = chunk / kChunksPerRow;
        const int column = (chunk - row * kChunksPerRow) * kChunkElements;
        const int out_row = fragment_row + row;
        const int out_column = base_column + column;
        if (out_row >= args.m) continue;
        const COut* src = exchange + row * kExchangeN + column;
        COut* dst =
            c_base + static_cast<long long>(out_row) * args.ldc + out_column;
        if (out_column + kChunkElements > args.n) {
          // A ragged edge takes the elements it covers, one at a time.
          for (int e = 0; e < kChunkElements; ++e) {
            if (out_column + e >= args.n) break;
            dst[e] = src[e];
          }
          continue;
        }
        if ((reinterpret_cast<std::uintptr_t>(dst) % kExchangeBytes) == 0u) {
          *reinterpret_cast<SixteenBytes*>(dst) =
              *reinterpret_cast<const SixteenBytes*>(src);
          continue;
        }
        // A destination the store cannot take in one piece.
        for (int e = 0; e < kChunkElements; ++e) dst[e] = src[e];
      }
      // The next exchange writes the tile again, so this one must be read.
      __syncwarp();
    }
  }
}

// Resolves the call, then launches one grid of that schedule.
template <typename PlanT, typename COut, bool TransA, bool TransB>
bool LaunchPlan(const GemmParams& params, const ComputeType* a,
                const ComputeType* b, COut* c) {
  const ResolvedGemm gemm =
      ResolveGemm(params, /*fp32_c=*/std::is_same_v<COut, float>);
  if (!IsTileCallable(gemm)) return false;

  const unsigned int grid_n =
      static_cast<unsigned int>((gemm.n + PlanT::kTileN - 1) / PlanT::kTileN);
  const unsigned int grid_m =
      static_cast<unsigned int>((gemm.m + PlanT::kTileM - 1) / PlanT::kTileM);
  cuda_backend::Launch(
      GemmTileKernel<PlanT, COut, TransA, TransB>,
      dim3(grid_n, grid_m, static_cast<unsigned int>(gemm.batch_count)),
      dim3(static_cast<unsigned int>(PlanT::kThreads)), 0, a, b, c, gemm);
  return true;
}

// The four operand forms of one store type.
template <typename PlanT, typename COut>
bool LaunchForm(const GemmParams& params, const ComputeType* a,
                const ComputeType* b, COut* c) {
  if (params.transpose_a) {
    return params.transpose_b
               ? LaunchPlan<PlanT, COut, true, true>(params, a, b, c)
               : LaunchPlan<PlanT, COut, true, false>(params, a, b, c);
  }
  return params.transpose_b
             ? LaunchPlan<PlanT, COut, false, true>(params, a, b, c)
             : LaunchPlan<PlanT, COut, false, false>(params, a, b, c);
}

// The sweep: one thunk per schedule, in the order of `kScheduleNames`.
using ScheduleThunk = bool (*)(const GemmParams&, const ComputeType*,
                               const ComputeType*, void*, bool);

template <typename PlanT>
bool Thunk(const GemmParams& params, const ComputeType* a, const ComputeType* b,
           void* c, bool fp32_c) {
  return fp32_c ? LaunchForm<PlanT>(params, a, b, static_cast<float*>(c))
                : LaunchForm<PlanT>(params, a, b, static_cast<ComputeType*>(c));
}

const ScheduleThunk kThunks[gemm_tile::kScheduleCount] = {
    &Thunk<gemm_tile::PlanBase>,
    &Thunk<gemm_tile::PlanSmall>,
    &Thunk<gemm_tile::PlanDeep>,
    &Thunk<gemm_tile::PlanWide>,
};

static_assert(sizeof(kThunks) / sizeof(kThunks[0]) == gemm_tile::kScheduleCount,
              "one thunk per name in the sweep");

bool LaunchWith(int schedule, const GemmParams& params, const ComputeType* a,
                const ComputeType* b, void* c, bool fp32_c) {
  const int index = (schedule >= 0 && schedule < gemm_tile::kScheduleCount)
                        ? schedule
                        : gemm_tile::kDefaultSchedule;
  return kThunks[index](params, a, b, c, fp32_c);
}

}  // namespace

bool LaunchGemmTile(const GemmParams& params, const ComputeType* a,
                    const ComputeType* b, ComputeType* c, int schedule) {
  return LaunchWith(schedule, params, a, b, c, /*fp32_c=*/false);
}

bool LaunchGemmTileToFp32(const GemmParams& params, const ComputeType* a,
                          const ComputeType* b, float* c, int schedule) {
  return LaunchWith(schedule, params, a, b, c, /*fp32_c=*/true);
}

}  // namespace kernels
}  // namespace nanochat
