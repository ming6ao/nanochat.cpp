// The hand-written tensor-core GEMM of docs/kernels.md Part 3
// (docs/tensor-core-gemm-plan.md Part B).
//
// One kernel per schedule over the universal call
// C = alpha * op(A) * op(B) + beta * C. The transpose flags reach the device as
// two stride pairs plus the fragment layout of the operand, so the four operand
// forms take one body each and share the staging code. The accumulators pass
// through a warp-sized fp32 scratch tile in shared memory, because `wmma`
// stores only an fp32 accumulator and C may be bf16; that one store path
// carries the bounds, the alpha and beta terms, and the fp32 weight-gradient
// form.
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

  // The epilogue: one warp at a time stores its accumulator tile into a shared
  // warp tile, and every thread applies alpha, beta, the bounds, and the store
  // type to that sub-tile. One warp tile of scratch instead of the whole tile
  // is what keeps a 128 by 128 schedule inside the static shared budget.
  float* scratch = reinterpret_cast<float*>(smem);
  const int warps = Plan::kWarpsM * Plan::kWarpsN;
  const int stride = static_cast<int>(blockDim.x);
  for (int owner = 0; owner < warps; ++owner) {
    __syncthreads();
    if (warp == owner) {
      for (int i = 0; i < kFragM; ++i) {
        for (int j = 0; j < kFragN; ++j) {
          gemm_bf16::StoreFp32(
              scratch + i * kMicroM * Plan::kWarpN + j * kMicroN, Plan::kWarpN,
              acc[i][j]);
        }
      }
    }
    __syncthreads();

    const int tile_m = (owner / Plan::kWarpsN) * Plan::kWarpM;
    const int tile_n = (owner % Plan::kWarpsN) * Plan::kWarpN;
    for (int index = static_cast<int>(threadIdx.x);
         index < Plan::kWarpM * Plan::kWarpN; index += stride) {
      const int i = index / Plan::kWarpN;
      const int j = index - i * Plan::kWarpN;
      const int row = m0 + tile_m + i;
      const int column = n0 + tile_n + j;
      if (row >= args.m || column >= args.n) continue;
      const long long offset = batch * args.stride_c +
                               static_cast<long long>(row) * args.ldc + column;
      float value = args.alpha * scratch[index];
      if (args.beta != 0.0f) value += args.beta * static_cast<float>(c[offset]);
      c[offset] = static_cast<COut>(value);
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
