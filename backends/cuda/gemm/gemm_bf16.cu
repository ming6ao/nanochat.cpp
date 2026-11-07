// The hand-written tensor-core GEMM of docs/kernels.md Part 3
// (docs/tensor-core-gemm-plan.md Part B).
//
// One kernel per schedule over the universal call
// C = alpha * op(A) * op(B) + beta * C. The transpose flags reach the device as
// two stride pairs, so the four operand forms are one body. The accumulators
// pass through an fp32 scratch tile in shared memory, because `wmma` stores
// only an fp32 accumulator and C may be bf16; that one store path carries the
// bounds, the alpha and beta terms, and the fp32 weight-gradient form.
//
// The k loop is the two-stage pipeline of gemm_tile.cuh: slice `n + 1` is
// copied while slice `n` is used, and a barrier after the arithmetic holds the
// buffer until every warp is done with it.
//
// See backends/cuda/gemm/gemm_tile.cuh for the stage, gemm_plan.h for the
// schedules, and gemm_launch.h for the entry points.

#if defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#include "backends/cuda/gemm/gemm_launch.h"

#include <cstddef>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/gemm_tile.cuh"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

namespace {

namespace wmma = nvcuda::wmma;

using gemm_tile::TileElement;

// C is `COut` and the accumulator is fp32, so one epilogue serves the bf16
// store of `Gemm` and the fp32 store of `GemmWgrad`.
template <typename PlanT, typename COut>
__global__ void GemmTileKernel(const ComputeType* a, const ComputeType* b,
                               COut* c, ResolvedGemm args) {
  using Plan = PlanT;
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
  const TileElement* a_base =
      a_tile + batch * args.stride_a +
      (args.transpose_a ? m0 : static_cast<long long>(m0) * args.lda);
  const TileElement* b_base =
      b_tile + batch * args.stride_b +
      (args.transpose_b ? static_cast<long long>(n0) * args.ldb : n0);
  const int a_stride_i = args.transpose_a ? 1 : args.lda;
  const int a_stride_k = args.transpose_a ? args.lda : 1;
  const int b_stride_k = args.transpose_b ? 1 : args.ldb;
  const int b_stride_j = args.transpose_b ? args.ldb : 1;

  // The region of C this warp owns.
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int warp_m = (warp / Plan::kWarpsN) * Plan::kWarpM;
  const int warp_n = (warp % Plan::kWarpsN) * Plan::kWarpN;

  wmma::fragment<wmma::accumulator, gemm_tile::kMicro, gemm_tile::kMicro,
                 gemm_tile::kMicro, float>
      acc[Plan::kFragsM][Plan::kFragsN];
  for (int i = 0; i < Plan::kFragsM; ++i) {
    for (int j = 0; j < Plan::kFragsN; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  // Stages one k slice into one buffer. Every thread calls it, so the two
  // ordering calls that follow are uniform across the block.
  auto stage_slice = [&](int k0, int buffer) {
    gemm_tile::StageSlice<Plan>(
        a_base, a_stride_i, a_stride_k, b_base, b_stride_k, b_stride_j, m0, n0,
        args.m, args.n, k0, args.k, stage_a + buffer * Plan::kStageAElements,
        stage_b + buffer * Plan::kStageBElements);
  };

  stage_slice(0, 0);
  gemm_tile::CommitStage();

  for (int slice = 0, k0 = 0; k0 < args.k; ++slice, k0 += Plan::kTileK) {
    const int buffer = slice % Plan::kStages;
    const bool more = (k0 + Plan::kTileK) < args.k;
    if (more) stage_slice(k0 + Plan::kTileK, (slice + 1) % Plan::kStages);
    gemm_tile::CommitStage();
    // Every iteration commits one group, so waiting for all but the newest
    // completes this slice's copy. The last iteration commits an empty group.
    gemm_tile::WaitStage(1);
    __syncthreads();

    const TileElement* a_slice = stage_a + buffer * Plan::kStageAElements;
    const TileElement* b_slice = stage_b + buffer * Plan::kStageBElements;
    for (int kk = 0; kk < Plan::kTileK; kk += gemm_tile::kMicro) {
      wmma::fragment<wmma::matrix_a, gemm_tile::kMicro, gemm_tile::kMicro,
                     gemm_tile::kMicro, TileElement, wmma::row_major>
          fa[Plan::kFragsM];
      wmma::fragment<wmma::matrix_b, gemm_tile::kMicro, gemm_tile::kMicro,
                     gemm_tile::kMicro, TileElement, wmma::row_major>
          fb[Plan::kFragsN];
      for (int i = 0; i < Plan::kFragsM; ++i) {
        wmma::load_matrix_sync(
            fa[i],
            a_slice + (warp_m + i * gemm_tile::kMicro) * Plan::kLeadA + kk,
            Plan::kLeadA);
      }
      for (int j = 0; j < Plan::kFragsN; ++j) {
        wmma::load_matrix_sync(
            fb[j],
            b_slice + kk * Plan::kLeadB + (warp_n + j * gemm_tile::kMicro),
            Plan::kLeadB);
      }
      for (int i = 0; i < Plan::kFragsM; ++i) {
        for (int j = 0; j < Plan::kFragsN; ++j) {
          wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
      }
    }
    // Every warp is done with this buffer, so the next slice may overwrite it.
    __syncthreads();
  }

  // The epilogue: the accumulators become an fp32 tile in shared memory, and
  // one predicated pass applies alpha, beta, the bounds, and the store type.
  float* scratch = reinterpret_cast<float*>(smem);
  for (int i = 0; i < Plan::kFragsM; ++i) {
    for (int j = 0; j < Plan::kFragsN; ++j) {
      wmma::store_matrix_sync(
          scratch + (warp_m + i * gemm_tile::kMicro) * Plan::kTileN +
              (warp_n + j * gemm_tile::kMicro),
          acc[i][j], Plan::kTileN, wmma::mem_row_major);
    }
  }
  __syncthreads();

  const int stride = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x);
       index < Plan::kTileM * Plan::kTileN; index += stride) {
    const int i = index / Plan::kTileN;
    const int j = index - i * Plan::kTileN;
    const int row = m0 + i;
    const int column = n0 + j;
    if (row >= args.m || column >= args.n) continue;
    const long long offset =
        batch * args.stride_c + static_cast<long long>(row) * args.ldc + column;
    float value = args.alpha * scratch[index];
    if (args.beta != 0.0f) value += args.beta * static_cast<float>(c[offset]);
    c[offset] = static_cast<COut>(value);
  }
}

// Resolves the call, then launches one grid of that schedule.
template <typename PlanT, typename COut>
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
      GemmTileKernel<PlanT, COut>,
      dim3(grid_n, grid_m, static_cast<unsigned int>(gemm.batch_count)),
      dim3(static_cast<unsigned int>(PlanT::kThreads)), 0, a, b, c, gemm);
  return true;
}

// The sweep: one thunk per schedule, in the order of `kScheduleNames`.
using ScheduleThunk = bool (*)(const GemmParams&, const ComputeType*,
                               const ComputeType*, void*, bool);

template <typename PlanT>
bool Thunk(const GemmParams& params, const ComputeType* a, const ComputeType* b,
           void* c, bool fp32_c) {
  return fp32_c ? LaunchPlan<PlanT>(params, a, b, static_cast<float*>(c))
                : LaunchPlan<PlanT>(params, a, b, static_cast<ComputeType*>(c));
}

const ScheduleThunk kThunks[gemm_tile::kScheduleCount] = {
    &Thunk<gemm_tile::PlanBase>,
    &Thunk<gemm_tile::PlanSmall>,
    &Thunk<gemm_tile::PlanDeep>,
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

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)
