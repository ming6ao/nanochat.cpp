// The hand-written tensor-core GEMM of docs/kernels.md Part 3
// (docs/tensor-core-gemm-plan.md Part B).
//
// One kernel over the universal call C = alpha * op(A) * op(B) + beta * C. The
// transpose flags reach the device as two stride pairs, so the four operand
// forms are one body. The accumulators pass through an fp32 scratch tile in
// shared memory, because `wmma` stores only an fp32 accumulator and C may be
// bf16; that one store path carries the bounds, the alpha and beta terms, and
// the fp32 weight-gradient form.
//
// See backends/cuda/gemm/gemm_tile.cuh for the geometry and gemm_launch.h for
// the entry points.

#if defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#include "backends/cuda/gemm/gemm_launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/gemm_tile.cuh"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

namespace {

namespace wmma = nvcuda::wmma;

using gemm_tile::GemmTileArgs;
using gemm_tile::TileElement;

// C is `COut` and the accumulator is fp32, so one epilogue serves the bf16
// store of `Gemm` and the fp32 store of `GemmWgrad`.
template <typename COut>
__global__ void GemmTileKernel(const ComputeType* a, const ComputeType* b,
                               COut* c, GemmTileArgs args) {
  // `alignas` leads: the emulator maps `__shared__` to `static`, and a standard
  // attribute may not follow a storage class.
  alignas(16) __shared__ TileElement smem[gemm_tile::kSmemElements];
  TileElement* stage_a = smem;
  TileElement* stage_b = smem + gemm_tile::kStageAElements;

  const int n0 = static_cast<int>(blockIdx.x) * gemm_tile::kTileN;
  const int m0 = static_cast<int>(blockIdx.y) * gemm_tile::kTileM;
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
  const int warp_m = (warp / gemm_tile::kWarpsN) * gemm_tile::kWarpM;
  const int warp_n = (warp % gemm_tile::kWarpsN) * gemm_tile::kWarpN;

  wmma::fragment<wmma::accumulator, gemm_tile::kMicro, gemm_tile::kMicro,
                 gemm_tile::kMicro, float>
      acc[gemm_tile::kFragsM][gemm_tile::kFragsN];
  for (int i = 0; i < gemm_tile::kFragsM; ++i) {
    for (int j = 0; j < gemm_tile::kFragsN; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  for (int k0 = 0; k0 < args.k; k0 += gemm_tile::kTileK) {
    gemm_tile::StageASlice(a_base, a_stride_i, a_stride_k, m0, args.m, k0,
                           args.k, stage_a);
    gemm_tile::StageBSlice(b_base, b_stride_k, b_stride_j, n0, args.n, k0,
                           args.k, stage_b);
    __syncthreads();

    for (int kk = 0; kk < gemm_tile::kTileK; kk += gemm_tile::kMicro) {
      wmma::fragment<wmma::matrix_a, gemm_tile::kMicro, gemm_tile::kMicro,
                     gemm_tile::kMicro, TileElement, wmma::row_major>
          fa[gemm_tile::kFragsM];
      wmma::fragment<wmma::matrix_b, gemm_tile::kMicro, gemm_tile::kMicro,
                     gemm_tile::kMicro, TileElement, wmma::row_major>
          fb[gemm_tile::kFragsN];
      for (int i = 0; i < gemm_tile::kFragsM; ++i) {
        wmma::load_matrix_sync(
            fa[i],
            stage_a + (warp_m + i * gemm_tile::kMicro) * gemm_tile::kLeadA + kk,
            gemm_tile::kLeadA);
      }
      for (int j = 0; j < gemm_tile::kFragsN; ++j) {
        wmma::load_matrix_sync(
            fb[j],
            stage_b + kk * gemm_tile::kLeadB + (warp_n + j * gemm_tile::kMicro),
            gemm_tile::kLeadB);
      }
      for (int i = 0; i < gemm_tile::kFragsM; ++i) {
        for (int j = 0; j < gemm_tile::kFragsN; ++j) {
          wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // The epilogue: the accumulators become an fp32 tile in shared memory, and
  // one predicated pass applies alpha, beta, the bounds, and the store type.
  float* scratch = reinterpret_cast<float*>(smem);
  for (int i = 0; i < gemm_tile::kFragsM; ++i) {
    for (int j = 0; j < gemm_tile::kFragsN; ++j) {
      wmma::store_matrix_sync(
          scratch + (warp_m + i * gemm_tile::kMicro) * gemm_tile::kTileN +
              (warp_n + j * gemm_tile::kMicro),
          acc[i][j], gemm_tile::kTileN, wmma::mem_row_major);
    }
  }
  __syncthreads();

  const int stride = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x);
       index < gemm_tile::kTileM * gemm_tile::kTileN; index += stride) {
    const int i = index / gemm_tile::kTileN;
    const int j = index - i * gemm_tile::kTileN;
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

// Settles the seam defaults the way `GemmImpl` does, then launches one grid.
template <typename COut>
bool LaunchTile(const GemmParams& params, const ComputeType* a,
                const ComputeType* b, COut* c) {
  if (params.batch_count <= 0 || params.m <= 0 || params.n <= 0 ||
      params.k <= 0) {
    return false;
  }

  GemmTileArgs args;
  args.m = params.m;
  args.n = params.n;
  args.k = params.k;
  args.transpose_a = params.transpose_a;
  args.transpose_b = params.transpose_b;
  args.lda =
      params.lda > 0 ? params.lda : (params.transpose_a ? params.m : params.k);
  args.ldb =
      params.ldb > 0 ? params.ldb : (params.transpose_b ? params.k : params.n);
  args.ldc = params.ldc > 0 ? params.ldc : params.n;
  const int rows_a = params.transpose_a ? params.k : params.m;
  const int rows_b = params.transpose_b ? params.n : params.k;
  args.stride_a = params.stride_a != 0
                      ? params.stride_a
                      : static_cast<long long>(rows_a) * args.lda;
  args.stride_b = params.stride_b != 0
                      ? params.stride_b
                      : static_cast<long long>(rows_b) * args.ldb;
  args.stride_c = params.stride_c != 0
                      ? params.stride_c
                      : static_cast<long long>(params.m) * args.ldc;
  args.alpha = params.alpha;
  args.beta = params.beta;

  const unsigned int grid_n = static_cast<unsigned int>(
      (args.n + gemm_tile::kTileN - 1) / gemm_tile::kTileN);
  const unsigned int grid_m = static_cast<unsigned int>(
      (args.m + gemm_tile::kTileM - 1) / gemm_tile::kTileM);
  cuda_backend::Launch(
      GemmTileKernel<COut>,
      dim3(grid_n, grid_m, static_cast<unsigned int>(params.batch_count)),
      dim3(static_cast<unsigned int>(gemm_tile::kThreads)), 0, a, b, c, args);
  return true;
}

}  // namespace

bool LaunchGemmTile(const GemmParams& params, const ComputeType* a,
                    const ComputeType* b, ComputeType* c) {
  return LaunchTile(params, a, b, c);
}

bool LaunchGemmTileToFp32(const GemmParams& params, const ComputeType* a,
                          const ComputeType* b, float* c) {
  return LaunchTile(params, a, b, c);
}

}  // namespace kernels
}  // namespace nanochat

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)
