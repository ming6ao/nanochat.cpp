#ifndef NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_CUH_
#define NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_CUH_

// The device side of the Phase 1 prototype: one batched fp32 GEMM per launch,
// 128 by 64 block tile, 8 by 4 micro-tile per thread, two double-buffered
// 8-deep k slices. Only a device translation unit may include this header; the
// host side (the op resolver and the launcher) is in muon_fused_gemm.h and
// muon_fused_gemm.cu, so the plain-`cc` test binary never sees CUDA types.
//
// Layout of one tile in shared memory, chosen so the inner product is
// broadcast-friendly:
//
//   s_a[slice][k][m]   the op(A) tile, m contiguous (128 floats per k)
//   s_b[slice][k][n]   the op(B) tile, n contiguous (64 floats per k)
//
// A thread owns rows `row0 .. row0 + 7` and columns `col0 .. col0 + 3`, with
// `row0 = (tid / 16) * 8` and `col0 = (tid % 16) * 4`, so the 256 threads cover
// the 128 by 64 tile exactly and each of the 32 accumulators is a register.
//
// See dev/kernels/muon_fused_gemm.h and dev/kernels/README.md.

#include <cstdint>

#include <cuda_runtime.h>

#include "dev/kernels/muon_fused_gemm.h"

namespace nanochat {
namespace dev {

namespace muon_gemm {

constexpr int kBm = kMuonGemmBlockM;
constexpr int kBn = kMuonGemmBlockN;
constexpr int kBk = kMuonGemmSliceK;
constexpr int kThreads = kMuonGemmBlockThreads;
constexpr int kMicroM = kMuonGemmMicroM;
constexpr int kMicroN = kMuonGemmMicroN;
// 64 columns of 4 per thread, and 16 thread rows of 8 rows each.
constexpr int kThreadCols = kBn / kMicroN;
constexpr int kThreadRows = kThreads / kThreadCols;
// The float4 lanes one 8-deep slice splits into along the contiguous extent.
constexpr int kQuadsPerSlice = kBk / 4;
// The threads that own a float4 of the B tile: 8 k rows of 16 quads.
constexpr int kBThreads = kBk * (kBn / 4);

static_assert(kThreadRows * kMicroM == kBm,
              "one block tile must be covered by the thread rows");
static_assert(kThreadRows * kThreadCols == kThreads,
              "one block must be covered by the thread grid");
static_assert(kMuonGemmSmemBytes ==
                  2 * (kBm + kBn) * kBk * static_cast<int>(sizeof(float)),
              "the two double-buffered slices are the whole shared budget");

// One float4 of the operand at `p` (a 16-byte aligned read).
__device__ __forceinline__ float4 LoadQuad(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ float4 MakeQuad(float x, float y, float z, float w) {
  float4 v;
  v.x = x;
  v.y = y;
  v.z = z;
  v.w = w;
  return v;
}

// The loaders come in pairs. `Gather` reads one thread's four elements of one
// k slice into registers and `Scatter` writes them into the shared tile, so a
// slice's two global reads are both in flight before either store waits on
// one: one memory round trip per slice instead of two.
//
// Each operand has two loaders, and they differ only in which extent of the
// stored matrix is contiguous: `transpose_a` says the stored `A` is `k` by `m`
// (so `m` is contiguous) while a plain `A` is `m` by `k` (so `k` is), and the
// same holds for `B`.

// --- op(A), transposed: stored k by m, so m is contiguous ----------------
//
// Thread `tid` owns four columns of one k row: `kk = tid / 32` of the slice and
// columns `m0 + quad * 4 .. + 3`.
__device__ __forceinline__ float4 GatherATransposed(const float* base, int ld,
                                                    int m0, int k0, int m_ext,
                                                    int k_ext, int tid) {
  const int quad = tid % (kBm / 4);
  const int kk = tid / (kBm / 4);
  const int gm = m0 + quad * 4;
  const int gk = k0 + kk;
  if ((gm + 4 <= m_ext) && (gk < k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gk) * ld + gm);
  }
  const float* row = base + static_cast<std::int64_t>(gk) * ld;
  const bool in_row = gk < k_ext;
  return MakeQuad(in_row && gm + 0 < m_ext ? row[gm + 0] : 0.0f,
                  in_row && gm + 1 < m_ext ? row[gm + 1] : 0.0f,
                  in_row && gm + 2 < m_ext ? row[gm + 2] : 0.0f,
                  in_row && gm + 3 < m_ext ? row[gm + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterATransposed(const float4& v, int tid,
                                                   float (*dst)[kBm]) {
  *reinterpret_cast<float4*>(&dst[tid / (kBm / 4)][(tid % (kBm / 4)) * 4]) = v;
}

// --- op(A), plain: stored m by k, so k is contiguous ---------------------
//
// Thread `tid` owns four k of one row: row `m0 + tid / 2` and k slices
// `k0 + (tid % 2) * 4 .. + 3`. The four values land in four k slices of the
// shared tile, which is the transpose the inner product wants.
__device__ __forceinline__ float4 GatherAPlain(const float* base, int ld,
                                               int m0, int k0, int m_ext,
                                               int k_ext, int tid) {
  const int mm = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  const int gm = m0 + mm;
  const int gk = k0 + kq;
  if ((gm < m_ext) && (gk + 4 <= k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gm) * ld + gk);
  }
  const float* row = base + static_cast<std::int64_t>(gm) * ld;
  const bool in_row = gm < m_ext;
  return MakeQuad(in_row && gk + 0 < k_ext ? row[gk + 0] : 0.0f,
                  in_row && gk + 1 < k_ext ? row[gk + 1] : 0.0f,
                  in_row && gk + 2 < k_ext ? row[gk + 2] : 0.0f,
                  in_row && gk + 3 < k_ext ? row[gk + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterAPlain(const float4& v, int tid,
                                              float (*dst)[kBm]) {
  const int mm = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  dst[kq + 0][mm] = v.x;
  dst[kq + 1][mm] = v.y;
  dst[kq + 2][mm] = v.z;
  dst[kq + 3][mm] = v.w;
}

// --- op(B), plain: stored k by n, so n is contiguous ---------------------
//
// Only `kBThreads` threads own B tile elements: thread `tid < 128` owns four
// columns of one k row, `kk = tid / 16` and columns
// `n0 + (tid % 16) * 4 .. + 3`.
__device__ __forceinline__ float4 GatherBPlain(const float* base, int ld,
                                               int n0, int k0, int n_ext,
                                               int k_ext, int tid) {
  const int kk = tid / (kBn / 4);
  const int quad = (tid % (kBn / 4)) * 4;
  const int gn = n0 + quad;
  const int gk = k0 + kk;
  if ((gn + 4 <= n_ext) && (gk < k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gk) * ld + gn);
  }
  const float* row = base + static_cast<std::int64_t>(gk) * ld;
  const bool in_row = gk < k_ext;
  return MakeQuad(in_row && gn + 0 < n_ext ? row[gn + 0] : 0.0f,
                  in_row && gn + 1 < n_ext ? row[gn + 1] : 0.0f,
                  in_row && gn + 2 < n_ext ? row[gn + 2] : 0.0f,
                  in_row && gn + 3 < n_ext ? row[gn + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterBPlain(const float4& v, int tid,
                                              float (*dst)[kBn]) {
  *reinterpret_cast<float4*>(&dst[tid / (kBn / 4)][(tid % (kBn / 4)) * 4]) = v;
}

// --- op(B), transposed: stored n by k, so k is contiguous ----------------
//
// Thread `tid < 128` owns four k of one column: column `n0 + tid / 2` and k
// slices `k0 + (tid % 2) * 4 .. + 3`.
__device__ __forceinline__ float4 GatherBTransposed(const float* base, int ld,
                                                    int n0, int k0, int n_ext,
                                                    int k_ext, int tid) {
  const int nn = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  const int gn = n0 + nn;
  const int gk = k0 + kq;
  if ((gn < n_ext) && (gk + 4 <= k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gn) * ld + gk);
  }
  const float* row = base + static_cast<std::int64_t>(gn) * ld;
  const bool in_row = gn < n_ext;
  return MakeQuad(in_row && gk + 0 < k_ext ? row[gk + 0] : 0.0f,
                  in_row && gk + 1 < k_ext ? row[gk + 1] : 0.0f,
                  in_row && gk + 2 < k_ext ? row[gk + 2] : 0.0f,
                  in_row && gk + 3 < k_ext ? row[gk + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterBTransposed(const float4& v, int tid,
                                                   float (*dst)[kBn]) {
  const int nn = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  dst[kq + 0][nn] = v.x;
  dst[kq + 1][nn] = v.y;
  dst[kq + 2][nn] = v.z;
  dst[kq + 3][nn] = v.w;
}

// One k slice of both operands of one block: both global reads are issued
// before either store consumes one.
__device__ __forceinline__ void LoadSlice(const float* ap, const float* bp,
                                          int lda, int ldb, bool transpose_a,
                                          bool transpose_b, int m0, int n0,
                                          int m, int n, int k, int k0, int tid,
                                          float (*sa)[kBm], float (*sb)[kBn]) {
  const float4 a_regs = transpose_a
                            ? GatherATransposed(ap, lda, m0, k0, m, k, tid)
                            : GatherAPlain(ap, lda, m0, k0, m, k, tid);
  float4 b_regs = MakeQuad(0.0f, 0.0f, 0.0f, 0.0f);
  if (tid < kBThreads) {
    b_regs = transpose_b ? GatherBTransposed(bp, ldb, n0, k0, n, k, tid)
                         : GatherBPlain(bp, ldb, n0, k0, n, k, tid);
  }
  if (transpose_a) {
    ScatterATransposed(a_regs, tid, sa);
  } else {
    ScatterAPlain(a_regs, tid, sa);
  }
  if (tid < kBThreads) {
    if (transpose_b) {
      ScatterBTransposed(b_regs, tid, sb);
    } else {
      ScatterBPlain(b_regs, tid, sb);
    }
  }
}

// One batched GEMM. Block `f` of the flat grid is one `(matrix, tile_i,
// tile_j)`: `p = f / (tiles_m * tiles_n)` names the matrix and the remainder
// names the tile, so the launch is a single integer grid for any batch.
__global__ void __launch_bounds__(kMuonGemmBlockThreads, kMuonGemmBlocksPerSm)
    MuonFusedGemmKernel(int m, int n, int k, const float* __restrict__ a,
                        const float* __restrict__ b, float* __restrict__ c,
                        int lda, int ldb, int ldc, bool transpose_a,
                        bool transpose_b, std::int64_t stride_a,
                        std::int64_t stride_b, std::int64_t stride_c) {
  const int tiles_m = (m + kBm - 1) / kBm;
  const int tiles_n = (n + kBn - 1) / kBn;
  const int per_matrix = tiles_m * tiles_n;
  const int flat = static_cast<int>(blockIdx.x);
  const int p = flat / per_matrix;
  const int within = flat - p * per_matrix;
  const int tile_j = within % tiles_n;
  const int tile_i = within / tiles_n;
  const int m0 = tile_i * kBm;
  const int n0 = tile_j * kBn;

  const float* ap = a + static_cast<std::int64_t>(p) * stride_a;
  const float* bp = b + static_cast<std::int64_t>(p) * stride_b;
  float* cp = c + static_cast<std::int64_t>(p) * stride_c;

  const int tid = static_cast<int>(threadIdx.x);
  const int row0 = (tid / kThreadCols) * kMicroM;
  const int col0 = (tid % kThreadCols) * kMicroN;

  __shared__ float s_a[2][kBk][kBm];
  __shared__ float s_b[2][kBk][kBn];

  float acc[kMicroM][kMicroN];
#pragma unroll
  for (int i = 0; i < kMicroM; ++i) {
#pragma unroll
    for (int j = 0; j < kMicroN; ++j) acc[i][j] = 0.0f;
  }

  int slice = 0;
  LoadSlice(ap, bp, lda, ldb, transpose_a, transpose_b, m0, n0, m, n, k, 0, tid,
            s_a[slice], s_b[slice]);
  __syncthreads();

  for (int k0 = 0; k0 < k; k0 += kBk) {
    if (k0 + kBk < k) {
      LoadSlice(ap, bp, lda, ldb, transpose_a, transpose_b, m0, n0, m, n, k,
                k0 + kBk, tid, s_a[slice ^ 1], s_b[slice ^ 1]);
    }
    // Hoisted row and column bases: the per-k address is then a constant
    // offset from one register, so the slice's loads cost no extra address
    // arithmetic.
    const float* a_row = &s_a[slice][0][row0];
    const float* b_col = &s_b[slice][0][col0];
// Two k at a time: 64 independent FMAs is enough to cover the FMA latency,
// and unrolling the whole slice pushes the accumulators past the register
// budget `__launch_bounds__(256, 3)` allows and spills.
#pragma unroll 2
    for (int kk = 0; kk < kBk; ++kk) {
      // The 32 accumulators of the 8 by 4 micro-tile: the eight rows of this
      // thread against the four columns of this thread, one statement each, so
      // every one of them stays a register.
      const float4 a0 = LoadQuad(a_row + kk * kBm);
      const float4 a1 = LoadQuad(a_row + kk * kBm + 4);
      const float4 bv = LoadQuad(b_col + kk * kBn);
      acc[0][0] = fmaf(a0.x, bv.x, acc[0][0]);
      acc[0][1] = fmaf(a0.x, bv.y, acc[0][1]);
      acc[0][2] = fmaf(a0.x, bv.z, acc[0][2]);
      acc[0][3] = fmaf(a0.x, bv.w, acc[0][3]);
      acc[1][0] = fmaf(a0.y, bv.x, acc[1][0]);
      acc[1][1] = fmaf(a0.y, bv.y, acc[1][1]);
      acc[1][2] = fmaf(a0.y, bv.z, acc[1][2]);
      acc[1][3] = fmaf(a0.y, bv.w, acc[1][3]);
      acc[2][0] = fmaf(a0.z, bv.x, acc[2][0]);
      acc[2][1] = fmaf(a0.z, bv.y, acc[2][1]);
      acc[2][2] = fmaf(a0.z, bv.z, acc[2][2]);
      acc[2][3] = fmaf(a0.z, bv.w, acc[2][3]);
      acc[3][0] = fmaf(a0.w, bv.x, acc[3][0]);
      acc[3][1] = fmaf(a0.w, bv.y, acc[3][1]);
      acc[3][2] = fmaf(a0.w, bv.z, acc[3][2]);
      acc[3][3] = fmaf(a0.w, bv.w, acc[3][3]);
      acc[4][0] = fmaf(a1.x, bv.x, acc[4][0]);
      acc[4][1] = fmaf(a1.x, bv.y, acc[4][1]);
      acc[4][2] = fmaf(a1.x, bv.z, acc[4][2]);
      acc[4][3] = fmaf(a1.x, bv.w, acc[4][3]);
      acc[5][0] = fmaf(a1.y, bv.x, acc[5][0]);
      acc[5][1] = fmaf(a1.y, bv.y, acc[5][1]);
      acc[5][2] = fmaf(a1.y, bv.z, acc[5][2]);
      acc[5][3] = fmaf(a1.y, bv.w, acc[5][3]);
      acc[6][0] = fmaf(a1.z, bv.x, acc[6][0]);
      acc[6][1] = fmaf(a1.z, bv.y, acc[6][1]);
      acc[6][2] = fmaf(a1.z, bv.z, acc[6][2]);
      acc[6][3] = fmaf(a1.z, bv.w, acc[6][3]);
      acc[7][0] = fmaf(a1.w, bv.x, acc[7][0]);
      acc[7][1] = fmaf(a1.w, bv.y, acc[7][1]);
      acc[7][2] = fmaf(a1.w, bv.z, acc[7][2]);
      acc[7][3] = fmaf(a1.w, bv.w, acc[7][3]);
    }
    __syncthreads();
    slice ^= 1;
  }

#pragma unroll
  for (int i = 0; i < kMicroM; ++i) {
    const int gm = m0 + row0 + i;
    if (gm >= m) break;
    float* row = cp + static_cast<std::int64_t>(gm) * ldc + n0 + col0;
#pragma unroll
    for (int j = 0; j < kMicroN; ++j) {
      if (n0 + col0 + j < n) row[j] = acc[i][j];
    }
  }
}

}  // namespace muon_gemm
}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_CUH_
