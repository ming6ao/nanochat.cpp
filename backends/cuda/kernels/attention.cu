// CUDA Attention family: causal / sliding-window / grouped-query attention,
// forward and backward, plus the saved softmax statistics
// `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// The query-key scores, the probability-value product, and every backward
// product are batched GEMMs through the cuBLAS seam; only the softmax and its
// gradient run as custom row kernels. This is the same decomposition the
// PyTorch math backend uses.
//
// The model's q/k/v layout is `[batch, seq, heads, head_dim]`, whose per-head
// stride is not constant across `(b, h)`, so a single strided-batched call
// cannot read it. For multi-head attention (the training and decode path) the
// operands are transposed into a contiguous `[batch, heads, seq, head_dim]`
// scratch, which lets one batched GEMM cover every head. That matters on
// Pascal: a per-head GEMM of `512 x 128` occupies a fraction of the device and
// runs at about 1.4-3 TFLOP/s, while the same work batched over the heads runs
// at about 6-7 TFLOP/s. Grouped-query attention keeps the per-head path,
// because several query heads share one key/value head and their outputs would
// collide in a single batched call.
//
// The softmax kernels mirror `backends/cpu/kernels.cc` and
// `backends/cuda/kernels/testing/sequence_ref.h` exactly: the row maximum over
// the *visible* keys, the causal plus sliding-window mask, the `(0, 0)`
// statistics for an empty window, and the `p * (dp - sum(p * dp))` softmax
// Jacobian.
//
// A windowed forward at `head_dim` 128 takes a fused tiled kernel instead of
// the cuBLAS decomposition. The kernel keeps the online-softmax state and the
// output accumulator in registers and skips fully masked key tiles, which is
// what turns the sliding window into saved work. The cuBLAS forward stays for
// full-causal attention and for every other shape. The cuBLAS backward stays
// for every shape. See `docs/attention.md` Part 2, section 5.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceMax;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::KeyAllowedDev;
using cuda_kernels::KvHeadCountDev;
using cuda_kernels::KvHeadDev;
using cuda_kernels::ToComputeDev;

constexpr int kSoftmaxThreads = 128;
constexpr int kTransposeThreads = 256;

// Base of head `h` in a `[batch, rows, heads, head_dim]` buffer.
inline std::int64_t HeadBase(int b, int rows, int heads, int h, int head_dim) {
  return (static_cast<std::int64_t>(b) * rows * heads + h) * head_dim;
}

// Base of head `h` in the `[batch, heads, seq, kv_len]` score scratch.
__host__ __device__ inline std::int64_t ScoreBase(const AttentionParams& p,
                                                  int b, int h) {
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  return (static_cast<std::int64_t>(b) * p.num_heads + h) * p.seq * kv_len;
}

// One batched `C = alpha * op(A) * op(B) + beta * C` over contiguous
// `[batch, rows, cols]` head buffers. The seam infers the leading dimensions
// and per-batch strides from the logical shape, so only the shape and the
// transpose flags are needed. `GemmMode` is descriptive only (the seam reduces
// every mode to this product), so all callers pass kForward.
void BatchedGemm(int m, int n, int k, int batch_count, float alpha, float beta,
                 bool transpose_a, bool transpose_b, const ComputeType* a,
                 const ComputeType* b, ComputeType* c) {
  GemmParams gp;
  gp.m = m;
  gp.n = n;
  gp.k = k;
  gp.batch_count = batch_count;
  gp.alpha = alpha;
  gp.beta = beta;
  gp.transpose_a = transpose_a;
  gp.transpose_b = transpose_b;
  Gemm(GemmMode::kForward, gp, a, b, c);
}

// One `C = alpha * op(A) * op(B) + beta * C` with explicit row strides, for the
// grouped-query fallback whose operands are rows of the model's
// `[batch, seq, heads, dim]` buffers (row stride `heads * dim`, not `dim`).
void StridedGemm(int m, int n, int k, float alpha, float beta, bool transpose_a,
                 bool transpose_b, int lda, int ldb, int ldc,
                 const ComputeType* a, const ComputeType* b, ComputeType* c) {
  GemmParams gp;
  gp.m = m;
  gp.n = n;
  gp.k = k;
  gp.alpha = alpha;
  gp.beta = beta;
  gp.transpose_a = transpose_a;
  gp.transpose_b = transpose_b;
  gp.lda = lda;
  gp.ldb = ldb;
  gp.ldc = ldc;
  Gemm(GemmMode::kForward, gp, a, b, c);
}

// Cached device scratch, keyed by slot: 0 is the `[batch, heads, seq, kv_len]`
// score/probability matrices, 1 is the transposed head buffers. Attention runs
// once per layer, so a cudaMalloc/free per call would dominate. The training
// graph is single-threaded, so no locking is needed.
ComputeType* CachedBuffer(int slot, std::size_t count) {
  static ComputeType* buffers[2] = {nullptr, nullptr};
  static std::size_t capacities[2] = {0, 0};
  if (count > capacities[slot]) {
    if (buffers[slot] != nullptr) Free(buffers[slot]);
    buffers[slot] =
        static_cast<ComputeType*>(Alloc(count * sizeof(ComputeType)));
    capacities[slot] = count;
  }
  return buffers[slot];
}

// --- Softmax kernels -------------------------------------------------------

// One block per `(b, h, t)` query row. Computes the visible-key maximum, the
// shifted exponentials, the sum, and the normalized probabilities, and writes
// the `(max, sum_exp)` statistics. Mirrors the reference: an empty window
// yields a zero row and zero statistics.
__global__ void AttentionSoftmaxForwardKernel(
    const AttentionParams params, const ComputeType* __restrict__ scores,
    ComputeType* __restrict__ probs, float* __restrict__ stats) {
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;
  const std::int64_t base =
      ScoreBase(params, b, h) + static_cast<std::int64_t>(t) * kv_len;
  const ComputeType* srow = scores + base;
  ComputeType* prow = probs + base;
  float* stat =
      stats +
      (static_cast<std::int64_t>(b) * params.num_heads + h) * params.seq * 2 +
      t * 2;
  const int tid = threadIdx.x;

  float local_max = -CUDART_INF_F;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    if (!KeyAllowedDev(params, qpos, j)) continue;
    local_max = fmaxf(local_max, AsFloatDev(srow[j]));
  }
  const float row_max = BlockReduceMax(local_max);
  if (row_max == -CUDART_INF_F) {
    for (int j = tid; j < kv_len; j += blockDim.x) prow[j] = ToComputeDev(0.0f);
    if (tid == 0) {
      stat[0] = 0.0f;
      stat[1] = 0.0f;
    }
    return;
  }

  float local_sum = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    float weight = 0.0f;
    if (KeyAllowedDev(params, qpos, j)) {
      weight = expf(AsFloatDev(srow[j]) - row_max);
    }
    prow[j] = ToComputeDev(weight);
    local_sum += weight;
  }
  const float sum_exp = BlockReduceSum(local_sum);
  const float inv = 1.0f / sum_exp;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    prow[j] = ToComputeDev(AsFloatDev(prow[j]) * inv);
  }
  if (tid == 0) {
    stat[0] = row_max;
    stat[1] = sum_exp;
  }
}

// One block per `(b, h, t)` query row. Rebuilds the probabilities from the
// scores and the saved statistics, reduces `sum(p * dp)`, and turns `dP` into
// `dS = p * (dp - sum(p * dp))` in place. Masked positions become zero.
__global__ void AttentionSoftmaxBackwardKernel(
    const AttentionParams params, const ComputeType* __restrict__ scores,
    const float* __restrict__ stats, ComputeType* __restrict__ probs,
    ComputeType* __restrict__ dprobs) {
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;
  const std::int64_t base =
      ScoreBase(params, b, h) + static_cast<std::int64_t>(t) * kv_len;
  const ComputeType* srow = scores + base;
  ComputeType* prow = probs + base;
  ComputeType* drow = dprobs + base;
  const float* stat =
      stats +
      (static_cast<std::int64_t>(b) * params.num_heads + h) * params.seq * 2 +
      t * 2;
  const int tid = threadIdx.x;

  const float row_max = stat[0];
  const float sum_exp = stat[1];
  if (sum_exp <= 0.0f) {
    for (int j = tid; j < kv_len; j += blockDim.x) {
      prow[j] = ToComputeDev(0.0f);
      drow[j] = ToComputeDev(0.0f);
    }
    return;
  }
  const float inv = 1.0f / sum_exp;

  float local_wdp = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    float pj = 0.0f;
    if (KeyAllowedDev(params, qpos, j)) {
      pj = expf(AsFloatDev(srow[j]) - row_max) * inv;
    }
    prow[j] = ToComputeDev(pj);
    local_wdp += pj * AsFloatDev(drow[j]);
  }
  const float weighted_dp = BlockReduceSum(local_wdp);
  for (int j = tid; j < kv_len; j += blockDim.x) {
    const float ds =
        KeyAllowedDev(params, qpos, j)
            ? AsFloatDev(prow[j]) * (AsFloatDev(drow[j]) - weighted_dp)
            : 0.0f;
    drow[j] = ToComputeDev(ds);
  }
}

// --- Head-layout transposes ------------------------------------------------

// `[batch, rows, heads, dim]` -> `[batch, heads, rows, dim]`.
__global__ void TransposeToHeadsKernel(long long total, int rows, int heads,
                                       int dim,
                                       const ComputeType* __restrict__ in,
                                       ComputeType* __restrict__ out) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = i % dim;
  const int h = (i / dim) % heads;
  const int r = (i / (dim * heads)) % rows;
  const int b =
      static_cast<int>(i / (static_cast<long long>(dim) * heads * rows));
  out[((static_cast<long long>(b) * heads + h) * rows + r) * dim + d] = in[i];
}

// `[batch, heads, rows, dim]` -> `[batch, rows, heads, dim]`.
__global__ void TransposeFromHeadsKernel(long long total, int rows, int heads,
                                         int dim,
                                         const ComputeType* __restrict__ in,
                                         ComputeType* __restrict__ out) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = i % dim;
  const int r = (i / dim) % rows;
  const int h = (i / (dim * rows)) % heads;
  const int b =
      static_cast<int>(i / (static_cast<long long>(dim) * rows * heads));
  out[((static_cast<long long>(b) * rows + r) * heads + h) * dim + d] = in[i];
}

void TransposeToHeads(int batch, int rows, int heads, int dim,
                      const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * rows * heads * dim;
  if (total <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeToHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, dim, in,
                       out);
}

void TransposeFromHeads(int batch, int rows, int heads, int dim,
                        const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * rows * heads * dim;
  if (total <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeFromHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, dim, in,
                       out);
}

// --- Per-head fallback (grouped-query attention) ---------------------------

// scores = Q @ K^T * scale, for one (b, h).
void GemmScores(const AttentionParams& p, int b, int h, float scale,
                const ComputeType* q, const ComputeType* k,
                ComputeType* scores) {
  const int dim = p.head_dim;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int kv_heads = KvHeadCountDev(p);
  const int kvh = KvHeadDev(h, p.num_heads, p.num_kv_heads);
  StridedGemm(p.seq, kv_len, dim, scale, 0.0f, /*transpose_a=*/false,
              /*transpose_b=*/true, p.num_heads * dim, kv_heads * dim, kv_len,
              q + HeadBase(b, p.seq, p.num_heads, h, dim),
              k + HeadBase(b, kv_len, kv_heads, kvh, dim),
              scores + ScoreBase(p, b, h));
}

// out = P @ V, for one (b, h).
void GemmOut(const AttentionParams& p, int b, int h, const ComputeType* probs,
             const ComputeType* v, ComputeType* out) {
  const int dim = p.head_dim;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int kv_heads = KvHeadCountDev(p);
  const int kvh = KvHeadDev(h, p.num_heads, p.num_kv_heads);
  StridedGemm(p.seq, dim, kv_len, 1.0f, 0.0f, /*transpose_a=*/false,
              /*transpose_b=*/false, kv_len, kv_heads * dim, p.num_heads * dim,
              probs + ScoreBase(p, b, h),
              v + HeadBase(b, kv_len, kv_heads, kvh, dim),
              out + HeadBase(b, p.seq, p.num_heads, h, dim));
}

// Whether the batched path applies. Multi-head only: grouped-query attention
// shares key/value heads across query heads, so a batched call would write the
// same key/value gradient from several batches.
bool UseBatchedPath(const AttentionParams& p) {
  return p.num_kv_heads <= 0 || p.num_kv_heads == p.num_heads;
}

// --- Fused tiled forward (windowed shapes) ---------------------------------

// The validated Phase 1 tile: `Br=32` query rows, `Bc=32` key/value rows staged
// per tile, 256 threads, two resident blocks per streaming multiprocessor. At
// `head_dim` 128 the key/value stage and the probability tile use 37,120 bytes,
// below the 48 KB Pascal block cap. See `docs/attention.md` Part 2.
// section 6 and `backends/cuda/kernels/README.md`.
constexpr int kFusedThreads = 256;
constexpr int kFusedBr = 32;
constexpr int kFusedBc = 32;
constexpr int kFusedMinBlocks = 2;

// Owned output dimensions per lane. Together with the 32 warp lanes this caps
// `head_dim` at 128, the only shape the fused forward covers.
constexpr int kFusedDimsPerLane = 4;

// Warp-wide maximum over the 32 lanes. Every lane in the warp must reach the
// call; the result is broadcast back to every lane.
__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Warp-wide sum over the 32 lanes, mirroring WarpMax.
__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// One thread block owns `kFusedBr` query rows for one `(batch, head)` pair. The
// warp `w` owns rows `t0 + w * kRowsPerWarp ..`; within a warp, lane `l` scores
// the key `j0 + sub*32 + l` for each 32-key sub-tile `sub` and owns output
// dimensions `l, l + 32, ...`.
//
// The running row maximum, the running denominator, and the output accumulator
// live in registers, so a fully masked key tile is skipped without touching
// them. The exponentials use `expf` to match the CPU reference.
__global__ void __launch_bounds__(kFusedThreads, kFusedMinBlocks)
    FusedAttentionForwardKernel(const AttentionParams params,
                                const ComputeType* __restrict__ q,
                                const ComputeType* __restrict__ k,
                                const ComputeType* __restrict__ v,
                                ComputeType* __restrict__ out,
                                float* __restrict__ stats) {
  constexpr int kWarps = kFusedThreads / 32;       // warps per block
  constexpr int kRowsPerWarp = kFusedBr / kWarps;  // query rows per warp
  constexpr int kSubTiles = kFusedBc / 32;         // 32-key sub-tiles per tile

  const int dim = params.head_dim;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const int kv_heads = KvHeadCountDev(params);
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));

  const int t0 = blockIdx.x * kFusedBr;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int kvh = KvHeadDev(h, heads, params.num_kv_heads);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // Shared memory: the key tile, the value tile, and the probability tile for
  // the current key tile. The key/value rows are padded by one element so the
  // strided per-lane reads hit distinct banks. The raw buffer is typed double
  // so it is 8-byte aligned, which keeps the float probability tile aligned in
  // the fp16 build too.
  extern __shared__ double shared_raw[];
  ComputeType* ks = reinterpret_cast<ComputeType*>(shared_raw);
  const int kv_stride = dim + 1;
  ComputeType* vs = ks + kFusedBc * kv_stride;
  float* ps = reinterpret_cast<float*>(vs + kFusedBc * kv_stride);

  // Per-lane online-softmax state for the rows this warp owns.
  float row_max[kRowsPerWarp];
  float row_sum[kRowsPerWarp];
  float o_acc[kRowsPerWarp][kFusedDimsPerLane];
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    row_max[i] = -CUDART_INF_F;
    row_sum[i] = 0.0f;
#pragma unroll
    for (int c = 0; c < kFusedDimsPerLane; ++c) o_acc[i][c] = 0.0f;
  }

  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kFusedBr - 1;
  const int num_key_tiles = (kv_len + kFusedBc - 1) / kFusedBc;

  for (int tile = 0; tile < num_key_tiles; ++tile) {
    const int j0 = tile * kFusedBc;

    // Skip a key tile that is entirely masked for every query row in the
    // block. This turns the sliding window into saved work rather than a mask
    // applied after a dense product. The conditions are conservative: they
    // only skip a tile that is provably invisible to all rows.
    if (params.causal && j0 > qpos_max) continue;
    if (params.window_left >= 0 &&
        j0 + kFusedBc - 1 < qpos_min - params.window_left) {
      continue;
    }
    if (params.window_right >= 0 && j0 > qpos_max + params.window_right) {
      continue;
    }

    // Stage the key and value tile. The global read follows the row-major
    // layout and coalesces; the padded shared write keeps the banks balanced.
    for (int idx = tid; idx < kFusedBc * dim; idx += kFusedThreads) {
      const int jr = idx / dim;
      const int d = idx - jr * dim;
      const int j = j0 + jr;
      float key_value = 0.0f;
      float value_value = 0.0f;
      if (j < kv_len) {
        const std::size_t offset =
            (static_cast<std::size_t>(b) * kv_len + j) *
                (static_cast<std::size_t>(kv_heads) * dim) +
            static_cast<std::size_t>(kvh) * dim + d;
        key_value = AsFloatDev(k[offset]);
        value_value = AsFloatDev(v[offset]);
      }
      ks[jr * kv_stride + d] = ToComputeDev(key_value);
      vs[jr * kv_stride + d] = ToComputeDev(value_value);
    }
    __syncthreads();

    // Process the tile one 32-key sub-tile at a time so the online-softmax
    // state and the register block stay fixed for every `kBc`.
#pragma unroll
    for (int sub = 0; sub < kSubTiles; ++sub) {
      const int jsub = j0 + sub * 32;

      // QK^T and the causal/window mask. Each lane scores its own key for the
      // rows the warp owns; the head-dimension dot product stays inside the
      // lane and needs no cross-lane reduction. The key element is loaded once
      // and reused across the rows.
      float s_reg[kRowsPerWarp];
      float tile_max[kRowsPerWarp];
      {
        const ComputeType* qrows[kRowsPerWarp];
        bool valid[kRowsPerWarp];
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
          const int r = warp * kRowsPerWarp + i;
          const int tq = t0 + r;
          valid[i] = tq < seq;
          // An invalid row points at the base so the dot stays in bounds; its
          // score is discarded below.
          qrows[i] =
              valid[i]
                  ? q + ((static_cast<std::size_t>(b) * seq + tq) * heads + h) *
                            dim
                  : q;
        }
        float dot[kRowsPerWarp];
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) dot[i] = 0.0f;
        const long long jabs = jsub + lane;
        if (jabs < kv_len) {
          const ComputeType* krow = ks + (sub * 32 + lane) * kv_stride;
#pragma unroll 4
          for (int d = 0; d < dim; ++d) {
            const float key_value = AsFloatDev(krow[d]);
#pragma unroll
            for (int i = 0; i < kRowsPerWarp; ++i) {
              dot[i] += AsFloatDev(qrows[i][d]) * key_value;
            }
          }
        }
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
          const int tq = t0 + warp * kRowsPerWarp + i;
          const long long qpos = static_cast<long long>(kv_len) - seq + tq;
          float score = -CUDART_INF_F;
          if (valid[i] && jabs < kv_len && KeyAllowedDev(params, qpos, jabs)) {
            score = scale * dot[i];
          }
          s_reg[i] = score;
          tile_max[i] = WarpMax(score);
        }
      }

      // Online-softmax update. `alpha` rescales the running denominator and
      // the output accumulator onto the new maximum. A row with no visible key
      // in this sub-tile keeps its state: alpha is one and the new sum is zero.
      float alpha[kRowsPerWarp];
#pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i) {
        const float previous = row_max[i];
        const float tile = tile_max[i];
        if (tile == -CUDART_INF_F) {
          alpha[i] = 1.0f;
        } else {
          const float updated = fmaxf(previous, tile);
          row_max[i] = updated;
          alpha[i] =
              (previous == -CUDART_INF_F) ? 0.0f : expf(previous - updated);
        }
      }

      // P = exp(S - row_max). The warp sum over the lanes is the sub-tile sum,
      // which the running denominator absorbs with the same rescale.
      float p_reg[kRowsPerWarp];
#pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i) {
        float p = 0.0f;
        if (tile_max[i] != -CUDART_INF_F) {
          p = expf(s_reg[i] - row_max[i]);
        }
        p_reg[i] = p;
        row_sum[i] = alpha[i] * row_sum[i] + WarpSum(p);
      }

      // Publish P and rescale the output accumulator.
#pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
        for (int c = 0; c < kFusedDimsPerLane; ++c) o_acc[i][c] *= alpha[i];
        ps[(warp * kRowsPerWarp + i) * kFusedBc + sub * 32 + lane] = p_reg[i];
      }
      __syncthreads();

      // O += P V for this sub-tile. Every lane reads every key through the
      // shared probability tile, so each owned output dimension is a private
      // accumulation and needs no cross-lane reduction. The value element is
      // loaded once and reused across the rows.
#pragma unroll 4
      for (int jr = sub * 32; jr < sub * 32 + 32; ++jr) {
        const ComputeType* vrow = vs + jr * kv_stride;
        float value_value[kFusedDimsPerLane];
#pragma unroll
        for (int c = 0; c < kFusedDimsPerLane; ++c) {
          const int d = lane + 32 * c;
          value_value[c] = (d < dim) ? AsFloatDev(vrow[d]) : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
          const float p = ps[(warp * kRowsPerWarp + i) * kFusedBc + jr];
#pragma unroll
          for (int c = 0; c < kFusedDimsPerLane; ++c) {
            o_acc[i][c] += p * value_value[c];
          }
        }
      }
      __syncthreads();
    }
  }

  // Normalize the online accumulator, then write the output and the saved
  // softmax statistics. A row with an empty window (zero denominator) writes a
  // zero row and the (0, 0) statistic pair, matching the reference contract.
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int r = warp * kRowsPerWarp + i;
    const int tq = t0 + r;
    if (tq >= seq) continue;
    const std::size_t out_base =
        ((static_cast<std::size_t>(b) * seq + tq) * heads + h) * dim;
    const std::size_t stat_base =
        ((static_cast<std::size_t>(b) * heads + h) * seq + tq) * 2;
    if (row_sum[i] <= 0.0f) {
#pragma unroll
      for (int c = 0; c < kFusedDimsPerLane; ++c) {
        const int d = lane + 32 * c;
        if (d < dim) out[out_base + d] = ToComputeDev(0.0f);
      }
      if (lane == 0) {
        stats[stat_base] = 0.0f;
        stats[stat_base + 1] = 0.0f;
      }
    } else {
      const float inverse = 1.0f / row_sum[i];
#pragma unroll
      for (int c = 0; c < kFusedDimsPerLane; ++c) {
        const int d = lane + 32 * c;
        if (d < dim) out[out_base + d] = ToComputeDev(o_acc[i][c] * inverse);
      }
      if (lane == 0) {
        stats[stat_base] = row_max[i];
        stats[stat_base + 1] = row_sum[i];
      }
    }
  }
}

// Launches the fused forward. The dispatch predicate already bounds the shape
// to `head_dim` 128, where the shared-memory tile fits.
void LaunchFusedAttention(const AttentionParams& params, const ComputeType* q,
                          const ComputeType* k, const ComputeType* v,
                          ComputeType* out, float* stats) {
  const std::size_t kv_stride = static_cast<std::size_t>(params.head_dim) + 1;
  const std::size_t shared_bytes =
      (2 * kFusedBc * kv_stride) * sizeof(ComputeType) +
      (kFusedBr * kFusedBc) * sizeof(float);
  const int query_tiles = (params.seq + kFusedBr - 1) / kFusedBr;
  const dim3 grid(query_tiles, params.num_heads, params.batch);
  const dim3 block(kFusedThreads);
  cuda_backend::Launch(FusedAttentionForwardKernel, grid, block, shared_bytes,
                       params, q, k, v, out, stats);
}

}  // namespace

// Whether the fused tiled forward covers this shape. Only the sliding-window
// path is promoted: the full-causal tile lost the Phase 1 gate to cuBLAS, and
// the tile register block covers only `head_dim` 128.
bool UseFusedAttention(const AttentionParams& p) {
  return p.window_left >= 0 && p.head_dim == 128;
}

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  if (UseFusedAttention(params)) {
    LaunchFusedAttention(params, q, k, v, out, stats);
    return;
  }
  const int heads = params.num_heads;
  const int dim = params.head_dim;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));
  const std::size_t matrix =
      static_cast<std::size_t>(params.batch) * heads * params.seq * kv_len;
  ComputeType* scores = CachedBuffer(0, 3 * matrix);
  ComputeType* probs = scores + matrix;
  const int rows = params.batch * heads * params.seq;

  if (UseBatchedPath(params)) {
    const std::size_t q_elems =
        static_cast<std::size_t>(params.batch) * heads * params.seq * dim;
    const std::size_t k_elems =
        static_cast<std::size_t>(params.batch) * heads * kv_len * dim;
    ComputeType* head = CachedBuffer(1, 2 * q_elems + 2 * k_elems);
    ComputeType* qT = head;
    ComputeType* kT = qT + q_elems;
    ComputeType* vT = kT + k_elems;
    ComputeType* outT = vT + k_elems;
    TransposeToHeads(params.batch, params.seq, heads, dim, q, qT);
    TransposeToHeads(params.batch, kv_len, heads, dim, k, kT);
    TransposeToHeads(params.batch, kv_len, heads, dim, v, vT);

    BatchedGemm(params.seq, kv_len, dim, params.batch * heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, qT, kT, scores);
    cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                         dim3(kSoftmaxThreads), 0, params, scores, probs,
                         stats);
    BatchedGemm(params.seq, dim, kv_len, params.batch * heads, 1.0f, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/false, probs, vT, outT);
    TransposeFromHeads(params.batch, params.seq, heads, dim, outT, out);
    return;
  }

  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      GemmScores(params, b, h, scale, q, k, scores);
    }
  }
  cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, probs, stats);
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      GemmOut(params, b, h, probs, v, out);
    }
  }
}

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int dim = params.head_dim;
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const int heads = params.num_heads;
  const int kv_heads =
      params.num_kv_heads > 0 ? params.num_kv_heads : params.num_heads;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));

  const std::size_t matrix =
      static_cast<std::size_t>(params.batch) * heads * params.seq * kv_len;
  ComputeType* scores = CachedBuffer(0, 3 * matrix);
  ComputeType* probs = scores + matrix;
  ComputeType* dprobs = probs + matrix;
  const int rows = params.batch * heads * params.seq;

  if (UseBatchedPath(params)) {
    const std::size_t q_elems =
        static_cast<std::size_t>(params.batch) * heads * params.seq * dim;
    const std::size_t k_elems =
        static_cast<std::size_t>(params.batch) * heads * kv_len * dim;
    ComputeType* head = CachedBuffer(1, 3 * q_elems + 4 * k_elems);
    ComputeType* qT = head;
    ComputeType* kT = qT + q_elems;
    ComputeType* vT = kT + k_elems;
    ComputeType* doutT = vT + k_elems;
    ComputeType* dqT = doutT + q_elems;
    ComputeType* dkT = dqT + q_elems;
    ComputeType* dvT = dkT + k_elems;
    TransposeToHeads(params.batch, params.seq, heads, dim, q, qT);
    TransposeToHeads(params.batch, kv_len, heads, dim, k, kT);
    TransposeToHeads(params.batch, kv_len, heads, dim, v, vT);
    TransposeToHeads(params.batch, params.seq, heads, dim, dout, doutT);

    const int batch_heads = params.batch * heads;
    // scores = Q @ K^T * scale; dP = dOut @ V^T.
    BatchedGemm(params.seq, kv_len, dim, batch_heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, qT, kT, scores);
    BatchedGemm(params.seq, kv_len, dim, batch_heads, 1.0f, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, doutT, vT, dprobs);
    cuda_backend::Launch(AttentionSoftmaxBackwardKernel, dim3(rows),
                         dim3(kSoftmaxThreads), 0, params, scores, stats, probs,
                         dprobs);

    // dQ = dS @ K * scale; dK = dS^T @ Q * scale; dV = P^T @ dOut.
    BatchedGemm(params.seq, dim, kv_len, batch_heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/false, dprobs, kT, dqT);
    BatchedGemm(kv_len, dim, params.seq, batch_heads, scale, 0.0f,
                /*transpose_a=*/true, /*transpose_b=*/false, dprobs, qT, dkT);
    BatchedGemm(kv_len, dim, params.seq, batch_heads, 1.0f, 0.0f,
                /*transpose_a=*/true, /*transpose_b=*/false, probs, doutT, dvT);

    TransposeFromHeads(params.batch, params.seq, heads, dim, dqT, dq);
    TransposeFromHeads(params.batch, kv_len, kv_heads, dim, dkT, dk);
    TransposeFromHeads(params.batch, kv_len, kv_heads, dim, dvT, dv);
    return;
  }

  // Grouped-query fallback: one GEMM per (b, h), accumulating the shared
  // key/value gradients, so the key/value buffers start at zero.
  const std::size_t kv_count =
      static_cast<std::size_t>(params.batch) * kv_len * kv_heads * dim;
  Memset(dk, 0, kv_count * sizeof(ComputeType));
  Memset(dv, 0, kv_count * sizeof(ComputeType));
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      GemmScores(params, b, h, scale, q, k, scores);
      StridedGemm(params.seq, kv_len, dim, 1.0f, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/true, params.num_heads * dim, kv_heads * dim,
                  kv_len,
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  v + HeadBase(b, kv_len, kv_heads, kvh, dim),
                  dprobs + ScoreBase(params, b, h));
    }
  }
  cuda_backend::Launch(AttentionSoftmaxBackwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, stats, probs,
                       dprobs);

  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      const ComputeType* ds = dprobs + ScoreBase(params, b, h);
      const ComputeType* qh =
          q + HeadBase(b, params.seq, params.num_heads, h, dim);
      const ComputeType* kh = k + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dqh = dq + HeadBase(b, params.seq, params.num_heads, h, dim);
      ComputeType* dkh = dk + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dvh = dv + HeadBase(b, kv_len, kv_heads, kvh, dim);

      // dQ = dS @ K * scale.
      StridedGemm(params.seq, dim, kv_len, scale, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/false, kv_len, kv_heads * dim,
                  params.num_heads * dim, ds, kh, dqh);
      // dK += dS^T @ Q * scale.
      StridedGemm(kv_len, dim, params.seq, scale, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, ds, qh, dkh);
      // dV += P^T @ dOut.
      StridedGemm(kv_len, dim, params.seq, 1.0f, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, probs + ScoreBase(params, b, h),
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  dvh);
    }
  }
}

}  // namespace kernels
}  // namespace nanochat
