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
// at about 6-7 TFLOP/s. Grouped-query attention takes the same batched path
// after its key and value heads are widened to the query head count, which is
// one transpose kernel each and no second pass over the scores. Its backward
// computes one key and value gradient per query head and folds the heads that
// share a key/value head into it once, because a batched call writes one output
// per batch item and cannot accumulate into a shared one.
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

// The same batched product with explicit leading dimensions and per-batch
// strides. The windowed backward passes a query/key sub-block of the transposed
// `[batch, heads, seq, dim]` head buffer, whose per-batch stride is the full
// `seq * dim`, not the sub-block size the seam would infer.
void BatchedGemmStrided(int m, int n, int k, int batch_count, float alpha,
                        float beta, bool transpose_a, bool transpose_b,
                        const ComputeType* a, const ComputeType* b,
                        ComputeType* c, int lda, std::int64_t stride_a, int ldb,
                        std::int64_t stride_b, int ldc, std::int64_t stride_c) {
  GemmParams gp;
  gp.m = m;
  gp.n = n;
  gp.k = k;
  gp.batch_count = batch_count;
  gp.alpha = alpha;
  gp.beta = beta;
  gp.transpose_a = transpose_a;
  gp.transpose_b = transpose_b;
  gp.lda = lda;
  gp.stride_a = stride_a;
  gp.ldb = ldb;
  gp.stride_b = stride_b;
  gp.ldc = ldc;
  gp.stride_c = stride_c;
  Gemm(GemmMode::kForward, gp, a, b, c);
}

// Cached device scratch, keyed by slot: 0 is the `[batch, heads, seq, kv_len]`
// score/probability matrices, 1 is the transposed head buffers. Attention runs
// once per layer, so a cudaMalloc/free per call would dominate. The training
// graph is single-threaded, so no locking is needed.
ComputeType* CachedBuffer(int slot, std::size_t count) {
  // Per device thread: the scratch belongs to the current device
  // (docs/client-runtime-design.md section 8.1).
  thread_local ComputeType* buffers[2] = {nullptr, nullptr};
  thread_local std::size_t capacities[2] = {0, 0};
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
      (static_cast<std::int64_t>(b) * params.num_heads + h) *
          (params.stats_row_stride > 0 ? params.stats_row_stride : params.seq) *
          2 +
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

// `[batch, rows, kv_heads, dim]` -> `[batch, heads, rows, dim]`. Query head `h`
// reads the key/value head it shares, `KvHeadDev(h)`, so one call widens a
// grouped-query key or value buffer to the query head count, and the same call
// is the plain transpose when the two counts match. The widening is what lets a
// batched GEMM cover every head: the seam infers one per-batch stride for the
// whole call, and shared heads do not step at that stride.
__global__ void TransposeToHeadsKernel(long long total, int rows, int heads,
                                       int kv_heads, int dim,
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
  const int kvh = KvHeadDev(h, heads, kv_heads);
  out[i] =
      in[((static_cast<long long>(b) * rows + r) * kv_heads + kvh) * dim + d];
}

void TransposeToHeads(int batch, int rows, int heads, int kv_heads, int dim,
                      const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * heads * rows * dim;
  if (total <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeToHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, kv_heads,
                       dim, in, out);
}

// `[batch, heads, rows, dim]` -> `[batch, rows, kv_heads, dim]`, summing in
// float the query heads that share a key/value head. The backward computes one
// key and value gradient per query head, because that is what one batched call
// can write, and folds them here. `KvHeadDev` is the map the softmax kernels
// read, so the last key/value head also takes the heads that the map clamps
// onto it. With one head per key and value this is the plain transpose back.
__global__ void TransposeFromHeadsKernel(long long total, int rows, int heads,
                                         int kv_heads, int dim,
                                         const ComputeType* __restrict__ in,
                                         ComputeType* __restrict__ out) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = i % dim;
  const int r = (i / dim) % rows;
  const int kvh = (i / (dim * rows)) % kv_heads;
  const int b =
      static_cast<int>(i / (static_cast<long long>(dim) * rows * kv_heads));
  const int group = heads / kv_heads > 0 ? heads / kv_heads : 1;
  const int first = kvh * group;
  const int last = kvh == kv_heads - 1 ? heads : (kvh + 1) * group;
  const std::int64_t base = static_cast<std::int64_t>(b) * heads * rows * dim +
                            static_cast<std::int64_t>(r) * dim + d;
  const std::int64_t stride = static_cast<std::int64_t>(rows) * dim;
  float acc = 0.0f;
  for (int h = first; h < last; ++h) {
    acc += AsFloatDev(in[base + static_cast<std::int64_t>(h) * stride]);
  }
  // The model's buffer keeps the head innermost, so the write is not the read
  // order.
  const long long out_index =
      (static_cast<long long>(b) * rows + r) * kv_heads * dim +
      static_cast<long long>(kvh) * dim + d;
  out[out_index] = ToComputeDev(acc);
}

void TransposeFromHeads(int batch, int rows, int heads, int kv_heads, int dim,
                        const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * kv_heads * rows * dim;
  if (total <= 0 || heads <= 0 || kv_heads <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeFromHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, kv_heads,
                       dim, in, out);
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
  // so it is 8-byte aligned.
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

// Whether the fused tiled forward covers this shape. The tile computes the
// query-key product with a scalar FMA loop, so it belongs on a card whose
// matmuls have no tensor cores to lose to: it took the Phase 1 gate on the
// GTX 1080 Ti. A bf16 build is a tensor-core build, because bf16 arithmetic
// needs the units, and there the tile loses: the 13 sliding-window layers of
// the `big` profile take 34.6 ms through it, against a 1.06 ms roofline bound
// for all 20 layers. The tile also covers only `head_dim` 128, and only a
// windowed shape, because the full-causal tile lost its gate to cuBLAS.
bool UseFusedAttention(const AttentionParams& p) {
  return !kComputeIsBf16 && p.window_left >= 0 && p.head_dim == 128;
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

  // One batched call per GEMM, at every head count. Grouped-query attention
  // widens its key and value heads to the query head count first, so the
  // per-batch stride is uniform across the batch and one call still owns the
  // whole output: no two batches write the same element.
  const int kv_heads = KvHeadCountDev(params);
  const std::size_t q_elems =
      static_cast<std::size_t>(params.batch) * heads * params.seq * dim;
  const std::size_t k_elems =
      static_cast<std::size_t>(params.batch) * heads * kv_len * dim;
  ComputeType* head = CachedBuffer(1, 2 * q_elems + 2 * k_elems);
  ComputeType* qT = head;
  ComputeType* kT = qT + q_elems;
  ComputeType* vT = kT + k_elems;
  ComputeType* outT = vT + k_elems;
  TransposeToHeads(params.batch, params.seq, heads, heads, dim, q, qT);
  TransposeToHeads(params.batch, kv_len, heads, kv_heads, dim, k, kT);
  TransposeToHeads(params.batch, kv_len, heads, kv_heads, dim, v, vT);

  const int batch_heads = params.batch * heads;
  BatchedGemm(params.seq, kv_len, dim, batch_heads, scale, 0.0f,
              /*transpose_a=*/false, /*transpose_b=*/true, qT, kT, scores);
  cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, probs, stats);
  BatchedGemm(params.seq, dim, kv_len, batch_heads, 1.0f, 0.0f,
              /*transpose_a=*/false, /*transpose_b=*/false, probs, vT, outT);
  TransposeFromHeads(params.batch, params.seq, heads, heads, dim, outT, out);
}

// True when the backward can run the banded query-block schedule. The fused
// windowed forward already skips masked tiles; this is its backward partner.
// It covers both multi-head and grouped-query attention.
bool UseWindowedBackward(const AttentionParams& p) {
  if (!p.causal || p.window_left <= 0 || p.window_right != 0) return false;
  if (p.kv_len > 0 && p.kv_len != p.seq) return false;
  // A window that spans the whole sequence is the dense case.
  return p.seq > p.window_left;
}

// The windowed backward: a banded loop over query blocks. Each block scores
// `Bq` queries against the `Bq + window` keys that any query in the block can
// see, so the dense `seq x seq` score matrix is never built. Adjacent blocks
// share keys, so `dK`/`dV` accumulate over the zeroed buffers. The saved
// forward statistics are shared through `stats_row_stride` and the query
// offset. See `docs/attention.md` Part 2.
//
// Every product is one batched call over `batch * heads`, which needs the
// operands in head-major order and, for grouped-query attention, needs the key
// and value heads widened to the query head count first. The fold at the end
// then sums the heads that share a key or value head, the same way the dense
// backward does.
void AttentionBackwardWindowed(const AttentionParams& params,
                               const ComputeType* q, const ComputeType* k,
                               const ComputeType* v, const float* stats,
                               const ComputeType* dout, ComputeType* dq,
                               ComputeType* dk, ComputeType* dv) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = KvHeadCountDev(params);
  const int dim = params.head_dim;
  const int window = params.window_left;
  const int Bq = window;
  const int batch_heads = batch * heads;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));

  const std::size_t q_elems =
      static_cast<std::size_t>(batch) * heads * seq * dim;
  // Seven transposed head buffers: q, k, v, dout, dq, dk, dv.
  ComputeType* head = CachedBuffer(1, 7 * q_elems);
  ComputeType* qT = head;
  ComputeType* kT = qT + q_elems;
  ComputeType* vT = kT + q_elems;
  ComputeType* doutT = vT + q_elems;
  ComputeType* dqT = doutT + q_elems;
  ComputeType* dkT = dqT + q_elems;
  ComputeType* dvT = dkT + q_elems;
  TransposeToHeads(batch, seq, heads, heads, dim, q, qT);
  TransposeToHeads(batch, seq, heads, kv_heads, dim, k, kT);
  TransposeToHeads(batch, seq, heads, kv_heads, dim, v, vT);
  TransposeToHeads(batch, seq, heads, heads, dim, dout, doutT);
  Memset(dqT, 0, q_elems * sizeof(ComputeType));
  Memset(dkT, 0, q_elems * sizeof(ComputeType));
  Memset(dvT, 0, q_elems * sizeof(ComputeType));

  // One block's score scratch: `[batch*heads, Bq, key_max]`, indexed through
  // `ScoreBase` as `[batch*heads, q_len, k_len]`.
  const int key_max = window + Bq;
  const std::size_t block_matrix =
      static_cast<std::size_t>(batch_heads) * Bq * key_max;
  ComputeType* scores = CachedBuffer(0, 3 * block_matrix);
  ComputeType* probs = scores + block_matrix;
  ComputeType* dprobs = probs + block_matrix;

  const std::int64_t head_stride =
      static_cast<std::int64_t>(seq) * dim;  // per (b, h) head buffer stride

  for (int q0 = 0; q0 < seq; q0 += Bq) {
    const int q_len = std::min(Bq, seq - q0);
    const int k0 = std::max(0, q0 - window);
    const int k_len = std::min(seq, q0 + q_len) - k0;
    const std::int64_t block_stride = static_cast<std::int64_t>(q_len) * k_len;

    // scores = Q K^T * scale; dP = dOut V^T.
    BatchedGemmStrided(q_len, k_len, dim, batch_heads, scale, 0.0f,
                       /*transpose_a=*/false, /*transpose_b=*/true,
                       qT + static_cast<std::int64_t>(q0) * dim,
                       kT + static_cast<std::int64_t>(k0) * dim, scores,
                       /*lda=*/dim, head_stride, /*ldb=*/dim, head_stride,
                       /*ldc=*/k_len, block_stride);
    BatchedGemmStrided(q_len, k_len, dim, batch_heads, 1.0f, 0.0f,
                       /*transpose_a=*/false, /*transpose_b=*/true,
                       doutT + static_cast<std::int64_t>(q0) * dim,
                       vT + static_cast<std::int64_t>(k0) * dim, dprobs,
                       /*lda=*/dim, head_stride, /*ldb=*/dim, head_stride,
                       /*ldc=*/k_len, block_stride);

    AttentionParams sub = params;
    sub.seq = q_len;
    sub.kv_len = k_len;
    sub.stats_row_stride = seq;
    cuda_backend::Launch(AttentionSoftmaxBackwardKernel,
                         dim3(batch_heads * q_len), dim3(kSoftmaxThreads), 0,
                         sub, scores, stats + static_cast<std::int64_t>(q0) * 2,
                         probs, dprobs);

    // dQ = dS K * scale. Each query block owns a disjoint query slice, so it
    // overwrites (beta = 0).
    BatchedGemmStrided(q_len, dim, k_len, batch_heads, scale, 0.0f,
                       /*transpose_a=*/false, /*transpose_b=*/false, dprobs,
                       kT + static_cast<std::int64_t>(k0) * dim,
                       dqT + static_cast<std::int64_t>(q0) * dim,
                       /*lda=*/k_len, block_stride, /*ldb=*/dim, head_stride,
                       /*ldc=*/dim, head_stride);
    // dK += dS^T Q * scale; dV += P^T dOut. Adjacent query blocks share keys,
    // so these accumulate (beta = 1) over the zeroed buffers.
    BatchedGemmStrided(k_len, dim, q_len, batch_heads, scale, 1.0f,
                       /*transpose_a=*/true, /*transpose_b=*/false, dprobs,
                       qT + static_cast<std::int64_t>(q0) * dim,
                       dkT + static_cast<std::int64_t>(k0) * dim,
                       /*lda=*/k_len, block_stride, /*ldb=*/dim, head_stride,
                       /*ldc=*/dim, head_stride);
    BatchedGemmStrided(k_len, dim, q_len, batch_heads, 1.0f, 1.0f,
                       /*transpose_a=*/true, /*transpose_b=*/false, probs,
                       doutT + static_cast<std::int64_t>(q0) * dim,
                       dvT + static_cast<std::int64_t>(k0) * dim,
                       /*lda=*/k_len, block_stride, /*ldb=*/dim, head_stride,
                       /*ldc=*/dim, head_stride);
  }

  TransposeFromHeads(batch, seq, heads, heads, dim, dqT, dq);
  TransposeFromHeads(batch, seq, heads, kv_heads, dim, dkT, dk);
  TransposeFromHeads(batch, seq, heads, kv_heads, dim, dvT, dv);
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

  // The banded backward never builds the dense `seq x seq` score matrix, so it
  // dispatches before that scratch is requested (docs/attention.md Part 2).
  if (UseWindowedBackward(params)) {
    AttentionBackwardWindowed(params, q, k, v, stats, dout, dq, dk, dv);
    return;
  }

  const std::size_t matrix =
      static_cast<std::size_t>(params.batch) * heads * params.seq * kv_len;
  ComputeType* scores = CachedBuffer(0, 3 * matrix);
  ComputeType* probs = scores + matrix;
  ComputeType* dprobs = probs + matrix;
  const int rows = params.batch * heads * params.seq;

  // One batched call per GEMM, at every head count. Grouped-query attention
  // widens its key and value heads to the query head count first, so the
  // per-batch stride is uniform, and it folds the query heads that share a
  // key/value head into that head once at the end instead of accumulating into
  // it once per launch.
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
  TransposeToHeads(params.batch, params.seq, heads, heads, dim, q, qT);
  TransposeToHeads(params.batch, kv_len, heads, kv_heads, dim, k, kT);
  TransposeToHeads(params.batch, kv_len, heads, kv_heads, dim, v, vT);
  TransposeToHeads(params.batch, params.seq, heads, heads, dim, dout, doutT);

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

  TransposeFromHeads(params.batch, params.seq, heads, heads, dim, dqT, dq);
  TransposeFromHeads(params.batch, kv_len, heads, kv_heads, dim, dkT, dk);
  TransposeFromHeads(params.batch, kv_len, heads, kv_heads, dim, dvT, dv);
}

}  // namespace kernels
}  // namespace nanochat
