// Attention-tile-v3 prototype: the fused fp32 tiled attention forward.
//
// This is the Phase 1 forward prototype from docs/flash-attention-pascal.md. It
// replaces the shipped "transpose + dense cuBLAS + row softmax + dense cuBLAS"
// decomposition with one kernel that keeps the online-softmax state and the
// output accumulator in registers and skips fully masked key tiles. It is
// dev-only; the CUDA backend is unchanged. See attention_tile_v3.h and
// dev/kernels/README.md.

#include "attention_tile_v3.h"

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::KeyAllowedDev;
using cuda_kernels::KvHeadCountDev;
using cuda_kernels::KvHeadDev;
using cuda_kernels::ToComputeDev;

// Candidate tile (docs/flash-attention-pascal.md section 5.2): one block of 256
// threads owns 32 query rows and walks the key/value sequence in tiles of 32
// rows. The key tile, the value tile, and the per-tile probability tile live in
// shared memory; the running maximum, the running denominator, and the output
// accumulator live in registers. The shared budget stays under the Pascal 48 KB
// limit, so two blocks stay resident per streaming multiprocessor.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;       // 8 warps per block
constexpr int kBr = 32;                     // query rows per block
constexpr int kBc = 32;                     // key/value rows per tile
constexpr int kRowsPerWarp = kBr / kWarps;  // 4 query rows per warp
constexpr int kDimsPerLane = 4;             // owned output dims, head_dim <= 128

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

// One thread block owns `kBr` query rows for one `(batch, head)` pair. The warp
// `w` owns rows `t0 + w * kRowsPerWarp ..`; within a warp, lane `l` scores the
// key `j0 + l` and owns output dimensions `l, l + 32, ...`.
__global__ void __launch_bounds__(kThreads, 2)
AttentionTileV3ForwardKernel(const AttentionParams params,
                             const ComputeType* __restrict__ q,
                             const ComputeType* __restrict__ k,
                             const ComputeType* __restrict__ v,
                             ComputeType* __restrict__ out,
                             float* __restrict__ stats) {
  const int dim = params.head_dim;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const int kv_heads = KvHeadCountDev(params);
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : rsqrtf(static_cast<float>(dim));

  const int t0 = blockIdx.x * kBr;
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
  ComputeType* vs = ks + kBc * kv_stride;
  float* ps = reinterpret_cast<float*>(vs + kBc * kv_stride);

  // Per-lane online-softmax state for the rows this warp owns.
  float row_max[kRowsPerWarp];
  float row_sum[kRowsPerWarp];
  float o_acc[kRowsPerWarp][kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    row_max[i] = -CUDART_INF_F;
    row_sum[i] = 0.0f;
#pragma unroll
    for (int c = 0; c < kDimsPerLane; ++c) o_acc[i][c] = 0.0f;
  }

  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kBr - 1;
  const int num_key_tiles = (kv_len + kBc - 1) / kBc;

  for (int tile = 0; tile < num_key_tiles; ++tile) {
    const int j0 = tile * kBc;

    // Skip a key tile that is entirely masked for every query row in the
    // block. This is what turns the sliding window into saved work rather than
    // a mask applied after a dense product. The conditions are conservative:
    // they only skip a tile that is provably invisible to all rows.
    if (params.causal && j0 > qpos_max) continue;
    if (params.window_left >= 0 &&
        j0 + kBc - 1 < qpos_min - params.window_left) {
      continue;
    }
    if (params.window_right >= 0 && j0 > qpos_max + params.window_right) {
      continue;
    }

    // Stage the key and value tile. The global read follows the row-major
    // layout and coalesces; the padded shared write keeps the banks balanced.
    for (int idx = tid; idx < kBc * dim; idx += kThreads) {
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

    // QK^T and the causal/window mask. Each lane scores its own key for the
    // rows the warp owns; the head-dimension dot product stays inside the lane
    // and needs no cross-lane reduction. The key element is loaded once and
    // reused across the four rows.
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
                ? q + ((static_cast<std::size_t>(b) * seq + tq) * heads + h) * dim
                : q;
      }
      float dot[kRowsPerWarp];
#pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i) dot[i] = 0.0f;
      const long long jabs = j0 + lane;
      if (jabs < kv_len) {
        const ComputeType* krow = ks + lane * kv_stride;
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

    // Online-softmax update. `alpha` rescales the running denominator and the
    // output accumulator onto the new maximum. A row with no visible key in
    // this tile keeps its state: alpha is one and the new tile sum is zero.
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

    // P = exp(S - row_max). The warp sum over the lanes is the tile sum, which
    // the running denominator absorbs with the same rescale.
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
      for (int c = 0; c < kDimsPerLane; ++c) o_acc[i][c] *= alpha[i];
      ps[(warp * kRowsPerWarp + i) * kBc + lane] = p_reg[i];
    }
    __syncthreads();

    // O += P V for this tile. Every lane reads every key through the shared
    // probability tile, so each owned output dimension is a private
    // accumulation and needs no cross-lane reduction. The value element is
    // loaded once and reused across the four rows.
#pragma unroll 4
    for (int jr = 0; jr < kBc; ++jr) {
      const ComputeType* vrow = vs + jr * kv_stride;
      float value_value[kDimsPerLane];
#pragma unroll
      for (int c = 0; c < kDimsPerLane; ++c) {
        const int d = lane + 32 * c;
        value_value[c] = (d < dim) ? AsFloatDev(vrow[d]) : 0.0f;
      }
#pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i) {
        const float p = ps[(warp * kRowsPerWarp + i) * kBc + jr];
#pragma unroll
        for (int c = 0; c < kDimsPerLane; ++c) {
          o_acc[i][c] += p * value_value[c];
        }
      }
    }
    __syncthreads();
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
      for (int c = 0; c < kDimsPerLane; ++c) {
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
      for (int c = 0; c < kDimsPerLane; ++c) {
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

}  // namespace

int AttentionTileV3Forward(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats) {
  const int dim = params.head_dim;
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      dim <= 0 || q == nullptr || k == nullptr || v == nullptr ||
      out == nullptr || stats == nullptr) {
    return 0;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return 0;
  // The candidate tile keeps `kDimsPerLane` output dimensions per lane, and 32
  // lanes cover the query row, so head_dim must fit in that register slice.
  if (dim > kDimsPerLane * 32) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t kv_stride = static_cast<std::size_t>(dim) + 1;
  const std::size_t shared_bytes =
      (2 * kBc * kv_stride) * sizeof(ComputeType) +
      (kBr * kBc) * sizeof(float);
  // Pascal caps a block at 48 KB of shared memory; the candidate tile uses
  // 37 KB at head_dim 128 and stays under the limit.
  if (shared_bytes > 48 * 1024) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int query_tiles = (params.seq + kBr - 1) / kBr;
  const dim3 grid(query_tiles, params.num_heads, params.batch);
  const dim3 block(kThreads);
  cuda_backend::Launch(AttentionTileV3ForwardKernel, grid, block, shared_bytes,
                       params, q, k, v, out, stats);
  return static_cast<int>(cudaGetLastError());
}

int AttentionTileV3Backward(const AttentionParams& params,
                            const ComputeType* q, const ComputeType* k,
                            const ComputeType* v, const float* stats,
                            const ComputeType* dout, ComputeType* dq,
                            ComputeType* dk, ComputeType* dv) {
  // Backward is the Phase 2 prototype and lands in a later wave. The parameters
  // stay unused on purpose so the forward prototype can compile and link.
  (void)params;
  (void)q;
  (void)k;
  (void)v;
  (void)stats;
  (void)dout;
  (void)dq;
  (void)dk;
  (void)dv;
  return static_cast<int>(cudaGetLastError());
}

}  // namespace dev
}  // namespace nanochat
