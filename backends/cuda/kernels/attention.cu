// CUDA Attention family: causal / sliding-window / grouped-query attention,
// forward and backward, plus the saved softmax statistics
// `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// One block per query row `(b, h, t)`, matching the CPU reference exactly:
// the block scans the visible key window, stabilises the softmax with the row
// maximum, and accumulates the value-weighted output. The backward recomputes
// the probabilities from the saved statistics, forms `p * (dp - sum(p * dp))`,
// and scatters the query/key/value gradients. Grouped-query attention lets
// several query heads share a key/value head, so the key/value gradients use
// atomics; the query gradient is private to its `(b, h, t)` row.
//
// The heavy lifting lives in the shared device helpers OnlineSoftmaxTile and
// SoftmaxGradTile (device_utils.cuh). See docs/kernels.md.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::KvHeadCountDev;
using cuda_kernels::KvHeadDev;
using cuda_kernels::OnlineSoftmaxTile;
using cuda_kernels::SoftmaxGradTile;

__global__ void AttentionForwardKernel(const AttentionParams params,
                                       float scale,
                                       const ComputeType* __restrict__ q,
                                       const ComputeType* __restrict__ k,
                                       const ComputeType* __restrict__ v,
                                       ComputeType* __restrict__ out,
                                       float* __restrict__ stats) {
  extern __shared__ float scratch[];
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = KvHeadCountDev(params);
  const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;

  const long long qbase =
      ((static_cast<long long>(b) * params.seq + t) * params.num_heads + h) *
      params.head_dim;
  const long long kvbase =
      static_cast<long long>(b) * kv_len * kv_heads * params.head_dim;
  float* stats_row =
      stats +
      ((static_cast<long long>(b) * params.num_heads + h) * params.seq + t) * 2;

  OnlineSoftmaxTile(params, q + qbase, k + kvbase, v + kvbase, kv_len, kvh,
                    qpos, scale, out + qbase, stats_row, scratch);
}

__global__ void AttentionBackwardKernel(
    const AttentionParams params, float scale,
    const ComputeType* __restrict__ q, const ComputeType* __restrict__ k,
    const ComputeType* __restrict__ v, const float* __restrict__ stats,
    const ComputeType* __restrict__ dout, ComputeType* __restrict__ dq,
    ComputeType* __restrict__ dk, ComputeType* __restrict__ dv) {
  extern __shared__ float scratch[];
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = KvHeadCountDev(params);
  const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;

  const long long qbase =
      ((static_cast<long long>(b) * params.seq + t) * params.num_heads + h) *
      params.head_dim;
  const long long kvbase =
      static_cast<long long>(b) * kv_len * kv_heads * params.head_dim;
  const float* stats_row =
      stats +
      ((static_cast<long long>(b) * params.num_heads + h) * params.seq + t) * 2;

  SoftmaxGradTile(params, q + qbase, k + kvbase, v + kvbase, kv_len, kvh, qpos,
                  scale, stats_row[0], stats_row[1], dout + qbase, dq + qbase,
                  dk + kvbase, dv + kvbase, scratch);
}

}  // namespace

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(params.head_dim));
  const int rows = params.batch * params.num_heads * params.seq;
  const int block = cuda_kernels::BlockSizeForDim(params.head_dim);
  const std::size_t shared = sizeof(float) * static_cast<std::size_t>(kv_len);
  cuda_backend::Launch(AttentionForwardKernel, dim3(rows), dim3(block), shared,
                       params, scale, q, k, v, out, stats);
}

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const int kv_heads =
      params.num_kv_heads > 0 ? params.num_kv_heads : params.num_heads;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(params.head_dim));

  // The query gradient is private to each (b, h, t) row; the key/value
  // gradients are shared by the GQA group and accumulated with atomics, so the
  // whole buffers are cleared first.
  const std::size_t q_count = static_cast<std::size_t>(params.batch) *
                              params.seq * params.num_heads * params.head_dim;
  const std::size_t kv_count = static_cast<std::size_t>(params.batch) * kv_len *
                               kv_heads * params.head_dim;
  Memset(dq, 0, q_count * sizeof(ComputeType));
  Memset(dk, 0, kv_count * sizeof(ComputeType));
  Memset(dv, 0, kv_count * sizeof(ComputeType));

  const int rows = params.batch * params.num_heads * params.seq;
  const int block = cuda_kernels::BlockSizeForDim(params.head_dim);
  const std::size_t shared =
      2 * sizeof(float) * static_cast<std::size_t>(kv_len);
  cuda_backend::Launch(AttentionBackwardKernel, dim3(rows), dim3(block), shared,
                       params, scale, q, k, v, stats, dout, dq, dk, dv);
}

}  // namespace kernels
}  // namespace nanochat
