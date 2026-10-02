// CUDA QkPrep family: fused RMSNorm -> RoPE -> scale (1.2 by default) on the
// query and key projections, in place. One block per (batch, seq, head) row,
// exactly like the CPU reference: the row is normalized, then each
// (d, d + head_dim/2) pair is rotated by the fp32 rotary table and scaled.
//
// Backward contract (frozen seam): the q/k buffers handed to QkPrepBackward
// hold the *saved pre-norm projection outputs* on entry and are overwritten
// with the gradient with respect to those outputs. The upstream dq/dk are the
// gradients of the final (RoPE'd, scaled) activation.
//
// See docs/kernels.md and backends/cuda/kernels/device_utils.cuh.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockSizeForDim;
using cuda_kernels::RotatePairInPlace;
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::RsqrtScale;
using cuda_kernels::ToComputeDev;

// One row is (batch, seq, head); `heads` selects the tensor's head count. The
// row position feeds the shared rotary table.
__global__ void QkPrepForwardKernel(const QkPrepParams params,
                                    const float* __restrict__ cos,
                                    const float* __restrict__ sin,
                                    ComputeType* __restrict__ tensor,
                                    int heads) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  ComputeType* rowp = tensor + static_cast<std::size_t>(row) * dim;

  const float sum_sq = RowReduceSumSq(rowp, dim);
  const float r = RsqrtScale(sum_sq, dim, params.eps);

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    float x1 = AsFloatDev(rowp[d]) * r;
    float x2 = AsFloatDev(rowp[half + d]) * r;
    RotatePairInPlace(x1, x2, crow[d], srow[d]);
    rowp[d] = ToComputeDev(x1 * params.scale);
    rowp[half + d] = ToComputeDev(x2 * params.scale);
  }
}

__global__ void QkPrepBackwardKernel(const QkPrepParams params,
                                     const float* __restrict__ cos,
                                     const float* __restrict__ sin,
                                     const ComputeType* __restrict__ grad,
                                     ComputeType* __restrict__ tensor,
                                     int heads) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  ComputeType* rowp = tensor + base;
  const ComputeType* grow = grad + base;

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  // Pull the upstream gradient back through the scale and the rotation, and
  // accumulate the two statistics the RMSNorm backward needs.
  float sum_sq = 0.0f;
  float dot = 0.0f;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = params.scale * (g1 * c - g2 * s);
    const float gd2 = params.scale * (g1 * s + g2 * c);
    sum_sq += x1 * x1 + x2 * x2;
    dot += x1 * gd + x2 * gd2;
  }
  sum_sq = cuda_kernels::BlockReduceSum(sum_sq);
  dot = cuda_kernels::BlockReduceSum(dot);

  const float r = RsqrtScale(sum_sq, dim, params.eps);
  const float coeff = r * r * r * dot / static_cast<float>(dim);

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = params.scale * (g1 * c - g2 * s);
    const float gd2 = params.scale * (g1 * s + g2 * c);
    rowp[d] = ToComputeDev(r * gd - coeff * x1);
    rowp[half + d] = ToComputeDev(r * gd2 - coeff * x2);
  }
}

}  // namespace

void QkPrepForward(const QkPrepParams& params, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    const int block = BlockSizeForDim(params.head_dim);
    cuda_backend::Launch(QkPrepForwardKernel, dim3(q_rows), dim3(block), 0,
                         params, cos, sin, q, params.num_heads);
  }
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (k_rows > 0 && params.head_dim > 0) {
    const int block = BlockSizeForDim(params.head_dim);
    cuda_backend::Launch(QkPrepForwardKernel, dim3(k_rows), dim3(block), 0,
                         params, cos, sin, k, params.num_kv_heads);
  }
}

void QkPrepBackward(const QkPrepParams& params, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    const int block = BlockSizeForDim(params.head_dim);
    cuda_backend::Launch(QkPrepBackwardKernel, dim3(q_rows), dim3(block), 0,
                         params, cos, sin, dq, q, params.num_heads);
  }
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (k_rows > 0 && params.head_dim > 0) {
    const int block = BlockSizeForDim(params.head_dim);
    cuda_backend::Launch(QkPrepBackwardKernel, dim3(k_rows), dim3(block), 0,
                         params, cos, sin, dk, k, params.num_kv_heads);
  }
}

}  // namespace kernels
}  // namespace nanochat
