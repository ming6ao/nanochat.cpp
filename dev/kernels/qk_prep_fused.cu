// dev/kernels QkPrep fusion prototype and its decomposed baseline.
//
// The fused body mirrors backends/cuda/kernels/qk_prep.cu line for line; the
// only structural change is that one kernel launch covers both q and k. The
// decomposed baseline uses the seam `RmsNormForward` plus a standalone
// RoPE-then-scale kernel, which is the honest alternative a fusion has to beat.
//
// See dev/kernels/qk_prep_fused.h and DESIGN.md section 3.

#include "dev/kernels/qk_prep_fused.h"

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockSizeForDim;
using cuda_kernels::RotatePairInPlace;
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::RsqrtScale;
using cuda_kernels::ToComputeDev;

// One row of the fused forward: RMSNorm over `head_dim`, then the RoPE rotation
// and scale on each (d, d + head_dim/2) pair, in place.
__device__ __forceinline__ void QkPrepRowForward(
    const QkPrepParams& params, const float* __restrict__ cos,
    const float* __restrict__ sin, ComputeType* __restrict__ tensor, int row,
    int heads) {
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

// One row of the fused backward. `tensor` holds the saved pre-norm row on
// entry and the input gradient on return; `grow` is the upstream gradient with
// respect to the final activation.
__device__ __forceinline__ void QkPrepRowBackward(
    const QkPrepParams& params, const float* __restrict__ cos,
    const float* __restrict__ sin, const ComputeType* __restrict__ grow,
    ComputeType* __restrict__ tensor, int row, int heads) {
  const int dim = params.head_dim;
  const int half = dim / 2;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  ComputeType* rowp = tensor + base;
  const ComputeType* grow_row = grow + base;

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  float sum_sq = 0.0f;
  float dot = 0.0f;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow_row[d]);
    const float g2 = AsFloatDev(grow_row[half + d]);
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
    const float g1 = AsFloatDev(grow_row[d]);
    const float g2 = AsFloatDev(grow_row[half + d]);
    const float gd = params.scale * (g1 * c - g2 * s);
    const float gd2 = params.scale * (g1 * s + g2 * c);
    rowp[d] = ToComputeDev(r * gd - coeff * x1);
    rowp[half + d] = ToComputeDev(r * gd2 - coeff * x2);
  }
}

// One grid dimension covers q then k; each block owns one row and takes the
// branch selected by its block index, so `__syncthreads` inside the row
// helpers stays block-local.
__global__ void QkPrepFusedForwardKernel(const QkPrepParams params,
                                         const float* __restrict__ cos,
                                         const float* __restrict__ sin,
                                         ComputeType* __restrict__ q,
                                         ComputeType* __restrict__ k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int row = blockIdx.x;
  if (row < q_rows) {
    QkPrepRowForward(params, cos, sin, q, row, params.num_heads);
  } else {
    QkPrepRowForward(params, cos, sin, k, row - q_rows, params.num_kv_heads);
  }
}

__global__ void QkPrepFusedBackwardKernel(
    const QkPrepParams params, const float* __restrict__ cos,
    const float* __restrict__ sin, const ComputeType* __restrict__ dq,
    const ComputeType* __restrict__ dk, ComputeType* __restrict__ q,
    ComputeType* __restrict__ k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int row = blockIdx.x;
  if (row < q_rows) {
    QkPrepRowBackward(params, cos, sin, dq, q, row, params.num_heads);
  } else {
    QkPrepRowBackward(params, cos, sin, dk, k, row - q_rows,
                      params.num_kv_heads);
  }
}

// Standalone RoPE + scale, the second half of the decomposed forward. Reads the
// RMSNorm output `in`, writes the final activation to `out`. `heads` selects q
// or k.
__global__ void RopeScaleForwardKernel(const QkPrepParams params,
                                       const float* __restrict__ cos,
                                       const float* __restrict__ sin,
                                       const ComputeType* __restrict__ in,
                                       ComputeType* __restrict__ out,
                                       int heads) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  const ComputeType* in_row = in + base;
  ComputeType* out_row = out + base;

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    float x1 = AsFloatDev(in_row[d]);
    float x2 = AsFloatDev(in_row[half + d]);
    RotatePairInPlace(x1, x2, crow[d], srow[d]);
    out_row[d] = ToComputeDev(x1 * params.scale);
    out_row[half + d] = ToComputeDev(x2 * params.scale);
  }
}

// The first half of the decomposed backward: pull the upstream gradient back
// through the scale and the rotation.
__global__ void RopeScaleBackwardKernel(const QkPrepParams params,
                                        const float* __restrict__ cos,
                                        const float* __restrict__ sin,
                                        const ComputeType* __restrict__ grad,
                                        ComputeType* __restrict__ out,
                                        int heads) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  const ComputeType* grow = grad + base;
  ComputeType* out_row = out + base;

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    out_row[d] = ToComputeDev(params.scale * (g1 * c - g2 * s));
    out_row[half + d] = ToComputeDev(params.scale * (g1 * s + g2 * c));
  }
}

void LaunchRowsForward(const QkPrepParams& params, const float* cos,
                       const float* sin, const ComputeType* in,
                       ComputeType* out, int rows, int heads) {
  if (rows <= 0 || params.head_dim <= 0) return;
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(RopeScaleForwardKernel, dim3(rows), dim3(block), 0,
                       params, cos, sin, in, out, heads);
}

void LaunchRowsBackward(const QkPrepParams& params, const float* cos,
                        const float* sin, const ComputeType* grad,
                        ComputeType* out, int rows, int heads) {
  if (rows <= 0 || params.head_dim <= 0) return;
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(RopeScaleBackwardKernel, dim3(rows), dim3(block), 0,
                       params, cos, sin, grad, out, heads);
}

}  // namespace

void QkPrepFusedForward(const QkPrepParams& params, const float* cos,
                        const float* sin, ComputeType* q, ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  const int rows = q_rows + k_rows;
  if (rows <= 0 || params.head_dim <= 0) return;
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(QkPrepFusedForwardKernel, dim3(rows), dim3(block), 0,
                       params, cos, sin, q, k);
}

void QkPrepFusedBackward(const QkPrepParams& params, const float* cos,
                         const float* sin, const ComputeType* dq,
                         const ComputeType* dk, ComputeType* q,
                         ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  const int rows = q_rows + k_rows;
  if (rows <= 0 || params.head_dim <= 0) return;
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(QkPrepFusedBackwardKernel, dim3(rows), dim3(block), 0,
                       params, cos, sin, dq, dk, q, k);
}

void QkPrepDecomposedForward(const QkPrepParams& params, const float* cos,
                             const float* sin, const ComputeType* q_in,
                             const ComputeType* k_in, ComputeType* q_normed,
                             ComputeType* k_normed, ComputeType* q_out,
                             ComputeType* k_out, float* q_rstd,
                             float* k_rstd) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    RmsNormParams rms;
    rms.rows = q_rows;
    rms.dim = params.head_dim;
    rms.eps = params.eps;
    nanochat::kernels::RmsNormForward(rms, q_in, q_normed, q_rstd);
    LaunchRowsForward(params, cos, sin, q_normed, q_out, q_rows,
                      params.num_heads);
  }
  if (k_rows > 0 && params.head_dim > 0) {
    RmsNormParams rms;
    rms.rows = k_rows;
    rms.dim = params.head_dim;
    rms.eps = params.eps;
    nanochat::kernels::RmsNormForward(rms, k_in, k_normed, k_rstd);
    LaunchRowsForward(params, cos, sin, k_normed, k_out, k_rows,
                      params.num_kv_heads);
  }
}

void QkPrepDecomposedBackward(
    const QkPrepParams& params, const float* cos, const float* sin,
    const ComputeType* dq, const ComputeType* dk, const ComputeType* q_in,
    const ComputeType* k_in, const float* q_rstd, const float* k_rstd,
    ComputeType* dq_normed, ComputeType* dk_normed, ComputeType* q_grad,
    ComputeType* k_grad) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    LaunchRowsBackward(params, cos, sin, dq, dq_normed, q_rows,
                       params.num_heads);
    RmsNormParams rms;
    rms.rows = q_rows;
    rms.dim = params.head_dim;
    rms.eps = params.eps;
    nanochat::kernels::RmsNormBackward(rms, q_in, dq_normed, q_rstd, q_grad);
  }
  if (k_rows > 0 && params.head_dim > 0) {
    LaunchRowsBackward(params, cos, sin, dk, dk_normed, k_rows,
                       params.num_kv_heads);
    RmsNormParams rms;
    rms.rows = k_rows;
    rms.dim = params.head_dim;
    rms.eps = params.eps;
    nanochat::kernels::RmsNormBackward(rms, k_in, dk_normed, k_rstd, k_grad);
  }
}

}  // namespace dev
}  // namespace nanochat
