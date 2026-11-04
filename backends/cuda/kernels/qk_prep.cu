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
// Both kernels carry the composition's arithmetic operation for operation, and
// not the reverse: `ops::QkPrepForward`/`ops::QkPrepBackward` in src/ops.cc
// stay the definition. The backward accumulates its two row statistics in
// double; both kernels derive `rstd` as `1.0f / std::sqrt(...)`, which is the
// `kRsqrt` pointwise code; and both round every product of the rotation
// separately.
//
// See docs/kernels.md and backends/cuda/kernels/device_utils.cuh.

#include <cmath>
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
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::ToComputeDev;

// The forward's pair rotation, with every float operation rounded to nearest.
// The composition is the `kRotatePair` pointwise code, whose definition is
// `alpha * (x1 * c + x2 * s)` and `alpha * (-x1 * s + x2 * c)` with every
// multiply and the add rounded separately (spec/pointwise_ops.md section 3.2).
// nvcc contracts `x1 * c + x2 * s` into an FMA by default, which leaves one
// product unrounded, and that alone puts the fused forward an ulp from the
// composition at every pair. The ulp is not cosmetic: AdamW's
// `adam_eps` amplifies one ulp on `backout_lambda` into 1.23e-4 of parameter
// error, over the 1e-4 trajectory tolerance of `//tests:oracle_cuda_test`. The
// intrinsics hold the source form to one rounding per operation, because the
// composition is the definition and the kernel moves to it. The scale multiply
// stays at the caller, exactly where the composition applies it. The shared
// `RotatePairInPlace` keeps its plain expressions: the other rotary callers
// have not been measured and this family does not move them.
__device__ __forceinline__ void RotateForwardPair(float& x1, float& x2, float c,
                                                  float s) {
  const float t1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
  const float t2 = __fsub_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  x1 = t1;
  x2 = t2;
}

// The backward's rotation and combine, with every float operation rounded to
// nearest. `QkPrepBackwardKernel` reproduces the composition's arithmetic, and
// the composition rounds each float operation: the transposed rotation of
// `kRotatePair` is `alpha * (g1 * c - g2 * s)` with the two products rounded
// before the subtract. nvcc contracts the same expression into an FMA by
// default, which keeps the product unrounded, and that alone leaves the fused
// pair 1 or 2 ulp from the composition at every element of the model shape. The
// intrinsics hold the source form to the composition's rounding. The forward's
// rotation has its own helper for the same reason (see `RotateForwardPair`
// above).
__device__ __forceinline__ float RotateBackwardFirst(float g1, float c,
                                                     float g2, float s,
                                                     float scale) {
  return __fmul_rn(scale, __fsub_rn(__fmul_rn(g1, c), __fmul_rn(g2, s)));
}

__device__ __forceinline__ float RotateBackwardSecond(float g1, float s,
                                                      float g2, float c,
                                                      float scale) {
  return __fmul_rn(scale, __fadd_rn(__fmul_rn(g1, s), __fmul_rn(g2, c)));
}

__device__ __forceinline__ float CombineBackward(float r, float grad,
                                                 float coeff, float x) {
  return __fsub_rn(__fmul_rn(r, grad), __fmul_rn(coeff, x));
}

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
  // The composition's `rstd`, not `rsqrtf`. `NormForward` runs the `kRsqrt`
  // pointwise code, which is `1.0f / std::sqrt` (backends/cpu/kernels.cc), and
  // it multiplies by the row's `1.0f / dim` before it adds `eps`; `RsqrtScale`
  // divides instead and calls `rsqrtf`. Each operation here is held to one
  // rounding so the value cannot move with the compiler's contraction choice.
  // Checked in the emitted PTX:
  // this form is `rcp.rn` / `mul.rn` / `add.rn` / `sqrt.rn` / `rcp.rn`, where
  // the old form was `rsqrt.approx.f32`.
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(__fadd_rn(__fmul_rn(sum_sq, inv_dim), params.eps));

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    // Round the normalized value to the compute type *before* the rotation,
    // because the composition does: its RMSNorm writes a compute-type tensor
    // and the rotation reads that one. Rounding after the rotation, or keeping
    // the value in fp32 across it, is the more accurate arithmetic, but the
    // fused pair is held to the composition's arithmetic and not the other way
    // round.
    float x1 = AsFloatDev(ToComputeDev(AsFloatDev(rowp[d]) * r));
    float x2 = AsFloatDev(ToComputeDev(AsFloatDev(rowp[half + d]) * r));
    RotateForwardPair(x1, x2, crow[d], srow[d]);
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
  // accumulate the two statistics the RMSNorm backward needs. The norm scale is
  // `1.0f / std::sqrt` of the float sum and the reciprocal is multiplied in
  // rather than divided, which is how the composition's `kRsqrt` step reads: a
  // `rsqrtf` would leave `r` one ulp away on every row, and AdamW's `adam_eps`
  // amplifies that past the trajectory tolerance.
  //
  // The two accumulators stay double here. The composition reduces through
  // `RowReduce`, whose device form accumulates in float, so the two agree to
  // within the declared bound rather than bit for bit; keeping double on the
  // fused side is the more accurate of the two and stays inside that bound. The
  // rotation itself stays float and rounded to nearest, as `kRotatePair`
  // defines it.
  double sum_sq = 0.0;
  double dot = 0.0;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = RotateBackwardFirst(g1, c, g2, s, params.scale);
    const float gd2 = RotateBackwardSecond(g1, s, g2, c, params.scale);
    sum_sq += static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
    dot += static_cast<double>(x1) * gd + static_cast<double>(x2) * gd2;
  }
  sum_sq = cuda_kernels::BlockReduceSum(sum_sq);
  dot = cuda_kernels::BlockReduceSum(dot);

  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
  const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = RotateBackwardFirst(g1, c, g2, s, params.scale);
    const float gd2 = RotateBackwardSecond(g1, s, g2, c, params.scale);
    rowp[d] = ToComputeDev(CombineBackward(r, gd, coeff, x1));
    rowp[half + d] = ToComputeDev(CombineBackward(r, gd2, coeff, x2));
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
