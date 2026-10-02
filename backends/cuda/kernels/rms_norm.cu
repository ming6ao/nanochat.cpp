// CUDA RmsNorm family: the plain forward/backward pair from the frozen seam
// plus the fused residual add + norm helper used by the tests. One block per
// row: the row is reduced cooperatively and then scaled in place. The math is
// a direct port of backends/cpu/kernels.cc.
//
// See docs/kernels.md and backends/cuda/kernels/device_utils.cuh.

#include "backends/cuda/kernels/rms_norm.h"

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::RowReduceDot;
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::RsqrtScale;
using cuda_kernels::ToComputeDev;

// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps).
__global__ void RmsNormForwardKernel(const RmsNormParams params,
                                     const ComputeType* __restrict__ x,
                                     ComputeType* __restrict__ out,
                                     float* __restrict__ rstd) {
  const int row = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(row) * params.dim;
  const ComputeType* xr = x + base;
  ComputeType* orow = out + base;

  const float sum_sq = RowReduceSumSq(xr, params.dim);
  const float r = RsqrtScale(sum_sq, params.dim, params.eps);
  if (threadIdx.x == 0) rstd[row] = r;

  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    orow[d] = ToComputeDev(AsFloatDev(xr[d]) * r);
  }
}

// dx = r * dy - (r^3 * dot(x, dy) / dim) * x.
__global__ void RmsNormBackwardKernel(const RmsNormParams params,
                                      const ComputeType* __restrict__ x,
                                      const ComputeType* __restrict__ dy,
                                      const float* __restrict__ rstd,
                                      ComputeType* __restrict__ dx) {
  const int row = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(row) * params.dim;
  const ComputeType* xr = x + base;
  const ComputeType* dyr = dy + base;
  ComputeType* dxr = dx + base;

  const float dot = RowReduceDot(xr, dyr, params.dim);
  const float r = rstd[row];
  const float coeff =
      r * r * r * dot / static_cast<float>(params.dim);

  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    dxr[d] =
        ToComputeDev(r * AsFloatDev(dyr[d]) - coeff * AsFloatDev(xr[d]));
  }
}

// residual[r, :] = x + residual; out[r, :] = residual * rstd.
__global__ void RmsNormFusedResidualForwardKernel(
    const RmsNormParams params, const ComputeType* __restrict__ x,
    ComputeType* __restrict__ residual, ComputeType* __restrict__ out,
    float* __restrict__ rstd) {
  const int row = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(row) * params.dim;
  const ComputeType* xr = x + base;
  ComputeType* rr = residual + base;
  ComputeType* orow = out + base;

  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    const float v = AsFloatDev(xr[d]) + AsFloatDev(rr[d]);
    sum_sq += v * v;
  }
  sum_sq = cuda_kernels::BlockReduceSum(sum_sq);
  const float r = RsqrtScale(sum_sq, params.dim, params.eps);
  if (threadIdx.x == 0) rstd[row] = r;

  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    const float v = AsFloatDev(xr[d]) + AsFloatDev(rr[d]);
    rr[d] = ToComputeDev(v);
    orow[d] = ToComputeDev(v * r);
  }
}

// Gradient of the fused residual: both x and the incoming residual receive the
// same value, since residual_out = x + residual_in.
__global__ void RmsNormFusedResidualBackwardKernel(
    const RmsNormParams params, const ComputeType* __restrict__ residual,
    const float* __restrict__ rstd, const ComputeType* __restrict__ dy,
    ComputeType* __restrict__ dx, ComputeType* __restrict__ dresidual) {
  const int row = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(row) * params.dim;
  const ComputeType* sr = residual + base;
  const ComputeType* dyr = dy + base;
  ComputeType* dxr = dx + base;
  ComputeType* drr = dresidual + base;

  const float dot = RowReduceDot(sr, dyr, params.dim);
  const float r = rstd[row];
  const float coeff =
      r * r * r * dot / static_cast<float>(params.dim);

  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    const float s = AsFloatDev(sr[d]);
    const float g = r * AsFloatDev(dyr[d]) - coeff * s;
    dxr[d] = ToComputeDev(g);
    drr[d] = ToComputeDev(g);
  }
}

}  // namespace

void RmsNormForward(const RmsNormParams& params, const ComputeType* x,
                    ComputeType* out, float* rstd) {
  if (params.rows <= 0 || params.dim <= 0) return;
  const int block = cuda_kernels::BlockSizeForDim(params.dim);
  cuda_backend::Launch(RmsNormForwardKernel, dim3(params.rows), dim3(block), 0,
                       params, x, out, rstd);
}

void RmsNormBackward(const RmsNormParams& params, const ComputeType* x,
                     const ComputeType* dy, const float* rstd,
                     ComputeType* dx) {
  if (params.rows <= 0 || params.dim <= 0) return;
  const int block = cuda_kernels::BlockSizeForDim(params.dim);
  cuda_backend::Launch(RmsNormBackwardKernel, dim3(params.rows), dim3(block), 0,
                       params, x, dy, rstd, dx);
}

}  // namespace kernels

namespace cuda_kernels {

void RmsNormForwardFusedResidual(const RmsNormParams& params,
                                 const ComputeType* x, ComputeType* residual,
                                 ComputeType* out, float* rstd) {
  if (params.rows <= 0 || params.dim <= 0) return;
  const int block = cuda_kernels::BlockSizeForDim(params.dim);
  cuda_backend::Launch(kernels::RmsNormFusedResidualForwardKernel,
                       dim3(params.rows), dim3(block), 0, params, x, residual,
                       out, rstd);
}

void RmsNormBackwardFusedResidual(const RmsNormParams& params,
                                  const ComputeType* residual,
                                  const float* rstd, const ComputeType* dy,
                                  ComputeType* dx, ComputeType* dresidual) {
  if (params.rows <= 0 || params.dim <= 0) return;
  const int block = cuda_kernels::BlockSizeForDim(params.dim);
  cuda_backend::Launch(kernels::RmsNormFusedResidualBackwardKernel,
                       dim3(params.rows), dim3(block), 0, params, residual,
                       rstd, dy, dx, dresidual);
}

}  // namespace cuda_kernels
}  // namespace nanochat
