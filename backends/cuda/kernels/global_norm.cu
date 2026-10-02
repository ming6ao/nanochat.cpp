// CUDA GlobalNorm family: the global L2 norm of a gradient buffer, written to
// `out_norm`, then a global, in-place clip when the norm exceeds `clip`. Three
// kernels: a block-partial reduction into a device accumulator, a one-thread
// finalize that publishes the norm and the scale, and a grid-stride scale.
//
// The accumulator is a `__device__` global rather than a fresh allocation so
// the entry point needs no scratch buffer and no host round trip. The backend
// runs everything on one stream, so the three launches are ordered and a single
// accumulator is race-free; the finalize kernel resets it for the next call.
//
// See docs/kernels.md.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::ToComputeDev;

// Sum of squares across the whole buffer, accumulated in double to match the
// CPU reference, then a published norm and clip scale.
__device__ double g_global_norm_sum = 0.0;
__device__ float g_global_norm_scale = 1.0f;

__global__ void GlobalNormPartialKernel(int n,
                                        const ComputeType* __restrict__ grads) {
  double local = 0.0;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double v = AsFloatDev(grads[i]);
    local += v * v;
  }
  const double block_sum = cuda_kernels::BlockReduceSum(local);
  if (threadIdx.x == 0) atomicAdd(&g_global_norm_sum, block_sum);
}

__global__ void GlobalNormFinalizeKernel(float clip, float* __restrict__ norm) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float value = static_cast<float>(sqrt(g_global_norm_sum));
  if (norm != nullptr) norm[0] = value;
  g_global_norm_scale = (clip > 0.0f && value > clip) ? (clip / value) : 1.0f;
  // Reset for the next invocation; the stream ordering makes this safe.
  g_global_norm_sum = 0.0;
}

__global__ void GlobalNormScaleKernel(int n, ComputeType* __restrict__ grads) {
  const float scale = g_global_norm_scale;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    grads[i] = ToComputeDev(AsFloatDev(grads[i]) * scale);
  }
}

}  // namespace

void GlobalNorm(int n, float clip, ComputeType* grads, float* out_norm) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  const int blocks = (n + kThreads - 1) / kThreads;
  const int grid = blocks < kMaxBlocks ? blocks : kMaxBlocks;

  cuda_backend::Launch(GlobalNormPartialKernel, dim3(grid), dim3(kThreads), 0,
                       n, grads);
  cuda_backend::Launch(GlobalNormFinalizeKernel, dim3(1), dim3(1), 0, clip,
                       out_norm);
  cuda_backend::Launch(GlobalNormScaleKernel, dim3(grid), dim3(kThreads), 0, n,
                       grads);
}

}  // namespace kernels
}  // namespace nanochat
