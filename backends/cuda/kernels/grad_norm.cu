// CUDA GradNorm family: the L2 norm of a set of gradient buffers, written to
// `out_norm`, plus the clip scale the caller folds into every update. Two
// kernels: a block-partial reduction into a device accumulator, and a
// one-thread finalize that publishes the norm and the scale. Nothing is
// rewritten, so there is no third scale pass and no staging buffer.
//
// `grads`/`counts` describe `num_spans` disjoint buffers. The partial kernel is
// launched with `blockIdx.z` selecting the span, so the whole set is one
// launch.
//
// The accumulator is a `__device__` global rather than a fresh allocation so
// the entry point needs no scratch buffer and no host round trip. The backend
// runs everything on one stream, so the launches are ordered and a single
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

// Sum of squares across every span, accumulated in double to match the CPU
// reference, then a published norm and clip scale.
__device__ double g_grad_norm_sum = 0.0;

// Accumulator for ScalarDot. A separate `__device__` global, reset by its own
// finalize kernel, so the two reductions cannot interfere.
__device__ double g_scalar_dot_sum = 0.0;

// One z index per span. A span narrower than the x grid leaves the surplus
// blocks with no elements; they still reach the block reduction and contribute
// zero, which keeps the kernel free of a divergent early exit.
__global__ void GradNormPartialKernel(const float* const* __restrict__ grads,
                                      const int* __restrict__ counts) {
  const int span = blockIdx.z;
  const int n = counts[span];
  const float* span_grads = grads[span];
  double local = 0.0;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double v = span_grads[i];
    local += v * v;
  }
  const double block_sum = cuda_kernels::BlockReduceSum(local);
  if (threadIdx.x == 0) atomicAdd(&g_grad_norm_sum, block_sum);
}

__global__ void GradNormFinalizeKernel(float clip, float* __restrict__ norm,
                                       float* __restrict__ scale) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float value = static_cast<float>(sqrt(g_grad_norm_sum));
  if (norm != nullptr) norm[0] = value;
  if (scale != nullptr) {
    scale[0] = (clip > 0.0f && value > clip) ? (clip / value) : 1.0f;
  }
  // Reset for the next invocation; the stream ordering makes this safe.
  g_grad_norm_sum = 0.0;
}

// Dot product of two equal-length buffers, accumulated in double across the
// grid, then folded into the single-element `out` by the finalize kernel.
__global__ void ScalarDotPartialKernel(int n, const ComputeType* __restrict__ a,
                                       const ComputeType* __restrict__ b) {
  double local = 0.0;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    local += static_cast<double>(AsFloatDev(a[i])) *
             static_cast<double>(AsFloatDev(b[i]));
  }
  const double block_sum = cuda_kernels::BlockReduceSum(local);
  if (threadIdx.x == 0) atomicAdd(&g_scalar_dot_sum, block_sum);
}

__global__ void ScalarDotFinalizeKernel(float* __restrict__ out, float scale,
                                        bool accumulate) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float value = scale * static_cast<float>(g_scalar_dot_sum);
  out[0] = accumulate ? out[0] + value : value;
  // Reset for the next invocation; the stream ordering makes this safe.
  g_scalar_dot_sum = 0.0;
}

}  // namespace

void GradNorm(const float* const* grads, const int* counts, int num_spans,
              float clip, float* out_norm, float* out_scale) {
  if (grads == nullptr || counts == nullptr || num_spans <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  // The span count can be in the hundreds (one per matrix parameter), and each
  // span takes a z index. Keep the whole launch to a few thousand blocks: the
  // reduction is memory-bound, so a handful of blocks per span already
  // saturates the device, and a block count that tracks the span count would
  // only add launch and atomics overhead.
  constexpr int kBlockBudget = 4096;
  // `gridDim.z` is 16-bit, so a span set larger than that is reduced in chunks,
  // each accumulating into the same device accumulator.
  constexpr int kMaxSpanChunk = 32768;

  for (int start = 0; start < num_spans; start += kMaxSpanChunk) {
    const int chunk =
        num_spans - start < kMaxSpanChunk ? num_spans - start : kMaxSpanChunk;
    // `counts` is device memory, so the grid cannot be sized from it here. The
    // budget is spread over the chunk instead: a short span set gets the full
    // grid, and a long one still covers the whole device.
    int grid = kBlockBudget / chunk;
    if (grid > kMaxBlocks) grid = kMaxBlocks;
    if (grid < 1) grid = 1;

    cuda_backend::Launch(GradNormPartialKernel, dim3(grid, 1, chunk),
                         dim3(kThreads), 0, grads + start, counts + start);
  }
  cuda_backend::Launch(GradNormFinalizeKernel, dim3(1), dim3(1), 0, clip,
                       out_norm, out_scale);
}

void ScalarDot(const ComputeType* a, const ComputeType* b, int n, float* out,
               float scale, bool accumulate) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  const int blocks = (n + kThreads - 1) / kThreads;
  const int grid = blocks < kMaxBlocks ? blocks : kMaxBlocks;

  cuda_backend::Launch(ScalarDotPartialKernel, dim3(grid), dim3(kThreads), 0, n,
                       a, b);
  cuda_backend::Launch(ScalarDotFinalizeKernel, dim3(1), dim3(1), 0, out, scale,
                       accumulate);
}

}  // namespace kernels
}  // namespace nanochat
