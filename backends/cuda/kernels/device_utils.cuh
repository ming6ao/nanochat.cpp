#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_

// Shared device-side helpers for the row and elementwise kernel families
// (RmsNorm, QkPrep, Pointwise, GlobalNorm). They remove duplication *without*
// merging kernel boundaries: each .cu still owns its own entry points, and the
// helpers below are header-only so they inline into every caller.
//
// The arithmetic mirrors backends/cpu/kernels.cc exactly: storage is
// ComputeType (float or Fp16), the working type is always float, and the
// reduction runs in float. The CPU reference accumulates in double; the device
// uses float, which is well inside the fp32 oracle tolerance (docs/testing.md).
//
// See docs/kernels.md ("Shared device helpers") and docs/backends.md.

#include <cuda_runtime.h>

#include <cstddef>

#include "nanochat/tensor.h"
#if defined(NANOCHAT_PRECISION_FP16)
#include <cuda_fp16.h>
#endif

namespace nanochat {
namespace cuda_kernels {

// Largest block the block-wide reduction below supports. A block declares a
// shared array of this size, so every launch must stay at or below it.
constexpr int kMaxBlockThreads = 1024;

// Storage <-> float at the device boundary. Fp16 is bit-compatible with
// __half, so the conversion goes through the CUDA intrinsics rather than a
// reinterpret_cast.
#if defined(NANOCHAT_PRECISION_FP16)
__device__ __forceinline__ float AsFloatDev(ComputeType v) {
  return __half2float(__ushort_as_half(v.bits));
}

__device__ __forceinline__ ComputeType ToComputeDev(float v) {
  ComputeType out;
  out.bits = __half_as_ushort(__float2half_rn(v));
  return out;
}
#else
__device__ __forceinline__ float AsFloatDev(ComputeType v) { return v; }

__device__ __forceinline__ ComputeType ToComputeDev(float v) { return v; }
#endif

// Block-wide sum over `val`. Every thread in the block must call it, and the
// result is broadcast back to every thread through shared memory. `blockDim.x`
// must be a power of two and at most kMaxBlockThreads.
template <typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
  __shared__ T shared[kMaxBlockThreads];
  const int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }
  return shared[0];
}

// Sum of squares of one row, reduced across the calling block. Used by
// RmsNorm and QkPrep to recover the normalization statistic.
__device__ __forceinline__ float RowReduceSumSq(const ComputeType* row,
                                                int dim) {
  float sum = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = AsFloatDev(row[d]);
    sum += v * v;
  }
  return BlockReduceSum(sum);
}

// Dot product of two rows, reduced across the calling block. Used by the
// RmsNorm and QkPrep backward passes.
__device__ __forceinline__ float RowReduceDot(const ComputeType* x,
                                              const ComputeType* y, int dim) {
  float dot = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    dot += AsFloatDev(x[d]) * AsFloatDev(y[d]);
  }
  return BlockReduceSum(dot);
}

// 1 / sqrt(mean(square) + eps), the RMSNorm scale for one row.
__device__ __forceinline__ float RsqrtScale(float sum_sq, int dim, float eps) {
  return rsqrtf(sum_sq / static_cast<float>(dim) + eps);
}

// In-place RoPE rotation of one (x1, x2) pair. The sign convention matches the
// CPU reference: (x1*c + x2*s, -x1*s + x2*c).
__device__ __forceinline__ void RotatePairInPlace(float& x1, float& x2,
                                                  float c, float s) {
  const float t1 = x1 * c + x2 * s;
  const float t2 = -x1 * s + x2 * c;
  x1 = t1;
  x2 = t2;
}

// Host-side power-of-two block size for a row of `dim` elements. Rows shorter
// than a warp still get a full warp so BlockReduceSum stays valid; long rows
// cap the block and stride.
inline int BlockSizeForDim(int dim) {
  int block = 32;
  while (block < dim && block < 256) block <<= 1;
  return block;
}

}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_
