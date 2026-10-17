// backends/cuda/kernels/fp8.cu -- the fp8 quantization kernel
// (docs/fp8-training.md section 4.2).
//
// The dynamic scale is computed in the same step: the kernel measures the amax
// of the tensor, derives the scale, and then the cast kernel writes the fp8
// operands. The device scalar `dequant` stays on the device; the host never
// reads the amax, so no synchronization occurs.
//
// The cast uses the CUDA conversion intrinsic. It follows the same
// round-to-nearest-even rule as `nanochat/fp8.h`, which the CPU reference and
// the emulation engine share (docs/correctness-gates.md section 3.2).

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/fp8.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {
namespace {

constexpr int kFp8Block = 256;

// The magnitude of a float as an order-preserving unsigned value, so one
// atomic or one reduction can find the maximum of the absolute values.
__device__ unsigned int OrderedMagnitude(float value) {
  return __float_as_uint(fabsf(value)) | 0x80000000u;
}

// One block reduces the whole tensor, derives the scale, and writes the
// dequantization value. The reduction is over one block, so no scratch buffer
// and no second launch are necessary.
__global__ void AmaxScaleKernel(const ComputeType* src, int count,
                                float amax_floor, float fp8_max,
                                float* dequant) {
  unsigned int local = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const unsigned int bits =
        OrderedMagnitude(cuda_kernels::AsFloatDev(src[i]));
    local = bits > local ? bits : local;
  }
  __shared__ unsigned int shared[kFp8Block];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = kFp8Block / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      shared[threadIdx.x] = shared[threadIdx.x + stride] > shared[threadIdx.x]
                                ? shared[threadIdx.x + stride]
                                : shared[threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float amax = __uint_as_float(shared[0] & 0x7FFFFFFFu);
    const float magnitude = amax > amax_floor ? amax : amax_floor;
    dequant[0] = magnitude / fp8_max;
  }
}

__global__ void CastKernel(const ComputeType* src, std::uint8_t* dst, int count,
                           const float* dequant, int e4m3) {
  const float scale = 1.0f / dequant[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const float scaled = cuda_kernels::AsFloatDev(src[i]) * scale;
    // The intrinsic saturates and rounds to nearest, ties to even.
    const __nv_fp8_storage_t bits =
        e4m3 ? __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3)
             : __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E5M2);
    dst[i] = static_cast<std::uint8_t>(bits);
  }
}

__global__ void SetOneKernel(float* value) { value[0] = 1.0f; }

}  // namespace

void QuantizeFp8(const QuantizeFp8Params& params, const ComputeType* src,
                 std::uint8_t* dst, float* dequant) {
  const int count = params.count;
  if (count <= 0 || src == nullptr || dst == nullptr) {
    if (dequant != nullptr) {
      cuda_backend::Launch(SetOneKernel, dim3(1), dim3(1), 0, dequant);
    }
    return;
  }
  const bool e4m3 = params.format == Fp8Format::kE4M3;
  const float fp8_max = e4m3 ? kFp8E4M3Max : kFp8E5M2Max;
  if (dequant != nullptr) {
    cuda_backend::Launch(AmaxScaleKernel, dim3(1), dim3(kFp8Block), 0, src,
                         count, params.amax_floor, fp8_max, dequant);
  }
  const int blocks = (count + kFp8Block - 1) / kFp8Block;
  const int grid = blocks < 4096 ? blocks : 4096;
  cuda_backend::Launch(CastKernel, dim3(grid), dim3(kFp8Block), 0, src, dst,
                       count, dequant, e4m3 ? 1 : 0);
}

}  // namespace kernels
}  // namespace nanochat
