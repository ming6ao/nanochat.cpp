// P0 CUDA/Pascal toolchain spike. One kernel, compiled for the default sm_61
// and for sm_75 with --config=sm_75. It exists only to prove that the local
// CUDA toolkit builds, links, and runs from Bazel on Pascal; the real kernel
// families land in P1 under backends/cuda/kernels.

#include "pascal_spike.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace nanochat {
namespace dev {

namespace {

__global__ void ScaleAddKernel(const float* a, const float* b, float alpha,
                               float beta, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * a[i] + beta * b[i];
  }
}

}  // namespace

int SpikeScaleAddHost(const float* a, const float* b, float alpha, float beta,
                      float* out, int n) {
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_out = nullptr;

  cudaError_t status = cudaMalloc(&dev_a, bytes);
  if (status == cudaSuccess) status = cudaMalloc(&dev_b, bytes);
  if (status == cudaSuccess) status = cudaMalloc(&dev_out, bytes);
  if (status == cudaSuccess) {
    status = cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    constexpr int kThreads = 128;
    const int blocks = (n + kThreads - 1) / kThreads;
    ScaleAddKernel<<<blocks, kThreads>>>(dev_a, dev_b, alpha, beta, dev_out, n);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaMemcpy(out, dev_out, bytes, cudaMemcpyDeviceToHost);
  }

  if (dev_a != nullptr) cudaFree(dev_a);
  if (dev_b != nullptr) cudaFree(dev_b);
  if (dev_out != nullptr) cudaFree(dev_out);
  return static_cast<int>(status);
}

int SpikeComputeCapability(int* major, int* minor) {
  cudaDeviceProp props{};
  const cudaError_t status = cudaGetDeviceProperties(&props, 0);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *major = props.major;
  *minor = props.minor;
  return 0;
}

}  // namespace dev
}  // namespace nanochat
