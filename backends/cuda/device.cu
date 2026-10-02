// CUDA device runtime: allocation, copies, memset, synchronization, and the
// capability report. This is the L0 backend layer (DESIGN.md section 2): it
// implements the device/memory half of nanochat/kernels.h and knows nothing
// about the model. See docs/backends.md and docs/kernels.md.

#include "device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace cuda_backend {

cudaStream_t Stream() { return nullptr; }

void CheckCuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "fatal: CUDA error in %s: %s\n", what,
               cudaGetErrorString(status));
  std::exit(1);
}

void CheckLastError(const char* what) { CheckCuda(cudaGetLastError(), what); }

}  // namespace cuda_backend

namespace kernels {

void* Alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = nullptr;
  const cudaError_t status = cudaMalloc(&ptr, bytes);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "fatal: cudaMalloc(%zu bytes) failed: %s\n", bytes,
                 cudaGetErrorString(status));
    std::exit(1);
  }
  return ptr;
}

void Free(void* ptr) {
  if (ptr == nullptr) return;
  cuda_backend::CheckCuda(cudaFree(ptr), "cudaFree");
}

void Memcpy(void* dst, const void* src, std::size_t bytes, CopyDir dir) {
  if (bytes == 0) return;
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  switch (dir) {
    case CopyDir::kHostToDevice:
      kind = cudaMemcpyHostToDevice;
      break;
    case CopyDir::kDeviceToHost:
      kind = cudaMemcpyDeviceToHost;
      break;
    case CopyDir::kDeviceToDevice:
      kind = cudaMemcpyDeviceToDevice;
      break;
    case CopyDir::kHostToHost:
      kind = cudaMemcpyHostToHost;
      break;
  }
  // Synchronous on purpose: the seam exposes no events, so a host read after a
  // device-to-host copy must already see the data.
  cuda_backend::CheckCuda(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
}

void Memset(void* ptr, int value, std::size_t bytes) {
  if (bytes == 0) return;
  cuda_backend::CheckCuda(cudaMemset(ptr, value, bytes), "cudaMemset");
}

void Synchronize() {
  cuda_backend::CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

Caps GetCaps() {
  Caps caps;
  caps.is_device = true;
  caps.has_cublas = true;
  caps.device_name = "cuda (no device)";

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return caps;

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return caps;

  caps.device_index = device;
  caps.compute_major = props.major;
  caps.compute_minor = props.minor;
  caps.total_memory_bytes = static_cast<std::size_t>(props.totalGlobalMem);
  // Tensor cores arrive with Volta (sm_70) and are the reason fp16 is wired for
  // the sm_75 build config; GEMM selects the tensor-op algorithm only when they
  // are present.
  caps.has_tensor_cores = props.major >= 7;
  caps.supports_fp32 = true;
  // __half storage and cuBLAS half GEMM need sm_53 or newer. Compute in fp32.
  caps.supports_fp16 =
      props.major > 5 || (props.major == 5 && props.minor >= 3);

  static char device_name[256];
  std::snprintf(device_name, sizeof(device_name), "%s", props.name);
  caps.device_name = device_name;
  return caps;
}

}  // namespace kernels
}  // namespace nanochat
