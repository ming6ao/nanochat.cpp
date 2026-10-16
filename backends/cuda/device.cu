// CUDA device runtime: allocation, copies, memset, synchronization, and the
// capability report. This is the L0 backend layer (DESIGN.md section 2): it
// implements the device/memory half of nanochat/kernels.h and knows nothing
// about the model. See docs/build.md and docs/kernels.md.

#include "backends/cuda/device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/mesh.h"
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
  // Tensor cores arrive with Volta (sm_70).
  caps.has_tensor_cores = props.major >= 7;
  caps.supports_fp32 = true;

  // Per device thread, so two device threads do not overwrite one name buffer
  // (docs/client-runtime-design.md section 8.1).
  thread_local char device_name[256];
  std::snprintf(device_name, sizeof(device_name), "%s", props.name);
  caps.device_name = device_name;
  return caps;
}

}  // namespace kernels

std::vector<DeviceInfo> EnumerateDevices() {
  // The CUDA runtime enumerates the local devices. `props.name` is a local
  // buffer, so the names are copied into a static list that keeps the pointers
  // valid for the life of the process (as `Caps::device_name` requires).
  std::vector<DeviceInfo> devices;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return devices;

  static std::vector<std::string> names;
  if (names.empty()) {
    names.reserve(static_cast<std::size_t>(count));
  }
  devices.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, i) != cudaSuccess) continue;
    if (names.size() <= static_cast<std::size_t>(i)) {
      names.emplace_back(props.name);
    }
    DeviceInfo info;
    info.ordinal = i;
    info.index = i;
    info.name = names[static_cast<std::size_t>(i)].c_str();
    info.total_memory_bytes = static_cast<std::size_t>(props.totalGlobalMem);
    info.compute_major = props.major;
    info.compute_minor = props.minor;
    info.is_device = true;
    info.has_tensor_cores = props.major >= 7;
    devices.push_back(info);
  }
  return devices;
}

void SelectDevice(int index) {
  cuda_backend::CheckCuda(cudaSetDevice(index), "cudaSetDevice");
}

}  // namespace nanochat
