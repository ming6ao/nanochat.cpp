// CUDA device runtime: allocation, copies, memset, synchronization, and the
// capability report. This is the L0 backend layer (DESIGN.md section 2): it
// implements the device/memory half of nanochat/kernels.h and knows nothing
// about the model. See docs/build.md and docs/kernels.md.

#include "backends/cuda/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backends/cuda/caps_names.h"
#include "nanochat/device_profile.h"
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

namespace {

// The fp8 path this backend ships: `QuantizeFp8` (kernels/fp8.cu) and
// `GemmScaledFp8` (gemm.cu's cublasLt branch) are compiled into every CUDA
// build, unconditionally. `spec/capability_bits.md` section 4: a device that
// offers no fp8 path declares neither fp8 bit, so the pair below can only be
// offered when this is true. It is a build fact and not a device fact, which is
// why it lives here and not in `cudaDeviceProp`.
constexpr bool kFp8PathBuilt = true;

// The four precision bits. They are the only bits whose truth the silicon
// decides, so they are the only ones `GetCaps()` computes rather than copies
// from the declaration; the set is named once so the mask arithmetic below
// cannot forget one of them.
constexpr std::uint64_t kPrecisionBits =
    kFeatureFp32 | kFeatureBf16 | kFeatureFp8Storage | kFeatureFp8Gemm;

}  // namespace

std::uint64_t DeclaredFeatureMask(const char* device_name) {
  if (device_name == nullptr || *device_name == '\0') return 0;
  // The lookup normalizes case and whitespace, so the vendor spelling a driver
  // reports ("NVIDIA GeForce GTX 1080 Ti") and the profile's join key
  // ("gtx1080ti") resolve to one entry. The projection is the same one the
  // simulator uses, so the two tiers cannot disagree about a profile.
  const DeviceProfile* profile = FindDeviceProfile(device_name);
  if (profile == nullptr) return 0;
  return CapsFromProfile(*profile).feature_mask;
}

std::uint64_t PrecisionFeatureBits(int major, int minor) {
  std::uint64_t bits = kFeatureFp32;
  // bf16 arithmetic needs Ampere's bf16 units.
  if (major >= 8) bits |= kFeatureBf16;
  // The fp8 E4M3/E5M2 tensor core arrives with Ada (sm_89); Hopper reports
  // 9.0 (docs/fp8-training.md section 5.3). `fp8_gemm` implies `fp8_storage`,
  // so the pair is offered together or not at all.
  if ((major > 8 || (major == 8 && minor >= 9)) && kFp8PathBuilt) {
    bits |= kFeatureFp8Storage | kFeatureFp8Gemm;
  }
  return bits;
}

}  // namespace cuda_backend

namespace kernels {

void* Alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  // Round the request up to a four-byte boundary. `AtomicAddDev` on a bf16
  // target updates the aligned 32-bit word that contains the element, because
  // there is no 16-bit atomic compare-and-swap. That word can extend two bytes
  // past the last element of a buffer with an odd element count, so the
  // allocation must cover it. The rounding changes no kernel's indexing and
  // costs at most three bytes.
  const std::size_t rounded = (bytes + 3u) & ~static_cast<std::size_t>(3u);
  void* ptr = nullptr;
  const cudaError_t status = cudaMalloc(&ptr, rounded);
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
  // A fact about the part, not a promise the backend makes: tensor cores
  // arrive with Volta (sm_70) and the flag stays a boolean
  // (spec/capability_bits.md section 5, "a boolean that is a fact stays a
  // boolean").
  caps.has_tensor_cores = props.major >= 7;

  // Per device thread, so two device threads do not overwrite one name buffer
  // (docs/client-runtime-design.md section 8.1). The spelling is the driver's
  // and is kept verbatim: it is the join key `FindDeviceProfile` normalizes.
  thread_local char device_name[256];
  std::snprintf(device_name, sizeof(device_name), "%s", props.name);
  caps.device_name = device_name;

  // ---- the capability mask ------------------------------------------------
  //
  // Two inputs, and no third one. The declaration supplies the architecture,
  // tier, stateful, and override bits (`backends/cuda/caps.json`, through the
  // derived profile entry), and the silicon supplies the four precision bits.
  //
  // The two are combined as `(declared without its precision bits) | (what the
  // silicon reports)` rather than as a union of everything, because the two
  // sides have different authority. A declaration may not *grant* a precision
  // bit: a profile written for the fastest member of a family must not make a
  // Pascal card claim bf16 (spec/capability_bits.md section 4). And a
  // declaration may not *withhold* one either, because this backend's
  // correctness does not depend on the caps.json derivation having run: the
  // silicon's precision is a fact `cudaDeviceProp` answers on its own.
  const std::uint64_t declared =
      cuda_backend::DeclaredFeatureMask(caps.device_name);
  const std::uint64_t silicon =
      cuda_backend::PrecisionFeatureBits(props.major, props.minor);
  const std::uint64_t mask =
      (declared & ~cuda_backend::kPrecisionBits) | silicon;

  // Fail loudly at the first capability report when a bit arrives that no
  // vocabulary name maps to. A silent unknown bit is the run-time lie
  // spec/capability_bits.md section 2 warns about: the planner would read a
  // promise nobody can look up, and the user would be told the wrong
  // implementation ran.
  cuda_backend::RequireNamedBits(mask, "caps.json / the device profile");
  caps.feature_mask = mask;

  // The three booleans and `Supports(DType)` are one answer spelled twice, so
  // they are read off the mask rather than recomputed from the part: a caller
  // that asks `supports_bf16` and a caller that asks `HasFeature(kFeatureBf16)`
  // cannot disagree. `supports_fp8` is the tensor-core path, which is the pair
  // (`fp8_gemm` implies `fp8_storage`, spec/capability_bits.md bit 8), so it
  // asks for both.
  caps.supports_fp32 = (mask & kFeatureFp32) != 0;
  caps.supports_bf16 = (mask & kFeatureBf16) != 0;
  caps.supports_fp8 = (mask & (kFeatureFp8Storage | kFeatureFp8Gemm)) ==
                      (kFeatureFp8Storage | kFeatureFp8Gemm);
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
