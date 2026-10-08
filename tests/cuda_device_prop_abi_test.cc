// tests/cuda_device_prop_abi_test.cc -- the fabricated `cudaDeviceProp` must
// sit at the vendor field offsets (docs/simulator.md section 5.1).
//
// The interposer writes a `cudaDevicePropPrefix` into the caller's real
// `cudaDeviceProp` and leaves the tail untouched. If the prefix's layout drifts
// from the toolkit's, every later field shifts: `GetCaps()` in
// `backends/cuda/device.cu` would read the wrong bytes and report compute
// capability 0. The check is a compile-time cross-check against the real
// `<driver_types.h>`, so drift breaks the build instead of a device query.
//
// It needs the CUDA toolkit headers but no device, driver, or broker: the
// asserts are all `constexpr`, and `main` only re-prints the values.

#include <driver_types.h>

#include <cstddef>
#include <cstdio>

#include "nanochat/sandbox.h"
#include "tools/cuda_sim/cuda_device_prop_prefix.h"

// The prefix mirrors the vendor struct through `minor`, so each field's offset
// must match the real `cudaDeviceProp` exactly.
static_assert(offsetof(cudaDevicePropPrefix, name) ==
                  offsetof(cudaDeviceProp, name),
              "cudaDevicePropPrefix.name must match cudaDeviceProp.name");
static_assert(offsetof(cudaDevicePropPrefix, uuid) ==
                  offsetof(cudaDeviceProp, uuid),
              "cudaDevicePropPrefix.uuid must match cudaDeviceProp.uuid");
static_assert(offsetof(cudaDevicePropPrefix, luid) ==
                  offsetof(cudaDeviceProp, luid),
              "cudaDevicePropPrefix.luid must match cudaDeviceProp.luid");
static_assert(offsetof(cudaDevicePropPrefix, luidDeviceNodeMask) ==
                  offsetof(cudaDeviceProp, luidDeviceNodeMask),
              "cudaDevicePropPrefix.luidDeviceNodeMask must match");
static_assert(offsetof(cudaDevicePropPrefix, totalGlobalMem) ==
                  offsetof(cudaDeviceProp, totalGlobalMem),
              "cudaDevicePropPrefix.totalGlobalMem must match");
static_assert(offsetof(cudaDevicePropPrefix, sharedMemPerBlock) ==
                  offsetof(cudaDeviceProp, sharedMemPerBlock),
              "cudaDevicePropPrefix.sharedMemPerBlock must match");
static_assert(offsetof(cudaDevicePropPrefix, regsPerBlock) ==
                  offsetof(cudaDeviceProp, regsPerBlock),
              "cudaDevicePropPrefix.regsPerBlock must match");
static_assert(offsetof(cudaDevicePropPrefix, warpSize) ==
                  offsetof(cudaDeviceProp, warpSize),
              "cudaDevicePropPrefix.warpSize must match");
static_assert(offsetof(cudaDevicePropPrefix, memPitch) ==
                  offsetof(cudaDeviceProp, memPitch),
              "cudaDevicePropPrefix.memPitch must match");
static_assert(offsetof(cudaDevicePropPrefix, maxThreadsPerBlock) ==
                  offsetof(cudaDeviceProp, maxThreadsPerBlock),
              "cudaDevicePropPrefix.maxThreadsPerBlock must match");
static_assert(offsetof(cudaDevicePropPrefix, maxThreadsDim) ==
                  offsetof(cudaDeviceProp, maxThreadsDim),
              "cudaDevicePropPrefix.maxThreadsDim must match");
static_assert(offsetof(cudaDevicePropPrefix, maxGridSize) ==
                  offsetof(cudaDeviceProp, maxGridSize),
              "cudaDevicePropPrefix.maxGridSize must match");
static_assert(offsetof(cudaDevicePropPrefix, clockRate) ==
                  offsetof(cudaDeviceProp, clockRate),
              "cudaDevicePropPrefix.clockRate must match");
static_assert(offsetof(cudaDevicePropPrefix, totalConstMem) ==
                  offsetof(cudaDeviceProp, totalConstMem),
              "cudaDevicePropPrefix.totalConstMem must match");
static_assert(offsetof(cudaDevicePropPrefix, major) ==
                  offsetof(cudaDeviceProp, major),
              "cudaDevicePropPrefix.major must match cudaDeviceProp.major");
static_assert(offsetof(cudaDevicePropPrefix, minor) ==
                  offsetof(cudaDeviceProp, minor),
              "cudaDevicePropPrefix.minor must match cudaDeviceProp.minor");

// The prefix is the head of the vendor struct, never larger than it.
static_assert(sizeof(cudaDevicePropPrefix) <= sizeof(cudaDeviceProp),
              "cudaDevicePropPrefix must fit inside cudaDeviceProp");
static_assert(sizeof(cudaUuidAbi) == sizeof(cudaUUID_t),
              "cudaUuidAbi must match cudaUUID_t");

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf(
      "cuda device prop abi: sizeof(prefix)=%zu sizeof(cudaDeviceProp)=%zu "
      "totalGlobalMem@%zu major@%zu minor@%zu\n",
      sizeof(cudaDevicePropPrefix), sizeof(cudaDeviceProp),
      offsetof(cudaDevicePropPrefix, totalGlobalMem),
      offsetof(cudaDevicePropPrefix, major),
      offsetof(cudaDevicePropPrefix, minor));
  return 0;
}
