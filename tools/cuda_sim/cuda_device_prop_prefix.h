#ifndef NANOCHAT_TOOLS_CUDA_SIM_CUDA_DEVICE_PROP_PREFIX_H_
#define NANOCHAT_TOOLS_CUDA_SIM_CUDA_DEVICE_PROP_PREFIX_H_

// The `cudaDeviceProp` prefix the API interposer fabricates
// (tools/cuda_sim/cuda_sim_abi.h, docs/simulator.md section 5.1).
//
// The mock must place its fields at the *vendor* offsets: the binary under test
// was compiled against the real `cudaDeviceProp`, and `GetCaps()` in
// backends/cuda/device.cu reads `major`, `minor`, and `totalGlobalMem` straight
// out of it. The fields the CUDA 12 layout has between `name` and
// `totalGlobalMem` -- `cudaUUID_t uuid` (16 bytes), `char luid[8]`, and
// `unsigned int luidDeviceNodeMask` -- are part of that layout. Omitting them,
// as an earlier revision did, shifts every later field and the device reports
// compute capability 0.
//
// This header stays free of the runtime typedefs (`cudaError_t`, `dim3`,
// `cudaStream_t`, ...) that `cuda_sim_abi.h` declares, so a test can include
// both it and the real `<driver_types.h>` and `static_assert` that the offsets
// agree. See //tests:cuda_device_prop_abi_test for that cross-check.

#include <cstddef>

// Mirrors `cudaUUID_t` (`struct CUuuid_st { char bytes[16]; }` in
// <driver_types.h>). Named apart so it can sit beside the real definition in a
// translation unit that includes both.
struct cudaUuidAbi {
  char bytes[16];
};

// The vendor layout through `minor`. The tail of the caller's larger
// `cudaDeviceProp` is left untouched.
struct cudaDevicePropPrefix {
  char name[256];
  cudaUuidAbi uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  std::size_t totalGlobalMem;
  std::size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  std::size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  std::size_t totalConstMem;
  int major;
  int minor;
};

#endif  // NANOCHAT_TOOLS_CUDA_SIM_CUDA_DEVICE_PROP_PREFIX_H_
