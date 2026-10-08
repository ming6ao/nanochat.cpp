#ifndef NANOCHAT_DEVICE_PROFILE_H_
#define NANOCHAT_DEVICE_PROFILE_H_

// Device capability table for the simulator (docs/simulator.md section 7).
//
// The table holds one entry per target accelerator. The simulator's reference
// engine (the CPU backend under `--config=sim`), the API interposer, and the
// emulation engine all read their capability data from here, so a simulated
// device is described in exactly one place.
//
// This header is additive and architect-owned. It stays *beside*
// `PeakFlopsForDevice` in `src/train.cc`: that function remains the authority
// for the MFU string patterns, and this table is the authority for the
// capability fields. The two are kept consistent by
// `//tests:device_profile_test`.
//
// The header includes only `nanochat/tensor.h`, so a backend may include it
// without reaching above the kernel seam (DESIGN.md section 2.1).

#include <cstddef>
#include <string>
#include <vector>

#include "nanochat/tensor.h"

namespace nanochat {

// One accelerator target. `name` is the lowercased join key: the same string
// `PeakFlopsForDevice` matches against its MFU patterns. It points at a string
// literal, so the pointer stays valid for the life of the process (which
// `Caps::device_name` requires).
struct DeviceProfile {
  const char* name = "";
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_memory_bytes = 0;

  bool has_tensor_cores = false;  // fp16/bf16 MMA units
  bool supports_fp16 = false;
  bool supports_bf16 = false;

  // Dense (non-sparse) peak rates. The fp32 rate is the CUDA-core rate; the
  // fp16/bf16 rates are the tensor-core rates on a device that has them. A
  // rate of 0 means the device has no path at that precision.
  double peak_fp32_flops = 0.0;
  double peak_fp16_flops = 0.0;
  double peak_bf16_flops = 0.0;

  int warp_size = 32;
  int max_threads_per_block = 1024;
  // The opt-in maximum, the value a launch must request through
  // `cudaFuncSetAttribute` before exceeding the 48 KB default.
  std::size_t max_shared_memory_per_block = 48 * 1024;
};

// The profile table. The returned pointers are stable for the life of the
// process and the entries are ordered by descending capability.
const std::vector<const DeviceProfile*>& AllDeviceProfiles();

// Looks up a profile by name. The comparison is on the lowercased name (and on
// the `device_name` spelling a run reports), so "H100" and "h100" resolve to
// the same entry. Returns null when nothing matches.
const DeviceProfile* FindDeviceProfile(const std::string& name);

// Projects a profile onto the vendor-free capability struct the kernel seam
// reports. The result describes a *device*: `is_device` and `has_cublas` are
// true, and the precision flags come from the profile. It changes capabilities,
// never numerics (docs/simulator.md section 4.4).
Caps CapsFromProfile(const DeviceProfile& profile);

}  // namespace nanochat

#endif  // NANOCHAT_DEVICE_PROFILE_H_
