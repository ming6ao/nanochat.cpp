#ifndef NANOCHAT_DEVICE_PROFILE_H_
#define NANOCHAT_DEVICE_PROFILE_H_

// Device capability table for the simulator (docs/simulator.md section 7).
//
// The table holds one entry per target accelerator. The simulator's reference
// engine (the CPU backend under `--config=sim`), the API interposer, and the
// emulation engine all read their capability data from here, so a simulated
// device is described in exactly one place.
//
// The rows are not maintained here. Every entry derives from
// `backends/<name>/caps.json`, and `src/device_profile.cc` holds the
// derivation: a device bring-up adds one file under `backends/` and no row, in
// this header or in the table it declares. The header declares the shape of a
// profile and the lookups over the derived list, and it holds no
// device-specific row and no device-specific branch, which keeps the "add
// hardware" evolution rule true for the table as well as for the seam
// (docs/model-and-device-bringup.md section 18).
//
// This header is additive and architect-owned. It is the one rate and
// capability source: `PeakFlopsFor` is the single accessor over a row's rates,
// and `PeakFlopsForDevice` in `src/train.cc` reads the row's
// `reference_peak_flops` rather than holding a table of its own
// (docs/performance-model.md section 6). `//tests:device_profile_test` keeps
// the two purposes from drifting.
//
// The header includes only `nanochat/tensor.h`, so a backend may include it
// without reaching above the kernel seam (docs/build.md, dependency
// invariants).

#include <cstddef>
#include <cstdint>
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
  bool supports_bf16 = false;
  // The rate the *reference* recipe's Model FLOPs Utilization line divides by,
  // and the precision it is quoted at. It is a separate field from the three
  // rates above because the reference counts at one precision per card while
  // the estimator assigns a precision per operation. Declaring it here keeps a
  // card's reference rate in the same file as its other rates, so
  // `PeakFlopsForDevice` reads one number instead of holding a second table
  // (docs/performance-model.md section 6).
  double reference_peak_flops = 0.0;
  BuildPrecision reference_precision = BuildPrecision::kFp32;
  // The E4M3/E5M2 tensor-core path. Ada reports 8.9 and Hopper reports 9.0;
  // Ampere and older have no fp8 mode (docs/fp8-training.md section 5.3).
  bool supports_fp8 = false;

  // Dense (non-sparse) peak rates. The fp32 rate is the CUDA-core rate; the
  // bf16 rate is the tensor-core rate on a device that has it. A rate of 0
  // means the device has no path at that precision. The fp8 rate is the
  // E4M3/E5M2 tensor-core rate (docs/performance-model.md section 6).
  double peak_fp32_flops = 0.0;
  double peak_bf16_flops = 0.0;
  double peak_fp8_flops = 0.0;

  // The peak HBM bandwidth, in bytes per second. The estimator divides the
  // operand bytes by this value (docs/performance-model.md section 6).
  double peak_bandwidth_bytes_per_second = 0.0;

  int warp_size = 32;
  int max_threads_per_block = 1024;
  // The opt-in maximum, the value a launch must request through
  // `cudaFuncSetAttribute` before exceeding the 48 KB default.
  std::size_t max_shared_memory_per_block = 48 * 1024;

  // The nominal cost of one kernel launch, in microseconds. The launch cost
  // is a device property, not a per-shape value
  // (docs/performance-model.md section 7).
  double launch_us = 5.0;
};

// The profile table. The returned pointers are stable for the life of the
// process and the entries are ordered by descending capability.
const std::vector<const DeviceProfile*>& AllDeviceProfiles();

// Looks up a profile by name. The comparison is on the lowercased, whitespace-
// stripped name (and on the `device_name` spelling a run reports), so a vendor
// spelling and the join key resolve to the same entry. Returns null when
// nothing matches.
const DeviceProfile* FindDeviceProfile(const std::string& name);

// The capability mask a declaration carries: the bits from
// `backends/<name>/caps.json`, `devices[].features`, as `kFeature*` values.
// Returns 0 when no declaration names the device, or when the declaration file
// is unreachable (a bare binary outside the runfiles tree). It is the declared
// promise, not a field of `DeviceProfile`: a reference backend declares bits
// without contributing a profile row, so this lookup answers for a device whose
// profile the table does not hold.
std::uint64_t DeclaredFeatureMask(const std::string& device_name);

// Projects a profile onto the vendor-free capability struct the kernel seam
// reports. The result describes a *device*: `is_device` and `has_cublas` are
// true, and the precision flags come from the profile. It changes capabilities,
// never numerics (docs/simulator.md section 4.4).
Caps CapsFromProfile(const DeviceProfile& profile);

// The dense peak FLOP/s `profile` declares at `precision`, or 0 when the device
// has no path at that precision. It is the one rate accessor in the tree: the
// benchmark's Model FLOPs Utilization share, the training log's Model FLOPs
// Utilization line, and the roofline estimator all reach a card's rate through
// it, so one card has one number per precision (docs/performance-model.md
// section 6). The precision type is `BuildPrecision` from
// `nanochat/tensor.h`, which is the tree's single precision vocabulary;
// `ComputePrecision` in `nanochat/mfu.h` is an alias for it.
double PeakFlopsFor(const DeviceProfile& profile, BuildPrecision precision);

// The same rate for a device named by `name`, resolved through
// `FindDeviceProfile`. Returns 0 when no declaration names the device.
double PeakFlopsFor(const std::string& name, BuildPrecision precision);

// The comma-and-space separated list of declared device names, in the table's
// order. It is the list a command prints with an unknown device, and the list a
// command validates against, so the accepted set is the declared set and no
// program holds a second copy of it (docs/simulator.md section 7).
std::string DeviceNameList();

// The message a command prints before it exits with status 2 when no
// declaration names `device_name`. It names the rejected string and lists
// `DeviceNameList()`, so the roofline report, the simulator, and the benchmark
// tool refuse an unknown device with one text and one status.
std::string DeviceNameError(const std::string& device_name);

}  // namespace nanochat

#endif  // NANOCHAT_DEVICE_PROFILE_H_
