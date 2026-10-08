// The device capability table for the simulator (docs/simulator.md section 7).
//
// The table is the single source of truth for a simulated device's capability
// fields; `PeakFlopsForDevice` in `src/train.cc` stays the authority for the
// MFU string patterns. `//tests:device_profile_test` holds the two consistent.
//
// The unit here is deliberately a leaf: it depends only on
// `nanochat/tensor.h`, so the CPU reference backend can link it (to answer
// `GetCaps()` under the simulator) without gaining a dependency on the
// workflow layer.

#include "nanochat/device_profile.h"

#include <cctype>
#include <string>

namespace nanochat {
namespace {

// One entry per target, ordered by descending capability. Rates are dense
// (non-sparse) and in FLOP/s. See docs/simulator.md section 7.
const DeviceProfile kProfiles[] = {
    // NVIDIA Hopper.
    {"h100", 9, 0, 80ull * 1024 * 1024 * 1024, true, true, true, 66.9e12,
     989e12, 989e12, 32, 1024, 227 * 1024},
    {"h200", 9, 0, 141ull * 1024 * 1024 * 1024, true, true, true, 67.0e12,
     989e12, 989e12, 32, 1024, 227 * 1024},
    // NVIDIA Ampere data center.
    {"a100", 8, 0, 80ull * 1024 * 1024 * 1024, true, true, true, 19.5e12,
     312e12, 312e12, 32, 1024, 163 * 1024},
    // NVIDIA Turing.
    {"t4", 7, 5, 16ull * 1024 * 1024 * 1024, true, true, false, 8.1e12, 65e12,
     0.0, 32, 1024, 64 * 1024},
    // Pascal (the development host). No tensor cores; bf16 has no path at all,
    // and fp16 is a storage format at a fractional CUDA-core rate.
    {"gtx1080ti", 6, 1, 11ull * 1024 * 1024 * 1024, false, true, false,
     11.34e12, 0.177e12, 0.0, 32, 1024, 48 * 1024},
};

std::string ToLower(const std::string& value) {
  std::string out = value;
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

// Drops whitespace so a vendor spelling such as "GTX 1080 Ti" normalizes to the
// table's "gtx1080ti" join key.
std::string StripWhitespace(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
  }
  return out;
}

}  // namespace

const std::vector<const DeviceProfile*>& AllDeviceProfiles() {
  static const std::vector<const DeviceProfile*> kAll = [] {
    std::vector<const DeviceProfile*> all;
    for (const DeviceProfile& profile : kProfiles) all.push_back(&profile);
    return all;
  }();
  return kAll;
}

const DeviceProfile* FindDeviceProfile(const std::string& name) {
  const std::string wanted = StripWhitespace(ToLower(name));
  if (wanted.empty()) return nullptr;
  // Exact match first: the profile name is the join key, and a run reports it
  // verbatim.
  for (const DeviceProfile* profile : AllDeviceProfiles()) {
    if (wanted == profile->name) return profile;
  }
  // Fall back to a substring match so a full device name such as
  // "NVIDIA H100 80GB HBM3" resolves to its profile.
  for (const DeviceProfile* profile : AllDeviceProfiles()) {
    if (wanted.find(profile->name) != std::string::npos) return profile;
  }
  return nullptr;
}

Caps CapsFromProfile(const DeviceProfile& profile) {
  Caps caps;
  caps.device_index = 0;
  caps.compute_major = profile.compute_major;
  caps.compute_minor = profile.compute_minor;
  caps.total_memory_bytes = profile.total_memory_bytes;
  caps.is_device = true;
  caps.has_cublas = true;
  caps.has_tensor_cores = profile.has_tensor_cores;
  caps.supports_fp32 = true;
  caps.supports_fp16 = profile.supports_fp16;
  caps.device_name = profile.name;
  return caps;
}

}  // namespace nanochat
