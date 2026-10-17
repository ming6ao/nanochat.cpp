// tests/device_profile_test.cc -- the device profile / MFU consistency gate
// (docs/simulator.md section 7).
//
// The simulator reads its capability data from `nanochat/device_profile.h` and
// its MFU peak rate from `PeakFlopsForDevice` in `src/train.cc`. The two tables
// must agree, or a simulated run would report a device whose peak rate is not
// the one the table names. The test is objective:
//
//   1. The join key is the lowercased device name.
//   2. Every profile name resolves to exactly one MFU entry (a nonzero rate).
//   3. For a shared entry, the MFU rate equals the profile's peak at the
//      precision the MFU table models: the bf16 tensor-core rate on a device
//      with bf16, the fp32 CUDA-core rate otherwise.
//   4. The fp8 rate and the fp8 flag agree, and an fp8 device reports compute
//      capability 8.9 or higher (docs/correctness-gates.md section 3.1).
//
// It needs no GPU, no fixture, and no broker: tier S0, a plain T0 CPU test.

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/device_profile.h"
#include "nanochat/mfu.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::Caps;
using nanochat::ComputePrecision;
using nanochat::DeviceProfile;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& what) {
  if (!(std::fabs(actual - expected) <= tolerance)) {
    Fail(Format("%s: got %.6g, want %.6g", what.c_str(), actual, expected));
  }
}

// The rate the MFU table models for this device: the tensor-core bf16 rate when
// the device has bf16, and the CUDA-core fp32 rate otherwise. The Turing T4 has
// tensor cores but no bf16, and its build runs fp32, so it reports the fp32
// rate. See docs/precision.md.
double MfuPeakFor(const DeviceProfile& profile) {
  if (profile.supports_bf16 && profile.peak_bf16_flops > 0.0) {
    return profile.peak_bf16_flops;
  }
  return profile.peak_fp32_flops;
}

void TestJoinKey() {
  // Case-insensitive lookup: the join key is the lowercased device name.
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    const DeviceProfile* found = nanochat::FindDeviceProfile(profile->name);
    ExpectTrue(found == profile,
               Format("FindDeviceProfile(\"%s\") resolves to the profile",
                      profile->name));
  }
  ExpectTrue(nanochat::FindDeviceProfile("H100") != nullptr,
             "lookup is case-insensitive");
  ExpectTrue(nanochat::FindDeviceProfile("h100") ==
                 nanochat::FindDeviceProfile("H100"),
             "H100 and h100 agree");
  ExpectTrue(nanochat::FindDeviceProfile("GTX 1080 Ti") ==
                 nanochat::FindDeviceProfile("gtx1080ti"),
             "a vendor spelling with spaces resolves");
  ExpectTrue(nanochat::FindDeviceProfile("no such device") == nullptr,
             "an unknown name has no profile");
  ExpectTrue(nanochat::FindDeviceProfile("") == nullptr,
             "the empty name has no profile");
}

void TestMfuConsistency() {
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    const std::string name = profile->name;
    const double mfu_peak = nanochat::PeakFlopsForDevice(name);
    ExpectTrue(
        mfu_peak > 0.0,
        Format("profile \"%s\" resolves to one MFU entry", name.c_str()));
    ExpectNear(
        mfu_peak, MfuPeakFor(*profile), 1.0,
        Format("MFU rate for \"%s\" matches the profile peak", name.c_str()));
    // The fp32 rate is always recorded; a bf16 rate is recorded on a device
    // that has bf16, and it is the rate the MFU table models there.
    ExpectTrue(profile->peak_fp32_flops > 0.0,
               Format("profile \"%s\" records an fp32 rate", name.c_str()));
    if (profile->supports_bf16) {
      ExpectTrue(profile->peak_bf16_flops > 0.0,
                 Format("profile \"%s\" records a bf16 rate", name.c_str()));
    }
  }
  // Every profile in the table is reachable by name, and the table has no
  // duplicate keys.
  std::vector<std::string> seen;
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    for (const std::string& name : seen) {
      ExpectTrue(name != profile->name,
                 Format("profile name \"%s\" is unique", profile->name));
    }
    seen.push_back(profile->name);
  }
  ExpectTrue(seen.size() >= 5, "the table covers the planned targets");
}

void TestLimitTable() {
  // The fp8 rate and the bandwidth column (docs/performance-model.md section
  // 6). Hopper carries both; the older targets carry a bandwidth only, and a
  // zero fp8 rate means "no fp8 path".
  const DeviceProfile* h100 = nanochat::FindDeviceProfile("h100");
  ExpectTrue(h100 != nullptr, "h100 is in the table");
  if (h100 != nullptr) {
    ExpectNear(h100->peak_fp8_flops, 1979e12, 1.0, "h100 fp8 rate");
    ExpectNear(h100->peak_bandwidth_bytes_per_second, 3.35e12, 1.0,
               "h100 bandwidth");
    ExpectNear(nanochat::PeakFlopsForCompute("h100", ComputePrecision::kFp8),
               1979e12, 1.0, "h100 explicit fp8 peak");
  }
  const DeviceProfile* h200 = nanochat::FindDeviceProfile("h200");
  ExpectTrue(h200 != nullptr, "h200 is in the table");
  if (h200 != nullptr) {
    ExpectNear(h200->peak_bandwidth_bytes_per_second, 4.8e12, 1.0,
               "h200 bandwidth");
  }
  const DeviceProfile* a100 = nanochat::FindDeviceProfile("a100");
  ExpectTrue(a100 != nullptr, "a100 is in the table");
  if (a100 != nullptr) {
    ExpectNear(a100->peak_fp8_flops, 0.0, 1e-9, "a100 has no fp8 path");
    ExpectNear(a100->peak_bandwidth_bytes_per_second, 2.0e12, 1.0,
               "a100 bandwidth");
  }
  const DeviceProfile* t4 = nanochat::FindDeviceProfile("t4");
  ExpectTrue(t4 != nullptr, "t4 is in the table");
  if (t4 != nullptr) {
    ExpectNear(t4->peak_bandwidth_bytes_per_second, 320e9, 1.0, "t4 bandwidth");
  }

  // Every row records a bandwidth, and the explicit-precision selector returns
  // exactly the profile field at every precision.
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    const std::string name = profile->name;
    ExpectTrue(profile->peak_bandwidth_bytes_per_second > 0.0,
               Format("profile \"%s\" records a bandwidth", name.c_str()));
    ExpectNear(nanochat::PeakFlopsForCompute(name, ComputePrecision::kFp32),
               profile->peak_fp32_flops, 1.0,
               Format("%s explicit fp32 peak", name.c_str()));
    ExpectNear(nanochat::PeakFlopsForCompute(name, ComputePrecision::kBf16),
               profile->peak_bf16_flops, 1.0,
               Format("%s explicit bf16 peak", name.c_str()));
    ExpectNear(nanochat::PeakFlopsForCompute(name, ComputePrecision::kFp8),
               profile->peak_fp8_flops, 1.0,
               Format("%s explicit fp8 peak", name.c_str()));
  }
  // An unknown device has no limit row.
  ExpectNear(
      nanochat::PeakFlopsForCompute("unknown device", ComputePrecision::kFp8),
      0.0, 1e-9, "unknown device has no fp8 peak");
}

void TestCapsProjection() {
  const DeviceProfile* h100 = nanochat::FindDeviceProfile("h100");
  ExpectTrue(h100 != nullptr, "h100 is in the table");
  if (h100 == nullptr) return;
  const Caps caps = nanochat::CapsFromProfile(*h100);
  ExpectTrue(caps.is_device, "a profile projects to a device");
  ExpectTrue(caps.has_cublas, "a profile projects a GEMM library");
  ExpectTrue(caps.has_tensor_cores, "h100 has tensor cores");
  ExpectTrue(caps.compute_major == 9 && caps.compute_minor == 0,
             "h100 is compute capability 9.0");
  ExpectTrue(caps.total_memory_bytes == h100->total_memory_bytes,
             "the memory budget is carried through");
  ExpectTrue(caps.device_name == h100->name,
             "the device name is the profile name");
  ExpectTrue(caps.Supports(nanochat::DType::kFp32), "h100 supports fp32");
  ExpectTrue(caps.Supports(nanochat::DType::kBf16), "h100 supports bf16");

  const DeviceProfile* pascal = nanochat::FindDeviceProfile("gtx1080ti");
  ExpectTrue(pascal != nullptr, "gtx1080ti is in the table");
  if (pascal == nullptr) return;
  const Caps pascal_caps = nanochat::CapsFromProfile(*pascal);
  ExpectTrue(!pascal_caps.has_tensor_cores, "Pascal has no tensor cores");
  ExpectTrue(pascal_caps.compute_major == 6 && pascal_caps.compute_minor == 1,
             "the host is compute capability 6.1");
  ExpectTrue(!pascal_caps.Supports(nanochat::DType::kBf16),
             "Pascal does not support bf16");
}

void TestFp8Capability() {
  // The fp8 rows (docs/simulator-precision.md section 3,
  // docs/correctness-gates.md section 3.1). Three rules hold:
  //   1. A nonzero fp8 rate means `supports_fp8` is true, and the opposite.
  //   2. An fp8 device reports compute capability 8.9 or higher.
  //   3. The `Caps` projection carries the flag, and `Supports` answers for
  //      both fp8 operand tags.
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    const std::string name = profile->name;
    const bool has_rate = profile->peak_fp8_flops > 0.0;
    ExpectTrue(
        has_rate == profile->supports_fp8,
        Format("profile \"%s\" ties its fp8 rate to the flag", name.c_str()));
    if (profile->supports_fp8) {
      const bool modern =
          profile->compute_major > 8 ||
          (profile->compute_major == 8 && profile->compute_minor >= 9);
      ExpectTrue(modern, Format("profile \"%s\" with fp8 is compute capability "
                                "8.9 or higher",
                                name.c_str()));
    }
    const Caps caps = nanochat::CapsFromProfile(*profile);
    ExpectTrue(caps.supports_fp8 == profile->supports_fp8,
               Format("caps carry the fp8 flag for \"%s\"", name.c_str()));
    ExpectTrue(
        caps.Supports(nanochat::DType::kFp8E4M3) == profile->supports_fp8,
        Format("Supports(kFp8E4M3) agrees for \"%s\"", name.c_str()));
    ExpectTrue(
        caps.Supports(nanochat::DType::kFp8E5M2) == profile->supports_fp8,
        Format("Supports(kFp8E5M2) agrees for \"%s\"", name.c_str()));
  }
  // An fp8 operand is one byte, and it is not a container storage code: the
  // checkpoint stays fp32 (docs/fp8-training.md section 3.1).
  ExpectTrue(nanochat::SizeOf(nanochat::DType::kFp8E4M3) == 1,
             "E4M3 is one byte");
  ExpectTrue(nanochat::SizeOf(nanochat::DType::kFp8E5M2) == 1,
             "E5M2 is one byte");
  ExpectTrue(!nanochat::IsStorageDType(nanochat::DType::kFp8E4M3),
             "E4M3 is not a checkpoint storage dtype");
  ExpectTrue(!nanochat::IsStorageDType(nanochat::DType::kFp8E5M2),
             "E5M2 is not a checkpoint storage dtype");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestJoinKey();
  TestMfuConsistency();
  TestLimitTable();
  TestCapsProjection();
  TestFp8Capability();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("device profile: all checks passed\n");
  return 0;
}
