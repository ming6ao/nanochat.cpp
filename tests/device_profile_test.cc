// tests/device_profile_test.cc -- the device profile / MFU consistency gate
// (docs/simulator.md section 7) and the caps.json derivation gate
// (docs/model-and-device-bringup.md sections 6 and 16).
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
//      capability 8.9 or higher (docs/testing.md Part 1, section 3.1).
//
// The table itself is derived, not maintained: every row comes from a
// `devices[]` entry in `backends/<name>/caps.json` through
// `src/device_profile.cc`. So the test also reads both declarations and holds
// the derivation to three rules:
//
//   5. Every declared field of a declared device appears in its derived row,
//      field for field, so a declaration cannot drift from the table.
//   6. The derived table has exactly one row per declared device: nothing
//      declared is dropped, and no row exists that no declaration owns (a
//      device that is not declared cannot be silently added to a header).
//   7. A declaration that is not an accelerator (`"is_device": false`, the
//      reference host) contributes no row, because it has no launch cost, no
//      HBM rate, and no MFU rate.
//
// It needs no GPU, no fixture, and no broker: tier S0, a plain T0 CPU test.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "nanochat/device_profile.h"
#include "nanochat/mfu.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/json.h"

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

// The declaration's path: the Bazel runfiles tree first, then the tree relative
// to the working directory, the same order `src/device_profile.cc` uses, so the
// test and the derivation read the same bytes.
std::string LocateCaps(const char* backend) {
  const std::string relative =
      std::string("backends/") + backend + "/caps.json";
  std::vector<std::string> candidates;
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.push_back(std::string(src_dir) + "/" + workspace + "/" +
                           relative);
    }
    candidates.push_back(std::string(src_dir) + "/_main/" + relative);
  }
  candidates.push_back(relative);
  std::string prefix;
  for (int level = 0; level < 4; ++level) {
    prefix += "../";
    candidates.push_back(prefix + relative);
  }
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate);
    if (in.good()) return candidate;
  }
  return std::string();
}

// The rate the reference recipe's Model FLOPs Utilization line divides by: the
// tensor-core bf16 rate on a device with bf16, and the CUDA-core fp32 rate
// otherwise. The Turing T4 has tensor cores but no bf16, so it reports the fp32
// rate. See docs/precision.md. A declaration states this rate and the precision
// it is quoted at, so the table below is a check on the declaration rather than
// a second copy of it.
struct ReferenceCandidate {
  double flops = 0.0;
  const char* precision = "fp32";
};

ReferenceCandidate MfuReference(const DeviceProfile& profile) {
  if (profile.supports_bf16 && profile.peak_bf16_flops > 0.0) {
    return {profile.peak_bf16_flops, "bf16"};
  }
  return {profile.peak_fp32_flops, "fp32"};
}

const char* PrecisionNameOf(nanochat::BuildPrecision precision) {
  switch (precision) {
    case nanochat::BuildPrecision::kBf16:
      return "bf16";
    case nanochat::BuildPrecision::kFp8:
      return "fp8";
    case nanochat::BuildPrecision::kFp32:
      break;
  }
  return "fp32";
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
  ExpectTrue(nanochat::FindDeviceProfile("NVIDIA T4") ==
                 nanochat::FindDeviceProfile("t4"),
             "a vendor spelling with spaces resolves");
  ExpectTrue(nanochat::FindDeviceProfile("no such device") == nullptr,
             "an unknown name has no profile");
  ExpectTrue(nanochat::FindDeviceProfile("") == nullptr,
             "the empty name has no profile");
  // A vendor spelling that contains one declared key resolves to it, so a run
  // reports a card the table holds.
  ExpectTrue(nanochat::FindDeviceProfile("NVIDIA H100 80GB HBM3") ==
                 nanochat::FindDeviceProfile("h100"),
             "a card's marketing name resolves to its declared row");
}

void TestUnknownDeviceRefusal() {
  // One message and one device list for every command, so the roofline report,
  // the simulator, and the benchmark tool refuse an unknown device the same way
  // (docs/performance-model.md section 6).
  const std::string list = nanochat::DeviceNameList();
  ExpectTrue(!list.empty(), "the declared device list is not empty");
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    ExpectTrue(list.find(profile->name) != std::string::npos,
               Format("the device list names \"%s\"", profile->name));
  }
  const std::string error = nanochat::DeviceNameError("bogus");
  ExpectTrue(error.find("bogus") != std::string::npos,
             "the refusal message names the rejected device");
  ExpectTrue(error.find("h100") != std::string::npos,
             "the refusal message lists the declared devices");
  // The list is the declared set: a name it does not carry has no row, so a
  // command that validates against the list accepts exactly the declared set.
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    ExpectTrue(nanochat::FindDeviceProfile(profile->name) != nullptr,
               Format("\"%s\" from the list resolves", profile->name));
  }
}

void TestMfuConsistency() {
  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    const std::string name = profile->name;
    // The reference rate is declared, and `PeakFlopsForDevice` reads the
    // declaration. These three checks are what keeps a card's reference rate
    // from drifting away from the card's other rates: the reference rate must
    // equal one of the declared rates, and it must be the rate the documented
    // rule selects (docs/performance-model.md section 6).
    const ReferenceCandidate candidate = MfuReference(*profile);
    ExpectTrue(
        profile->reference_peak_flops > 0.0,
        Format("profile \"%s\" declares a reference peak rate", name.c_str()));
    ExpectTrue(std::string(PrecisionNameOf(profile->reference_precision)) ==
                   candidate.precision,
               Format("profile \"%s\" quotes its reference rate at %s",
                      name.c_str(), candidate.precision));
    ExpectNear(
        profile->reference_peak_flops,
        nanochat::PeakFlopsFor(*profile, profile->reference_precision), 1.0,
        Format("profile \"%s\" reference rate is one of its declared rates",
               name.c_str()));
    const double mfu_peak = nanochat::PeakFlopsForDevice(name);
    ExpectTrue(
        mfu_peak > 0.0,
        Format("profile \"%s\" resolves to one reference rate", name.c_str()));
    ExpectNear(mfu_peak, profile->reference_peak_flops, 1.0,
               Format("PeakFlopsForDevice(\"%s\") reads the declaration",
                      name.c_str()));
    // The fp32 rate is always recorded; a bf16 rate is recorded on a device
    // that has bf16, and it is the rate the reference line models there.
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
  ExpectTrue(seen.size() >= 2, "the table covers the declared cards");
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

  const DeviceProfile* t4 = nanochat::FindDeviceProfile("t4");
  ExpectTrue(t4 != nullptr, "t4 is in the table");
  if (t4 == nullptr) return;
  const Caps t4_caps = nanochat::CapsFromProfile(*t4);
  ExpectTrue(t4_caps.has_tensor_cores, "Turing has tensor cores");
  ExpectTrue(t4_caps.compute_major == 7 && t4_caps.compute_minor == 5,
             "T4 is compute capability 7.5");
  ExpectTrue(!t4_caps.Supports(nanochat::DType::kBf16),
             "T4 does not support bf16");
  ExpectTrue(!t4_caps.Supports(nanochat::DType::kFp8E4M3),
             "T4 has no fp8 path");
}

void TestFp8Capability() {
  // The fp8 rows (docs/simulator.md Part 7, section 3,
  // docs/testing.md Part 1, section 3.1). Three rules hold:
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
  // checkpoint stays fp32 (docs/precision.md Part 1, section 3.1).
  ExpectTrue(nanochat::SizeOf(nanochat::DType::kFp8E4M3) == 1,
             "E4M3 is one byte");
  ExpectTrue(nanochat::SizeOf(nanochat::DType::kFp8E5M2) == 1,
             "E5M2 is one byte");
  ExpectTrue(!nanochat::IsStorageDType(nanochat::DType::kFp8E4M3),
             "E4M3 is not a checkpoint storage dtype");
  ExpectTrue(!nanochat::IsStorageDType(nanochat::DType::kFp8E5M2),
             "E5M2 is not a checkpoint storage dtype");
}

// ---------------------------------------------------------------------------
// The derivation: backends/*/caps.json -> the profile table
// ---------------------------------------------------------------------------

// One `devices[]` entry, reduced to the fields the derivation projects. A
// missing field is a failure rather than a default, because the derivation
// silently keeps the `DeviceProfile` default and a declaration would then drift
// from the table one zero at a time.
struct DeclaredEntry {
  std::string name;
  bool is_device = false;
  int compute_major = 0;
  int compute_minor = 0;
  double total_memory_bytes = 0.0;
  bool has_tensor_cores = false;
  bool supports_bf16 = false;
  bool supports_fp8 = false;
  double peak_fp32_flops = 0.0;
  double peak_bf16_flops = 0.0;
  double peak_fp8_flops = 0.0;
  double peak_bandwidth_bytes_per_second = 0.0;
  // The reference recipe's rate and the precision it is quoted at. They are
  // required fields as well, so a declaration cannot omit the number
  // `PeakFlopsForDevice` reads.
  double reference_peak_flops = 0.0;
  std::string reference_precision;
  int warp_size = 0;
  int max_threads_per_block = 0;
  double max_shared_memory_per_block = 0.0;
  double launch_us = 0.0;
};

bool NeedNumber(const nanochat::json::Value& entry, const char* key,
                const std::string& where, double* out) {
  const nanochat::json::Value* value = entry.Find(key);
  if (value == nullptr || !value->is_number()) {
    Fail(where + " declares no numeric \"" + key + "\"");
    return false;
  }
  *out = value->number();
  return true;
}

bool NeedInt(const nanochat::json::Value& entry, const char* key,
             const std::string& where, int* out) {
  const nanochat::json::Value* value = entry.Find(key);
  if (value == nullptr || !value->Integer(out)) {
    Fail(where + " declares no integral \"" + key + "\"");
    return false;
  }
  return true;
}

bool NeedBool(const nanochat::json::Value& entry, const char* key,
              const std::string& where, bool* out) {
  const nanochat::json::Value* value = entry.Find(key);
  if (value == nullptr || !value->is_bool()) {
    Fail(where + " declares no boolean \"" + key + "\"");
    return false;
  }
  *out = value->boolean();
  return true;
}

// Reads one declaration into `out`. Returns false when the file is absent or
// unparsable; the caller reports that by name.
bool ReadDeclaredEntries(const char* backend, std::vector<DeclaredEntry>* out) {
  const std::string path = LocateCaps(backend);
  if (path.empty()) {
    Fail(
        Format("backends/%s/caps.json is not in the runfiles: the derivation "
               "has no declaration to read",
               backend));
    return false;
  }
  nanochat::json::Value root;
  std::string error;
  if (!nanochat::json::ParseFile(path, &root, &error)) {
    Fail(path + " does not parse: " + error);
    return false;
  }
  const nanochat::json::Value* devices = root.Find("devices");
  if (devices == nullptr || !devices->is_array()) {
    Fail(path + " declares no devices array");
    return false;
  }
  for (const nanochat::json::Value& entry : devices->array()) {
    const nanochat::json::Value* name = entry.Find("name");
    if (name == nullptr || !name->is_string() || name->string().empty()) {
      Fail(path + " has a devices[] entry with no name");
      continue;
    }
    DeclaredEntry declared;
    declared.name = name->string();
    const std::string where = path + ": device \"" + declared.name + "\"";
    if (!NeedBool(entry, "is_device", where, &declared.is_device)) continue;
    if (!NeedInt(entry, "compute_major", where, &declared.compute_major)) {
      continue;
    }
    if (!NeedInt(entry, "compute_minor", where, &declared.compute_minor)) {
      continue;
    }
    if (!NeedNumber(entry, "total_memory_bytes", where,
                    &declared.total_memory_bytes)) {
      continue;
    }
    if (!NeedBool(entry, "has_tensor_cores", where,
                  &declared.has_tensor_cores)) {
      continue;
    }
    if (!NeedBool(entry, "supports_bf16", where, &declared.supports_bf16)) {
      continue;
    }
    if (!NeedBool(entry, "supports_fp8", where, &declared.supports_fp8)) {
      continue;
    }
    if (!NeedNumber(entry, "peak_fp32_flops", where,
                    &declared.peak_fp32_flops)) {
      continue;
    }
    if (!NeedNumber(entry, "peak_bf16_flops", where,
                    &declared.peak_bf16_flops)) {
      continue;
    }
    if (!NeedNumber(entry, "peak_fp8_flops", where, &declared.peak_fp8_flops)) {
      continue;
    }
    if (!NeedNumber(entry, "peak_bandwidth_bytes_per_second", where,
                    &declared.peak_bandwidth_bytes_per_second)) {
      continue;
    }
    if (!NeedNumber(entry, "reference_peak_flops", where,
                    &declared.reference_peak_flops)) {
      continue;
    }
    const nanochat::json::Value* precision = entry.Find("reference_precision");
    if (precision == nullptr || !precision->is_string()) {
      Fail(where + " declares no string \"reference_precision\"");
      continue;
    }
    declared.reference_precision = precision->string();
    if (!NeedInt(entry, "warp_size", where, &declared.warp_size)) continue;
    if (!NeedInt(entry, "max_threads_per_block", where,
                 &declared.max_threads_per_block)) {
      continue;
    }
    if (!NeedNumber(entry, "max_shared_memory_per_block", where,
                    &declared.max_shared_memory_per_block)) {
      continue;
    }
    if (!NeedNumber(entry, "launch_us", where, &declared.launch_us)) continue;
    out->push_back(declared);
  }
  return true;
}

// The join key the derivation uses: lowercased, whitespace removed.
std::string NormalizeName(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
    out.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

void TestDeclarationsDeriveTheTable() {
  std::vector<DeclaredEntry> declared;
  const bool read_cpu = ReadDeclaredEntries("cpu", &declared);
  const bool read_cuda = ReadDeclaredEntries("cuda", &declared);
  ExpectTrue(read_cpu, "the reference backend declares its devices");
  ExpectTrue(read_cuda, "the CUDA backend declares its devices");
  if (!read_cpu || !read_cuda) return;

  std::vector<std::string> declared_devices;
  std::vector<std::string> declared_hosts;
  for (const DeclaredEntry& entry : declared) {
    const std::string name = NormalizeName(entry.name);
    const DeviceProfile* profile = nanochat::FindDeviceProfile(name);
    if (!entry.is_device) {
      // Rule 7: a declaration that is not an accelerator declares its mask and
      // nothing else. It has no launch cost, no HBM rate and no MFU rate, so it
      // contributes no row.
      declared_hosts.push_back(name);
      ExpectTrue(profile == nullptr,
                 Format("the declaration \"%s\" is not an accelerator, so it "
                        "contributes no profile row",
                        name.c_str()));
      continue;
    }
    // Rule 5: every declared field appears in the derived row, so a
    // declaration cannot drift from the table.
    declared_devices.push_back(name);
    ExpectTrue(
        profile != nullptr,
        Format("the declared device \"%s\" has a derived row", name.c_str()));
    if (profile == nullptr) continue;
    ExpectTrue(profile->name == name,
               Format("the row for \"%s\" carries the declaration's name",
                      name.c_str()));
    ExpectTrue(profile->compute_major == entry.compute_major &&
                   profile->compute_minor == entry.compute_minor,
               Format("%s: compute capability %d.%d derives from the "
                      "declaration",
                      name.c_str(), entry.compute_major, entry.compute_minor));
    ExpectTrue(profile->total_memory_bytes ==
                   static_cast<std::size_t>(entry.total_memory_bytes),
               Format("%s: the memory budget derives from the declaration",
                      name.c_str()));
    ExpectTrue(profile->has_tensor_cores == entry.has_tensor_cores,
               Format("%s: has_tensor_cores derives from the declaration",
                      name.c_str()));
    ExpectTrue(
        profile->supports_bf16 == entry.supports_bf16,
        Format("%s: supports_bf16 derives from the declaration", name.c_str()));
    ExpectTrue(
        profile->supports_fp8 == entry.supports_fp8,
        Format("%s: supports_fp8 derives from the declaration", name.c_str()));
    ExpectNear(
        profile->peak_fp32_flops, entry.peak_fp32_flops, 1.0,
        Format("%s: the fp32 peak derives from the declaration", name.c_str()));
    ExpectNear(
        profile->peak_bf16_flops, entry.peak_bf16_flops, 1.0,
        Format("%s: the bf16 peak derives from the declaration", name.c_str()));
    ExpectNear(
        profile->peak_fp8_flops, entry.peak_fp8_flops, 1.0,
        Format("%s: the fp8 peak derives from the declaration", name.c_str()));
    ExpectNear(
        profile->peak_bandwidth_bytes_per_second,
        entry.peak_bandwidth_bytes_per_second, 1.0,
        Format("%s: the bandwidth derives from the declaration", name.c_str()));
    ExpectNear(profile->reference_peak_flops, entry.reference_peak_flops, 1.0,
               Format("%s: the reference rate derives from the declaration",
                      name.c_str()));
    ExpectTrue(
        std::string(PrecisionNameOf(profile->reference_precision)) ==
            entry.reference_precision,
        Format("%s: the reference precision derives from the declaration",
               name.c_str()));
    ExpectTrue(
        profile->warp_size == entry.warp_size,
        Format("%s: the warp size derives from the declaration", name.c_str()));
    ExpectTrue(profile->max_threads_per_block == entry.max_threads_per_block,
               Format("%s: the thread bound derives from the declaration",
                      name.c_str()));
    ExpectTrue(profile->max_shared_memory_per_block ==
                   static_cast<std::size_t>(entry.max_shared_memory_per_block),
               Format("%s: the shared-memory bound derives from the "
                      "declaration",
                      name.c_str()));
    ExpectNear(profile->launch_us, entry.launch_us, 1e-9,
               Format("%s: the launch cost derives from the declaration",
                      name.c_str()));
  }

  // Rule 6: the table is exactly the declared devices. A row that no
  // declaration owns is a hand-added row, and a declared device without a row
  // is a dropped declaration; both are failures rather than a smaller table.
  const std::vector<const DeviceProfile*>& table =
      nanochat::AllDeviceProfiles();
  ExpectTrue(table.size() == declared_devices.size(),
             Format("the derived table has one row per declared device "
                    "(%zu rows, %zu declared)",
                    table.size(), declared_devices.size()));
  for (const DeviceProfile* profile : table) {
    const auto count =
        std::count(declared_devices.begin(), declared_devices.end(),
                   std::string(profile->name));
    ExpectTrue(count == 1,
               Format("profile \"%s\" is declared exactly once across "
                      "backends/*/caps.json",
                      profile->name));
  }

  // The no-row-per-device rule stated the other way: a name no declaration
  // carries has no row, so a device is brought up by adding a `devices[]`
  // entry and not by editing a header.
  for (const char* undeclared :
       {"not-a-declared-device", "b200", "rtx5090", "mi300"}) {
    ExpectTrue(nanochat::FindDeviceProfile(undeclared) == nullptr,
               Format("the undeclared name \"%s\" has no row: a device is "
                      "added by a declaration, never by a table edit",
                      undeclared));
  }
  for (const std::string& host : declared_hosts) {
    ExpectTrue(nanochat::FindDeviceProfile(host) == nullptr,
               Format("the declared host \"%s\" has no accelerator row",
                      host.c_str()));
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestJoinKey();
  TestUnknownDeviceRefusal();
  TestMfuConsistency();
  TestLimitTable();
  TestCapsProjection();
  TestFp8Capability();
  TestDeclarationsDeriveTheTable();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("device profile: all checks passed\n");
  return 0;
}
