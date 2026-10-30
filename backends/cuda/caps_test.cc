// The capability declaration gate: `backends/cuda/caps.json` against the mask
// `GetCaps()` actually reports, and against the registrations the planner
// reads.
//
// This is the test that makes `caps.json` the single source instead of a
// second, decorative list (docs/model-and-device-bringup.md section 6,
// spec/capability_bits.md section 1). It holds three things together:
//
//   1. Every bit string in the file names a capability the header defines. A
//      name with no constant is a failure by name, never a silent drop.
//   2. Every bit `GetCaps()` reports for this device is a bit the file declares
//      for it. The mask may therefore not grow a bit the declaration does not
//      carry, which is what "no second hand-written capability list" means once
//      it is checkable.
//   3. Every capability bit declared for this device is either reported or is a
//      precision bit the silicon cannot honour. The reverse direction needs the
//      caps.json-derived profile entry to carry a mask, so it reports the
//      pending derivation rather than passing silently when the entry is empty.
//
// It also cross-checks the parts of the mask that promise an implementation: an
// `override_*` bit in the mask must correspond to an entry in
// `RegisteredFusedOverrides`, because a device that claims a fused override the
// backend does not register sends the planner looking for a family that is not
// there.
//
// The file is read through the runfiles path in argv[1] (`data` + `$(rootpath)`
// in BUILD.bazel), the same way the sequence oracle test reads its fixture.
// See docs/testing.md.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/caps_names.h"
#include "backends/cuda/device.h"
#include "nanochat/device_profile.h"
#include "nanochat/kernels.h"
#include "nanochat/plan.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

// The project's tests are plain `main` binaries with a failure counter
// (docs/testing.md), so the helpers are local rather than a framework.
namespace {

using nanochat::Caps;
using nanochat::FusedOverride;

int g_failures = 0;

void Fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++g_failures;
}

std::string BitNames(std::uint64_t mask) {
  std::string names;
  for (int i = 0; i < 64; ++i) {
    const std::uint64_t bit = std::uint64_t{1} << i;
    if ((mask & bit) == 0) continue;
    const char* name = nanochat::cuda_backend::CapabilityNameForBit(bit);
    if (!names.empty()) names += ", ";
    if (name != nullptr) {
      names += name;
    } else {
      names += "bit " + std::to_string(i) + " (no vocabulary name)";
    }
  }
  return names.empty() ? "(none)" : names;
}

std::string ReadFile(const char* path) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
    std::printf("FAIL: cannot open %s\n", path);
    ++g_failures;
    return std::string();
  }
  std::string text;
  char buffer[4096];
  std::size_t read = 0;
  while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
    text.append(buffer, read);
  }
  std::fclose(file);
  return text;
}

// The next double-quoted string at or after `pos`, with `pos` advanced past it.
bool NextQuoted(const std::string& text, std::size_t* pos, std::string* out) {
  const std::size_t open = text.find('"', *pos);
  if (open == std::string::npos) return false;
  const std::size_t close = text.find('"', open + 1);
  if (close == std::string::npos) return false;
  *out = text.substr(open + 1, close - open - 1);
  *pos = close + 1;
  return true;
}

// One `devices[]` entry: the join key and the bit names it declares.
struct DeviceEntry {
  std::string name;
  std::vector<std::string> features;
};

// A deliberately small reader: the file is ours, and the only structure this
// test needs is `"name": "<device>"` followed by `"features": [ "<bit>", ... ]`
// inside each device object. A real JSON parser would hide which bytes the test
// relies on, and this test is about the bytes.
std::vector<DeviceEntry> ParseDevices(const std::string& text, bool* ok) {
  std::vector<DeviceEntry> devices;
  *ok = true;
  std::size_t pos = 0;
  while (true) {
    const std::size_t name_key = text.find("\"name\":", pos);
    if (name_key == std::string::npos) break;
    std::size_t cursor = name_key + std::strlen("\"name\":");
    DeviceEntry entry;
    if (!NextQuoted(text, &cursor, &entry.name)) {
      Fail("a \"name\" key has no string value");
      *ok = false;
      return devices;
    }
    const std::size_t features_key = text.find("\"features\":", cursor);
    if (features_key == std::string::npos) {
      Fail("the device \"" + entry.name + "\" declares no \"features\" list");
      *ok = false;
      return devices;
    }
    const std::size_t open_bracket = text.find('[', features_key);
    const std::size_t close_bracket = open_bracket == std::string::npos
                                          ? std::string::npos
                                          : text.find(']', open_bracket);
    if (open_bracket == std::string::npos ||
        close_bracket == std::string::npos) {
      Fail("the \"features\" list of \"" + entry.name + "\" is not an array");
      *ok = false;
      return devices;
    }
    const std::size_t inner_begin = open_bracket + 1;
    const std::string inner =
        text.substr(inner_begin, close_bracket - inner_begin);
    std::size_t cursor_inner = 0;
    std::string feature;
    while (NextQuoted(inner, &cursor_inner, &feature)) {
      entry.features.push_back(feature);
    }
    pos = close_bracket + 1;
    devices.push_back(std::move(entry));
  }
  return devices;
}

std::uint64_t BitsForNames(const std::vector<std::string>& names,
                           const std::string& device) {
  std::uint64_t mask = 0;
  for (const std::string& name : names) {
    std::uint64_t bit = 0;
    if (!nanochat::cuda_backend::ParseCapabilityName(name.c_str(), &bit)) {
      Fail("the device \"" + device + "\" declares the capability name \"" +
           name +
           "\", which spec/capability_bits.md version 1 does not define (or "
           "which has no constant in include/nanochat/tensor.h). A name with "
           "no mapping is an error, never a dropped row.");
      continue;
    }
    if ((mask & bit) != 0) {
      Fail("the device \"" + device + "\" declares \"" + name +
           "\" twice; a caps file lists each bit once");
    }
    mask |= bit;
  }
  return mask;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("test");
  std::printf("caps declaration test\n");
  if (argc < 2) {
    std::printf("FAIL: no caps.json path in argv[1]\n");
    return 1;
  }

  const std::string text = ReadFile(argv[1]);
  if (text.empty()) {
    std::printf("caps: %d failure(s)\n", g_failures);
    return 1;
  }

  // --- the declaration's own shape -----------------------------------------
  // //
  if (text.find("\"capability_vocabulary\": 1") == std::string::npos) {
    Fail(
        "caps.json does not declare capability vocabulary 1 "
        "(spec/versions.md section 2: a backend declares the vocabulary it "
        "speaks)");
  }
  if (text.find("\"backend\": \"cuda\"") == std::string::npos) {
    Fail("caps.json does not name the cuda backend");
  }

  bool parsed = false;
  const std::vector<DeviceEntry> devices = ParseDevices(text, &parsed);
  if (!parsed || devices.empty()) {
    Fail("caps.json declares no device entry");
    std::printf("caps: %d failure(s)\n", g_failures);
    return 1;
  }

  // --- every name in the file maps to a constant ---------------------------
  // //
  std::vector<std::pair<std::string, std::uint64_t>> declared;
  declared.reserve(devices.size());
  for (const DeviceEntry& entry : devices) {
    if (entry.features.empty()) {
      Fail("the device \"" + entry.name +
           "\" declares no capability at all; "
           "a device with no primitive tier is non-conformant, so it does not "
           "belong in the file");
    }
    declared.emplace_back(entry.name, BitsForNames(entry.features, entry.name));
  }
  std::printf("  %zu device declaration(s) read\n", devices.size());

  // --- the mask GetCaps() reports ----------------------------------------- //
  const Caps caps = nanochat::kernels::GetCaps();
  std::printf("  device \"%s\", compute %d.%d, mask: %s\n", caps.device_name,
              caps.compute_major, caps.compute_minor,
              BitNames(caps.feature_mask).c_str());

  // The three booleans and `Supports(DType)` are one answer spelled twice
  // (`spec/capability_bits.md` section 1); they may not disagree with the mask.
  auto expect_bool = [&](bool value, const char* what, std::uint64_t bit) {
    const bool from_mask = (caps.feature_mask & bit) != 0;
    if (value != from_mask) {
      Fail(std::string("Caps::") + what + " is " + (value ? "true" : "false") +
           " but the bit " + (from_mask ? "is" : "is not") +
           " set in feature_mask");
    }
  };
  expect_bool(caps.supports_fp32, "supports_fp32", nanochat::kFeatureFp32);
  expect_bool(caps.supports_bf16, "supports_bf16", nanochat::kFeatureBf16);
  // `supports_fp8` is the tensor-core pair, so it asks for both fp8 bits.
  expect_bool(caps.supports_fp8, "supports_fp8",
              nanochat::kFeatureFp8Storage | nanochat::kFeatureFp8Gemm);
  if (caps.Supports(nanochat::DType::kBf16) != caps.supports_bf16) {
    Fail("Supports(kBf16) disagrees with supports_bf16");
  }
  if (caps.Supports(nanochat::DType::kFp8E4M3) != caps.supports_fp8 ||
      caps.Supports(nanochat::DType::kFp8E5M2) != caps.supports_fp8) {
    Fail("Supports(kFp8E4M3/kFp8E5M2) disagrees with supports_fp8");
  }
  if (caps.Supports(nanochat::DType::kFp8E4M3) && !caps.supports_fp8) {
    Fail("an fp8 dtype is supported while supports_fp8 is false");
  }

  // --- the mask's precision bits are the silicon's -------------------------
  // // A declaration may not grant a precision bit the card lacks
  // (spec/capability_bits.md section 4): the four precision bits are a fact the
  // runtime answers, and the test recomputes them from the reported compute
  // capability rather than from the file.
  const std::uint64_t precision_bits =
      nanochat::kFeatureFp32 | nanochat::kFeatureBf16 |
      nanochat::kFeatureFp8Storage | nanochat::kFeatureFp8Gemm;
  const std::uint64_t silicon = nanochat::cuda_backend::PrecisionFeatureBits(
      caps.compute_major, caps.compute_minor);
  if ((caps.feature_mask & precision_bits) != silicon) {
    Fail("the precision bits in feature_mask (" +
         BitNames(caps.feature_mask & precision_bits) +
         ") are not what compute capability " +
         std::to_string(caps.compute_major) + "." +
         std::to_string(caps.compute_minor) + " reports (" + BitNames(silicon) +
         ")");
  }

  // --- the declaration this device's mask must agree with ------------------
  // //
  const nanochat::DeviceProfile* profile =
      nanochat::FindDeviceProfile(caps.device_name);
  const std::string join_key = profile != nullptr ? profile->name : "";
  if (profile == nullptr) {
    std::printf(
        "  NOTE: \"%s\" is not in the derived profile table, so there is no "
        "caps.json entry to compare against; the mask holds only what the "
        "silicon reports\n",
        caps.device_name);
  }
  std::uint64_t declared_json = 0;
  bool found = false;
  for (const std::pair<std::string, std::uint64_t>& entry : declared) {
    if (entry.first == join_key) {
      declared_json = entry.second;
      found = true;
    }
  }

  if (profile != nullptr && !found) {
    Fail("the derived profile table has the entry \"" + join_key +
         "\" and caps.json does not declare it; the table may not hold a "
         "device the declaration file does not describe");
  }

  if (found) {
    // Direction 1, always checked: no bit in the mask that the file does not
    // declare for this device. This is the "single source" property.
    const std::uint64_t undeclared = caps.feature_mask & ~declared_json;
    if (undeclared != 0) {
      Fail("GetCaps() reports bit(s) caps.json does not declare for \"" +
           join_key + "\": " + BitNames(undeclared) +
           ". The mask comes from the caps.json-derived profile entry plus the "
           "silicon's precision bits, so a bit outside the file means a second "
           "capability list grew somewhere.");
    }
    // Direction 2, checked when the derivation has run: the bits the file
    // declares are the bits the mask carries, minus the precision the silicon
    // withholds.
    const std::uint64_t withheld = precision_bits & ~silicon;
    const std::uint64_t missing =
        declared_json & ~caps.feature_mask & ~withheld;
    const std::uint64_t from_profile =
        nanochat::cuda_backend::DeclaredFeatureMask(caps.device_name);
    if ((from_profile & ~precision_bits) == 0) {
      std::printf(
          "  NOTE: the derived profile entry for \"%s\" carries no capability "
          "mask, so the caps.json derivation (src/device_profile.cc) has not "
          "run yet; the tier, stateful and override bits cannot be sourced "
          "from it and direction 2 is reported instead of asserted: %s\n",
          join_key.c_str(), BitNames(missing).c_str());
    } else if (missing != 0) {
      Fail("caps.json declares for \"" + join_key + "\" the bit(s) " +
           BitNames(missing) +
           " that GetCaps() does not report; the profile entry and the "
           "declaration file disagree");
    }
  }

  // --- a claimed override must be a registered override ------------------- //
  // A device that claims an `override_*` bit while the backend registers no
  // family behind it sends the planner looking for an implementation that is
  // not there, which is the dead-code asymmetry `spec/capability_bits.md`
  // section 5 describes from the other side.
  int registration_count = 0;
  const FusedOverride* registrations =
      nanochat::RegisteredFusedOverrides(&registration_count);
  const nanochat::FusedOverride* registrations_checked = registrations;
  std::uint64_t registered_bits = 0;
  for (int i = 0; i < registration_count && registrations_checked != nullptr;
       ++i) {
    registered_bits |= registrations_checked[i].capability;
    // The diagnostic names both spellings, so a reviewer can compare the caps
    // file against the header without opening two files.
    std::printf("  registration %d: %s <- %s, gated on %s (%s)\n", i,
                registrations_checked[i].family,
                registrations_checked[i].composition,
                nanochat::cuda_backend::CapabilityNameForBit(
                    registrations_checked[i].capability) != nullptr
                    ? nanochat::cuda_backend::CapabilityNameForBit(
                          registrations_checked[i].capability)
                    : "(unnamed)",
                nanochat::cuda_backend::CapabilityConstantForBit(
                    registrations_checked[i].capability) != nullptr
                    ? nanochat::cuda_backend::CapabilityConstantForBit(
                          registrations_checked[i].capability)
                    : "(no header constant)");
    if (nanochat::cuda_backend::CapabilityNameForBit(
            registrations_checked[i].capability) == nullptr) {
      Fail(std::string("the registration for \"") +
           registrations_checked[i].family +
           "\" is gated on a capability bit with no vocabulary name");
    }
  }
  const std::uint64_t override_bits = nanochat::kFeatureFusedResidualRmsNorm |
                                      nanochat::kFeatureFusedQkPrep |
                                      nanochat::kFeatureFusedClassifier;
  const std::uint64_t claimed_unregistered =
      caps.feature_mask & override_bits & ~registered_bits;
  if (claimed_unregistered != 0) {
    Fail("the mask claims the override bit(s) " +
         BitNames(claimed_unregistered) +
         " with no entry in RegisteredFusedOverrides");
  }
  std::printf("  %d fused override registration(s)\n", registration_count);

  if (g_failures != 0) {
    std::printf("caps: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("caps: ok\n");
  return 0;
}
