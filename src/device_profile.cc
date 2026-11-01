// The device capability table for the simulator (docs/simulator.md section 7).
//
// The table is *derived*: every row comes from `backends/<name>/caps.json`, the
// declaration that backend ships. A device bring-up therefore adds a file (a
// `devices[]` entry in that backend's declaration) and no row here, in
// `include/nanochat/device_profile.h`, or anywhere else
// (docs/model-and-device-bringup.md section 6: one source; section 16 P3).
//
// `PeakFlopsForDevice` in `src/train.cc` stays the authority for the MFU string
// patterns, and this file is the authority for the capability fields.
// `//tests:device_profile_test` holds the two consistent: it resolves every
// derived profile through the MFU table and compares the rates.
//
// Two views come out of the same parse:
//
//   * `AllDeviceProfiles` / `FindDeviceProfile` / `CapsFromProfile`, the shape
//     `include/nanochat/device_profile.h` declares. A caps entry that is not an
//     accelerator (`"is_device": false`, the reference host) declares no row:
//     it has no launch cost, no HBM rate and no tensor cores, and the MFU table
//     has no rate for it.
//   * `DeclaredFeatureMaskForDevice` (`src/device_caps.h`), the mechanism mask
//     the same entry declares. It lives beside the profile instead of inside it
//     because `DeviceProfile` has no mask field and that header is frozen for
//     this campaign; the planner is its consumer.
//
// The unit is deliberately a leaf: it depends only on `nanochat/tensor.h` plus
// the two `src/` helpers that read the files (the small JSON reader and the
// capability vocabulary), so the CPU reference backend can link it (to answer
// `GetCaps()` under the simulator) without gaining a dependency on the workflow
// layer.

#include "nanochat/device_profile.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "nanochat/tensor.h"
#include "src/capability_names.h"
#include "src/device_caps.h"
#include "src/json.h"

namespace nanochat {
namespace {

// The backends whose declarations are read. A new backend adds one entry here
// and one file label to this target's `data` in `src/BUILD.bazel`; a new
// *device* adds nothing but a `devices[]` entry in a caps file.
constexpr const char* kBackendNames[] = {"cpu", "cuda"};
constexpr int kBackendCount =
    static_cast<int>(sizeof(kBackendNames) / sizeof(kBackendNames[0]));

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

std::string Normalize(const std::string& value) {
  return StripWhitespace(ToLower(value));
}

bool FileExists(const std::string& path) {
  std::ifstream in(path);
  return in.good();
}

// The declaration's path, from the Bazel runfiles first and from the source
// tree second, so a simulated run and a test agree on the bytes.
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
  // Walk up from the working directory as well, because a binary launched from
  // a subdirectory (`bazel-bin`, a checkpoint directory) still has the tree
  // above it. The search is a path lookup, not a second source of truth: the
  // first file that exists is the declaration.
  std::string prefix;
  for (int level = 0; level < 4; ++level) {
    prefix += "../";
    candidates.push_back(prefix + relative);
  }
  for (const std::string& candidate : candidates) {
    if (FileExists(candidate)) return candidate;
  }
  return std::string();
}

// The mechanism mask of one `devices[]` entry's `features` array. A name this
// build cannot spell contributes no bit; the backend's own caps test is what
// rejects an unnameable declaration, because a backend may be built and tested
// without `src/` (DESIGN.md section 2.1).
std::uint64_t FeaturesMask(const json::Value& device) {
  const json::Value* features = device.Find("features");
  if (features == nullptr || !features->is_array()) return 0;
  std::uint64_t mask = 0;
  for (const json::Value& entry : features->array()) {
    if (!entry.is_string()) continue;
    std::uint64_t bit = 0;
    if (ParseCapabilityName(entry.string(), &bit)) mask |= bit;
  }
  return mask;
}

bool WantNumber(const json::Value& device, const char* name, double* out) {
  const json::Value* value = device.Find(name);
  if (value == nullptr || !value->is_number()) return false;
  *out = value->number();
  return true;
}

bool WantInt(const json::Value& device, const char* name, int* out) {
  const json::Value* value = device.Find(name);
  if (value == nullptr || !value->Integer(out)) return false;
  return true;
}

bool WantBool(const json::Value& device, const char* name, bool* out) {
  const json::Value* value = device.Find(name);
  if (value == nullptr || !value->is_bool()) return false;
  *out = value->boolean();
  return true;
}

// One declaration entry, before the capability fields are projected.
struct DeclaredDevice {
  std::string name;  // normalized join key
  bool is_device = false;
  std::uint64_t mask = 0;
  DeviceProfile profile;
};

// Reads one caps file into `out`. An unreadable or malformed declaration is
// skipped rather than fatal: a build that links no backend with a declaration
// has an empty table, which every caller already handles (an unknown name has
// no profile).
void ReadDeclaration(const char* backend, std::vector<DeclaredDevice>* out) {
  const std::string path = LocateCaps(backend);
  if (path.empty()) return;
  json::Value root;
  if (!json::ParseFile(path, &root, nullptr)) return;
  if (!root.is_object()) return;
  int vocabulary = 0;
  const json::Value* version = root.Find("capability_vocabulary");
  if (version == nullptr || !version->Integer(&vocabulary) || vocabulary != 1) {
    return;
  }
  const json::Value* devices = root.Find("devices");
  if (devices == nullptr || !devices->is_array()) return;
  for (const json::Value& entry : devices->array()) {
    if (!entry.is_object()) continue;
    const json::Value* name = entry.Find("name");
    if (name == nullptr || !name->is_string()) continue;
    DeclaredDevice declared;
    declared.name = Normalize(name->string());
    if (declared.name.empty()) continue;
    if (!WantBool(entry, "is_device", &declared.is_device)) {
      declared.is_device = false;
    }
    declared.mask = FeaturesMask(entry);

    DeviceProfile& profile = declared.profile;
    int major = 0;
    int minor = 0;
    WantInt(entry, "compute_major", &major);
    WantInt(entry, "compute_minor", &minor);
    profile.compute_major = major;
    profile.compute_minor = minor;
    double memory = 0.0;
    if (WantNumber(entry, "total_memory_bytes", &memory)) {
      profile.total_memory_bytes = static_cast<std::size_t>(memory);
    }
    bool flag = false;
    if (WantBool(entry, "has_tensor_cores", &flag))
      profile.has_tensor_cores = flag;
    if (WantBool(entry, "supports_bf16", &flag)) profile.supports_bf16 = flag;
    if (WantBool(entry, "supports_fp8", &flag)) profile.supports_fp8 = flag;
    double rate = 0.0;
    if (WantNumber(entry, "peak_fp32_flops", &rate))
      profile.peak_fp32_flops = rate;
    if (WantNumber(entry, "peak_bf16_flops", &rate))
      profile.peak_bf16_flops = rate;
    if (WantNumber(entry, "peak_fp8_flops", &rate))
      profile.peak_fp8_flops = rate;
    if (WantNumber(entry, "peak_bandwidth_bytes_per_second", &rate)) {
      profile.peak_bandwidth_bytes_per_second = rate;
    }
    int count = 0;
    if (WantInt(entry, "warp_size", &count)) profile.warp_size = count;
    if (WantInt(entry, "max_threads_per_block", &count)) {
      profile.max_threads_per_block = count;
    }
    if (WantNumber(entry, "max_shared_memory_per_block", &memory)) {
      profile.max_shared_memory_per_block = static_cast<std::size_t>(memory);
    }
    if (WantNumber(entry, "launch_us", &rate)) profile.launch_us = rate;
    out->push_back(std::move(declared));
  }
}

// The whole derived table, built once. The name strings live in `names`, which
// is filled before any profile points at it and is never modified afterwards,
// so `DeviceProfile::name` stays valid for the life of the process (the
// contract `nanochat/device_profile.h` states and `Caps::device_name` relies
// on).
struct DerivedTable {
  std::vector<std::string> names;
  std::vector<DeviceProfile> rows;
  std::vector<const DeviceProfile*> all;
  std::map<std::string, std::uint64_t> declared_masks;
};

const DerivedTable& Table() {
  static const DerivedTable kTable = [] {
    DerivedTable table;
    std::vector<DeclaredDevice> declared;
    for (int i = 0; i < kBackendCount; ++i) {
      ReadDeclaration(kBackendNames[i], &declared);
    }
    for (const DeclaredDevice& device : declared) {
      // The declaration of a device that is not an accelerator is still a
      // declaration: the planner reads its mask even though it contributes no
      // profile row (no launch cost, no HBM rate, no tensor cores).
      table.declared_masks[device.name] = device.mask;
      if (!device.is_device) continue;
      table.names.push_back(device.name);
    }
    // Ordered by descending capability, which is what the header promises: the
    // newest compute capability first, and the faster card first within one.
    std::sort(
        table.names.begin(), table.names.end(),
        [&declared](const std::string& a, const std::string& b) {
          const auto find = [&declared](const std::string& name) {
            for (const DeclaredDevice& device : declared) {
              if (device.name == name) return &device;
            }
            return static_cast<const DeclaredDevice*>(nullptr);
          };
          const DeclaredDevice* left = find(a);
          const DeclaredDevice* right = find(b);
          if (left == nullptr || right == nullptr) return a < b;
          if (left->profile.compute_major != right->profile.compute_major) {
            return left->profile.compute_major > right->profile.compute_major;
          }
          if (left->profile.compute_minor != right->profile.compute_minor) {
            return left->profile.compute_minor > right->profile.compute_minor;
          }
          return left->profile.peak_fp32_flops > right->profile.peak_fp32_flops;
        });
    table.rows.reserve(table.names.size());
    for (std::size_t i = 0; i < table.names.size(); ++i) {
      DeviceProfile profile;
      for (const DeclaredDevice& device : declared) {
        if (device.name == table.names[i]) {
          profile = device.profile;
          break;
        }
      }
      // The name points into `names`, which is final by now.
      profile.name = table.names[i].c_str();
      table.rows.push_back(profile);
    }
    table.all.reserve(table.rows.size());
    for (const DeviceProfile& profile : table.rows) {
      table.all.push_back(&profile);
    }
    return table;
  }();
  return kTable;
}

}  // namespace

const std::vector<const DeviceProfile*>& AllDeviceProfiles() {
  return Table().all;
}

const DeviceProfile* FindDeviceProfile(const std::string& name) {
  const std::string wanted = Normalize(name);
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

std::uint64_t DeclaredFeatureMaskForDevice(const std::string& device_name) {
  const DerivedTable& table = Table();
  const std::string wanted = Normalize(device_name);
  if (wanted.empty()) return 0;
  const auto exact = table.declared_masks.find(wanted);
  if (exact != table.declared_masks.end()) return exact->second;
  // The same substring fallback the profile lookup uses, so a vendor spelling
  // and the join key reach one declaration.
  if (const DeviceProfile* profile = FindDeviceProfile(wanted)) {
    const auto found = table.declared_masks.find(profile->name);
    if (found != table.declared_masks.end()) return found->second;
  }
  for (const auto& entry : table.declared_masks) {
    if (wanted.find(entry.first) != std::string::npos) return entry.second;
  }
  return 0;
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
  caps.supports_bf16 = profile.supports_bf16;
  caps.supports_fp8 = profile.supports_fp8;
  caps.device_name = profile.name;
  // The mechanism mask comes from the same declaration the profile does, so a
  // simulated device reports the bits its caps file names rather than a second
  // hand-written list (`spec/capability_bits.md` section 1).
  caps.feature_mask = DeclaredFeatureMaskForDevice(profile.name);
  return caps;
}

}  // namespace nanochat
