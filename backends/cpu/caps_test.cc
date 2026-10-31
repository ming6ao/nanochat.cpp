// backends/cpu/caps_test.cc -- the self-check of the reference declaration.
//
// `caps.json` is data, so nothing in it can be a compile error and nothing in
// it can be caught by the compiler. This test is the gate instead: every bit it
// declares is a name from `spec/capability_bits.md` section 3 **and** resolves
// to a `kFeature*` constant in the frozen `include/nanochat/tensor.h` through
// `caps_names.h`, the declaration carries no fused-override bit (the
// section 7.4 consequence: on this backend the planner resolves every
// composition to the composition path), and the utilities, the primitives, the
// attention variants, and the stateful entries the reference must supply are
// all present.
//
// The file is a flat object of scalars and arrays, so a scan is enough to read
// it and the backend gains no JSON dependency (DESIGN.md section 2.1: a backend
// depends on the seam and nothing above it).
//
// Run with `tools/nanochat test //backends/cpu:caps_test`.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "backends/cpu/caps_names.h"
#include "nanochat/sandbox.h"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// The declaration's path, from the Bazel runfiles first and from the source
// tree second, so the test runs both under `tools/nanochat test` and by hand.
std::string LocateCaps() {
  std::vector<std::string> candidates;
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/backends/cpu/caps.json");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/backends/cpu/caps.json");
  }
  candidates.emplace_back("backends/cpu/caps.json");
  candidates.emplace_back("../backends/cpu/caps.json");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate);
    if (in.good()) return candidate;
  }
  return std::string();
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The strings of the JSON array that follows `"key"`. An absent key yields an
// empty list, which every check below treats as a failure.
std::vector<std::string> Strings(const std::string& text,
                                 const std::string& key) {
  std::vector<std::string> out;
  const std::string quoted = "\"" + key + "\"";
  std::size_t pos = text.find(quoted);
  if (pos == std::string::npos) return out;
  pos = text.find('[', pos + quoted.size());
  if (pos == std::string::npos) return out;
  for (std::size_t i = pos + 1; i < text.size(); ++i) {
    if (text[i] == ']') break;
    if (text[i] != '"') continue;
    const std::size_t end = text.find('"', i + 1);
    if (end == std::string::npos) break;
    out.push_back(text.substr(i + 1, end - i - 1));
    i = end;
  }
  return out;
}

bool Contains(const std::vector<std::string>& list, const std::string& name) {
  for (const std::string& entry : list) {
    if (entry == name) return true;
  }
  return false;
}

// The vocabulary of `spec/capability_bits.md` section 3, in declaration order.
// The test carries the list because `spec/` is prose and the backend may not
// reach into it.
const char* const kVocabulary[] = {
    "gqa",
    "rope",
    "sliding_window",
    "value_gate",
    "fp32",
    "bf16",
    "fp8_storage",
    "fp8_gemm",
    "primitive_tier_complete",
    "stateful_adamw",
    "stateful_muon",
    "override_rms_norm",
    "override_qk_prep",
    "override_classifier",
    "override_value_gate",
};

// The feature set the reference declaration states: every mechanism of the
// vocabulary except the four fused overrides and the fp8 pair. A device that
// declares an override offers a single-kernel replacement; the reference
// deliberately offers none, so every composition is served by the composition
// path here. The fp8 pair is absent because this backend has no tensor-core fp8
// path (`supports_fp8` is false): its fp8 emulation is the oracle a GPU's fp8
// bits are measured against, not a mechanism this device claims to run.
const char* const kReferenceFeatures[] = {
    "gqa",           "rope", "sliding_window",          "value_gate",
    "fp32",          "bf16", "primitive_tier_complete", "stateful_adamw",
    "stateful_muon",
};

// The six device utilities (`docs/kernels.md`: Alloc, Free, Memcpy, Memset,
// Synchronize, GetCaps).
const char* const kUtilities[] = {
    "alloc", "free", "memcpy", "memset", "synchronize", "get_caps",
};

// The nine op codes of `spec/pointwise_ops.md` section 3.
const char* const kPointwiseCodes[] = {
    "kScale", "kScaleAdd", "kGateMul", "kReluSquare", "kSoftcap",
    "kExp",   "kRsqrt",    "kLog",     "kRowScale",
};

// The four combine ops of `ReduceOp` (`include/nanochat/kernels.h`).
const char* const kReduceOps[] = {
    "kSumSq",
    "kSum",
    "kMax",
    "kLogSumExp",
};

// The four families that are compositions on this backend. None of them may be
// declared as a primitive: a primitive is what a composition is built from, and
// these are what the composition tier builds.
const char* const kCompositionFamilies[] = {
    "rms_norm",
    "qk_prep",
    "classifier",
    "value_gate",
};

void CheckBits(const std::vector<std::string>& features) {
  Check(!features.empty(), "caps.json declares a features array");
  for (const std::string& feature : features) {
    bool known = false;
    for (const char* name : kVocabulary) {
      if (feature == name) known = true;
    }
    Check(known, "feature is a vocabulary name: " + feature);
    // The name has to reach a `kFeature*` constant in the frozen header. A
    // string with no constant is the silent lie `spec/capability_bits.md`
    // section 2 forbids: the planner would never see the bit.
    const std::uint64_t bit =
        nanochat::cpu_backend::CapabilityBitForName(feature.c_str());
    Check(bit != 0,
          "feature maps to a kFeature constant in include/nanochat/tensor.h "
          "through backends/cpu/caps_names.h: " +
              feature);
    Check(feature.compare(0, 9, "override_") != 0,
          "the reference declaration offers no fused override: " + feature);
  }
  // Set equality in both directions, so a typo, an addition, or a removal is a
  // failure rather than a silently different claim.
  for (const char* name : kReferenceFeatures) {
    Check(Contains(features, name), std::string("declared feature: ") + name);
  }
  Check(features.size() ==
            sizeof(kReferenceFeatures) / sizeof(kReferenceFeatures[0]),
        "caps.json declares exactly the reference feature set");
}

// The mapping table itself: one row per spellable bit, no duplicate name, no
// duplicate bit, and every row's constant is one the frozen header defines
// (which the initializer guarantees at compile time).
void CheckNameTable() {
  using nanochat::cpu_backend::kCapabilityNameCount;
  using nanochat::cpu_backend::kCapabilityNames;
  Check(kCapabilityNameCount == 14,
        "caps_names.h spells fourteen of the fifteen vocabulary bits");
  for (std::size_t i = 0; i < kCapabilityNameCount; ++i) {
    Check(
        kCapabilityNames[i].bit != 0,
        std::string("caps_names.h row has a bit: ") + kCapabilityNames[i].name);
    for (std::size_t j = i + 1; j < kCapabilityNameCount; ++j) {
      Check(std::string(kCapabilityNames[i].name) != kCapabilityNames[j].name,
            std::string("caps_names.h row is unique: ") +
                kCapabilityNames[i].name);
      Check(kCapabilityNames[i].bit != kCapabilityNames[j].bit,
            std::string("caps_names.h bit is unique: ") +
                kCapabilityNames[i].name);
    }
  }
  // The one vocabulary bit this build cannot spell, and the one constant with
  // no vocabulary name, are both absent on purpose.
  Check(nanochat::cpu_backend::CapabilityBitForName("override_value_gate") == 0,
        "caps_names.h has no constant for override_value_gate");
  Check(nanochat::cpu_backend::CapabilityBitForName("not_a_bit") == 0,
        "an unknown name maps to no bit");
  Check(nanochat::cpu_backend::CapabilityBitForName("stateful_adamw") ==
            nanochat::kFeatureAdamW,
        "stateful_adamw maps to the header's kFeatureAdamW");
  Check(nanochat::cpu_backend::CapabilityBitForName("stateful_muon") ==
            nanochat::kFeatureMuon,
        "stateful_muon maps to the header's kFeatureMuon");
}

void CheckUtilities(const std::vector<std::string>& utilities) {
  for (const char* name : kUtilities) {
    Check(Contains(utilities, name), std::string("declared utility: ") + name);
  }
  Check(utilities.size() == sizeof(kUtilities) / sizeof(kUtilities[0]),
        "caps.json declares exactly the six device utilities");
}

void CheckPrimitives(const std::vector<std::string>& primitives) {
  // Both reductions are required primitives (D-001): a device that omits one
  // declares no `primitive_tier_complete`.
  Check(Contains(primitives, "row_reduce"), "declared primitive: row_reduce");
  Check(Contains(primitives, "span_reduce"), "declared primitive: span_reduce");
  Check(Contains(primitives, "pointwise"), "declared primitive: pointwise");
  Check(Contains(primitives, "gemm"), "declared primitive: gemm");
  Check(Contains(primitives, "attention_forward"),
        "declared primitive: attention_forward");
  Check(Contains(primitives, "attention_backward"),
        "declared primitive: attention_backward");
  Check(Contains(primitives, "embedding_forward"),
        "declared primitive: embedding_forward");
  Check(Contains(primitives, "embedding_backward"),
        "declared primitive: embedding_backward");
  // A composition is not a primitive, and the reference does not pretend it is.
  for (const char* name : kCompositionFamilies) {
    Check(!Contains(primitives, name),
          std::string("a composition is not declared as a primitive: ") + name);
  }
}

void CheckNames(const std::string& text) {
  const std::vector<std::string> codes = Strings(text, "pointwise_codes");
  for (const char* name : kPointwiseCodes) {
    Check(Contains(codes, name), std::string("declared op code: ") + name);
  }
  Check(codes.size() == sizeof(kPointwiseCodes) / sizeof(kPointwiseCodes[0]),
        "caps.json declares all nine op codes");
  const std::vector<std::string> reduces = Strings(text, "reduce_ops");
  for (const char* name : kReduceOps) {
    Check(Contains(reduces, name), std::string("declared combine op: ") + name);
  }
  Check(reduces.size() == sizeof(kReduceOps) / sizeof(kReduceOps[0]),
        "caps.json declares all four combine ops");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  const std::string path = LocateCaps();
  Check(!path.empty(), "backends/cpu/caps.json is reachable");
  if (path.empty()) {
    std::printf("caps: the declaration was not found\n");
    return 1;
  }
  const std::string text = ReadFile(path);

  // The declaration's own identity and the two contract versions it belongs to
  // (spec/versions.md section 2): capability vocabulary 1 and kernel seam 2.
  // The file format is shared with every other backend's `caps.json`, so one
  // derivation reads all of them.
  Check(text.find("\"format\": \"nanochat.caps/1\"") != std::string::npos,
        "caps.json declares the shared nanochat.caps/1 format");
  Check(text.find("\"capability_vocabulary\": 1") != std::string::npos,
        "caps.json declares capability vocabulary 1");
  Check(text.find("\"backend\": \"cpu\"") != std::string::npos,
        "caps.json names the cpu backend");
  Check(text.find("\"seam\": 2") != std::string::npos,
        "caps.json declares kernel seam 2");
  Check(text.find("\"devices\"") != std::string::npos,
        "caps.json declares a devices array");

  CheckNameTable();
  CheckBits(Strings(text, "features"));
  CheckUtilities(Strings(text, "utilities"));
  CheckPrimitives(Strings(text, "primitives"));
  CheckNames(text);

  // The two stateful entries this device claims (bits 10 and 11).
  const std::vector<std::string> stateful = Strings(text, "stateful");
  Check(Contains(stateful, "adamw"), "declared stateful entry: adamw");
  Check(Contains(stateful, "muon"), "declared stateful entry: muon");
  Check(stateful.size() == 2,
        "caps.json declares exactly the two stateful entries");

  // Both attention variants at one tile size (`capability_bits.md` bit 3): the
  // variants differ only in the mask, so a device that runs one at a tile runs
  // the other at the same tile.
  const std::vector<std::string> variants = Strings(text, "variants");
  Check(Contains(variants, "full"), "declared attention variant: full");
  Check(Contains(variants, "sliding"), "declared attention variant: sliding");
  Check(variants.size() == 2, "caps.json declares exactly the two variants");
  Check(text.find("\"tile_rows\"") != std::string::npos,
        "caps.json declares the attention tile the variants run at");

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("cpu caps: all checks passed\n");
  return 0;
}
