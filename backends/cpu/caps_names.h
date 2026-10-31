#ifndef NANOCHAT_BACKENDS_CPU_CAPS_NAMES_H_
#define NANOCHAT_BACKENDS_CPU_CAPS_NAMES_H_

// The one place a capability *name* meets a capability *bit* in this directory.
//
// `spec/capability_bits.md` section 2 states the naming rule: the vocabulary
// name is the JSON string in `caps.json` and the C++ constant in
// `include/nanochat/tensor.h` is `kFeature` + upper camel case of the same
// name. The two lists are owned by different people and a one-character
// difference is a silent lie: the header's bit is never set, nobody notices,
// and the planner reads a device that offers a capability as a device that does
// not.
//
// This header therefore holds the mapping as *data* - one row per bit, both
// spellings on the row - so `backends/cpu/caps.json` is checked against the
// header rather than trusted, and so a declared string with no constant fails
// loudly instead of being dropped. `include/nanochat/tensor.h` is frozen for
// this campaign: the table records what the header spells, not what the rule
// would prefer, and it adds no alias.
//
// Two facts about the frozen header are recorded here rather than papered over:
//
//   * Five rows use the header's spelling, not the mechanical one:
//     `kFeatureAdamW` and `kFeatureMuon` carry `stateful_adamw` and
//     `stateful_muon`, and `kFeatureFusedResidualRmsNorm`,
//     `kFeatureFusedQkPrep` and `kFeatureFusedClassifier` carry
//     `override_rms_norm`, `override_qk_prep` and `override_classifier`.
//   * `override_value_gate` has no constant at all, so it has no row: a string
//     without a constant would be a promise the planner can never see. The row
//     arrives with the header addition `kFeatureOverrideValueGate`, and no
//     other line of this file changes when it does. `kFeatureFusedAttention`
//     (bit 13) has a constant but no vocabulary name, so it has no row either:
//     nothing in this backend fuses attention, and an unused constant is not a
//     capability.
//
// Backend-internal and host-only: no vendor header, no CUDA type, so the caps
// declaration check and a future derivation share one table.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "nanochat/tensor.h"

namespace nanochat {
namespace cpu_backend {

// One row of the capability vocabulary this backend can spell.
//
//   `name`      the vocabulary spelling, verbatim from
//               `spec/capability_bits.md` section 3 and from the JSON in
//               `backends/cpu/caps.json`. No alias, no abbreviation.
//   `bit`       the `kFeature*` constant from `include/nanochat/tensor.h`.
//   `constant`  the header's spelling of that constant, carried only so a
//               diagnostic can name the header symbol a reviewer has to look
//               at. It is a comment in string form and is never parsed.
struct CapabilityNameEntry {
  const char* name;
  std::uint64_t bit;
  const char* constant;
};

// The fourteen bits this backend can name today, in the order
// `spec/capability_bits.md` section 3 lists them.
inline constexpr CapabilityNameEntry kCapabilityNames[] = {
    // Architecture (bits 1 to 4).
    {"gqa", kFeatureGqa, "kFeatureGqa"},
    {"rope", kFeatureRope, "kFeatureRope"},
    {"sliding_window", kFeatureSlidingWindow, "kFeatureSlidingWindow"},
    {"value_gate", kFeatureValueGate, "kFeatureValueGate"},
    // Precision (bits 5 to 8).
    {"fp32", kFeatureFp32, "kFeatureFp32"},
    {"bf16", kFeatureBf16, "kFeatureBf16"},
    {"fp8_storage", kFeatureFp8Storage, "kFeatureFp8Storage"},
    {"fp8_gemm", kFeatureFp8Gemm, "kFeatureFp8Gemm"},
    // Tier (bit 9).
    {"primitive_tier_complete", kFeaturePrimitiveTierComplete,
     "kFeaturePrimitiveTierComplete"},
    // Stateful (bits 10 and 11). The header spells these `kFeatureAdamW` and
    // `kFeatureMuon`; those constants are the ones used, with no alias added.
    {"stateful_adamw", kFeatureAdamW, "kFeatureAdamW"},
    {"stateful_muon", kFeatureMuon, "kFeatureMuon"},
    // Override (bits 12 to 14). The header spells these `kFeatureFused*`; the
    // reference declares none of them, and the rows exist so the drift is
    // visible and so a future declaration cannot invent a third spelling.
    {"override_rms_norm", kFeatureFusedResidualRmsNorm,
     "kFeatureFusedResidualRmsNorm"},
    {"override_qk_prep", kFeatureFusedQkPrep, "kFeatureFusedQkPrep"},
    {"override_classifier", kFeatureFusedClassifier, "kFeatureFusedClassifier"},
};

// The number of rows, for a consumer that wants to report the size of the
// vocabulary it knows.
inline constexpr std::size_t kCapabilityNameCount =
    sizeof(kCapabilityNames) / sizeof(kCapabilityNames[0]);

// The bit a vocabulary name maps to, or 0 when this build cannot spell the
// name. Bit 0 is `kFeatureGqa`, so 0 is never a valid answer and a caller can
// use it as "unmapped" without a second out-parameter. A consumer must treat 0
// as a hard error and name the offending string: dropping a name silently is
// the failure mode this table exists to prevent.
inline std::uint64_t CapabilityBitForName(const char* name) {
  if (name == nullptr) return 0;
  for (const CapabilityNameEntry& entry : kCapabilityNames) {
    if (std::strcmp(entry.name, name) == 0) return entry.bit;
  }
  return 0;
}

}  // namespace cpu_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CPU_CAPS_NAMES_H_
