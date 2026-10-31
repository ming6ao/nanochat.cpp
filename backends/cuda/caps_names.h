#ifndef NANOCHAT_BACKENDS_CUDA_CAPS_NAMES_H_
#define NANOCHAT_BACKENDS_CUDA_CAPS_NAMES_H_

// The one place a capability *name* meets a capability *bit* in this backend.
//
// `spec/capability_bits.md` section 2 states the naming rule: the vocabulary
// name is the JSON string in `caps.json`, and the C++ constant in
// `include/nanochat/tensor.h` is `kFeature` + upper camel case of the same
// name. The two lists are written by different owners, and a one-character
// difference is a silent lie: the header's bit is never set and nobody
// notices. This file therefore holds the mapping as *data* — one row per bit,
// both spellings on the row — so `backends/cuda/caps.json` can be checked
// against the header instead of trusted, and so an unmapped name fails loudly
// instead of being dropped.
//
// This header is backend-internal and host-only: it includes no CUDA header and
// no vendor type, so the CUDA runtime, a host test, and the caps declaration
// check all share one table.

#include <cstddef>
#include <cstdint>

#include "nanochat/tensor.h"

namespace nanochat {
namespace cuda_backend {

// One row of the capability vocabulary this backend can spell.
//
//   `name`      the vocabulary spelling, verbatim from
//               `spec/capability_bits.md` section 3 and from the JSON in
//               `backends/cuda/caps.json`. Nothing else may be used as a
//               string: no alias, no abbreviation.
//   `bit`       the `kFeature*` constant from `include/nanochat/tensor.h`.
//   `constant`  the header's spelling of that constant, carried only so a
//               diagnostic can name the header symbol a reviewer has to look
//               at. It is a comment in string form and is never parsed.
//
// The header's spelling is *not* the mechanical one for five of the rows below
// (see `kRenamedByHeader`), and one vocabulary bit has no header constant at
// all (see `HasVocabularyNameWithoutHeaderConstant`). Both are recorded here
// rather than papered over: the numbers are the header's, the strings are the
// spec's, and a reader must be able to see which name belongs to which bit
// without reading two files side by side.
struct CapabilityNameEntry {
  const char* name;
  std::uint64_t bit;
  const char* constant;
};

// The fourteen bits this backend can declare today.
//
// The vocabulary has fifteen. `override_value_gate` (bit 15) is absent because
// `include/nanochat/tensor.h` has no constant for it, and a name declared here
// without a constant would be a promise the planner can never see: the bit
// would never be set, so a registered override behind it would be dead code
// that looks live (`spec/capability_bits.md` section 5 explicitly forbids
// registering an override under a bit that does not exist). The row returns
// with the one-line header addition `kFeatureOverrideValueGate`; nothing else
// in this file changes when it does.
//
// Two further facts about the frozen header are recorded at the bottom of this
// file rather than here: the extra `kFeatureFusedAttention` constant, which has
// no vocabulary name in version 1, and the five constants whose spelling is not
// the mechanical `kFeature` + upper camel case form.
inline constexpr CapabilityNameEntry kCapabilityNames[] = {
    // Architecture (spec/capability_bits.md bits 1 to 4).
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
    // `kFeatureMuon`; the vocabulary names are `stateful_adamw` and
    // `stateful_muon`. Same bits, different spelling, and the row says so.
    {"stateful_adamw", kFeatureAdamW, "kFeatureAdamW"},
    {"stateful_muon", kFeatureMuon, "kFeatureMuon"},
    // Fused override (bits 12 to 14). The header spells these
    // `kFeatureFusedResidualRmsNorm`, `kFeatureFusedQkPrep` and
    // `kFeatureFusedClassifier`; the vocabulary names are `override_rms_norm`,
    // `override_qk_prep` and `override_classifier`.
    {"override_rms_norm", kFeatureFusedResidualRmsNorm,
     "kFeatureFusedResidualRmsNorm"},
    {"override_qk_prep", kFeatureFusedQkPrep, "kFeatureFusedQkPrep"},
    {"override_classifier", kFeatureFusedClassifier, "kFeatureFusedClassifier"},
};

// The constants above are the five whose header spelling differs from the
// mechanical rule. Listed once, as a comment, because the repair pass that
// renames them must edit `include/nanochat/tensor.h` and this table together:
//   stateful_adamw          kFeatureAdamW                  ->
//   kFeatureStatefulAdamw stateful_muon           kFeatureMuon ->
//   kFeatureStatefulMuon override_rms_norm       kFeatureFusedResidualRmsNorm
//   -> kFeatureOverrideRmsNorm override_qk_prep        kFeatureFusedQkPrep ->
//   kFeatureOverrideQkPrep override_classifier     kFeatureFusedClassifier ->
//   kFeatureOverrideClassifier
//
// `kFeatureFusedAttention` is a seventh divergence in the other direction: the
// header declares the bit and `spec/capability_bits.md` version 1 has no name
// for it, so no caps file may declare it and this backend registers no
// `attention` override. It is deliberately absent from the table above; the
// check below fails loudly if it ever reaches a device mask.

inline constexpr int kCapabilityNameCount =
    static_cast<int>(sizeof(kCapabilityNames) / sizeof(kCapabilityNames[0]));

// The bits the vocabulary can name. Anything outside this mask has no name and
// may not appear in a device's `Caps::feature_mask`.
constexpr std::uint64_t AllNamedCapabilityBits() {
  std::uint64_t all = 0;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    all |= kCapabilityNames[i].bit;
  }
  return all;
}

// True when `bit` is a single bit (and not zero).
constexpr bool IsSingleBit(std::uint64_t bit) {
  return bit != 0 && (bit & (bit - 1)) == 0;
}

// True when no two names claim one bit and no name is zero or multi-bit. The
// table is data, so the property is checked once at compile time instead of
// being believed.
constexpr bool CapabilityNamesAreDistinct() {
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (!IsSingleBit(kCapabilityNames[i].bit)) return false;
    for (int j = 0; j < i; ++j) {
      if (kCapabilityNames[i].bit == kCapabilityNames[j].bit) return false;
    }
  }
  return true;
}

static_assert(CapabilityNamesAreDistinct(),
              "backends/cuda/caps_names.h maps two vocabulary names to one "
              "capability bit, or maps a name to a zero or multi-bit constant. "
              "A duplicated row is the silent lie spec/capability_bits.md "
              "section 2 describes: fix the row, do not drop it.");

// Parses a vocabulary name. Returns false when the name is not in the table:
// an unknown name is an error, never rounded to the nearest known one
// (`spec/versions.md` section 3.1, "a reader must be able to see the name and
// refuse it by name"). `out` is left untouched on failure.
bool ParseCapabilityName(const char* name, std::uint64_t* out);

// The vocabulary name of `bit`, or null when the bit has no name. Used by
// diagnostics so a failure names what a reviewer can look up.
const char* CapabilityNameForBit(std::uint64_t bit);

// The header constant name for `bit`, or null. Diagnostics only.
const char* CapabilityConstantForBit(std::uint64_t bit);

// Reports, on stderr, every bit in `mask` that has no vocabulary name, then
// aborts. `what` names the source of the mask in the message. This is the
// startup check `backends/cuda/device.cu` runs on the mask it built, so a
// profile entry that carries an unnameable bit fails at the first `GetCaps()`
// rather than producing a plan nobody can explain.
void RequireNamedBits(std::uint64_t mask, const char* what);

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_CAPS_NAMES_H_
