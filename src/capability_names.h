#ifndef NANOCHAT_SRC_CAPABILITY_NAMES_H_
#define NANOCHAT_SRC_CAPABILITY_NAMES_H_

#include <cstdint>
#include <string>

#include "nanochat/tensor.h"

// The one place a capability *name* meets a capability *bit* above the kernel
// seam.
//
// `spec/capability_bits.md` section 2 states the naming rule: the vocabulary
// name is the string a `caps.json` and a model's `requires` carry, and the C++
// constant lives in `include/nanochat/tensor.h`. A backend owns its own copy of
// the mapping (`backends/<name>/caps_names.h`, backend-internal, checked by
// that backend's caps test). This table is the copy the *workflow* needs, and
// it cannot borrow one from a backend: `src/` compiles against either backend,
// so including `backends/cuda/caps_names.h` would point the workflow at one of
// them and would break the backend-selection boundary in `nanochat.bzl`.
//
// The two consumers are the ones that read a *string* rather than a constant:
// the spec reader, which turns `requires` into `ModelSpec::requires_features`
// (`src/model_spec.cc`), and the planner, whose rejection message must name the
// missing bits by their vocabulary spelling and state the consequence in the
// model's own terms (`src/plan.cc`).
//
// The header's spelling is not mechanical for five rows (`kFeatureAdamW`,
// `kFeatureMuon`, `kFeatureFusedResidualRmsNorm`, `kFeatureFusedQkPrep`,
// `kFeatureFusedClassifier`), and `override_value_gate` has no constant at all
// in the frozen header. Both facts are recorded on the row rather than papered
// over: the table spells the header's constants, and the missing constant is
// one line of header addition reported by this campaign's node report.
//
// This header is private to `src/`.

namespace nanochat {

struct CapabilityNameEntry {
  const char* name;      // the vocabulary spelling, verbatim
  std::uint64_t bit;     // the kFeature* constant
  const char* constant;  // the header spelling, for diagnostics only
  // The consequence of the model requiring this bit on a device that does not
  // declare it, in the model's own terms (`spec/capability_bits.md` section 4).
  // A rejection names the missing bit and this sentence, and never says
  // "unsupported device".
  const char* consequence;
};

// The fourteen bits this build can spell, in the order
// `spec/capability_bits.md` section 3 lists them.
inline constexpr CapabilityNameEntry kCapabilityNames[] = {
    {"gqa", kFeatureGqa, "kFeatureGqa",
     "group-query attention has no group mapping, so a block whose key/value "
     "heads are fewer than its query heads cannot run"},
    {"rope", kFeatureRope, "kFeatureRope",
     "the position blocks have no rotary path, so the query and key rows "
     "cannot be rotated by their sequence index"},
    {"sliding_window", kFeatureSlidingWindow, "kFeatureSlidingWindow",
     "the blocks that declare a windowed mask would have to run full context, "
     "which is a different meaning than the one the spec declares"},
    {"value_gate", kFeatureValueGate, "kFeatureValueGate",
     "the value-gated residual has no gate, so its value rows cannot be "
     "formed"},
    {"fp32", kFeatureFp32, "kFeatureFp32",
     "the model's numeric contract assumes single-precision arithmetic, which "
     "this device does not compute in"},
    {"bf16", kFeatureBf16, "kFeatureBf16",
     "the model's declared arithmetic precision has no bfloat16 path on this "
     "device, so the graph cannot run at the precision it declares"},
    {"fp8_storage", kFeatureFp8Storage, "kFeatureFp8Storage",
     "the model's numeric contract is an fp8 contract, and this device cannot "
     "produce one-byte operands to feed it"},
    {"fp8_gemm", kFeatureFp8Gemm, "kFeatureFp8Gemm",
     "the fp8 layer set has no fp8 matmul on this device, and the model "
     "declares no dense fallback for it"},
    {"primitive_tier_complete", kFeaturePrimitiveTierComplete,
     "kFeaturePrimitiveTierComplete",
     "the primitive tier is incomplete, so no composition can be resolved to "
     "a primitive this device supplies and no plan can be built at all"},
    {"stateful_adamw", kFeatureAdamW, "kFeatureAdamW",
     "the training graph has no stateful AdamW update on this device, so the "
     "model can be evaluated but not trained"},
    {"stateful_muon", kFeatureMuon, "kFeatureMuon",
     "the training graph has no stateful Muon update on this device, so the "
     "matrix parameters cannot be trained"},
    {"override_rms_norm", kFeatureFusedResidualRmsNorm,
     "kFeatureFusedResidualRmsNorm",
     "the fused residual-plus-norm family is not offered; the rms_norm "
     "composition runs instead, which is the same meaning at a different "
     "speed"},
    {"override_qk_prep", kFeatureFusedQkPrep, "kFeatureFusedQkPrep",
     "the fused norm-rotate-scale family is not offered; the qk_prep "
     "composition runs instead, which is the same meaning at a different "
     "speed"},
    {"override_classifier", kFeatureFusedClassifier, "kFeatureFusedClassifier",
     "the fused classifier family is not offered; the classifier composition "
     "runs instead, which is the same meaning at a different speed"},
};

inline constexpr int kCapabilityNameCount =
    static_cast<int>(sizeof(kCapabilityNames) / sizeof(kCapabilityNames[0]));

// Every bit this build can name. A mask outside it carries a bit no vocabulary
// entry describes, which is a run-time lie a caller must reject by name rather
// than ignore.
std::uint64_t AllNamedCapabilityBits();

// Parses a vocabulary name. Returns false for a name this build does not
// define, and never rounds an unknown name to a known one
// (`spec/versions.md` section 3.1). `*out` is untouched on failure.
bool ParseCapabilityName(const std::string& name, std::uint64_t* out);

// The vocabulary name of `bit`, or null when the bit has no name.
const char* CapabilityNameForBit(std::uint64_t bit);

// The header constant spelling of `bit`, for diagnostics. Null when the bit
// has no row.
const char* CapabilityConstantForBit(std::uint64_t bit);

// The model-terms consequence of `bit` being missing (see the table above).
// Null when the bit has no row.
const char* CapabilityConsequence(std::uint64_t bit);

// True when `bit` is one of the override bits: a model never requires one,
// because an implementation choice is not a semantic
// (`spec/capability_bits.md` section 3).
bool IsOverrideCapability(std::uint64_t bit);

// True when `bit` is one of the stateful training bits.
bool IsStatefulCapability(std::uint64_t bit);

// "rope, sliding_window" for the bits set in `mask`, in vocabulary order. An
// empty mask spells "(none)". The spelling is what a rejection and a plan
// report print, so it is the vocabulary name and not the constant.
std::string FormatCapabilityList(std::uint64_t mask);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_CAPABILITY_NAMES_H_
