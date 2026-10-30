// The CUDA backend's fused-override registrations (include/nanochat/plan.h).
//
// A fused override is a single-kernel replacement for one composition. The
// composition defines the semantics and the override is an optimization
// (DESIGN.md section 3.1, docs/model-and-device-bringup.md section 7.4
// condition 1), so the two must satisfy the same golden trace within the
// committed tolerance. `backends/cuda/fused_override_trace_test.cc` is that
// gate: one trace comparison per family below, at the T1 tiny shapes, and the
// comparison has to survive before the fused family may be removed
// (docs/model-and-device-bringup.md section 7.6).
//
// The table is data. It names a composition, a family, and a capability bit,
// and nothing else: no graph, no block index, no kernel symbol, and no shape
// predicate. The plan matches on the composition, requires the bit to be set in
// `Caps::feature_mask`, and either selects the family or records it as a
// declined fallback. A registration whose bit is clear is ignored rather than
// rejected, so a backend can land an entry before its family passes the gate;
// the bit, not the entry, is the switch.
//
// This backend implements five fused families, and registers three of them.
// The other two are blocked on the capability vocabulary, not on the code:
//
//   * `value_gate` — `spec/capability_bits.md` bit 15 is `override_value_gate`,
//     and `include/nanochat/tensor.h` has no constant for it. A registration
//     under a bit that no header constant spells would never be set in any
//     `feature_mask`, so the planner would never resolve it: dead code that
//     looks live, which `spec/capability_bits.md` section 5 forbids. The
//     one-line follow-up is `kFeatureOverrideValueGate` in the header plus one
//     row in `backends/cuda/caps_names.h`; the trace test already covers the
//     family.
//   * `grad_norm` — vocabulary 1 has no override bit at all: `GradNorm` is a
//     composition with no fused counterpart in the partition
//     (docs/model-and-device-bringup.md section 7.3), and
//     `spec/capability_bits.md` section 5 names it explicitly. Nothing is
//     registered here, and the family keeps its trace comparison against the
//     composition all the same, so a later removal has the gate section 7.6
//     requires.
//
// `attention` is deliberately unregistered too: `include/nanochat/tensor.h`
// declares `kFeatureFusedAttention` (the v2 candidate of DESIGN.md section 3.1)
// and `spec/capability_bits.md` version 1 gives it no name, so no device could
// declare it and no caps file here does.
//
// See docs/kernels.md and backends/cuda/README.md.

#include <cstdio>
#include <cstdlib>

#include "backends/cuda/caps_names.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"

namespace nanochat {

namespace {

// One entry per family this backend implements *and* can gate. The `family`
// string is what the plan prints as the implementation it chose, so it is the
// name a reader of the plan sees: the family's own name, not a kernel symbol.
constexpr FusedOverride kFusedOverrides[] = {
    // residual add + RmsNorm, for the `rms_norm` composition. The fused kernel
    // is `backends/cuda/kernels/rms_norm.cu`, which also carries the fused
    // residual variant (`backends/cuda/kernels/rms_norm.h`).
    {"rms_norm", "rms_norm", kFeatureFusedResidualRmsNorm},
    // norm on head_dim + rope + scale, for the `qk_prep` composition. This is
    // the reference case DESIGN.md section 3.1 already requires the fused pair
    // to carry the same golden trace as.
    {"qk_prep", "qk_prep", kFeatureFusedQkPrep},
    // logit softcap + cross-entropy + the unpadded-vocabulary slice, for the
    // `classifier` composition.
    {"classifier", "classifier", kFeatureFusedClassifier},
};

constexpr int kFusedOverrideCount =
    static_cast<int>(sizeof(kFusedOverrides) / sizeof(kFusedOverrides[0]));

// Every entry is checked once, on the first registration read.
//
// The checks are the ones this translation unit can make on its own: a bit no
// vocabulary name maps to is a fatal error rather than a registration the
// planner silently skips, because that bit can never be set in any
// `feature_mask` and the entry would be dead code that looks live
// (`spec/capability_bits.md` section 5). The bit is spelled in the message so a
// reviewer can look it up.
//
// Membership of `composition` in `nanochat/plan.h`'s closed `CompositionKind`
// list is checked by the planner's own registration test rather than here: the
// spellings and the parser belong to `src/plan.cc`, and a backend that carried
// its own copy of the name list would be the second list this file exists to
// avoid. `fused_override_trace_test.cc` holds the composition strings to the
// compositions the trace comparison covers, so a renamed composition fails
// there.
void VerifyRegistrationsOnce() {
  static bool verified = false;
  if (verified) return;
  verified = true;
  for (int i = 0; i < kFusedOverrideCount; ++i) {
    const FusedOverride& entry = kFusedOverrides[i];
    if (entry.composition == nullptr || *entry.composition == '\0' ||
        entry.family == nullptr || *entry.family == '\0') {
      std::fprintf(stderr,
                   "fatal: fused override %d does not name both a composition "
                   "and a family; the plan matches on the first and reports "
                   "the second\n",
                   i);
      std::exit(1);
    }
    if (cuda_backend::CapabilityNameForBit(entry.capability) == nullptr) {
      std::fprintf(
          stderr,
          "fatal: fused override %d (family \"%s\") is gated on a "
          "capability bit that spec/capability_bits.md version 1 does "
          "not name (backends/cuda/caps_names.h has no row for it), so "
          "no device can set it and the plan would never resolve the "
          "entry\n",
          i, entry.family);
      std::exit(1);
    }
  }
}

}  // namespace

const FusedOverride* RegisteredFusedOverrides(int* count) {
  VerifyRegistrationsOnce();
  if (count != nullptr) *count = kFusedOverrideCount;
  return kFusedOverrides;
}

}  // namespace nanochat
