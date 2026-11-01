// The device-and-graph plan: `include/nanochat/plan.h`.
//
// The plan is where a model and a device meet, and it is the *only* place they
// meet (`spec/decisions.md` D-005). It is pure: it reads the `ModelSpec`, the
// graph kind, the device's capability answer and the backend's registration
// table, and it touches nothing else. It allocates no device memory, launches
// no kernel, warms no cache, and mutates no global state, so a bring-up check
// and a test can build a plan for a device that does not exist yet, and two
// calls with equal inputs return the same plan.
//
// Three rules shape every decision below.
//
//   1. A composition is the definition and an override is an optimization
//      (`spec/capability_bits.md` section 3, bits 12 to 15). So the composition
//      path is always placeable when the primitive tier is complete, and a
//      fused family wins only when the device declares its `override_*` bit and
//      the spec's policy allows one. The losing candidate is recorded, because
//      the plan is observable (`tools/nanochat doctor`) and a user who asks
//      "why did my path change" reads the same sentence whether the resolution
//      succeeded or failed.
//   2. A rejection names the missing bit by its vocabulary spelling and states
//      the consequence in the model's own terms
//      (`spec/capability_bits.md` section 4). It never says "unsupported
//      device": a device that lacks one bit is conformant for every other bit,
//      and that sentence hides which promise is missing.
//   3. The plan never changes the graph. It chooses an implementation for
//      pieces the graph already has, so a registration cannot add a block, an
//      edge, or a composition.
//
// The shape of a resolution, which is also the shape of the rejection text:
//
//   model requires:  ftp32, bf16, fp8_gemm
//   device offers:   fp32, bf16, fp8_storage
//   resolution:      rejected - missing fp8_gemm
//   consequence:     the fp8 layer set has no fp8 matmul on this device; the
//                    model declares no dense fallback for it

#include "nanochat/plan.h"

#include <cstring>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "src/capability_names.h"
#include "src/device_caps.h"
#include "src/plan_internal.h"

namespace nanochat {
namespace {

// The implementation string the plan reports for the composition path: the
// composition tier itself (`src/ops.cc`), which is the definition of every
// kind's semantics. A fused family reports its own name instead, so a reader of
// the plan sees which family ran and whether it was the definition or an
// optimization.
constexpr const char* kCompositionImplementation = "composition";

// The name table of `CompositionKind`. The spellings travel: a caps file, a
// registration and a plan report all use them verbatim.
struct CompositionRow {
  CompositionKind kind;
  const char* name;
};
constexpr CompositionRow kCompositions[] = {
    {CompositionKind::kEmbedding, "embedding"},
    {CompositionKind::kRmsNorm, "rms_norm"},
    {CompositionKind::kQkPrep, "qk_prep"},
    {CompositionKind::kAttention, "attention"},
    {CompositionKind::kMlp, "mlp"},
    {CompositionKind::kClassifier, "classifier"},
    {CompositionKind::kValueGate, "value_gate"},
    {CompositionKind::kGradNorm, "grad_norm"},
    {CompositionKind::kAdamW, "adam_w"},
    {CompositionKind::kMuon, "muon"},
};
constexpr int kCompositionCount =
    static_cast<int>(sizeof(kCompositions) / sizeof(kCompositions[0]));

// True when every block of the spec declares the value-gated residual *and* the
// embedding table that feeds it. The value gate is a composition the graph runs
// only in that case, so a plan for a model without value embeddings has nothing
// to place for it.
bool SpecRunsValueGate(const ModelSpec& spec) {
  if (!spec.embeddings.value_embedding) return false;
  for (const BlockSpec& block : spec.blocks) {
    if (block.residual == ResidualKind::kBlend) return true;
  }
  return false;
}

// True when `graph` runs `composition` at the spec's shapes. A forward-only
// graph runs no backward-only composition, and the optimizer compositions are
// not part of any inference graph.
bool GraphRunsComposition(GraphKind graph, CompositionKind composition,
                          const ModelSpec& spec) {
  const bool training = graph == GraphKind::kTrain;
  switch (composition) {
    case CompositionKind::kGradNorm:
    case CompositionKind::kAdamW:
    case CompositionKind::kMuon:
      return training;
    case CompositionKind::kValueGate:
      return SpecRunsValueGate(spec);
    case CompositionKind::kEmbedding:
    case CompositionKind::kRmsNorm:
    case CompositionKind::kQkPrep:
    case CompositionKind::kAttention:
    case CompositionKind::kMlp:
    case CompositionKind::kClassifier:
      return true;
  }
  return false;
}

// The device bits one composition needs. The primitive tier is the floor: every
// composition is built over the primitives, so an incomplete tier places
// nothing. The rest are the semantic bits the composition's arithmetic implies.
std::uint64_t RequiredBits(CompositionKind composition, const ModelSpec& spec) {
  const std::uint64_t tier =
      spec.policy.require_primitive_tier ? kFeaturePrimitiveTierComplete : 0;
  switch (composition) {
    case CompositionKind::kQkPrep:
      return tier | kFeatureRope;
    case CompositionKind::kAttention: {
      std::uint64_t need = tier;
      if (spec.dimensions.num_heads != spec.dimensions.num_kv_heads) {
        need |= kFeatureGqa;
      }
      for (const BlockSpec& block : spec.blocks) {
        if (block.attention == AttentionKind::kSlidingWindow) {
          need |= kFeatureSlidingWindow;
        }
      }
      return need;
    }
    case CompositionKind::kValueGate:
      return tier | kFeatureValueGate;
    case CompositionKind::kAdamW:
      return tier;
    case CompositionKind::kMuon:
      return tier;
    case CompositionKind::kEmbedding:
    case CompositionKind::kRmsNorm:
    case CompositionKind::kMlp:
    case CompositionKind::kClassifier:
    case CompositionKind::kGradNorm:
      return tier;
  }
  return tier;
}

// The registration whose composition is `name`, or null. The first match wins,
// so a backend lists its preferred family first.
const FusedOverride* FindRegistration(const FusedOverride* registrations,
                                      int count, const char* name) {
  if (registrations == nullptr) return nullptr;
  for (int i = 0; i < count; ++i) {
    const FusedOverride& entry = registrations[i];
    if (entry.composition == nullptr || entry.family == nullptr) continue;
    if (std::strcmp(entry.composition, name) == 0) return &entry;
  }
  return nullptr;
}

// The four-line rejection block of `spec/capability_bits.md` section 4. One
// consequence line per missing bit, so a reader learns every promise that is
// missing in the model's own terms.
std::string RejectionText(const ModelSpec& spec, const Caps& device,
                          std::uint64_t mask, std::uint64_t required,
                          std::uint64_t missing) {
  std::string text;
  text += "model requires:  " + FormatCapabilityList(required) + "\n";
  text += "device offers:   " + FormatCapabilityList(mask);
  if (device.device_name != nullptr) {
    text += std::string(" (") + device.device_name + ")";
  }
  text += "\n";
  text += "resolution:      rejected - missing " +
          FormatCapabilityList(missing) + "\n";
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    const std::uint64_t bit = kCapabilityNames[i].bit;
    if ((missing & bit) == 0) continue;
    const char* consequence = CapabilityConsequence(bit);
    text += std::string("consequence:     \"") + kCapabilityNames[i].name +
            "\" is not offered: ";
    text += consequence != nullptr ? consequence : "the model cannot run";
    text += "\n";
  }
  text += "spec:            " + spec.name + "\n";
  return text;
}

}  // namespace

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

const char* GraphKindName(GraphKind kind) {
  switch (kind) {
    case GraphKind::kTrain:
      return "train";
    case GraphKind::kPrefill:
      return "prefill";
    case GraphKind::kDecode:
      return "decode";
    case GraphKind::kEval:
      return "eval";
    case GraphKind::kGenerate:
      return "generate";
  }
  return "unknown";
}

const char* CompositionName(CompositionKind kind) {
  for (const CompositionRow& row : kCompositions) {
    if (row.kind == kind) return row.name;
  }
  return "unknown";
}

bool ParseCompositionName(const char* name, CompositionKind* out) {
  if (name == nullptr || out == nullptr) return false;
  for (const CompositionRow& row : kCompositions) {
    if (std::strcmp(row.name, name) == 0) {
      *out = row.kind;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// The device's answer
// ---------------------------------------------------------------------------

std::uint64_t EffectiveDeviceMask(const Caps& device) {
  if (device.feature_mask != 0) return device.feature_mask;
  // The reference backend reports the host and no mask; its declaration is
  // `backends/cpu/caps.json`, which is the single source for what it offers.
  if (device.device_name != nullptr) {
    return DeclaredFeatureMaskForDevice(device.device_name);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// BuildPlan
// ---------------------------------------------------------------------------

Plan BuildPlanWithDevice(const PlanRequest& request, const Caps& device,
                         const FusedOverride* registrations, int count,
                         std::string* why_not) {
  if (why_not != nullptr) why_not->clear();

  Plan plan;
  plan.graph = request.graph;
  plan.device_index = request.device_index;
  plan.ok = false;

  if (request.model == nullptr) {
    if (why_not != nullptr) *why_not = "no model spec was given to BuildPlan";
    return plan;
  }
  const ModelSpec& spec = *request.model;
  plan.spec_name = spec.name;

  std::string spec_error;
  if (!ValidateModelSpec(spec, &spec_error)) {
    if (why_not != nullptr) {
      *why_not = "the model spec is invalid: " + spec_error;
    }
    return plan;
  }

  const std::uint64_t mask = EffectiveDeviceMask(device);
  std::uint64_t required = spec.requires_features;
  if (spec.policy.require_primitive_tier) {
    required |= kFeaturePrimitiveTierComplete;
  }
  const std::uint64_t missing = required & ~mask;
  if (missing != 0) {
    // The model's own declaration cannot be satisfied: no plan exists, and the
    // reason is a named promise rather than a device verdict.
    if (why_not != nullptr) {
      *why_not = RejectionText(spec, device, mask, required, missing);
    }
    return plan;
  }

  // Every composition the graph runs gets one decision. They are emitted in the
  // closed order `CompositionKind` declares, so two plans for the same inputs
  // compare equal.
  for (const CompositionRow& row : kCompositions) {
    const CompositionKind composition = row.kind;
    if (!GraphRunsComposition(request.graph, composition, spec)) continue;

    CompositionChoice choice;
    choice.composition = composition;
    choice.fused = false;

    // The device bits this composition's arithmetic needs. The tier is implied
    // by the requires check above when the policy asks for it; the semantic
    // bits are repeated here so a per-graph composition (the optimizer entries)
    // is placed or reported on its own terms.
    const std::uint64_t need = RequiredBits(composition, spec);
    const std::uint64_t need_missing = need & ~mask;
    if (need_missing != 0) {
      PlanGap gap;
      gap.composition = composition;
      gap.reason = std::string("\"") + row.name +
                   "\" is not placed: this device does not declare " +
                   FormatCapabilityList(need_missing) +
                   ", and the composition "
                   "cannot be built over a primitive the device lacks";
      for (int i = 0; i < kCapabilityNameCount; ++i) {
        const std::uint64_t bit = kCapabilityNames[i].bit;
        if ((need_missing & bit) == 0) continue;
        const char* consequence = CapabilityConsequence(bit);
        if (consequence != nullptr)
          gap.reason += std::string("; ") + consequence;
      }
      plan.gaps.push_back(gap);
      continue;
    }

    // A fused family may win, but only when the spec allows an optimization and
    // the device declares the bit that gates it. The registration carries no
    // shape predicate by contract (`include/nanochat/plan.h`): the plan may not
    // re-decide shapes, so "the shapes match" means the composition is one this
    // graph runs at the spec's declared dimensions, which is the branch we are
    // already in.
    const FusedOverride* registration =
        FindRegistration(registrations, count, row.name);
    bool fused = false;
    if (registration != nullptr) {
      if (!spec.policy.allow_fused_overrides) {
        DeclinedImpl declined;
        declined.implementation = registration->family;
        declined.reason =
            "the spec's policy.allow_fused_overrides is false and "
            "an implementation choice is not a semantic";
        choice.declined.push_back(declined);
      } else if (registration->capability == 0 ||
                 (registration->capability & mask) !=
                     registration->capability) {
        DeclinedImpl declined;
        declined.implementation = registration->family;
        declined.reason = std::string("the device does not declare ") +
                          FormatCapabilityList(registration->capability) +
                          ", so this fused family is not offered here";
        choice.declined.push_back(declined);
      } else {
        // The family wins; the composition is the fallback that lost because a
        // faster implementation of the same meaning is available.
        DeclinedImpl declined;
        declined.implementation = kCompositionImplementation;
        declined.reason = std::string("the device declares ") +
                          FormatCapabilityList(registration->capability) +
                          " and the fused family \"" + registration->family +
                          "\" implements the same composition faster";
        choice.declined.push_back(declined);
        choice.implementation = registration->family;
        choice.fused = true;
        fused = true;
      }
    }
    if (!fused) {
      // The composition path: the tier is complete (checked above), so the
      // composition is placeable by construction.
      choice.implementation = kCompositionImplementation;
    }
    plan.choices.push_back(choice);
  }

  plan.ok = plan.gaps.empty();
  if (!plan.ok && why_not != nullptr) {
    std::string text = "the plan for \"" + spec.name + "\" on " +
                       (device.device_name != nullptr ? device.device_name
                                                      : "(unknown device)") +
                       " (" + GraphKindName(request.graph) +
                       ") has unplaced compositions:\n";
    for (const PlanGap& gap : plan.gaps) {
      text += "  " + gap.reason + "\n";
    }
    *why_not = text;
  }
  return plan;
}

// The empty registration list a backend with no fused family answers with.
//
// `include/nanochat/plan.h` declares this function and a backend that has fused
// families defines it (`backends/cuda/fused_overrides.cc`). A backend with none
// - the reference CPU backend today - defines nothing, which would leave the
// symbol undefined for every host binary that links the workflow. This weak
// definition is that answer: a build that links a backend's strong definition
// uses it instead, and a build that does not reports "no fused family", which
// is exactly what "compiles and links with no override code at all" means in
// the header comment.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
const FusedOverride* RegisteredFusedOverrides(int* count) {
  if (count != nullptr) *count = 0;
  return nullptr;
}

Plan BuildPlan(const PlanRequest& request, std::string* why_not) {
  const Caps device = kernels::GetCaps();
  int count = 0;
  const FusedOverride* registrations = RegisteredFusedOverrides(&count);
  return BuildPlanWithDevice(request, device, registrations, count, why_not);
}

}  // namespace nanochat
