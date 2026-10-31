// tests/plan_gate_test.cc -- the bring-up gate for the device-and-graph plan
// (docs/model-and-device-bringup.md section 9.1 rule 1 and section 16 P3).
//
// The gate is one sentence: **a plan is built under T0 CPU with no device
// present**. That is why `BuildPlan` is pure and why `src/plan_internal.h`
// exposes the fabricated-device seam: a plan can be decided before the silicon
// exists, and a *rejection* can be read before it costs a queue slot.
//
// Four things are asserted here, each with a fabricated capability mask and a
// fabricated registration table. No device is opened, no checkpoint is read,
// and no kernel runs.
//
//   1. A complete reference mask builds a plan under T0 CPU. Every composition
//      resolves to the composition path, because the reference declaration
//      (`backends/cpu/caps.json`) offers no `override_*` bit: with no fused
//      family declared, the composition is what runs
//      (`spec/capability_bits.md` section 3, bits 1 to 15).
//   2. A mask missing a bit the model requires is **rejected**, and the message
//      carries both the missing bit's vocabulary name and the consequence in
//      the model's own terms - never "unsupported device"
//      (`spec/capability_bits.md` section 4).
//   3. A mask that satisfies the model's `requires` but lacks a bit one
//      composition needs leaves that composition **unplaced**, and the gap
//      names the composition, the missing bit, and the consequence. A rejection
//      that says only "cannot run" is not a gate: the gap is the diagnosis.
//   4. A mask that declares an override bit resolves that composition to the
//      fused family and records the composition as the declined fallback, so
//      the report explains the choice whether the resolution succeeded or not.
//
// The fabricated device answers are passed through `BuildPlanWithDevice`
// (`src/plan_internal.h`), which is the same pure function `BuildPlan` calls
// after it reads `kernels::GetCaps()`.
//
// Run with `tools/nanochat test //tests:plan_gate_test`.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/model_spec.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"
#include "src/plan_internal.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void Check(bool ok, const std::string& what) {
  if (!ok) Fail(what);
}

// The bits the reference declaration offers (`backends/cpu/caps.json`): the
// mechanisms the flagship model's semantics need, the two stateful entries, and
// no `override_*` bit at all.
std::uint64_t ReferenceMask() {
  return nanochat::kFeatureGqa | nanochat::kFeatureRope |
         nanochat::kFeatureSlidingWindow | nanochat::kFeatureValueGate |
         nanochat::kFeatureFp32 | nanochat::kFeatureBf16 |
         nanochat::kFeaturePrimitiveTierComplete | nanochat::kFeatureAdamW |
         nanochat::kFeatureMuon;
}

// A fabricated device answer. `is_device` is false and `has_cublas` is false:
// this is the T0 CPU host, with no accelerator attached, which is the only
// environment the gate runs in.
nanochat::Caps FabricatedDevice(std::uint64_t mask,
                                const char* device_name = "cpu") {
  nanochat::Caps caps;
  caps.device_index = 0;
  caps.is_device = false;
  caps.has_cublas = false;
  caps.has_tensor_cores = false;
  caps.supports_fp32 = true;
  caps.supports_bf16 = (mask & nanochat::kFeatureBf16) != 0;
  caps.supports_fp8 = (mask & nanochat::kFeatureFp8Gemm) != 0;
  caps.device_name = device_name;
  caps.feature_mask = mask;
  return caps;
}

const nanochat::CompositionChoice* FindChoice(const nanochat::Plan& plan,
                                              nanochat::CompositionKind kind) {
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    if (choice.composition == kind) return &choice;
  }
  return nullptr;
}

bool Mentions(const std::vector<nanochat::DeclinedImpl>& declined,
              const std::string& implementation) {
  for (const nanochat::DeclinedImpl& entry : declined) {
    if (entry.implementation == implementation) return true;
  }
  return false;
}

const nanochat::DeclinedImpl* FindDeclined(
    const std::vector<nanochat::DeclinedImpl>& declined,
    const std::string& implementation) {
  for (const nanochat::DeclinedImpl& entry : declined) {
    if (entry.implementation == implementation) return &entry;
  }
  return nullptr;
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// 1. A complete reference mask builds a plan under T0 CPU with no device.
void CheckReferenceMaskBuildsAPlan() {
  const nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  request.device_index = -1;  // the backend's default device: none is present

  std::string why_not;
  const nanochat::Caps device = FabricatedDevice(ReferenceMask());
  Check(!device.is_device && !device.has_cublas,
        "the fabricated device is the host, with no accelerator and no GEMM "
        "library");
  const nanochat::Plan plan =
      nanochat::BuildPlanWithDevice(request, device, nullptr, 0, &why_not);
  Check(plan.ok, "a complete reference mask builds a plan: " + why_not);
  Check(plan.gaps.empty(), "the reference mask leaves no composition unplaced");
  Check(!plan.choices.empty(),
        "the plan carries one decision per composition the graph runs");
  Check(plan.spec_name == spec.name,
        "the plan names the spec it was built for");

  // Every composition resolves to the composition tier: the reference
  // declaration offers no override bit, so the composition path is what runs.
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    Check(!choice.fused,
          std::string("composition ") +
              nanochat::CompositionName(choice.composition) +
              " runs the composition path when no override bit is declared");
    Check(choice.implementation == "composition",
          std::string("composition ") +
              nanochat::CompositionName(choice.composition) +
              " spells the composition tier");
    Check(choice.declined.empty(),
          std::string("composition ") +
              nanochat::CompositionName(choice.composition) +
              " declines nothing when the backend registers no fused family");
  }

  // A forward-only graph runs no optimizer composition.
  nanochat::PlanRequest prefill = request;
  prefill.graph = nanochat::GraphKind::kPrefill;
  const nanochat::Plan inference =
      nanochat::BuildPlanWithDevice(prefill, device, nullptr, 0, nullptr);
  Check(inference.ok, "the prefill graph builds a plan under the same mask");
  Check(FindChoice(inference, nanochat::CompositionKind::kAdamW) == nullptr,
        "a forward-only graph places no adam_w composition");
  Check(FindChoice(inference, nanochat::CompositionKind::kMuon) == nullptr,
        "a forward-only graph places no muon composition");

  // The plan is a pure function of its inputs: two calls agree, decision for
  // decision, so a bring-up check can compare two plans.
  const nanochat::Plan again = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(ReferenceMask()), nullptr, 0, nullptr);
  Check(again.choices.size() == plan.choices.size(),
        "two calls with equal inputs return the same number of decisions");
  for (std::size_t i = 0; i < again.choices.size() && i < plan.choices.size();
       ++i) {
    Check(
        again.choices[i].composition == plan.choices[i].composition &&
            again.choices[i].implementation == plan.choices[i].implementation &&
            again.choices[i].fused == plan.choices[i].fused,
        "two calls with equal inputs agree on every decision");
  }
}

// 2. A mask that does not satisfy the spec's `requires` is a named rejection.
void CheckMissingBitIsRejected() {
  nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  spec.name = "requires-bf16";
  spec.requires_features |= nanochat::kFeatureBf16;
  const std::uint64_t mask = ReferenceMask() & ~nanochat::kFeatureBf16;

  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  std::string why_not;
  const nanochat::Plan plan = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(mask), nullptr, 0, &why_not);
  Check(!plan.ok, "a mask missing bf16 is rejected");
  Check(plan.gaps.empty(),
        "a rejection at the requires check places nothing, so it reports no "
        "gap");
  Check(Contains(why_not, "bf16"),
        "the rejection names the missing bit by its vocabulary spelling: " +
            why_not);
  Check(Contains(why_not, "consequence"),
        "the rejection states a consequence: " + why_not);
  Check(Contains(why_not, "bfloat16 path"),
        "the consequence is the vocabulary entry's, in the model's terms: " +
            why_not);
  Check(Contains(why_not, "resolution"),
        "the rejection has the resolution line: " + why_not);
  Check(Contains(why_not, "model requires:"),
        "the rejection restates what the model requires: " + why_not);
  Check(!Contains(why_not, "unsupported device"),
        "the rejection never says \"unsupported device\": " + why_not);
  Check(
      plan.choices.empty(),
      "a rejected plan places nothing, because no composition can be trusted");

  // The same model on the complete mask resolves, so the rejection is about the
  // bit and not about the model.
  const nanochat::Plan accepted = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(ReferenceMask() | nanochat::kFeatureBf16),
      nullptr, 0, nullptr);
  Check(accepted.ok, "the same model resolves once the device declares bf16");
}

// 3. A composition whose own bit is missing is reported as a named gap.
void CheckUnplacedCompositionNamesItsGap() {
  // The model does not require rope, so the `requires` check passes; the
  // qk_prep composition still needs it. This is the case a "the model declares
  // it, so it must work" gate misses.
  nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  spec.name = "composition-gap-probe";
  spec.requires_features &= ~nanochat::kFeatureRope;
  const std::uint64_t mask = ReferenceMask() & ~nanochat::kFeatureRope;

  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  std::string why_not;
  const nanochat::Plan plan = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(mask), nullptr, 0, &why_not);
  Check(!plan.ok,
        "a mask missing rope leaves the qk_prep composition unplaced");
  Check(!plan.gaps.empty(), "the rejection reports the unplaced composition");
  Check(Contains(why_not, "unplaced compositions"),
        "the summary names the unplaced compositions: " + why_not);
  bool named_qk_prep = false;
  for (const nanochat::PlanGap& gap : plan.gaps) {
    if (gap.composition != nanochat::CompositionKind::kQkPrep) continue;
    named_qk_prep = true;
    Check(Contains(gap.reason, "qk_prep"),
          "the gap names the composition that could not be placed: " +
              gap.reason);
    Check(Contains(gap.reason, "rope"),
          "the gap names the missing bit: " + gap.reason);
    Check(Contains(gap.reason, "rotary path"),
          "the gap states the consequence in the model's terms: " + gap.reason);
    Check(!Contains(gap.reason, "unsupported device"),
          "the gap never says \"unsupported device\": " + gap.reason);
  }
  Check(named_qk_prep,
        "the qk_prep composition is the one the mask cannot place");
  // Every other composition is still placed: a gap is one composition's
  // verdict, not a whole-plan refusal.
  Check(FindChoice(plan, nanochat::CompositionKind::kMlp) != nullptr,
        "the compositions that do not need rope are still placed");
  Check(FindChoice(plan, nanochat::CompositionKind::kQkPrep) == nullptr,
        "the unplaced composition carries no choice");
}

// 4. A declared override bit selects the fused family and records the
// composition as the declined fallback.
void CheckOverrideBitSelectsTheFusedFamily() {
  const nanochat::FusedOverride registrations[] = {
      {"rms_norm", "rms_norm", nanochat::kFeatureFusedResidualRmsNorm},
  };
  nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  spec.name = "override-probe";

  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;

  // (a) The bit is declared: the family wins, the composition is recorded.
  const nanochat::Plan fused = nanochat::BuildPlanWithDevice(
      request,
      FabricatedDevice(ReferenceMask() |
                       nanochat::kFeatureFusedResidualRmsNorm),
      registrations, 1, nullptr);
  Check(fused.ok, "a mask declaring the override bit builds a plan");
  const nanochat::CompositionChoice* choice =
      FindChoice(fused, nanochat::CompositionKind::kRmsNorm);
  Check(choice != nullptr, "the plan decides the rms_norm composition");
  if (choice != nullptr) {
    Check(choice->fused, "the declared override bit selects the fused family");
    Check(
        choice->implementation == "rms_norm",
        "the plan names the fused family that won: " + choice->implementation);
    Check(Mentions(choice->declined, "composition"),
          "the composition path is recorded as the declined fallback");
    const nanochat::DeclinedImpl* declined =
        FindDeclined(choice->declined, "composition");
    Check(
        declined != nullptr && Contains(declined->reason, "override_rms_norm"),
        "the declined entry names the capability bit that decided it");
  }

  // (b) The bit is absent: the registration is recorded as a declined fallback
  // and the composition runs, so a backend can land a family before its gate.
  const nanochat::Plan composed = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(ReferenceMask()), registrations, 1, nullptr);
  Check(composed.ok, "the same plan builds with the override bit absent");
  choice = FindChoice(composed, nanochat::CompositionKind::kRmsNorm);
  Check(choice != nullptr, "the plan still decides the rms_norm composition");
  if (choice != nullptr) {
    Check(!choice->fused,
          "a clear override bit leaves the composition running");
    Check(choice->implementation == "composition",
          "the composition path is what runs with no override bit");
    Check(Mentions(choice->declined, "rms_norm"),
          "the unselected fused family is recorded as a declined fallback");
    const nanochat::DeclinedImpl* declined =
        FindDeclined(choice->declined, "rms_norm");
    Check(
        declined != nullptr && Contains(declined->reason, "override_rms_norm"),
        "the declined family's reason names the bit the device lacks");
  }

  // (c) A spec that forbids optimizations keeps the composition even when the
  // bit is declared: an implementation choice is not a semantic.
  nanochat::ModelSpec policy_off = spec;
  policy_off.policy.allow_fused_overrides = false;
  nanochat::PlanRequest strict = request;
  strict.model = &policy_off;
  const nanochat::Plan forced = nanochat::BuildPlanWithDevice(
      strict,
      FabricatedDevice(ReferenceMask() |
                       nanochat::kFeatureFusedResidualRmsNorm),
      registrations, 1, nullptr);
  Check(forced.ok, "the policy-off spec still builds a plan");
  choice = FindChoice(forced, nanochat::CompositionKind::kRmsNorm);
  Check(choice != nullptr && !choice->fused,
        "policy.allow_fused_overrides = false keeps the composition");
}

// 5. A device that reports no mask at all is read through its declaration: the
// reference backend reports the host and zero bits, and silence is not a
// statement that the device offers nothing.
void CheckSilentDeviceReadsItsDeclaration() {
  const nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;

  // A zero mask with the reference device's name: the declaration in
  // `backends/cpu/caps.json` is the answer.
  const nanochat::Plan declared = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(0), nullptr, 0, nullptr);
  Check(
      declared.ok,
      "a device that declares no mask is read through backends/cpu/caps.json");
  Check(nanochat::EffectiveDeviceMask(FabricatedDevice(0)) != 0,
        "the effective mask falls back to the declaration rather than to zero");

  // A device with no name and no mask has declared nothing, and the tier bit it
  // is missing is named.
  std::string why_not;
  const nanochat::Plan anonymous = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(0, nullptr), nullptr, 0, &why_not);
  Check(!anonymous.ok,
        "a device that names nothing and declares nothing cannot build a plan");
  Check(Contains(why_not, "primitive_tier_complete"),
        "the rejection names the missing tier bit: " + why_not);
}

// The plan's own name table: a caps file and a registration speak these
// spellings, so a rename is a contract change and this check is what notices.
void CheckCompositionNames() {
  const nanochat::CompositionKind kinds[] = {
      nanochat::CompositionKind::kEmbedding,
      nanochat::CompositionKind::kRmsNorm,
      nanochat::CompositionKind::kQkPrep,
      nanochat::CompositionKind::kAttention,
      nanochat::CompositionKind::kMlp,
      nanochat::CompositionKind::kClassifier,
      nanochat::CompositionKind::kValueGate,
      nanochat::CompositionKind::kGradNorm,
      nanochat::CompositionKind::kAdamW,
      nanochat::CompositionKind::kMuon};
  for (nanochat::CompositionKind kind : kinds) {
    const char* name = nanochat::CompositionName(kind);
    nanochat::CompositionKind parsed = nanochat::CompositionKind::kEmbedding;
    Check(nanochat::ParseCompositionName(name, &parsed) && parsed == kind,
          std::string("the composition name \"") + name + "\" parses back");
  }
  nanochat::CompositionKind unused = nanochat::CompositionKind::kEmbedding;
  Check(!nanochat::ParseCompositionName("not_a_composition", &unused),
        "an unknown composition name is refused, not rounded");
}

}  // namespace

int main() {
  CheckReferenceMaskBuildsAPlan();
  CheckMissingBitIsRejected();
  CheckUnplacedCompositionNamesItsGap();
  CheckOverrideBitSelectsTheFusedFamily();
  CheckSilentDeviceReadsItsDeclaration();
  CheckCompositionNames();

  if (g_failures != 0) {
    std::fprintf(stderr, "plan_gate_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("plan_gate_test: all checks passed\n");
  return 0;
}
