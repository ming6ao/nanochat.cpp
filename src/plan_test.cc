// src/plan_test.cc -- the planner's gate (docs/model-and-device-bringup.md
// section 9.1 rule 1, section 16 P3).
//
// The three cases below are the three things a plan can do, each with a
// *fabricated* capability mask: no device is present, no checkpoint is opened,
// and no kernel runs. That is the whole reason `BuildPlan` is pure - a bring-up
// can be tested before the silicon exists.
//
//   1. A complete reference mask builds a plan under T0 CPU. Every composition
//      resolves to the composition path, because the reference declaration
//      offers no `override_*` bit: with no fused family declared, the
//      composition is what runs (`spec/capability_bits.md` section 3, bit 12).
//   2. A mask missing a bit the model requires is rejected, and the message
//      carries **both** the missing bit's vocabulary name and the consequence in
//      the model's own terms - never "unsupported device"
//      (`spec/capability_bits.md` section 4).
//   3. A mask that declares an override bit resolves that composition to the
//      fused family and records the composition as a declined fallback, so the
//      report explains the choice whether the resolution succeeded or not.
//
// The fabricated device answers are passed through `BuildPlanWithDevice`
// (`src/plan_internal.h`), which is the same pure function `BuildPlan` calls
// after it reads `kernels::GetCaps()`.
//
// Run with `tools/nanochat test //src:plan_test`.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/model_spec.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"
#include "src/capability_names.h"
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

nanochat::Caps FabricatedDevice(std::uint64_t mask) {
  nanochat::Caps caps;
  caps.device_index = 0;
  caps.is_device = false;  // T0: the host, with no accelerator attached
  caps.has_cublas = false;
  caps.has_tensor_cores = false;
  caps.supports_fp32 = true;
  caps.supports_bf16 = (mask & nanochat::kFeatureBf16) != 0;
  caps.supports_fp8 = false;
  caps.device_name = "cpu";
  caps.feature_mask = mask;
  return caps;
}

const nanochat::CompositionChoice* FindChoice(
    const nanochat::Plan& plan, nanochat::CompositionKind kind) {
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

// Case 1: a complete reference mask, no device, every composition placed on the
// composition path.
void CheckReferenceMaskBuildsAPlan() {
  const nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  request.device_index = 0;
  std::string why_not;
  const nanochat::Plan plan =
      nanochat::BuildPlanWithDevice(request, FabricatedDevice(ReferenceMask()),
                                    nullptr, 0, &why_not);
  Check(plan.ok, "a complete reference mask builds a plan: " + why_not);
  Check(plan.gaps.empty(), "the reference mask leaves no composition unplaced");
  Check(!plan.choices.empty(), "the plan carries one decision per composition");
  Check(plan.spec_name == spec.name, "the plan names the spec it was built for");

  // Every composition resolves to the composition tier: `caps.json` declares no
  // override bit, so the composition path is what runs.
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    Check(!choice.fused,
          std::string("composition ") +
              nanochat::CompositionName(choice.composition) +
              " runs the composition path when no override bit is declared");
    Check(choice.implementation == "composition",
          std::string("composition ") +
              nanochat::CompositionName(choice.composition) +
              " spells the composition tier");
  }

  // A forward-only graph runs no optimizer composition.
  nanochat::PlanRequest prefill = request;
  prefill.graph = nanochat::GraphKind::kPrefill;
  const nanochat::Plan inference =
      nanochat::BuildPlanWithDevice(prefill, FabricatedDevice(ReferenceMask()),
                                    nullptr, 0, nullptr);
  Check(inference.ok, "the prefill graph builds a plan");
  Check(FindChoice(inference, nanochat::CompositionKind::kAdamW) == nullptr,
        "a forward-only graph places no adam_w composition");

  // The plan is a pure function of its inputs: two calls agree, decision for
  // decision.
  const nanochat::Plan again =
      nanochat::BuildPlanWithDevice(request, FabricatedDevice(ReferenceMask()),
                                    nullptr, 0, nullptr);
  Check(again.choices.size() == plan.choices.size(),
        "two calls with equal inputs return the same number of decisions");
  for (std::size_t i = 0; i < again.choices.size() && i < plan.choices.size();
       ++i) {
    Check(again.choices[i].composition == plan.choices[i].composition &&
              again.choices[i].implementation ==
                  plan.choices[i].implementation &&
              again.choices[i].fused == plan.choices[i].fused,
          "two calls with equal inputs agree on every decision");
  }
}

// Case 2: a mask missing a required bit is a named rejection carrying the bit and
// the consequence.
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
  Check(why_not.find("bf16") != std::string::npos,
        "the rejection names the missing bit: " + why_not);
  Check(why_not.find("consequence") != std::string::npos,
        "the rejection states a consequence: " + why_not);
  Check(why_not.find("bfloat16 path") != std::string::npos,
        "the consequence is the vocabulary entry's, in the model's terms: " +
            why_not);
  Check(why_not.find("resolution") != std::string::npos,
        "the rejection has the resolution line: " + why_not);
  Check(why_not.find("unsupported device") == std::string::npos,
        "the rejection never says \"unsupported device\"");
  Check(plan.choices.empty(),
        "a rejected plan places nothing, because no composition can be trusted");

  // The same model on the complete mask resolves, so the rejection is about the
  // bit and not about the model.
  const nanochat::Plan accepted = nanochat::BuildPlanWithDevice(
      request, FabricatedDevice(ReferenceMask() | nanochat::kFeatureBf16),
      nullptr, 0, nullptr);
  Check(accepted.ok, "the same model resolves once the device declares bf16");
}

// Case 3: a declared override bit selects the fused family and records the
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
      FabricatedDevice(ReferenceMask() | nanochat::kFeatureFusedResidualRmsNorm),
      registrations, 1, nullptr);
  Check(fused.ok, "a mask declaring the override bit builds a plan");
  const nanochat::CompositionChoice* choice =
      FindChoice(fused, nanochat::CompositionKind::kRmsNorm);
  Check(choice != nullptr, "the plan decides the rms_norm composition");
  if (choice != nullptr) {
    Check(choice->fused, "the declared override bit selects the fused family");
    Check(choice->implementation == "rms_norm",
          "the plan names the fused family that won: " + choice->implementation);
    Check(Mentions(choice->declined, "composition"),
          "the composition path is recorded as the declined fallback");
    bool names_the_bit = false;
    for (const nanochat::DeclinedImpl& entry : choice->declined) {
      if (entry.implementation == "composition" &&
          entry.reason.find("override_rms_norm") != std::string::npos) {
        names_the_bit = true;
      }
    }
    Check(names_the_bit,
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
    Check(!choice->fused, "a clear override bit leaves the composition running");
    Check(choice->implementation == "composition",
          "the composition path is what runs with no override bit");
    Check(Mentions(choice->declined, "rms_norm"),
          "the unselected fused family is recorded as a declined fallback");
  }

  // (c) A spec that forbids optimizations keeps the composition even when the
  // bit is declared.
  nanochat::ModelSpec policy_off = spec;
  policy_off.policy.allow_fused_overrides = false;
  nanochat::PlanRequest strict = request;
  strict.model = &policy_off;
  const nanochat::Plan forced = nanochat::BuildPlanWithDevice(
      strict,
      FabricatedDevice(ReferenceMask() | nanochat::kFeatureFusedResidualRmsNorm),
      registrations, 1, nullptr);
  choice = FindChoice(forced, nanochat::CompositionKind::kRmsNorm);
  Check(choice != nullptr && !choice->fused,
        "policy.allow_fused_overrides = false keeps the composition");
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
  CheckOverrideBitSelectsTheFusedFamily();
  CheckCompositionNames();

  if (g_failures != 0) {
    std::fprintf(stderr, "plan_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("plan_test: all checks passed\n");
  return 0;
}
