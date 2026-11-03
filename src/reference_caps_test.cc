// The reference backend's capability report (`docs/api-surface.md` R4b).
//
// Two facts, and they are coupled. `kernels::GetCaps()` answers with the nine
// bits `backends/cpu/caps.json` declares for the `cpu` entry, read through
// `nanochat/device_profile.h`, and none of those bits is an override bit. The
// second fact is what keeps the portable composition on the CPU leg: the
// `Fused` gate in `src/ops.cc` reads the bit before it takes a backend family,
// and the planner declines the fused family for the same reason.

#include <cstdint>
#include <cstdio>
#include <string>

#include "nanochat/kernels.h"
#include "nanochat/model_spec.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"
#include "src/capability_names.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "reference caps: FAILED - %s\n", what.c_str());
    ++g_failures;
  }
}

int BitCount(std::uint64_t mask) {
  int count = 0;
  for (; mask != 0; mask &= mask - 1) ++count;
  return count;
}

// The reference feature set the vocabulary implies: every named bit except the
// four overrides and the fp8 pair. `spec/capability_bits.md` section 3 makes an
// override bit device-only, and `backends/cpu/caps.json` declares no fp8 bit,
// because its fp8 emulation is the oracle a GPU's fp8 bits are measured against
// rather than a mechanism this device offers.
std::uint64_t ExpectedReferenceMask() {
  std::uint64_t overrides = 0;
  for (int i = 0; i < nanochat::kCapabilityNameCount; ++i) {
    const std::uint64_t bit = nanochat::kCapabilityNames[i].bit;
    if (nanochat::IsOverrideCapability(bit)) overrides |= bit;
  }
  return nanochat::AllNamedCapabilityBits() & ~overrides &
         ~(nanochat::kFeatureFp8Storage | nanochat::kFeatureFp8Gemm);
}

// The declaration is the answer, and the three precision flags are the same
// answer spelled twice.
void TestReportedMask() {
  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  Check(!caps.is_device,
        "the reference backend reports the host, not a device");
  Check(caps.feature_mask == ExpectedReferenceMask(),
        "the reference backend reports every declared bit (the vocabulary "
        "minus the four overrides and the fp8 pair)");
  Check(BitCount(caps.feature_mask) == 9,
        "the reference declaration carries nine bits");
  Check(caps.supports_fp32 == caps.HasFeature(nanochat::kFeatureFp32),
        "supports_fp32 agrees with the fp32 bit");
  Check(caps.supports_bf16 == caps.HasFeature(nanochat::kFeatureBf16),
        "supports_bf16 agrees with the bf16 bit");
  Check(!caps.supports_fp8, "the reference backend claims no fp8 path");
}

// No override bit is set, so `ops::` takes the composition, and the planner
// makes the same call for the same reason.
void TestCompositionStaysOnTheReferenceBackend() {
  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  Check(!caps.HasFeature(nanochat::kFeatureOverrideRmsNorm),
        "no override_rms_norm bit, so ops::RmsNormForward runs the "
        "composition");
  Check(!caps.HasFeature(nanochat::kFeatureOverrideQkPrep),
        "no override_qk_prep bit, so ops::QkPrepForward runs the composition");
  Check(!caps.HasFeature(nanochat::kFeatureOverrideClassifier),
        "no override_classifier bit, so ops::ClassifierForward runs the "
        "composition");
  Check(!caps.HasFeature(nanochat::kFeatureFusedAttention),
        "no fused-attention bit, so ops::AttentionForward runs the primitive");

  const nanochat::ModelSpec spec = nanochat::DefaultModelSpec();
  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  std::string why_not;
  const nanochat::Plan plan = nanochat::BuildPlan(request, &why_not);
  Check(plan.ok,
        "the reference backend builds a plan for the default spec: " + why_not);
  const nanochat::CompositionChoice* rms_norm = nullptr;
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    if (choice.composition == nanochat::CompositionKind::kRmsNorm) {
      rms_norm = &choice;
    }
  }
  Check(rms_norm != nullptr, "the plan carries the rms_norm composition");
  if (rms_norm != nullptr) {
    Check(!rms_norm->fused,
          "rms_norm resolves to the composition, not to a fused family");
    Check(rms_norm->implementation == "composition",
          "the plan names the composition as the implementation: " +
              rms_norm->implementation);
  }
}

}  // namespace

int main() {
  TestReportedMask();
  TestCompositionStaysOnTheReferenceBackend();
  if (g_failures != 0) {
    std::fprintf(stderr, "reference caps: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("reference caps: checks passed\n");
  return 0;
}
