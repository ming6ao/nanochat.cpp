#ifndef NANOCHAT_SRC_PLAN_INTERNAL_H_
#define NANOCHAT_SRC_PLAN_INTERNAL_H_

#include <cstdint>
#include <string>

#include "nanochat/plan.h"
#include "nanochat/tensor.h"

// The pure seam behind `BuildPlan` (`include/nanochat/plan.h`).
//
// The public `BuildPlan` reads the device from `kernels::GetCaps()` and the
// registrations from `RegisteredFusedOverrides`, which is what a launch does.
// This overload takes both as arguments, so a caller can build a plan for a
// *fabricated* capability mask and a *fabricated* registration table: a device
// that does not exist yet, a mask with one bit cleared, a registration whose
// capability bit the device does not declare. `src/plan_test.cc` is that
// caller, and it is the whole reason the seam exists: the gate of
// docs/model-and-device-bringup.md section 16 P3 is "a plan built under T0 CPU
// with no device present".
//
// The seam is as pure as the public call: it reads its arguments, allocates no
// device memory, launches no kernel, and mutates nothing. It is declared in
// `src/` and not in the public header because the fabrication is a test
// capability, not a runtime one.
//
// This header is private to `src/`.

namespace nanochat {

// The capability mask a plan treats as the device's answer.
//
// `Caps::feature_mask` is the authority when it is non-empty. When a backend
// reports no mask at all - the reference backend today, whose declaration lives
// in `backends/cpu/caps.json` and is not yet wired into `GetCaps()` - the plan
// reads that declaration through the device-profile derivation instead, because
// silence is not a statement that a device offers nothing, and treating it as
// one would reject every plan on the reference backend. A mask that reports the
// host with zero bits is exactly the case this resolves.
std::uint64_t EffectiveDeviceMask(const Caps& device);

// `BuildPlan`, with the device's answer and the registration list supplied by
// the caller. `registrations` may be null when `count` is 0.
Plan BuildPlanWithDevice(const PlanRequest& request, const Caps& device,
                         const FusedOverride* registrations, int count,
                         std::string* why_not);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_PLAN_INTERNAL_H_
