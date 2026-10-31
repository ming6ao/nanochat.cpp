#ifndef NANOCHAT_SRC_DEVICE_CAPS_H_
#define NANOCHAT_SRC_DEVICE_CAPS_H_

#include <cstdint>
#include <string>

// The capability mask a device's *declaration* carries:
// `backends/<name>/caps.json`, `devices[].features`, mapped through the
// vocabulary table in `src/capability_names.h`.
//
// `src/device_profile.cc` derives two views from the same parsed files: the
// capability *fields* a profile exposes (`nanochat/device_profile.h`) and the
// mechanism mask this function returns. The mask is not a field of
// `DeviceProfile`, because that header is frozen for this campaign and because a
// profile describes a device's shape while the mask is the device's promise.
//
// This is a private `src/` header.

namespace nanochat {

// The bits `backends/*/caps.json` declare for the device whose normalized name
// is `device_name`. Returns 0 when no declaration names it, which the planner
// reads as "this backend has not declared anything for that device".
std::uint64_t DeclaredFeatureMaskForDevice(const std::string& device_name);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_DEVICE_CAPS_H_
