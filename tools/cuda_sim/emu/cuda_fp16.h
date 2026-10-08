#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP16_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP16_H_

// The emulator's `<cuda_fp16.h>` (docs/simulator.md section 9.2).
//
// The project's `Fp16` (include/nanochat/tensor.h) is bit-compatible with the
// device `__half`, so the conversions go through the project's float <-> half
// helpers rather than a second implementation. Only the surface the kernel
// families use is provided: the bit casts, the round-to-nearest conversions,
// and the scalar half atomic the fp16 storage path calls on sm_70+.

#include <cstdint>

#include "nanochat/tensor.h"
#include "tools/cuda_sim/emu/cuda_emu.h"

struct __half {
  std::uint16_t bits;
};

struct __half2 {
  __half x;
  __half y;
};

namespace nanochat {
namespace emu {

inline float HalfToFloat(const __half& value) {
  return HalfBitsToFloat(value.bits);
}

inline __half FloatToHalf(float value) {
  __half result;
  result.bits = FloatToHalfBits(value);
  return result;
}

}  // namespace emu
}  // namespace nanochat

inline float __half2float(__half value) {
  return nanochat::emu::HalfToFloat(value);
}
inline __half __float2half_rn(float value) {
  return nanochat::emu::FloatToHalf(value);
}
inline __half __ushort_as_half(unsigned short bits) {
  __half result;
  result.bits = bits;
  return result;
}
inline unsigned short __half_as_ushort(__half value) { return value.bits; }

// The scalar half atomic the fp16 storage path uses on sm_70 and newer. It is
// the same read-modify-write as the float form, on half storage.
inline __half atomicAdd(__half* address, __half value) {
  std::lock_guard<std::mutex> lock(::nanochat::emu::AtomicMutex());
  const __half previous = *address;
  *address = nanochat::emu::FloatToHalf(nanochat::emu::HalfToFloat(previous) +
                                        nanochat::emu::HalfToFloat(value));
  return previous;
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP16_H_
