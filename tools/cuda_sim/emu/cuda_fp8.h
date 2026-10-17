#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP8_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP8_H_

// The emulator's `<cuda_fp8.h>` (docs/simulator-precision.md section 6.1).
//
// The CUDA conversion intrinsics map to the shared rounding header
// `nanochat/fp8.h`, so the emulated device code and the CPU reference use one
// implementation (docs/simulator-precision.md invariant 2, docs/correctness-
// gates.md section 3.2). A second implementation would break the exact
// simulator leg.

#include <cstdint>

#include "nanochat/fp8.h"

using __nv_fp8_storage_t = std::uint8_t;

enum __nv_saturation_t { __NV_NOSAT = 0, __NV_SATFINITE = 1 };
enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1 };

struct __nv_fp8_e4m3 {
  __nv_fp8_storage_t __x = 0;
};

struct __nv_fp8_e5m2 {
  __nv_fp8_storage_t __x = 0;
};

// The host mapping of `__nv_cvt_float_to_fp8`. The device intrinsic saturates
// and rounds to nearest, ties to even; the shared header does the same, so the
// saturation argument does not change the result.
inline __nv_fp8_storage_t __nv_cvt_float_to_fp8(
    float value, __nv_saturation_t /*saturation*/,
    __nv_fp8_interpretation_t interpretation) {
  return interpretation == __NV_E4M3 ? nanochat::Fp8E4M3Encode(value)
                                     : nanochat::Fp8E5M2Encode(value);
}

inline float __nv_cvt_fp8_to_float(__nv_fp8_storage_t bits,
                                   __nv_fp8_interpretation_t interpretation) {
  return interpretation == __NV_E4M3 ? nanochat::Fp8E4M3Decode(bits)
                                     : nanochat::Fp8E5M2Decode(bits);
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_FP8_H_
