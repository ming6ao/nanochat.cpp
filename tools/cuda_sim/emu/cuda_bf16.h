#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_BF16_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_BF16_H_

// The emulator's `<cuda_bf16.h>` (docs/simulator.md section 9.2).
//
// bf16 maps to `nanochat::Bf16`, so one rounding function serves the device
// code, the emulator, and the CPU reference. Only the scalar type, the
// two-lane pair, and the conversions a tensor-core tile uses are supplied.

#include <cstdint>

#include "nanochat/tensor.h"

using __nv_bfloat16 = ::nanochat::Bf16;

// The two-lane pair: the low lane in `x`, the high lane in `y`.
struct __nv_bfloat162 {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
};

inline __nv_bfloat16 __float2bfloat16(float value) {
  return ::nanochat::Bf16FromFloat(value);
}

inline __nv_bfloat16 __float2bfloat16_rn(float value) {
  return ::nanochat::Bf16FromFloat(value);
}

inline float __bfloat162float(__nv_bfloat16 value) {
  return ::nanochat::Bf16BitsToFloat(value.bits);
}

inline __nv_bfloat162 __floats2bfloat162_rn(float low, float high) {
  __nv_bfloat162 pair;
  pair.x = ::nanochat::Bf16FromFloat(low);
  pair.y = ::nanochat::Bf16FromFloat(high);
  return pair;
}

inline float __low2float(__nv_bfloat162 pair) {
  return ::nanochat::Bf16BitsToFloat(pair.x.bits);
}

inline float __high2float(__nv_bfloat162 pair) {
  return ::nanochat::Bf16BitsToFloat(pair.y.bits);
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_BF16_H_
