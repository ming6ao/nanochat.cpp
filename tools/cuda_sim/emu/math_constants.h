#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_MATH_CONSTANTS_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_MATH_CONSTANTS_H_

// The emulator's `<math_constants.h>` (docs/simulator.md section 9.2).
//
// CUDA exposes these as device constants; on the host they are compile-time
// constants with the same value, so a kernel body that writes
// `-CUDART_INF_F` keeps its meaning.

#define CUDART_INF_F (__builtin_inff())
#define CUDART_INF_D (__builtin_inf())
#define CUDART_NAN_F (__builtin_nanf(""))
#define CUDART_NAN_D (__builtin_nan(""))
#define CUDART_PI_F (3.141592654f)

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_MATH_CONSTANTS_H_
