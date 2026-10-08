#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_RUNTIME_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_RUNTIME_H_

// The emulator's `<cuda_runtime.h>` (docs/simulator.md section 9.2).
//
// An emulated translation unit compiles a real `backends/cuda/**` source with
// the host compiler; this header shadows the CUDA runtime header so the device
// vocabulary resolves to the emulation instead of to the toolkit. Everything
// lives in `cuda_emu.h`, which doubles as the documented mapping table.

#include "tools/cuda_sim/emu/cuda_emu.h"

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_RUNTIME_H_
