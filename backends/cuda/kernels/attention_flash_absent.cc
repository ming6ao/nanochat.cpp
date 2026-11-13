// The stand-in for the flash attention family (docs/attention.md Part 7,
// docs/simulator.md section 9.2).
//
// An fp32 build has no bf16 `mma` operand to stage, and the emulator has no
// tensor-core primitive because it compiles every kernel with the host
// compiler. Both link this file instead of `attention_flash.cu`, so
// `attention.cu` asks the same three questions in every build and the dispatch
// stays out of the preprocessor.
//
// `local_defines = precision_defines()` keeps `ComputeType` the same on both
// sides of the call: this file and its caller compile at the workspace
// precision.

#include "backends/cuda/kernels/attention_flash.h"

namespace nanochat {
namespace kernels {

bool UseFusedAttentionFlash(const AttentionParams&) { return false; }

void LaunchFlashAttention(const AttentionParams&, const ComputeType*,
                          const ComputeType*, const ComputeType*, ComputeType*,
                          float*) {}

void LaunchFlashAttentionBackward(const AttentionParams&, const ComputeType*,
                                  const ComputeType*, const ComputeType*,
                                  const float*, const ComputeType*,
                                  ComputeType*, ComputeType*, ComputeType*) {}

}  // namespace kernels
}  // namespace nanochat
