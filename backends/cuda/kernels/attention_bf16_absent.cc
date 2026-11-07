// The stand-in for the bf16 tensor-core attention tile
// (docs/attention.md Part 6, docs/simulator.md section 9.2).
//
// An fp32 build has no bf16 operands to stage, and the emulator has no
// tensor-core primitive because it compiles every kernel with the host
// compiler. Both link this file instead of `attention_bf16.cu`, so
// `attention.cu` asks the same two questions in every build and the dispatch
// stays out of the preprocessor.
//
// `local_defines = precision_defines()` keeps `ComputeType` the same on both
// sides of the call: this file and its caller compile at the workspace
// precision.

#include "backends/cuda/kernels/attention_bf16.h"

namespace nanochat {
namespace kernels {

bool UseFusedAttentionBf16(const AttentionParams&) { return false; }

void LaunchBf16Attention(const AttentionParams&, const ComputeType*,
                         const ComputeType*, const ComputeType*, ComputeType*,
                         float*) {}

}  // namespace kernels
}  // namespace nanochat
