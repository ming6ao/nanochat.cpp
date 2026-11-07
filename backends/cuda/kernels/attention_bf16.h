#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_BF16_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_BF16_H_

// What `attention.cu` asks of the bf16 tensor-core tile of the windowed
// forward: whether this device takes it, and the launch itself. The tile is the
// device source `attention_bf16.cu`, which only a bf16 or fp8 build compiles;
// this header names no vendor type, so a host pass reads it
// (docs/attention.md Part 6).
//
// The other builds link `attention_bf16_absent.cc`, which answers the same two
// questions with false and nothing. Both sources define the pair, so
// `attention.cu` holds no guard and cannot tell which one it got
// (docs/simulator.md section 9.2).

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// Whether this build and this device take the tile for this shape. The shape
// conditions are in `attention_dispatch.h`; this adds the device answer.
bool UseFusedAttentionBf16(const AttentionParams& p);

// Launches the tile. The caller has already asked `UseFusedAttentionBf16`; the
// stand-in returns without launching anything.
void LaunchBf16Attention(const AttentionParams& params, const ComputeType* q,
                         const ComputeType* k, const ComputeType* v,
                         ComputeType* out, float* stats);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_BF16_H_
