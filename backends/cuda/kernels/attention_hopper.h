#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_HOPPER_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_HOPPER_H_

// What `attention.cu` asks of the Hopper forward tile of `docs/attention.md`
// Part 8: whether this device takes it, and the one launch. The tile is the
// device source `attention_hopper.cu`, which states `archs = ["sm_90a"]` on its
// own target, so the default build compiles no code of this header's subject.
// This header names no vendor type, so a host pass reads it.
//
// Every other build links `attention_hopper_absent.cc`, which answers the same
// two questions with false and nothing. Both sources define the pair, so
// `attention.cu` holds no guard and cannot tell which one it got
// (docs/simulator.md section 9.2).

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// Whether this build and this device take the tile for this shape. The shape
// conditions are in `attention_dispatch.h`; this adds the device answers: the
// card runs the warpgroup instruction, and its shared window accepts one block
// of the tile.
bool UseHopperAttention(const AttentionParams& p);

// Launches the fused forward tile. The caller has already asked
// `UseHopperAttention`; the stand-in returns without launching anything.
void LaunchHopperAttention(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_HOPPER_H_
