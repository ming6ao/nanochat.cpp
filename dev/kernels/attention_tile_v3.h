#ifndef NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_
#define NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_

// Prototype build surface for the attention-tile-v3 experiment. This header
// fixes the entry points that the later kernel waves fill in, so the target,
// the GPU test, and the benchmark can compile and link from the first wave.
//
// This is dev-only scaffolding: nothing here is part of the frozen seam, and
// the CUDA backend is unchanged. See DESIGN.md section 3 and
// dev/kernels/README.md.

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// Forward prototype. On entry `q`/`k`/`v` hold `[batch, kv_len, heads, dim]`
// activations; on return `out` holds the attention output and `stats` holds the
// per-query softmax statistics. Returns 0 on success.
int AttentionTileV3Forward(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats);

// Backward prototype. `stats` holds the saved softmax statistics from the
// forward pass. On return `dq`/`dk`/`dv` hold the input gradients. Returns 0 on
// success.
int AttentionTileV3Backward(const AttentionParams& params,
                            const ComputeType* q, const ComputeType* k,
                            const ComputeType* v, const float* stats,
                            const ComputeType* dout, ComputeType* dq,
                            ComputeType* dk, ComputeType* dv);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_
