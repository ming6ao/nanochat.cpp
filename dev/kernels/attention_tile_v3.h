#ifndef NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_
#define NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_

// Prototype for the Pascal-native tiled (flash) attention experiment. The
// forward pass is implemented in attention_tile_v3.cu; the backward pass is
// still a stub that a later wave fills in.
//
// This is dev-only scaffolding: nothing here is part of the frozen seam, and
// the CUDA backend is unchanged. See DESIGN.md section 3,
// docs/flash-attention-pascal.md section 5, and dev/kernels/README.md.

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// Forward prototype. `q` holds `[batch, seq, num_heads, head_dim]`; `k` and `v`
// hold `[batch, kv_len, num_kv_heads, head_dim]` with `kv_len = params.kv_len`
// when that is positive and `params.seq` otherwise. On return `out` holds the
// attention output in the query layout and `stats` holds the per-query softmax
// statistics `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// One thread block owns 32 query rows for one `(batch, head)` pair and walks
// the key and value sequence in tiles of 32 rows. The running row maximum, the
// running denominator, and the output accumulator live in registers; fully
// masked key tiles are skipped. Returns 0 on success and a nonzero CUDA status
// when the kernel cannot be launched.
int AttentionTileV3Forward(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats);

// Backward prototype. `stats` holds the saved softmax statistics from the
// forward pass. On return `dq`/`dk`/`dv` hold the input gradients. Returns 0 on
// success. Not implemented yet.
int AttentionTileV3Backward(const AttentionParams& params,
                            const ComputeType* q, const ComputeType* k,
                            const ComputeType* v, const float* stats,
                            const ComputeType* dout, ComputeType* dq,
                            ComputeType* dk, ComputeType* dv);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_
