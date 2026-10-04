#ifndef NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_
#define NANOCHAT_DEV_KERNELS_ATTENTION_TILE_V3_H_

// Prototype for the Pascal-native tiled (flash) attention experiment. The
// forward pass is implemented in attention_tile_v3.cu; the backward pass is
// still a stub that a later wave fills in.
//
// The forward tile is compiled for a fixed set of launch configurations so the
// Phase 1 tile-size sweep can select one at run time. See
// AttentionTileV3Config, AttentionTileV3Supported, and the configured
// AttentionTileV3Forward overload below.
//
// This is dev-only scaffolding: nothing here is part of the frozen seam, and
// the CUDA backend is unchanged. See DESIGN.md section 3,
// docs/flash-attention-pascal.md section 5, and dev/kernels/README.md.

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// Compile-time tile shape and launch configuration for the forward prototype.
// The kernel is instantiated for a small set of these so the tile sweep can
// choose one without a rebuild. `br` is the query rows per block, `bc` the
// key/value rows staged per shared-memory tile (a multiple of 32, the warp
// width), `threads` the threads per block, and `min_blocks` the
// `__launch_bounds__` minimum resident blocks per streaming multiprocessor.
struct AttentionTileV3Config {
  int br = 32;
  int bc = 32;
  int threads = 256;
  int min_blocks = 2;
};

// Returns true when the prototype has a compiled kernel for `config`. A
// supported configuration can still be rejected at launch time when its shared
// memory does not fit at the requested `head_dim`; check the launch status.
bool AttentionTileV3Supported(const AttentionTileV3Config& config);

// Forward prototype with an explicit tile configuration. `q` holds
// `[batch, seq, num_heads, head_dim]`; `k` and `v` hold
// `[batch, kv_len, num_kv_heads, head_dim]` with `kv_len = params.kv_len` when
// that is positive and `params.seq` otherwise. On return `out` holds the
// attention output in the query layout and `stats` holds the per-query softmax
// statistics `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// One thread block owns `config.br` query rows for one `(batch, head)` pair and
// walks the key and value sequence in tiles of `config.bc` rows. The running
// row maximum, the running denominator, and the output accumulator live in
// registers; fully masked key tiles are skipped. Returns 0 on success and a
// nonzero CUDA status when the configuration is unknown or cannot be launched.
int AttentionTileV3Forward(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats,
                           const AttentionTileV3Config& config);

// Default forward prototype (Br=32, Bc=32, 256 threads, 2 blocks per SM). Same
// contract as the configured overload above.
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
