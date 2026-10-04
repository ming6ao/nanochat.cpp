// Attention-tile-v3 prototype, first-wave stub. The file fixes the CUDA
// translation unit and the target wiring; the tile kernel implementations land
// in a later wave. Each entry point clears and reports the CUDA status so the
// GPU test can prove the device path is reachable.
//
// dev-only; the CUDA backend is unchanged. See attention_tile_v3.h.

#include "attention_tile_v3.h"

#include <cuda_runtime.h>

namespace nanochat {
namespace dev {

int AttentionTileV3Forward(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats) {
  // Kernel stub: no launch yet. The parameters stay unused on purpose.
  (void)params;
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)stats;
  return static_cast<int>(cudaGetLastError());
}

int AttentionTileV3Backward(const AttentionParams& params,
                            const ComputeType* q, const ComputeType* k,
                            const ComputeType* v, const float* stats,
                            const ComputeType* dout, ComputeType* dq,
                            ComputeType* dk, ComputeType* dv) {
  // Kernel stub: no launch yet. The parameters stay unused on purpose.
  (void)params;
  (void)q;
  (void)k;
  (void)v;
  (void)stats;
  (void)dout;
  (void)dq;
  (void)dk;
  (void)dv;
  return static_cast<int>(cudaGetLastError());
}

}  // namespace dev
}  // namespace nanochat
