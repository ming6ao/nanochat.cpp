// Standalone benchmark for the attention-tile-v3 prototype. The fused fp32
// tiled forward has landed; this wave enters the forward entry point on real
// device buffers, and the device-event timing loop and the tile-size sweep
// arrive with the Phase 1 sweep wave.
//
// Run it through the project entry point (which takes the GPU broker):
//
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/attention_tile_v3_bench
//
// See docs/performance.md.

#include <cstddef>
#include <cstdio>

#include "attention_tile_v3.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

int main() {
  nanochat::RequireSandboxOrDie("bench");

  nanochat::AttentionParams params;
  params.batch = 2;
  params.seq = 64;
  params.num_heads = 2;
  params.num_kv_heads = 2;
  params.head_dim = 8;
  params.causal = true;
  params.window_left = -1;
  params.window_right = 0;

  const int dim = params.head_dim;
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int q_count = params.batch * params.seq * params.num_heads * dim;
  const int kv_count = params.batch * kv_len * params.num_kv_heads * dim;
  const int stats_count = nanochat::AttentionStatsCount(params);

  nanochat::ComputeType* q = static_cast<nanochat::ComputeType*>(
      nanochat::kernels::Alloc(sizeof(nanochat::ComputeType) * q_count));
  nanochat::ComputeType* k = static_cast<nanochat::ComputeType*>(
      nanochat::kernels::Alloc(sizeof(nanochat::ComputeType) * kv_count));
  nanochat::ComputeType* v = static_cast<nanochat::ComputeType*>(
      nanochat::kernels::Alloc(sizeof(nanochat::ComputeType) * kv_count));
  nanochat::ComputeType* out = static_cast<nanochat::ComputeType*>(
      nanochat::kernels::Alloc(sizeof(nanochat::ComputeType) * q_count));
  float* stats = static_cast<float*>(
      nanochat::kernels::Alloc(sizeof(float) * stats_count));

  // A zeroed input keeps the bench deterministic and free of triangular
  // denominators, so it measures the tile's launch and loop cost only.
  nanochat::kernels::Memset(q, 0, sizeof(nanochat::ComputeType) * q_count);
  nanochat::kernels::Memset(k, 0, sizeof(nanochat::ComputeType) * kv_count);
  nanochat::kernels::Memset(v, 0, sizeof(nanochat::ComputeType) * kv_count);

  const int status = nanochat::dev::AttentionTileV3Forward(
      params, q, k, v, out, stats);
  nanochat::kernels::Synchronize();

  nanochat::kernels::Free(q);
  nanochat::kernels::Free(k);
  nanochat::kernels::Free(v);
  nanochat::kernels::Free(out);
  nanochat::kernels::Free(stats);

  if (status != 0) {
    std::fprintf(stderr, "attention_tile_v3_bench: forward failed\n");
    return 1;
  }
  std::printf(
      "attention_tile_v3_bench: forward entry point ok; the device-event "
      "timing loop arrives with the Phase 1 sweep wave\n");
  return 0;
}
