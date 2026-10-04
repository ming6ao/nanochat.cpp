// Standalone benchmark for the attention-tile-v3 prototype. This wave ships the
// build surface only, so the benchmark reports that the tile kernels have not
// landed yet; the device-event timing loop arrives with the kernels.
//
// Run it through the project entry point (which takes the GPU broker):
//
//   tools/nanochat bench -- ./bazel-bin/dev/kernels/attention_tile_v3_bench
//
// See docs/performance.md.

#include <cstdio>

#include "attention_tile_v3.h"
#include "nanochat/sandbox.h"

int main() {
  nanochat::RequireSandboxOrDie("bench");

  nanochat::AttentionParams params;
  params.batch = 2;
  params.seq = 8;
  params.num_heads = 2;
  params.num_kv_heads = 2;
  params.head_dim = 4;

  const int status = nanochat::dev::AttentionTileV3Forward(
      params, nullptr, nullptr, nullptr, nullptr, nullptr);
  if (status != 0) {
    std::fprintf(stderr, "attention_tile_v3_bench: forward stub failed\n");
    return 1;
  }

  std::printf(
      "attention_tile_v3_bench: prototype stub; the tile kernels have not "
      "landed yet\n");
  return 0;
}
