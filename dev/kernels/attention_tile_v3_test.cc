// T1 GPU smoke test for the attention-tile-v3 prototype. This wave ships the
// build surface only, so the test proves that the target compiles, links, and
// enters both entry points under the sandbox guard. The shape checks and the
// finite-difference checks land with the tile kernels in a later wave.
//
// Tiny shapes only, per docs/testing.md: this is a gate, not an iteration tool.

#include <cstdio>

#include "attention_tile_v3.h"
#include "nanochat/sandbox.h"

#if defined(NANOCHAT_PRECISION_FP16) && defined(NANOCHAT_PRECISION_FP32)
#error "exactly one precision must be selected"
#endif
#if !defined(NANOCHAT_PRECISION_FP16) && !defined(NANOCHAT_PRECISION_FP32)
#error "no precision selected; --config=fp32 / --config=fp16 sets it"
#endif

int main() {
  nanochat::RequireSandboxOrDie("test");

  nanochat::AttentionParams params;
  params.batch = 2;
  params.seq = 8;
  params.num_heads = 2;
  params.num_kv_heads = 2;
  params.head_dim = 4;

  const int forward_status = nanochat::dev::AttentionTileV3Forward(
      params, nullptr, nullptr, nullptr, nullptr, nullptr);
  if (forward_status != 0) {
    std::fprintf(stderr, "attention_tile_v3: forward stub failed\n");
    return 1;
  }

  const int backward_status = nanochat::dev::AttentionTileV3Backward(
      params, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);
  if (backward_status != 0) {
    std::fprintf(stderr, "attention_tile_v3: backward stub failed\n");
    return 1;
  }

  std::printf("attention_tile_v3: forward and backward stubs ok\n");
  return 0;
}
