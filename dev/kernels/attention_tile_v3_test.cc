// T1 GPU correctness test for the attention-tile-v3 fused forward prototype.
// It runs the device tiled forward against the host reference in
// sequence_ref.h over the causal, sliding-window, grouped-query, KV-cache
// offset, non-causal, and empty-window contracts.
//
// Shapes: the plan's tiny gate (B=2, T=8, head_dim=4) plus a few shallow
// multi-tile shapes (T=70) that cross the candidate Br=32 / Bc=32 tile
// boundaries so the tile loop and the masked-tile skip are actually exercised.
// These still run in microseconds. The backward entry point is still a stub,
// so this test only checks that it links and returns cleanly.
//
// Tiny shapes only, per docs/testing.md: this is a gate, not an iteration tool.

#include <cstdio>
#include <string>
#include <vector>

#include "attention_tile_v3.h"
#include "gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "sequence_ref.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ComputeType;
using nanochat::dev::seqref::AttentionForward;
using nanochat::dev::Check;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// The reference computes the dot product and the exponentials in double; the
// device tile accumulates in float. fp16 storage widens the gap, so the
// tolerance follows the selected precision.
#if defined(NANOCHAT_PRECISION_FP16)
constexpr double kOutTol = 5e-3;
constexpr double kStatsTol = 1e-2;
#else
constexpr double kOutTol = 1e-4;
constexpr double kStatsTol = 1e-4;
#endif

// Runs one tiny forward case on the device and compares the output and the
// saved softmax statistics against the host reference.
void ExpectForward(const AttentionParams& p, const char* name) {
  const int dim = p.head_dim;
  const int heads = p.num_heads;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : heads;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int q_count = p.batch * p.seq * heads * dim;
  const int kv_count = p.batch * kv_len * kv_heads * dim;
  const int stats_count = nanochat::AttentionStatsCount(p);

  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(kv_count, &rng);
  const std::vector<float> v0 = RandomVec(kv_count, &rng);

  DevBuf<ComputeType> q(ToStorage(q0));
  DevBuf<ComputeType> k(ToStorage(k0));
  DevBuf<ComputeType> v(ToStorage(v0));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(stats_count);

  const int status = nanochat::dev::AttentionTileV3Forward(
      p, q.ptr, k.ptr, v.ptr, out.ptr, stats.ptr);
  Check(status == 0, (std::string(name) + " launch").c_str());
  nanochat::kernels::Synchronize();

  std::vector<float> ref_stats;
  const std::vector<float> ref_out =
      AttentionForward(p, q0, k0, v0, &ref_stats);
  CheckVectorClose(FromStorage(out.Download()), ref_out, kOutTol,
                   (std::string(name) + " forward out").c_str());
  CheckVectorClose(stats.Download(), ref_stats, kStatsTol,
                   (std::string(name) + " forward stats").c_str());
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("attention_tile_v3 gpu test\n");

  // Baseline tiny shape from the plan: B=2, T=8, head_dim=4.
  AttentionParams tiny;
  tiny.batch = 2;
  tiny.seq = 8;
  tiny.num_heads = 2;
  tiny.num_kv_heads = 2;
  tiny.head_dim = 4;
  tiny.causal = true;
  tiny.window_left = -1;
  tiny.window_right = 0;
  tiny.kv_len = 0;
  tiny.scale = 0.0f;
  ExpectForward(tiny, "tiny-mha-causal");

  AttentionParams tiny_gqa = tiny;
  tiny_gqa.num_kv_heads = 1;
  ExpectForward(tiny_gqa, "tiny-gqa-causal");

  // Multi-tile shapes: T=70 crosses three Br=32 query tiles and three Bc=32
  // key tiles, so the tile loop, the running online-softmax rescale, and the
  // causal tile skip all run.
  AttentionParams tile;
  tile.batch = 1;
  tile.seq = 70;
  tile.num_heads = 2;
  tile.num_kv_heads = 2;
  tile.head_dim = 8;
  tile.causal = true;
  tile.window_left = -1;
  tile.window_right = 0;
  ExpectForward(tile, "tile-mha-causal");

  // Grouped-query attention plus a left sliding window, which forces both the
  // head map and the left-window tile skip.
  AttentionParams tile_gqa_window = tile;
  tile_gqa_window.num_heads = 4;
  tile_gqa_window.num_kv_heads = 2;
  tile_gqa_window.window_left = 8;
  ExpectForward(tile_gqa_window, "tile-gqa-window");

  // Full non-causal attention with grouped-query heads.
  AttentionParams noncausal;
  noncausal.batch = 1;
  noncausal.seq = 70;
  noncausal.num_heads = 2;
  noncausal.num_kv_heads = 1;
  noncausal.head_dim = 8;
  noncausal.causal = false;
  noncausal.window_left = -1;
  noncausal.window_right = -1;
  ExpectForward(noncausal, "tile-gqa-noncausal");

  // Non-causal with both window bounds finite: exercises the left and the
  // right masked-tile skip in the same kernel run.
  AttentionParams window_both;
  window_both.batch = 1;
  window_both.seq = 70;
  window_both.num_heads = 2;
  window_both.num_kv_heads = 2;
  window_both.head_dim = 8;
  window_both.causal = false;
  window_both.window_left = 6;
  window_both.window_right = 6;
  ExpectForward(window_both, "tile-window-both");

  // KV-cache offset: kv_len differs from seq, so the absolute query position
  // starts at kv_len - seq and the key tiles before the window must still be
  // walked.
  AttentionParams kv_offset;
  kv_offset.batch = 2;
  kv_offset.seq = 8;
  kv_offset.num_heads = 2;
  kv_offset.num_kv_heads = 2;
  kv_offset.head_dim = 8;
  kv_offset.causal = true;
  kv_offset.window_left = -1;
  kv_offset.window_right = 0;
  kv_offset.kv_len = 70;
  ExpectForward(kv_offset, "kv-cache-offset");

  // Empty-window contract: with window_left == 0 and causal set, the allowed
  // key is exactly j == qpos. Rows with a negative absolute query position
  // have no visible key, so both the reference and the kernel must write a
  // zero output row and the (0, 0) statistic pair.
  AttentionParams empty;
  empty.batch = 2;
  empty.seq = 8;
  empty.num_heads = 2;
  empty.num_kv_heads = 2;
  empty.head_dim = 4;
  empty.causal = true;
  empty.window_left = 0;
  empty.window_right = -1;
  empty.kv_len = 4;
  ExpectForward(empty, "empty-window");

  // The backward entry point is still a Phase 2 stub; confirm it links and
  // returns its guard status without an error.
  const int backward_status = nanochat::dev::AttentionTileV3Backward(
      tiny, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);
  Check(backward_status == 0, "backward entry point");

  if (Failures() != 0) {
    std::printf("attention_tile_v3: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("attention_tile_v3: all checks passed\n");
  return 0;
}
