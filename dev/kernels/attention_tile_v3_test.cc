// T1 GPU correctness test for the attention-tile-v3 fused forward prototype.
//
// Two comparisons prove the forward tile:
//   1. The tiny gate from the plan (B=2, T=8, head_dim=4, heads=2) plus a few
//      shallow multi-tile shapes (T=70) against the host reference in
//      sequence_ref.h. These exercise the tile loop, the masked-tile skip, the
//      online-softmax rescale, grouped-query heads, the KV-cache offset, and
//      the empty-window contract.
//   2. The production training shape (batch=8, seq=512, heads=4, kv-heads=4,
//      head_dim=128) against the shipped `kernels::AttentionForward` for
//      causal, sliding-window 128, a left-only window, a right-only window,
//      grouped-query, and the KV-cache offset.
//
// The backward entry point is still a stub, so this test only checks that it
// links and returns cleanly. Tiny shapes only for the host-reference gate, per
// docs/testing.md: this is a gate, not an iteration tool.

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
constexpr double kShippedTol = 2e-2;
#else
constexpr double kOutTol = 1e-4;
constexpr double kStatsTol = 1e-4;
constexpr double kShippedTol = 2e-3;
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

// Runs one production-shape forward on the device tile and on the shipped
// `kernels::AttentionForward`, then compares the output and the softmax
// statistics. This is a device-to-device check, so it isolates the tile from
// the host reference's double-precision arithmetic.
void ExpectShippedMatch(const AttentionParams& p, const char* name) {
  const int dim = p.head_dim;
  const int heads = p.num_heads;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : heads;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int q_count = p.batch * p.seq * heads * dim;
  const int kv_count = p.batch * kv_len * kv_heads * dim;
  const int stats_count = nanochat::AttentionStatsCount(p);

  Rng rng;
  DevBuf<ComputeType> q(ToStorage(RandomVec(q_count, &rng)));
  DevBuf<ComputeType> k(ToStorage(RandomVec(kv_count, &rng)));
  DevBuf<ComputeType> v(ToStorage(RandomVec(kv_count, &rng)));
  DevBuf<ComputeType> out_tile(q_count);
  DevBuf<ComputeType> out_shipped(q_count);
  DevBuf<float> stats_tile(stats_count);
  DevBuf<float> stats_shipped(stats_count);

  const int status = nanochat::dev::AttentionTileV3Forward(
      p, q.ptr, k.ptr, v.ptr, out_tile.ptr, stats_tile.ptr);
  Check(status == 0, (std::string(name) + " tile launch").c_str());
  nanochat::kernels::AttentionForward(p, q.ptr, k.ptr, v.ptr, out_shipped.ptr,
                                      stats_shipped.ptr);
  nanochat::kernels::Synchronize();

  CheckVectorClose(FromStorage(out_tile.Download()),
                   FromStorage(out_shipped.Download()), kShippedTol,
                   (std::string(name) + " shipped out").c_str());
  CheckVectorClose(stats_tile.Download(), stats_shipped.Download(), kShippedTol,
                   (std::string(name) + " shipped stats").c_str());
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("attention_tile_v3 gpu test\n");

  // Baseline tiny shape from the plan: B=2, T=8, head_dim=4, heads=2.
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

  // Non-default tile configurations must reproduce the reference too. These
  // cover the sweep axes (Br, Bc, block size, launch bounds) at the tiny gate,
  // so a tile that the benchmark selects is also proven correct.
  const nanochat::dev::AttentionTileV3Config configs[] = {
      {16, 32, 256, 2},  // Br sweep
      {64, 32, 256, 2},  // Br sweep
      {32, 64, 256, 2},  // Bc sweep
      {32, 32, 128, 2},  // block-size sweep
      {32, 32, 512, 2},  // block-size sweep
      {32, 32, 256, 1},  // launch-bounds sweep
      {32, 32, 256, 3},  // launch-bounds sweep
  };
  for (const nanochat::dev::AttentionTileV3Config& config : configs) {
    const int dim = tile.head_dim;
    const int heads = tile.num_heads;
    const int kv_heads = tile.num_kv_heads;
    const int kv_len = tile.kv_len > 0 ? tile.kv_len : tile.seq;
    const int q_count = tile.batch * tile.seq * heads * dim;
    const int kv_count = tile.batch * kv_len * kv_heads * dim;
    const int stats_count = nanochat::AttentionStatsCount(tile);

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
        tile, q.ptr, k.ptr, v.ptr, out.ptr, stats.ptr, config);
    Check(status == 0, "sweep launch");
    nanochat::kernels::Synchronize();

    std::vector<float> ref_stats;
    const std::vector<float> ref_out =
        AttentionForward(tile, q0, k0, v0, &ref_stats);
    char name[96];
    std::snprintf(name, sizeof(name), "sweep-br%d-bc%d-thr%d-mb%d", config.br,
                  config.bc, config.threads, config.min_blocks);
    CheckVectorClose(FromStorage(out.Download()), ref_out, kOutTol,
                     (std::string(name) + " out").c_str());
    CheckVectorClose(stats.Download(), ref_stats, kStatsTol,
                     (std::string(name) + " stats").c_str());
  }

  // Production training shape: batch=8, seq=512, heads=4, kv-heads=4,
  // head_dim=128. The tile must match the shipped `kernels::AttentionForward`
  // across the mask and layout contracts.
  AttentionParams d8;
  d8.batch = 8;
  d8.seq = 512;
  d8.num_heads = 4;
  d8.num_kv_heads = 4;
  d8.head_dim = 128;
  d8.causal = true;
  d8.window_left = -1;
  d8.window_right = 0;
  d8.kv_len = 0;
  d8.scale = 0.0f;
  ExpectShippedMatch(d8, "d8_s512-causal");

  AttentionParams d8_window = d8;
  d8_window.window_left = 128;
  ExpectShippedMatch(d8_window, "d8_s512-win128");

  // Left-only window: a non-causal mask that keeps a finite past but every
  // future key.
  AttentionParams d8_left = d8;
  d8_left.causal = false;
  d8_left.window_left = 128;
  d8_left.window_right = -1;
  ExpectShippedMatch(d8_left, "d8_s512-left-window");

  // Right-only window: a non-causal mask that keeps a finite future but every
  // past key.
  AttentionParams d8_right = d8;
  d8_right.causal = false;
  d8_right.window_left = -1;
  d8_right.window_right = 128;
  ExpectShippedMatch(d8_right, "d8_s512-right-window");

  // Grouped-query attention: 8 query heads share 2 key/value heads.
  AttentionParams d8_gqa = d8;
  d8_gqa.batch = 2;
  d8_gqa.num_heads = 8;
  d8_gqa.num_kv_heads = 2;
  ExpectShippedMatch(d8_gqa, "d8_s512-gqa");

  // KV-cache offset: 640 key/value rows for 512 query rows, so the absolute
  // query position starts at 128 and the causal tile skip must account for it.
  AttentionParams d8_kv = d8;
  d8_kv.kv_len = 640;
  ExpectShippedMatch(d8_kv, "d8_s512-kv-offset");

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
