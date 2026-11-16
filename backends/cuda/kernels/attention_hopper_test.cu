// T1 GPU correctness test for the Hopper attention forward tile of
// `docs/attention.md` Part 8: the warpgroup product over a key ring and a value
// ring that the copy engine fills, with one producer warpgroup and two consumer
// warpgroups.
//
// Every case is compared with `seqref::AttentionForward`, which is the same
// host reference the family test uses: the output over all rows and the two
// saved statistics, at the bf16 tolerance of `docs/testing.md`. The cases cover
// the three ways the tile can be wrong without being wrong everywhere: the mask
// of a tile that is not fully visible, the tail that is, and the layout of the
// query. The last case runs the seam and the tile on one set of operands and
// requires them to agree bit for bit, so a dispatch that routes to the family
// fails here instead of passing quietly.
//
//   tools/nanochat test --gpu //backends/cuda/kernels:attention_hopper_test
//
// The target states `archs = ["sm_90a"]`, so it runs on the H100 leg only. A
// card that does not answer for the tile fails the first check instead of
// skipping the run, because a silent skip would report a pass for a kernel that
// never ran.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/attention_hopper.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AttentionParams;
using nanochat::AttentionStatsCount;
using nanochat::ComputeType;
using nanochat::dev::Check;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::seqref::AttentionForward;

// The bf16 bound of docs/testing.md. The tile rounds the operands, the
// probability, and the value product, so the fp32 tolerance of the family would
// fail on the rounding alone.
constexpr double kTolerance = 3e-2;

// The model-stream rows `(b, t, h)` and the head-major rows `(b, h, t)` hold
// the same numbers; `to_heads` picks the direction.
std::vector<float> PermuteQuery(const std::vector<float>& rows, int batch,
                                int seq, int heads, int dim, bool to_heads) {
  std::vector<float> out(rows.size());
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          const std::size_t stream =
              ((static_cast<std::size_t>(b) * seq + t) * heads + h) * dim + d;
          const std::size_t head_major =
              ((static_cast<std::size_t>(b) * heads + h) * seq + t) * dim + d;
          out[to_heads ? head_major : stream] =
              rows[to_heads ? stream : head_major];
        }
      }
    }
  }
  return out;
}

// One shape: the tile's output and both statistics against the host reference.
void RunCase(const AttentionParams& p, const char* name, bool q_head_major) {
  std::printf("case: %s\n", name);
  std::fflush(stdout);
  const int dim = p.head_dim;
  const int heads = p.num_heads;
  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : heads;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int q_count = p.batch * p.seq * heads * dim;
  const int kv_count = p.batch * kv_len * kv_heads * dim;
  const int stats_count = AttentionStatsCount(p);

  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(kv_count, &rng);
  const std::vector<float> v0 = RandomVec(kv_count, &rng);

  AttentionParams device_p = p;
  device_p.q_head_major = q_head_major;
  const std::vector<float> q_device =
      q_head_major ? PermuteQuery(q0, p.batch, p.seq, heads, dim,
                                  /*to_heads=*/true)
                   : q0;

  DevBuf<ComputeType> q(ToStorage(q_device));
  DevBuf<ComputeType> k(ToStorage(k0));
  DevBuf<ComputeType> v(ToStorage(v0));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(stats_count);
  nanochat::kernels::LaunchHopperAttention(device_p, q.ptr, k.ptr, v.ptr,
                                           out.ptr, stats.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_stats;
  const std::vector<float> ref_out =
      AttentionForward(p, q0, k0, v0, &ref_stats);
  const std::string base(name);
  CheckVectorClose(FromStorage(out.Download()), ref_out, kTolerance,
                   (base + " fwd out").c_str());
  CheckVectorClose(stats.Download(), ref_stats, kTolerance,
                   (base + " fwd stats").c_str());
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("attention_hopper: starting\n");
  std::fflush(stdout);

  // The tile answers for this card and this shape with no variable set, which
  // is the promoted default: every case below runs on the kernel and not on a
  // stand-in, and the seam must route to the same kernel (case `seam`).
  AttentionParams production;
  production.batch = 2;
  production.seq = 512;
  production.num_heads = 4;
  production.num_kv_heads = 4;
  production.head_dim = 128;
  production.causal = true;
  production.window_left = -1;
  production.window_right = 0;
  production.kv_len = production.seq;
  Check(nanochat::kernels::UseHopperAttention(production),
        "the Hopper tile answers for the production shape");

  // A head width of 64 is the family's and not this tile's, so the predicate
  // must refuse it. The card answers yes for the tile, so this is the shape
  // rule under test.
  AttentionParams narrow = production;
  narrow.head_dim = 64;
  Check(!nanochat::kernels::UseHopperAttention(narrow),
        "the Hopper tile refuses a head width of 64");

  // Dense causal: the boundary tile of every query block takes the mask, and
  // the interior tiles take the unmasked path.
  RunCase(production, "causal-d128", /*q_head_major=*/false);
  RunCase(production, "causal-d128-head-major", /*q_head_major=*/true);

  // The sliding window: the leading tiles are skipped, the two boundary tiles
  // are masked per row, and a whole tile can be hidden for some rows of a block
  // and visible for others.
  AttentionParams windowed = production;
  windowed.window_left = 128;
  RunCase(windowed, "window-d128", /*q_head_major=*/false);
  RunCase(windowed, "window-d128-head-major", /*q_head_major=*/true);

  // Grouped-query attention: four query heads over two key-value heads, so the
  // map that turns a query head into a key-value head is under test.
  AttentionParams grouped = production;
  grouped.num_heads = 8;
  grouped.num_kv_heads = 2;
  grouped.window_left = 128;
  RunCase(grouped, "gqa-window-d128", /*q_head_major=*/false);

  // A sequence of one query block: the block is the boundary tile from its
  // first row to its last, and the key ring holds one tile.
  AttentionParams single = production;
  single.seq = 128;
  single.kv_len = single.seq;
  single.batch = 1;
  RunCase(single, "single-block-d128", /*q_head_major=*/false);

  // A sequence that is not a multiple of the query block: the last block holds
  // rows past the sequence, which must stay zeroed and must not be written.
  // The last key tile of the last block also runs past the sequence, so the box
  // the engine copies is wider than the rows that exist.
  AttentionParams tail = production;
  tail.batch = 1;
  tail.seq = 200;
  tail.kv_len = tail.seq;
  RunCase(tail, "tail-d128", /*q_head_major=*/false);
  AttentionParams tail_windowed = tail;
  tail_windowed.window_left = 100;
  RunCase(tail_windowed, "tail-window-d128", /*q_head_major=*/false);

  // Two query blocks over three key tiles, with a window that starts and ends
  // inside a tile: the per-block key range and the statistics offset of the
  // second block are cheapest to get wrong here.
  AttentionParams boundary = production;
  boundary.batch = 1;
  boundary.seq = 384;
  boundary.kv_len = boundary.seq;
  boundary.window_left = 130;
  RunCase(boundary, "block-boundary-window-d128", /*q_head_major=*/false);

  // The seam must route to this kernel and not to the family, and the two must
  // agree bit for bit: one launch of one kernel on one set of operands, so any
  // difference is the dispatch.
  {
    AttentionParams p = production;
    p.window_left = 128;
    const int dim = p.head_dim;
    const int heads = p.num_heads;
    const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
    const int q_count = p.batch * p.seq * heads * dim;
    const int kv_count = p.batch * kv_len * p.num_kv_heads * dim;
    const int stats_count = AttentionStatsCount(p);
    Rng rng;
    DevBuf<ComputeType> q(ToStorage(RandomVec(q_count, &rng)));
    DevBuf<ComputeType> k(ToStorage(RandomVec(kv_count, &rng)));
    DevBuf<ComputeType> v(ToStorage(RandomVec(kv_count, &rng)));
    DevBuf<ComputeType> seam_out(q_count);
    DevBuf<float> seam_stats(stats_count);
    DevBuf<ComputeType> direct_out(q_count);
    DevBuf<float> direct_stats(stats_count);
    std::printf("case: seam\n");
    std::fflush(stdout);
    nanochat::kernels::AttentionForward(p, q.ptr, k.ptr, v.ptr, seam_out.ptr,
                                        seam_stats.ptr);
    nanochat::kernels::LaunchHopperAttention(p, q.ptr, k.ptr, v.ptr,
                                             direct_out.ptr, direct_stats.ptr);
    nanochat::kernels::Synchronize();
    CheckVectorClose(FromStorage(seam_out.Download()),
                     FromStorage(direct_out.Download()), 0.0,
                     "seam fwd out equals the tile");
    CheckVectorClose(seam_stats.Download(), direct_stats.Download(), 0.0,
                     "seam fwd stats equal the tile");
  }

  if (Failures() != 0) {
    std::printf("attention_hopper: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("attention_hopper: all checks passed\n");
  return 0;
}
