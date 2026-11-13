// T1 GPU correctness + finite-difference test for the Attention family:
// causal / sliding-window / grouped-query attention, forward and backward,
// with the saved softmax statistics. Tiny attention shapes only
// (B=2, T=8, head_dim=4, num_heads=2, and four query heads over two key/value
// heads for the group fold). See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ComputeType;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::seqref::AttentionBackward;
using nanochat::dev::seqref::AttentionForward;

// Model-stream rows `(b, t, h)` and head-major rows `(b, h, t)` hold the same
// values; `to_heads` picks the direction.
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

// The head-major query at one shape: the same numbers in the layout the
// batched products read. No arithmetic changes, so every result must match the
// model-stream run bit for bit; the query gradient is the one that moves.
void RunHeadMajorCase(const AttentionParams& p, const char* name) {
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
  const std::vector<float> w = RandomVec(q_count, &rng);

  AttentionParams heads_p = p;
  heads_p.q_head_major = true;

  DevBuf<ComputeType> q_rm(ToStorage(q0));
  DevBuf<ComputeType> k_rm(ToStorage(k0));
  DevBuf<ComputeType> v_rm(ToStorage(v0));
  DevBuf<ComputeType> out_rm(q_count);
  DevBuf<float> stats_rm(stats_count);
  nanochat::kernels::AttentionForward(p, q_rm.ptr, k_rm.ptr, v_rm.ptr,
                                      out_rm.ptr, stats_rm.ptr);
  DevBuf<ComputeType> dout_rm(ToStorage(w));
  DevBuf<ComputeType> dq_rm(q_count);
  DevBuf<ComputeType> dk_rm(kv_count);
  DevBuf<ComputeType> dv_rm(kv_count);
  nanochat::kernels::AttentionBackward(p, q_rm.ptr, k_rm.ptr, v_rm.ptr,
                                       stats_rm.ptr, dout_rm.ptr, dq_rm.ptr,
                                       dk_rm.ptr, dv_rm.ptr);

  DevBuf<ComputeType> q_hm(ToStorage(
      PermuteQuery(q0, p.batch, p.seq, heads, dim, /*to_heads=*/true)));
  DevBuf<ComputeType> k_hm(ToStorage(k0));
  DevBuf<ComputeType> v_hm(ToStorage(v0));
  DevBuf<ComputeType> out_hm(q_count);
  DevBuf<float> stats_hm(stats_count);
  nanochat::kernels::AttentionForward(heads_p, q_hm.ptr, k_hm.ptr, v_hm.ptr,
                                      out_hm.ptr, stats_hm.ptr);
  DevBuf<ComputeType> dout_hm(ToStorage(w));
  DevBuf<ComputeType> dq_hm(q_count);
  DevBuf<ComputeType> dk_hm(kv_count);
  DevBuf<ComputeType> dv_hm(kv_count);
  nanochat::kernels::AttentionBackward(heads_p, q_hm.ptr, k_hm.ptr, v_hm.ptr,
                                       stats_hm.ptr, dout_hm.ptr, dq_hm.ptr,
                                       dk_hm.ptr, dv_hm.ptr);
  nanochat::kernels::Synchronize();

  const std::string base(name);
  CheckVectorClose(FromStorage(out_hm.Download()),
                   FromStorage(out_rm.Download()), 0.0,
                   (base + " fwd out head-major").c_str());
  CheckVectorClose(stats_hm.Download(), stats_rm.Download(), 0.0,
                   (base + " fwd stats head-major").c_str());
  const std::vector<float> dq_back =
      PermuteQuery(FromStorage(dq_hm.Download()), p.batch, p.seq, heads, dim,
                   /*to_heads=*/false);
  CheckVectorClose(dq_back, FromStorage(dq_rm.Download()), 0.0,
                   (base + " bwd dq head-major").c_str());
  CheckVectorClose(FromStorage(dk_hm.Download()), FromStorage(dk_rm.Download()),
                   0.0, (base + " bwd dk head-major").c_str());
  CheckVectorClose(FromStorage(dv_hm.Download()), FromStorage(dv_rm.Download()),
                   0.0, (base + " bwd dv head-major").c_str());
}

// The tolerance of a comparison against the fp32 host reference. A bf16 build
// rounds the operands, the stored probability, and every product, so the fp32
// value 1e-4 would fail on the rounding alone; 3e-2 is the bf16 bound of
// docs/testing.md. A finite difference over bf16 output is noise, not gradient
// information, so only the fp32 build differentiates.
constexpr double kFamilyTolerance = nanochat::kComputeIsBf16 ? 3e-2 : 1e-4;

void RunCase(const AttentionParams& p, const char* name,
             bool differentiate = true, bool q_head_major = false) {
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
  const std::vector<float> w = RandomVec(q_count, &rng);

  // A head-major query carries the same numbers in the layout the fused
  // `qk_prep` writes; the reference reads the model stream, so the case sends
  // the permuted copy to the device and permutes the query gradient back.
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
  nanochat::kernels::AttentionForward(device_p, q.ptr, k.ptr, v.ptr, out.ptr,
                                      stats.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_stats;
  const std::vector<float> ref_out =
      AttentionForward(p, q0, k0, v0, &ref_stats);
  CheckVectorClose(FromStorage(out.Download()), ref_out, kFamilyTolerance,
                   (std::string(name) + " fwd out").c_str());
  CheckVectorClose(stats.Download(), ref_stats, kFamilyTolerance,
                   (std::string(name) + " fwd stats").c_str());

  DevBuf<ComputeType> dout(ToStorage(w));
  DevBuf<ComputeType> dq(q_count);
  DevBuf<ComputeType> dk(kv_count);
  DevBuf<ComputeType> dv(kv_count);
  nanochat::kernels::AttentionBackward(device_p, q.ptr, k.ptr, v.ptr, stats.ptr,
                                       dout.ptr, dq.ptr, dk.ptr, dv.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_dq;
  std::vector<float> ref_dk;
  std::vector<float> ref_dv;
  AttentionBackward(p, q0, k0, v0, ref_stats, w, &ref_dq, &ref_dk, &ref_dv);
  const std::vector<float> dq_device =
      q_head_major ? PermuteQuery(FromStorage(dq.Download()), p.batch, p.seq,
                                  heads, dim, /*to_heads=*/false)
                   : FromStorage(dq.Download());
  CheckVectorClose(dq_device, ref_dq, kFamilyTolerance,
                   (std::string(name) + " bwd dq").c_str());
  CheckVectorClose(FromStorage(dk.Download()), ref_dk, kFamilyTolerance,
                   (std::string(name) + " bwd dk").c_str());
  CheckVectorClose(FromStorage(dv.Download()), ref_dv, kFamilyTolerance,
                   (std::string(name) + " bwd dv").c_str());

  if (!differentiate) return;
  // A bf16 build stores the loss output, so a central difference of 1e-2 over
  // it measures the rounding, not the gradient. The fp32 build is the one that
  // differentiates (docs/testing.md).
  if (nanochat::kComputeIsBf16) return;
  auto loss_at = [&](const std::vector<float>& qv, const std::vector<float>& kv,
                     const std::vector<float>& vv) {
    std::vector<float> s;
    const std::vector<float> o = AttentionForward(p, qv, kv, vv, &s);
    double loss = 0.0;
    for (std::size_t i = 0; i < o.size(); ++i) loss += o[i] * w[i];
    return loss;
  };
  CheckFiniteDifference(
      q0, FromStorage(dq.Download()),
      [&](const std::vector<float>& qv) { return loss_at(qv, k0, v0); }, 3e-3,
      (std::string(name) + " dq").c_str());
  CheckFiniteDifference(
      k0, FromStorage(dk.Download()),
      [&](const std::vector<float>& kv) { return loss_at(q0, kv, v0); }, 3e-3,
      (std::string(name) + " dk").c_str());
  CheckFiniteDifference(
      v0, FromStorage(dv.Download()),
      [&](const std::vector<float>& vv) { return loss_at(q0, k0, vv); }, 3e-3,
      (std::string(name) + " dv").c_str());
}

// A forward-only comparison, for the shapes the fused tiles cover. The tiles
// are forward paths, so the backward is not part of this check.
void RunForwardCase(const AttentionParams& p, const char* name, double tol,
                    bool q_head_major = false) {
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
  nanochat::kernels::AttentionForward(device_p, q.ptr, k.ptr, v.ptr, out.ptr,
                                      stats.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_stats;
  const std::vector<float> ref_out =
      AttentionForward(p, q0, k0, v0, &ref_stats);
  CheckVectorClose(FromStorage(out.Download()), ref_out, tol,
                   (std::string(name) + " fwd out").c_str());
  CheckVectorClose(stats.Download(), ref_stats, tol,
                   (std::string(name) + " fwd stats").c_str());
}

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++Failures();
  }
}

// The branch conditions of `docs/attention.md` Part 4, Part 6 section 5, and
// Part 7 section 5. The build chooses which family is live, so the check
// states the invariants that hold in either build.
void RunDispatchCase() {
  AttentionParams p;
  p.batch = 1;
  p.seq = 8;
  p.num_heads = 2;
  p.num_kv_heads = 2;
  p.head_dim = 128;
  p.causal = true;
  p.window_left = 4;
  p.window_right = 0;

  const bool fp32_tile = nanochat::kernels::FusedAttentionFp32Covers(p);
  const bool bf16_tile = nanochat::kernels::FusedAttentionBf16Covers(p, true);
  Expect(fp32_tile != bf16_tile,
         "exactly one `wmma` tile covers a windowed head-width-128 shape");
  Expect(!nanochat::kernels::FusedAttentionBf16Covers(p, false),
         "a device without tensor cores keeps the cuBLAS path");

  // The flash family of Part 7: a bf16 build on a tensor-core device takes the
  // windowed shape, and it is the branch `attention.cu` selects first.
  const bool flash = nanochat::kernels::FusedAttentionFlashCovers(p, true);
  Expect(flash == nanochat::kComputeIsBf16,
         "the flash family covers the windowed head-width-128 shape");
  Expect(!nanochat::kernels::FusedAttentionFlashCovers(p, false),
         "a device without tensor cores keeps the cuBLAS path");

  AttentionParams other = p;
  other.head_dim = 64;
  Expect(!nanochat::kernels::FusedAttentionFp32Covers(other),
         "head width 64 stays on cuBLAS in an fp32 build");
  Expect(!nanochat::kernels::FusedAttentionBf16Covers(other, true),
         "head width 64 keeps the cuBLAS path for the `wmma` tile");
  Expect(nanochat::kernels::FusedAttentionFlashCovers(other, true) ==
             nanochat::kComputeIsBf16,
         "the flash family covers head width 64");

  other = p;
  other.window_left = -1;
  Expect(!nanochat::kernels::FusedAttentionFp32Covers(other),
         "full-causal attention stays on cuBLAS for the fp32 tile");
  Expect(!nanochat::kernels::FusedAttentionBf16Covers(other, true),
         "full-causal attention keeps the `wmma` tile out");
  Expect(nanochat::kernels::FusedAttentionFlashCovers(other, true) ==
             nanochat::kComputeIsBf16,
         "the flash family covers full-causal attention");

  other = p;
  other.q_head_major = true;
  Expect(!nanochat::kernels::FusedAttentionFp32Covers(other),
         "a head-major query keeps the batched path for the fp32 tile");
  Expect(!nanochat::kernels::FusedAttentionBf16Covers(other, true),
         "a head-major query keeps the `wmma` tile out");
  Expect(nanochat::kernels::FusedAttentionFlashCovers(other, true) ==
             nanochat::kComputeIsBf16,
         "the flash family reads the head-major query");

  other = p;
  other.kv_len = 16;
  Expect(!nanochat::kernels::FusedAttentionFlashCovers(other, true),
         "a decode step keeps the cuBLAS path");

  other = p;
  other.causal = false;
  other.window_right = -1;
  Expect(!nanochat::kernels::FusedAttentionFlashCovers(other, true),
         "a two-sided window keeps the cuBLAS path");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("attention gpu test\n");

  AttentionParams base;
  base.batch = 2;
  base.seq = 8;
  base.num_heads = 2;
  base.num_kv_heads = 2;
  base.head_dim = 4;
  base.causal = true;
  base.window_left = -1;
  base.window_right = 0;
  base.kv_len = 0;
  base.scale = 0.0f;

  RunCase(base, "mha-causal");
  RunHeadMajorCase(base, "mha-causal");

  AttentionParams gqa = base;
  gqa.num_kv_heads = 1;
  RunCase(gqa, "gqa-causal");

  // Four query heads over two key/value heads. The forward widens both shared
  // heads into one batched call, and the backward folds each pair back. One
  // key/value head cannot check the fold's key/value row order, because that
  // order collapses to the same index when there is one head.
  AttentionParams gqa_pair = base;
  gqa_pair.num_heads = 4;
  gqa_pair.num_kv_heads = 2;
  RunCase(gqa_pair, "gqa-two-kv-heads");

  AttentionParams window = base;
  window.window_left = 3;
  RunCase(window, "sliding-window");

  // Grouped-query attention with a sliding window exercises the banded
  // backward, where the fold must still land in the model's key/value order.
  AttentionParams gqa_window = gqa;
  gqa_window.window_left = 3;
  RunCase(gqa_window, "gqa-sliding-window");

  // The same banded shape with two key/value heads, so the fold's two leading
  // indices cannot collapse into one another.
  AttentionParams gqa_pair_window = gqa_pair;
  gqa_pair_window.window_left = 3;
  RunCase(gqa_pair_window, "gqa-two-kv-heads-sliding-window");
  RunHeadMajorCase(gqa_pair_window, "gqa-two-kv-heads-sliding-window");

  AttentionParams noncausal = base;
  noncausal.causal = false;
  noncausal.window_right = -1;
  RunCase(noncausal, "noncausal");

  AttentionParams decode = base;
  decode.kv_len = 10;
  RunCase(decode, "kv-cache-offset");
  RunHeadMajorCase(decode, "kv-cache-offset");

  // The fused softmax pair holds one row in registers when the row is at most
  // 128 threads times 8 elements, so a row of 1024 is the last length that
  // takes it and every shorter case above takes it too. This case sits exactly
  // on the bound, where an off-by-one in the unrolled loop would drop the last
  // element of the row and leave the sum of the weights below one.
  AttentionParams bound = base;
  bound.batch = 1;
  bound.seq = 1024;
  bound.num_heads = 1;
  bound.num_kv_heads = 1;
  RunCase(bound, "softmax-register-bound",
          /*differentiate=*/false);

  RunDispatchCase();

  // The fused tiles at head width 128, the only width they cover. A bf16 build
  // takes the tensor-core tile and an fp32 build takes the CUDA-core tile, so
  // these cases cover whichever tile the build has against the host reference.
  // The bf16 tolerance is the one of docs/testing.md: the operands, the stored
  // probability, and the value product all round to bf16, while the reference
  // stays in fp32.
  const double fused_tol = nanochat::kComputeIsBf16 ? 3e-2 : 1e-4;
  AttentionParams wide = base;
  wide.seq = 16;
  wide.num_kv_heads = 2;
  wide.head_dim = 128;
  wide.window_left = 5;
  RunForwardCase(wide, "window-d128", fused_tol);

  AttentionParams wide_full = wide;
  wide_full.window_left = 15;
  RunForwardCase(wide_full, "window-d128-near-full", fused_tol);

  AttentionParams wide_gqa = wide;
  wide_gqa.num_heads = 4;
  wide_gqa.num_kv_heads = 2;
  RunForwardCase(wide_gqa, "gqa-window-d128", fused_tol);

  // A decode-shaped call: fewer query rows than keys, so the mask runs in model
  // positions rather than in local ones.
  AttentionParams wide_decode = wide;
  wide_decode.seq = 8;
  wide_decode.kv_len = 24;
  wide_decode.window_left = 5;
  RunForwardCase(wide_decode, "kv-cache-window-d128", fused_tol);

  // A bidirectional window: no causal bound, so only the two window bounds
  // decide which key tiles are visible.
  AttentionParams wide_bidir = wide;
  wide_bidir.causal = false;
  wide_bidir.window_left = 5;
  wide_bidir.window_right = 3;
  RunForwardCase(wide_bidir, "bidirectional-window-d128", fused_tol);

  // A key length that ends inside the last key tile, so the staging path zeroes
  // a partial row.
  AttentionParams wide_tail = wide;
  wide_tail.seq = 20;
  wide_tail.kv_len = 0;
  wide_tail.window_left = 6;
  RunForwardCase(wide_tail, "window-d128-tile-tail", fused_tol);

  // The flash family of `docs/attention.md` Part 7 covers both passes. A bf16
  // build routes these cases to the flash forward tile and the two flash
  // backward kernels; an fp32 build keeps the `wmma` tile, the banded backward,
  // and the dense decomposition, so every case covers whichever the build has.
  // The bf16 build never differentiates, because a finite difference over a
  // bf16 output measures the rounding (docs/testing.md).
  RunCase(wide, "window-d128-fwd-bwd", /*differentiate=*/false);
  RunCase(wide_full, "window-d128-near-full-bwd", /*differentiate=*/false);
  RunCase(wide_gqa, "gqa-window-d128-fwd-bwd", /*differentiate=*/false);
  RunCase(wide, "window-d128-head-major-bwd", /*differentiate=*/false,
          /*q_head_major=*/true);
  RunCase(wide_gqa, "gqa-window-d128-head-major-bwd",
          /*differentiate=*/false, /*q_head_major=*/true);

  // Full causal attention is a flash case too: the family keeps the window in
  // the mask, so a window of -1 needs no other path.
  AttentionParams wide_causal = wide;
  wide_causal.window_left = -1;
  RunCase(wide_causal, "causal-d128-fwd-bwd", /*differentiate=*/false);

  // Causality off with the right bound taken from the window: the mask is the
  // only thing that bounds the keys from above.
  AttentionParams wide_left = wide;
  wide_left.causal = false;
  wide_left.window_left = 5;
  RunCase(wide_left, "left-window-d128-fwd-bwd", /*differentiate=*/false);

  // Head width 64, the other width the family covers.
  AttentionParams narrow = wide;
  narrow.head_dim = 64;
  RunCase(narrow, "window-d64-fwd-bwd", /*differentiate=*/false);
  AttentionParams narrow_causal = narrow;
  narrow_causal.window_left = -1;
  RunCase(narrow_causal, "causal-d64-fwd-bwd", /*differentiate=*/false);

  // Fewer key rows than query rows: the leading query rows have no visible key
  // at all, so they take the empty-window contract, and the key tiles end
  // inside the last block.
  AttentionParams wide_short_kv = wide;
  wide_short_kv.kv_len = 12;
  RunCase(wide_short_kv, "window-d128-short-kv-fwd-bwd",
          /*differentiate=*/false);

  // A sequence that spans two query blocks and several key tiles, with a
  // window that starts and ends inside a tile. The per-block key range, the
  // statistics offset of the second block, and the skip of the tiles a block
  // cannot see are all cheapest to get wrong here.
  AttentionParams wide_long = wide;
  wide_long.batch = 1;
  wide_long.seq = 200;
  wide_long.window_left = 100;
  RunCase(wide_long, "window-d128-block-boundary-fwd-bwd",
          /*differentiate=*/false);
  RunCase(wide_long, "window-d128-block-boundary-head-major-bwd",
          /*differentiate=*/false, /*q_head_major=*/true);

  if (Failures() != 0) {
    std::printf("attention: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("attention: all checks passed\n");
  return 0;
}
