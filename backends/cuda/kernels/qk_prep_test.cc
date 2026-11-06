// T1 GPU correctness + finite-difference test for the QkPrep family (fused
// RMSNorm -> RoPE -> scale on q and k). Two tiny attention shapes -- B=2, T=8,
// head_dim=4, num_heads=2, plus a grouped-query variant -- and two model-shape
// cases at head_dim 128: the grouped-query geometry the 20-layer shape runs,
// and a nine-row case that leaves the warp schedule's last block partly idle.
// Every case compares at kQkPrepPathAtol. See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/row_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::QkPrepParams;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::rowref::QkPrepBackward;
using nanochat::dev::rowref::QkPrepForward;

// The bound of a comparison between two code paths inside one build follows the
// storage precision (docs/testing.md: "a test that compares two code paths
// inside one build derives its bound from the storage precision"). 1e-6 is the
// honest fp32 number here: both paths then hold the same rows in fp32, and the
// only difference left is the order and the width of the two row statistics,
// which is far below a part in a million. A bf16 build rounds the saved rows
// and the gradients themselves, so it needs the coarse arm.
constexpr double kQkPrepPathAtol = nanochat::kComputeIsBf16 ? 1e-4 : 1e-6;

// The synthetic tables of the two tiny cases: one fixed angle per (position,
// pair). Kept so no existing check changes meaning when the model shape brings
// the real base in.
void SyntheticTables(int seq, int half, std::vector<float>* cos,
                     std::vector<float>* sin) {
  cos->assign(static_cast<std::size_t>(seq) * half, 0.0f);
  sin->assign(static_cast<std::size_t>(seq) * half, 0.0f);
  for (int t = 0; t < seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float theta =
          0.1f * static_cast<float>(t + 1) * static_cast<float>(d + 1);
      (*cos)[static_cast<std::size_t>(t) * half + d] = std::cos(theta);
      (*sin)[static_cast<std::size_t>(t) * half + d] = std::sin(theta);
    }
  }
}

// The tables a model shape carries, fp32 of shape [seq, head_dim / 2] as the
// seam says: angle(t, d) = t * rope_base ** (-2d / head_dim), the same
// arithmetic as `TrainModel::BuildRope` in src/model.cc and the `rope`
// obligation of spec/capability_bits.md. `rope_base` is 100000.0f in every
// shipped shape (models/nanochat-default/spec.json).
void ModelTables(int seq, int head_dim, float rope_base,
                 std::vector<float>* cos, std::vector<float>* sin) {
  const int half = head_dim / 2;
  cos->assign(static_cast<std::size_t>(seq) * half, 0.0f);
  sin->assign(static_cast<std::size_t>(seq) * half, 0.0f);
  for (int t = 0; t < seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float exponent =
          static_cast<float>(2 * d) / static_cast<float>(head_dim);
      const float inv_freq = 1.0f / std::pow(rope_base, exponent);
      const float angle = static_cast<float>(t) * inv_freq;
      (*cos)[static_cast<std::size_t>(t) * half + d] = std::cos(angle);
      (*sin)[static_cast<std::size_t>(t) * half + d] = std::sin(angle);
    }
  }
}

// One case. `cos` and `sin` are the rotary tables of the shape the case runs
// at, and `bound` is the absolute bound this case compares the two paths at, so
// a case carries its own bound instead of sharing one hardcoded number.
// `check_grad` runs the finite-difference sweep: it costs two reference forward
// passes per element, so a shape of a few thousand elements carries it and the
// model shape does not (see the call site).// Model-stream rows `(b, t, h)` to
// head-major rows `(b, h, t)`.
std::vector<float> PermuteToHeads(const std::vector<float>& rows, int batch,
                                  int seq, int heads, int dim) {
  std::vector<float> out(rows.size());
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          const std::size_t src =
              ((static_cast<std::size_t>(b) * seq + t) * heads + h) * dim + d;
          const std::size_t dst =
              ((static_cast<std::size_t>(b) * heads + h) * seq + t) * dim + d;
          out[dst] = rows[src];
        }
      }
    }
  }
  return out;
}

// The head-major query: `QkPrepForward` reads the saved rows from `q` and
// writes `[batch, heads, seq, dim]` to `q_out`; `QkPrepBackward` reads the
// upstream query gradient in that layout. The values must equal the
// model-stream result bit for bit, because only the destination index moves.
void RunHeadMajorCase(const QkPrepParams& p, double bound,
                      const std::vector<float>& cos,
                      const std::vector<float>& sin) {
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;
  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(k_count, &rng);
  const std::vector<float> wq = RandomVec(q_count, &rng);
  const std::vector<float> wk = RandomVec(k_count, &rng);
  DevBuf<float> dcos(cos);
  DevBuf<float> dsin(sin);

  QkPrepParams row = p;
  QkPrepParams heads = p;
  heads.q_head_major = true;

  // Forward in both layouts.
  DevBuf<ComputeType> q_rm(ToStorage(q0));
  DevBuf<ComputeType> k_rm(ToStorage(k0));
  nanochat::kernels::QkPrepForward(row, dcos.ptr, dsin.ptr, q_rm.ptr, k_rm.ptr);

  DevBuf<ComputeType> q_in(ToStorage(q0));
  DevBuf<ComputeType> q_hm(q_count);
  DevBuf<ComputeType> k_hm(ToStorage(k0));
  heads.q_out = q_hm.ptr;
  nanochat::kernels::QkPrepForward(heads, dcos.ptr, dsin.ptr, q_in.ptr,
                                   k_hm.ptr);
  nanochat::kernels::Synchronize();

  // The two forward results must agree once both are head-major, and the key
  // must not move at all.
  const std::vector<float> want_hm = PermuteToHeads(
      FromStorage(q_rm.Download()), p.batch, p.seq, p.num_heads, p.head_dim);
  CheckVectorClose(FromStorage(q_hm.Download()), want_hm, 0.0,
                   "qk fwd q head-major");
  CheckVectorClose(FromStorage(k_hm.Download()), FromStorage(k_rm.Download()),
                   0.0, "qk fwd k head-major");

  // The widened key: every query head of a group receives its key/value
  // head's prepared row, so the batched products can read it unwidened.
  const int k_widen = p.batch * p.seq * p.num_heads * p.head_dim;
  DevBuf<ComputeType> k_in(ToStorage(k0));
  DevBuf<ComputeType> q_side(ToStorage(q0));
  DevBuf<ComputeType> k_widened(k_widen);
  QkPrepParams k_heads = p;
  k_heads.k_out = k_widened.ptr;
  nanochat::kernels::QkPrepForward(k_heads, dcos.ptr, dsin.ptr, q_side.ptr,
                                   k_in.ptr);
  nanochat::kernels::Synchronize();
  const int group = p.num_heads / p.num_kv_heads;
  std::vector<float> wide(static_cast<std::size_t>(k_widen), 0.0f);
  const std::vector<float> k_row = FromStorage(k_rm.Download());
  for (int b = 0; b < p.batch; ++b) {
    for (int t = 0; t < p.seq; ++t) {
      for (int dh = 0; dh < p.num_heads; ++dh) {
        const int kh = dh / group;
        for (int d = 0; d < p.head_dim; ++d) {
          const std::size_t src =
              ((static_cast<std::size_t>(b) * p.seq + t) * p.num_kv_heads +
               kh) *
                  p.head_dim +
              d;
          const std::size_t dst =
              ((static_cast<std::size_t>(b) * p.seq + t) * p.num_heads + dh) *
                  p.head_dim +
              d;
          wide[dst] = k_row[src];
        }
      }
    }
  }
  CheckVectorClose(
      FromStorage(k_widened.Download()),
      PermuteToHeads(wide, p.batch, p.seq, p.num_heads, p.head_dim), 0.0,
      "qk fwd k widened");

  // Backward in both layouts.
  DevBuf<ComputeType> q_saved_rm(ToStorage(q0));
  DevBuf<ComputeType> k_saved_rm(ToStorage(k0));
  DevBuf<ComputeType> dq_rm(ToStorage(wq));
  DevBuf<ComputeType> dk_rm(ToStorage(wk));
  nanochat::kernels::QkPrepBackward(row, dcos.ptr, dsin.ptr, dq_rm.ptr,
                                    dk_rm.ptr, q_saved_rm.ptr, k_saved_rm.ptr);

  DevBuf<ComputeType> q_saved_hm(ToStorage(q0));
  DevBuf<ComputeType> k_saved_hm(ToStorage(k0));
  DevBuf<ComputeType> dq_hm(
      ToStorage(PermuteToHeads(wq, p.batch, p.seq, p.num_heads, p.head_dim)));
  DevBuf<ComputeType> dk_hm(ToStorage(wk));
  nanochat::kernels::QkPrepBackward(heads, dcos.ptr, dsin.ptr, dq_hm.ptr,
                                    dk_hm.ptr, q_saved_hm.ptr, k_saved_hm.ptr);
  nanochat::kernels::Synchronize();

  CheckVectorClose(FromStorage(q_saved_hm.Download()),
                   FromStorage(q_saved_rm.Download()), 0.0,
                   "qk bwd q head-major");
  CheckVectorClose(FromStorage(k_saved_hm.Download()),
                   FromStorage(k_saved_rm.Download()), 0.0,
                   "qk bwd k head-major");

  // And the model-stream point against the composition, at the case's bound.
  std::vector<float> ref_q = q0;
  std::vector<float> ref_k = k0;
  QkPrepForward(p, cos, sin, &ref_q, &ref_k);
  CheckVectorClose(FromStorage(q_rm.Download()), ref_q, bound, "qk fwd q");
  CheckVectorClose(FromStorage(k_rm.Download()), ref_k, bound, "qk fwd k");
}

void RunCase(const QkPrepParams& p, double bound, const std::vector<float>& cos,
             const std::vector<float>& sin, bool check_grad) {
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;

  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(k_count, &rng);
  const std::vector<float> wq = RandomVec(q_count, &rng);
  const std::vector<float> wk = RandomVec(k_count, &rng);

  DevBuf<float> dcos(cos);
  DevBuf<float> dsin(sin);

  // Forward.
  DevBuf<ComputeType> q(ToStorage(q0));
  DevBuf<ComputeType> k(ToStorage(k0));
  nanochat::kernels::QkPrepForward(p, dcos.ptr, dsin.ptr, q.ptr, k.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_q = q0;
  std::vector<float> ref_k = k0;
  QkPrepForward(p, cos, sin, &ref_q, &ref_k);
  CheckVectorClose(FromStorage(q.Download()), ref_q, bound, "qk fwd q");
  CheckVectorClose(FromStorage(k.Download()), ref_k, bound, "qk fwd k");

  // Backward: the saved buffers hold the pre-norm projections.
  DevBuf<ComputeType> dq(ToStorage(wq));
  DevBuf<ComputeType> dk(ToStorage(wk));
  DevBuf<ComputeType> q_saved(ToStorage(q0));
  DevBuf<ComputeType> k_saved(ToStorage(k0));
  nanochat::kernels::QkPrepBackward(p, dcos.ptr, dsin.ptr, dq.ptr, dk.ptr,
                                    q_saved.ptr, k_saved.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_dq = q0;
  std::vector<float> ref_dk = k0;
  QkPrepBackward(p, cos, sin, wq, wk, &ref_dq, &ref_dk);
  const std::vector<float> got_dq = FromStorage(q_saved.Download());
  const std::vector<float> got_dk = FromStorage(k_saved.Download());
  CheckVectorClose(got_dq, ref_dq, bound, "qk bwd dq");
  CheckVectorClose(got_dk, ref_dk, bound, "qk bwd dk");

  if (check_grad) {
    auto loss_at = [&](const std::vector<float>& qv,
                       const std::vector<float>& kv) {
      std::vector<float> qb = qv;
      std::vector<float> kb = kv;
      QkPrepForward(p, cos, sin, &qb, &kb);
      double loss = 0.0;
      for (std::size_t i = 0; i < qb.size(); ++i) loss += qb[i] * wq[i];
      for (std::size_t i = 0; i < kb.size(); ++i) loss += kb[i] * wk[i];
      return loss;
    };
    CheckFiniteDifference(
        q0, got_dq,
        [&](const std::vector<float>& qv) { return loss_at(qv, k0); }, 2e-3,
        "qk dq");
    CheckFiniteDifference(
        k0, got_dk,
        [&](const std::vector<float>& kv) { return loss_at(q0, kv); }, 2e-3,
        "qk dk");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("qk_prep gpu test\n");

  // The tiny cases keep the synthetic tables of the row that first wrote them.
  // They now compare at kQkPrepPathAtol too: against the composition's
  // arithmetic the backward is bit-identical at this size and the forward is
  // 3.6e-07 away, so 1e-4 was the bound of that first row and not a measured
  // floor. Tightening a bound is allowed; widening never is.
  constexpr int kTinySeq = 8;
  constexpr int kTinyHeadDim = 4;
  std::vector<float> tiny_cos;
  std::vector<float> tiny_sin;
  SyntheticTables(kTinySeq, kTinyHeadDim / 2, &tiny_cos, &tiny_sin);

  QkPrepParams p;
  p.batch = 2;
  p.seq = kTinySeq;
  p.num_heads = 2;
  p.num_kv_heads = 2;
  p.head_dim = kTinyHeadDim;
  p.eps = 1e-6f;
  p.scale = 1.2f;
  RunCase(p, kQkPrepPathAtol, tiny_cos, tiny_sin, /*check_grad=*/true);

  // Grouped-query variant: two query heads share one key/value head.
  QkPrepParams gqa = p;
  gqa.num_kv_heads = 1;
  RunCase(gqa, kQkPrepPathAtol, tiny_cos, tiny_sin, /*check_grad=*/true);
  // The head-major query on the block schedule (head_dim 4).
  RunHeadMajorCase(p, kQkPrepPathAtol, tiny_cos, tiny_sin);
  RunHeadMajorCase(gqa, kQkPrepPathAtol, tiny_cos, tiny_sin);

  // The model shape: head_dim 128, seq 64, batch 2, the real rotary base
  // 100000, scale 1.2, and the grouped-query geometry the 20-layer shape runs.
  constexpr int kModelHeadDim = 128;
  constexpr int kModelSeq = 64;
  constexpr float kRopeBase = 100000.0f;
  std::vector<float> model_cos;
  std::vector<float> model_sin;
  ModelTables(kModelSeq, kModelHeadDim, kRopeBase, &model_cos, &model_sin);

  QkPrepParams model;
  model.batch = 2;
  model.seq = kModelSeq;
  model.num_heads = 2;
  model.num_kv_heads = 1;
  model.head_dim = kModelHeadDim;
  model.eps = 1e-6f;
  model.scale = 1.2f;
  // No finite difference here: the sweep costs two reference forward passes per
  // element, which is about 9 s at 32768 elements, and the gradient's shape is
  // the tiny cases' check. This case exists to compare the two code paths, and
  // it does that at four bounds (forward q and k, backward dq and dk). Measured
  // on this case at kQkPrepPathAtol: forward 7.15e-07, backward 0.00, the
  // backward being bit-identical once its two row statistics accumulate in
  // double.
  RunCase(model, kQkPrepPathAtol, model_cos, model_sin, /*check_grad=*/false);
  // The head-major query on the warp schedule (head_dim 128).
  RunHeadMajorCase(model, kQkPrepPathAtol, model_cos, model_sin);

  // A ragged row count on the warp schedule: nine rows over three blocks, so
  // the third block runs one warp and the other three exit. The warp schedule
  // must mask the tail by warp, not by lane.
  QkPrepParams ragged = model;
  ragged.batch = 3;
  ragged.seq = 3;
  ragged.num_heads = 1;
  ragged.num_kv_heads = 1;
  RunCase(ragged, kQkPrepPathAtol, model_cos, model_sin, /*check_grad=*/false);

  if (Failures() != 0) {
    std::printf("qk_prep: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("qk_prep: all checks passed\n");
  return 0;
}
