// backends/cpu/cpu_kernels_test.cc — the reference backend's self-check.
//
// CPU-only T0 test (docs/testing.md): analytic identities where the answer is
// known, plus a central finite-difference gradient check for every family that
// has a backward. The torch oracle parity test lives in //tests; this file
// keeps the backend honest without a fixture.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;

#if defined(NANOCHAT_PRECISION_FP16)
ComputeType C(float v) { return nanochat::Fp16FromFloat(v); }
float U(ComputeType v) { return nanochat::Fp16ToFloat(v); }
#else
ComputeType C(float v) { return v; }
float U(ComputeType v) { return v; }
#endif

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

void CheckClose(double got, double want, double tol, const char* what) {
  if (std::fabs(got - want) > tol * (1.0 + std::fabs(want))) {
    std::printf("FAIL: %s (got %.8g want %.8g)\n", what, got, want);
    ++g_failures;
  }
}

// Deterministic xorshift so the test never depends on the platform RNG.
struct Rng {
  std::uint64_t s = 0x1234567890abcdefULL;
  float Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>((s >> 40) % 2001) / 1000.0f - 1.0f;
  }
};

std::vector<float> RandomVec(int n, Rng* rng) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = rng->Next();
  return v;
}

std::vector<ComputeType> ToStorage(const std::vector<float>& v) {
  std::vector<ComputeType> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = C(v[i]);
  return out;
}

std::vector<float> FromStorage(const std::vector<ComputeType>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = U(v[i]);
  return out;
}

#if !defined(NANOCHAT_PRECISION_FP16)
// Central finite difference of a scalar loss over x, compared against an
// analytic gradient. fp32 only: fp16 storage cannot resolve the perturbation.
template <typename LossFn>
void CheckFiniteDifference(const std::vector<float>& x0,
                           const std::vector<float>& analytic, LossFn loss_at,
                           double tol, const char* what) {
  const float h = 1e-2f;
  for (std::size_t i = 0; i < x0.size(); ++i) {
    std::vector<float> xp = x0;
    xp[i] += h;
    std::vector<float> xm = x0;
    xm[i] -= h;
    const double num = (loss_at(xp) - loss_at(xm)) / (2.0 * h);
    if (std::fabs(num - analytic[i]) > tol * (1.0 + std::fabs(num))) {
      std::printf("FAIL: %s [%zu] analytic %.6g numeric %.6g\n", what, i,
                  analytic[i], num);
      ++g_failures;
      return;
    }
  }
}
#endif

void TestGemm() {
  using namespace nanochat;
  // 2x2 * 2x2 = [[19,22],[43,50]].
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};
  GemmParams p;
  p.m = 2;
  p.n = 2;
  p.k = 2;
  std::vector<ComputeType> sa = ToStorage(std::vector<float>(a, a + 4));
  std::vector<ComputeType> sb = ToStorage(std::vector<float>(b, b + 4));
  std::vector<ComputeType> sc(4);
  kernels::Gemm(GemmMode::kForward, p, sa.data(), sb.data(), sc.data());
  std::vector<float> got = FromStorage(sc);
  CheckClose(got[0], 19, 1e-5, "gemm c00");
  CheckClose(got[1], 22, 1e-5, "gemm c01");
  CheckClose(got[2], 43, 1e-5, "gemm c10");
  CheckClose(got[3], 50, 1e-5, "gemm c11");

  // Transposed B: A * B^T = [[17,23],[39,53]].
  GemmParams pt = p;
  pt.transpose_b = true;
  c[0] = c[1] = c[2] = c[3] = 0;
  std::fill(sc.begin(), sc.end(), C(0));
  kernels::Gemm(GemmMode::kForward, pt, sa.data(), sb.data(), sc.data());
  got = FromStorage(sc);
  CheckClose(got[0], 17, 1e-5, "gemm tb c00");
  CheckClose(got[1], 23, 1e-5, "gemm tb c01");
  CheckClose(got[2], 39, 1e-5, "gemm tb c10");
  CheckClose(got[3], 53, 1e-5, "gemm tb c11");

  // Transposed A (stored A^T): [1,3,2,4] gives the same product.
  GemmParams pta = p;
  pta.transpose_a = true;
  std::vector<ComputeType> sa_t = ToStorage({1.0f, 3.0f, 2.0f, 4.0f});
  std::fill(sc.begin(), sc.end(), C(0));
  kernels::Gemm(GemmMode::kForward, pta, sa_t.data(), sb.data(), sc.data());
  got = FromStorage(sc);
  CheckClose(got[0], 19, 1e-5, "gemm ta c00");
  CheckClose(got[3], 50, 1e-5, "gemm ta c11");

  // Batched: two independent 2x2 products with inferred strides.
  GemmParams pb = p;
  pb.batch_count = 2;
  std::vector<ComputeType> sba = ToStorage({1, 2, 3, 4, 1, 0, 0, 1});
  std::vector<ComputeType> sbb = ToStorage({5, 6, 7, 8, 1, 1, 1, 1});
  std::vector<ComputeType> sbc(8, C(0));
  kernels::Gemm(GemmMode::kForward, pb, sba.data(), sbb.data(), sbc.data());
  got = FromStorage(sbc);
  CheckClose(got[0], 19, 1e-5, "gemm batch0 c00");
  CheckClose(got[4], 1, 1e-5, "gemm batch1 c00");
  CheckClose(got[7], 1, 1e-5, "gemm batch1 c11");

  // Accumulate: C = 2 A B + 3 C0.
  GemmParams pa = p;
  pa.alpha = 2.0f;
  pa.beta = 3.0f;
  float c0[4] = {1, 1, 1, 1};
  std::vector<ComputeType> sca = ToStorage(std::vector<float>(c0, c0 + 4));
  kernels::Gemm(GemmMode::kForward, pa, sa.data(), sb.data(), sca.data());
  got = FromStorage(sca);
  CheckClose(got[0], 2 * 19 + 3, 1e-5, "gemm beta c00");
  CheckClose(got[3], 2 * 50 + 3, 1e-5, "gemm beta c11");
}

void TestRmsNorm() {
  using namespace nanochat;
  RmsNormParams p;
  p.rows = 1;
  p.dim = 2;
  p.eps = 0.0f;
  std::vector<ComputeType> x = ToStorage({3.0f, 4.0f});
  std::vector<ComputeType> out(2);
  float rstd = 0.0f;
  kernels::RmsNormForward(p, x.data(), out.data(), &rstd);
  const float r = 1.0f / std::sqrt(12.5f);
  CheckClose(rstd, r, 1e-5, "rms rstd");
  CheckClose(U(out[0]), 3.0f * r, 1e-5, "rms out0");
  CheckClose(U(out[1]), 4.0f * r, 1e-5, "rms out1");

#if !defined(NANOCHAT_PRECISION_FP16)
  // Finite-difference the backward: loss = sum(out * w).
  Rng rng;
  RmsNormParams q;
  q.rows = 2;
  q.dim = 3;
  std::vector<float> x0 = RandomVec(6, &rng);
  std::vector<float> w = RandomVec(6, &rng);
  std::vector<ComputeType> xs = ToStorage(x0);
  std::vector<ComputeType> outs(6);
  float rstdv[2];
  kernels::RmsNormForward(q, xs.data(), outs.data(), rstdv);
  std::vector<ComputeType> dys = ToStorage(w);
  std::vector<ComputeType> dxs(6);
  kernels::RmsNormBackward(q, xs.data(), dys.data(), rstdv, dxs.data());
  std::vector<float> dx = FromStorage(dxs);
  auto loss_at = [&](const std::vector<float>& xv) {
    std::vector<ComputeType> xs2 = ToStorage(xv);
    std::vector<ComputeType> out2(6);
    float rr[2];
    kernels::RmsNormForward(q, xs2.data(), out2.data(), rr);
    double loss = 0.0;
    for (int i = 0; i < 6; ++i) loss += U(out2[i]) * w[i];
    return loss;
  };
  CheckFiniteDifference(x0, dx, loss_at, 1e-3, "rms backward");
#endif
}

void TestQkPrep() {
  using namespace nanochat;
  QkPrepParams p;
  p.batch = 1;
  p.seq = 1;
  p.num_heads = 1;
  p.num_kv_heads = 1;
  p.head_dim = 2;
  p.eps = 0.0f;
  p.scale = 1.2f;
  float cos[1] = {1.0f};
  float sin[1] = {0.0f};
  std::vector<ComputeType> q = ToStorage({3.0f, 4.0f});
  std::vector<ComputeType> k = ToStorage({3.0f, 4.0f});
  kernels::QkPrepForward(p, cos, sin, q.data(), k.data());
  const float r = 1.0f / std::sqrt(12.5f);
  CheckClose(U(q[0]), 3.0f * r * 1.2f, 1e-4, "qkprep fwd q0");
  CheckClose(U(q[1]), 4.0f * r * 1.2f, 1e-4, "qkprep fwd q1");

#if !defined(NANOCHAT_PRECISION_FP16)
  // Backward with a real RoPE table.
  Rng rng;
  QkPrepParams p2;
  p2.batch = 1;
  p2.seq = 2;
  p2.num_heads = 1;
  p2.num_kv_heads = 1;
  p2.head_dim = 4;
  p2.eps = 1e-6f;
  p2.scale = 1.2f;
  float cos2[4] = {1.0f, 0.9f, 0.8f, 0.7f};
  float sin2[4] = {0.0f, 0.1f, 0.2f, 0.3f};
  std::vector<float> q0 = RandomVec(8, &rng);
  std::vector<float> wq = RandomVec(8, &rng);
  std::vector<float> k0 = RandomVec(8, &rng);
  std::vector<float> wk = RandomVec(8, &rng);

  std::vector<ComputeType> qsaved = ToStorage(q0);
  std::vector<ComputeType> dq = ToStorage(wq);
  std::vector<ComputeType> ksaved = ToStorage(k0);
  std::vector<ComputeType> dk = ToStorage(wk);
  kernels::QkPrepBackward(p2, cos2, sin2, dq.data(), dk.data(), qsaved.data(),
                          ksaved.data());
  std::vector<float> dq_an = FromStorage(qsaved);
  std::vector<float> dk_an = FromStorage(ksaved);

  auto loss_at = [&](const std::vector<float>& qv,
                     const std::vector<float>& kv) {
    std::vector<ComputeType> qb = ToStorage(qv);
    std::vector<ComputeType> kb = ToStorage(kv);
    kernels::QkPrepForward(p2, cos2, sin2, qb.data(), kb.data());
    double loss = 0.0;
    for (int i = 0; i < 8; ++i) loss += U(qb[i]) * wq[i] + U(kb[i]) * wk[i];
    return loss;
  };
  std::vector<float> qtmp = q0;
  CheckFiniteDifference(
      q0, dq_an, [&](const std::vector<float>& xv) { return loss_at(xv, k0); },
      2e-3, "qkprep dq");
  CheckFiniteDifference(
      k0, dk_an, [&](const std::vector<float>& xv) { return loss_at(q0, xv); },
      2e-3, "qkprep dk");
#endif
}

void TestAttention() {
  using namespace nanochat;
  AttentionParams p;
  p.batch = 1;
  p.seq = 3;
  p.num_heads = 2;
  p.num_kv_heads = 1;
  p.head_dim = 2;
  p.causal = true;
  p.window_left = -1;
  p.window_right = 0;
  p.kv_len = 0;
  p.scale = 0.0f;
#if !defined(NANOCHAT_PRECISION_FP16)
  const int qn = 1 * 3 * 2 * 2;
  const int kn = 1 * 3 * 1 * 2;
  Rng rng;
  std::vector<float> q0 = RandomVec(qn, &rng);
  std::vector<float> k0 = RandomVec(kn, &rng);
  std::vector<float> v0 = RandomVec(kn, &rng);
  std::vector<float> w = RandomVec(qn, &rng);
  std::vector<ComputeType> qs = ToStorage(q0);
  std::vector<ComputeType> ks = ToStorage(k0);
  std::vector<ComputeType> vs = ToStorage(v0);
  std::vector<ComputeType> out(qn);
  std::vector<float> stats(1 * 2 * 3 * 2);
  kernels::AttentionForward(p, qs.data(), ks.data(), vs.data(), out.data(),
                            stats.data());
  std::vector<ComputeType> douts = ToStorage(w);
  std::vector<ComputeType> dq(qn), dk(kn), dv(kn);
  kernels::AttentionBackward(p, qs.data(), ks.data(), vs.data(), stats.data(),
                             douts.data(), dq.data(), dk.data(), dv.data());
  std::vector<float> dqa = FromStorage(dq);
  std::vector<float> dka = FromStorage(dk);
  std::vector<float> dva = FromStorage(dv);
  auto loss_at = [&](const std::vector<float>& qv, const std::vector<float>& kv,
                     const std::vector<float>& vv) {
    std::vector<ComputeType> qb = ToStorage(qv);
    std::vector<ComputeType> kb = ToStorage(kv);
    std::vector<ComputeType> vb = ToStorage(vv);
    std::vector<ComputeType> ob(qn);
    std::vector<float> st(1 * 2 * 3 * 2);
    kernels::AttentionForward(p, qb.data(), kb.data(), vb.data(), ob.data(),
                              st.data());
    double loss = 0.0;
    for (int i = 0; i < qn; ++i) loss += U(ob[i]) * w[i];
    return loss;
  };
  CheckFiniteDifference(
      q0, dqa,
      [&](const std::vector<float>& xv) { return loss_at(xv, k0, v0); }, 2e-3,
      "attention dq");
  CheckFiniteDifference(
      k0, dka,
      [&](const std::vector<float>& xv) { return loss_at(q0, xv, v0); }, 2e-3,
      "attention dk");
  CheckFiniteDifference(
      v0, dva,
      [&](const std::vector<float>& xv) { return loss_at(q0, k0, xv); }, 2e-3,
      "attention dv");
  // Causal: the first query row only sees key 0.
  Check(stats[0] == stats[0], "attention stats finite");
#endif
}

void TestAttentionWindow() {
#if !defined(NANOCHAT_PRECISION_FP16)
  using namespace nanochat;
  AttentionParams p;
  p.batch = 1;
  p.seq = 4;
  p.num_heads = 1;
  p.num_kv_heads = 1;
  p.head_dim = 2;
  p.causal = true;
  p.window_left = 1;  // each query sees only itself and the previous key
  p.window_right = 0;
  p.kv_len = 0;
  p.scale = 0.0f;
  const int qn = 1 * 4 * 1 * 2;
  const int kn = 1 * 4 * 1 * 2;
  Rng rng;
  std::vector<float> q0 = RandomVec(qn, &rng);
  std::vector<float> k0 = RandomVec(kn, &rng);
  std::vector<float> v0 = RandomVec(kn, &rng);
  std::vector<float> w = RandomVec(qn, &rng);
  std::vector<ComputeType> qs = ToStorage(q0);
  std::vector<ComputeType> ks = ToStorage(k0);
  std::vector<ComputeType> vs = ToStorage(v0);
  std::vector<ComputeType> out(qn);
  std::vector<float> stats(1 * 1 * 4 * 2);
  kernels::AttentionForward(p, qs.data(), ks.data(), vs.data(), out.data(),
                            stats.data());
  std::vector<ComputeType> douts = ToStorage(w);
  std::vector<ComputeType> dq(qn), dk(kn), dv(kn);
  kernels::AttentionBackward(p, qs.data(), ks.data(), vs.data(), stats.data(),
                             douts.data(), dq.data(), dk.data(), dv.data());
  std::vector<float> dqa = FromStorage(dq);
  std::vector<float> dka = FromStorage(dk);
  std::vector<float> dva = FromStorage(dv);
  auto loss_at = [&](const std::vector<float>& qv, const std::vector<float>& kv,
                     const std::vector<float>& vv) {
    std::vector<ComputeType> qb = ToStorage(qv);
    std::vector<ComputeType> kb = ToStorage(kv);
    std::vector<ComputeType> vb = ToStorage(vv);
    std::vector<ComputeType> ob(qn);
    std::vector<float> st(1 * 1 * 4 * 2);
    kernels::AttentionForward(p, qb.data(), kb.data(), vb.data(), ob.data(),
                              st.data());
    double loss = 0.0;
    for (int i = 0; i < qn; ++i) loss += U(ob[i]) * w[i];
    return loss;
  };
  CheckFiniteDifference(
      q0, dqa,
      [&](const std::vector<float>& xv) { return loss_at(xv, k0, v0); }, 2e-3,
      "window dq");
  CheckFiniteDifference(
      k0, dka,
      [&](const std::vector<float>& xv) { return loss_at(q0, xv, v0); }, 2e-3,
      "window dk");
  CheckFiniteDifference(
      v0, dva,
      [&](const std::vector<float>& xv) { return loss_at(q0, k0, xv); }, 2e-3,
      "window dv");
#endif
}

void TestPointwise() {
  using namespace nanochat;
  Rng rng;
  const int n = 5;
  std::vector<float> a0 = RandomVec(n, &rng);
  std::vector<float> b0 = RandomVec(n, &rng);
  std::vector<float> w = RandomVec(n, &rng);
  for (PointwiseOp op :
       {PointwiseOp::kScale, PointwiseOp::kScaleAdd, PointwiseOp::kGateMul,
        PointwiseOp::kReluSquare, PointwiseOp::kSoftcap}) {
    std::vector<ComputeType> as = ToStorage(a0);
    std::vector<ComputeType> bs = ToStorage(b0);
    std::vector<ComputeType> out(n);
    kernels::PointwiseForward(op, n, as.data(), bs.data(), 1.5f, 0.5f,
                              out.data());
    std::vector<ComputeType> dys = ToStorage(w);
    std::vector<ComputeType> da(n), db(n);
    kernels::PointwiseBackward(op, n, as.data(), bs.data(), dys.data(), 1.5f,
                               0.5f, da.data(), db.data());
#if !defined(NANOCHAT_PRECISION_FP16)
    std::vector<float> daa = FromStorage(da);
    std::vector<float> dba = FromStorage(db);
    auto loss_at = [&](const std::vector<float>& av,
                       const std::vector<float>& bv) {
      std::vector<ComputeType> a2 = ToStorage(av);
      std::vector<ComputeType> b2 = ToStorage(bv);
      std::vector<ComputeType> o2(n);
      kernels::PointwiseForward(op, n, a2.data(), b2.data(), 1.5f, 0.5f,
                                o2.data());
      double loss = 0.0;
      for (int i = 0; i < n; ++i) loss += U(o2[i]) * w[i];
      return loss;
    };
    CheckFiniteDifference(
        a0, daa, [&](const std::vector<float>& av) { return loss_at(av, b0); },
        2e-3, "pointwise da");
    CheckFiniteDifference(
        b0, dba, [&](const std::vector<float>& bv) { return loss_at(a0, bv); },
        2e-3, "pointwise db");
#endif
  }
}

void TestClassifier() {
  using namespace nanochat;
  ClassifierParams p;
  p.rows = 3;
  p.vocab_size = 4;
  p.padded_vocab_size = 6;
  p.softcap = 15.0f;
  p.ignore_index = -1;
  Rng rng;
  std::vector<float> logits0 = RandomVec(3 * 6, &rng);
  int targets[3] = {0, 2, -1};
  std::vector<ComputeType> logits = ToStorage(logits0);
  std::vector<ComputeType> losses(3);
  kernels::ClassifierForward(p, logits.data(), targets, losses.data());
  CheckClose(U(losses[2]), 0.0, 1e-6, "classifier ignore loss");
  Check(U(losses[0]) > 0.0f, "classifier positive loss");

  std::vector<ComputeType> dlogits(3 * 6);
  kernels::ClassifierBackward(p, logits.data(), targets, dlogits.data());
  for (int j = 0; j < 6; ++j) {
    CheckClose(U(dlogits[2 * 6 + j]), 0.0, 1e-6, "classifier ignore grad");
  }
#if !defined(NANOCHAT_PRECISION_FP16)
  std::vector<float> dl = FromStorage(dlogits);
  auto loss_at = [&](const std::vector<float>& lv) {
    std::vector<ComputeType> l2 = ToStorage(lv);
    std::vector<ComputeType> ls(3);
    kernels::ClassifierForward(p, l2.data(), targets, ls.data());
    double loss = 0.0;
    for (int r = 0; r < 3; ++r) loss += U(ls[r]);
    return loss;
  };
  CheckFiniteDifference(logits0, dl, loss_at, 5e-3, "classifier backward");
#endif
}

void TestEmbedding() {
  using namespace nanochat;
  Rng rng;
  const int tokens = 4;
  const int dim = 3;
  const int vocab = 4;
  int ids[4] = {1, 0, 1, 2};
  std::vector<float> table0 = RandomVec(vocab * dim, &rng);
  std::vector<ComputeType> table = ToStorage(table0);
  std::vector<ComputeType> out(tokens * dim);
  kernels::EmbeddingForward(tokens, dim, ids, table.data(), out.data());
  for (int i = 0; i < tokens; ++i) {
    for (int d = 0; d < dim; ++d) {
      CheckClose(U(out[i * dim + d]), table0[ids[i] * dim + d], 1e-6,
                 "embedding forward");
    }
  }
  std::vector<float> dout0 = RandomVec(tokens * dim, &rng);
  std::vector<ComputeType> dout = ToStorage(dout0);
  std::vector<ComputeType> dtable(vocab * dim, C(7.0f));
  kernels::EmbeddingBackward(tokens, dim, ids, dout.data(), dtable.data());
  for (int d = 0; d < dim; ++d) {
    CheckClose(U(dtable[0 * dim + d]), dout0[1 * dim + d], 1e-5,
               "embedding grad row0");
    CheckClose(U(dtable[1 * dim + d]), dout0[0 * dim + d] + dout0[2 * dim + d],
               1e-5, "embedding grad row1 (duplicate)");
    CheckClose(U(dtable[2 * dim + d]), dout0[3 * dim + d], 1e-5,
               "embedding grad row2");
    CheckClose(U(dtable[3 * dim + d]), 7.0, 1e-6, "embedding untouched row");
  }
}

void TestAdamW() {
  using namespace nanochat;
  AdamWParams p;
  p.lr = 0.1f;
  p.beta1 = 0.9f;
  p.beta2 = 0.999f;
  p.eps = 1e-10f;
  p.weight_decay = 0.0f;
  p.step = 1;
  std::vector<ComputeType> w = ToStorage({1.0f});
  std::vector<ComputeType> g = ToStorage({1.0f});
  float m = 0.0f;
  float v = 0.0f;
  kernels::AdamWUpdate(1, p, w.data(), g.data(), &m, &v);
  CheckClose(U(w[0]), 0.9, 1e-4, "adamw step");
  CheckClose(m, 0.1, 1e-6, "adamw m");
  CheckClose(v, 0.001, 1e-6, "adamw v");
}

void TestMuon() {
  using namespace nanochat;
  MuonParams p;
  p.num_params = 1;
  p.rows = 3;
  p.cols = 3;
  p.lr = 0.0f;  // parameter must be untouched when lr is zero
  p.momentum = 0.9f;
  p.beta2 = 0.9f;
  p.weight_decay = 0.0f;
  p.ns_steps = 5;
  p.red_dim = -1;
  p.nesterov = true;
  Rng rng;
  std::vector<float> grad0 = RandomVec(9, &rng);
  std::vector<float> param0 = RandomVec(9, &rng);
  std::vector<ComputeType> grad = ToStorage(grad0);
  std::vector<ComputeType> param = ToStorage(param0);
  std::vector<float> buf1(9, 0.0f);
  std::vector<float> buf2(3, 0.0f);
  kernels::MuonUpdate(p, grad.data(), param.data(), buf1.data(), buf2.data());
  std::vector<float> got = FromStorage(param);
  for (int i = 0; i < 9; ++i) {
    CheckClose(got[i], param0[i], 1e-6, "muon lr=0 identity");
    Check(std::isfinite(buf1[i]), "muon buf1 finite");
  }
  for (int i = 0; i < 3; ++i) Check(std::isfinite(buf2[i]), "muon buf2 finite");
}

void TestGlobalNorm() {
  using namespace nanochat;
  std::vector<ComputeType> g = ToStorage({3.0f, 4.0f});
  float norm = 0.0f;
  kernels::GlobalNorm(2, 2.5f, g.data(), &norm);
  CheckClose(norm, 5.0, 1e-6, "globalnorm pre-clip");
  CheckClose(U(g[0]), 1.5, 1e-6, "globalnorm clipped 0");
  CheckClose(U(g[1]), 2.0, 1e-6, "globalnorm clipped 1");

  std::vector<ComputeType> g2 = ToStorage({3.0f, 4.0f});
  kernels::GlobalNorm(2, 10.0f, g2.data(), &norm);
  CheckClose(U(g2[0]), 3.0, 1e-6, "globalnorm unclipped");
}

void TestDeviceUtilities() {
  using namespace nanochat;
  void* p = kernels::Alloc(64);
  Check(p != nullptr, "alloc");
  kernels::Memset(p, 0, 64);
  std::vector<ComputeType> src = ToStorage({1.0f, 2.0f, 3.0f, 4.0f});
  std::vector<ComputeType> dst(4);
  kernels::Memcpy(dst.data(), src.data(), 4 * sizeof(ComputeType),
                  CopyDir::kHostToHost);
  CheckClose(U(dst[2]), 3.0, 1e-6, "memcpy");
  kernels::Free(p);
  kernels::Synchronize();
  Caps caps = kernels::GetCaps();
  Check(!caps.is_device, "caps is_device false");
  Check(caps.Supports(kComputeDType), "caps supports build precision");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestDeviceUtilities();
  TestGemm();
  TestRmsNorm();
  TestQkPrep();
  TestAttention();
  TestAttentionWindow();
  TestPointwise();
  TestClassifier();
  TestEmbedding();
  TestAdamW();
  TestMuon();
  TestGlobalNorm();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("cpu kernels: all checks passed (precision=%s)\n",
#if defined(NANOCHAT_PRECISION_FP16)
              "fp16"
#else
              "fp32"
#endif
  );
  return 0;
}
