// T1 GPU correctness + finite-difference test for the dev/kernels QkPrep fusion
// prototype. It checks the combined fused kernel and the explicit decomposed
// path against the host reference in row_ref.h, cross-checks the prototype
// against the seam `kernels::QkPrepForward`/`Backward`, and finite-differences
// the fused backward. Tiny attention shape: B=2, T=8, head_dim=4, num_heads=2,
// plus a grouped-query variant. See dev/kernels/README.md and DESIGN.md
// section 3.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/row_ref.h"
#include "dev/kernels/qk_prep_fused.h"
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

void MakeRopeTables(const QkPrepParams& p, std::vector<float>* cos,
                    std::vector<float>* sin) {
  const int half = p.head_dim / 2;
  cos->assign(static_cast<std::size_t>(p.seq) * half, 0.0f);
  sin->assign(static_cast<std::size_t>(p.seq) * half, 0.0f);
  for (int t = 0; t < p.seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float theta =
          0.1f * static_cast<float>(t + 1) * static_cast<float>(d + 1);
      (*cos)[static_cast<std::size_t>(t) * half + d] = std::cos(theta);
      (*sin)[static_cast<std::size_t>(t) * half + d] = std::sin(theta);
    }
  }
}

void RunCase(const QkPrepParams& p, const char* name) {
  const int half = p.head_dim / 2;
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;

  std::vector<float> cos;
  std::vector<float> sin;
  MakeRopeTables(p, &cos, &sin);

  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(k_count, &rng);
  const std::vector<float> wq = RandomVec(q_count, &rng);
  const std::vector<float> wk = RandomVec(k_count, &rng);

  DevBuf<float> dcos(cos);
  DevBuf<float> dsin(sin);

  // Reference activations and gradients.
  std::vector<float> ref_q = q0;
  std::vector<float> ref_k = k0;
  QkPrepForward(p, cos, sin, &ref_q, &ref_k);
  std::vector<float> ref_dq = q0;
  std::vector<float> ref_dk = k0;
  QkPrepBackward(p, cos, sin, wq, wk, &ref_dq, &ref_dk);

  // --- Prototype combined fused forward --------------------------------
  DevBuf<ComputeType> fq(ToStorage(q0));
  DevBuf<ComputeType> fk(ToStorage(k0));
  nanochat::dev::QkPrepFusedForward(p, dcos.ptr, dsin.ptr, fq.ptr, fk.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(fq.Download()), ref_q, 1e-4,
                   (std::string(name) + " fused fwd q").c_str());
  CheckVectorClose(FromStorage(fk.Download()), ref_k, 1e-4,
                   (std::string(name) + " fused fwd k").c_str());

  // --- Prototype vs the seam fused kernel ------------------------------
  DevBuf<ComputeType> sq(ToStorage(q0));
  DevBuf<ComputeType> sk(ToStorage(k0));
  nanochat::kernels::QkPrepForward(p, dcos.ptr, dsin.ptr, sq.ptr, sk.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(fq.Download()), FromStorage(sq.Download()), 1e-4,
                   (std::string(name) + " fused==seam q").c_str());
  CheckVectorClose(FromStorage(fk.Download()), FromStorage(sk.Download()), 1e-4,
                   (std::string(name) + " fused==seam k").c_str());

  // --- Decomposed forward ----------------------------------------------
  DevBuf<ComputeType> q_in(ToStorage(q0));
  DevBuf<ComputeType> k_in(ToStorage(k0));
  DevBuf<ComputeType> q_normed(q_count);
  DevBuf<ComputeType> k_normed(k_count);
  DevBuf<ComputeType> q_out(q_count);
  DevBuf<ComputeType> k_out(k_count);
  DevBuf<float> q_rstd(p.batch * p.seq * p.num_heads);
  DevBuf<float> k_rstd(p.batch * p.seq * p.num_kv_heads);
  nanochat::dev::QkPrepDecomposedForward(
      p, dcos.ptr, dsin.ptr, q_in.ptr, k_in.ptr, q_normed.ptr, k_normed.ptr,
      q_out.ptr, k_out.ptr, q_rstd.ptr, k_rstd.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(q_out.Download()), ref_q, 1e-4,
                   (std::string(name) + " decomp fwd q").c_str());
  CheckVectorClose(FromStorage(k_out.Download()), ref_k, 1e-4,
                   (std::string(name) + " decomp fwd k").c_str());

  // --- Prototype combined fused backward -------------------------------
  DevBuf<ComputeType> bq(ToStorage(q0));
  DevBuf<ComputeType> bk(ToStorage(k0));
  DevBuf<ComputeType> ddq(ToStorage(wq));
  DevBuf<ComputeType> ddk(ToStorage(wk));
  nanochat::dev::QkPrepFusedBackward(p, dcos.ptr, dsin.ptr, ddq.ptr, ddk.ptr,
                                     bq.ptr, bk.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(bq.Download()), ref_dq, 1e-4,
                   (std::string(name) + " fused bwd dq").c_str());
  CheckVectorClose(FromStorage(bk.Download()), ref_dk, 1e-4,
                   (std::string(name) + " fused bwd dk").c_str());

  // --- Decomposed backward ---------------------------------------------
  DevBuf<ComputeType> dq_normed(q_count);
  DevBuf<ComputeType> dk_normed(k_count);
  DevBuf<ComputeType> q_grad(q_count);
  DevBuf<ComputeType> k_grad(k_count);
  nanochat::dev::QkPrepDecomposedBackward(
      p, dcos.ptr, dsin.ptr, ddq.ptr, ddk.ptr, q_in.ptr, k_in.ptr, q_rstd.ptr,
      k_rstd.ptr, dq_normed.ptr, dk_normed.ptr, q_grad.ptr, k_grad.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(q_grad.Download()), ref_dq, 1e-4,
                   (std::string(name) + " decomp bwd dq").c_str());
  CheckVectorClose(FromStorage(k_grad.Download()), ref_dk, 1e-4,
                   (std::string(name) + " decomp bwd dk").c_str());

#if !defined(NANOCHAT_PRECISION_FP16)
  // Finite-difference the prototype fused backward against the reference
  // forward: d/dq of sum(ref_q(q) * wq) + sum(ref_k * wk). Recompute the
  // reference forward in fp32 so the perturbation resolves.
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
      q0, FromStorage(bq.Download()),
      [&](const std::vector<float>& qv) { return loss_at(qv, k0); }, 2e-3,
      (std::string(name) + " fused fd dq").c_str());
  CheckFiniteDifference(
      k0, FromStorage(bk.Download()),
      [&](const std::vector<float>& kv) { return loss_at(q0, kv); }, 2e-3,
      (std::string(name) + " fused fd dk").c_str());
  (void)half;
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("qk_prep fused prototype gpu test\n");

  QkPrepParams p;
  p.batch = 2;
  p.seq = 8;
  p.num_heads = 2;
  p.num_kv_heads = 2;
  p.head_dim = 4;
  p.eps = 1e-6f;
  p.scale = 1.2f;
  RunCase(p, "mha");

  // Grouped-query variant: two query heads share one key/value head.
  QkPrepParams gqa = p;
  gqa.num_kv_heads = 1;
  RunCase(gqa, "gqa");

  if (Failures() != 0) {
    std::printf("qk_prep_fused: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("qk_prep_fused: all checks passed\n");
  return 0;
}
