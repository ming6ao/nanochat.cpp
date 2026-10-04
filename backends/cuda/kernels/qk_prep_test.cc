// T1 GPU correctness + finite-difference test for the QkPrep family (fused
// RMSNorm -> RoPE -> scale on q and k). Tiny attention shape: B=2, T=8,
// head_dim=4, num_heads=2, plus a grouped-query variant. See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/row_ref.h"

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
using namespace nanochat::dev::rowref;

void RunCase(const QkPrepParams& p) {
  const int half = p.head_dim / 2;
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;

  Rng rng;
  std::vector<float> cos(static_cast<std::size_t>(p.seq) * half, 0.0f);
  std::vector<float> sin(static_cast<std::size_t>(p.seq) * half, 0.0f);
  for (int t = 0; t < p.seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float theta =
          0.1f * static_cast<float>(t + 1) * static_cast<float>(d + 1);
      cos[static_cast<std::size_t>(t) * half + d] = std::cos(theta);
      sin[static_cast<std::size_t>(t) * half + d] = std::sin(theta);
    }
  }

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
  CheckVectorClose(FromStorage(q.Download()), ref_q, 1e-4, "qk fwd q");
  CheckVectorClose(FromStorage(k.Download()), ref_k, 1e-4, "qk fwd k");

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
  CheckVectorClose(got_dq, ref_dq, 1e-4, "qk bwd dq");
  CheckVectorClose(got_dk, ref_dk, 1e-4, "qk bwd dk");

#if !defined(NANOCHAT_PRECISION_FP16)
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
      q0, got_dq, [&](const std::vector<float>& qv) { return loss_at(qv, k0); },
      2e-3, "qk dq");
  CheckFiniteDifference(
      k0, got_dk, [&](const std::vector<float>& kv) { return loss_at(q0, kv); },
      2e-3, "qk dk");
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("qk_prep gpu test\n");

  QkPrepParams p;
  p.batch = 2;
  p.seq = 8;
  p.num_heads = 2;
  p.num_kv_heads = 2;
  p.head_dim = 4;
  p.eps = 1e-6f;
  p.scale = 1.2f;
  RunCase(p);

  // Grouped-query variant: two query heads share one key/value head.
  QkPrepParams gqa = p;
  gqa.num_kv_heads = 1;
  RunCase(gqa);

  if (Failures() != 0) {
    std::printf("qk_prep: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("qk_prep: all checks passed\n");
  return 0;
}
