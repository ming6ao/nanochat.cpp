// T1 GPU correctness + finite-difference test for the Attention family:
// causal / sliding-window / grouped-query attention, forward and backward,
// with the saved softmax statistics. Tiny attention shapes only
// (B=2, T=8, head_dim=4, num_heads=2). See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"

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
using namespace nanochat::dev::seqref;

void RunCase(const AttentionParams& p, const char* name) {
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

  DevBuf<ComputeType> q(ToStorage(q0));
  DevBuf<ComputeType> k(ToStorage(k0));
  DevBuf<ComputeType> v(ToStorage(v0));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(stats_count);
  nanochat::kernels::AttentionForward(p, q.ptr, k.ptr, v.ptr, out.ptr,
                                      stats.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_stats;
  const std::vector<float> ref_out =
      AttentionForward(p, q0, k0, v0, &ref_stats);
  CheckVectorClose(FromStorage(out.Download()), ref_out, 1e-4,
                   (std::string(name) + " fwd out").c_str());
  CheckVectorClose(stats.Download(), ref_stats, 1e-4,
                   (std::string(name) + " fwd stats").c_str());

  DevBuf<ComputeType> dout(ToStorage(w));
  DevBuf<ComputeType> dq(q_count);
  DevBuf<ComputeType> dk(kv_count);
  DevBuf<ComputeType> dv(kv_count);
  nanochat::kernels::AttentionBackward(p, q.ptr, k.ptr, v.ptr, stats.ptr,
                                       dout.ptr, dq.ptr, dk.ptr, dv.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_dq;
  std::vector<float> ref_dk;
  std::vector<float> ref_dv;
  AttentionBackward(p, q0, k0, v0, ref_stats, w, &ref_dq, &ref_dk, &ref_dv);
  CheckVectorClose(FromStorage(dq.Download()), ref_dq, 1e-4,
                   (std::string(name) + " bwd dq").c_str());
  CheckVectorClose(FromStorage(dk.Download()), ref_dk, 1e-4,
                   (std::string(name) + " bwd dk").c_str());
  CheckVectorClose(FromStorage(dv.Download()), ref_dv, 1e-4,
                   (std::string(name) + " bwd dv").c_str());

#if !defined(NANOCHAT_PRECISION_FP16)
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
#endif
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

  AttentionParams gqa = base;
  gqa.num_kv_heads = 1;
  RunCase(gqa, "gqa-causal");

  AttentionParams window = base;
  window.window_left = 3;
  RunCase(window, "sliding-window");

  AttentionParams noncausal = base;
  noncausal.causal = false;
  noncausal.window_right = -1;
  RunCase(noncausal, "noncausal");

  AttentionParams decode = base;
  decode.kv_len = 10;
  RunCase(decode, "kv-cache-offset");

  if (Failures() != 0) {
    std::printf("attention: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("attention: all checks passed\n");
  return 0;
}
