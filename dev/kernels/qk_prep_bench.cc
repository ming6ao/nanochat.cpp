// Micro-benchmark for the QkPrep fusion boundary: the decomposed path
// (RmsNormForward + standalone RoPE/scale) against the seam fused kernel and
// the prototype combined fused kernel. Tiny and medium shapes; wall-clock with
// one synchronize per batch. Run it through the broker:
//
//   tools/nanochat bench -- <qk_prep_bench binary>
//
// See dev/kernels/README.md and DESIGN.md section 3.

#include <chrono>
#include <cstdio>
#include <vector>

#include "dev/kernels/qk_prep_fused.h"
#include "gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::QkPrepParams;
using nanochat::dev::DevBuf;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

template <typename Fn>
double TimeIt(const char* name, int iters, Fn&& fn) {
  fn();
  nanochat::kernels::Synchronize();
  // Best of a few rounds: microsecond kernels are sensitive to host noise.
  double best = 1e30;
  for (int round = 0; round < 3; ++round) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    nanochat::kernels::Synchronize();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double ms = elapsed / iters;
    if (ms < best) best = ms;
  }
  std::printf("    %-34s %10.4f ms/iter\n", name, best);
  return best;
}

void BenchShape(const QkPrepParams& p, int iters) {
  const int half = p.head_dim / 2;
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;
  const int q_rows = p.batch * p.seq * p.num_heads;
  const int k_rows = p.batch * p.seq * p.num_kv_heads;

  Rng rng;
  std::vector<float> cos(static_cast<std::size_t>(p.seq) * half, 0.5f);
  std::vector<float> sin(static_cast<std::size_t>(p.seq) * half, 0.5f);
  DevBuf<float> dcos(cos);
  DevBuf<float> dsin(sin);
  DevBuf<ComputeType> q(ToStorage(RandomVec(q_count, &rng)));
  DevBuf<ComputeType> k(ToStorage(RandomVec(k_count, &rng)));

  std::printf("  shape b=%d t=%d h=%d kvh=%d d=%d\n", p.batch, p.seq,
              p.num_heads, p.num_kv_heads, p.head_dim);

  // Forward: decomposed vs seam fused vs prototype combined fused.
  DevBuf<ComputeType> q_normed(q_count);
  DevBuf<ComputeType> k_normed(k_count);
  DevBuf<ComputeType> q_out(q_count);
  DevBuf<ComputeType> k_out(k_count);
  DevBuf<float> q_rstd(q_rows);
  DevBuf<float> k_rstd(k_rows);
  const double decomp_fwd = TimeIt("fwd decomposed", iters, [&] {
    nanochat::dev::QkPrepDecomposedForward(
        p, dcos.ptr, dsin.ptr, q.ptr, k.ptr, q_normed.ptr, k_normed.ptr,
        q_out.ptr, k_out.ptr, q_rstd.ptr, k_rstd.ptr);
  });
  const double seam_fwd = TimeIt("fwd seam fused", iters, [&] {
    nanochat::kernels::QkPrepForward(p, dcos.ptr, dsin.ptr, q.ptr, k.ptr);
  });
  const double proto_fwd = TimeIt("fwd prototype combined fused", iters, [&] {
    nanochat::dev::QkPrepFusedForward(p, dcos.ptr, dsin.ptr, q.ptr, k.ptr);
  });

  // Backward: same three variants.
  DevBuf<ComputeType> dq(ToStorage(RandomVec(q_count, &rng)));
  DevBuf<ComputeType> dk(ToStorage(RandomVec(k_count, &rng)));
  DevBuf<ComputeType> q_pre(ToStorage(RandomVec(q_count, &rng)));
  DevBuf<ComputeType> k_pre(ToStorage(RandomVec(k_count, &rng)));
  DevBuf<ComputeType> scratch_q(q_count);
  DevBuf<ComputeType> scratch_k(k_count);
  DevBuf<ComputeType> grad_q(q_count);
  DevBuf<ComputeType> grad_k(k_count);
  const double decomp_bwd = TimeIt("bwd decomposed", iters, [&] {
    nanochat::dev::QkPrepDecomposedBackward(
        p, dcos.ptr, dsin.ptr, dq.ptr, dk.ptr, q_pre.ptr, k_pre.ptr,
        q_rstd.ptr, k_rstd.ptr, scratch_q.ptr, scratch_k.ptr, grad_q.ptr,
        grad_k.ptr);
  });
  const double seam_bwd = TimeIt("bwd seam fused", iters, [&] {
    nanochat::kernels::QkPrepBackward(p, dcos.ptr, dsin.ptr, dq.ptr, dk.ptr,
                                      q_pre.ptr, k_pre.ptr);
  });
  const double proto_bwd = TimeIt("bwd prototype combined fused", iters, [&] {
    nanochat::dev::QkPrepFusedBackward(p, dcos.ptr, dsin.ptr, dq.ptr, dk.ptr,
                                       q_pre.ptr, k_pre.ptr);
  });

  std::printf("    fwd speedup decomp/proto %.2fx  decomp/seam %.2fx\n",
              decomp_fwd / proto_fwd, decomp_fwd / seam_fwd);
  std::printf("    bwd speedup decomp/proto %.2fx  decomp/seam %.2fx\n",
              decomp_bwd / proto_bwd, decomp_bwd / seam_bwd);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf("qk_prep fusion benchmark\n");

  QkPrepParams tiny;
  tiny.batch = 2;
  tiny.seq = 8;
  tiny.num_heads = 2;
  tiny.num_kv_heads = 2;
  tiny.head_dim = 4;
  tiny.eps = 1e-6f;
  tiny.scale = 1.2f;
  BenchShape(tiny, 200);

  QkPrepParams medium;
  medium.batch = 8;
  medium.seq = 512;
  medium.num_heads = 8;
  medium.num_kv_heads = 8;
  medium.head_dim = 64;
  medium.eps = 1e-6f;
  medium.scale = 1.2f;
  BenchShape(medium, 100);

  return 0;
}
