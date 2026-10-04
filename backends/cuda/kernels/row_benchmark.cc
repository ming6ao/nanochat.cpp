// Standalone micro-benchmark for the row/elementwise families, llm.c dev/cuda
// style. It mirrors the family tests' shapes: a tiny attention shape and
// a medium training shape. Wall-clock timing with a synchronize per batch;
// this is a smoke benchmark, not a tuned T3 profile. Run it through the broker:
//
//   tools/nanochat bench -- <row_benchmark binary>
//
// See backends/cuda/kernels/README.md and docs/testing.md.

#include <chrono>
#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::dev::DevBuf;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

template <typename Fn>
double TimeIt(const char* name, int iters, Fn&& fn) {
  // Warm up, then time a batch and synchronize once.
  fn();
  nanochat::kernels::Synchronize();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  nanochat::kernels::Synchronize();
  const auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - start).count() / iters;
  std::printf("  %-28s %10.3f ms/iter\n", name, ms);
  return ms;
}

void BenchRmsNorm(Rng* rng, int rows, int dim, int iters) {
  nanochat::RmsNormParams p;
  p.rows = rows;
  p.dim = dim;
  p.eps = 1e-6f;
  DevBuf<ComputeType> x(ToStorage(RandomVec(rows * dim, rng)));
  DevBuf<ComputeType> out(rows * dim);
  DevBuf<float> rstd(rows);
  char label[64];
  std::snprintf(label, sizeof(label), "rmsnorm r=%d d=%d", rows, dim);
  TimeIt(label, iters, [&] {
    nanochat::kernels::RmsNormForward(p, x.ptr, out.ptr, rstd.ptr);
  });
}

void BenchQkPrep(Rng* rng, int batch, int seq, int heads, int dim, int iters) {
  nanochat::QkPrepParams p;
  p.batch = batch;
  p.seq = seq;
  p.num_heads = heads;
  p.num_kv_heads = heads;
  p.head_dim = dim;
  p.eps = 1e-6f;
  p.scale = 1.2f;
  const int half = dim / 2;
  DevBuf<float> cos(
      std::vector<float>(static_cast<std::size_t>(seq) * half, 0.5f));
  DevBuf<float> sin(
      std::vector<float>(static_cast<std::size_t>(seq) * half, 0.5f));
  const int n = batch * seq * heads * dim;
  DevBuf<ComputeType> q(ToStorage(RandomVec(n, rng)));
  DevBuf<ComputeType> k(ToStorage(RandomVec(n, rng)));
  char label[64];
  std::snprintf(label, sizeof(label), "qkprep b=%d t=%d h=%d d=%d", batch, seq,
                heads, dim);
  TimeIt(label, iters, [&] {
    nanochat::kernels::QkPrepForward(p, cos.ptr, sin.ptr, q.ptr, k.ptr);
  });
}

void BenchPointwise(Rng* rng, int n, int iters) {
  DevBuf<ComputeType> a(ToStorage(RandomVec(n, rng)));
  DevBuf<ComputeType> b(ToStorage(RandomVec(n, rng)));
  DevBuf<ComputeType> out(n);
  char label[64];
  std::snprintf(label, sizeof(label), "pointwise n=%d", n);
  TimeIt(label, iters, [&] {
    nanochat::kernels::PointwiseForward(nanochat::PointwiseOp::kScaleAdd, n,
                                        a.ptr, b.ptr, 1.0f, 1.0f, out.ptr);
  });
}

void BenchGlobalNorm(Rng* rng, int n, int iters) {
  DevBuf<ComputeType> g(ToStorage(RandomVec(n, rng)));
  DevBuf<float> norm(1);
  char label[64];
  std::snprintf(label, sizeof(label), "globalnorm n=%d", n);
  // A tiny clip keeps the scale pass exercised every iteration.
  TimeIt(label, iters,
         [&] { nanochat::kernels::GlobalNorm(n, 1e-3f, g.ptr, norm.ptr); });
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  Rng rng;
  std::printf("row kernel benchmark\n");

  BenchRmsNorm(&rng, 16, 4, 200);      // tiny attention shape
  BenchRmsNorm(&rng, 4096, 768, 50);   // medium training shape
  BenchQkPrep(&rng, 2, 8, 2, 4, 200);  // tiny attention shape
  BenchQkPrep(&rng, 8, 512, 8, 64, 20);
  BenchPointwise(&rng, 1 << 16, 200);
  BenchPointwise(&rng, 1 << 20, 50);
  BenchGlobalNorm(&rng, 1 << 16, 200);
  BenchGlobalNorm(&rng, 1 << 20, 50);
  return 0;
}
