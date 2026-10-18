// tests/gemm_threads_test.cc -- the CPU GEMM thread-independence guard.
//
// `GemmImpl` parallelizes over output row tiles. Each output is one accumulator
// over a fixed reduction order, and the packed `b` panel is shared read-only.
// So the result is identical at any thread count. The numeric trace relies on
// that: it pins the kernel reductions to one thread for a stable golden, and it
// lets the GEMM use the CPUs the sandbox granted, through
// `NANOHAT_GEMM_THREADS`.
//
// This test pins the property. It runs the forward, input-gradient, and
// weight-gradient GEMMs at one thread and at two, and compares the bytes.
//
// CPU only: tier S0. On a build without OpenMP the guard holds trivially.
// docs/numerics-integration.md section 6.2 owns the trace.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::GemmMode;
using nanochat::GemmParams;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

// A deterministic input that is not uniform, so a wrong reduction shows.
std::vector<ComputeType> MakeInput(std::size_t count, int seed) {
  std::vector<ComputeType> out(count);
  std::uint32_t state = static_cast<std::uint32_t>(seed) * 2654435761u + 1u;
  for (std::size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit =
        static_cast<float>((state >> 8) & 0xffffu) / 65535.0f - 0.5f;
    out[i] = nanochat::ComputeFromFloat(unit * 2.0f);
  }
  return out;
}

void SetThreads(int threads) {
  char buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%d", threads);
  setenv("NANOCHAT_GEMM_THREADS", buffer, 1);
}

// The compute-type GEMM, forward or input gradient, as raw output bytes.
std::vector<std::uint8_t> RunCompute(GemmMode mode, const GemmParams& params,
                                     const std::vector<ComputeType>& a,
                                     const std::vector<ComputeType>& b,
                                     int threads) {
  SetThreads(threads);
  std::vector<ComputeType> c(static_cast<std::size_t>(params.m) * params.n,
                             nanochat::ComputeFromFloat(0.0f));
  nanochat::kernels::Gemm(mode, params, a.data(), b.data(), c.data());
  std::vector<std::uint8_t> bytes(c.size() * sizeof(ComputeType));
  std::memcpy(bytes.data(), c.data(), bytes.size());
  return bytes;
}

// The weight-gradient GEMM, which writes fp32.
std::vector<std::uint8_t> RunWgrad(const GemmParams& params,
                                   const std::vector<ComputeType>& a,
                                   const std::vector<ComputeType>& b,
                                   int threads) {
  SetThreads(threads);
  std::vector<float> c(static_cast<std::size_t>(params.m) * params.n, 0.0f);
  nanochat::kernels::GemmWgrad(params, a.data(), b.data(), c.data());
  std::vector<std::uint8_t> bytes(c.size() * sizeof(float));
  std::memcpy(bytes.data(), c.data(), bytes.size());
  return bytes;
}

void CheckEqual(const std::vector<std::uint8_t>& one,
                const std::vector<std::uint8_t>& two, const std::string& what) {
  if (one.size() != two.size()) {
    Fail(what + ": the sizes differ");
    return;
  }
  if (std::memcmp(one.data(), two.data(), one.size()) != 0) {
    Fail(what + ": the one-thread and two-thread bytes differ");
  }
}

// The operand sizes follow the stored shapes: A is [m, k] when `transpose_a` is
// false and [k, m] when it is true, so `m * k` elements; B is [k, n] or [n, k],
// so `k * n` elements.
void CheckFamily(const std::string& what, GemmMode mode, int m, int n, int k,
                 bool transpose_a, bool transpose_b, bool wgrad) {
  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = 1.0f;
  params.beta = wgrad ? 1.0f : 0.0f;
  params.transpose_a = transpose_a;
  params.transpose_b = transpose_b;

  const std::vector<ComputeType> a =
      MakeInput(static_cast<std::size_t>(m) * k, 3);
  const std::vector<ComputeType> b =
      MakeInput(static_cast<std::size_t>(k) * n, 5);

  // Ask for one thread and for two, so the guard does not depend on the host's
  // CPU count.
  if (wgrad) {
    CheckEqual(RunWgrad(params, a, b, 1), RunWgrad(params, a, b, 2), what);
    return;
  }
  CheckEqual(RunCompute(mode, params, a, b, 1),
             RunCompute(mode, params, a, b, 2), what);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  // The forward role: A is [m, k], B is the stored weight [n, k].
  CheckFamily("forward", GemmMode::kForward, 64, 64, 64,
              /*transpose_a=*/false, /*transpose_b=*/true, /*wgrad=*/false);
  // The input-gradient role: A is [m, k], B is the weight [k, n].
  CheckFamily("dgrad", GemmMode::kDgrad, 64, 48, 32, /*transpose_a=*/false,
              /*transpose_b=*/false, /*wgrad=*/false);
  // The weight-gradient role: A is the stored [k, m], B is [k, n].
  CheckFamily("wgrad", GemmMode::kWgrad, 48, 32, 64, /*transpose_a=*/true,
              /*transpose_b=*/false, /*wgrad=*/true);
  // A row count that leaves a partial tile, so the tail path is covered.
  CheckFamily("forward partial tile", GemmMode::kForward, 70, 64, 64, false,
              true, false);
  CheckFamily("dgrad partial tile", GemmMode::kDgrad, 70, 48, 32, false, false,
              false);

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("gemm threads: all checks passed\n");
  return 0;
}
