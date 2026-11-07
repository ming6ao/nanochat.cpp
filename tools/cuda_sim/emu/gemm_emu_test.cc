// The acceptance check for the hand-written tensor-core GEMM
// (docs/tensor-core-gemm-plan.md sections A.4 and B.2).
//
// It walks C = alpha * op(A) * op(B) + beta * C over the four operand layouts,
// a ragged shape with a k tail, a batch of two, and the fp32 store of the
// weight-gradient form. Every case runs the real kernel
// (`backends/cuda/gemm/gemm_bf16.cu`) through the emulation engine, and
// compares against a host reference.
//
// What this checks: the staged layout and its padding, the four stride pairs,
// the sub-tile addressing, the accumulation, the epilogue's alpha, beta, and
// bounds, the store type, and the batch base arithmetic. What it cannot check:
// the fragment ownership layout, a bank conflict, the register count, the
// occupancy, and the rate. Those need a card.
//
// CPU only. The kernel reads host pointers, because the emulator runs every
// fiber on the launching host thread.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "backends/cuda/gemm/gemm_launch.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"
#include "tools/cuda_sim/emu/emu_kernels.h"

namespace {

using nanochat::ComputeType;
using nanochat::GemmParams;
using nanochat::kernels::EmuLaunchGemmTile;
using nanochat::kernels::EmuLaunchGemmTileToFp32;

int g_failures = 0;

// A bf16 store rounds once, so a relative half-unit in the last place is the
// whole budget. An fp32 store keeps the accumulator, so only the accumulation
// order differs.
constexpr float kBf16Relative = 1e-2f;
constexpr float kFp32Relative = 1e-5f;
constexpr float kScale = 0.25f;

void Fail(const char* what, const char* case_name) {
  std::fprintf(stderr, "FAIL: %s (%s)\n", what, case_name);
  ++g_failures;
}

// One operand value. The multiplier must be coprime with the modulus: a shared
// factor gives a constant operand, and a constant operand hides a wrong
// sub-tile offset.
float OperandValue(int index, int modulus, int shift) {
  return kScale * static_cast<float>((index * 5 + shift) % modulus - 3);
}

// C starts at a small nonzero value, so a beta term is visible.
std::vector<ComputeType> SeedC(int m, int n, int batch) {
  std::vector<ComputeType> c(static_cast<std::size_t>(batch) * m * n);
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = ComputeType(0.125f * static_cast<float>(static_cast<int>(i % 5)));
  }
  return c;
}

// One case: run the tile, then compare every element with a double reference.
template <typename COut>
void RunCase(const char* name, int m, int n, int k, int batch, bool ta, bool tb,
             float alpha, float beta) {
  const int rows_a = ta ? k : m;
  const int rows_b = tb ? n : k;
  const int lda = ta ? m : k;
  const int ldb = tb ? k : n;
  const int ldc = n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(m) * ldc;

  std::vector<ComputeType> host_a(static_cast<std::size_t>(batch) * stride_a);
  std::vector<ComputeType> host_b(static_cast<std::size_t>(batch) * stride_b);
  for (std::size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = ComputeType(OperandValue(static_cast<int>(i), 9, 0));
  }
  for (std::size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = ComputeType(OperandValue(static_cast<int>(i), 7, 2));
  }
  const std::vector<ComputeType> host_c = SeedC(m, n, batch);

  std::vector<COut> got(host_c.size());
  for (std::size_t i = 0; i < host_c.size(); ++i) {
    got[i] = static_cast<COut>(host_c[i]);
  }

  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.batch_count = batch;
  params.alpha = alpha;
  params.beta = beta;
  params.transpose_a = ta;
  params.transpose_b = tb;

  bool took = false;
  if constexpr (std::is_same_v<COut, float>) {
    took = EmuLaunchGemmTileToFp32(params, host_a.data(), host_b.data(),
                                   got.data());
  } else {
    took = EmuLaunchGemmTile(params, host_a.data(), host_b.data(), got.data());
  }
  if (!took) {
    Fail("the tile took the call", name);
    return;
  }

  const double relative =
      std::is_same_v<COut, float> ? kFp32Relative : kBf16Relative;
  for (int bi = 0; bi < batch; ++bi) {
    const ComputeType* a = host_a.data() + bi * stride_a;
    const ComputeType* b = host_b.data() + bi * stride_b;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int l = 0; l < k; ++l) {
          const float av = static_cast<float>(
              ta ? a[static_cast<std::ptrdiff_t>(l) * lda + i]
                 : a[static_cast<std::ptrdiff_t>(i) * lda + l]);
          const float bv = static_cast<float>(
              tb ? b[static_cast<std::ptrdiff_t>(j) * ldb + l]
                 : b[static_cast<std::ptrdiff_t>(l) * ldb + j]);
          sum += static_cast<double>(av) * static_cast<double>(bv);
        }
        const std::size_t index = static_cast<std::size_t>(bi) * stride_c +
                                  static_cast<std::size_t>(i) * ldc + j;
        const double want =
            static_cast<double>(alpha) * sum +
            static_cast<double>(beta) * static_cast<double>(host_c[index]);
        const double diff = std::fabs(static_cast<double>(got[index]) - want);
        if (diff > relative * (1.0 + std::fabs(want))) {
          std::fprintf(stderr,
                       "FAIL: %s missed at element %zu (batch %d): got %f "
                       "want %f\n",
                       name, index, bi, static_cast<double>(got[index]), want);
          ++g_failures;
          return;
        }
      }
    }
  }
}

// A call the tile cannot express is refused and writes nothing.
void CheckRefusal() {
  const char* kCase = "refusal";
  std::vector<ComputeType> a(4, ComputeType(1.0f));
  std::vector<ComputeType> b(4, ComputeType(1.0f));
  std::vector<ComputeType> c(4, ComputeType(2.0f));

  GemmParams params;
  params.m = 2;
  params.n = 2;
  params.k = 0;
  if (EmuLaunchGemmTile(params, a.data(), b.data(), c.data())) {
    Fail("an empty k extent is refused", kCase);
  }
  for (const ComputeType& value : c) {
    if (static_cast<float>(value) != 2.0f) {
      Fail("a refused call writes nothing", kCase);
      break;
    }
  }

  params.k = 2;
  params.m = 0;
  if (EmuLaunchGemmTile(params, a.data(), b.data(), c.data())) {
    Fail("an empty m extent is refused", kCase);
  }
}

}  // namespace

int main() {
  // The four operand layouts. The first is a whole tile pair, so the staging
  // loops, the sub-tile addressing, and the epilogue all run their full path.
  RunCase<ComputeType>("plain 128x64x32", 128, 64, 32, 1, false, false, 1.0f,
                       0.0f);
  RunCase<ComputeType>("transpose A 96x64x48", 96, 64, 48, 1, true, false, 1.0f,
                       0.0f);
  RunCase<ComputeType>("transpose B 128x48x32", 128, 48, 32, 1, false, true,
                       1.0f, 0.0f);
  RunCase<ComputeType>("both 40x24x16", 40, 24, 16, 1, true, true, 1.0f, 0.0f);

  // A ragged shape with a k tail and an in-place accumulate, then a batch of
  // two, so the per-batch base and stride arithmetic runs.
  RunCase<ComputeType>("ragged 45x23x50 alpha/beta", 45, 23, 50, 1, false,
                       false, 0.5f, 0.25f);
  RunCase<ComputeType>("batch 2 32x16x32 beta", 32, 16, 32, 2, false, false,
                       1.0f, 0.5f);

  // The weight-gradient form: fp32 out of bf16 operands.
  RunCase<float>("wgrad 64x32x32", 64, 32, 32, 1, false, true, 1.0f, 0.0f);

  CheckRefusal();

  if (g_failures != 0) {
    std::fprintf(stderr, "gemm_emu: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("gemm_emu: ok\n");
  return 0;
}
