// The off-device acceptance check for the fp32 shared-memory GEMM
// (docs/kernels.md Part 3, section 6 phase 3).
//
// It walks C = alpha * op(A) * op(B) + beta * C over the four operand layouts,
// a ragged shape with a k tail, a batch above one, and both schedules. Every
// case runs the real kernel (`backends/cuda/gemm/gemm_fp32.cu`) through the
// emulation engine, and compares against a host reference in double.
//
// What this checks: the staged layout, the four stride pairs, the thread-tile
// addressing, the double buffer's index, the accumulation, the epilogue's
// alpha, beta, and bounds, the sixteen-byte store and its fallback, and the
// batch base arithmetic. What it cannot check: a bank conflict, the register
// count, the residency, and the rate. Those need a card.
//
// CPU only. The kernel reads host pointers, because the emulator runs every
// fiber on the launching host thread.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "backends/cuda/gemm/gemm_fp32_plan.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"
#include "tools/cuda_sim/emu/emu_kernels.h"

namespace {

using nanochat::GemmParams;
using nanochat::kernels::EmuLaunchGemmFp32;

int g_failures = 0;

//: The two sides sum the same products in different orders, so the budget is a
//: few units in the last place of an fp32 sum.
constexpr double kRelative = 1e-5;
constexpr double kAbsolute = 1e-5;

void Fail(const char* case_name, const char* what) {
  std::fprintf(stderr, "FAIL: %s: %s\n", case_name, what);
  ++g_failures;
}

// One operand value. The multiplier is coprime with the modulus: a shared
// factor gives a constant operand, and a constant operand hides a wrong
// sub-tile offset.
float OperandValue(int index, int modulus, int shift) {
  return 0.25f * static_cast<float>((index * 5 + shift) % modulus - 3);
}

// The host reference: one product for each output element, summed in double.
void Reference(int m, int n, int k, int batch, bool transpose_a,
               bool transpose_b, float alpha, float beta,
               const std::vector<float>& a, const std::vector<float>& b,
               const std::vector<float>& init, std::vector<float>* out) {
  const int rows_a = transpose_a ? k : m;
  const int rows_b = transpose_b ? n : k;
  const int lda = transpose_a ? m : k;
  const int ldb = transpose_b ? k : n;
  const int ldc = n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(m) * ldc;
  out->assign(init.begin(), init.end());
  for (int slice = 0; slice < batch; ++slice) {
    const float* a_base = a.data() + static_cast<long long>(slice) * stride_a;
    const float* b_base = b.data() + static_cast<long long>(slice) * stride_b;
    const long long c_base_offset = static_cast<long long>(slice) * stride_c;
    float* c_base = out->data() + c_base_offset;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          const float a_value =
              transpose_a ? a_base[static_cast<long long>(kk) * lda + i]
                          : a_base[static_cast<long long>(i) * lda + kk];
          const float b_value =
              transpose_b ? b_base[static_cast<long long>(j) * ldb + kk]
                          : b_base[static_cast<long long>(kk) * ldb + j];
          sum += static_cast<double>(a_value) * static_cast<double>(b_value);
        }
        const long long offset = static_cast<long long>(i) * ldc + j;
        c_base[offset] = static_cast<float>(
            static_cast<double>(alpha) * sum +
            static_cast<double>(beta) *
                static_cast<double>(init[c_base_offset + offset]));
      }
    }
  }
}

void RunCase(const char* name, int m, int n, int k, int batch, bool transpose_a,
             bool transpose_b, float alpha, float beta, int schedule) {
  const int rows_a = transpose_a ? k : m;
  const int rows_b = transpose_b ? n : k;
  const int lda = transpose_a ? m : k;
  const int ldb = transpose_b ? k : n;
  const int ldc = n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(m) * ldc;

  std::vector<float> a(static_cast<std::size_t>(batch) * stride_a);
  std::vector<float> b(static_cast<std::size_t>(batch) * stride_b);
  std::vector<float> init(static_cast<std::size_t>(batch) * stride_c);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = OperandValue(static_cast<int>(i), 9, 0);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = OperandValue(static_cast<int>(i), 7, 2);
  }
  for (std::size_t i = 0; i < init.size(); ++i) {
    init[i] = 0.125f * static_cast<float>(static_cast<int>(i % 5)) - 0.25f;
  }

  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.batch_count = batch;
  params.alpha = alpha;
  params.beta = beta;
  params.transpose_a = transpose_a;
  params.transpose_b = transpose_b;

  std::vector<float> want;
  Reference(m, n, k, batch, transpose_a, transpose_b, alpha, beta, a, b, init,
            &want);

  const int first = schedule < 0 ? 0 : schedule;
  const int last = schedule < 0 ? nanochat::kernels::gemm_fp32::kScheduleCount
                                : schedule + 1;
  for (int index = first; index < last; ++index) {
    std::vector<float> got = init;
    const bool took =
        EmuLaunchGemmFp32(params, a.data(), b.data(), got.data(), index);
    if (!took) {
      Fail(name, "the launcher refused a case it claims to serve");
      continue;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double difference =
          std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
      const double budget =
          kAbsolute + kRelative * std::fabs(static_cast<double>(want[i]));
      if (difference <= budget) continue;
      std::fprintf(stderr,
                   "FAIL: %s schedule %d: element %zu reads %.9g and the "
                   "reference reads %.9g\n",
                   name, index, i, static_cast<double>(got[i]),
                   static_cast<double>(want[i]));
      ++g_failures;
      break;
    }
  }
}

}  // namespace

int main() {
  // The four operand layouts. The first is a whole tile pair, so the staging
  // loops, the thread-tile addressing, and the epilogue all run their full
  // path.
  RunCase("plain 128x64x32", 128, 64, 32, 1, false, false, 1.0f, 0.0f, -1);
  RunCase("transpose A 96x64x48", 96, 64, 48, 1, true, false, 1.0f, 0.0f, -1);
  RunCase("transpose B 128x48x32", 128, 48, 32, 1, false, true, 1.0f, 0.0f, -1);
  RunCase("both 40x24x16", 40, 24, 16, 1, true, true, 1.0f, 0.0f, -1);

  // A ragged shape with a k tail and an in-place accumulate, then a batch above
  // one, so the per-batch base and stride arithmetic runs.
  RunCase("ragged 45x23x50 alpha/beta", 45, 23, 50, 1, false, false, 0.5f,
          0.25f, -1);
  RunCase("batch 3 32x16x32 beta", 32, 16, 32, 3, false, false, 1.0f, 0.5f, -1);
  RunCase("batch 2 transpose A 32x24x40", 32, 24, 40, 2, true, false, 1.0f,
          0.0f, -1);

  // The edges: one element, and a k extent of many slices.
  RunCase("single 1x1x1", 1, 1, 1, 1, false, false, 1.0f, 0.0f, -1);
  RunCase("deep k 8x8x1024", 8, 8, 1024, 1, false, true, 1.0f, 0.0f, -1);

  // Every schedule over a shape that spans more than one k slice and more than
  // one tile of each extent. A wrong buffer index fails the schedules whose k
  // covers more than one slice.
  RunCase("sweep 200x130x96", 200, 130, 96, 1, false, true, 1.0f, 0.25f, -1);

  if (g_failures != 0) {
    std::fprintf(stderr, "gemm_fp32_emu: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("gemm_fp32_emu: ok\n");
  return 0;
}
