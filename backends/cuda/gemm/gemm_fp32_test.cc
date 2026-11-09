// The device gate of the fp32 shared-memory GEMM
// (docs/kernels.md Part 3, section 6 phase 3).
//
// The reference is a host loop in double, and not cuBLAS. The kernel
// accumulates fp32 products in fp32, so the only difference between the two
// sides is the order of the sum, and a double reference states that difference
// once instead of twice.
//
// The cases cover the whole `GemmParams` contract: the four operand forms, a
// ragged shape whose k extent is not a whole slice, a batch above one, the
// alpha and the beta terms, and every schedule of the sweep.
//
// A run needs an fp32 build and a card. The build compiles this source only in
// an fp32 build, because the kernel reads fp32 operands.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "backends/cuda/gemm/gemm_fp32_launch.h"
#include "backends/cuda/gemm/gemm_fp32_plan.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::CopyDir;
using nanochat::GemmParams;
using nanochat::kernels::Alloc;
using nanochat::kernels::Free;
using nanochat::kernels::LaunchGemmFp32;
using nanochat::kernels::Memcpy;
using nanochat::kernels::Synchronize;

int g_failures = 0;

//: The two sides sum the same products in different orders, so the budget is a
//: few units in the last place of an fp32 sum of at most 256 terms.
constexpr double kRelative = 1e-4;
constexpr double kAbsolute = 1e-4;

void Fail(const char* name, const char* what) {
  std::fprintf(stderr, "FAIL: %s: %s\n", name, what);
  ++g_failures;
}

float OperandValue(int index, int modulus, int shift) {
  return 0.25f * static_cast<float>((index * 5 + shift) % modulus - 3);
}

struct Case {
  const char* name;
  int m;
  int n;
  int k;
  int batch;
  bool transpose_a;
  bool transpose_b;
  float alpha;
  float beta;
};

// The host reference: one product for each output element, summed in double.
void Reference(const Case& c, const std::vector<float>& a,
               const std::vector<float>& b, const std::vector<float>& init,
               std::vector<float>* out) {
  const int rows_a = c.transpose_a ? c.k : c.m;
  const int rows_b = c.transpose_b ? c.n : c.k;
  const int lda = c.transpose_a ? c.m : c.k;
  const int ldb = c.transpose_b ? c.k : c.n;
  const int ldc = c.n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(c.m) * ldc;
  out->assign(init.begin(), init.end());
  for (int batch = 0; batch < c.batch; ++batch) {
    const float* a_base = a.data() + static_cast<long long>(batch) * stride_a;
    const float* b_base = b.data() + static_cast<long long>(batch) * stride_b;
    const long long c_base_offset = static_cast<long long>(batch) * stride_c;
    float* c_base = out->data() + c_base_offset;
    for (int i = 0; i < c.m; ++i) {
      for (int j = 0; j < c.n; ++j) {
        double sum = 0.0;
        for (int kk = 0; kk < c.k; ++kk) {
          const float a_value =
              c.transpose_a ? a_base[static_cast<long long>(kk) * lda + i]
                            : a_base[static_cast<long long>(i) * lda + kk];
          const float b_value =
              c.transpose_b ? b_base[static_cast<long long>(j) * ldb + kk]
                            : b_base[static_cast<long long>(kk) * ldb + j];
          sum += static_cast<double>(a_value) * static_cast<double>(b_value);
        }
        const long long offset = static_cast<long long>(i) * ldc + j;
        c_base[offset] = static_cast<float>(
            static_cast<double>(c.alpha) * sum +
            static_cast<double>(c.beta) *
                static_cast<double>(init[c_base_offset + offset]));
      }
    }
  }
}

void RunCase(const Case& c) {
  const int rows_a = c.transpose_a ? c.k : c.m;
  const int rows_b = c.transpose_b ? c.n : c.k;
  const int lda = c.transpose_a ? c.m : c.k;
  const int ldb = c.transpose_b ? c.k : c.n;
  const int ldc = c.n;
  const long long stride_a = static_cast<long long>(rows_a) * lda;
  const long long stride_b = static_cast<long long>(rows_b) * ldb;
  const long long stride_c = static_cast<long long>(c.m) * ldc;

  std::vector<float> host_a(static_cast<std::size_t>(c.batch) * stride_a);
  std::vector<float> host_b(static_cast<std::size_t>(c.batch) * stride_b);
  std::vector<float> host_init(static_cast<std::size_t>(c.batch) * stride_c);
  for (std::size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = OperandValue(static_cast<int>(i), 9, 0);
  }
  for (std::size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = OperandValue(static_cast<int>(i), 7, 2);
  }
  for (std::size_t i = 0; i < host_init.size(); ++i) {
    host_init[i] = 0.125f * static_cast<float>(static_cast<int>(i % 5)) - 0.25f;
  }

  void* dev_a = Alloc(host_a.size() * sizeof(float));
  void* dev_b = Alloc(host_b.size() * sizeof(float));
  void* dev_c = Alloc(host_init.size() * sizeof(float));
  Memcpy(dev_a, host_a.data(), host_a.size() * sizeof(float),
         CopyDir::kHostToDevice);
  Memcpy(dev_b, host_b.data(), host_b.size() * sizeof(float),
         CopyDir::kHostToDevice);

  GemmParams params;
  params.m = c.m;
  params.n = c.n;
  params.k = c.k;
  params.batch_count = c.batch;
  params.alpha = c.alpha;
  params.beta = c.beta;
  params.transpose_a = c.transpose_a;
  params.transpose_b = c.transpose_b;

  std::vector<float> want;
  Reference(c, host_a, host_b, host_init, &want);
  std::vector<float> got(want.size());

  for (int schedule = 0;
       schedule < nanochat::kernels::gemm_fp32::kScheduleCount; ++schedule) {
    Memcpy(dev_c, host_init.data(), host_init.size() * sizeof(float),
           CopyDir::kHostToDevice);
    const bool took = LaunchGemmFp32(params, static_cast<const float*>(dev_a),
                                     static_cast<const float*>(dev_b),
                                     static_cast<float*>(dev_c), schedule);
    Synchronize();
    if (!took) {
      Fail(c.name, "the launcher refused a case it claims to serve");
      continue;
    }
    Memcpy(got.data(), dev_c, got.size() * sizeof(float),
           CopyDir::kDeviceToHost);
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double difference =
          std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
      const double budget =
          kAbsolute + kRelative * std::fabs(static_cast<double>(want[i]));
      if (difference <= budget) continue;
      std::fprintf(stderr,
                   "FAIL: %s schedule %d: element %zu reads %.9g and the "
                   "reference reads %.9g\n",
                   c.name, schedule, i, static_cast<double>(got[i]),
                   static_cast<double>(want[i]));
      ++g_failures;
      break;
    }
  }

  Free(dev_c);
  Free(dev_b);
  Free(dev_a);
}

}  // namespace

int main() {
  const Case kCases[] = {
      // The four operand forms, at a shape past one tile on both extents and
      // past one k slice.
      {"plain 300x200x40", 300, 200, 40, 1, false, false, 1.0f, 0.0f},
      {"transpose B 300x200x40", 300, 200, 40, 1, false, true, 1.0f, 0.0f},
      {"transpose A 300x200x40", 300, 200, 40, 1, true, false, 1.0f, 0.0f},
      {"both 300x200x40", 300, 200, 40, 1, true, true, 1.0f, 0.0f},
      // The alpha and beta terms, with C read and scaled.
      {"beta plain 129x67x33", 129, 67, 33, 1, false, false, 0.5f, 0.25f},
      {"beta transpose B 129x67x33", 129, 67, 33, 1, false, true, 0.5f, 0.25f},
      // A batch above one, and a k extent that is not a whole slice.
      {"batch 64x64x17 of 3", 64, 64, 17, 3, false, false, 1.0f, 0.5f},
      {"batch transpose A 64x48x23 of 2", 64, 48, 23, 2, true, false, 1.0f,
       0.0f},
      // A single tile, and a shape whose output is narrower than one tile.
      {"one tile 128x128x16", 128, 128, 16, 1, false, false, 1.0f, 0.0f},
      {"narrow 1x1x1", 1, 1, 1, 1, false, false, 1.0f, 0.0f},
      {"wide k 8x8x1024", 8, 8, 1024, 1, false, true, 1.0f, 0.0f},
  };
  for (const Case& c : kCases) RunCase(c);

  if (g_failures != 0) {
    std::fprintf(stderr, "gemm fp32 gpu test: %d check(s) failed\n",
                 g_failures);
    return 1;
  }
  std::printf("gemm fp32 gpu test: ok\n");
  return 0;
}
