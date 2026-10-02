// T1 GPU correctness for the CUDA device runtime and the cuBLAS GEMM. Host
// code only: it calls the sealed nanochat/kernels.h API, so it exercises
// exactly the seam the rest of the project links against and stays free of
// vendor headers. Tiny shapes only, per docs/testing.md.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::kComputeDType;
using nanochat::kernels::Alloc;
using nanochat::kernels::Free;
using nanochat::kernels::Gemm;
using nanochat::kernels::GetCaps;
using nanochat::kernels::Memcpy;
using nanochat::kernels::Memset;
using nanochat::kernels::Synchronize;

#if defined(NANOCHAT_PRECISION_FP16)
float ToFloat(nanochat::Fp16 value) { return nanochat::Fp16ToFloat(value); }
#else
float ToFloat(float value) { return value; }
#endif

ComputeType FromFloat(float value) {
#if defined(NANOCHAT_PRECISION_FP16)
  return nanochat::Fp16FromFloat(value);
#else
  return value;
#endif
}

#if defined(NANOCHAT_PRECISION_FP16)
// Recorded fp16 GEMM tolerance. cuBLAS on Pascal (sm_61) has no fp32-compute
// half path and runs native HGEMM (fp16 accumulation), so the sum over k
// rounds per term; Volta+ uses the fp32-accumulating tensor-op path. The
// reference itself accumulates in fp32, and the observed worst case over
// these tiny shapes is well inside 1e-2.
constexpr float kTolerance = 1e-2f;
#else
constexpr float kTolerance = 1e-3f;
#endif

double g_worst_error = 0.0;

// Stores A and B in the layout the transpose flags imply, computes the host
// reference for C = alpha * op(A) * op(B) + beta * C, runs the device GEMM, and
// compares. All three modes share this universal formula.
bool RunCase(GemmMode mode, bool ta, bool tb, int m, int n, int k, int batch,
             float alpha, float beta) {
  const int rows_a = ta ? k : m;
  const int lda = ta ? m : k;
  const int rows_b = tb ? n : k;
  const int ldb = tb ? k : n;
  const int ldc = n;

  const std::int64_t stride_a = static_cast<std::int64_t>(rows_a) * lda;
  const std::int64_t stride_b = static_cast<std::int64_t>(rows_b) * ldb;
  const std::int64_t stride_c = static_cast<std::int64_t>(m) * ldc;

  const std::size_t total_a = static_cast<std::size_t>(batch) * stride_a;
  const std::size_t total_b = static_cast<std::size_t>(batch) * stride_b;
  const std::size_t total_c = static_cast<std::size_t>(batch) * stride_c;

  std::vector<ComputeType> host_a(total_a);
  std::vector<ComputeType> host_b(total_b);
  std::vector<ComputeType> host_c(total_c);
  std::vector<float> ref_a(total_a);
  std::vector<float> ref_b(total_b);
  std::vector<float> ref_c(total_c);

  for (std::size_t i = 0; i < total_a; ++i) {
    host_a[i] = FromFloat(0.25f * static_cast<float>(static_cast<int>(i % 9) - 4));
    ref_a[i] = ToFloat(host_a[i]);
  }
  for (std::size_t i = 0; i < total_b; ++i) {
    host_b[i] = FromFloat(0.2f * static_cast<float>(static_cast<int>(i % 7) - 3));
    ref_b[i] = ToFloat(host_b[i]);
  }
  for (std::size_t i = 0; i < total_c; ++i) {
    host_c[i] = FromFloat(0.1f * static_cast<float>(static_cast<int>(i % 5)));
    ref_c[i] = ToFloat(host_c[i]);
  }

  for (int bi = 0; bi < batch; ++bi) {
    const float* a = ref_a.data() + bi * stride_a;
    const float* b = ref_b.data() + bi * stride_b;
    float* c = ref_c.data() + bi * stride_c;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int l = 0; l < k; ++l) {
          const float av = ta ? a[l * lda + i] : a[i * lda + l];
          const float bv = tb ? b[j * ldb + l] : b[l * ldb + j];
          acc += av * bv;
        }
        c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
      }
    }
  }

  void* dev_a = Alloc(total_a * sizeof(ComputeType));
  void* dev_b = Alloc(total_b * sizeof(ComputeType));
  void* dev_c = Alloc(total_c * sizeof(ComputeType));
  Memcpy(dev_a, host_a.data(), total_a * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(dev_b, host_b.data(), total_b * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(dev_c, host_c.data(), total_c * sizeof(ComputeType),
         CopyDir::kHostToDevice);

  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.batch_count = batch;
  params.alpha = alpha;
  params.beta = beta;
  params.transpose_a = ta;
  params.transpose_b = tb;
  Gemm(mode, params, static_cast<const ComputeType*>(dev_a),
       static_cast<const ComputeType*>(dev_b),
       static_cast<ComputeType*>(dev_c));
  Synchronize();
  Memcpy(host_c.data(), dev_c, total_c * sizeof(ComputeType),
         CopyDir::kDeviceToHost);

  Free(dev_a);
  Free(dev_b);
  Free(dev_c);

  for (std::size_t i = 0; i < total_c; ++i) {
    const float got = ToFloat(host_c[i]);
    const double diff = std::fabs(got - ref_c[i]);
    if (diff > g_worst_error) g_worst_error = diff;
    if (diff > kTolerance) {
      std::fprintf(stderr,
                   "gemm mismatch (mode=%d ta=%d tb=%d m=%d n=%d k=%d "
                   "batch=%d at %zu): got %f expected %f\n",
                   static_cast<int>(mode), ta ? 1 : 0, tb ? 1 : 0, m, n, k,
                   batch, i, got, ref_c[i]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  const nanochat::Caps caps = GetCaps();
  if (!caps.is_device || !caps.has_cublas) {
    std::fprintf(stderr, "cuda_runtime: no device or cuBLAS available\n");
    return 1;
  }
  if (!caps.Supports(kComputeDType)) {
    std::fprintf(stderr, "cuda_runtime: build precision not supported\n");
    return 1;
  }
  std::printf("cuda_runtime: device %s, compute capability %d.%d\n",
              caps.device_name, caps.compute_major, caps.compute_minor);

  // Device round trip: alloc, copy in, memset, copy back, free.
  {
    constexpr std::size_t kCount = 64;
    const std::size_t bytes = kCount * sizeof(ComputeType);
    void* buffer = Alloc(bytes);
    std::vector<ComputeType> host(kCount, FromFloat(1.0f));
    Memcpy(buffer, host.data(), bytes, CopyDir::kHostToDevice);
    Memset(buffer, 0, bytes);
    Memcpy(host.data(), buffer, bytes, CopyDir::kDeviceToHost);
    Free(buffer);
    Synchronize();
    for (std::size_t i = 0; i < kCount; ++i) {
      if (ToFloat(host[i]) != 0.0f) {
        std::fprintf(stderr, "cuda_runtime: memset round trip failed\n");
        return 1;
      }
    }
  }

  struct Case {
    GemmMode mode;
    bool ta;
    bool tb;
    int m;
    int n;
    int k;
    int batch;
    float alpha;
    float beta;
  };
  const Case cases[] = {
      {GemmMode::kForward, false, false, 4, 3, 5, 1, 1.0f, 0.0f},
      {GemmMode::kForward, false, true, 3, 4, 5, 1, 0.5f, 0.25f},
      {GemmMode::kForward, true, false, 3, 4, 5, 1, 1.0f, 0.0f},
      {GemmMode::kForward, true, true, 4, 3, 5, 1, 1.0f, 0.5f},
      {GemmMode::kForward, false, false, 2, 3, 4, 3, 1.0f, 0.0f},
      {GemmMode::kDgrad, true, false, 3, 2, 4, 1, 1.0f, 0.0f},
      {GemmMode::kWgrad, false, true, 2, 4, 3, 1, 1.0f, 0.0f},
  };

  for (const Case& c : cases) {
    if (!RunCase(c.mode, c.ta, c.tb, c.m, c.n, c.k, c.batch, c.alpha, c.beta)) {
      return 1;
    }
  }

  std::printf("cuda_runtime: ok, %zu gemm cases verified",
              sizeof(cases) / sizeof(cases[0]));
  std::printf(" (max abs error %.3g, tolerance %.3g)\n", g_worst_error,
              kTolerance);
  return 0;
}
