#ifndef NANOCHAT_DEV_KERNELS_GPU_TEST_UTILS_H_
#define NANOCHAT_DEV_KERNELS_GPU_TEST_UTILS_H_

// Shared host-side scaffolding for the dev/kernels GPU tests: storage
// conversion, a deterministic RNG, a device buffer, tolerance checks, and the
// central finite-difference gradient helper. Header-only; every test binary
// gets its own copy. See docs/testing.md.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace dev {

#if defined(NANOCHAT_PRECISION_FP16)
inline ComputeType C(float v) { return Fp16FromFloat(v); }
inline float U(ComputeType v) { return Fp16ToFloat(v); }
#else
inline ComputeType C(float v) { return v; }
inline float U(ComputeType v) { return v; }
#endif

inline std::vector<ComputeType> ToStorage(const std::vector<float>& v) {
  std::vector<ComputeType> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = C(v[i]);
  return out;
}

inline std::vector<float> FromStorage(const std::vector<ComputeType>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = U(v[i]);
  return out;
}

// Deterministic xorshift so the tests never depend on the platform RNG.
struct Rng {
  std::uint64_t s = 0x1234567890abcdefULL;
  float Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>((s >> 40) % 2001) / 1000.0f - 1.0f;
  }
};

inline std::vector<float> RandomVec(int n, Rng* rng) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = rng->Next();
  return v;
}

inline int& Failures() {
  static int failures = 0;
  return failures;
}

inline void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++Failures();
  }
}

inline void CheckClose(double got, double want, double tol, const char* what) {
  if (std::fabs(got - want) > tol * (1.0 + std::fabs(want))) {
    std::printf("FAIL: %s (got %.8g want %.8g)\n", what, got, want);
    ++Failures();
  }
}

inline void CheckVectorClose(const std::vector<float>& got,
                             const std::vector<float>& want, double tol,
                             const char* what) {
  if (got.size() != want.size()) {
    std::printf("FAIL: %s (size %zu vs %zu)\n", what, got.size(), want.size());
    ++Failures();
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(got[i] - want[i]);
    if (diff > worst) worst = diff;
    if (diff > tol * (1.0 + std::fabs(want[i]))) {
      std::printf("FAIL: %s [%zu] got %.8g want %.8g\n", what, i, got[i],
                  want[i]);
      ++Failures();
      return;
    }
  }
  std::printf("  %-24s max abs error %.3g\n", what, worst);
}

// A typed device buffer that owns its allocation. Non-copyable.
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  int count = 0;

  explicit DevBuf(int n) : count(n) {
    ptr = static_cast<T*>(nanochat::kernels::Alloc(sizeof(T) * n));
  }
  explicit DevBuf(const std::vector<T>& host) : count((int)host.size()) {
    ptr = static_cast<T*>(nanochat::kernels::Alloc(sizeof(T) * host.size()));
    nanochat::kernels::Memcpy(ptr, host.data(), sizeof(T) * host.size(),
                              nanochat::CopyDir::kHostToDevice);
  }
  ~DevBuf() { nanochat::kernels::Free(ptr); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;

  void Upload(const std::vector<T>& host) {
    nanochat::kernels::Memcpy(ptr, host.data(), sizeof(T) * host.size(),
                              nanochat::CopyDir::kHostToDevice);
  }
  std::vector<T> Download() const {
    std::vector<T> host(count);
    nanochat::kernels::Memcpy(host.data(), ptr, sizeof(T) * count,
                              nanochat::CopyDir::kDeviceToHost);
    return host;
  }
};

#if !defined(NANOCHAT_PRECISION_FP16)
// Central finite difference of a scalar loss over x, compared against an
// analytic gradient. fp32 only: fp16 storage cannot resolve the perturbation.
template <typename LossFn>
void CheckFiniteDifference(const std::vector<float>& x0,
                           const std::vector<float>& analytic, LossFn loss_at,
                           double tol, const char* what) {
  const float h = 1e-2f;
  double worst = 0.0;
  for (std::size_t i = 0; i < x0.size(); ++i) {
    std::vector<float> xp = x0;
    xp[i] += h;
    std::vector<float> xm = x0;
    xm[i] -= h;
    const double num = (loss_at(xp) - loss_at(xm)) / (2.0 * h);
    const double diff = std::fabs(num - analytic[i]);
    if (diff > worst) worst = diff;
    if (diff > tol * (1.0 + std::fabs(num))) {
      std::printf("FAIL: %s [%zu] analytic %.6g numeric %.6g\n", what, i,
                  analytic[i], num);
      ++Failures();
      return;
    }
  }
  std::printf("  finite-diff %-14s max abs error %.3g\n", what, worst);
}
#else
// fp16 storage cannot resolve the finite-difference perturbation, so the check
// is skipped (docs/testing.md). Provide the symbol so the family tests that
// import it still compile under `--config=fp16`.
template <typename LossFn>
void CheckFiniteDifference(const std::vector<float>&, const std::vector<float>&,
                           LossFn, double, const char* what) {
  std::printf("  finite-diff %-14s skipped (fp16 build)\n", what);
}
#endif

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_GPU_TEST_UTILS_H_
