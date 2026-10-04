// T1 GPU smoke test for the row/elementwise families. Tiny shapes, host code
// only, exercising the sealed nanochat/kernels.h API plus the backend-internal
// fused residual helper. The exhaustive correctness and finite-difference
// tests sit next to the kernels; this target proves the package builds, links,
// and runs on the device. See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/rms_norm.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::kernels::Alloc;
using nanochat::kernels::Free;
using nanochat::kernels::Memcpy;
using nanochat::kernels::Synchronize;

#if defined(NANOCHAT_PRECISION_FP16)
ComputeType C(float v) { return nanochat::Fp16FromFloat(v); }
float U(ComputeType v) { return nanochat::Fp16ToFloat(v); }
// Recorded fp16 tolerance for a value that has passed through one or two
// half-rounded elementwise steps. The half ulp near 1.0 is 2^-10 ~= 9.8e-4,
// so 2e-3 (relative, through CheckClose's (1 + |want|) factor) leaves room for
// a couple of roundings without hiding a wrong result. The fp32 build keeps
// the tight 1e-4.
constexpr double kOutTol = 2e-3;
#else
ComputeType C(float v) { return v; }
float U(ComputeType v) { return v; }
constexpr double kOutTol = 1e-4;
#endif

int g_failures = 0;

void CheckClose(double got, double want, double tol, const char* what) {
  if (std::fabs(got - want) > tol * (1.0 + std::fabs(want))) {
    std::printf("FAIL: %s (got %.8g want %.8g)\n", what, got, want);
    ++g_failures;
  }
}

// Device buffer that uploads on construction and downloads on demand.
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  int count = 0;
  explicit DevBuf(const std::vector<T>& host) : count((int)host.size()) {
    ptr = static_cast<T*>(Alloc(sizeof(T) * host.size()));
    Memcpy(ptr, host.data(), sizeof(T) * host.size(), CopyDir::kHostToDevice);
  }
  ~DevBuf() { Free(ptr); }
  std::vector<T> Download() const {
    std::vector<T> host(count);
    Memcpy(host.data(), ptr, sizeof(T) * count, CopyDir::kDeviceToHost);
    return host;
  }
};

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  // RmsNorm forward: {3, 4} with eps = 0 normalizes to unit RMS.
  {
    nanochat::RmsNormParams p;
    p.rows = 1;
    p.dim = 2;
    p.eps = 0.0f;
    DevBuf<ComputeType> x(std::vector<ComputeType>{C(3.0f), C(4.0f)});
    DevBuf<ComputeType> out(std::vector<ComputeType>{C(0.0f), C(0.0f)});
    DevBuf<float> rstd(std::vector<float>{0.0f});
    nanochat::kernels::RmsNormForward(p, x.ptr, out.ptr, rstd.ptr);
    Synchronize();
    const float r = 1.0f / std::sqrt(12.5f);
    const std::vector<ComputeType> o = out.Download();
    const std::vector<float> rs = rstd.Download();
    CheckClose(rs[0], r, 1e-4, "rms rstd");
    CheckClose(U(o[0]), 3.0f * r, kOutTol, "rms out0");
    CheckClose(U(o[1]), 4.0f * r, kOutTol, "rms out1");
  }

  // QkPrep forward with an identity rotation: normed row scaled by 1.2.
  {
    nanochat::QkPrepParams p;
    p.batch = 1;
    p.seq = 1;
    p.num_heads = 1;
    p.num_kv_heads = 1;
    p.head_dim = 2;
    p.eps = 0.0f;
    p.scale = 1.2f;
    DevBuf<float> cos(std::vector<float>{1.0f});
    DevBuf<float> sin(std::vector<float>{0.0f});
    DevBuf<ComputeType> q(std::vector<ComputeType>{C(3.0f), C(4.0f)});
    DevBuf<ComputeType> k(std::vector<ComputeType>{C(3.0f), C(4.0f)});
    nanochat::kernels::QkPrepForward(p, cos.ptr, sin.ptr, q.ptr, k.ptr);
    Synchronize();
    const float r = 1.0f / std::sqrt(12.5f) * 1.2f;
    const std::vector<ComputeType> qh = q.Download();
    CheckClose(U(qh[0]), 3.0f * r, kOutTol, "qkprep q0");
    CheckClose(U(qh[1]), 4.0f * r, kOutTol, "qkprep q1");
  }

  // Pointwise scale-add.
  {
    const int n = 3;
    DevBuf<ComputeType> a(std::vector<ComputeType>{C(1.0f), C(-2.0f), C(0.5f)});
    DevBuf<ComputeType> b(std::vector<ComputeType>{C(2.0f), C(3.0f), C(-1.0f)});
    DevBuf<ComputeType> out(std::vector<ComputeType>(n, C(0.0f)));
    nanochat::kernels::PointwiseForward(nanochat::PointwiseOp::kScaleAdd, n,
                                        a.ptr, b.ptr, 0.5f, 0.25f, out.ptr);
    Synchronize();
    const std::vector<ComputeType> o = out.Download();
    CheckClose(U(o[0]), 0.5f * 1.0f + 0.25f * 2.0f, kOutTol, "pointwise 0");
    CheckClose(U(o[1]), 0.5f * -2.0f + 0.25f * 3.0f, kOutTol, "pointwise 1");
  }

  // GlobalNorm: {3, 4} has norm 5, clipped to 2.5 -> {1.5, 2}.
  {
    DevBuf<ComputeType> g(std::vector<ComputeType>{C(3.0f), C(4.0f)});
    DevBuf<float> norm(std::vector<float>{0.0f});
    nanochat::kernels::GlobalNorm(2, 2.5f, g.ptr, norm.ptr);
    Synchronize();
    const std::vector<ComputeType> gh = g.Download();
    const std::vector<float> nh = norm.Download();
    CheckClose(nh[0], 5.0, 1e-4, "globalnorm pre-clip");
    CheckClose(U(gh[0]), 1.5, kOutTol, "globalnorm clipped 0");
    CheckClose(U(gh[1]), 2.0, kOutTol, "globalnorm clipped 1");
  }

  // Fused residual add + norm: out = rmsnorm(x + residual).
  {
    nanochat::RmsNormParams p;
    p.rows = 1;
    p.dim = 2;
    p.eps = 0.0f;
    DevBuf<ComputeType> x(std::vector<ComputeType>{C(1.0f), C(1.0f)});
    DevBuf<ComputeType> res(std::vector<ComputeType>{C(2.0f), C(3.0f)});
    DevBuf<ComputeType> out(std::vector<ComputeType>{C(0.0f), C(0.0f)});
    DevBuf<float> rstd(std::vector<float>{0.0f});
    nanochat::cuda_kernels::RmsNormForwardFusedResidual(p, x.ptr, res.ptr,
                                                        out.ptr, rstd.ptr);
    Synchronize();
    const std::vector<ComputeType> rh = res.Download();
    const std::vector<ComputeType> oh = out.Download();
    CheckClose(U(rh[0]), 3.0, kOutTol, "fused residual 0");
    CheckClose(U(rh[1]), 4.0, kOutTol, "fused residual 1");
    const float r = 1.0f / std::sqrt(12.5f);
    CheckClose(U(oh[0]), 3.0f * r, kOutTol, "fused out0");
    CheckClose(U(oh[1]), 4.0f * r, kOutTol, "fused out1");
  }

  if (g_failures != 0) {
    std::printf("row kernels: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("row kernels: smoke checks passed\n");
  return 0;
}
