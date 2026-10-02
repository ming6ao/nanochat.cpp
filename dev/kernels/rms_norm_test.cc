// T1 GPU correctness + finite-difference test for the RmsNorm family,
// including the backend-internal fused residual variant. Tiny shapes only.
// The device results are compared against the host reference in row_ref.h and
// the backward is checked by central differences of that reference forward.
// See docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/rms_norm.h"
#include "gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "row_ref.h"

namespace {

using nanochat::ComputeType;
using nanochat::RmsNormParams;
using nanochat::dev::CheckClose;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::C;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::U;
using namespace nanochat::dev::rowref;

void TestForwardBackward() {
  Rng rng;
  // Two shapes: the tiny attention head and an odd, non-power-of-two width.
  const std::pair<int, int> shapes[] = {{16, 4}, {5, 7}};
  for (const auto& shape : shapes) {
    RmsNormParams p;
    p.rows = shape.first;
    p.dim = shape.second;
    p.eps = 1e-6f;

    const std::vector<float> x0 = RandomVec(p.rows * p.dim, &rng);
    const std::vector<float> dy0 = RandomVec(p.rows * p.dim, &rng);

    DevBuf<ComputeType> x(ToStorage(x0));
    DevBuf<ComputeType> out(p.rows * p.dim);
    DevBuf<float> rstd(p.rows);
    nanochat::kernels::RmsNormForward(p, x.ptr, out.ptr, rstd.ptr);
    nanochat::kernels::Synchronize();

    std::vector<float> ref_rstd;
    const std::vector<float> ref_out = RmsNormForward(p, x0, &ref_rstd);

    CheckVectorClose(FromStorage(out.Download()), ref_out, 1e-4, "rms fwd out");
    CheckVectorClose(rstd.Download(), ref_rstd, 1e-4, "rms fwd rstd");

    DevBuf<ComputeType> dy(ToStorage(dy0));
    DevBuf<ComputeType> dx(p.rows * p.dim);
    nanochat::kernels::RmsNormBackward(p, x.ptr, dy.ptr, rstd.ptr, dx.ptr);
    nanochat::kernels::Synchronize();

    const std::vector<float> ref_dx =
        RmsNormBackward(p, x0, dy0, ref_rstd);
    CheckVectorClose(FromStorage(dx.Download()), ref_dx, 1e-4, "rms bwd dx");

#if !defined(NANOCHAT_PRECISION_FP16)
    // Finite-difference the device backward against the reference forward.
    // The kernel uses the reference's rstd, which the forward above shows is
    // equal to the device's within tolerance.
    auto loss_at = [&](const std::vector<float>& xv) {
      std::vector<float> rr;
      const std::vector<float> o = RmsNormForward(p, xv, &rr);
      double loss = 0.0;
      for (std::size_t i = 0; i < o.size(); ++i) loss += o[i] * dy0[i];
      return loss;
    };
    CheckFiniteDifference(x0, FromStorage(dx.Download()), loss_at, 2e-3,
                          "rms dx");
#endif
  }
}

void TestFusedResidual() {
  Rng rng;
  RmsNormParams p;
  p.rows = 16;
  p.dim = 4;
  p.eps = 1e-6f;

  const std::vector<float> x0 = RandomVec(p.rows * p.dim, &rng);
  const std::vector<float> res0 = RandomVec(p.rows * p.dim, &rng);
  const std::vector<float> dy0 = RandomVec(p.rows * p.dim, &rng);

  DevBuf<ComputeType> x(ToStorage(x0));
  DevBuf<ComputeType> res(ToStorage(res0));
  DevBuf<ComputeType> out(p.rows * p.dim);
  DevBuf<float> rstd(p.rows);
  nanochat::cuda_kernels::RmsNormForwardFusedResidual(p, x.ptr, res.ptr,
                                                      out.ptr, rstd.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_res = res0;
  std::vector<float> ref_rstd;
  std::vector<float> ref_out;
  FusedResidualForward(p, x0, &ref_res, &ref_out, &ref_rstd);

  CheckVectorClose(FromStorage(res.Download()), ref_res, 1e-4, "fused resid");
  CheckVectorClose(FromStorage(out.Download()), ref_out, 1e-4, "fused out");
  CheckVectorClose(rstd.Download(), ref_rstd, 1e-4, "fused rstd");

  DevBuf<ComputeType> dy(ToStorage(dy0));
  DevBuf<ComputeType> dx(p.rows * p.dim);
  DevBuf<ComputeType> dres(p.rows * p.dim);
  nanochat::cuda_kernels::RmsNormBackwardFusedResidual(
      p, res.ptr, rstd.ptr, dy.ptr, dx.ptr, dres.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> ref_g =
      FusedResidualBackward(p, ref_res, ref_rstd, dy0);
  CheckVectorClose(FromStorage(dx.Download()), ref_g, 1e-4, "fused dx");
  CheckVectorClose(FromStorage(dres.Download()), ref_g, 1e-4, "fused dres");

#if !defined(NANOCHAT_PRECISION_FP16)
  auto loss_at = [&](const std::vector<float>& xv,
                     const std::vector<float>& rv) {
    RmsNormParams p_loss = p;
    std::vector<float> rr;
    std::vector<float> s = rv;
    std::vector<float> o;
    FusedResidualForward(p_loss, xv, &s, &o, &rr);
    double loss = 0.0;
    for (std::size_t i = 0; i < o.size(); ++i) loss += o[i] * dy0[i];
    return loss;
  };
  CheckFiniteDifference(
      x0, FromStorage(dx.Download()),
      [&](const std::vector<float>& xv) { return loss_at(xv, res0); }, 2e-3,
      "fused dx");
  CheckFiniteDifference(
      res0, FromStorage(dres.Download()),
      [&](const std::vector<float>& rv) { return loss_at(x0, rv); }, 2e-3,
      "fused dres");
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("rms_norm gpu test\n");
  TestForwardBackward();
  TestFusedResidual();
  if (Failures() != 0) {
    std::printf("rms_norm: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("rms_norm: all checks passed\n");
  return 0;
}
