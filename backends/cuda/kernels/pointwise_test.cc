// T1 GPU correctness + finite-difference test for the Pointwise family:
// scale, scale-add, gate-mul, relu^2, and softcap, forward and backward. Tiny
// shapes only. See docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/row_ref.h"

namespace {

using nanochat::ComputeType;
using nanochat::PointwiseOp;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using namespace nanochat::dev::rowref;

constexpr float kAlpha = 1.5f;
constexpr float kBeta = 0.5f;

bool UsesB(PointwiseOp op) {
  return op == PointwiseOp::kScaleAdd || op == PointwiseOp::kGateMul;
}

void TestOp(PointwiseOp op, const char* name, const std::vector<float>& a0,
            const std::vector<float>& b0, const std::vector<float>& w) {
  const int n = static_cast<int>(a0.size());

  DevBuf<ComputeType> a(ToStorage(a0));
  DevBuf<ComputeType> b(ToStorage(b0));
  DevBuf<ComputeType> out(n);
  nanochat::kernels::PointwiseForward(op, n, a.ptr, b.ptr, kAlpha, kBeta,
                                      out.ptr);
  nanochat::kernels::Synchronize();
  const std::vector<float> ref_out =
      PointwiseForward(op, a0, b0, kAlpha, kBeta);
  CheckVectorClose(FromStorage(out.Download()), ref_out, 1e-4, name);

  DevBuf<ComputeType> dy(ToStorage(w));
  DevBuf<ComputeType> da(n);
  DevBuf<ComputeType> db(n);
  nanochat::kernels::PointwiseBackward(op, n, a.ptr, b.ptr, dy.ptr, kAlpha,
                                       kBeta, da.ptr, db.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_da(n, 0.0f);
  std::vector<float> ref_db(n, 0.0f);
  PointwiseBackward(op, a0, b0, w, kAlpha, kBeta, &ref_da, &ref_db);
  CheckVectorClose(FromStorage(da.Download()), ref_da, 1e-4, "pointwise da");
  CheckVectorClose(FromStorage(db.Download()), ref_db, 1e-4, "pointwise db");

#if !defined(NANOCHAT_PRECISION_FP16)
  auto loss_at = [&](const std::vector<float>& av,
                     const std::vector<float>& bv) {
    const std::vector<float> o = PointwiseForward(op, av, bv, kAlpha, kBeta);
    double loss = 0.0;
    for (int i = 0; i < n; ++i) loss += o[i] * w[i];
    return loss;
  };
  CheckFiniteDifference(
      a0, ref_da, [&](const std::vector<float>& av) { return loss_at(av, b0); },
      2e-3, "pointwise da");
  if (UsesB(op)) {
    CheckFiniteDifference(
        b0, ref_db,
        [&](const std::vector<float>& bv) { return loss_at(a0, bv); }, 2e-3,
        "pointwise db");
  }
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("pointwise gpu test\n");

  Rng rng;
  const int n = 8;
  const std::vector<float> a0 = RandomVec(n, &rng);
  const std::vector<float> b0 = RandomVec(n, &rng);
  const std::vector<float> w = RandomVec(n, &rng);

  TestOp(PointwiseOp::kScale, "scale", a0, b0, w);
  TestOp(PointwiseOp::kScaleAdd, "scale-add", a0, b0, w);
  TestOp(PointwiseOp::kGateMul, "gate-mul", a0, b0, w);
  TestOp(PointwiseOp::kReluSquare, "relu-square", a0, b0, w);
  TestOp(PointwiseOp::kSoftcap, "softcap", a0, b0, w);

  if (Failures() != 0) {
    std::printf("pointwise: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("pointwise: all checks passed\n");
  return 0;
}
