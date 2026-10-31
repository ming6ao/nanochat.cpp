// T1 GPU correctness + finite-difference test for the Pointwise family: the
// nine codes of spec/pointwise_ops.md -- scale, scale-add, gate-mul, relu^2,
// softcap, exp, rsqrt, log, and the row broadcast -- forward and backward. Tiny
// shapes only. See docs/testing.md.
//
// The forward gate asserts the code's *declared absolute bound* and never bit
// equality (spec/pointwise_ops.md section 4.1): a device `expf` and a host
// `std::exp` are both correct and disagree in the last place, so only an
// absolute bound states what a downstream sum receives. Each code is fed an
// operand set inside its own envelope, which is the second half of the
// declaration.

#include <cmath>
#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/row_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::PointwiseOp;
using nanochat::PointwiseParams;
using nanochat::dev::Check;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::rowref::PointwiseBackward;
using nanochat::dev::rowref::PointwiseForward;

constexpr float kAlpha = 1.5f;
constexpr float kBeta = 0.5f;

// The declared bounds of spec/pointwise_ops.md section 4, one constant per code
// so a call site cannot quietly use another code's number.
constexpr double kScaleBound = 8e-6;
constexpr double kScaleAddBound = 2e-5;
constexpr double kGateMulBound = 4e-6;  // bounded, not bit-exact
constexpr double kReluSquareBound = 6e-5;
constexpr double kSoftcapBound = 8e-6;
constexpr double kExpBound = 3e-7;
constexpr double kRsqrtBound = 1e-6;
constexpr double kLogBound = 2e-5;
constexpr double kRowScaleBound = 8e-6;

// The fp32 oracle gate for a gradient, scaled by (1 + |want|). The spec's bound
// is stated for the code's output, so the gradient is compared at the tier's
// ordinary fp32 gate (docs/testing.md) against a double-precision reference.
constexpr double kGradTol = 1e-4;

PointwiseParams Make(PointwiseOp op, int count, int row_width = 0) {
  PointwiseParams p;
  p.op = op;
  p.count = count;
  p.row_width = row_width;
  p.alpha = kAlpha;
  p.beta = kBeta;
  return p;
}

// `b` is read by the codes that name it, and it is the dense operand of the row
// broadcast. `alpha`/`beta` are set on every call and ignored by the codes that
// do not read them, which is what section 3 of the spec says they may be.
bool UsesB(PointwiseOp op) {
  return op == PointwiseOp::kScaleAdd || op == PointwiseOp::kGateMul ||
         op == PointwiseOp::kRowScale;
}

// The number of entries in `a`: one per row for the broadcast, one per element
// otherwise, and the number of entries in `da` for the same reason.
int ACount(const PointwiseParams& p) {
  return p.op == PointwiseOp::kRowScale ? p.count / p.row_width : p.count;
}

// Assert the code's declared absolute bound, element by element.
// `CheckVectorClose` scales its tolerance by (1 + |want|), which is the right
// gate for a relative fp32 comparison and the wrong one for an absolute bound
// near zero, so the bound is asserted directly here.
void CheckWithinBound(const std::vector<float>& got,
                      const std::vector<float>& want, double bound,
                      const char* what) {
  if (got.size() != want.size()) {
    std::printf("FAIL: %s (size %zu vs %zu)\n", what, got.size(), want.size());
    ++Failures();
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(got[i]) - want[i]);
    if (diff > worst) worst = diff;
    if (diff > bound) {
      std::printf("FAIL: %s [%zu] got %.9g want %.9g |err| %.3g > bound %.3g\n",
                  what, i, got[i], want[i], diff, bound);
      ++Failures();
      return;
    }
  }
  std::printf("  %-24s max abs error %.3g (bound %.3g)\n", what, worst, bound);
}

// One code end to end: the forward against its declared bound, the backward
// against the double-precision host reference, and a central finite difference
// of that reference forward.
void TestOp(const PointwiseParams& p, const char* name,
            const std::vector<float>& a0, const std::vector<float>& b0,
            const std::vector<float>& w, double bound) {
  const int n = p.count;
  const int na = ACount(p);

  DevBuf<ComputeType> a(ToStorage(a0));
  DevBuf<ComputeType> b(ToStorage(b0));
  DevBuf<ComputeType> out(n);
  nanochat::kernels::PointwiseForward(p, a.ptr, b.ptr, out.ptr);
  nanochat::kernels::Synchronize();
  const std::vector<float> ref_out = PointwiseForward(p, a0, b0);
  CheckWithinBound(FromStorage(out.Download()), ref_out, bound, name);

  DevBuf<ComputeType> dy(ToStorage(w));
  DevBuf<ComputeType> da(na);
  DevBuf<ComputeType> db(n);
  nanochat::kernels::PointwiseBackward(p, a.ptr, b.ptr, dy.ptr, da.ptr, db.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_da(na, 0.0f);
  std::vector<float> ref_db(n, 0.0f);
  PointwiseBackward(p, a0, b0, w, &ref_da, &ref_db);
  CheckVectorClose(FromStorage(da.Download()), ref_da, kGradTol, "da");
  CheckVectorClose(FromStorage(db.Download()), ref_db, kGradTol, "db");

  auto loss_at = [&](const std::vector<float>& av,
                     const std::vector<float>& bv) {
    const std::vector<float> o = PointwiseForward(p, av, bv);
    double loss = 0.0;
    for (int i = 0; i < n; ++i) loss += static_cast<double>(o[i]) * w[i];
    return loss;
  };
  CheckFiniteDifference(
      a0, ref_da, [&](const std::vector<float>& av) { return loss_at(av, b0); },
      2e-3, "da");
  if (UsesB(p.op)) {
    CheckFiniteDifference(
        b0, ref_db,
        [&](const std::vector<float>& bv) { return loss_at(a0, bv); }, 2e-3,
        "db");
  }
}

// A signed operand set in [-1, 1], which is inside the envelope of every code
// whose operand may be negative.
std::vector<float> Signed(int n, Rng* rng) { return RandomVec(n, rng); }

// Scale a signed set into [lo, hi].
std::vector<float> Ranged(const std::vector<float>& signed_set, float lo,
                          float hi) {
  std::vector<float> out(signed_set.size());
  const float mid = 0.5f * (lo + hi);
  const float half = 0.5f * (hi - lo);
  for (std::size_t i = 0; i < signed_set.size(); ++i) {
    out[i] = mid + half * signed_set[i];
  }
  return out;
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("pointwise gpu test\n");

  Rng rng;
  const int n = 8;
  const std::vector<float> a0 = Signed(n, &rng);
  const std::vector<float> b0 = Signed(n, &rng);
  const std::vector<float> w = Signed(n, &rng);

  // --- The five arithmetic/softcap codes ---------------------------------
  // Every one stays inside its declared envelope with `a` and `b` in [-1, 1].
  TestOp(Make(PointwiseOp::kScale, n), "scale", a0, b0, w, kScaleBound);
  TestOp(Make(PointwiseOp::kScaleAdd, n), "scale-add", a0, b0, w,
         kScaleAddBound);
  TestOp(Make(PointwiseOp::kGateMul, n), "gate-mul", a0, b0, w, kGateMulBound);
  TestOp(Make(PointwiseOp::kReluSquare, n), "relu-square", a0, b0, w,
         kReluSquareBound);
  TestOp(Make(PointwiseOp::kSoftcap, n), "softcap", a0, b0, w, kSoftcapBound);

  // --- The three transcendental codes ------------------------------------
  // Each needs an operand from its own declared envelope: `kExp` reads
  // a in [-88, 0] so its output stays at most one, `kRsqrt` reads a in
  // [0.25, 4], and `kLog` reads a in (0, 1e6]. A signed operand would leave two
  // of the three domains entirely (a negative `a` has no real log or rsqrt), so
  // each gets a mapped set instead.
  const std::vector<float> exp_a = Ranged(a0, -2.0f, 0.0f);
  const std::vector<float> positive_a = Ranged(a0, 0.4f, 2.4f);
  TestOp(Make(PointwiseOp::kExp, n), "exp", exp_a, b0, w, kExpBound);
  TestOp(Make(PointwiseOp::kRsqrt, n), "rsqrt", positive_a, b0, w, kRsqrtBound);
  TestOp(Make(PointwiseOp::kLog, n), "log", positive_a, b0, w, kLogBound);

  // --- kRowScale: the row broadcast --------------------------------------
  // The value-gate shape of spec/pointwise_ops.md section 3.1: B=2, T=8, two KV
  // heads, head_dim 4 gives 32 dense rows of width 4 over 16 row scalars, so
  // `a` holds `count / row_width` values and `b`/`out` hold `count`.
  {
    const int rows = 2 * 8 * 2;
    const int width = 4;
    const int count = rows * width;
    const std::vector<float> scalars = Ranged(Signed(rows, &rng), -4.0f, 4.0f);
    const std::vector<float> dense = Ranged(Signed(count, &rng), -8.0f, 8.0f);
    const std::vector<float> dyv = Ranged(Signed(count, &rng), -1.0f, 1.0f);
    const PointwiseParams p = Make(PointwiseOp::kRowScale, count, width);
    TestOp(p, "row-scale", scalars, dense, dyv, kRowScaleBound);

    DevBuf<ComputeType> a(ToStorage(scalars));
    DevBuf<ComputeType> b(ToStorage(dense));
    DevBuf<ComputeType> out(count);
    nanochat::kernels::PointwiseForward(p, a.ptr, b.ptr, out.ptr);
    nanochat::kernels::Synchronize();
    const std::vector<float> want = FromStorage(out.Download());

    // The code reads no scalar of the parameter block: its field list is
    // `row_width` alone (spec/pointwise_ops.md section 3), so a stale `alpha`
    // and `beta` must not move the answer.
    PointwiseParams stale = p;
    stale.alpha = 3.5f;
    stale.beta = -2.0f;
    DevBuf<ComputeType> stale_out(count);
    nanochat::kernels::PointwiseForward(stale, a.ptr, b.ptr, stale_out.ptr);
    nanochat::kernels::Synchronize();
    CheckWithinBound(FromStorage(stale_out.Download()), want, 0.0,
                     "row-scale ignores alpha");

    // `out` may alias the dense operand (section 2): the residual accumulation
    // and every composition reuse their buffers in place, so the same call
    // through one buffer must give the same answer.
    DevBuf<ComputeType> aliased(ToStorage(dense));
    nanochat::kernels::PointwiseForward(p, a.ptr, aliased.ptr, aliased.ptr);
    nanochat::kernels::Synchronize();
    CheckWithinBound(FromStorage(aliased.Download()), want, 0.0,
                     "row-scale aliased out");

    // A width of 0, a width that exceeds the count, and a count that is not a
    // multiple of it are all declared errors (section 3.1). The seam has no
    // error return, so the rejection is a message naming the count and the
    // width plus no write at all: the buffer must come back untouched rather
    // than holding a plausible wrong row.
    DevBuf<ComputeType> dy(ToStorage(dyv));
    for (int bad_width : {0, 5, 7}) {
      PointwiseParams reject = p;
      reject.row_width = bad_width;
      DevBuf<ComputeType> untouched(
          ToStorage(std::vector<float>(count, 42.0f)));
      nanochat::kernels::PointwiseForward(reject, a.ptr, b.ptr, untouched.ptr);
      nanochat::kernels::Synchronize();
      bool unchanged = true;
      for (float v : FromStorage(untouched.Download())) {
        if (v != 42.0f) unchanged = false;
      }
      Check(unchanged, "row-scale rejects a bad width without writing");

      DevBuf<ComputeType> da_reject(ToStorage(std::vector<float>(rows, 42.0f)));
      DevBuf<ComputeType> db_reject(
          ToStorage(std::vector<float>(count, 42.0f)));
      nanochat::kernels::PointwiseBackward(reject, a.ptr, b.ptr, dy.ptr,
                                           da_reject.ptr, db_reject.ptr);
      nanochat::kernels::Synchronize();
      unchanged = true;
      for (float v : FromStorage(da_reject.Download())) {
        if (v != 42.0f) unchanged = false;
      }
      for (float v : FromStorage(db_reject.Download())) {
        if (v != 42.0f) unchanged = false;
      }
      Check(unchanged, "row-scale backward rejects a bad width");
    }
  }

  if (Failures() != 0) {
    std::printf("pointwise: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("pointwise: all checks passed\n");
  return 0;
}
