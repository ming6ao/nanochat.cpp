// tests/kernel_emu_test.cc -- the emulation engine's acceptance check
// (docs/simulator.md sections 9.3 and 12.3, invariant 13).
//
// The test links two implementations of the same seam entry points:
//
//   * `nanochat::kernels::RmsNormForward` and friends from `//backends/cpu`,
//     the reference loops;
//   * `nanochat::kernels::EmuRmsNormForward` and friends from
//     `//tools/cuda_sim/emu:emulated_kernels`, the *real* device kernels from
//     `backends/cuda/kernels/{rms_norm,pointwise}.cu`, compiled by the host
//     compiler against the emulator's CUDA stubs.
//
// Comparing the two validates the emulator and the device code together: the
// device body is unchanged, so a mismatch is either an emulation defect or a
// genuine divergence between the device arithmetic and the reference. The
// reference accumulates in double and the device in float, so the comparison is
// tolerance-based. The atol/rtol split below is local to this test;
// docs/testing.md records only the backend's overall tolerance, not a split.
//
// It also covers the two things the first cut missed: the warp shuffle path
// (the attention `WarpMax`/`WarpSum` pattern) and the fused-residual RmsNorm
// exports, which have no seam entry point but are part of the device family.
//
// CPU only: no GPU, no driver, no toolkit. The launch geometry is chosen so
// that the block-wide reductions, the grid stride, and the multi-block path are
// all exercised.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "tools/cuda_sim/emu/cuda_emu.h"
#include "tools/cuda_sim/emu/emu_kernels.h"

namespace {

using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::PointwiseOp;
using nanochat::RmsNormParams;
using nanochat::cuda_kernels::EmuRmsNormBackwardFusedResidual;
using nanochat::cuda_kernels::EmuRmsNormForwardFusedResidual;
using nanochat::kernels::EmuPointwiseBackward;
using nanochat::kernels::EmuPointwiseForward;
using nanochat::kernels::EmuRmsNormBackward;
using nanochat::kernels::EmuRmsNormForward;
using nanochat::kernels::PointwiseBackward;
using nanochat::kernels::PointwiseForward;
using nanochat::kernels::RmsNormBackward;
using nanochat::kernels::RmsNormForward;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

// The tolerance for this comparison. docs/testing.md records the backend's
// overall tolerance (fp32 tight, fp16 looser because storage quantizes to
// half); the atol/rtol split here is the test's own.
double Atol() {
#if defined(NANOCHAT_PRECISION_FP16)
  return 2e-3;
#else
  return 1e-5;
#endif
}

double Rtol() {
#if defined(NANOCHAT_PRECISION_FP16)
  return 2e-3;
#else
  return 1e-4;
#endif
}

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want,
                 const std::string& what) {
  if (got.size() != want.size()) {
    Fail(Format("%s: size %zu vs %zu", what.c_str(), got.size(), want.size()));
    return;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double limit =
        Atol() +
        Rtol() * std::max<double>({std::fabs(got[i]), std::fabs(want[i])});
    if (!(std::fabs(static_cast<double>(got[i]) -
                    static_cast<double>(want[i])) <= limit)) {
      Fail(Format("%s[%zu]: emulated %.9g vs reference %.9g", what.c_str(), i,
                  got[i], want[i]));
      return;
    }
  }
}

// A deterministic, well-conditioned input. Large enough that the fp16 build
// does not saturate, and varied enough that a wrong reduction order shows up.
std::vector<float> MakeInput(std::size_t count, float scale, int seed) {
  std::vector<float> out(count);
  std::uint32_t state = static_cast<std::uint32_t>(seed) * 2654435761u + 1u;
  for (std::size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit =
        static_cast<float>((state >> 8) & 0xffffu) / 65535.0f - 0.5f;
    out[i] = unit * scale;
  }
  return out;
}

ComputeType ToStorage(float value) {
#if defined(NANOCHAT_PRECISION_FP16)
  return nanochat::Fp16FromFloat(value);
#else
  return value;
#endif
}

float ToFloat(ComputeType value) {
#if defined(NANOCHAT_PRECISION_FP16)
  return nanochat::Fp16ToFloat(value);
#else
  return value;
#endif
}

std::vector<ComputeType> ToStorage(const std::vector<float>& values) {
  std::vector<ComputeType> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) out[i] = ToStorage(values[i]);
  return out;
}

std::vector<float> ToFloat(const std::vector<ComputeType>& values) {
  std::vector<float> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) out[i] = ToFloat(values[i]);
  return out;
}

void CheckRmsNormForward(int rows, int dim) {
  const RmsNormParams params{rows, dim, 1e-5f};
  const std::size_t count = static_cast<std::size_t>(rows) * dim;
  const std::vector<ComputeType> input =
      ToStorage(MakeInput(count, 2.0f, 11 + rows * 100 + dim));

  std::vector<ComputeType> reference_out(count);
  std::vector<ComputeType> emulated_out(count);
  std::vector<float> reference_rstd(static_cast<std::size_t>(rows));
  std::vector<float> emulated_rstd(static_cast<std::size_t>(rows));
  RmsNormForward(params, input.data(), reference_out.data(),
                 reference_rstd.data());
  EmuRmsNormForward(params, input.data(), emulated_out.data(),
                    emulated_rstd.data());

  const std::string what = Format("rms_norm forward rows=%d dim=%d", rows, dim);
  ExpectClose(ToFloat(emulated_out), ToFloat(reference_out), what);
  ExpectClose(emulated_rstd, reference_rstd, what + " rstd");
}

void CheckRmsNormBackward(int rows, int dim) {
  const RmsNormParams params{rows, dim, 1e-5f};
  const std::size_t count = static_cast<std::size_t>(rows) * dim;
  const std::vector<ComputeType> x =
      ToStorage(MakeInput(count, 2.0f, 23 + rows * 100 + dim));
  const std::vector<ComputeType> dy =
      ToStorage(MakeInput(count, 0.75f, 41 + rows * 100 + dim));

  // The saved statistic is the forward's, so it is computed once by the
  // reference and handed to both backward paths.
  std::vector<float> rstd(static_cast<std::size_t>(rows));
  std::vector<ComputeType> forward_out(count);
  RmsNormForward(params, x.data(), forward_out.data(), rstd.data());

  std::vector<ComputeType> reference = dy;
  std::vector<ComputeType> emulated = dy;
  RmsNormBackward(params, x.data(), dy.data(), rstd.data(), reference.data());
  EmuRmsNormBackward(params, x.data(), dy.data(), rstd.data(), emulated.data());

  ExpectClose(ToFloat(emulated), ToFloat(reference),
              Format("rms_norm backward rows=%d dim=%d", rows, dim));
}

void CheckPointwiseForward(PointwiseOp op, int n) {
  const std::vector<ComputeType> a =
      ToStorage(MakeInput(static_cast<std::size_t>(n), 3.0f, 7));
  const std::vector<ComputeType> b =
      ToStorage(MakeInput(static_cast<std::size_t>(n), 1.5f, 9));
  const float alpha = 0.75f;
  const float beta = 1.25f;

  std::vector<ComputeType> reference(static_cast<std::size_t>(n));
  std::vector<ComputeType> emulated(static_cast<std::size_t>(n));
  PointwiseForward(op, n, a.data(), b.data(), alpha, beta, reference.data());
  EmuPointwiseForward(op, n, a.data(), b.data(), alpha, beta, emulated.data());

  ExpectClose(ToFloat(emulated), ToFloat(reference),
              Format("pointwise forward op=%d n=%d", static_cast<int>(op), n));
}

void CheckPointwiseBackward(PointwiseOp op, int n) {
  const std::vector<ComputeType> a =
      ToStorage(MakeInput(static_cast<std::size_t>(n), 3.0f, 13));
  const std::vector<ComputeType> b =
      ToStorage(MakeInput(static_cast<std::size_t>(n), 1.5f, 17));
  const std::vector<ComputeType> dy =
      ToStorage(MakeInput(static_cast<std::size_t>(n), 0.5f, 19));
  const float alpha = 0.75f;
  const float beta = 1.25f;

  // The backward entry point accepts a null gradient pointer for either operand
  // (the device kernel checks `da != nullptr` / `db != nullptr`). Assert the
  // null-output contract by comparing whichever outputs are requested in each
  // of the three configurations: both, `da` only, and `db` only.
  for (int config = 0; config < 3; ++config) {
    const bool want_da = config != 2;
    const bool want_db = config != 1;
    std::vector<ComputeType> reference_da(static_cast<std::size_t>(n));
    std::vector<ComputeType> reference_db(static_cast<std::size_t>(n));
    std::vector<ComputeType> emulated_da(static_cast<std::size_t>(n));
    std::vector<ComputeType> emulated_db(static_cast<std::size_t>(n));
    ComputeType* ref_a = want_da ? reference_da.data() : nullptr;
    ComputeType* ref_b = want_db ? reference_db.data() : nullptr;
    ComputeType* emu_a = want_da ? emulated_da.data() : nullptr;
    ComputeType* emu_b = want_db ? emulated_db.data() : nullptr;
    PointwiseBackward(op, n, a.data(), b.data(), dy.data(), alpha, beta, ref_a,
                      ref_b);
    EmuPointwiseBackward(op, n, a.data(), b.data(), dy.data(), alpha, beta,
                         emu_a, emu_b);
    if (want_da) {
      ExpectClose(ToFloat(emulated_da), ToFloat(reference_da),
                  Format("pointwise backward op=%d n=%d da config=%d",
                         static_cast<int>(op), n, config));
    }
    if (want_db) {
      ExpectClose(ToFloat(emulated_db), ToFloat(reference_db),
                  Format("pointwise backward op=%d n=%d db config=%d",
                         static_cast<int>(op), n, config));
    }
  }
}

// The emulator's warp shuffle, on the exact five-exchange pattern
// attention.cu's `WarpMax`/`WarpSum` use (`__shfl_xor_sync` for offsets 1, 2,
// 4, 8, 16). The second warp barrier in `Shfl` is what makes sequential
// exchanges safe: without it a released lane can overwrite its slot before a
// peer reads it.
__global__ void WarpMaxKernel(const float* in, float* out, int n) {
  const int lane = threadIdx.x;
  float value = lane < n ? in[lane] : -3.0e38f;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = std::fmax(value, other);
  }
  out[lane] = value;
}

void CheckWarpShuffles() {
  std::vector<float> in(32);
  std::vector<float> out(32, 0.0f);
  for (int i = 0; i < 32; ++i) {
    in[static_cast<std::size_t>(i)] =
        3.0f * std::sin(static_cast<float>(i) * 0.7f) - 0.25f;
  }
  const float want = *std::max_element(in.begin(), in.end());
  nanochat::emu::LaunchEmulatedKernel(WarpMaxKernel, dim3(1), dim3(32), 0,
                                      in.data(), out.data(), 32);
  for (int lane = 0; lane < 32; ++lane) {
    ExpectClose({out[static_cast<std::size_t>(lane)]}, {want},
                Format("warp shuffle lane %d", lane));
  }
}

// The fused-residual RmsNorm exports are private to the RmsNorm family (they
// are not in the frozen seam), so an independent host reference stands in for
// the CPU kernel. Semantics: residual_out = x + residual_in; out = rmsnorm of
// that.
void CheckRmsNormFusedResidualForward(int rows, int dim) {
  const RmsNormParams params{rows, dim, 1e-5f};
  const std::size_t count = static_cast<std::size_t>(rows) * dim;
  const std::vector<ComputeType> x =
      ToStorage(MakeInput(count, 2.0f, 101 + rows * 100 + dim));
  const std::vector<ComputeType> incoming =
      ToStorage(MakeInput(count, 1.5f, 211 + rows * 100 + dim));

  std::vector<ComputeType> residual = incoming;
  std::vector<ComputeType> out(count);
  std::vector<float> rstd(static_cast<std::size_t>(rows));
  EmuRmsNormForwardFusedResidual(params, x.data(), residual.data(), out.data(),
                                 rstd.data());

  std::vector<float> want_residual(count);
  std::vector<float> want_out(count);
  std::vector<float> want_rstd(static_cast<std::size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const std::size_t i = static_cast<std::size_t>(row) * dim + d;
      const double v = static_cast<double>(ToFloat(x[i])) +
                       static_cast<double>(ToFloat(incoming[i]));
      want_residual[i] = static_cast<float>(v);
      sum_sq += v * v;
    }
    const double r = 1.0 / std::sqrt(sum_sq / dim + 1e-5);
    want_rstd[static_cast<std::size_t>(row)] = static_cast<float>(r);
    for (int d = 0; d < dim; ++d) {
      const std::size_t i = static_cast<std::size_t>(row) * dim + d;
      want_out[i] = static_cast<float>(want_residual[i] * r);
    }
  }
  const std::string what =
      Format("rms_norm fused residual forward rows=%d dim=%d", rows, dim);
  ExpectClose(ToFloat(residual), want_residual, what + " residual");
  ExpectClose(ToFloat(out), want_out, what + " out");
  ExpectClose(rstd, want_rstd, what + " rstd");
}

void CheckRmsNormFusedResidualBackward(int rows, int dim) {
  const RmsNormParams params{rows, dim, 1e-5f};
  const std::size_t count = static_cast<std::size_t>(rows) * dim;
  const std::vector<ComputeType> x =
      ToStorage(MakeInput(count, 2.0f, 307 + rows * 100 + dim));
  const std::vector<ComputeType> incoming =
      ToStorage(MakeInput(count, 1.5f, 401 + rows * 100 + dim));
  const std::vector<ComputeType> dy =
      ToStorage(MakeInput(count, 0.75f, 503 + rows * 100 + dim));

  std::vector<ComputeType> residual = incoming;
  std::vector<ComputeType> out(count);
  std::vector<float> rstd(static_cast<std::size_t>(rows));
  EmuRmsNormForwardFusedResidual(params, x.data(), residual.data(), out.data(),
                                 rstd.data());

  std::vector<ComputeType> dx(count);
  std::vector<ComputeType> dresidual(count);
  EmuRmsNormBackwardFusedResidual(params, residual.data(), rstd.data(),
                                  dy.data(), dx.data(), dresidual.data());

  // Both gradients receive the same value: g = r*dy - (r^3*dot(v,dy)/dim)*v.
  std::vector<float> want(count);
  for (int row = 0; row < rows; ++row) {
    double dot = 0.0;
    for (int d = 0; d < dim; ++d) {
      const std::size_t i = static_cast<std::size_t>(row) * dim + d;
      dot += static_cast<double>(ToFloat(residual[i])) *
             static_cast<double>(ToFloat(dy[i]));
    }
    const double r = rstd[static_cast<std::size_t>(row)];
    const double coeff = r * r * r * dot / dim;
    for (int d = 0; d < dim; ++d) {
      const std::size_t i = static_cast<std::size_t>(row) * dim + d;
      want[i] =
          static_cast<float>(r * static_cast<double>(ToFloat(dy[i])) -
                             coeff * static_cast<double>(ToFloat(residual[i])));
    }
  }
  const std::string what =
      Format("rms_norm fused residual backward rows=%d dim=%d", rows, dim);
  ExpectClose(ToFloat(dx), want, what + " dx");
  ExpectClose(ToFloat(dresidual), want, what + " dresidual");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  // One block per row, several blocks in the grid.
  CheckRmsNormForward(1, 4);
  CheckRmsNormForward(4, 16);
  CheckRmsNormForward(3, 128);
  CheckRmsNormForward(2, 256);
  CheckRmsNormBackward(1, 4);
  CheckRmsNormBackward(4, 16);
  CheckRmsNormBackward(3, 128);

  // A single block for a short vector, and a grid-strided walk for a long one.
  for (PointwiseOp op :
       {PointwiseOp::kScale, PointwiseOp::kScaleAdd, PointwiseOp::kGateMul,
        PointwiseOp::kReluSquare, PointwiseOp::kSoftcap}) {
    CheckPointwiseForward(op, 7);
    CheckPointwiseForward(op, 64);
    CheckPointwiseForward(op, 1000);
    CheckPointwiseBackward(op, 7);
    CheckPointwiseBackward(op, 64);
    CheckPointwiseBackward(op, 1000);
  }

  // The warp shuffle path (the attention WarpMax/WarpSum pattern) and the
  // fused-residual RmsNorm exports.
  CheckWarpShuffles();
  CheckRmsNormFusedResidualForward(1, 4);
  CheckRmsNormFusedResidualForward(3, 128);
  CheckRmsNormFusedResidualBackward(2, 16);
  CheckRmsNormFusedResidualBackward(3, 128);

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  const nanochat::emu::LaunchStats& stats = nanochat::emu::Stats();
  std::printf(
      "kernel emu: all checks passed (precision=%s, %lld launches, "
      "max block %zu)\n",
#if defined(NANOCHAT_PRECISION_FP16)
      "fp16",
#else
      "fp32",
#endif
      static_cast<long long>(stats.launches), stats.max_block_threads);
  return 0;
}
