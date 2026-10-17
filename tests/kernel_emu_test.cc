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
// overall tolerance; the atol/rtol split here is the test's own.
//
// The emulated kernels store `ComputeType`, so a bf16 build rounds every result
// to 7 mantissa bits while the fp64 host reference below does not. The bounds
// therefore carry that storage granularity in the bf16 build.
#if defined(NANOCHAT_PRECISION_FP32)
double Atol() { return 1e-5; }

double Rtol() { return 1e-4; }
#else
double Atol() { return 0.01; }

double Rtol() { return 0.02; }
#endif

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

// A deterministic, well-conditioned input. Large enough that the comparison
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

ComputeType ToStorage(float value) { return value; }

float ToFloat(ComputeType value) { return value; }

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

// The emulated fp8 cast against the CPU reference. Both use the shared rounding
// header, so the bytes and the dequantization value must match exactly
// (docs/simulator-precision.md invariants 2 and 13).
void CheckQuantizeFp8() {
  constexpr int kCount = 64;
  const std::vector<float> source =
      MakeInput(static_cast<std::size_t>(kCount), 2.0f, 7);
  std::vector<ComputeType> input(static_cast<std::size_t>(kCount));
  for (int i = 0; i < kCount; ++i) {
    input[static_cast<std::size_t>(i)] =
        nanochat::ComputeFromFloat(source[static_cast<std::size_t>(i)]);
  }
  for (nanochat::Fp8Format format :
       {nanochat::Fp8Format::kE4M3, nanochat::Fp8Format::kE5M2}) {
    std::vector<std::uint8_t> reference(static_cast<std::size_t>(kCount), 0);
    std::vector<std::uint8_t> emulated(static_cast<std::size_t>(kCount), 0);
    float reference_dequant = 0.0f;
    float emulated_dequant = 0.0f;
    nanochat::QuantizeFp8Params params;
    params.count = kCount;
    params.format = format;
    nanochat::kernels::QuantizeFp8(params, input.data(), reference.data(),
                                   &reference_dequant);
    nanochat::kernels::EmuQuantizeFp8(params, input.data(), emulated.data(),
                                      &emulated_dequant);
    ExpectClose({reference_dequant}, {emulated_dequant}, "fp8 dequantization");
    for (int i = 0; i < kCount; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      if (reference[index] != emulated[index]) {
        Fail(Format("fp8 byte %d differs: emulated 0x%02X vs reference 0x%02X",
                    i, emulated[index], reference[index]));
        return;
      }
    }
  }
}

// The value gate: the emulated kernel against the CPU reference. This is the
// `extern __shared__` path the executed-source rewrite covers
// (docs/simulator-precision.md section 6.2).
void CheckValueGate(int rows, int kv_heads, int head_dim) {
  const int hidden = 32;
  const int gate_channels = 12;
  const int value_count = rows * kv_heads * head_dim;
  const int gate_count = rows * kv_heads;
  const int h_count = rows * hidden;
  const int w_count = kv_heads * gate_channels;

  const std::vector<float> h_f = MakeInput(h_count, 1.0f, 11);
  const std::vector<float> ve_f = MakeInput(value_count, 1.0f, 13);
  const std::vector<float> w_f = MakeInput(w_count, 1.0f, 17);
  std::vector<ComputeType> h(h_count), ve(value_count), w(w_count);
  for (int i = 0; i < h_count; ++i) h[i] = nanochat::ComputeFromFloat(h_f[i]);
  for (int i = 0; i < value_count; ++i) {
    ve[i] = nanochat::ComputeFromFloat(ve_f[i]);
  }
  for (int i = 0; i < w_count; ++i) w[i] = nanochat::ComputeFromFloat(w_f[i]);

  const ComputeType zero = nanochat::ComputeFromFloat(0.0f);
  std::vector<ComputeType> v_ref(value_count, zero);
  std::vector<ComputeType> gate_ref(gate_count, zero);
  std::vector<ComputeType> v_emu(value_count, zero);
  std::vector<ComputeType> gate_emu(gate_count, zero);
  nanochat::kernels::ValueGateForward(rows, hidden, kv_heads, head_dim,
                                      h.data(), ve.data(), w.data(),
                                      v_ref.data(), gate_ref.data());
  nanochat::kernels::EmuValueGateForward(rows, hidden, kv_heads, head_dim,
                                         h.data(), ve.data(), w.data(),
                                         v_emu.data(), gate_emu.data());
  ExpectClose(ToFloat(v_ref), ToFloat(v_emu), "value gate forward v");
  ExpectClose(ToFloat(gate_ref), ToFloat(gate_emu), "value gate forward gate");

  // The backward. `gate_w_grad` and `dh` accumulate, so they start at zero.
  const std::vector<float> dv_f = MakeInput(value_count, 1.0f, 19);
  std::vector<ComputeType> dv(value_count);
  for (int i = 0; i < value_count; ++i) {
    dv[i] = nanochat::ComputeFromFloat(dv_f[i]);
  }
  std::vector<float> wgrad_ref(w_count, 0.0f);
  std::vector<float> wgrad_emu(w_count, 0.0f);
  std::vector<ComputeType> dh_ref(h_count, zero);
  std::vector<ComputeType> dh_emu(h_count, zero);
  std::vector<ComputeType> dve_ref(value_count, zero);
  std::vector<ComputeType> dve_emu(value_count, zero);
  nanochat::kernels::ValueGateBackward(
      rows, hidden, kv_heads, head_dim, h.data(), ve.data(), w.data(),
      gate_ref.data(), dv.data(), wgrad_ref.data(), dh_ref.data(),
      dve_ref.data());
  nanochat::kernels::EmuValueGateBackward(
      rows, hidden, kv_heads, head_dim, h.data(), ve.data(), w.data(),
      gate_ref.data(), dv.data(), wgrad_emu.data(), dh_emu.data(),
      dve_emu.data());
  ExpectClose(wgrad_ref, wgrad_emu, "value gate backward gate_w_grad");
  ExpectClose(ToFloat(dh_ref), ToFloat(dh_emu), "value gate backward dh");
  ExpectClose(ToFloat(dve_ref), ToFloat(dve_emu), "value gate backward dve");
}

// Attention: the emulated kernels against the CPU reference. The fused
// windowed forward uses `extern __shared__`, which the executed-source rewrite
// covers (docs/simulator-precision.md section 6.2).
void CheckAttention(const nanochat::AttentionParams& params) {
  const int dim = params.head_dim;
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int q_count = params.batch * params.seq * params.num_heads * dim;
  const int k_count = params.batch * kv_len * params.num_kv_heads * dim;
  const int stats_count = params.batch * params.num_heads * params.seq * 2;

  const std::vector<float> q_f = MakeInput(q_count, 1.0f, 23);
  const std::vector<float> k_f = MakeInput(k_count, 1.0f, 29);
  const std::vector<float> v_f = MakeInput(k_count, 1.0f, 31);
  std::vector<ComputeType> q(q_count), k(k_count), v(k_count);
  for (int i = 0; i < q_count; ++i) q[i] = nanochat::ComputeFromFloat(q_f[i]);
  for (int i = 0; i < k_count; ++i) k[i] = nanochat::ComputeFromFloat(k_f[i]);
  for (int i = 0; i < k_count; ++i) v[i] = nanochat::ComputeFromFloat(v_f[i]);

  const ComputeType zero = nanochat::ComputeFromFloat(0.0f);
  std::vector<ComputeType> out_ref(q_count, zero);
  std::vector<ComputeType> out_emu(q_count, zero);
  std::vector<float> stats_ref(stats_count, 0.0f);
  std::vector<float> stats_emu(stats_count, 0.0f);
  nanochat::kernels::AttentionForward(params, q.data(), k.data(), v.data(),
                                      out_ref.data(), stats_ref.data());
  nanochat::kernels::EmuAttentionForward(params, q.data(), k.data(), v.data(),
                                         out_emu.data(), stats_emu.data());
  ExpectClose(ToFloat(out_ref), ToFloat(out_emu), "attention forward out");
  ExpectClose(stats_ref, stats_emu, "attention forward stats");
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

  // The fp8 cast: the emulated device code against the CPU reference.
  CheckQuantizeFp8();

  // The value gate, which uses `extern __shared__`.
  CheckValueGate(3, 2, 8);
  CheckValueGate(5, 4, 4);

  // Attention: the cuBLAS path, then the fused windowed path, which also uses
  // `extern __shared__`.
  {
    nanochat::AttentionParams params;
    params.batch = 2;
    params.seq = 8;
    params.num_heads = 2;
    params.num_kv_heads = 2;
    params.head_dim = 8;
    params.causal = true;
    CheckAttention(params);
  }
  {
    nanochat::AttentionParams params;
    params.batch = 1;
    params.seq = 8;
    params.num_heads = 1;
    params.num_kv_heads = 1;
    params.head_dim = 128;
    params.causal = true;
    params.window_left = 4;
    CheckAttention(params);
  }

  // The `extern __shared__` path ran: the dynamic shared buffer was sized.
  if (nanochat::emu::Stats().max_dynamic_shared_bytes == 0) {
    Fail("the dynamic shared-memory path was exercised");
  }

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  const nanochat::emu::LaunchStats& stats = nanochat::emu::Stats();
  std::printf(
      "kernel emu: all checks passed (precision=fp32, %lld launches, "
      "max block %zu)\n",
      static_cast<long long>(stats.launches), stats.max_block_threads);
  return 0;
}
