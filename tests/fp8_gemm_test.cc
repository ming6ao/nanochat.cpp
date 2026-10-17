// tests/fp8_gemm_test.cc -- the scaled fp8 seam gate
// (docs/correctness-gates.md section 3.3, docs/fp8-training.md sections 4-5).
//
// The test drives the two new seam entry points, `QuantizeFp8` and
// `GemmScaledFp8`, on the CPU reference backend and compares them against an
// independent reference:
//
//   1. `QuantizeFp8` derives the scale from the amax, floors it, and writes the
//      dequantization value.
//   2. The scaled GEMM matches a direct triple loop over the decoded operands,
//      for the forward, dgrad, and wgrad operand roles.
//   3. The `beta` accumulation and the mixed E5M2/E4M3 operands behave.
//
// It is CPU-only: tier S0, a plain T0 test. It runs under `--config=sim` too,
// where the same arithmetic is checked with the simulated capability table.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/fp8.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::ComputeType;
using nanochat::Fp8Format;
using nanochat::GemmMode;
using nanochat::QuantizeFp8Params;
using nanochat::ScaledGemmParams;

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

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

void ExpectNear(double actual, double expected, double tolerance,
                const std::string& what) {
  if (!(std::fabs(actual - expected) <= tolerance)) {
    Fail(Format("%s: got %.9g, want %.9g (tol %.3g)", what.c_str(), actual,
                expected, tolerance));
  }
}

// A deterministic generator, so the test has a fixed input and no dependency on
// a random library.
std::uint32_t g_state = 0x12345678u;
float NextFloat() {
  g_state = g_state * 1664525u + 1013904223u;
  return static_cast<float>((g_state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

float DecodeFp8(std::uint8_t bits, Fp8Format format) {
  return format == Fp8Format::kE4M3 ? nanochat::Fp8E4M3Decode(bits)
                                    : nanochat::Fp8E5M2Decode(bits);
}

// Logical op(A)[i, p] of a dense A. Stored [m, k] when `transpose` is false,
// and [k, m] when it is true.
double AValue(const std::uint8_t* a, int m, int k, bool transpose, int i, int p,
              Fp8Format format) {
  if (!transpose) return DecodeFp8(a[i * k + p], format);
  return DecodeFp8(a[p * m + i], format);
}

double BValue(const std::uint8_t* b, int k, int n, bool transpose, int p, int j,
              Fp8Format format) {
  if (!transpose) return DecodeFp8(b[p * n + j], format);
  return DecodeFp8(b[j * k + p], format);
}

void TestQuantize() {
  // A tensor whose amax is 0.5, so the E4M3 scale is 448 / 0.5 = 896.
  const int count = 8;
  std::vector<float> source = {0.5f, -0.5f, 0.25f, -0.125f,
                               0.0f, 0.1f,  -0.2f, 0.3f};
  std::vector<ComputeType> compute(count);
  for (int i = 0; i < count; ++i) {
    compute[i] = nanochat::ComputeFromFloat(source[i]);
  }
  std::vector<std::uint8_t> bytes(count);
  float dequant = 0.0f;
  QuantizeFp8Params params;
  params.count = count;
  params.format = Fp8Format::kE4M3;
  nanochat::kernels::QuantizeFp8(params, compute.data(), bytes.data(),
                                 &dequant);
  // The amax is 0.5, so the dequantization value is 0.5 / 448. The value is a
  // float, so the comparison is relative.
  ExpectNear(dequant, 0.5 / 448.0, 0.5 / 448.0 * 1e-6,
             "E4M3 dequantization value");
  // Every byte is the cast of the value at the derived scale, and the amax maps
  // to the format maximum.
  const float scale = 1.0f / dequant;
  for (int i = 0; i < count; ++i) {
    const std::uint8_t expected = nanochat::Fp8E4M3Encode(source[i] * scale);
    ExpectTrue(bytes[i] == expected,
               Format("quantize byte %d matches the cast", i));
  }
  ExpectTrue(bytes[0] == 0x7E || bytes[1] == 0xFE,
             "the amax saturates the format");

  // An all-zero tensor keeps the floor, so the scale is finite.
  std::vector<ComputeType> zeros(count, nanochat::ComputeFromFloat(0.0f));
  float zero_dequant = 0.0f;
  nanochat::kernels::QuantizeFp8(params, zeros.data(), bytes.data(),
                                 &zero_dequant);
  ExpectNear(zero_dequant, 1e-12 / 448.0, 1e-12 / 448.0 * 1e-6,
             "all-zero dequantization");
  for (int i = 0; i < count; ++i) {
    ExpectTrue(bytes[i] == 0, Format("all-zero byte %d is zero", i));
  }
}

// One scaled GEMM case, checked against a direct triple loop. The output type
// follows the mode: fp32 for the weight gradient, the compute type otherwise.
void CheckGemm(const std::string& name, GemmMode mode, int m, int n, int k,
               bool transpose_a, bool transpose_b, Fp8Format format_a,
               Fp8Format format_b, float beta) {
  const int a_size = m * k;
  const int b_size = k * n;

  std::vector<ComputeType> a_source(a_size);
  std::vector<ComputeType> b_source(b_size);
  for (int i = 0; i < a_size; ++i) {
    a_source[i] = nanochat::ComputeFromFloat(NextFloat());
  }
  for (int i = 0; i < b_size; ++i) {
    b_source[i] = nanochat::ComputeFromFloat(NextFloat());
  }

  std::vector<std::uint8_t> a_bytes(a_size);
  std::vector<std::uint8_t> b_bytes(b_size);
  float scale_a = 0.0f;
  float scale_b = 0.0f;
  QuantizeFp8Params params;
  params.count = a_size;
  params.format = format_a;
  nanochat::kernels::QuantizeFp8(params, a_source.data(), a_bytes.data(),
                                 &scale_a);
  params.count = b_size;
  params.format = format_b;
  nanochat::kernels::QuantizeFp8(params, b_source.data(), b_bytes.data(),
                                 &scale_b);

  const bool fp32_output = mode == GemmMode::kWgrad;
  std::vector<ComputeType> c_compute;
  std::vector<float> c_float;
  if (fp32_output) {
    c_float.assign(static_cast<std::size_t>(m) * n, 0.1f);
  } else {
    c_compute.assign(static_cast<std::size_t>(m) * n,
                     nanochat::ComputeFromFloat(0.1f));
  }
  const float alpha = 1.0f;
  ScaledGemmParams gemm;
  gemm.m = m;
  gemm.n = n;
  gemm.k = k;
  gemm.alpha = alpha;
  gemm.beta = beta;
  gemm.transpose_a = transpose_a;
  gemm.transpose_b = transpose_b;
  gemm.format_a = format_a;
  gemm.format_b = format_b;
  gemm.scale_a = &scale_a;
  gemm.scale_b = &scale_b;
  void* output = fp32_output ? static_cast<void*>(c_float.data())
                             : static_cast<void*>(c_compute.data());
  nanochat::kernels::GemmScaledFp8(mode, gemm, a_bytes.data(), b_bytes.data(),
                                   output);

  // The direct reference, in double.
  double worst = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p) {
        const double av =
            AValue(a_bytes.data(), m, k, transpose_a, i, p, format_a);
        const double bv =
            BValue(b_bytes.data(), k, n, transpose_b, p, j, format_b);
        acc += av * bv;
      }
      const std::size_t index = static_cast<std::size_t>(i) * n + j;
      const double got =
          fp32_output
              ? static_cast<double>(c_float[index])
              : static_cast<double>(nanochat::ComputeToFloat(c_compute[index]));
      const double expected = alpha * (acc * scale_a * scale_b) + beta * 0.1;
      const double error = std::fabs(got - expected);
      // A combined bound: the compute type stores the forward result
      // (docs/fp8-training.md section 7), so a bf16 build rounds the output and
      // a small result carries an absolute, not relative, error.
      const double relative = error / (std::fabs(expected) + 1.0);
      if (relative > worst) worst = relative;
    }
  }
  // The fp8 relative tolerance (docs/simulator-precision.md section 7).
  ExpectTrue(worst <= 1e-2, Format("%s relative error %.3g is within 1e-2",
                                   name.c_str(), worst));
}

void TestGemm() {
  // The forward role: A is the activation [m, k], B is the weight [n, k].
  CheckGemm("forward E4M3", GemmMode::kForward, 64, 96, 32,
            /*transpose_a=*/false, /*transpose_b=*/true, Fp8Format::kE4M3,
            Fp8Format::kE4M3, 0.0f);
  // The dgrad role: A is the output gradient [m, k], B is the weight [k, n].
  CheckGemm("dgrad E5M2/E4M3", GemmMode::kDgrad, 64, 32, 96,
            /*transpose_a=*/false, /*transpose_b=*/false, Fp8Format::kE5M2,
            Fp8Format::kE4M3, 0.0f);
  // The wgrad role: A is the output gradient [k, m] (transposed), B is the
  // activation [k, n]. The output is fp32, so it also tests the accumulation.
  CheckGemm("wgrad E5M2/E4M3", GemmMode::kWgrad, 96, 32, 64,
            /*transpose_a=*/true, /*transpose_b=*/false, Fp8Format::kE5M2,
            Fp8Format::kE4M3, 0.5f);
  // The accumulation form: beta is not zero.
  CheckGemm("forward with beta", GemmMode::kForward, 64, 96, 32,
            /*transpose_a=*/false, /*transpose_b=*/true, Fp8Format::kE4M3,
            Fp8Format::kE4M3, 0.5f);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestQuantize();
  TestGemm();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("fp8 gemm: all checks passed\n");
  return 0;
}
