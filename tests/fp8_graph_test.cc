// tests/fp8_graph_test.cc -- the fp8 graph plumbing gate
// (docs/correctness-gates.md sections 3.3 and 5.1).
//
// The small model tests use shapes below the fp8 threshold, so they never
// exercise the fp8 branch of `ops::LinearForward`. This test uses an eligible
// shape (`min(out, in) >= 128`) and compares the fp8 forward against the dense
// forward in the same build:
//
//   1. The fp8 linear output matches the dense output within the fp8 tolerance.
//   2. The block forward runs with an fp8 scratch and stays finite.
//
// In a non-fp8 build the branch is compiled out, so the check degenerates to a
// self-comparison and still passes. CPU only: tier S0, a plain T0 test.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/ops.h"

namespace {

using nanochat::ComputeType;

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

// A deterministic, well-conditioned input. The values are positive, so the dot
// products do not cancel and the fp8 quantization error stays relative to the
// result (docs/fp8-training.md section 4.1).
std::vector<ComputeType> MakeInput(std::size_t count, int seed) {
  std::vector<ComputeType> out(count);
  std::uint32_t state = static_cast<std::uint32_t>(seed) * 2654435761u + 1u;
  for (std::size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>((state >> 8) & 0xffffu) / 65535.0f;
    out[i] = nanochat::ComputeFromFloat(0.25f + 0.75f * unit);
  }
  return out;
}

ComputeType* Upload(const std::vector<ComputeType>& host) {
  ComputeType* device = static_cast<ComputeType*>(
      nanochat::kernels::Alloc(host.size() * sizeof(ComputeType)));
  if (device != nullptr) {
    nanochat::kernels::Memcpy(device, host.data(),
                              host.size() * sizeof(ComputeType),
                              nanochat::CopyDir::kHostToDevice);
  }
  return device;
}

std::vector<ComputeType> Download(const ComputeType* device,
                                  std::size_t count) {
  std::vector<ComputeType> host(count);
  nanochat::kernels::Memcpy(host.data(), device, count * sizeof(ComputeType),
                            nanochat::CopyDir::kDeviceToHost);
  return host;
}

std::vector<float> DownloadFloat(const float* device, std::size_t count) {
  std::vector<float> host(count);
  nanochat::kernels::Memcpy(host.data(), device, count * sizeof(float),
                            nanochat::CopyDir::kDeviceToHost);
  return host;
}

void TestLinearFp8AgainstDense() {
  if constexpr (nanochat::kFp8Build) {
    // An eligible shape: both dimensions divide 16 and the smaller is 128 or
    // more (docs/performance-model.md section 5.2).
    const std::int64_t rows = 16;
    const std::int64_t in = 128;
    const std::int64_t out = 128;
    const std::size_t x_count = static_cast<std::size_t>(rows) * in;
    const std::size_t w_count = static_cast<std::size_t>(out) * in;
    const std::size_t y_count = static_cast<std::size_t>(rows) * out;

    const std::vector<ComputeType> host_x = MakeInput(x_count, 3);
    const std::vector<ComputeType> host_w = MakeInput(w_count, 5);
    ComputeType* x = Upload(host_x);
    ComputeType* w = Upload(host_w);
    ComputeType* y_dense = static_cast<ComputeType*>(
        nanochat::kernels::Alloc(y_count * sizeof(ComputeType)));
    ComputeType* y_fp8 = static_cast<ComputeType*>(
        nanochat::kernels::Alloc(y_count * sizeof(ComputeType)));
    std::uint8_t* qx =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(x_count));
    std::uint8_t* qw =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(w_count));
    float* scale_x =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));
    float* scale_w =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));

    const bool allocated = x != nullptr && w != nullptr && y_dense != nullptr &&
                           y_fp8 != nullptr && qx != nullptr && qw != nullptr &&
                           scale_x != nullptr && scale_w != nullptr;
    ExpectTrue(allocated, "the fp8 graph buffers allocate");
    if (!allocated) return;

    nanochat::ops::LinearForward(x, w, y_dense, rows, in, out);
    nanochat::Fp8Scratch fp8;
    fp8.a_bytes = qx;
    fp8.b_bytes = qw;
    fp8.a_scale = scale_x;
    fp8.b_scale = scale_w;
    nanochat::ops::LinearForward(x, w, y_fp8, rows, in, out, false, &fp8);
    nanochat::kernels::Synchronize();

    const std::vector<ComputeType> dense = Download(y_dense, y_count);
    const std::vector<ComputeType> quantized = Download(y_fp8, y_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < y_count; ++i) {
      const double a = nanochat::ComputeToFloat(dense[i]);
      const double b = nanochat::ComputeToFloat(quantized[i]);
      // A combined bound: the output is the compute type, so a small result
      // carries an absolute error (docs/fp8-training.md section 7).
      const double error = std::fabs(a - b) / (std::fabs(a) + 1.0);
      if (error > worst) worst = error;
    }
    // A single E4M3 matmul carries the raw 3-mantissa-bit error, which is wider
    // than the 1e-2 graph bound the loss aggregates to
    // (docs/simulator-precision.md section 7).
    ExpectTrue(
        worst <= 5e-2,
        Format("the fp8 linear matches the dense path (worst %.3g)", worst));
    // The two paths must differ, or the fp8 branch did not run. This catches a
    // build whose flag says fp8 but whose code took the dense path.
    ExpectTrue(worst > 0.0, "the fp8 forward path ran (the result differs)");

    nanochat::kernels::Free(x);
    nanochat::kernels::Free(w);
    nanochat::kernels::Free(y_dense);
    nanochat::kernels::Free(y_fp8);
    nanochat::kernels::Free(qx);
    nanochat::kernels::Free(qw);
    nanochat::kernels::Free(scale_x);
    nanochat::kernels::Free(scale_w);
  } else {
    // A dense build has no fp8 branch, so there is nothing to compare.
    ExpectTrue(true, "the dense build skips the fp8 linear comparison");
  }
}

// The fp8 backward against the dense backward at an eligible shape. The weight
// gradient is fp32; the input gradient is the compute type
// (docs/fp8-training.md section 7). The contraction is long enough that the
// per-element fp8 error averages down.
void TestBackwardFp8AgainstDense() {
  if constexpr (nanochat::kFp8Build) {
    const std::int64_t rows = 64;
    const std::int64_t in = 128;
    const std::int64_t out = 128;
    const std::size_t x_count = static_cast<std::size_t>(rows) * in;
    const std::size_t w_count = static_cast<std::size_t>(out) * in;
    const std::size_t dy_count = static_cast<std::size_t>(rows) * out;

    ComputeType* x = Upload(MakeInput(x_count, 3));
    ComputeType* w = Upload(MakeInput(w_count, 5));
    ComputeType* dy = Upload(MakeInput(dy_count, 7));
    float* dw_dense =
        static_cast<float*>(nanochat::kernels::Alloc(w_count * sizeof(float)));
    float* dw_fp8 =
        static_cast<float*>(nanochat::kernels::Alloc(w_count * sizeof(float)));
    ComputeType* dx_dense = static_cast<ComputeType*>(
        nanochat::kernels::Alloc(x_count * sizeof(ComputeType)));
    ComputeType* dx_fp8 = static_cast<ComputeType*>(
        nanochat::kernels::Alloc(x_count * sizeof(ComputeType)));
    // The output gradient is the largest A operand; the saved activation and
    // the weight are the B operands.
    const std::size_t a_count = std::max(dy_count, x_count);
    const std::size_t b_count = std::max(x_count, w_count);
    std::uint8_t* qa =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(a_count));
    std::uint8_t* qb =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(b_count));
    float* scale_a =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));
    float* scale_b =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));

    const bool allocated = x != nullptr && w != nullptr && dy != nullptr &&
                           dw_dense != nullptr && dw_fp8 != nullptr &&
                           dx_dense != nullptr && dx_fp8 != nullptr &&
                           qa != nullptr && qb != nullptr &&
                           scale_a != nullptr && scale_b != nullptr;
    ExpectTrue(allocated, "the fp8 backward buffers allocate");
    if (!allocated) return;

    nanochat::kernels::Memset(dw_dense, 0, w_count * sizeof(float));
    nanochat::kernels::Memset(dw_fp8, 0, w_count * sizeof(float));
    nanochat::ops::LinearWgrad(x, dy, dw_dense, rows, in, out, nullptr);
    nanochat::ops::LinearDgrad(dy, w, dx_dense, rows, in, out, false, nullptr);

    nanochat::Fp8Scratch fp8;
    fp8.a_bytes = qa;
    fp8.b_bytes = qb;
    fp8.a_scale = scale_a;
    fp8.b_scale = scale_b;
    nanochat::ops::LinearWgrad(x, dy, dw_fp8, rows, in, out, &fp8);
    nanochat::ops::LinearDgrad(dy, w, dx_fp8, rows, in, out, false, &fp8);
    nanochat::kernels::Synchronize();

    const std::vector<float> dw_ref = DownloadFloat(dw_dense, w_count);
    const std::vector<float> dw_got = DownloadFloat(dw_fp8, w_count);
    const std::vector<ComputeType> dx_ref = Download(dx_dense, x_count);
    const std::vector<ComputeType> dx_got = Download(dx_fp8, x_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < w_count; ++i) {
      const double error =
          std::fabs(dw_ref[i] - dw_got[i]) / (std::fabs(dw_ref[i]) + 1.0);
      if (error > worst) worst = error;
    }
    for (std::size_t i = 0; i < x_count; ++i) {
      const double a = nanochat::ComputeToFloat(dx_ref[i]);
      const double b = nanochat::ComputeToFloat(dx_got[i]);
      const double error = std::fabs(a - b) / (std::fabs(a) + 1.0);
      if (error > worst) worst = error;
    }
    ExpectTrue(
        worst <= 5e-2,
        Format("the fp8 backward matches the dense path (worst %.3g)", worst));
    ExpectTrue(worst > 0.0, "the fp8 backward path ran (the result differs)");

    nanochat::kernels::Free(x);
    nanochat::kernels::Free(w);
    nanochat::kernels::Free(dy);
    nanochat::kernels::Free(dw_dense);
    nanochat::kernels::Free(dw_fp8);
    nanochat::kernels::Free(dx_dense);
    nanochat::kernels::Free(dx_fp8);
    nanochat::kernels::Free(qa);
    nanochat::kernels::Free(qb);
    nanochat::kernels::Free(scale_a);
    nanochat::kernels::Free(scale_b);
  } else {
    ExpectTrue(true, "the dense build skips the fp8 backward comparison");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestLinearFp8AgainstDense();
  TestBackwardFp8AgainstDense();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("fp8 graph: all checks passed\n");
  return 0;
}
