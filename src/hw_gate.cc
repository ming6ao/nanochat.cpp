// src/hw_gate.cc -- the hardware correctness gate
// (docs/correctness-gates.md section 4).
//
// The simulator cannot prove the tensor-core arithmetic, the `cublasLt` result,
// the operand layout, or the fused attention tile. This gate runs those cases
// on the real device against a host reference and reports the first mismatch.
//
// The reference lives here, not in a committed fixture. It is small: the fp8
// conversion comes from `nanochat/fp8.h`, the GEMM is a double triple loop over
// the decoded operands, and the attention is a naive softmax. The device
// results are compared to that reference.
//
// Every case uses the seam, so the gate compiles and links on any backend. A
// caller runs it only when the device is real (`caps.is_device`).

#include "src/hw_gate.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/fp8.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace {

// The gate collects every failure, but the caller wants the first one.
bool g_failed = false;
std::string g_first_failure;
// Cases run by the last call, so the result counts itself.
int g_cases = 0;

// The scaled fp8 GEMM cases `CheckScaledGemmCases` defines. They cannot run
// below compute capability 8.9, so the skip branch counts them instead.
constexpr int kScaledGemmCases = 6;

void Fail(const std::string& what) {
  if (!g_failed) g_first_failure = what;
  g_failed = true;
  std::fprintf(stderr, "hw gate: %s\n", what.c_str());
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

// A deterministic input in [0.25, 1.0). The values are positive, so the dot
// products do not cancel and the fp8 error stays relative to the result.
float NextValue(std::uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  const float unit = static_cast<float>((*state >> 8) & 0xffffu) / 65535.0f;
  return 0.25f + 0.75f * unit;
}

ComputeType* Upload(const std::vector<float>& host) {
  std::vector<ComputeType> converted(host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    converted[i] = ComputeFromFloat(host[i]);
  }
  ComputeType* device = static_cast<ComputeType*>(
      kernels::Alloc(host.size() * sizeof(ComputeType)));
  if (device != nullptr && !host.empty()) {
    kernels::Memcpy(device, converted.data(), host.size() * sizeof(ComputeType),
                    CopyDir::kHostToDevice);
  }
  return device;
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  if (count > 0) {
    kernels::Memcpy(host.data(), device, count * sizeof(float),
                    CopyDir::kDeviceToHost);
  }
  return host;
}

std::vector<ComputeType> DownloadCompute(const ComputeType* device,
                                         std::size_t count) {
  std::vector<ComputeType> host(count);
  if (count > 0) {
    kernels::Memcpy(host.data(), device, count * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
  }
  return host;
}

float DownloadScalar(const float* device) {
  float value = 0.0f;
  kernels::Memcpy(&value, device, sizeof(float), CopyDir::kDeviceToHost);
  return value;
}

double DecodeFp8(std::uint8_t bits, Fp8Format format) {
  return format == Fp8Format::kE4M3 ? static_cast<double>(Fp8E4M3Decode(bits))
                                    : static_cast<double>(Fp8E5M2Decode(bits));
}

// Case 1: the fp8 conversion. The device cast must match the shared rounding
// header, byte for byte, and derive the same scale.
void CheckConversion() {
  ++g_cases;
  const int count = 256;
  std::vector<float> values(count);
  std::uint32_t state = 7u;
  for (int i = 0; i < count; ++i)
    values[i] = NextValue(&state) * (i % 2 ? -1.0f : 1.0f);
  values[0] = 1e30f;  // saturates
  values[1] = -1e30f;
  values[2] = 0.0f;

  ComputeType* input = Upload(values);
  std::uint8_t* quantized = static_cast<std::uint8_t*>(
      kernels::Alloc(static_cast<std::size_t>(count)));
  float* scale = static_cast<float*>(kernels::Alloc(sizeof(float)));
  if (input == nullptr || quantized == nullptr || scale == nullptr) {
    Fail("the conversion buffers allocate");
    return;
  }

  for (Fp8Format format : {Fp8Format::kE4M3, Fp8Format::kE5M2}) {
    QuantizeFp8Params params;
    params.count = count;
    params.format = format;
    kernels::QuantizeFp8(params, input, quantized, scale);
    kernels::Synchronize();
    const float device_scale = DownloadScalar(scale);
    std::vector<std::uint8_t> host_bytes(static_cast<std::size_t>(count));
    kernels::Memcpy(host_bytes.data(), quantized,
                    static_cast<std::size_t>(count), CopyDir::kDeviceToHost);

    float amax = 0.0f;
    for (float value : values) {
      amax = std::fmax(amax, std::fabs(value));
    }
    const float magnitude = amax > 1e-12f ? amax : 1e-12f;
    const float fp8_max =
        format == Fp8Format::kE4M3 ? kFp8E4M3Max : kFp8E5M2Max;
    const float expected_scale = magnitude / fp8_max;
    if (std::fabs(device_scale - expected_scale) > expected_scale * 1e-3) {
      Fail(Format("the fp8 scale: got %.9g, want %.9g", device_scale,
                  expected_scale));
      continue;
    }
    const float inv = 1.0f / device_scale;
    for (int i = 0; i < count; ++i) {
      const std::uint8_t expected = format == Fp8Format::kE4M3
                                        ? Fp8E4M3Encode(values[i] * inv)
                                        : Fp8E5M2Encode(values[i] * inv);
      if (host_bytes[static_cast<std::size_t>(i)] != expected) {
        Fail(Format("the fp8 byte %d: got 0x%02X, want 0x%02X", i,
                    host_bytes[static_cast<std::size_t>(i)], expected));
        break;
      }
    }
  }

  kernels::Free(input);
  kernels::Free(quantized);
  kernels::Free(scale);
}

// One scaled GEMM case against a host triple loop. `fp32_output` selects the
// weight-gradient output type (docs/fp8-training.md section 7).
void CheckScaledGemm(const std::string& name, int m, int n, int k,
                     bool transpose_a, bool transpose_b, Fp8Format format_a,
                     Fp8Format format_b, float beta, bool fast_accum,
                     bool fp32_output) {
  ++g_cases;
  const int a_count = m * k;
  const int b_count = k * n;
  const int c_count = m * n;

  std::vector<float> a_values(a_count), b_values(b_count);
  std::uint32_t state = 11u;
  for (int i = 0; i < a_count; ++i) a_values[i] = NextValue(&state);
  for (int i = 0; i < b_count; ++i) b_values[i] = NextValue(&state);

  ComputeType* a_source = Upload(a_values);
  ComputeType* b_source = Upload(b_values);
  std::uint8_t* a_bytes = static_cast<std::uint8_t*>(
      kernels::Alloc(static_cast<std::size_t>(a_count)));
  std::uint8_t* b_bytes = static_cast<std::uint8_t*>(
      kernels::Alloc(static_cast<std::size_t>(b_count)));
  float* a_scale = static_cast<float*>(kernels::Alloc(sizeof(float)));
  float* b_scale = static_cast<float*>(kernels::Alloc(sizeof(float)));
  if (a_source == nullptr || b_source == nullptr || a_bytes == nullptr ||
      b_bytes == nullptr || a_scale == nullptr || b_scale == nullptr) {
    Fail(name + ": the buffers allocate");
    return;
  }

  QuantizeFp8Params quantize;
  quantize.count = a_count;
  quantize.format = format_a;
  kernels::QuantizeFp8(quantize, a_source, a_bytes, a_scale);
  quantize.count = b_count;
  quantize.format = format_b;
  kernels::QuantizeFp8(quantize, b_source, b_bytes, b_scale);

  std::vector<std::uint8_t> host_a(static_cast<std::size_t>(a_count));
  std::vector<std::uint8_t> host_b(static_cast<std::size_t>(b_count));
  kernels::Memcpy(host_a.data(), a_bytes, static_cast<std::size_t>(a_count),
                  CopyDir::kDeviceToHost);
  kernels::Memcpy(host_b.data(), b_bytes, static_cast<std::size_t>(b_count),
                  CopyDir::kDeviceToHost);
  const float scale_a = DownloadScalar(a_scale);
  const float scale_b = DownloadScalar(b_scale);

  // The output buffer. The device stores the compute type for the forward and
  // the input gradient, and fp32 for the weight gradient.
  const std::size_t element_bytes =
      fp32_output ? sizeof(float) : sizeof(ComputeType);
  void* output =
      kernels::Alloc(static_cast<std::size_t>(c_count) * element_bytes);
  if (output == nullptr) {
    Fail(name + ": the output buffer allocates");
    return;
  }
  if (fp32_output) {
    std::vector<float> initial(static_cast<std::size_t>(c_count), 0.1f);
    kernels::Memcpy(output, initial.data(),
                    static_cast<std::size_t>(c_count) * sizeof(float),
                    CopyDir::kHostToDevice);
  } else {
    std::vector<ComputeType> initial(static_cast<std::size_t>(c_count),
                                     ComputeFromFloat(0.1f));
    kernels::Memcpy(output, initial.data(),
                    static_cast<std::size_t>(c_count) * sizeof(ComputeType),
                    CopyDir::kHostToDevice);
  }

  ScaledGemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = 1.0f;
  params.beta = beta;
  params.transpose_a = transpose_a;
  params.transpose_b = transpose_b;
  params.fast_accum = fast_accum;
  params.format_a = format_a;
  params.format_b = format_b;
  params.scale_a = a_scale;
  params.scale_b = b_scale;
  kernels::GemmScaledFp8(GemmMode::kForward, params, a_bytes, b_bytes, output);
  kernels::Synchronize();

  std::vector<double> got(static_cast<std::size_t>(c_count));
  if (fp32_output) {
    const std::vector<float> host = Download(static_cast<const float*>(output),
                                             static_cast<std::size_t>(c_count));
    for (int i = 0; i < c_count; ++i) {
      got[static_cast<std::size_t>(i)] = host[static_cast<std::size_t>(i)];
    }
  } else {
    const std::vector<ComputeType> host =
        DownloadCompute(static_cast<const ComputeType*>(output),
                        static_cast<std::size_t>(c_count));
    for (int i = 0; i < c_count; ++i) {
      got[static_cast<std::size_t>(i)] =
          ComputeToFloat(host[static_cast<std::size_t>(i)]);
    }
  }

  // The host reference: a double triple loop over the decoded operands.
  double worst = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p) {
        const double av =
            DecodeFp8(transpose_a ? host_a[static_cast<std::size_t>(p) * m + i]
                                  : host_a[static_cast<std::size_t>(i) * k + p],
                      format_a);
        const double bv =
            DecodeFp8(transpose_b ? host_b[static_cast<std::size_t>(j) * k + p]
                                  : host_b[static_cast<std::size_t>(p) * n + j],
                      format_b);
        acc += av * bv;
      }
      const std::size_t index = static_cast<std::size_t>(i) * n + j;
      const double expected = acc * scale_a * scale_b + beta * 0.1;
      const double error = std::fabs(got[index] - expected);
      const double relative = error / (std::fabs(expected) + 1.0);
      if (relative > worst) worst = relative;
    }
  }
  if (worst > 5e-2) {
    Fail(Format("%s: worst relative error %.3g is above 5e-2", name.c_str(),
                worst));
  }

  kernels::Free(a_source);
  kernels::Free(b_source);
  kernels::Free(a_bytes);
  kernels::Free(b_bytes);
  kernels::Free(a_scale);
  kernels::Free(b_scale);
  kernels::Free(output);
}

// Cases 2 to 6: the three operand roles, the three layouts, and the
// fast-accumulation flag.
void CheckScaledGemmCases() {
  // Fast accumulation is an Ada (8.9) feature. Hopper (9.0) supports the
  // scaled fp8 GEMM but rejects the descriptor attribute, so ask for it only
  // where the driver accepts it (`backends/cuda/gemm.cu`).
  const Caps caps = kernels::GetCaps();
  const bool fast_accum = caps.compute_major == 8 && caps.compute_minor == 9;
  CheckScaledGemm("forward E4M3", 32, 32, 32, /*transpose_a=*/false,
                  /*transpose_b=*/true, Fp8Format::kE4M3, Fp8Format::kE4M3,
                  /*beta=*/0.0f, fast_accum, /*fp32_output=*/false);
  CheckScaledGemm("dgrad E5M2/E4M3", 32, 32, 32, false, false, Fp8Format::kE5M2,
                  Fp8Format::kE4M3, 0.0f, false, false);
  CheckScaledGemm("wgrad E5M2/E4M3", 32, 32, 32, /*transpose_a=*/true, false,
                  Fp8Format::kE5M2, Fp8Format::kE4M3, /*beta=*/0.5f, false,
                  /*fp32_output=*/true);
  // A non-square layout, so a wrong stride shows.
  CheckScaledGemm("layout non-square", 16, 48, 32, false, true,
                  Fp8Format::kE4M3, Fp8Format::kE4M3, 0.0f, fast_accum, false);
  // The fast-accumulation pair. The flag changes the reduction, and both
  // results stay inside the bound.
  CheckScaledGemm("fast accum on", 16, 64, 32, false, true, Fp8Format::kE4M3,
                  Fp8Format::kE4M3, 0.0f, fast_accum, false);
  CheckScaledGemm("fast accum off", 16, 64, 32, false, true, Fp8Format::kE4M3,
                  Fp8Format::kE4M3, 0.0f, false, false);
}

// Case 7: the attention forward at one small causal shape, against a naive
// host softmax.
void CheckAttention() {
  ++g_cases;
  const int batch = 1;
  const int seq = 8;
  const int heads = 2;
  const int head_dim = 8;
  const int count = batch * seq * heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> q_values(count), k_values(count), v_values(count);
  std::uint32_t state = 37u;
  for (int i = 0; i < count; ++i) q_values[i] = NextValue(&state) - 0.5f;
  for (int i = 0; i < count; ++i) k_values[i] = NextValue(&state) - 0.5f;
  for (int i = 0; i < count; ++i) v_values[i] = NextValue(&state) - 0.5f;

  ComputeType* q = Upload(q_values);
  ComputeType* k = Upload(k_values);
  ComputeType* v = Upload(v_values);
  ComputeType* out = static_cast<ComputeType*>(
      kernels::Alloc(static_cast<std::size_t>(count) * sizeof(ComputeType)));
  float* stats = static_cast<float*>(kernels::Alloc(
      static_cast<std::size_t>(batch) * heads * seq * 2 * sizeof(float)));
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      stats == nullptr) {
    Fail("the attention buffers allocate");
    return;
  }

  AttentionParams params;
  params.batch = batch;
  params.seq = seq;
  params.num_heads = heads;
  params.num_kv_heads = heads;
  params.head_dim = head_dim;
  params.causal = true;
  params.scale = scale;
  kernels::AttentionForward(params, q, k, v, out, stats);
  kernels::Synchronize();
  const std::vector<ComputeType> host_out =
      DownloadCompute(out, static_cast<std::size_t>(count));

  // The host reference: q/k/v are [batch, seq, heads, head_dim].
  double worst = 0.0;
  std::vector<double> row(static_cast<std::size_t>(seq));
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < seq; ++i) {
      double max_logit = -1e30;
      for (int j = 0; j <= i; ++j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          const std::size_t qi =
              (static_cast<std::size_t>(i) * heads + h) * head_dim + d;
          const std::size_t kj =
              (static_cast<std::size_t>(j) * heads + h) * head_dim + d;
          dot += static_cast<double>(q_values[qi]) * k_values[kj];
        }
        row[static_cast<std::size_t>(j)] = dot * scale;
        max_logit = std::fmax(max_logit, row[static_cast<std::size_t>(j)]);
      }
      double sum = 0.0;
      for (int j = 0; j <= i; ++j) {
        row[static_cast<std::size_t>(j)] =
            std::exp(row[static_cast<std::size_t>(j)] - max_logit);
        sum += row[static_cast<std::size_t>(j)];
      }
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= i; ++j) {
          const std::size_t vj =
              (static_cast<std::size_t>(j) * heads + h) * head_dim + d;
          acc += row[static_cast<std::size_t>(j)] / sum * v_values[vj];
        }
        const std::size_t index =
            (static_cast<std::size_t>(i) * heads + h) * head_dim + d;
        const double got = ComputeToFloat(host_out[index]);
        const double error = std::fabs(got - acc) / (std::fabs(acc) + 1.0);
        if (error > worst) worst = error;
      }
    }
  }
  if (worst > 5e-2) {
    Fail(Format("attention forward: worst relative error %.3g is above 5e-2",
                worst));
  }

  kernels::Free(q);
  kernels::Free(k);
  kernels::Free(v);
  kernels::Free(out);
  kernels::Free(stats);
}

}  // namespace

HardwareGateResult RunHardwareCorrectnessGate() {
  g_failed = false;
  g_first_failure.clear();
  g_cases = 0;

  CheckConversion();
  // The scaled fp8 GEMM needs the Ada tensor-core path (docs/fp8-training.md
  // section 5.3), which `Caps::supports_fp8` declares. Do not ask a device for
  // arithmetic it does not have. The CPU reference runs every case, because it
  // has no tensor cores to disagree with.
  const Caps caps = kernels::GetCaps();
  int skipped = 0;
  std::string skipped_reason;
  if (!caps.is_device || caps.supports_fp8) {
    CheckScaledGemmCases();
  } else {
    skipped = kScaledGemmCases;
    skipped_reason = Format(
        "the fp8 scaled GEMM needs compute capability 8.9 or 9.0, and this "
        "device reports %d.%d",
        caps.compute_major, caps.compute_minor);
    std::fprintf(stderr, "hw gate: skipped %d fp8 cases: %s\n", skipped,
                 skipped_reason.c_str());
  }
  CheckAttention();

  HardwareGateResult result;
  result.passed = !g_failed;
  result.cases = g_cases;
  result.skipped = skipped;
  result.first_failure = g_first_failure;
  result.skipped_reason = skipped_reason;
  return result;
}

}  // namespace nanochat
