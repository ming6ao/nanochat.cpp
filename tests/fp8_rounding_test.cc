// tests/fp8_rounding_test.cc -- the shared fp8 conversion gate
// (docs/correctness-gates.md section 3.2, docs/simulator-precision.md
// section 4).
//
// The CPU reference backend and the emulation engine share one conversion
// header, `nanochat/fp8.h`, so the simulator leg is exact by construction. This
// test pins that header against the format definition:
//
//   1. Hand-computed encodings pin the code-to-value formula and the sign.
//   2. Every finite code decodes and re-encodes to itself.
//   3. A sweep of points near every adjacent code pair matches an independent
//      direct scan that applies the tie rule, so the binary search is checked.
//   4. Saturation, infinity, NaN, and the sign bit behave as documented.
//
// It needs no GPU, no fixture, and no broker: tier S0, a plain T0 CPU test.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "nanochat/fp8.h"
#include "nanochat/sandbox.h"

namespace {

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

void ExpectCode(std::uint8_t actual, std::uint8_t expected,
                const std::string& what) {
  if (actual != expected) {
    Fail(Format("%s: got 0x%02X, want 0x%02X", what.c_str(), actual, expected));
  }
}

// The value of one unsigned code, written independently of the header. It is
// the format definition itself: a subnormal scales the mantissa by the smallest
// normal exponent, and a normal adds the implicit leading one.
double CodeValue(int code, int mantissa_bits, int bias, int min_exponent) {
  const int exponent = code >> mantissa_bits;
  const int mantissa = code & ((1 << mantissa_bits) - 1);
  if (exponent == 0) {
    return std::ldexp(static_cast<double>(mantissa),
                      min_exponent - mantissa_bits);
  }
  return std::ldexp(static_cast<double>((1 << mantissa_bits) + mantissa),
                    exponent - bias - mantissa_bits);
}

// The nearest code by a direct scan, with a tie to the even code. This is an
// independent check of the binary search in the header.
int NearestCode(double magnitude, int mantissa_bits, int bias, int min_exponent,
                int max_code) {
  int best = 0;
  double best_gap =
      std::fabs(CodeValue(0, mantissa_bits, bias, min_exponent) - magnitude);
  for (int code = 1; code <= max_code; ++code) {
    const double gap = std::fabs(
        CodeValue(code, mantissa_bits, bias, min_exponent) - magnitude);
    const bool better =
        gap < best_gap || (gap == best_gap && code % 2 == 0 && best % 2 != 0);
    if (better) {
      best = code;
      best_gap = gap;
    }
  }
  return best;
}

// One format under test, with the two public entry points.
struct FormatSpec {
  const char* name;
  int mantissa_bits;
  int bias;
  int min_exponent;
  int max_code;
  double max_value;
  std::uint8_t (*encode)(float);
  float (*decode)(std::uint8_t);
  float (*round)(float);
};

const FormatSpec kFormats[] = {
    {"E4M3", 3, 7, -6, 0x7E, 448.0, &nanochat::Fp8E4M3Encode,
     &nanochat::Fp8E4M3Decode, &nanochat::Fp8E4M3Round},
    {"E5M2", 2, 15, -14, 0x7B, 57344.0, &nanochat::Fp8E5M2Encode,
     &nanochat::Fp8E5M2Decode, &nanochat::Fp8E5M2Round},
};

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

void TestKnownValues() {
  // E4M3: bias 7, mantissa 3, subnormal quantum 2^-9.
  ExpectCode(nanochat::Fp8E4M3Encode(0.0f), 0x00, "E4M3 +0");
  ExpectCode(nanochat::Fp8E4M3Encode(-0.0f), 0x80, "E4M3 -0");
  ExpectCode(nanochat::Fp8E4M3Encode(1.0f), 0x38, "E4M3 1.0");
  ExpectCode(nanochat::Fp8E4M3Encode(-1.0f), 0xB8, "E4M3 -1.0");
  ExpectCode(nanochat::Fp8E4M3Encode(2.0f), 0x40, "E4M3 2.0");
  ExpectCode(nanochat::Fp8E4M3Encode(0.5f), 0x30, "E4M3 0.5");
  ExpectCode(nanochat::Fp8E4M3Encode(448.0f), 0x7E, "E4M3 max");
  ExpectCode(nanochat::Fp8E4M3Encode(-448.0f), 0xFE, "E4M3 -max");
  ExpectCode(nanochat::Fp8E4M3Encode(449.0f), 0x7E, "E4M3 saturates above max");
  ExpectCode(nanochat::Fp8E4M3Encode(1e30f), 0x7E,
             "E4M3 saturates a huge value");
  ExpectCode(nanochat::Fp8E4M3Encode(kInf), 0x7E, "E4M3 saturates +inf");
  ExpectCode(nanochat::Fp8E4M3Encode(-kInf), 0xFE, "E4M3 saturates -inf");
  ExpectCode(nanochat::Fp8E4M3Encode(kNaN), 0x7F, "E4M3 NaN");
  ExpectCode(nanochat::Fp8E4M3Encode(std::ldexp(1.0f, -9)), 0x01,
             "E4M3 smallest subnormal");
  ExpectCode(nanochat::Fp8E4M3Encode(std::ldexp(1.0f, -10)), 0x00,
             "E4M3 below half a subnormal rounds to zero");
  // A tie goes to the even code: 1.0625 is between 0x38 (1.0) and 0x39 (1.125).
  ExpectCode(nanochat::Fp8E4M3Encode(1.0625f), 0x38, "E4M3 tie to even");
  // 1.1875 is between 0x39 (1.125) and 0x3A (1.25).
  ExpectCode(nanochat::Fp8E4M3Encode(1.1875f), 0x3A, "E4M3 tie to even");
  ExpectTrue(nanochat::Fp8E4M3Decode(0x38) == 1.0f, "E4M3 0x38 decodes to 1.0");
  ExpectTrue(nanochat::Fp8E4M3Decode(0x7E) == 448.0f,
             "E4M3 0x7E decodes to the maximum");
  ExpectTrue(std::isnan(nanochat::Fp8E4M3Decode(0x7F)), "E4M3 0x7F is NaN");

  // E5M2: bias 15, mantissa 2, subnormal quantum 2^-16.
  ExpectCode(nanochat::Fp8E5M2Encode(0.0f), 0x00, "E5M2 +0");
  ExpectCode(nanochat::Fp8E5M2Encode(-0.0f), 0x80, "E5M2 -0");
  ExpectCode(nanochat::Fp8E5M2Encode(1.0f), 0x3C, "E5M2 1.0");
  ExpectCode(nanochat::Fp8E5M2Encode(2.0f), 0x40, "E5M2 2.0");
  ExpectCode(nanochat::Fp8E5M2Encode(57344.0f), 0x7B, "E5M2 max");
  ExpectCode(nanochat::Fp8E5M2Encode(-57344.0f), 0xFB, "E5M2 -max");
  ExpectCode(nanochat::Fp8E5M2Encode(57345.0f), 0x7B, "E5M2 saturates");
  ExpectCode(nanochat::Fp8E5M2Encode(kInf), 0x7B, "E5M2 saturates +inf");
  ExpectCode(nanochat::Fp8E5M2Encode(kNaN), 0x7F, "E5M2 NaN");
  ExpectCode(nanochat::Fp8E5M2Encode(std::ldexp(1.0f, -16)), 0x01,
             "E5M2 smallest subnormal");
  // A tie goes to the even code: 1.125 is between 0x3C (1.0) and 0x3D (1.25).
  ExpectCode(nanochat::Fp8E5M2Encode(1.125f), 0x3C, "E5M2 tie to even");
  // 1.375 is between 0x3D (1.25) and 0x3E (1.5).
  ExpectCode(nanochat::Fp8E5M2Encode(1.375f), 0x3E, "E5M2 tie to even");
  ExpectTrue(nanochat::Fp8E5M2Decode(0x3C) == 1.0f, "E5M2 0x3C decodes to 1.0");
  ExpectTrue(nanochat::Fp8E5M2Decode(0x7B) == 57344.0f,
             "E5M2 0x7B decodes to the maximum");
  ExpectTrue(std::isinf(nanochat::Fp8E5M2Decode(0x7C)) &&
                 !std::signbit(nanochat::Fp8E5M2Decode(0x7C)),
             "E5M2 0x7C is +inf");
  ExpectTrue(std::isinf(nanochat::Fp8E5M2Decode(0xFC)) &&
                 std::signbit(nanochat::Fp8E5M2Decode(0xFC)),
             "E5M2 0xFC is -inf");
  ExpectTrue(std::isnan(nanochat::Fp8E5M2Decode(0x7F)), "E5M2 0x7F is NaN");
}

void TestRoundTrip() {
  for (const FormatSpec& format : kFormats) {
    for (int code = 0; code <= format.max_code; ++code) {
      const float magnitude = format.decode(static_cast<std::uint8_t>(code));
      ExpectTrue(std::isfinite(magnitude),
                 Format("%s code 0x%02X decodes finite", format.name, code));
      ExpectTrue(std::fabs(magnitude) <= format.max_value,
                 Format("%s code 0x%02X is within range", format.name, code));
      ExpectCode(format.encode(magnitude), static_cast<std::uint8_t>(code),
                 Format("%s code 0x%02X round trip", format.name, code));
      // The sign bit is separate from the magnitude code.
      ExpectCode(
          format.encode(-magnitude), static_cast<std::uint8_t>(code | 0x80),
          Format("%s code 0x%02X negative round trip", format.name, code));
      ExpectTrue(
          std::signbit(format.decode(static_cast<std::uint8_t>(code | 0x80))),
          Format("%s code 0x%02X decodes negative", format.name, code));
    }
  }
}

void TestNearest() {
  for (const FormatSpec& format : kFormats) {
    std::vector<double> points;
    points.push_back(0.0);
    for (int code = 0; code <= format.max_code; ++code) {
      const double value = CodeValue(code, format.mantissa_bits, format.bias,
                                     format.min_exponent);
      points.push_back(value);
      if (code == format.max_code) continue;
      const double next = CodeValue(code + 1, format.mantissa_bits, format.bias,
                                    format.min_exponent);
      const double mid = 0.5 * (value + next);
      points.push_back(mid);
      points.push_back(mid + (next - value) * 1e-3);
      points.push_back(mid - (next - value) * 1e-3);
      points.push_back(value + (mid - value) * 0.25);
      points.push_back(value + (mid - value) * 0.75);
    }
    points.push_back(format.max_value * 1.5);
    points.push_back(format.max_value * 1000.0);

    int mismatches = 0;
    for (double raw : points) {
      const float input = static_cast<float>(raw);
      if (!std::isfinite(input)) continue;
      const double magnitude = std::fabs(static_cast<double>(input));
      const int expected =
          NearestCode(magnitude, format.mantissa_bits, format.bias,
                      format.min_exponent, format.max_code);
      const int actual = format.encode(input) & 0x7F;
      if (actual != expected) {
        if (mismatches < 5) {
          Fail(Format("%s nearest at %.9g: got 0x%02X, want 0x%02X",
                      format.name, magnitude, actual, expected));
        }
        ++mismatches;
      }
    }
    ExpectTrue(
        mismatches == 0,
        Format("%s: every sweep point matched the direct scan", format.name));
  }
}

void TestMonotonic() {
  for (const FormatSpec& format : kFormats) {
    int previous = 0;
    const int kSteps = 4000;
    for (int i = 0; i <= kSteps; ++i) {
      const double raw =
          format.max_value * 1.2 * static_cast<double>(i) / kSteps;
      const int code = format.encode(static_cast<float>(raw)) & 0x7F;
      if (code < previous) {
        Fail(Format("%s encode decreases at %.9g: 0x%02X after 0x%02X",
                    format.name, raw, code, previous));
        break;
      }
      previous = code;
    }
  }
}

void TestSpecialValues() {
  for (const FormatSpec& format : kFormats) {
    const std::uint8_t max_code = static_cast<std::uint8_t>(format.max_code);
    ExpectTrue(std::isnan(format.round(kNaN)),
               Format("%s round(NaN) is NaN", format.name));
    ExpectCode(format.encode(format.max_value * 2.0f), max_code,
               Format("%s saturates above the maximum", format.name));
    ExpectCode(format.encode(static_cast<float>(format.max_value)), max_code,
               Format("%s keeps the maximum", format.name));
    // The smallest subnormal is kept; half of it rounds to zero on the tie.
    const double quantum =
        std::ldexp(1.0, format.min_exponent - format.mantissa_bits);
    ExpectCode(format.encode(static_cast<float>(quantum)), 0x01,
               Format("%s keeps the smallest subnormal", format.name));
    ExpectCode(format.encode(static_cast<float>(quantum * 0.5)), 0x00,
               Format("%s rounds half a subnormal to zero", format.name));
    ExpectCode(format.encode(static_cast<float>(quantum * 0.49)), 0x00,
               Format("%s rounds just under half to zero", format.name));
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestKnownValues();
  TestRoundTrip();
  TestNearest();
  TestMonotonic();
  TestSpecialValues();
  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("fp8 rounding: all checks passed\n");
  return 0;
}
