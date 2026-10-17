#ifndef NANOCHAT_FP8_H_
#define NANOCHAT_FP8_H_

// The 8-bit floating point conversion, in one place
// (docs/fp8-training.md section 4.1, docs/simulator-precision.md section 4).
//
// The CPU reference backend and the emulation engine both include this header,
// so the two agree by construction. The device kernels use the CUDA conversion
// intrinsics, which follow the same round-to-nearest-even rule
// (docs/correctness-gates.md section 3.2).
//
// Two formats:
//   E4M3  four exponent bits, three mantissa bits, largest value 448. The
//         `fn` encoding has no infinity: the code 0x7F is NaN.
//   E5M2  five exponent bits, two mantissa bits, largest value 57344. The
//         exponent field of all ones carries the infinity and the NaN.
//
// The conversion rounds to nearest with ties to even. It also saturates: a
// value above the format maximum becomes the maximum, not a NaN. The design
// clamps the scaled operand first, and the saturation keeps a later overflow
// safe.
//
// This header is host-only and vendor-free (DESIGN.md section 2.1 rule 3), so a
// backend may include it without reaching above the kernel seam.

#include <cmath>
#include <cstdint>
#include <limits>

namespace nanochat {

// The largest finite value of each format, and the unsigned code that carries
// it. A positive code is the magnitude; the sign bit is separate.
inline constexpr float kFp8E4M3Max = 448.0f;
inline constexpr float kFp8E5M2Max = 57344.0f;

inline constexpr std::uint8_t kFp8E4M3MaxCode = 0x7E;
inline constexpr std::uint8_t kFp8E5M2MaxCode = 0x7B;

// The canonical NaN of each format.
inline constexpr std::uint8_t kFp8E4M3NaN = 0x7F;
inline constexpr std::uint8_t kFp8E5M2NaN = 0x7F;

namespace fp8_detail {

// The value of one unsigned code, for a format with `mantissa_bits` mantissa
// bits, an exponent bias `bias`, and the smallest normal exponent
// `min_exponent`. The unsigned code range is contiguous and it increases in
// value.
inline double CodeValue(int code, int mantissa_bits, int bias,
                        int min_exponent) {
  const int exponent = code >> mantissa_bits;
  const int mantissa = code & ((1 << mantissa_bits) - 1);
  if (exponent == 0) {
    return std::ldexp(static_cast<double>(mantissa),
                      min_exponent - mantissa_bits);
  }
  return std::ldexp(static_cast<double>((1 << mantissa_bits) + mantissa),
                    exponent - bias - mantissa_bits);
}

// Rounds a finite magnitude to the format grid. A tie goes to the even code.
inline std::uint8_t RoundMagnitude(double magnitude, int mantissa_bits,
                                   int bias, int min_exponent, int max_code) {
  if (!(magnitude > 0.0)) return 0;
  const double max_value =
      CodeValue(max_code, mantissa_bits, bias, min_exponent);
  if (magnitude >= max_value) return static_cast<std::uint8_t>(max_code);
  // The largest code whose value does not exceed the magnitude.
  int lower = 0;
  int upper = max_code;
  while (lower < upper) {
    const int mid = (lower + upper + 1) / 2;
    if (CodeValue(mid, mantissa_bits, bias, min_exponent) <= magnitude) {
      lower = mid;
    } else {
      upper = mid - 1;
    }
  }
  if (lower == max_code) return static_cast<std::uint8_t>(max_code);
  const double low_value = CodeValue(lower, mantissa_bits, bias, min_exponent);
  const double high_value =
      CodeValue(lower + 1, mantissa_bits, bias, min_exponent);
  const double low_gap = magnitude - low_value;
  const double high_gap = high_value - magnitude;
  if (low_gap < high_gap) return static_cast<std::uint8_t>(lower);
  if (high_gap < low_gap) return static_cast<std::uint8_t>(lower + 1);
  return static_cast<std::uint8_t>((lower % 2 == 0) ? lower : lower + 1);
}

inline std::uint8_t Encode(float value, int mantissa_bits, int bias,
                           int min_exponent, int max_code,
                           std::uint8_t nan_code) {
  if (std::isnan(value)) return nan_code;
  const bool negative = std::signbit(value);
  double magnitude = std::fabs(static_cast<double>(value));
  const double max_value =
      CodeValue(max_code, mantissa_bits, bias, min_exponent);
  if (std::isinf(magnitude) || magnitude > max_value) magnitude = max_value;
  const std::uint8_t code =
      RoundMagnitude(magnitude, mantissa_bits, bias, min_exponent, max_code);
  return static_cast<std::uint8_t>(code | (negative ? 0x80u : 0x00u));
}

inline float Decode(std::uint8_t bits, int mantissa_bits, int bias,
                    int min_exponent, int max_code, bool has_infinity) {
  const int code = bits & 0x7F;
  if (code > max_code) {
    // E5M2 spends the first code above the finite range on infinity. Every
    // other code above the range is a NaN. E4M3 has no infinity.
    if (has_infinity && code == max_code + 1) {
      return (bits & 0x80u) ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
  }
  const double magnitude = CodeValue(code, mantissa_bits, bias, min_exponent);
  return static_cast<float>((bits & 0x80u) ? -magnitude : magnitude);
}

}  // namespace fp8_detail

// E4M3: the forward operand format (docs/fp8-training.md section 3.1).
inline std::uint8_t Fp8E4M3Encode(float value) {
  return fp8_detail::Encode(value, /*mantissa_bits=*/3, /*bias=*/7,
                            /*min_exponent=*/-6, kFp8E4M3MaxCode, kFp8E4M3NaN);
}

inline float Fp8E4M3Decode(std::uint8_t bits) {
  return fp8_detail::Decode(bits, /*mantissa_bits=*/3, /*bias=*/7,
                            /*min_exponent=*/-6, kFp8E4M3MaxCode,
                            /*has_infinity=*/false);
}

// The value that a store to E4M3 keeps: it quantizes and dequantizes.
inline float Fp8E4M3Round(float value) {
  return Fp8E4M3Decode(Fp8E4M3Encode(value));
}

// E5M2: the backward operand format (docs/fp8-training.md section 3.1).
inline std::uint8_t Fp8E5M2Encode(float value) {
  return fp8_detail::Encode(value, /*mantissa_bits=*/2, /*bias=*/15,
                            /*min_exponent=*/-14, kFp8E5M2MaxCode, kFp8E5M2NaN);
}

inline float Fp8E5M2Decode(std::uint8_t bits) {
  return fp8_detail::Decode(bits, /*mantissa_bits=*/2, /*bias=*/15,
                            /*min_exponent=*/-14, kFp8E5M2MaxCode,
                            /*has_infinity=*/true);
}

inline float Fp8E5M2Round(float value) {
  return Fp8E5M2Decode(Fp8E5M2Encode(value));
}

}  // namespace nanochat

#endif  // NANOCHAT_FP8_H_
