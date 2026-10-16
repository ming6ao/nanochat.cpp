#ifndef NANOCHAT_TENSOR_H_
#define NANOCHAT_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

// Core value types shared by every layer: the precision alias, the storage
// dtype tags, and the backend capability report. This header is vendor-free
// on purpose (DESIGN.md section 2.1): no
// CUDA/HIP header is ever reachable from it, so a backend can be replaced as a
// build/link decision. See docs/build.md.

namespace nanochat {

// ---------------------------------------------------------------------------
// Precision (docs/build.md)
// ---------------------------------------------------------------------------
//
// Selected at build time by the toolchain (`--config=fp32`, which defines
// NANOCHAT_PRECISION_FP32). There is one precision per build and no runtime
// dtype dispatch. fp32 is the only compute type today; the bf16 compute path is
// Stage 2 of docs/precision.md.

// 16-bit bfloat16 storage: the upper half of the fp32 bit pattern. It is a
// storage dtype only, so no compute build maps to it yet (docs/precision.md).
struct Bf16 {
  std::uint16_t bits = 0;
};

inline bool operator==(Bf16 a, Bf16 b) { return a.bits == b.bits; }
inline bool operator!=(Bf16 a, Bf16 b) { return a.bits != b.bits; }

// Round-to-nearest-even float -> bfloat16. Inf and NaN keep their top bits.
inline std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return static_cast<std::uint16_t>(bits >> 16);
  }
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

// bfloat16 -> float. Exact for every encoding.
inline float Bf16BitsToFloat(std::uint16_t bf16) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline Bf16 Bf16FromFloat(float value) {
  Bf16 result;
  result.bits = FloatToBf16Bits(value);
  return result;
}

inline float Bf16ToFloat(Bf16 value) { return Bf16BitsToFloat(value.bits); }

// The arithmetic type for this build. fp32 is the only compute type today.
#if !defined(NANOCHAT_PRECISION_FP32)
#error "no precision selected: define NANOCHAT_PRECISION_FP32"
#endif
using ComputeType = float;

// Converts between the build compute type and float. A checkpoint load uses
// these to convert a storage dtype into the compute dtype. The fp32 compute
// type needs no conversion; the calls stay for a future compute type.
inline ComputeType ComputeFromFloat(float value) { return value; }

inline float ComputeToFloat(ComputeType value) { return value; }

// ---------------------------------------------------------------------------
// Dtype tags and element sizes
// ---------------------------------------------------------------------------

// The dtype tags. The first three are the container storage codes, so their
// values are the on-disk contract (docs/precision.md section 5). The code 3
// belonged to `kFp16`. Stage 1 of docs/precision.md retired that tag and keeps
// the code reserved.
enum class DType {
  kFp32 = 0,
  kBf16 = 1,
  kInt8 = 2,
};

inline std::size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return 1;
    case DType::kBf16:
      return 2;
    case DType::kFp32:
    default:
      return 4;
  }
}

// True for a dtype the checkpoint container can store.
inline bool IsStorageDType(DType dtype) {
  return dtype == DType::kFp32 || dtype == DType::kBf16 ||
         dtype == DType::kInt8;
}

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
  static constexpr DType value = DType::kFp32;
  static constexpr std::size_t size = 4;
};

template <>
struct DTypeTraits<Bf16> {
  static constexpr DType value = DType::kBf16;
  static constexpr std::size_t size = 2;
};

// The storage dtype that backs ComputeType in this build.
inline constexpr DType kComputeDType = DTypeTraits<ComputeType>::value;
inline constexpr std::size_t kComputeTypeSize = DTypeTraits<ComputeType>::size;

// ---------------------------------------------------------------------------
// Backend capabilities
// ---------------------------------------------------------------------------

// A vendor-free description of the active device. `kernels::GetCaps()` returns
// one of these; the host fails fast at startup when the build's precision is
// not in the supported set (docs/build.md).
struct Caps {
  int device_index = 0;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_memory_bytes = 0;

  bool is_device = false;         // false for the CPU reference backend
  bool has_cublas = false;        // a batched GEMM library is available
  bool has_tensor_cores = false;  // fp16/bf16 MMA units
  bool supports_fp32 = true;

  const char* device_name = "cpu";

  // True when the device can run arithmetic at `dtype`. Only the compute
  // dtypes are meaningful: fp32 today. bf16 and int8 are storage dtypes
  // (docs/precision.md section 5).
  bool Supports(DType dtype) const {
    if (dtype == DType::kFp32) return supports_fp32;
    return false;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_TENSOR_H_
