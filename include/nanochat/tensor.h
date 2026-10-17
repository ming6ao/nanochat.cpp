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
// Selected at build time by the toolchain: `--config=fp32` defines
// NANOCHAT_PRECISION_FP32 and `--config=bf16` defines NANOCHAT_PRECISION_BF16.
// There is one precision per build and no runtime dtype dispatch. The
// persistent state stays fp32 in every build; see `GradientType` below and
// docs/precision.md section 5.

#if defined(__CUDACC__)
#define NANOCHAT_HOST_DEVICE __host__ __device__
#else
#define NANOCHAT_HOST_DEVICE
#endif

// 16-bit bfloat16 value: the upper half of the fp32 bit pattern. It is the
// compute type in a bf16 build, and it stays available as a storage type in
// every build.
struct Bf16 {
  std::uint16_t bits = 0;

  NANOCHAT_HOST_DEVICE Bf16() = default;
  NANOCHAT_HOST_DEVICE Bf16(float value);
  NANOCHAT_HOST_DEVICE operator float() const;
  NANOCHAT_HOST_DEVICE Bf16& operator+=(float value);
  NANOCHAT_HOST_DEVICE Bf16& operator-=(float value);
  NANOCHAT_HOST_DEVICE Bf16& operator*=(float value);
  NANOCHAT_HOST_DEVICE Bf16& operator/=(float value);
};

inline bool operator==(Bf16 a, Bf16 b) { return a.bits == b.bits; }
inline bool operator!=(Bf16 a, Bf16 b) { return a.bits != b.bits; }

// Round-to-nearest-even float -> bfloat16. Inf and NaN keep their top bits.
NANOCHAT_HOST_DEVICE inline std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits = 0;
#if defined(__CUDA_ARCH__)
  bits = __float_as_uint(value);
#else
  std::memcpy(&bits, &value, sizeof(bits));
#endif
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return static_cast<std::uint16_t>(bits >> 16);
  }
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

// bfloat16 -> float. Exact for every encoding.
NANOCHAT_HOST_DEVICE inline float Bf16BitsToFloat(std::uint16_t bf16) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

NANOCHAT_HOST_DEVICE inline Bf16::Bf16(float value)
    : bits(FloatToBf16Bits(value)) {}

NANOCHAT_HOST_DEVICE inline Bf16::operator float() const {
  return Bf16BitsToFloat(bits);
}

NANOCHAT_HOST_DEVICE inline Bf16& Bf16::operator+=(float value) {
  *this = Bf16(static_cast<float>(*this) + value);
  return *this;
}

NANOCHAT_HOST_DEVICE inline Bf16& Bf16::operator-=(float value) {
  *this = Bf16(static_cast<float>(*this) - value);
  return *this;
}

NANOCHAT_HOST_DEVICE inline Bf16& Bf16::operator*=(float value) {
  *this = Bf16(static_cast<float>(*this) * value);
  return *this;
}

NANOCHAT_HOST_DEVICE inline Bf16& Bf16::operator/=(float value) {
  *this = Bf16(static_cast<float>(*this) / value);
  return *this;
}

NANOCHAT_HOST_DEVICE inline Bf16 Bf16FromFloat(float value) {
  Bf16 result;
  result.bits = FloatToBf16Bits(value);
  return result;
}

NANOCHAT_HOST_DEVICE inline float Bf16ToFloat(Bf16 value) {
  return Bf16BitsToFloat(value.bits);
}

#undef NANOCHAT_HOST_DEVICE

// The arithmetic type for this build. The bf16 build stores compute tensors as
// `Bf16`; the fp32 build stores them as `float`.
#if defined(NANOCHAT_PRECISION_FP32) && defined(NANOCHAT_PRECISION_BF16)
#error "select exactly one precision"
#elif defined(NANOCHAT_PRECISION_BF16)
using ComputeType = Bf16;
#elif defined(NANOCHAT_PRECISION_FP32)
using ComputeType = float;
#else
#error \
    "no precision selected: define NANOCHAT_PRECISION_FP32 or NANOCHAT_PRECISION_BF16"
#endif

// Persistent parameter gradients and optimizer state use fp32 in every build.
using GradientType = float;

// Converts between the build compute type and float. Checkpoint loads and
// arithmetic kernels use these at each dtype boundary.
inline ComputeType ComputeFromFloat(float value) { return ComputeType(value); }

inline float ComputeToFloat(ComputeType value) {
  return static_cast<float>(value);
}

// ---------------------------------------------------------------------------
// Dtype tags and element sizes
// ---------------------------------------------------------------------------

// The dtype tags. The first three are the container storage codes, so their
// values are the on-disk contract (docs/precision.md section 5). The code 3
// belonged to `kFp16`. Stage 1 of docs/precision.md retired that tag and keeps
// the code reserved. The two fp8 tags are GEMM operand tags, not storage codes:
// the checkpoint container stays fp32, so `IsStorageDType` rejects both
// (docs/fp8-training.md section 3.1).
enum class DType {
  kFp32 = 0,
  kBf16 = 1,
  kInt8 = 2,
  kFp8E4M3 = 4,  // forward operands, one byte
  kFp8E5M2 = 5,  // backward operands, one byte
};

inline std::size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
    case DType::kFp8E4M3:
    case DType::kFp8E5M2:
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
  bool supports_bf16 = false;
  bool supports_fp8 = false;  // the E4M3/E5M2 tensor-core path (sm_89+)

  const char* device_name = "cpu";

  // True when the device can run arithmetic at `dtype`. Only the compute
  // dtypes are meaningful; int8 is storage-only (docs/precision.md section 5).
  bool Supports(DType dtype) const {
    if (dtype == DType::kFp32) return supports_fp32;
    if (dtype == DType::kBf16) return supports_bf16;
    if (dtype == DType::kFp8E4M3 || dtype == DType::kFp8E5M2) {
      return supports_fp8;
    }
    return false;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_TENSOR_H_
