#ifndef NANOCHAT_TENSOR_H_
#define NANOCHAT_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

// Core value types shared by every layer: the precision alias, the storage
// dtype tags, a lightweight tensor descriptor, and the backend capability
// report. This header is vendor-free on purpose (DESIGN.md section 2.1): no
// CUDA/HIP header is ever reachable from it, so a backend can be replaced as a
// build/link decision. See docs/precision.md and docs/backends.md.

namespace nanochat {

// ---------------------------------------------------------------------------
// Precision (docs/precision.md)
// ---------------------------------------------------------------------------
//
// Selected at build time by the toolchain (`--config=fp32` / `--config=fp16`,
// which define NANOCHAT_PRECISION_FP32 / NANOCHAT_PRECISION_FP16). There is one
// precision per build and no runtime dtype dispatch.

// 16-bit IEEE-754 storage. Bit-compatible with CUDA `__half`, so a backend may
// reinterpret_cast between Fp16 and the vendor type without a vendor header
// crossing the seam. Host code converts through FloatToHalf / HalfToFloat.
struct Fp16 {
  std::uint16_t bits = 0;
};

inline bool operator==(Fp16 a, Fp16 b) { return a.bits == b.bits; }
inline bool operator!=(Fp16 a, Fp16 b) { return a.bits != b.bits; }

// Round-to-nearest-even float -> half. Correct for normals, subnormals, and
// the inf/NaN encodings.
inline std::uint16_t FloatToHalfBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) {  // inf or NaN
    const std::uint32_t payload = (magnitude > 0x7f800000u) ? 0x0200u : 0u;
    return static_cast<std::uint16_t>(sign | 0x7c00u | payload);
  }
  if (magnitude >= 0x47800000u) {  // overflow (> 65504) rounds to inf
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (magnitude < 0x38800000u) {  // subnormal or zero (< 2^-14)
    const std::uint32_t exponent = (magnitude >> 23) & 0xffu;
    if (exponent < 103u) return static_cast<std::uint16_t>(sign);
    std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    const int shift = 126 - static_cast<int>(exponent);
    std::uint32_t half = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  const std::uint32_t exponent = (magnitude >> 23) & 0xffu;
  const std::uint32_t mantissa = magnitude & 0x7fffffu;
  std::uint32_t half = ((exponent - 112u) << 10) | (mantissa >> 13);
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
  return static_cast<std::uint16_t>(sign | half);
}

// Half -> float. Exact for every half encoding.
inline float HalfBitsToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1fu;
  std::uint32_t mantissa = half & 0x3ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // signed zero
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);  // inf / NaN
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline Fp16 Fp16FromFloat(float value) {
  Fp16 result;
  result.bits = FloatToHalfBits(value);
  return result;
}

inline float Fp16ToFloat(Fp16 value) { return HalfBitsToFloat(value.bits); }

// The arithmetic type for this build: `float` for fp32, `Fp16` for fp16.
#if defined(NANOCHAT_PRECISION_FP16) && defined(NANOCHAT_PRECISION_FP32)
#error "exactly one precision must be selected"
#elif defined(NANOCHAT_PRECISION_FP16)
using ComputeType = Fp16;
#elif defined(NANOCHAT_PRECISION_FP32)
using ComputeType = float;
#else
#error "no precision selected: define NANOCHAT_PRECISION_FP32 or FP16"
#endif

// ---------------------------------------------------------------------------
// Dtype tags and element sizes
// ---------------------------------------------------------------------------

enum class DType {
  kFp32 = 0,
  kFp16 = 1,
};

inline std::size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kFp16:
      return 2;
    case DType::kFp32:
    default:
      return 4;
  }
}

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
  static constexpr DType value = DType::kFp32;
  static constexpr std::size_t size = 4;
};

template <>
struct DTypeTraits<Fp16> {
  static constexpr DType value = DType::kFp16;
  static constexpr std::size_t size = 2;
};

// The storage dtype that backs ComputeType in this build.
inline constexpr DType kComputeDType = DTypeTraits<ComputeType>::value;
inline constexpr std::size_t kComputeTypeSize = DTypeTraits<ComputeType>::size;

// ---------------------------------------------------------------------------
// Tensor: a non-owning descriptor
// ---------------------------------------------------------------------------
//
// Tensors never own storage here. Device memory is handed out by the backend
// (kernels::Alloc) and the fixed forward/backward graphs carve it into named
// workspace slots (docs/model.md). A Tensor is therefore just a typed view:
// pointer, rank, shape, and strides.

inline constexpr int kMaxTensorRank = 4;

struct Tensor {
  void* data = nullptr;
  DType dtype = kComputeDType;
  int rank = 0;
  std::int64_t shape[kMaxTensorRank] = {0, 0, 0, 0};
  std::int64_t stride[kMaxTensorRank] = {0, 0, 0, 0};

  std::int64_t dim(int axis) const {
    return (axis >= 0 && axis < rank) ? shape[axis] : 0;
  }

  std::int64_t numel() const {
    std::int64_t total = 1;
    for (int axis = 0; axis < rank; ++axis) total *= shape[axis];
    return total;
  }

  // True when the strides describe a dense row-major layout.
  bool IsContiguous() const {
    std::int64_t expected = 1;
    for (int axis = rank - 1; axis >= 0; --axis) {
      if (shape[axis] != 1 && stride[axis] != expected) return false;
      expected *= shape[axis];
    }
    return true;
  }

  // Builds a dense, row-major view over `data`.
  static Tensor Contiguous(void* data, DType dtype,
                           std::initializer_list<std::int64_t> dims) {
    Tensor tensor;
    tensor.data = data;
    tensor.dtype = dtype;
    tensor.rank = static_cast<int>(dims.size());
    std::int64_t stride = 1;
    int axis = tensor.rank;
    for (auto it = dims.end(); it != dims.begin();) {
      --it;
      --axis;
      tensor.shape[axis] = *it;
      tensor.stride[axis] = stride;
      stride *= *it;
    }
    return tensor;
  }
};

// ---------------------------------------------------------------------------
// Backend capabilities
// ---------------------------------------------------------------------------

// A vendor-free description of the active device. `kernels::GetCaps()` returns
// one of these; the host fails fast at startup when the build's precision is
// not in the supported set (docs/precision.md).
struct Caps {
  int device_index = 0;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_memory_bytes = 0;

  bool is_device = false;        // false for the CPU reference backend
  bool has_cublas = false;       // a batched GEMM library is available
  bool has_tensor_cores = false; // fp16/bf16 MMA units
  bool supports_fp32 = true;
  bool supports_fp16 = false;

  const char* device_name = "cpu";

  bool Supports(DType dtype) const {
    return dtype == DType::kFp16 ? supports_fp16 : supports_fp32;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_TENSOR_H_
