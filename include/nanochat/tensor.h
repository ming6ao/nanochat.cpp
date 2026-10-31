#ifndef NANOCHAT_TENSOR_H_
#define NANOCHAT_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

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
// Selected at build time by the toolchain: `--config=fp32`, `--config=bf16`,
// or `--config=fp8`. Each sets one value, `NANOCHAT_PRECISION`, which maps
// to `BuildPrecision` below.
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

// The build precision. The numbers match `precision_defines()` in nanochat.bzl,
// so the build macro maps straight to this enum.
enum class BuildPrecision { kFp32 = 0, kBf16 = 1, kFp8 = 2 };

#if !defined(NANOCHAT_PRECISION)
#error \
    "no precision selected: build with --config=fp32, --config=bf16, or --config=fp8"
#else
static_assert(NANOCHAT_PRECISION >= 0 && NANOCHAT_PRECISION <= 2,
              "NANOCHAT_PRECISION is out of range");
#endif

inline constexpr BuildPrecision kBuildPrecision =
    static_cast<BuildPrecision>(NANOCHAT_PRECISION);

// True when the compute type is bf16. The fp8 build computes in bf16 and
// quantizes the eligible matmuls (docs/fp8-training.md section 3.3).
inline constexpr bool kComputeIsBf16 = kBuildPrecision != BuildPrecision::kFp32;

// True in the fp8 build. The graph reads `if constexpr (kFp8Build)` instead of
// a preprocessor block.
inline constexpr bool kFp8Build = kBuildPrecision == BuildPrecision::kFp8;

// The arithmetic type for this build. The bf16 and fp8 builds store compute
// tensors as `Bf16`; the fp32 build stores them as `float`.
using ComputeType = std::conditional_t<kComputeIsBf16, Bf16, float>;

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

// The closed list of mechanism bits `Caps::feature_mask` carries. Every bit
// names a *mechanism* the seam can run. No bit names a model and no bit names a
// device: a model asks for mechanisms through `ModelSpec::requires_features`,
// and a device answers through its own `backends/<name>/caps.json` (see
// nanochat/device_profile.h), so adding a model or a device adds no bit here.
//
// The names below are the authority for the *values*; spec/capability_bits.md
// holds the identical names for the prose and for the caps files. The list is
// closed: a consumer that tests a bit that does not exist should fail to
// compile rather than read a silent false.

// The bit at `index`. A named helper rather than a shift at each constant, so
// the table below stays one name per line and readable.
constexpr std::uint64_t CapabilityBit(int index) {
  return std::uint64_t{1} << index;
}

// Attention with fewer key/value heads than query heads (GQA).
inline constexpr std::uint64_t kFeatureGqa = CapabilityBit(0);
// Rotary position embeddings.
inline constexpr std::uint64_t kFeatureRope = CapabilityBit(1);
// A causal mask with a left window.
inline constexpr std::uint64_t kFeatureSlidingWindow = CapabilityBit(2);
// The ResFormer value gate (`ValueGateForward` / `ValueGateBackward`).
inline constexpr std::uint64_t kFeatureValueGate = CapabilityBit(3);
// bf16 compute and storage.
inline constexpr std::uint64_t kFeatureBf16 = CapabilityBit(4);
// fp32 compute and storage.
inline constexpr std::uint64_t kFeatureFp32 = CapabilityBit(5);
// The fp8 storage codes (one byte of E4M3 or E5M2 in a container).
inline constexpr std::uint64_t kFeatureFp8Storage = CapabilityBit(6);
// The scaled fp8 GEMM path (`GemmScaledFp8` with `QuantizeFp8`).
inline constexpr std::uint64_t kFeatureFp8Gemm = CapabilityBit(7);
// Every primitive family in nanochat/kernels.h resolves and passes its gate.
// This is the non-fused tier: a spec that requires it cannot run on a backend
// that still has holes in the seam.
inline constexpr std::uint64_t kFeaturePrimitiveTierComplete = CapabilityBit(8);
// The two stateful training entries: AdamW and Muon.
inline constexpr std::uint64_t kFeatureAdamW = CapabilityBit(9);
inline constexpr std::uint64_t kFeatureMuon = CapabilityBit(10);

// One fused-override bit per composition whose fusion is legal under the
// boundary rule, that is per composition a backend may substitute with a
// registered fused family (nanochat/plan.h). Each bit is set only when that
// family exists *and* passed its gate; the plan reads it to decide whether the
// override competes with the primitive implementation.
//
//   kFeatureFusedResidualRmsNorm   residual add + RmsNorm, for `rms_norm`
//   kFeatureFusedQkPrep            norm + rope + scale, for `qk_prep`
//   kFeatureFusedAttention         attention with qk_prep folded in, for
//                                  `attention` (the v2 candidate)
//   kFeatureFusedClassifier        softcap + cross-entropy + the logits scale,
//                                  for `classifier`
inline constexpr std::uint64_t kFeatureFusedResidualRmsNorm = CapabilityBit(11);
inline constexpr std::uint64_t kFeatureFusedQkPrep = CapabilityBit(12);
inline constexpr std::uint64_t kFeatureFusedAttention = CapabilityBit(13);
inline constexpr std::uint64_t kFeatureFusedClassifier = CapabilityBit(14);

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

  // The mechanism bits: the OR of the kFeature* constants above. This sits
  // beside the three precision flags rather than replacing them, because those
  // three are the contract every existing caller reads and `Supports(DType)` is
  // a dtype query, not a bit test. Zero claims nothing beyond the three flags,
  // which is what a backend that reports no extra mechanism says.
  std::uint64_t feature_mask = 0;

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

  // True when every bit in `feature` is set in `feature_mask`, so one call
  // answers a single-bit query and a combination (a spec's `requires_features`
  // mask) alike. A query of 0 is vacuously true.
  bool HasFeature(std::uint64_t feature) const {
    return (feature & feature_mask) == feature;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_TENSOR_H_
