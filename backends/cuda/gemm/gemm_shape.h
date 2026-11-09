#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_SHAPE_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_SHAPE_H_

// The resolved form of one GEMM call, and the rule that decides whether the
// hand tile can serve it (docs/tensor-core-gemm-plan.md section B.6).
//
// `ResolveGemm` settles every default the seam allows, by the same rules
// `GemmImpl` in `backends/cuda/gemm.cu` applies. One function serves the shape
// table, the launcher, and the kernel argument, so the three cannot disagree.
//
// Precision free: the struct holds extents, strides, flags, and the two
// scalars, and no element type. A host test can therefore check the rules.

#include <cstdint>

#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

struct ResolvedGemm {
  int m = 0;
  int n = 0;
  int k = 0;
  int batch_count = 1;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool transpose_a = false;
  bool transpose_b = false;
  // True when C is stored as fp32, which is the weight-gradient form. It is
  // part of the shape, because it selects the store path.
  bool fp32_c = false;
};

// `params` with every leading dimension and stride settled. `fp32_c` states the
// store type of the call.
inline ResolvedGemm ResolveGemm(const GemmParams& params, bool fp32_c) {
  ResolvedGemm gemm;
  gemm.m = params.m;
  gemm.n = params.n;
  gemm.k = params.k;
  gemm.batch_count = params.batch_count;
  gemm.transpose_a = params.transpose_a;
  gemm.transpose_b = params.transpose_b;
  gemm.alpha = params.alpha;
  gemm.beta = params.beta;
  gemm.fp32_c = fp32_c;
  gemm.lda =
      params.lda > 0 ? params.lda : (params.transpose_a ? params.m : params.k);
  gemm.ldb =
      params.ldb > 0 ? params.ldb : (params.transpose_b ? params.k : params.n);
  gemm.ldc = params.ldc > 0 ? params.ldc : params.n;
  const int rows_a = params.transpose_a ? params.k : params.m;
  const int rows_b = params.transpose_b ? params.n : params.k;
  gemm.stride_a = params.stride_a != 0
                      ? params.stride_a
                      : static_cast<std::int64_t>(rows_a) * gemm.lda;
  gemm.stride_b = params.stride_b != 0
                      ? params.stride_b
                      : static_cast<std::int64_t>(rows_b) * gemm.ldb;
  gemm.stride_c = params.stride_c != 0
                      ? params.stride_c
                      : static_cast<std::int64_t>(params.m) * gemm.ldc;
  return gemm;
}

// True when the tile can serve this call at all. It refuses an empty extent and
// nothing else: which shapes are *fast enough* is the shape table's question.
inline bool IsTileCallable(const ResolvedGemm& gemm) {
  return gemm.m > 0 && gemm.n > 0 && gemm.k > 0 && gemm.batch_count > 0;
}

// --- The Hopper mainloop's own rule (docs/kernels.md Part 3 section 5)
// --------
//
// The CUTLASS mainloop of `cutlass_bf16.cu` stages both operands with the
// Tensor Memory Accelerator and stores C with it, so every address it takes
// must sit on a sixteen-byte boundary. A call that breaks one of those
// boundaries is not an error: it takes the next path, which is the hand tile
// and then cuBLAS.
//
// This rule is precision free and device free, so `dispatch_test` checks it on
// the host. The device and architecture test lives in the launcher, because it
// needs the driver.

// The smallest output the Hopper mainloop serves, in elements of C. Below it
// the launch and the tensor-map build cost more than the product, so the hand
// tile keeps those shapes: the 5 by 12 Muon group is 60 outputs and 0.005 ms.
//
// The value is an argument and not a measurement, and no sweep varies it yet.
// Section 11.6 item 3 of docs/kernels.md records that: once the candidates of
// section 11.4 are measured per shape, a fixed crossover is one more argument
// the measurement can replace.
inline constexpr int kCutlassMinOutputElements = 64 * 64;

// The sixteen-byte boundary, in bytes, that every address shares.
inline constexpr std::uintptr_t kCutlassAlignmentBytes = 16;

// One requirement the Hopper mainloop states, in the order the rule tests them.
// `kNone` means it takes the call.
enum class CutlassRefusal {
  kNone = 0,
  // A zero extent: there is no product.
  kExtent,
  // Fewer output elements than `kCutlassMinOutputElements`.
  kSize,
  // A leading dimension or a batch stride that leaves the boundary.
  kStride,
  // A pointer that does not share the boundary.
  kPointer,
};

// `refusal` as a report, a log line, or a benchmark column spells it.
inline const char* CutlassRefusalName(CutlassRefusal refusal) {
  switch (refusal) {
    case CutlassRefusal::kNone:
      return "none";
    case CutlassRefusal::kExtent:
      return "extent";
    case CutlassRefusal::kSize:
      return "size";
    case CutlassRefusal::kStride:
      return "stride";
    case CutlassRefusal::kPointer:
      return "pointer";
  }
  return "none";
}

// True when a count of bf16 elements shares the boundary. The operands are bf16
// in both store forms, because the compute type is.
inline bool CutlassAlignedBf16(int elements) {
  return (static_cast<std::int64_t>(elements) * 2) %
             static_cast<std::int64_t>(kCutlassAlignmentBytes) ==
         0;
}

// True when a count of C elements shares the boundary. A weight gradient stores
// fp32, which needs four elements, and an activation stores bf16, which needs
// eight.
inline bool CutlassAlignedC(int elements, bool fp32_c) {
  const std::int64_t bytes = fp32_c ? 4 : 2;
  return (static_cast<std::int64_t>(elements) * bytes) %
             static_cast<std::int64_t>(kCutlassAlignmentBytes) ==
         0;
}

// True when a count of fp8 elements shares the boundary. An fp8 element is one
// byte, so the same sixteen-byte boundary is sixteen elements and not eight
// (docs/kernels.md section 10.4).
inline bool CutlassAlignedFp8(int elements) {
  return static_cast<std::int64_t>(elements) %
             static_cast<std::int64_t>(kCutlassAlignmentBytes) ==
         0;
}

// True when a host or device address shares the boundary.
inline bool CutlassAlignedPointer(const void* pointer) {
  if (pointer == nullptr) return false;
  return reinterpret_cast<std::uintptr_t>(pointer) % kCutlassAlignmentBytes ==
         0;
}

// The two refusals both arms share, in the order they test them: an empty
// product, then fewer outputs than one tile.
inline CutlassRefusal JudgeExtents(const ResolvedGemm& gemm) {
  if (!IsTileCallable(gemm)) return CutlassRefusal::kExtent;
  if (static_cast<std::int64_t>(gemm.m) * gemm.n < kCutlassMinOutputElements) {
    return CutlassRefusal::kSize;
  }
  return CutlassRefusal::kNone;
}

// The refusal that governs this call. `a`, `b`, and `c` are the operand and
// result pointers of the call, so the rule can state the pointer requirement
// without the caller testing it twice.
inline CutlassRefusal JudgeCutlass(const ResolvedGemm& gemm, const void* a,
                                   const void* b, const void* c) {
  if (const CutlassRefusal shared = JudgeExtents(gemm);
      shared != CutlassRefusal::kNone) {
    return shared;
  }
  // The batch strides only matter past one batch, and `ResolveGemm` has already
  // given every one of them a value.
  if (!CutlassAlignedBf16(gemm.lda) || !CutlassAlignedBf16(gemm.ldb) ||
      !CutlassAlignedC(gemm.ldc, gemm.fp32_c) ||
      (gemm.batch_count > 1 &&
       (!CutlassAlignedBf16(gemm.stride_a) ||
        !CutlassAlignedBf16(gemm.stride_b) ||
        !CutlassAlignedC(gemm.stride_c, gemm.fp32_c)))) {
    return CutlassRefusal::kStride;
  }
  if (!CutlassAlignedPointer(a) || !CutlassAlignedPointer(b) ||
      !CutlassAlignedPointer(c)) {
    return CutlassRefusal::kPointer;
  }
  return CutlassRefusal::kNone;
}

// True when the Hopper mainloop takes this call.
inline bool IsCutlassCallable(const ResolvedGemm& gemm, const void* a,
                              const void* b, const void* c) {
  return JudgeCutlass(gemm, a, b, c) == CutlassRefusal::kNone;
}

// The refusal that governs an fp8 call (docs/kernels.md section 10.4). The
// extents, the size floor, and the pointers follow `JudgeCutlass`; the strides
// do not, because one operand element is one byte instead of two, and a batch
// is refused outright because the fp8 seam states no batch stride.
inline CutlassRefusal JudgeScaledFp8(const ResolvedGemm& gemm, const void* a,
                                     const void* b, const void* c) {
  if (const CutlassRefusal shared = JudgeExtents(gemm);
      shared != CutlassRefusal::kNone) {
    return shared;
  }
  if (gemm.batch_count > 1) return CutlassRefusal::kExtent;
  if (!CutlassAlignedFp8(gemm.lda) || !CutlassAlignedFp8(gemm.ldb) ||
      !CutlassAlignedC(gemm.ldc, gemm.fp32_c)) {
    return CutlassRefusal::kStride;
  }
  if (!CutlassAlignedPointer(a) || !CutlassAlignedPointer(b) ||
      !CutlassAlignedPointer(c)) {
    return CutlassRefusal::kPointer;
  }
  return CutlassRefusal::kNone;
}

// True when two calls need the same tile schedule. The scalars are excluded: a
// shape is measured at one alpha and beta, and the tile applies any pair.
inline bool SameShape(const ResolvedGemm& a, const ResolvedGemm& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k &&
         a.batch_count == b.batch_count && a.lda == b.lda && a.ldb == b.ldb &&
         a.ldc == b.ldc && a.stride_a == b.stride_a &&
         a.stride_b == b.stride_b && a.stride_c == b.stride_c &&
         a.transpose_a == b.transpose_a && a.transpose_b == b.transpose_b &&
         a.fp32_c == b.fp32_c;
}

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_SHAPE_H_
