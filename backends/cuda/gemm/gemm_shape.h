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
