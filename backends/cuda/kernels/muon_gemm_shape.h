#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_GEMM_SHAPE_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_GEMM_SHAPE_H_

// The three GEMM shapes of one Polar Express iteration, for a group of
// `num_params` matrices of `rows` by `cols` (docs/model.md section The Muon
// step).
//
// `muon.cu` issues these three and the hand GEMM's shape table reads the same
// three, so the two cannot drift. Tall (`rows > cols`) is `A = X^T X` and
// `prod = X B`; wide is `A = X X^T` and `prod = B X`.
//
// Header-only and precision free: they fill `GemmParams`, and the seam settles
// the leading dimensions and the strides.

#include <cstdint>

#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {
namespace muon_shape {

// The smaller extent, and the element counts both layouts are built from.
inline int MinDim(int rows, int cols) { return rows < cols ? rows : cols; }

inline std::int64_t MatElements(int rows, int cols) {
  return static_cast<std::int64_t>(rows) * cols;
}

inline std::int64_t MinSquare(int rows, int cols) {
  const std::int64_t min_dim = MinDim(rows, cols);
  return min_dim * min_dim;
}

// `A = X^T X` (tall) or `A = X X^T` (wide): `min_dim` by `min_dim`, over the
// other extent.
inline GemmParams SquareGemm(int rows, int cols, int num_params) {
  const std::int64_t mat = MatElements(rows, cols);
  const std::int64_t min_sq = MinSquare(rows, cols);
  GemmParams params;
  params.batch_count = num_params;
  params.stride_a = mat;
  params.stride_b = mat;
  params.stride_c = min_sq;
  if (rows > cols) {
    params.m = cols;
    params.n = cols;
    params.k = rows;
    params.transpose_a = true;
  } else {
    params.m = rows;
    params.n = rows;
    params.k = cols;
    params.transpose_b = true;
  }
  return params;
}

// `A2 = A A`, over the `min_dim` by `min_dim` square, in place.
inline GemmParams SquareSquareGemm(int rows, int cols, int num_params) {
  const int min_dim = MinDim(rows, cols);
  const std::int64_t min_sq = MinSquare(rows, cols);
  GemmParams params;
  params.m = min_dim;
  params.n = min_dim;
  params.k = min_dim;
  params.batch_count = num_params;
  params.stride_a = min_sq;
  params.stride_b = min_sq;
  params.stride_c = min_sq;
  return params;
}

// `prod = X B` (tall) or `prod = B X` (wide): `rows` by `cols`. The caller
// passes the operands in the order the shape names: `(x, b)` for tall and
// `(b, x)` for wide.
inline GemmParams ApplyGemm(int rows, int cols, int num_params) {
  const std::int64_t mat = MatElements(rows, cols);
  const std::int64_t min_sq = MinSquare(rows, cols);
  GemmParams params;
  params.m = rows;
  params.n = cols;
  params.batch_count = num_params;
  params.stride_c = mat;
  if (rows > cols) {
    params.k = cols;
    params.stride_a = mat;
    params.stride_b = min_sq;
  } else {
    params.k = rows;
    params.stride_a = min_sq;
    params.stride_b = mat;
  }
  return params;
}

}  // namespace muon_shape
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_GEMM_SHAPE_H_
