// Host side of the Phase 1 prototype: resolve one Polar Express form into the
// batched GEMM the kernel runs, state the same op in the seam's terms for the
// cuBLAS comparison, and launch the flat grid.
//
// Everything here is dev-only. `backends/cuda` does not depend on this file,
// and the CUDA backend's Muon path is unchanged. See
// dev/kernels/muon_fused_gemm.h, dev/kernels/muon_fused_gemm.cuh, and
// dev/kernels/README.md.

#include "dev/kernels/muon_fused_gemm.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "dev/kernels/muon_fused_gemm.cuh"

namespace nanochat {
namespace dev {

const char* MuonGemmFormName(MuonGemmForm form) {
  switch (form) {
    case MuonGemmForm::kXtX:
      return "XtX";
    case MuonGemmForm::kXXt:
      return "XXt";
    case MuonGemmForm::kAA:
      return "AA";
    case MuonGemmForm::kXB:
      return "XB";
    case MuonGemmForm::kBX:
    default:
      return "BX";
  }
}

MuonGemmOp MuonPolarGemm(MuonGemmForm form, MuonSquareSide side, int rows,
                         int cols, int batch, const float* x,
                         const float* square, float* out) {
  const int min_dim = rows < cols ? rows : cols;
  const std::int64_t mat = static_cast<std::int64_t>(rows) * cols;
  const std::int64_t min_sq = static_cast<std::int64_t>(min_dim) * min_dim;

  MuonGemmOp op;
  op.batch = batch;
  switch (form) {
    case MuonGemmForm::kXtX:
      // A = X^T X, the tall branch: reductions over `rows`, so the product is
      // `cols` by `cols` and the stored `x` is read transposed on the left.
      op.m = cols;
      op.n = cols;
      op.k = rows;
      op.a = x;
      op.lda = cols;
      op.transpose_a = true;
      op.b = x;
      op.ldb = cols;
      op.c = out;
      op.ldc = min_dim;
      op.stride_a = mat;
      op.stride_b = mat;
      break;
    case MuonGemmForm::kXXt:
      // A = X X^T, the wide branch: reductions over `cols`, so the product is
      // `rows` by `rows` and the stored `x` is read transposed on the right.
      op.m = rows;
      op.n = rows;
      op.k = cols;
      op.a = x;
      op.lda = cols;
      op.b = x;
      op.ldb = cols;
      op.transpose_b = true;
      op.c = out;
      op.ldc = min_dim;
      op.stride_a = mat;
      op.stride_b = mat;
      break;
    case MuonGemmForm::kAA:
      // A2 = A A over the square min-by-min buffer: no transpose either side.
      op.m = min_dim;
      op.n = min_dim;
      op.k = min_dim;
      op.a = square;
      op.lda = min_dim;
      op.b = square;
      op.ldb = min_dim;
      op.c = out;
      op.ldc = min_dim;
      op.stride_a = min_sq;
      op.stride_b = min_sq;
      break;
    case MuonGemmForm::kXB:
    case MuonGemmForm::kBX: {
      // prod = X B (tall) or prod = B X (wide). Both are plain operand times
      // plain operand, so `transpose_a` and `transpose_b` are clear in both and
      // only the side the square operand sits on separates them: form `BX`
      // names the wide-branch product, and the selector names the side.
      const bool square_left =
          form == MuonGemmForm::kBX || side == MuonSquareSide::kLeft;
      const int reduction = square_left ? rows : cols;
      op.m = rows;
      op.n = cols;
      op.k = reduction;
      op.a = square_left ? square : x;
      op.lda = square_left ? min_dim : cols;
      op.b = square_left ? x : square;
      op.ldb = square_left ? cols : min_dim;
      op.c = out;
      op.ldc = cols;
      op.stride_a = square_left ? min_sq : mat;
      op.stride_b = square_left ? mat : min_sq;
      break;
    }
  }
  // The output is always the tight `m` by `n` matrix of the product, whatever
  // the form: cols by cols for X^T X, rows by rows for X X^T, min by min for
  // A A, and rows by cols for either application product.
  op.stride_c = static_cast<std::int64_t>(op.m) * op.n;
  return op;
}

GemmParams MuonGemmSeamOp(const MuonGemmOp& op) {
  GemmParams p;
  p.m = op.m;
  p.n = op.n;
  p.k = op.k;
  p.batch_count = op.batch;
  p.transpose_a = op.transpose_a;
  p.transpose_b = op.transpose_b;
  p.lda = op.lda;
  p.ldb = op.ldb;
  p.ldc = op.ldc;
  p.stride_a = op.stride_a;
  p.stride_b = op.stride_b;
  p.stride_c = op.stride_c;
  return p;
}

int MuonGemmBlockCount(const MuonGemmOp& op) {
  if (op.batch <= 0 || op.m <= 0 || op.n <= 0) return 0;
  const int tiles_m = (op.m + kMuonGemmBlockM - 1) / kMuonGemmBlockM;
  const int tiles_n = (op.n + kMuonGemmBlockN - 1) / kMuonGemmBlockN;
  return op.batch * tiles_m * tiles_n;
}

double MuonGemmFlops(const MuonGemmOp& op) {
  return 2.0 * static_cast<double>(op.m) * op.n * op.k * op.batch;
}

void MuonFusedGemm(const MuonGemmOp& op) {
  if (op.batch <= 0 || op.m <= 0 || op.n <= 0 || op.k < 0) return;

  // The leading dimensions and the per-matrix strides are the seam's: zero
  // means the tight dense row-major one.
  const int lda = op.lda > 0 ? op.lda : (op.transpose_a ? op.m : op.k);
  const int ldb = op.ldb > 0 ? op.ldb : (op.transpose_b ? op.k : op.n);
  const int ldc = op.ldc > 0 ? op.ldc : op.n;
  const std::int64_t stride_a =
      op.stride_a > 0
          ? op.stride_a
          : static_cast<std::int64_t>(op.transpose_a ? op.k : op.m) * lda;
  const std::int64_t stride_b =
      op.stride_b > 0
          ? op.stride_b
          : static_cast<std::int64_t>(op.transpose_b ? op.n : op.k) * ldb;
  const std::int64_t stride_c =
      op.stride_c > 0 ? op.stride_c : static_cast<std::int64_t>(op.m) * ldc;

  const int blocks = MuonGemmBlockCount(op);
  cuda_backend::Launch(muon_gemm::MuonFusedGemmKernel, dim3(blocks),
                       dim3(kMuonGemmBlockThreads), 0, op.m, op.n, op.k, op.a,
                       op.b, op.c, lda, ldb, ldc, op.transpose_a,
                       op.transpose_b, stride_a, stride_b, stride_c);
}

}  // namespace dev
}  // namespace nanochat
