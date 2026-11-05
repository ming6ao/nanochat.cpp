#ifndef NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_H_
#define NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_H_

// Phase 1 prototype: the device-side batched fp32 GEMM of the Muon Polar
// Express iteration (docs/muon-fused-kernel.md section 4). `backends/cuda`
// never depends on this header; nothing here is part of the frozen seam.
//
// `MuonUpdate` issues three batched GEMMs per Polar Express iteration through
// the seam `kernels::Gemm`, which is one cuBLAS call per GEMM
// (backends/cuda/kernels/muon.cu). The prototype runs the same three products
// itself, so a later fused launch can carry the whole iteration (three
// products and the two elementwise passes) instead of five dependent launches.
// The arithmetic is the necessary one: `C = op(A) * op(B)` over a batch of
// row-major matrices, in fp32, with no alpha/beta scaling.
//
// The block schedule is the section 4 one, stated once here so the kernel, the
// benchmark, and the test cannot drift:
//
//   * 256 threads per block, `__launch_bounds__(256, 3)`;
//   * block tile 128 by 64 (`kMuonGemmBlockM` by `kMuonGemmBlockN`);
//   * micro-tile 8 by 4 per thread (`kMuonGemmMicroM` by `kMuonGemmMicroN`),
//     which is 32 fp32 accumulators per thread in 16 by 16 thread groups;
//   * two 8-deep k slices (`kMuonGemmSliceK`), double buffered, which is
//     `2 * (128 + 64) * 8` floats of shared memory, 12 KB;
//   * one flat `(matrix, tile_i, tile_j)` index per block, so a launch is a
//     single integer grid however many matrices the batch holds.
//
// The benchmark binary is `muon_fused_gemm_bench`, the correctness test is
// `muon_fused_gemm_gpu_test`, and the measured gate is in
// dev/kernels/README.md.

#include <cstdint>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

// The block schedule of the prototype.
constexpr int kMuonGemmBlockThreads = 256;
constexpr int kMuonGemmBlockM = 128;
constexpr int kMuonGemmBlockN = 64;
constexpr int kMuonGemmSliceK = 8;
constexpr int kMuonGemmMicroM = 8;
constexpr int kMuonGemmMicroN = 4;

// The blocks per SM the kernel is compiled for, and the shared memory the two
// double-buffered slices occupy.
constexpr int kMuonGemmBlocksPerSm = 3;
constexpr int kMuonGemmSmemBytes = 2 * (kMuonGemmBlockM + kMuonGemmBlockN) *
                                   kMuonGemmSliceK *
                                   static_cast<int>(sizeof(float));

// The five operand forms of the Polar Express iteration. The first four are
// expressible with a transpose flag pair; the fifth is not, which is what
// `MuonSquareSide` is for.
//
//   kXtX  A = X^T X   tall branch (rows > cols): transpose_a, operands (x, x)
//   kXXt  A = X X^T   wide branch (rows <= cols): transpose_b, operands (x, x)
//   kAA   A2 = A A    both operands are the square matrix
//   kXB   prod = X B  tall branch: plain x plain, square operand on the right
//   kBX   prod = B X  wide branch: plain x plain, square operand on the left
enum class MuonGemmForm {
  kXtX,
  kXXt,
  kAA,
  kXB,
  kBX,
};

// Name of one form, as a report row spells it: `XtX`, `XXt`, `AA`, `XB`, `BX`.
const char* MuonGemmFormName(MuonGemmForm form);

// Which side of a plain product the min-by-min square operand sits on.
//
// `transpose_a` and `transpose_b` cannot express the wide-branch product: both
// `X B` (tall) and `B X` (wide) are plain operand times plain operand, and the
// only difference is which of the two is the square matrix. The selector names
// that side, so the pair (flags, side) selects every form `MuonUpdate` issues.
enum class MuonSquareSide {
  kRight,  // square operand is `b`: C = X * square   (`X B`)
  kLeft,   // square operand is `a`: C = square * X   (`B X`)
};

// One batched GEMM in the seam's convention: `C[p] = op(A[p]) * op(B[p])` for
// `p` in `[0, batch)`, row-major, fp32.
//
// `op(A[p])` is `m` by `k` and `op(B[p])` is `k` by `n`. `transpose_a` says the
// stored `A` is `k` by `m` (so the leading dimension strides `k` rows of `m`);
// `transpose_b` says the stored `B` is `n` by `k`. A leading dimension of 0
// means the tight one, and a stride of 0 means the tight matrix size, exactly
// as `nanochat::GemmParams` reads them.
struct MuonGemmOp {
  int m = 0;
  int n = 0;
  int k = 0;
  int batch = 0;
  const float* a = nullptr;
  const float* b = nullptr;
  float* c = nullptr;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  bool transpose_a = false;
  bool transpose_b = false;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
};

// The op of one form of the Polar Express iteration, resolved from the group's
// extents and the three buffers `MuonUpdate` holds. `x` is the stacked working
// matrix `[batch, rows, cols]`, `square` is one stacked min-by-min buffer
// (`A`, `B`, or `A2` depending on the form), and `out` receives the product.
//
// The tall branch (`rows > cols`) reduces over `rows` and the wide branch
// (`rows <= cols`) over `cols`, so the min-by-min extent is `min(rows, cols)`
// in both. The returned op names the transpose flags for the four expressible
// forms and carries `side` for the wide-branch product.
MuonGemmOp MuonPolarGemm(MuonGemmForm form, MuonSquareSide side, int rows,
                         int cols, int batch, const float* x,
                         const float* square, float* out);

// The same op as a seam `GemmParams`, i.e. the cuBLAS call `muon.cu` issues for
// this product. The correctness test compares against it and the benchmark
// times it, so the prototype is measured against the incumbent and not against
// a second hand-rolled reference.
GemmParams MuonGemmSeamOp(const MuonGemmOp& op);

// The blocks one op launches: `batch * ceil(m / 128) * ceil(n / 64)`.
int MuonGemmBlockCount(const MuonGemmOp& op);

// The necessary arithmetic of one op, `2 * m * n * k * batch`, in FLOPs.
double MuonGemmFlops(const MuonGemmOp& op);

// Runs the op on the backend stream. The grid is one flat integer: block `f`
// decodes to `(matrix, tile_i, tile_j)` and writes that tile of that matrix.
void MuonFusedGemm(const MuonGemmOp& op);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_MUON_FUSED_GEMM_H_
