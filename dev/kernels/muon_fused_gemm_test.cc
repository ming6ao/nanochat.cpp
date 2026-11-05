// T1 GPU correctness test for the Phase 1 prototype: the device-side batched
// fp32 GEMM of the Muon Polar Express iteration, over all five operand forms
// (X^T X, X X^T, A A, X B, and the wide branch B X) and both sides of the
// wide-branch selector.
//
// The reference is the seam `kernels::Gemm`, which is the cuBLAS call
// `MuonUpdate` issues for the same product (backends/cuda/kernels/muon.cu), so
// the prototype is checked against the incumbent and not against a second
// hand-rolled reference. It runs one matrix per call, which is the seam's
// `cublasSgemm` path. The shapes include tiles that are not whole multiples
// of the 128 by 64 block tile and columns that are not a multiple of four, so
// both the vectorized and the scalar gather paths run.
//
// The prototype is fp32-only, so a bf16 or fp8 build skips it.
// See dev/kernels/muon_fused_gemm.h and dev/kernels/README.md.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "dev/kernels/muon_fused_gemm.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

#if NANOCHAT_PRECISION == 0

using nanochat::ComputeType;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::MuonFusedGemm;
using nanochat::dev::MuonGemmForm;
using nanochat::dev::MuonGemmFormName;
using nanochat::dev::MuonGemmOp;
using nanochat::dev::MuonGemmSeamOp;
using nanochat::dev::MuonPolarGemm;
using nanochat::dev::MuonSquareSide;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;

// One stacked group: `batch` matrices of `rows` by `cols`.
struct Case {
  const char* name;
  int rows;
  int cols;
  int batch;
};

constexpr Case kCases[] = {
    // The tall and wide branches of the optimizer's group layout, on tiles
    // smaller than one block tile.
    {"tall", 40, 24, 3},
    {"wide", 24, 40, 3},
    // Whole block tiles (128 by 64) with every operand 16-byte aligned, which
    // is the vectorized float4 loader path.
    {"tiled tall", 256, 128, 2},
    {"tiled wide", 128, 256, 2},
    // Partial tiles and a column count that is not a multiple of four, which
    // is the scalar gather path.
    {"ragged", 130, 70, 2},
    // A square group, where rows == cols and the wide branch applies.
    {"square", 32, 32, 2},
};

// Runs one form and compares it against the seam GEMM on the same operands.
void CheckForm(const Case& c, MuonGemmForm form, MuonSquareSide side,
               const DevBuf<float>& x, const DevBuf<float>& square,
               DevBuf<float>& out) {
  const MuonGemmOp op = MuonPolarGemm(form, side, c.rows, c.cols, c.batch,
                                      x.ptr, square.ptr, out.ptr);
  // The form fixes the product's extent, so the output slice is `m` by `n`.
  const int out_count = c.batch * op.m * op.n;

  // The reference is cuBLAS through the seam, one matrix per call: the seam's
  // batched path builds a `cublasLt` plan per shape and rejects some of these
  // tiny shapes, while the single-matrix path is a plain `cublasSgemm` call.
  // The comparison is therefore per matrix and still against cuBLAS.
  DevBuf<float> want(out_count);
  const GemmParams batched = MuonGemmSeamOp(op);
  for (int p = 0; p < c.batch; ++p) {
    GemmParams gp = batched;
    gp.batch_count = 1;
    gp.stride_a = 0;
    gp.stride_b = 0;
    gp.stride_c = 0;
    nanochat::kernels::Gemm(GemmMode::kForward, gp, op.a + p * op.stride_a,
                            op.b + p * op.stride_b, want.ptr + p * op.stride_c);
  }
  MuonFusedGemm(op);
  nanochat::kernels::Synchronize();

  const std::vector<float> got = out.Download();
  const std::vector<float> ref = want.Download();
  std::vector<float> got_view(got.begin(), got.begin() + out_count);
  std::vector<float> ref_view(ref.begin(), ref.begin() + out_count);
  const char* side_name =
      side == MuonSquareSide::kLeft ? "square-left" : "square-right";
  const std::string label = std::string(c.name) + " " + MuonGemmFormName(form) +
                            " " + side_name + " " + std::to_string(c.rows) +
                            "x" + std::to_string(c.cols) + " b" +
                            std::to_string(c.batch);
  std::printf("  %-44s m=%d n=%d k=%d batch=%d\n", label.c_str(), op.m, op.n,
              op.k, op.batch);
  CheckVectorClose(got_view, ref_view, 1e-4, label.c_str());
}

void RunCase(const Case& c) {
  const int min_dim = std::min(c.rows, c.cols);
  const int max_dim = std::max(c.rows, c.cols);
  const int mat = c.rows * c.cols;
  const int min_sq = min_dim * min_dim;

  Rng rng;
  rng.s ^= static_cast<std::uint64_t>(mat) * 0x9e3779b97f4a7c15ULL +
           static_cast<std::uint64_t>(c.batch);
  DevBuf<float> x(RandomVec(c.batch * mat, &rng));
  DevBuf<float> square(RandomVec(c.batch * min_sq, &rng));
  // The largest output any form writes is the widest square product.
  DevBuf<float> out(c.batch * max_dim * max_dim);

  // `MuonUpdate` picks the branch from the group's extents: rows > cols is the
  // tall branch, everything else is wide (muon.cu). Each form is exercised on
  // the branch that issues it.
  const bool tall = c.rows > c.cols;
  if (tall) {
    CheckForm(c, MuonGemmForm::kXtX, MuonSquareSide::kRight, x, square, out);
  } else {
    CheckForm(c, MuonGemmForm::kXXt, MuonSquareSide::kRight, x, square, out);
  }
  CheckForm(c, MuonGemmForm::kAA, MuonSquareSide::kRight, x, square, out);
  if (tall) {
    CheckForm(c, MuonGemmForm::kXB, MuonSquareSide::kRight, x, square, out);
  } else {
    CheckForm(c, MuonGemmForm::kBX, MuonSquareSide::kLeft, x, square, out);
    // The wide-branch selector: the plain application form with the selector
    // naming the left side. It must reach the same product as the dedicated
    // wide form above, so a caller never needs a second entry point for it.
    CheckForm(c, MuonGemmForm::kXB, MuonSquareSide::kLeft, x, square, out);
  }
}

#endif  // NANOCHAT_PRECISION == 0

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("muon_fused_gemm prototype gpu test\n");
  std::printf("  block %d threads, tile %dx%d, micro %dx%d, %d KB smem\n",
              nanochat::dev::kMuonGemmBlockThreads,
              nanochat::dev::kMuonGemmBlockM, nanochat::dev::kMuonGemmBlockN,
              nanochat::dev::kMuonGemmMicroM, nanochat::dev::kMuonGemmMicroN,
              nanochat::dev::kMuonGemmSmemBytes / 1024);

#if NANOCHAT_PRECISION == 0
  for (const Case& c : kCases) RunCase(c);
  if (Failures() != 0) {
    std::printf("muon_fused_gemm: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("muon_fused_gemm: all checks passed\n");
#else
  std::printf("  skipped: the prototype is fp32-only\n");
#endif
  return 0;
}
