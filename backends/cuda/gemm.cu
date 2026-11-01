// cuBLAS-backed GEMM for the CUDA backend. One entry point serves forward,
// dgrad, and wgrad: the host graph supplies the operand roles and the
// transpose flags, so every mode is the universal
//
//     C = alpha * op(A) * op(B) + beta * C
//
// with op(X) = X^T when the corresponding transpose flag is set. cuBLAS is
// column-major, so the call swaps the operands and uses the standard
// row-major-to-column-major trick; see the block comment inside Gemm.
// See docs/build.md and docs/kernels.md.

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "backends/cuda/device.h"

namespace nanochat {
namespace kernels {

namespace {

// cuBLAS operand type of the build compute type.
constexpr cudaDataType_t kDataType = kComputeIsBf16 ? CUDA_R_16BF : CUDA_R_32F;

void CheckCublas(cublasStatus_t status, const char* what) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  std::fprintf(stderr, "fatal: cuBLAS error in %s: code %d (%s)\n", what,
               static_cast<int>(status), cublasGetStatusString(status));
  std::exit(1);
}

cublasHandle_t Handle() {
  // One handle per device thread: a cuBLAS handle binds to the current device,
  // so a shared handle would target the wrong card in a multi-device process
  // (docs/client-runtime-design.md section 8.1).
  thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CheckCublas(cublasCreate(&handle), "cublasCreate");
  }
  return handle;
}

// The `cublasLt` handle, for the scaled fp8 path
// (docs/fp8-training.md section 5.2). One handle per device thread.
cublasLtHandle_t LtHandle() {
  thread_local cublasLtHandle_t handle = nullptr;
  if (handle == nullptr) {
    CheckCublas(cublasLtCreate(&handle), "cublasLtCreate");
  }
  return handle;
}

// The `cublasLt` data type of one fp8 format.
cudaDataType_t LtDataType(Fp8Format format) {
  return format == Fp8Format::kE4M3 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
}

void GemmImpl(const GemmParams& params, const ComputeType* a,
              const ComputeType* b, void* c, cudaDataType_t c_type) {
  if (params.batch_count <= 0 || params.m <= 0 || params.n <= 0) return;
  if (params.k < 0) return;

  const bool ta = params.transpose_a;
  const bool tb = params.transpose_b;
  const int lda = params.lda > 0 ? params.lda : (ta ? params.m : params.k);
  const int ldb = params.ldb > 0 ? params.ldb : (tb ? params.k : params.n);
  const int ldc = params.ldc > 0 ? params.ldc : params.n;
  const std::int64_t stride_a =
      params.stride_a != 0
          ? params.stride_a
          : static_cast<std::int64_t>(ta ? params.k : params.m) * lda;
  const std::int64_t stride_b =
      params.stride_b != 0
          ? params.stride_b
          : static_cast<std::int64_t>(tb ? params.n : params.k) * ldb;
  const std::int64_t stride_c = params.stride_c != 0
                                    ? params.stride_c
                                    : static_cast<std::int64_t>(params.m) * ldc;
  const cublasOperation_t transa = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  const float alpha = params.alpha;
  const float beta = params.beta;
  const void* blas_a = reinterpret_cast<const void*>(b);
  const void* blas_b = reinterpret_cast<const void*>(a);

  if (params.batch_count == 1 && kDataType == CUDA_R_32F &&
      c_type == CUDA_R_32F) {
    CheckCublas(cublasSgemm(Handle(), transa, transb, params.n, params.m,
                            params.k, &alpha, static_cast<const float*>(blas_a),
                            ldb, static_cast<const float*>(blas_b), lda, &beta,
                            static_cast<float*>(c), ldc),
                "cublasSgemm");
    return;
  }

  if (params.batch_count == 1) {
    CheckCublas(
        cublasGemmEx(Handle(), transa, transb, params.n, params.m, params.k,
                     &alpha, blas_a, kDataType, ldb, blas_b, kDataType, lda,
                     &beta, c, c_type, ldc, compute_type, algo),
        "cublasGemmEx");
    return;
  }

  CheckCublas(cublasGemmStridedBatchedEx(
                  Handle(), transa, transb, params.n, params.m, params.k,
                  &alpha, blas_a, kDataType, ldb, stride_b, blas_b, kDataType,
                  lda, stride_a, &beta, c, c_type, ldc, stride_c,
                  params.batch_count, compute_type, algo),
              "cublasGemmStridedBatchedEx");
}

}  // namespace

void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c) {
  (void)mode;
  GemmImpl(params, a, b, c, kDataType);
}

void GemmWgrad(const GemmParams& params, const ComputeType* a,
               const ComputeType* b, float* c) {
  GemmImpl(params, a, b, c, CUDA_R_32F);
}

void GemmScaledFp8(GemmMode mode, const ScaledGemmParams& params,
                   const std::uint8_t* a, const std::uint8_t* b, void* c) {
  if (params.m <= 0 || params.n <= 0 || params.k <= 0) return;
  if (params.batch_count > 1) {
    std::fprintf(stderr,
                 "fatal: GemmScaledFp8 does not support a batch count above "
                 "one yet\n");
    std::exit(1);
  }
  // The column-major swap, the same trick as the dense path above: compute
  // C^T = op(ltA) * op(ltB) with ltA the stored B and ltB the stored A
  // (docs/fp8-training.md section 7). A row-major [r, c] operand is a
  // column-major [c, r] matrix with leading dimension `c`, which is the
  // descriptor's row count after the swap.
  const bool ta = params.transpose_a;
  const bool tb = params.transpose_b;
  const int lt_a_rows = tb ? params.k : params.n;
  const int lt_a_cols = tb ? params.n : params.k;
  const int lt_b_rows = ta ? params.m : params.k;
  const int lt_b_cols = ta ? params.k : params.m;
  const int lt_c_rows = params.n;
  const int lt_c_cols = params.m;

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  CheckCublas(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  const cublasOperation_t op_a = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  CheckCublas(cublasLtMatmulDescSetAttribute(
                  op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
              "cublasLt set TRANSA");
  CheckCublas(cublasLtMatmulDescSetAttribute(
                  op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
              "cublasLt set TRANSB");
  // The operand roles are swapped, so the matmul A scale belongs to the stored
  // B and the matmul B scale belongs to the stored A.
  if (params.scale_b != nullptr) {
    CheckCublas(cublasLtMatmulDescSetAttribute(
                    op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &params.scale_b, sizeof(params.scale_b)),
                "cublasLt set A scale");
  }
  if (params.scale_a != nullptr) {
    CheckCublas(cublasLtMatmulDescSetAttribute(
                    op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &params.scale_a, sizeof(params.scale_a)),
                "cublasLt set B scale");
  }
  if (params.fast_accum) {
    // Fast accumulation is an Ada (8.9) feature (docs/fp8-training.md section
    // 5.3). Hopper (9.0) supports the scaled fp8 GEMM but not this descriptor
    // attribute, and `cublasLt` answers status 7 for it there. On any other
    // card `CheckCublas` would abort from a device thread, so the unsupported
    // case exits with a message instead. The capability is a per-device fact,
    // so ask once per device thread, like `Handle()` above.
    static thread_local const bool kFastAccumSupported = [] {
      const Caps caps = GetCaps();
      return caps.compute_major == 8 && caps.compute_minor == 9;
    }();
    if (!kFastAccumSupported) {
      const Caps caps = GetCaps();
      std::fprintf(stderr,
                   "fatal: fast accumulation needs compute capability 8.9 "
                   "(docs/fp8-training.md section 5.3); this device reports "
                   "%d.%d\n",
                   caps.compute_major, caps.compute_minor);
      std::exit(1);
    }
    const int32_t fast = 1;
    CheckCublas(
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                       &fast, sizeof(fast)),
        "cublasLt set fast accum");
  }

  CheckCublas(cublasLtMatrixLayoutCreate(&a_desc, LtDataType(params.format_b),
                                         lt_a_rows, lt_a_cols, lt_a_rows),
              "cublasLt layout A");
  CheckCublas(cublasLtMatrixLayoutCreate(&b_desc, LtDataType(params.format_a),
                                         lt_b_rows, lt_b_cols, lt_b_rows),
              "cublasLt layout B");
  // The weight gradient keeps fp32; the forward and the input gradient store
  // the compute type (docs/fp8-training.md section 7).
  const cudaDataType_t c_type =
      mode == GemmMode::kWgrad ? CUDA_R_32F : kDataType;
  CheckCublas(cublasLtMatrixLayoutCreate(&c_desc, c_type, lt_c_rows, lt_c_cols,
                                         lt_c_rows),
              "cublasLt layout C");
  CheckCublas(cublasLtMatmulPreferenceCreate(&pref),
              "cublasLtMatmulPreferenceCreate");
  std::size_t workspace_request = 32u * 1024u * 1024u;
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_request, sizeof(workspace_request)),
              "cublasLt preference workspace");

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  CheckCublas(cublasLtMatmulAlgoGetHeuristic(LtHandle(), op_desc, a_desc,
                                             b_desc, c_desc, c_desc, pref, 1,
                                             &heuristic, &returned),
              "cublasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    std::fprintf(stderr, "fatal: no cublasLt fp8 algorithm for this shape\n");
    std::exit(1);
  }

  void* workspace = nullptr;
  if (heuristic.workspaceSize > 0) workspace = Alloc(heuristic.workspaceSize);
  const float alpha = params.alpha;
  const float beta = params.beta;
  CheckCublas(
      cublasLtMatmul(LtHandle(), op_desc, &alpha, b, a_desc, a, b_desc, &beta,
                     c, c_desc, c, c_desc, &heuristic.algo, workspace,
                     heuristic.workspaceSize, cuda_backend::Stream()),
      "cublasLtMatmul");
  if (workspace != nullptr) Free(workspace);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
}

}  // namespace kernels
}  // namespace nanochat
