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
#if defined(NANOCHAT_PRECISION_BF16)
constexpr cudaDataType_t kDataType = CUDA_R_16BF;
#else
constexpr cudaDataType_t kDataType = CUDA_R_32F;
#endif

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

}  // namespace kernels
}  // namespace nanochat
