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

// The operand type of the build compute type. fp32 is the only compute type
// today (docs/precision.md).
constexpr cudaDataType_t kDataType = CUDA_R_32F;

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

}  // namespace

void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c) {
  // The mode is descriptive: forward, dgrad, and wgrad all reduce to the same
  // GEMM once the host has chosen the operands, as kernels.h documents.
  (void)mode;
  if (params.batch_count <= 0 || params.m <= 0 || params.n <= 0) return;
  if (params.k < 0) return;

  const bool ta = params.transpose_a;
  const bool tb = params.transpose_b;

  // Leading dimensions of the *stored* operands. Zero means "infer a dense
  // row-major layout from the logical shape": op(A) is m x k, op(B) is k x n,
  // C is m x n.
  const int lda = params.lda > 0 ? params.lda : (ta ? params.m : params.k);
  const int ldb = params.ldb > 0 ? params.ldb : (tb ? params.k : params.n);
  const int ldc = params.ldc > 0 ? params.ldc : params.n;

  // Per-batch element strides; zero means the dense size of the stored operand.
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

  // Row-major C = op(A) * op(B) is, after transposing the whole product,
  // C^T = op(B)^T * op(A)^T. cuBLAS computes a column-major product, so pass
  // op(B)'s buffer as its A operand and op(A)'s buffer as its B operand, with
  // the row-major buffers reinterpreted column-major (which supplies the
  // trailing transpose). The m/n extents therefore swap as well.
  const cublasOperation_t transa = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = ta ? CUBLAS_OP_T : CUBLAS_OP_N;

  // fp32 uses fp32 accumulation on every device.
  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

  const float alpha_f = params.alpha;
  const float beta_f = params.beta;
  const void* alpha = &alpha_f;
  const void* beta = &beta_f;
  const void* blas_a = reinterpret_cast<const void*>(b);
  const void* blas_b = reinterpret_cast<const void*>(a);
  void* blas_c = reinterpret_cast<void*>(c);

  // A single GEMM runs faster through the classic API than through the
  // strided-batched API with `CUBLAS_GEMM_DEFAULT`. At the language-model head
  // shape (M=16384, N=32768, K=768) `cublasSgemm` reaches about 8.8 TFLOP/s
  // and `cublasGemmStridedBatchedEx` about 3.6 TFLOP/s on Pascal. The two
  // calls take the same operands and flags, so the result is unchanged.
  // See docs/performance.md.
  if (params.batch_count == 1) {
    CheckCublas(cublasSgemm(Handle(), transa, transb, /*m=*/params.n,
                            /*n=*/params.m, /*k=*/params.k, &alpha_f,
                            static_cast<const float*>(blas_a), ldb,
                            static_cast<const float*>(blas_b), lda, &beta_f,
                            static_cast<float*>(blas_c), ldc),
                "cublasSgemm");
    return;
  }

  CheckCublas(cublasGemmStridedBatchedEx(
                  Handle(), transa, transb, /*m=*/params.n, /*n=*/params.m,
                  /*k=*/params.k, alpha, blas_a, kDataType, ldb, stride_b,
                  blas_b, kDataType, lda, stride_a, beta, blas_c, kDataType,
                  ldc, stride_c, params.batch_count, compute_type, algo),
              "cublasGemmStridedBatchedEx");
}

}  // namespace kernels
}  // namespace nanochat
