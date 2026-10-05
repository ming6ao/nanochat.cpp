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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "backends/cuda/device.h"

namespace nanochat {
namespace kernels {

namespace {

#if defined(NANOCHAT_PRECISION_FP16)
// Fp16 is bit-compatible with __half; the alias only pins the layout so the
// reinterpret_cast below is checkable.
using CudaScalar = __half;
constexpr cudaDataType_t kDataType = CUDA_R_16F;
static_assert(sizeof(ComputeType) == sizeof(CudaScalar),
              "Fp16 must match __half storage");
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
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CheckCublas(cublasCreate(&handle), "cublasCreate");
  }
  return handle;
}

// Whether the active device exposes fp16 tensor cores (sm_70+). Cached because
// Gemm consults it on every call and cudaGetDeviceProperties is expensive. Only
// the fp16 build branches on it.
#if defined(NANOCHAT_PRECISION_FP16)
bool HasTensorCores() {
  static const bool has_tensor_cores = GetCaps().has_tensor_cores;
  return has_tensor_cores;
}
#endif

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

  // fp16 needs a different cuBLAS compute type per architecture. Volta and
  // later run the fp16 tensor-op path with fp32 accumulation
  // (CUBLAS_COMPUTE_32F). Pascal (sm_60/sm_61) has no tensor cores and its
  // cuBLAS cannot launch the pseudo-fp16 combination (half storage with
  // CUBLAS_COMPUTE_32F): it returns CUBLAS_STATUS_EXECUTION_FAILED. There the
  // only working path is native HGEMM with CUBLAS_COMPUTE_16F, whose alpha and
  // beta are __half. Both are approximate modes; the host graph and the
  // optimizer state stay fp32 wherever the seam requires it.
  // See docs/build.md.
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#if defined(NANOCHAT_PRECISION_FP16)
  if (HasTensorCores()) {
    algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  } else {
    compute_type = CUBLAS_COMPUTE_16F;
  }
#endif

  const float alpha_f = params.alpha;
  const float beta_f = params.beta;
  const void* alpha = &alpha_f;
  const void* beta = &beta_f;
#if defined(NANOCHAT_PRECISION_FP16)
  const __half alpha_h = __float2half_rn(alpha_f);
  const __half beta_h = __float2half_rn(beta_f);
  if (compute_type == CUBLAS_COMPUTE_16F) {
    alpha = &alpha_h;
    beta = &beta_h;
  }
#endif
  const void* blas_a = reinterpret_cast<const void*>(b);
  const void* blas_b = reinterpret_cast<const void*>(a);
  void* blas_c = reinterpret_cast<void*>(c);

#if !defined(NANOCHAT_PRECISION_FP16)
  // A single GEMM runs faster through the classic API than through the
  // strided-batched API with `CUBLAS_GEMM_DEFAULT`. At the language-model head
  // shape (M=16384, N=32768, K=768) `cublasSgemm` reaches about 8.8 TFLOP/s
  // and `cublasGemmStridedBatchedEx` about 3.6 TFLOP/s on Pascal. The two
  // calls take the same operands and flags, so the result is unchanged. The
  // fp16 build keeps the batched path, which carries its own tensor-core
  // branch. See docs/performance.md.
  if (params.batch_count == 1) {
    CheckCublas(cublasSgemm(Handle(), transa, transb, /*m=*/params.n,
                            /*n=*/params.m, /*k=*/params.k, &alpha_f,
                            static_cast<const float*>(blas_a), ldb,
                            static_cast<const float*>(blas_b), lda, &beta_f,
                            static_cast<float*>(blas_c), ldc),
                "cublasSgemm");
    return;
  }
#endif

  CheckCublas(cublasGemmStridedBatchedEx(
                  Handle(), transa, transb, /*m=*/params.n, /*n=*/params.m,
                  /*k=*/params.k, alpha, blas_a, kDataType, ldb, stride_b,
                  blas_b, kDataType, lda, stride_a, beta, blas_c, kDataType,
                  ldc, stride_c, params.batch_count, compute_type, algo),
              "cublasGemmStridedBatchedEx");
}

}  // namespace kernels
}  // namespace nanochat
