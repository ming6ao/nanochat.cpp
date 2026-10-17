#ifndef NANOCHAT_TOOLS_CUDA_SIM_CUDA_SIM_ABI_H_
#define NANOCHAT_TOOLS_CUDA_SIM_CUDA_SIM_ABI_H_

// The CUDA and cuBLAS ABI the interposer mirrors, plus the control surface the
// S0 test uses (docs/simulator.md sections 5.1 and 5.2).
//
// The interposer is a shared library that exports the CUDA runtime, the cuBLAS
// entry points, and the fat-binary hooks the loader calls. `tools/nanochat
// simulate --suite api` preloads it with `LD_PRELOAD`, so the calls from
// `backends/cuda/device.cu`, `backends/cuda/gemm.cu`, and every
// compiler-generated launch stub bind to it. It never calls the real runtime.
//
// The declarations below repeat the vendor ABI's *values* rather than including
// a toolkit header: the library must build on a host with no CUDA toolkit, and
// the values are what the binary under test was compiled against.

#include <cstddef>
#include <cstdint>

// The fabricated `cudaDeviceProp` layout lives in its own runtime-free header
// so //tests:cuda_device_prop_abi_test can cross-check it against the real
// <driver_types.h> without colliding with the runtime typedefs below.
#include "tools/cuda_sim/cuda_device_prop_prefix.h"

// ---------------------------------------------------------------------------
// The CUDA runtime subset
// ---------------------------------------------------------------------------

using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;

using cudaMemcpyKindAbi = int;
constexpr cudaMemcpyKindAbi kCudaMemcpyHostToHost = 0;
constexpr cudaMemcpyKindAbi kCudaMemcpyHostToDevice = 1;
constexpr cudaMemcpyKindAbi kCudaMemcpyDeviceToHost = 2;
constexpr cudaMemcpyKindAbi kCudaMemcpyDeviceToDevice = 3;
constexpr cudaMemcpyKindAbi kCudaMemcpyDefault = 4;

constexpr cudaError_t kCudaSuccess = 0;
constexpr cudaError_t kCudaErrorInvalidValue = 1;
constexpr cudaError_t kCudaErrorInvalidDevicePointer = 17;
constexpr cudaError_t kCudaErrorInvalidConfiguration = 9;
constexpr cudaError_t kCudaErrorInvalidDevice = 10;
constexpr cudaError_t kCudaErrorInvalidResourceHandle = 33;

// The attribute `gemm.cu`'s neighbours would set before a >48 KB launch. The
// tree sets none today; the interposer's rule checks the ordering anyway, and
// the synthetic case in the test exercises it.
constexpr int kCudaFuncAttributeMaxDynamicSharedMemorySize = 8;

// The documented default dynamic shared-memory limit per block.
constexpr int kDefaultSharedMemoryLimit = 48 * 1024;

struct dim3 {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
};

// ---------------------------------------------------------------------------
// The cuBLAS subset
// ---------------------------------------------------------------------------

using cublasStatus_t = int;
using cublasHandle_t = void*;
using cublasOperation_t = int;
using cublasComputeType_t = int;
using cublasGemmAlgo_t = int;
using cublasDataType_t = int;

constexpr cublasStatus_t kCublasStatusSuccess = 0;
constexpr cublasStatus_t kCublasStatusNotInitialized = 1;
constexpr cublasStatus_t kCublasStatusInvalidValue = 7;

constexpr cublasOperation_t kCublasOpN = 0;
constexpr cublasOperation_t kCublasOpT = 1;
constexpr cublasOperation_t kCublasOpC = 2;

constexpr cublasComputeType_t kCublasCompute16F = 64;
constexpr cublasComputeType_t kCublasCompute32F = 68;
constexpr cublasComputeType_t kCublasCompute64F = 70;

constexpr cublasGemmAlgo_t kCublasGemmDefault = -1;
constexpr cublasGemmAlgo_t kCublasGemmDefaultTensorOp = 99;

constexpr cublasDataType_t kCudaR16F = 2;
constexpr cublasDataType_t kCudaR32F = 0;

// ---------------------------------------------------------------------------
// The cublasLt subset (docs/simulator-precision.md section 5)
// ---------------------------------------------------------------------------

using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
using cublasLtMatmulPreference_t = void*;
using cublasLtMatmulDescAttributes_t = int;
using cublasLtMatmulPreferenceAttributes_t = int;
using cudaDataType_t = int;

constexpr cudaDataType_t kCudaR16BF = 14;
constexpr cudaDataType_t kCudaR8FE4M3 = 28;
constexpr cudaDataType_t kCudaR8FE5M2 = 29;

// The descriptor attributes the fp8 path sets, and the preference attribute it
// queries. The values are the vendor ABI's in CUDA 12.
constexpr cublasLtMatmulDescAttributes_t kCublasLtDescTransA = 3;
constexpr cublasLtMatmulDescAttributes_t kCublasLtDescTransB = 4;
constexpr cublasLtMatmulDescAttributes_t kCublasLtDescAScalePointer = 17;
constexpr cublasLtMatmulDescAttributes_t kCublasLtDescBScalePointer = 18;
constexpr cublasLtMatmulDescAttributes_t kCublasLtDescFastAccum = 25;
constexpr cublasLtMatmulPreferenceAttributes_t kCublasLtPrefMaxWorkspace = 1;

// The algorithm descriptor is 8 words; the heuristic result that carries it
// must match the vendor layout, because the caller reads `workspaceSize`.
struct cublasLtMatmulAlgo_t {
  std::uint64_t data[8];
};

struct cublasLtMatmulHeuristicResult_t {
  cublasLtMatmulAlgo_t algo;
  std::size_t workspaceSize;
  cublasStatus_t state;
  float wavesCount;
  int reserved[4];
};

// ---------------------------------------------------------------------------
// The control surface the S0 API test uses
// ---------------------------------------------------------------------------

extern "C" {

// The runtime entry points, declared here because the interposer is the only
// definition in the S0 build. The signatures are the vendor ABI's.
cudaError_t cudaMalloc(void** pointer, std::size_t bytes);
cudaError_t cudaFree(void* pointer);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                       cudaMemcpyKindAbi kind);
cudaError_t cudaMemset(void* pointer, int value, std::size_t bytes);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties_v2(cudaDevicePropPrefix* prop, int device);
const char* cudaGetErrorString(cudaError_t status);
cudaError_t cudaGetLastError(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block,
                             void** args, std::size_t shared, void* stream);
cudaError_t cudaFuncSetAttribute(const void* func, int attribute, int value);
void** __cudaRegisterFatBinary(void* fat_cubin);
void __cudaRegisterFatBinaryEnd(void** handle);
void __cudaUnregisterFatBinary(void** handle);
void __cudaRegisterFunction(void** handle, const void* host_function,
                            const char* device_function,
                            const char* device_name, int thread_limit,
                            void* tid, void* bid, void* block_dim,
                            void* grid_dim, int* warp_size);

// The cuBLAS entry points the backend uses.
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
const char* cublasGetStatusString(cublasStatus_t status);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* a, int lda,
                              const float* b, int ldb, const float* beta,
                              float* c, int ldc);
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* a,
    cublasDataType_t a_type, int lda, long long stride_a, const void* b,
    cublasDataType_t b_type, int ldb, long long stride_b, const void* beta,
    void* c, cublasDataType_t c_type, int ldc, long long stride_c,
    int batch_count, cublasComputeType_t compute_type, cublasGemmAlgo_t algo);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* a,
                            cublasDataType_t a_type, int lda, const void* b,
                            cublasDataType_t b_type, int ldb, const void* beta,
                            void* c, cublasDataType_t c_type, int ldc,
                            cublasComputeType_t compute_type,
                            cublasGemmAlgo_t algo);

// The cublasLt entry points the scaled fp8 path uses.
cublasStatus_t cublasLtCreate(cublasLtHandle_t* light_handle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t light_handle);
const char* cublasLtGetStatusString(cublasStatus_t status);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmul_desc,
                                        cublasComputeType_t compute_type,
                                        cudaDataType_t scale_type);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmul_desc);
cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmul_desc, cublasLtMatmulDescAttributes_t attribute,
    const void* buffer, std::size_t size_in_bytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmul_desc, cublasLtMatmulDescAttributes_t attribute,
    void* buffer, std::size_t size_in_bytes, std::size_t* size_written);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* layout,
                                          cudaDataType_t type,
                                          std::uint64_t rows,
                                          std::uint64_t cols, std::int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout,
                                                int attribute,
                                                const void* buffer,
                                                std::size_t size_in_bytes);
cublasStatus_t cublasLtMatmulPreferenceCreate(
    cublasLtMatmulPreference_t* preference);
cublasStatus_t cublasLtMatmulPreferenceDestroy(
    cublasLtMatmulPreference_t preference);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t preference,
    cublasLtMatmulPreferenceAttributes_t attribute, const void* buffer,
    std::size_t size_in_bytes);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t light_handle, cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc, cublasLtMatrixLayout_t d_desc,
    cublasLtMatmulPreference_t preference, int requested_algo_count,
    cublasLtMatmulHeuristicResult_t heuristic_results[],
    int* return_algo_count);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t light_handle,
                              cublasLtMatmulDesc_t compute_desc,
                              const void* alpha, const void* a,
                              cublasLtMatrixLayout_t a_desc, const void* b,
                              cublasLtMatrixLayout_t b_desc, const void* beta,
                              const void* c, cublasLtMatrixLayout_t c_desc,
                              void* d, cublasLtMatrixLayout_t d_desc,
                              const cublasLtMatmulAlgo_t* algo, void* workspace,
                              std::size_t workspace_size, cudaStream_t stream);

// Clears the log, the allocation registry, the cuBLAS handles, and the
// per-function attribute table. The binding canary is re-armed.
void nanochat_sim_reset(void);

// The whole API log: one JSON object per line, oldest first.
const char* nanochat_sim_log(void);

// The number of records in the log.
int nanochat_sim_record_count(void);

// The profile the run simulates, or the empty string when unset.
const char* nanochat_sim_profile(void);
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_CUDA_SIM_ABI_H_
