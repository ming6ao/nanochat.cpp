// cuBLAS-backed GEMM for the CUDA backend. One entry point serves forward,
// dgrad, and wgrad: the host graph supplies the operand roles and the
// transpose flags, so every mode is the universal
//
//     C = alpha * op(A) * op(B) + beta * C
//
// with op(X) = X^T when the corresponding transpose flag is set. cuBLAS is
// column-major, so the call swaps the operands and uses the standard
// row-major-to-column-major trick; see the block comment inside Gemm.
//
// The batched form runs through `cublasLt` with a plan cached per shape. A plan
// holds the descriptors, the algorithm, and the workspace of one shape, so the
// heuristic query and the workspace allocation happen once instead of once per
// call. The first plan of a shape also times its candidate algorithms and keeps
// the fastest, because the heuristic's first choice was the cost, not the
// query. A shape whose plan cannot be built falls back to
// `cublasGemmStridedBatchedEx`.
//
// A shape may also take the hand tensor-core tile, and every other shape keeps
// cuBLAS (`docs/tensor-core-gemm-plan.md`). The build selects the tile's source
// and this file asks `kComputeIsBf16`, so no preprocessor guard is needed.
//
// See docs/build.md and docs/kernels.md.

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/dispatch.h"
#include "backends/cuda/gemm/gemm_launch.h"

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
  // (docs/distributed.md Part 3, section 8.1).
  thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CheckCublas(cublasCreate(&handle), "cublasCreate");
  }
  return handle;
}

// The `cublasLt` handle, for the scaled fp8 path
// (docs/precision.md Part 1, section 5.2). One handle per device thread.
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

// --- The cached cublasLt plan ----------------------------------------------
//
// A batched GEMM of one shape pays three costs per call that do not depend on
// the arithmetic: the descriptor build, the heuristic query, and the workspace
// allocation. The optimizer issues tens of these calls per step at shapes that
// never change, so a plan is built once per shape and replayed.

// Everything a plan depends on: the three extents, the batch count, the
// leading dimensions, the per-batch strides, the transpose flags, and whether
// the output stays fp32 (the weight-gradient form).
struct PlanKey {
  int m = 0;
  int n = 0;
  int k = 0;
  int batch = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
  bool transpose_a = false;
  bool transpose_b = false;
  bool fp32_c = false;

  bool operator<(const PlanKey& other) const {
    return std::tie(m, n, k, batch, lda, ldb, ldc, stride_a, stride_b, stride_c,
                    transpose_a, transpose_b, fp32_c) <
           std::tie(other.m, other.n, other.k, other.batch, other.lda,
                    other.ldb, other.ldc, other.stride_a, other.stride_b,
                    other.stride_c, other.transpose_a, other.transpose_b,
                    other.fp32_c);
  }
};

// One shape's descriptors, algorithm, and workspace. The operand roles are the
// column-major swap of `GemmImpl`: the matmul's A is the stored B, and the
// matmul's B is the stored A.
struct LtPlan {
  bool usable = false;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  void* workspace = nullptr;
  std::size_t workspace_size = 0;
};

// The candidate count the one-time autotune compares, and the iterations it
// times each candidate over. The query answers the same list on every card, so
// the tune is a property of the shape and the card, not of a step.
constexpr int kMaxCandidates = 8;
constexpr int kTuneWarmup = 2;
constexpr int kTuneIters = 3;

// One plan cache per device thread: a descriptor and an algorithm bind to a
// device, so a shared cache would target the wrong card
// (docs/distributed.md Part 3, section 8.1). The cache outlives the thread
// on purpose, like the two handles above: destroying a cuBLAS object during
// static destruction needs a live context.
using PlanCache = std::map<PlanKey, LtPlan>;

PlanCache& Plans() {
  thread_local PlanCache* plans = nullptr;
  if (plans == nullptr) plans = new PlanCache();
  return *plans;
}

// The matmul call of one plan, with the operand roles of the swap. `stored_a`
// is the seam's `a`, which is the matmul's B.
void LtMatmul(const LtPlan& plan, float alpha, float beta, const void* stored_a,
              const void* stored_b, void* c) {
  CheckCublas(
      cublasLtMatmul(LtHandle(), plan.op_desc, &alpha, stored_b, plan.a_desc,
                     stored_a, plan.b_desc, &beta, c, plan.c_desc, c,
                     plan.c_desc, &plan.algo, plan.workspace,
                     plan.workspace_size, cuda_backend::Stream()),
      "cublasLtMatmul");
}

// Times one candidate over the given pointers and answers the best mean
// milliseconds. The device is idle between rounds, so the events measure the
// candidate's own work.
double TimeCandidate(const cublasLtMatmulHeuristicResult_t& candidate,
                     const cublasLtMatmulDesc_t op_desc,
                     const cublasLtMatrixLayout_t a_desc,
                     const cublasLtMatrixLayout_t b_desc,
                     const cublasLtMatrixLayout_t c_desc, void* workspace,
                     float alpha, float beta, const void* stored_a,
                     const void* stored_b, void* c) {
  for (int i = 0; i < kTuneWarmup; ++i) {
    CheckCublas(cublasLtMatmul(LtHandle(), op_desc, &alpha, stored_b, a_desc,
                               stored_a, b_desc, &beta, c, c_desc, c, c_desc,
                               &candidate.algo, workspace,
                               candidate.workspaceSize, cuda_backend::Stream()),
                "cublasLtMatmul tune");
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_backend::CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
  cuda_backend::CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
  cuda_backend::CheckCuda(cudaEventRecord(start, nullptr), "cudaEventRecord");
  for (int i = 0; i < kTuneIters; ++i) {
    CheckCublas(cublasLtMatmul(LtHandle(), op_desc, &alpha, stored_b, a_desc,
                               stored_a, b_desc, &beta, c, c_desc, c, c_desc,
                               &candidate.algo, workspace,
                               candidate.workspaceSize, cuda_backend::Stream()),
                "cublasLtMatmul tune");
  }
  cuda_backend::CheckCuda(cudaEventRecord(stop, nullptr), "cudaEventRecord");
  cuda_backend::CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float elapsed_ms = 0.0f;
  cuda_backend::CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                          "cudaEventElapsedTime");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(elapsed_ms) / kTuneIters;
}

// Builds the plan of one key. `beta` is the caller's, and a plan is only
// autotuned when it overwrites its output: an accumulating GEMM would add its
// own result again for every tuning call.
bool BuildPlan(const PlanKey& key, cudaDataType_t c_type, float alpha,
               float beta, const void* stored_a, const void* stored_b, void* c,
               LtPlan* plan) {
  CheckCublas(
      cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  const cublasOperation_t op_a = key.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = key.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  CheckCublas(
      cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_a, sizeof(op_a)),
      "cublasLt set TRANSA");
  CheckCublas(
      cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_b, sizeof(op_b)),
      "cublasLt set TRANSB");

  // The stored shapes after the column-major swap.
  const int lt_a_rows = key.transpose_b ? key.k : key.n;
  const int lt_a_cols = key.transpose_b ? key.n : key.k;
  const int lt_b_rows = key.transpose_a ? key.m : key.k;
  const int lt_b_cols = key.transpose_a ? key.k : key.m;
  const int lt_c_rows = key.n;
  const int lt_c_cols = key.m;
  const cudaDataType_t c_datatype = key.fp32_c ? CUDA_R_32F : c_type;

  CheckCublas(cublasLtMatrixLayoutCreate(&plan->a_desc, kDataType, lt_a_rows,
                                         lt_a_cols, key.ldb),
              "cublasLt layout A");
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->b_desc, kDataType, lt_b_rows,
                                         lt_b_cols, key.lda),
              "cublasLt layout B");
  CheckCublas(cublasLtMatrixLayoutCreate(&plan->c_desc, c_datatype, lt_c_rows,
                                         lt_c_cols, key.ldc),
              "cublasLt layout C");

  // The batch, when there is one: the count and the per-matrix stride of each
  // operand. The strides are element counts, as cuBLAS takes them.
  if (key.batch > 1) {
    const int count = key.batch;
    const std::int64_t stride_a = key.stride_b;
    const std::int64_t stride_b = key.stride_a;
    const std::int64_t stride_c = key.stride_c;
    cublasLtMatrixLayout_t layouts[3] = {plan->a_desc, plan->b_desc,
                                         plan->c_desc};
    const std::int64_t strides[3] = {stride_a, stride_b, stride_c};
    for (int i = 0; i < 3; ++i) {
      CheckCublas(cublasLtMatrixLayoutSetAttribute(
                      layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
                      sizeof(count)),
                  "cublasLt set batch count");
      CheckCublas(cublasLtMatrixLayoutSetAttribute(
                      layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                      &strides[i], sizeof(strides[i])),
                  "cublasLt set batch stride");
    }
  }

  cublasLtMatmulPreference_t pref = nullptr;
  CheckCublas(cublasLtMatmulPreferenceCreate(&pref),
              "cublasLtMatmulPreferenceCreate");
  std::size_t workspace_request = 32u * 1024u * 1024u;
  CheckCublas(cublasLtMatmulPreferenceSetAttribute(
                  pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_request, sizeof(workspace_request)),
              "cublasLt preference workspace");

  cublasLtMatmulHeuristicResult_t candidates[kMaxCandidates]{};
  int returned = 0;
  CheckCublas(
      cublasLtMatmulAlgoGetHeuristic(
          LtHandle(), plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc,
          plan->c_desc, pref, kMaxCandidates, candidates, &returned),
      "cublasLtMatmulAlgoGetHeuristic");
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned <= 0) {
    std::fprintf(stderr,
                 "cublasLt: no algorithm for [%d, %d, %d] batch %d; the "
                 "batched path falls back to cublasGemmStridedBatchedEx\n",
                 key.m, key.n, key.k, key.batch);
    return false;
  }

  // The workspace is the largest any candidate asks for, so the one
  // allocation serves whichever is chosen.
  for (int i = 0; i < returned; ++i) {
    plan->workspace_size =
        std::max(plan->workspace_size, candidates[i].workspaceSize);
  }
  if (plan->workspace_size > 0) {
    plan->workspace = Alloc(plan->workspace_size);
  }

  // The first candidate is the heuristic's choice. When the GEMM overwrites
  // its output, the others are timed too and the fastest wins: the choice,
  // not the query, is what the plan exists to fix.
  plan->algo = candidates[0].algo;
  if (beta == 0.0f && returned > 1) {
    double best_ms = TimeCandidate(candidates[0], plan->op_desc, plan->a_desc,
                                   plan->b_desc, plan->c_desc, plan->workspace,
                                   alpha, beta, stored_a, stored_b, c);
    int best = 0;
    for (int i = 1; i < returned; ++i) {
      if (candidates[i].workspaceSize > plan->workspace_size) continue;
      const double ms = TimeCandidate(
          candidates[i], plan->op_desc, plan->a_desc, plan->b_desc,
          plan->c_desc, plan->workspace, alpha, beta, stored_a, stored_b, c);
      if (ms < best_ms) {
        best_ms = ms;
        best = i;
      }
    }
    plan->algo = candidates[best].algo;
  }
  plan->usable = true;
  return true;
}

// The batched path: one cached plan per shape, built on the first call of that
// shape and replayed after it. Answers false when the shape has no plan, so
// the caller can fall back.
bool GemmBatched(const GemmParams& params, const ComputeType* a,
                 const ComputeType* b, void* c, cudaDataType_t c_type, int lda,
                 int ldb, int ldc, std::int64_t stride_a, std::int64_t stride_b,
                 std::int64_t stride_c) {
  PlanKey key;
  key.m = params.m;
  key.n = params.n;
  key.k = params.k;
  key.batch = params.batch_count;
  key.lda = lda;
  key.ldb = ldb;
  key.ldc = ldc;
  key.stride_a = stride_a;
  key.stride_b = stride_b;
  key.stride_c = stride_c;
  key.transpose_a = params.transpose_a;
  key.transpose_b = params.transpose_b;
  key.fp32_c = c_type == CUDA_R_32F;

  PlanCache& plans = Plans();
  auto found = plans.find(key);
  if (found == plans.end()) {
    LtPlan plan;
    const bool built =
        BuildPlan(key, c_type, params.alpha, params.beta, a, b, c, &plan);
    found = plans.emplace(key, plan).first;
    if (!built) return false;
  }
  if (!found->second.usable) return false;
  LtMatmul(found->second, params.alpha, params.beta, a, b, c);
  return true;
}

// The hand tile, when the shape table selects it
// (docs/tensor-core-gemm-plan.md section B.6). It answers false for every shape
// the table does not hold, for every call the tile refuses, and for a store
// type the tile does not serve, so cuBLAS keeps every one of those.
//
// The first line is the whole fp32 answer: a tile needs bf16 operands, so the
// discarded branch leaves the table, the switch, and the launchers unnamed.
bool TryGemmTile(const GemmParams& params, const ComputeType* a,
                 const ComputeType* b, void* c, cudaDataType_t c_type) {
  if constexpr (!kComputeIsBf16) return false;
  const bool fp32_c = c_type == CUDA_R_32F;
  if (!fp32_c && c_type != kDataType) return false;
  const ResolvedGemm gemm = ResolveGemm(params, fp32_c);
  const int schedule = GemmTileScheduleFor(gemm, GemmTileModeFromEnv());
  if (schedule < 0) return false;
  return fp32_c ? LaunchGemmTileToFp32(params, a, b, static_cast<float*>(c),
                                       schedule)
                : LaunchGemmTile(params, a, b, static_cast<ComputeType*>(c),
                                 schedule);
}

void GemmImpl(const GemmParams& params, const ComputeType* a,
              const ComputeType* b, void* c, cudaDataType_t c_type) {
  if (params.batch_count <= 0 || params.m <= 0 || params.n <= 0) return;
  if (params.k < 0) return;

  if (TryGemmTile(params, a, b, c, c_type)) return;

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

  if (GemmBatched(params, a, b, c, c_type, lda, ldb, ldc, stride_a, stride_b,
                  stride_c)) {
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
  // (docs/precision.md Part 1, section 7). A row-major [r, c] operand is a
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
    // Fast accumulation is an Ada (8.9) feature (docs/precision.md Part 1,
    // section
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
                   "(docs/precision.md Part 1, section 5.3); this device "
                   "reports "
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
  // the compute type (docs/precision.md Part 1, section 7).
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
