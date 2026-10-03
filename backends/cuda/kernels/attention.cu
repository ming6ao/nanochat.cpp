// CUDA Attention family: causal / sliding-window / grouped-query attention,
// forward and backward, plus the saved softmax statistics
// `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// The query-key scores, the probability-value product, and every backward
// product are batched GEMMs through the cuBLAS seam; only the softmax and its
// gradient run as custom row kernels. This is the same decomposition the
// PyTorch math backend uses.
//
// The model's q/k/v layout is `[batch, seq, heads, head_dim]`, whose per-head
// stride is not constant across `(b, h)`, so a single strided-batched call
// cannot read it. For multi-head attention (the training and decode path) the
// operands are transposed into a contiguous `[batch, heads, seq, head_dim]`
// scratch, which lets one batched GEMM cover every head. That matters on
// Pascal: a per-head GEMM of `512 x 128` occupies a fraction of the device and
// runs at about 1.4-3 TFLOP/s, while the same work batched over the heads runs
// at about 6-7 TFLOP/s. Grouped-query attention keeps the per-head path,
// because several query heads share one key/value head and their outputs would
// collide in a single batched call.
//
// The softmax kernels mirror `backends/cpu/kernels.cc` and
// `dev/kernels/sequence_ref.h` exactly: the row maximum over the *visible*
// keys, the causal plus sliding-window mask, the `(0, 0)` statistics for an
// empty window, and the `p * (dp - sum(p * dp))` softmax Jacobian.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceMax;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::KeyAllowedDev;
using cuda_kernels::KvHeadCountDev;
using cuda_kernels::KvHeadDev;
using cuda_kernels::ToComputeDev;

constexpr int kSoftmaxThreads = 128;
constexpr int kTransposeThreads = 256;

// Base of head `h` in a `[batch, rows, heads, head_dim]` buffer.
inline std::int64_t HeadBase(int b, int rows, int heads, int h, int head_dim) {
  return (static_cast<std::int64_t>(b) * rows * heads + h) * head_dim;
}

// Base of head `h` in the `[batch, heads, seq, kv_len]` score scratch.
__host__ __device__ inline std::int64_t ScoreBase(const AttentionParams& p,
                                                  int b, int h) {
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  return (static_cast<std::int64_t>(b) * p.num_heads + h) * p.seq * kv_len;
}

// One batched `C = alpha * op(A) * op(B) + beta * C` over contiguous
// `[batch, rows, cols]` head buffers. The seam infers the leading dimensions
// and per-batch strides from the logical shape, so only the shape and the
// transpose flags are needed. `GemmMode` is descriptive only (the seam reduces
// every mode to this product), so all callers pass kForward.
void BatchedGemm(int m, int n, int k, int batch_count, float alpha, float beta,
                 bool transpose_a, bool transpose_b, const ComputeType* a,
                 const ComputeType* b, ComputeType* c) {
  GemmParams gp;
  gp.m = m;
  gp.n = n;
  gp.k = k;
  gp.batch_count = batch_count;
  gp.alpha = alpha;
  gp.beta = beta;
  gp.transpose_a = transpose_a;
  gp.transpose_b = transpose_b;
  Gemm(GemmMode::kForward, gp, a, b, c);
}

// One `C = alpha * op(A) * op(B) + beta * C` with explicit row strides, for the
// grouped-query fallback whose operands are rows of the model's
// `[batch, seq, heads, dim]` buffers (row stride `heads * dim`, not `dim`).
void StridedGemm(int m, int n, int k, float alpha, float beta, bool transpose_a,
                 bool transpose_b, int lda, int ldb, int ldc,
                 const ComputeType* a, const ComputeType* b, ComputeType* c) {
  GemmParams gp;
  gp.m = m;
  gp.n = n;
  gp.k = k;
  gp.alpha = alpha;
  gp.beta = beta;
  gp.transpose_a = transpose_a;
  gp.transpose_b = transpose_b;
  gp.lda = lda;
  gp.ldb = ldb;
  gp.ldc = ldc;
  Gemm(GemmMode::kForward, gp, a, b, c);
}

// Cached device scratch, keyed by slot: 0 is the `[batch, heads, seq, kv_len]`
// score/probability matrices, 1 is the transposed head buffers. Attention runs
// once per layer, so a cudaMalloc/free per call would dominate. The training
// graph is single-threaded, so no locking is needed.
ComputeType* CachedBuffer(int slot, std::size_t count) {
  static ComputeType* buffers[2] = {nullptr, nullptr};
  static std::size_t capacities[2] = {0, 0};
  if (count > capacities[slot]) {
    if (buffers[slot] != nullptr) Free(buffers[slot]);
    buffers[slot] =
        static_cast<ComputeType*>(Alloc(count * sizeof(ComputeType)));
    capacities[slot] = count;
  }
  return buffers[slot];
}

// --- Softmax kernels -------------------------------------------------------

// One block per `(b, h, t)` query row. Computes the visible-key maximum, the
// shifted exponentials, the sum, and the normalized probabilities, and writes
// the `(max, sum_exp)` statistics. Mirrors the reference: an empty window
// yields a zero row and zero statistics.
__global__ void AttentionSoftmaxForwardKernel(
    const AttentionParams params, const ComputeType* __restrict__ scores,
    ComputeType* __restrict__ probs, float* __restrict__ stats) {
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;
  const std::int64_t base =
      ScoreBase(params, b, h) + static_cast<std::int64_t>(t) * kv_len;
  const ComputeType* srow = scores + base;
  ComputeType* prow = probs + base;
  float* stat =
      stats +
      (static_cast<std::int64_t>(b) * params.num_heads + h) * params.seq * 2 +
      t * 2;
  const int tid = threadIdx.x;

  float local_max = -CUDART_INF_F;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    if (!KeyAllowedDev(params, qpos, j)) continue;
    local_max = fmaxf(local_max, AsFloatDev(srow[j]));
  }
  const float row_max = BlockReduceMax(local_max);
  if (row_max == -CUDART_INF_F) {
    for (int j = tid; j < kv_len; j += blockDim.x) prow[j] = ToComputeDev(0.0f);
    if (tid == 0) {
      stat[0] = 0.0f;
      stat[1] = 0.0f;
    }
    return;
  }

  float local_sum = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    float weight = 0.0f;
    if (KeyAllowedDev(params, qpos, j)) {
      weight = expf(AsFloatDev(srow[j]) - row_max);
    }
    prow[j] = ToComputeDev(weight);
    local_sum += weight;
  }
  const float sum_exp = BlockReduceSum(local_sum);
  const float inv = 1.0f / sum_exp;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    prow[j] = ToComputeDev(AsFloatDev(prow[j]) * inv);
  }
  if (tid == 0) {
    stat[0] = row_max;
    stat[1] = sum_exp;
  }
}

// One block per `(b, h, t)` query row. Rebuilds the probabilities from the
// scores and the saved statistics, reduces `sum(p * dp)`, and turns `dP` into
// `dS = p * (dp - sum(p * dp))` in place. Masked positions become zero.
__global__ void AttentionSoftmaxBackwardKernel(
    const AttentionParams params, const ComputeType* __restrict__ scores,
    const float* __restrict__ stats, ComputeType* __restrict__ probs,
    ComputeType* __restrict__ dprobs) {
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;
  const std::int64_t base =
      ScoreBase(params, b, h) + static_cast<std::int64_t>(t) * kv_len;
  const ComputeType* srow = scores + base;
  ComputeType* prow = probs + base;
  ComputeType* drow = dprobs + base;
  const float* stat =
      stats +
      (static_cast<std::int64_t>(b) * params.num_heads + h) * params.seq * 2 +
      t * 2;
  const int tid = threadIdx.x;

  const float row_max = stat[0];
  const float sum_exp = stat[1];
  if (sum_exp <= 0.0f) {
    for (int j = tid; j < kv_len; j += blockDim.x) {
      prow[j] = ToComputeDev(0.0f);
      drow[j] = ToComputeDev(0.0f);
    }
    return;
  }
  const float inv = 1.0f / sum_exp;

  float local_wdp = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    float pj = 0.0f;
    if (KeyAllowedDev(params, qpos, j)) {
      pj = expf(AsFloatDev(srow[j]) - row_max) * inv;
    }
    prow[j] = ToComputeDev(pj);
    local_wdp += pj * AsFloatDev(drow[j]);
  }
  const float weighted_dp = BlockReduceSum(local_wdp);
  for (int j = tid; j < kv_len; j += blockDim.x) {
    const float ds =
        KeyAllowedDev(params, qpos, j)
            ? AsFloatDev(prow[j]) * (AsFloatDev(drow[j]) - weighted_dp)
            : 0.0f;
    drow[j] = ToComputeDev(ds);
  }
}

// --- Head-layout transposes ------------------------------------------------

// `[batch, rows, heads, dim]` -> `[batch, heads, rows, dim]`.
__global__ void TransposeToHeadsKernel(long long total, int rows, int heads,
                                       int dim,
                                       const ComputeType* __restrict__ in,
                                       ComputeType* __restrict__ out) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = i % dim;
  const int h = (i / dim) % heads;
  const int r = (i / (dim * heads)) % rows;
  const int b =
      static_cast<int>(i / (static_cast<long long>(dim) * heads * rows));
  out[((static_cast<long long>(b) * heads + h) * rows + r) * dim + d] = in[i];
}

// `[batch, heads, rows, dim]` -> `[batch, rows, heads, dim]`.
__global__ void TransposeFromHeadsKernel(long long total, int rows, int heads,
                                         int dim,
                                         const ComputeType* __restrict__ in,
                                         ComputeType* __restrict__ out) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = i % dim;
  const int r = (i / dim) % rows;
  const int h = (i / (dim * rows)) % heads;
  const int b =
      static_cast<int>(i / (static_cast<long long>(dim) * rows * heads));
  out[((static_cast<long long>(b) * rows + r) * heads + h) * dim + d] = in[i];
}

void TransposeToHeads(int batch, int rows, int heads, int dim,
                      const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * rows * heads * dim;
  if (total <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeToHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, dim, in,
                       out);
}

void TransposeFromHeads(int batch, int rows, int heads, int dim,
                        const ComputeType* in, ComputeType* out) {
  const long long total = static_cast<long long>(batch) * rows * heads * dim;
  if (total <= 0) return;
  const int grid =
      static_cast<int>((total + kTransposeThreads - 1) / kTransposeThreads);
  cuda_backend::Launch(TransposeFromHeadsKernel, dim3(grid),
                       dim3(kTransposeThreads), 0, total, rows, heads, dim, in,
                       out);
}

// --- Per-head fallback (grouped-query attention) ---------------------------

// scores = Q @ K^T * scale, for one (b, h).
void GemmScores(const AttentionParams& p, int b, int h, float scale,
                const ComputeType* q, const ComputeType* k,
                ComputeType* scores) {
  const int dim = p.head_dim;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int kv_heads = KvHeadCountDev(p);
  const int kvh = KvHeadDev(h, p.num_heads, p.num_kv_heads);
  StridedGemm(p.seq, kv_len, dim, scale, 0.0f, /*transpose_a=*/false,
              /*transpose_b=*/true, p.num_heads * dim, kv_heads * dim, kv_len,
              q + HeadBase(b, p.seq, p.num_heads, h, dim),
              k + HeadBase(b, kv_len, kv_heads, kvh, dim),
              scores + ScoreBase(p, b, h));
}

// out = P @ V, for one (b, h).
void GemmOut(const AttentionParams& p, int b, int h, const ComputeType* probs,
             const ComputeType* v, ComputeType* out) {
  const int dim = p.head_dim;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int kv_heads = KvHeadCountDev(p);
  const int kvh = KvHeadDev(h, p.num_heads, p.num_kv_heads);
  StridedGemm(p.seq, dim, kv_len, 1.0f, 0.0f, /*transpose_a=*/false,
              /*transpose_b=*/false, kv_len, kv_heads * dim, p.num_heads * dim,
              probs + ScoreBase(p, b, h),
              v + HeadBase(b, kv_len, kv_heads, kvh, dim),
              out + HeadBase(b, p.seq, p.num_heads, h, dim));
}

// Whether the batched path applies. Multi-head only: grouped-query attention
// shares key/value heads across query heads, so a batched call would write the
// same key/value gradient from several batches.
bool UseBatchedPath(const AttentionParams& p) {
  return p.num_kv_heads <= 0 || p.num_kv_heads == p.num_heads;
}

}  // namespace

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const int heads = params.num_heads;
  const int dim = params.head_dim;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));
  const std::size_t matrix =
      static_cast<std::size_t>(params.batch) * heads * params.seq * kv_len;
  ComputeType* scores = CachedBuffer(0, 3 * matrix);
  ComputeType* probs = scores + matrix;
  const int rows = params.batch * heads * params.seq;

  if (UseBatchedPath(params)) {
    const std::size_t q_elems =
        static_cast<std::size_t>(params.batch) * heads * params.seq * dim;
    const std::size_t k_elems =
        static_cast<std::size_t>(params.batch) * heads * kv_len * dim;
    ComputeType* head = CachedBuffer(1, 2 * q_elems + 2 * k_elems);
    ComputeType* qT = head;
    ComputeType* kT = qT + q_elems;
    ComputeType* vT = kT + k_elems;
    ComputeType* outT = vT + k_elems;
    TransposeToHeads(params.batch, params.seq, heads, dim, q, qT);
    TransposeToHeads(params.batch, kv_len, heads, dim, k, kT);
    TransposeToHeads(params.batch, kv_len, heads, dim, v, vT);

    BatchedGemm(params.seq, kv_len, dim, params.batch * heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, qT, kT, scores);
    cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                         dim3(kSoftmaxThreads), 0, params, scores, probs,
                         stats);
    BatchedGemm(params.seq, dim, kv_len, params.batch * heads, 1.0f, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/false, probs, vT, outT);
    TransposeFromHeads(params.batch, params.seq, heads, dim, outT, out);
    return;
  }

  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      GemmScores(params, b, h, scale, q, k, scores);
    }
  }
  cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, probs, stats);
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      GemmOut(params, b, h, probs, v, out);
    }
  }
}

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int dim = params.head_dim;
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const int heads = params.num_heads;
  const int kv_heads =
      params.num_kv_heads > 0 ? params.num_kv_heads : params.num_heads;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));

  const std::size_t matrix =
      static_cast<std::size_t>(params.batch) * heads * params.seq * kv_len;
  ComputeType* scores = CachedBuffer(0, 3 * matrix);
  ComputeType* probs = scores + matrix;
  ComputeType* dprobs = probs + matrix;
  const int rows = params.batch * heads * params.seq;

  if (UseBatchedPath(params)) {
    const std::size_t q_elems =
        static_cast<std::size_t>(params.batch) * heads * params.seq * dim;
    const std::size_t k_elems =
        static_cast<std::size_t>(params.batch) * heads * kv_len * dim;
    ComputeType* head = CachedBuffer(1, 3 * q_elems + 4 * k_elems);
    ComputeType* qT = head;
    ComputeType* kT = qT + q_elems;
    ComputeType* vT = kT + k_elems;
    ComputeType* doutT = vT + k_elems;
    ComputeType* dqT = doutT + q_elems;
    ComputeType* dkT = dqT + q_elems;
    ComputeType* dvT = dkT + k_elems;
    TransposeToHeads(params.batch, params.seq, heads, dim, q, qT);
    TransposeToHeads(params.batch, kv_len, heads, dim, k, kT);
    TransposeToHeads(params.batch, kv_len, heads, dim, v, vT);
    TransposeToHeads(params.batch, params.seq, heads, dim, dout, doutT);

    const int batch_heads = params.batch * heads;
    // scores = Q @ K^T * scale; dP = dOut @ V^T.
    BatchedGemm(params.seq, kv_len, dim, batch_heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, qT, kT, scores);
    BatchedGemm(params.seq, kv_len, dim, batch_heads, 1.0f, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/true, doutT, vT, dprobs);
    cuda_backend::Launch(AttentionSoftmaxBackwardKernel, dim3(rows),
                         dim3(kSoftmaxThreads), 0, params, scores, stats, probs,
                         dprobs);

    // dQ = dS @ K * scale; dK = dS^T @ Q * scale; dV = P^T @ dOut.
    BatchedGemm(params.seq, dim, kv_len, batch_heads, scale, 0.0f,
                /*transpose_a=*/false, /*transpose_b=*/false, dprobs, kT, dqT);
    BatchedGemm(kv_len, dim, params.seq, batch_heads, scale, 0.0f,
                /*transpose_a=*/true, /*transpose_b=*/false, dprobs, qT, dkT);
    BatchedGemm(kv_len, dim, params.seq, batch_heads, 1.0f, 0.0f,
                /*transpose_a=*/true, /*transpose_b=*/false, probs, doutT, dvT);

    TransposeFromHeads(params.batch, params.seq, heads, dim, dqT, dq);
    TransposeFromHeads(params.batch, kv_len, kv_heads, dim, dkT, dk);
    TransposeFromHeads(params.batch, kv_len, kv_heads, dim, dvT, dv);
    return;
  }

  // Grouped-query fallback: one GEMM per (b, h), accumulating the shared
  // key/value gradients, so the key/value buffers start at zero.
  const std::size_t kv_count =
      static_cast<std::size_t>(params.batch) * kv_len * kv_heads * dim;
  Memset(dk, 0, kv_count * sizeof(ComputeType));
  Memset(dv, 0, kv_count * sizeof(ComputeType));
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      GemmScores(params, b, h, scale, q, k, scores);
      StridedGemm(params.seq, kv_len, dim, 1.0f, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/true, params.num_heads * dim, kv_heads * dim,
                  kv_len,
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  v + HeadBase(b, kv_len, kv_heads, kvh, dim),
                  dprobs + ScoreBase(params, b, h));
    }
  }
  cuda_backend::Launch(AttentionSoftmaxBackwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, stats, probs,
                       dprobs);

  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      const ComputeType* ds = dprobs + ScoreBase(params, b, h);
      const ComputeType* qh =
          q + HeadBase(b, params.seq, params.num_heads, h, dim);
      const ComputeType* kh = k + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dqh = dq + HeadBase(b, params.seq, params.num_heads, h, dim);
      ComputeType* dkh = dk + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dvh = dv + HeadBase(b, kv_len, kv_heads, kvh, dim);

      // dQ = dS @ K * scale.
      StridedGemm(params.seq, dim, kv_len, scale, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/false, kv_len, kv_heads * dim,
                  params.num_heads * dim, ds, kh, dqh);
      // dK += dS^T @ Q * scale.
      StridedGemm(kv_len, dim, params.seq, scale, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, ds, qh, dkh);
      // dV += P^T @ dOut.
      StridedGemm(kv_len, dim, params.seq, 1.0f, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, probs + ScoreBase(params, b, h),
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  dvh);
    }
  }
}

}  // namespace kernels
}  // namespace nanochat
