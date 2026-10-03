// CUDA Attention family: causal / sliding-window / grouped-query attention,
// forward and backward, plus the saved softmax statistics
// `[batch, num_heads, seq, 2] = (max, sum_exp)`.
//
// The query-key scores, the probability-value product, and every backward
// product are batched GEMMs through the cuBLAS seam; only the softmax and its
// gradient run as custom row kernels. This is the same decomposition the
// PyTorch math backend uses, and it matters on Pascal: a block-per-query-row
// kernel re-reads the whole K/V row set once per query row, so its traffic is
// `O(seq^2 * head_dim)` and it runs far below the memory ceiling. A GEMM tiles
// the `seq x seq` output and reuses each K/V tile across the query rows, which
// makes the K/V traffic `O(seq * head_dim)`.
//
// Layout is the model's `[batch, seq, heads, head_dim]` for q/k/v/out, so the
// per-head row stride is `heads * head_dim` (or `kv_heads * head_dim` for
// k/v), and the score/probability matrices are `[batch, heads, seq, kv_len]`.
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

// One `C = alpha * op(A) * op(B) + beta * C`. `GemmMode` is descriptive only
// (the seam reduces every mode to this product), so the host wrappers below
// all pass kForward and let the operands carry the meaning.
void BatchedGemm(int m, int n, int k, float alpha, float beta, bool transpose_a,
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

// Reusable device scratch for the `[batch, heads, seq, kv_len]` matrices:
// `scores` (QK^T), `probs` (softmax output), and `dprobs` (dP, then dS). The
// allocation is cached across calls because attention runs once per layer and
// a cudaMalloc/free per call would dominate. The training graph is
// single-threaded, so no locking is needed.
ComputeType* ScratchBuffer(std::size_t count) {
  static ComputeType* buffer = nullptr;
  static std::size_t capacity = 0;
  if (count > capacity) {
    if (buffer != nullptr) Free(buffer);
    buffer = static_cast<ComputeType*>(Alloc(count * sizeof(ComputeType)));
    capacity = count;
  }
  return buffer;
}

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

// scores = Q @ K^T * scale, for one (b, h).
void GemmScores(const AttentionParams& p, int b, int h, float scale,
                const ComputeType* q, const ComputeType* k,
                ComputeType* scores) {
  const int dim = p.head_dim;
  const int kv_len = p.kv_len > 0 ? p.kv_len : p.seq;
  const int kv_heads = KvHeadCountDev(p);
  const int kvh = KvHeadDev(h, p.num_heads, p.num_kv_heads);
  BatchedGemm(p.seq, kv_len, dim, scale, 0.0f, /*transpose_a=*/false,
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
  BatchedGemm(p.seq, dim, kv_len, 1.0f, 0.0f, /*transpose_a=*/false,
              /*transpose_b=*/false, kv_len, kv_heads * dim, p.num_heads * dim,
              probs + ScoreBase(p, b, h),
              v + HeadBase(b, kv_len, kv_heads, kvh, dim),
              out + HeadBase(b, p.seq, p.num_heads, h, dim));
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
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(params.head_dim));
  const std::size_t matrix = static_cast<std::size_t>(params.batch) *
                             params.num_heads * params.seq * kv_len;
  ComputeType* scores = ScratchBuffer(3 * matrix);
  ComputeType* probs = scores + matrix;

  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < params.num_heads; ++h) {
      GemmScores(params, b, h, scale, q, k, scores);
    }
  }
  const int rows = params.batch * params.num_heads * params.seq;
  cuda_backend::Launch(AttentionSoftmaxForwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, probs, stats);
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < params.num_heads; ++h) {
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
  const int kv_heads =
      params.num_kv_heads > 0 ? params.num_kv_heads : params.num_heads;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(dim));

  // dk/dv accumulate across the grouped query heads, so they start at zero.
  // dq is written with beta = 0 and needs no clear.
  const std::size_t kv_count =
      static_cast<std::size_t>(params.batch) * kv_len * kv_heads * dim;
  Memset(dk, 0, kv_count * sizeof(ComputeType));
  Memset(dv, 0, kv_count * sizeof(ComputeType));

  const std::size_t matrix = static_cast<std::size_t>(params.batch) *
                             params.num_heads * params.seq * kv_len;
  ComputeType* scores = ScratchBuffer(3 * matrix);
  ComputeType* probs = scores + matrix;
  ComputeType* dprobs = probs + matrix;

  // Recompute the scores, then dP = dOut @ V^T.
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < params.num_heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      GemmScores(params, b, h, scale, q, k, scores);
      BatchedGemm(params.seq, kv_len, dim, 1.0f, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/true, params.num_heads * dim, kv_heads * dim,
                  kv_len,
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  v + HeadBase(b, kv_len, kv_heads, kvh, dim),
                  dprobs + ScoreBase(params, b, h));
    }
  }

  const int rows = params.batch * params.num_heads * params.seq;
  cuda_backend::Launch(AttentionSoftmaxBackwardKernel, dim3(rows),
                       dim3(kSoftmaxThreads), 0, params, scores, stats, probs,
                       dprobs);

  // dQ = dS @ K * scale; dK += dS^T @ Q * scale; dV += P^T @ dOut.
  for (int b = 0; b < params.batch; ++b) {
    for (int h = 0; h < params.num_heads; ++h) {
      const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
      const ComputeType* ds = dprobs + ScoreBase(params, b, h);
      const ComputeType* qh =
          q + HeadBase(b, params.seq, params.num_heads, h, dim);
      const ComputeType* kh = k + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dqh = dq + HeadBase(b, params.seq, params.num_heads, h, dim);
      ComputeType* dkh = dk + HeadBase(b, kv_len, kv_heads, kvh, dim);
      ComputeType* dvh = dv + HeadBase(b, kv_len, kv_heads, kvh, dim);

      // dQ = dS @ K * scale.
      BatchedGemm(params.seq, dim, kv_len, scale, 0.0f, /*transpose_a=*/false,
                  /*transpose_b=*/false, kv_len, kv_heads * dim,
                  params.num_heads * dim, ds, kh, dqh);
      // dK += dS^T @ Q * scale.
      BatchedGemm(kv_len, dim, params.seq, scale, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, ds, qh, dkh);
      // dV += P^T @ dOut.
      BatchedGemm(kv_len, dim, params.seq, 1.0f, 1.0f, /*transpose_a=*/true,
                  /*transpose_b=*/false, kv_len, params.num_heads * dim,
                  kv_heads * dim, probs + ScoreBase(params, b, h),
                  dout + HeadBase(b, params.seq, params.num_heads, h, dim),
                  dvh);
    }
  }
}

}  // namespace kernels
}  // namespace nanochat
