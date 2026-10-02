// Local capture-safe mirrors of the decode-path seam ops. The bodies are ports
// of backends/cuda/kernels/{rms_norm,qk_prep,attention,pointwise,embedding}.cu
// and reuse the shared device helpers; the only difference is that every entry
// point takes an explicit stream so CUDA graph capture can record it.
//
// See dev/kernels/decode_kernels.h.

#include "dev/kernels/decode_kernels.h"

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::BlockSizeForDim;
using cuda_kernels::KvHeadCountDev;
using cuda_kernels::KvHeadDev;
using cuda_kernels::OnlineSoftmaxTile;
using cuda_kernels::RotatePairInPlace;
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::RsqrtScale;
using cuda_kernels::ToComputeDev;

__global__ void DecodeEmbeddingKernel(int tokens, int dim,
                                      const int* __restrict__ ids,
                                      const ComputeType* __restrict__ table,
                                      ComputeType* __restrict__ out) {
  const long long n = static_cast<long long>(tokens) * dim;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long idx =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const long long token = idx / dim;
    const int d = static_cast<int>(idx - token * dim);
    out[idx] = table[static_cast<long long>(ids[token]) * dim + d];
  }
}

__global__ void DecodeRmsNormKernel(const RmsNormParams params,
                                    const ComputeType* __restrict__ x,
                                    ComputeType* __restrict__ out,
                                    float* __restrict__ rstd) {
  const int row = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(row) * params.dim;
  const ComputeType* xr = x + base;
  ComputeType* orow = out + base;

  const float sum_sq = RowReduceSumSq(xr, params.dim);
  const float r = RsqrtScale(sum_sq, params.dim, params.eps);
  if (threadIdx.x == 0) rstd[row] = r;
  for (int d = threadIdx.x; d < params.dim; d += blockDim.x) {
    orow[d] = ToComputeDev(AsFloatDev(xr[d]) * r);
  }
}

__device__ __forceinline__ void DecodeQkPrepRow(
    const QkPrepParams& params, const float* __restrict__ cos,
    const float* __restrict__ sin, ComputeType* __restrict__ tensor, int row,
    int heads) {
  const int dim = params.head_dim;
  const int half = dim / 2;
  ComputeType* rowp = tensor + static_cast<std::size_t>(row) * dim;

  const float sum_sq = RowReduceSumSq(rowp, dim);
  const float r = RsqrtScale(sum_sq, dim, params.eps);

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    float x1 = AsFloatDev(rowp[d]) * r;
    float x2 = AsFloatDev(rowp[half + d]) * r;
    RotatePairInPlace(x1, x2, crow[d], srow[d]);
    rowp[d] = ToComputeDev(x1 * params.scale);
    rowp[half + d] = ToComputeDev(x2 * params.scale);
  }
}

__global__ void DecodeQkPrepKernel(const QkPrepParams params,
                                   const float* __restrict__ cos,
                                   const float* __restrict__ sin,
                                   ComputeType* __restrict__ q,
                                   ComputeType* __restrict__ k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int row = blockIdx.x;
  if (row < q_rows) {
    DecodeQkPrepRow(params, cos, sin, q, row, params.num_heads);
  } else {
    DecodeQkPrepRow(params, cos, sin, k, row - q_rows, params.num_kv_heads);
  }
}

__global__ void DecodeAttentionKernel(const AttentionParams params, float scale,
                                      const ComputeType* __restrict__ q,
                                      const ComputeType* __restrict__ k,
                                      const ComputeType* __restrict__ v,
                                      ComputeType* __restrict__ out,
                                      float* __restrict__ stats) {
  extern __shared__ float scratch[];
  const int row = blockIdx.x;
  const int t = row % params.seq;
  const int h = (row / params.seq) % params.num_heads;
  const int b = row / (params.seq * params.num_heads);
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = KvHeadCountDev(params);
  const int kvh = KvHeadDev(h, params.num_heads, params.num_kv_heads);
  const long long qpos = static_cast<long long>(kv_len) - params.seq + t;

  const long long qbase =
      ((static_cast<long long>(b) * params.seq + t) * params.num_heads + h) *
      params.head_dim;
  const long long kvbase =
      static_cast<long long>(b) * kv_len * kv_heads * params.head_dim;
  float* stats_row =
      stats +
      ((static_cast<long long>(b) * params.num_heads + h) * params.seq + t) * 2;

  OnlineSoftmaxTile(params, q + qbase, k + kvbase, v + kvbase, kv_len, kvh,
                    qpos, scale, out + qbase, stats_row, scratch);
}

__device__ __forceinline__ float DecodeSigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void DecodePointwiseKernel(const PointwiseOp op, int n,
                                      const ComputeType* __restrict__ a,
                                      const ComputeType* __restrict__ b,
                                      float alpha, float beta,
                                      ComputeType* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float av = AsFloatDev(a[i]);
  float result = 0.0f;
  switch (op) {
    case PointwiseOp::kScale:
      result = alpha * av;
      break;
    case PointwiseOp::kScaleAdd:
      result = alpha * av + beta * AsFloatDev(b[i]);
      break;
    case PointwiseOp::kGateMul:
      result = DecodeSigmoid(av) * AsFloatDev(b[i]);
      break;
    case PointwiseOp::kReluSquare: {
      const float relu = av > 0.0f ? av : 0.0f;
      result = relu * relu;
      break;
    }
    case PointwiseOp::kSoftcap: {
      const float cap = alpha;
      result = cap * tanhf(av / cap);
      break;
    }
  }
  out[i] = ToComputeDev(result);
}

inline int GridSize(int n, int block) {
  return n > 0 ? (n + block - 1) / block : 1;
}

}  // namespace

void DecodeEmbeddingFwd(cudaStream_t stream, int tokens, int dim,
                        const int* ids_dev, const ComputeType* table,
                        ComputeType* out) {
  if (tokens <= 0 || dim <= 0) return;
  constexpr int kThreads = 256;
  const long long n = static_cast<long long>(tokens) * dim;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  DecodeEmbeddingKernel<<<blocks, kThreads, 0, stream>>>(tokens, dim, ids_dev,
                                                         table, out);
}

void DecodeRmsNormFwd(cudaStream_t stream, int rows, int dim, float eps,
                      const ComputeType* x, ComputeType* out, float* rstd) {
  if (rows <= 0 || dim <= 0) return;
  RmsNormParams params;
  params.rows = rows;
  params.dim = dim;
  params.eps = eps;
  const int block = BlockSizeForDim(dim);
  DecodeRmsNormKernel<<<rows, block, 0, stream>>>(params, x, out, rstd);
}

void DecodeQkPrepFwd(cudaStream_t stream, const QkPrepParams& params,
                     const float* cos, const float* sin, ComputeType* q,
                     ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  const int rows = q_rows + k_rows;
  if (rows <= 0 || params.head_dim <= 0) return;
  const int block = BlockSizeForDim(params.head_dim);
  DecodeQkPrepKernel<<<rows, block, 0, stream>>>(params, cos, sin, q, k);
}

void DecodeAttentionFwd(cudaStream_t stream, const AttentionParams& params,
                        const ComputeType* q, const ComputeType* k,
                        const ComputeType* v, ComputeType* out, float* stats) {
  if (params.batch <= 0 || params.seq <= 0 || params.num_heads <= 0 ||
      params.head_dim <= 0) {
    return;
  }
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / sqrtf(static_cast<float>(params.head_dim));
  const int rows = params.batch * params.num_heads * params.seq;
  const int block = BlockSizeForDim(params.head_dim);
  const std::size_t shared = sizeof(float) * static_cast<std::size_t>(kv_len);
  DecodeAttentionKernel<<<rows, block, shared, stream>>>(params, scale, q, k, v,
                                                         out, stats);
}

void DecodePointwiseFwd(cudaStream_t stream, PointwiseOp op, int n,
                        const ComputeType* a, const ComputeType* b, float alpha,
                        float beta, ComputeType* out) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  DecodePointwiseKernel<<<GridSize(n, kThreads), kThreads, 0, stream>>>(
      op, n, a, b, alpha, beta, out);
}

}  // namespace dev
}  // namespace nanochat
