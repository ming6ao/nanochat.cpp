#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_

// Shared device-side helpers for the row/elementwise families (RmsNorm,
// QkPrep, Pointwise, GlobalNorm) and the sequence families (Attention,
// Classifier, Embedding). They remove duplication *without* merging kernel
// boundaries: each .cu still owns its own entry points, and the helpers below
// are header-only so they inline into every caller.
//
// The arithmetic mirrors backends/cpu/kernels.cc exactly: storage is
// ComputeType (float or Fp16), the working type is always float, and the
// reduction runs in float. The CPU reference accumulates in double; the device
// uses float, which is well inside the fp32 oracle tolerance (docs/testing.md).
//
// See docs/kernels.md ("Shared device helpers") and docs/backends.md.

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"
#if defined(NANOCHAT_PRECISION_FP16)
#include <cuda_fp16.h>
#endif

namespace nanochat {
namespace cuda_kernels {

// Largest block the block-wide reduction below supports. A block declares a
// shared array of this size, so every launch must stay at or below it.
constexpr int kMaxBlockThreads = 1024;

// Storage <-> float at the device boundary. Fp16 is bit-compatible with
// __half, so the conversion goes through the CUDA intrinsics rather than a
// reinterpret_cast.
#if defined(NANOCHAT_PRECISION_FP16)
__device__ __forceinline__ float AsFloatDev(ComputeType v) {
  return __half2float(__ushort_as_half(v.bits));
}

__device__ __forceinline__ ComputeType ToComputeDev(float v) {
  ComputeType out;
  out.bits = __half_as_ushort(__float2half_rn(v));
  return out;
}
#else
__device__ __forceinline__ float AsFloatDev(ComputeType v) { return v; }

__device__ __forceinline__ ComputeType ToComputeDev(float v) { return v; }
#endif

// Atomic add of a float into ComputeType storage. fp32 uses the native atomic.
// fp16 uses the native scalar __half atomic on sm_70+, where the compiler
// exposes one. Pascal (sm_60/sm_61) has neither a scalar __half atomicAdd nor
// a 16-bit atomicCAS, so the fallback packs the target half into its aligned
// 32-bit word and CASes the word. Only the target 16 bits change; the
// neighbouring half is written back unchanged, which keeps the operation safe
// even when the pair straddles a workspace slot boundary.
//
// The attention backward needs this because grouped-query attention lets
// several query heads accumulate into the same key/value gradient row.
__device__ __forceinline__ void AtomicAddDev(ComputeType* addr, float value) {
#if defined(NANOCHAT_PRECISION_FP16)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  atomicAdd(reinterpret_cast<__half*>(addr), __float2half_rn(value));
#else
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(addr);
  unsigned int* word = reinterpret_cast<unsigned int*>(
      address & ~static_cast<std::uintptr_t>(3));
  const unsigned int shift = (address & 2u) != 0 ? 16u : 0u;
  const unsigned int mask = 0xffffu << shift;
  unsigned int assumed;
  unsigned int old = *word;
  do {
    assumed = old;
    const __half current = __ushort_as_half(
        static_cast<unsigned short>((assumed >> shift) & 0xffffu));
    const unsigned int updated = static_cast<unsigned int>(
        __half_as_ushort(__float2half_rn(__half2float(current) + value)));
    old = atomicCAS(word, assumed, (assumed & ~mask) | (updated << shift));
  } while (assumed != old);
#endif
#else
  atomicAdd(addr, value);
#endif
}

// Block-wide sum over `val`. Every thread in the block must call it, and the
// result is broadcast back to every thread through shared memory. `blockDim.x`
// must be a power of two and at most kMaxBlockThreads.
template <typename T>
__device__ __forceinline__ T BlockReduceSum(T val) {
  __shared__ T shared[kMaxBlockThreads];
  const int tid = threadIdx.x;
  // The array is reused by every call in the kernel. Barrier before writing so
  // a second reduction cannot overwrite `shared[0]` before every thread has
  // consumed the previous result (a real, nondeterministic race observed in
  // QkPrep backward, which reduces `sum_sq` and then `dot`).
  __syncthreads();
  shared[tid] = val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }
  return shared[0];
}

// Block-wide maximum, mirroring BlockReduceSum. `blockDim.x` must be a power
// of two and at most kMaxBlockThreads. Used by the attention and classifier
// row reductions (the softmax stabilizer).
template <typename T>
__device__ __forceinline__ T BlockReduceMax(T val) {
  __shared__ T shared[kMaxBlockThreads];
  const int tid = threadIdx.x;
  // See BlockReduceSum: guard the reuse of the shared array across calls.
  __syncthreads();
  shared[tid] = val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
    __syncthreads();
  }
  return shared[0];
}

// Sum of squares of one row, reduced across the calling block. Used by
// RmsNorm and QkPrep to recover the normalization statistic.
__device__ __forceinline__ float RowReduceSumSq(const ComputeType* row,
                                                int dim) {
  float sum = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = AsFloatDev(row[d]);
    sum += v * v;
  }
  return BlockReduceSum(sum);
}

// Dot product of two rows, reduced across the calling block. Used by the
// RmsNorm and QkPrep backward passes.
__device__ __forceinline__ float RowReduceDot(const ComputeType* x,
                                              const ComputeType* y, int dim) {
  float dot = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    dot += AsFloatDev(x[d]) * AsFloatDev(y[d]);
  }
  return BlockReduceSum(dot);
}

// 1 / sqrt(mean(square) + eps), the RMSNorm scale for one row.
__device__ __forceinline__ float RsqrtScale(float sum_sq, int dim, float eps) {
  return rsqrtf(sum_sq / static_cast<float>(dim) + eps);
}

// In-place RoPE rotation of one (x1, x2) pair. The sign convention matches the
// CPU reference: (x1*c + x2*s, -x1*s + x2*c).
__device__ __forceinline__ void RotatePairInPlace(float& x1, float& x2, float c,
                                                  float s) {
  const float t1 = x1 * c + x2 * s;
  const float t2 = -x1 * s + x2 * c;
  x1 = t1;
  x2 = t2;
}

// ---------------------------------------------------------------------------
// Attention (shared by the forward and backward tiles)
// ---------------------------------------------------------------------------

// Contiguous-grouping GQA head mapping, matching the CPU reference: each
// key/value head serves `num_heads / num_kv_heads` consecutive query heads.
__device__ __forceinline__ int KvHeadDev(int h, int num_heads,
                                         int num_kv_heads) {
  if (num_kv_heads <= 0 || num_kv_heads >= num_heads) return h;
  const int group = num_heads / num_kv_heads;
  const int kv = h / (group > 0 ? group : 1);
  return kv < num_kv_heads ? kv : num_kv_heads - 1;
}

// Effective key/value head count: `num_kv_heads == 0` means multi-head (no
// grouping), the same default the CPU reference uses.
__device__ __forceinline__ int KvHeadCountDev(const AttentionParams& p) {
  return p.num_kv_heads > 0 ? p.num_kv_heads : p.num_heads;
}

// Whether query row at absolute position `qpos` may attend to key `j`. Mirrors
// the CPU reference's KeyAllowed: causal mask plus the sliding window.
__device__ __forceinline__ bool KeyAllowedDev(const AttentionParams& p,
                                              long long qpos, long long j) {
  if (p.causal && j > qpos) return false;
  if (p.window_left >= 0 && qpos - j > p.window_left) return false;
  if (p.window_right >= 0 && j - qpos > p.window_right) return false;
  return true;
}

// Dot product of a query row and one key row.
__device__ __forceinline__ float AttentionDot(const ComputeType* q_row,
                                              const ComputeType* k_row,
                                              int dim) {
  float dot = 0.0f;
  for (int d = 0; d < dim; ++d) {
    dot += AsFloatDev(q_row[d]) * AsFloatDev(k_row[d]);
  }
  return dot;
}

// Online softmax tile: the forward pass for one query row, cooperatively
// computed by the calling block. `k`/`v` point at the batch's key/value base.
// The row max and sum_exp are reduced across the block and written to
// `stats_row`; each thread owns output dimensions `tid, tid + blockDim, ...`.
//
// The scaled query-key score is computed once per key and cached in the
// caller-provided `scratch` buffer (at least `kv_len` floats of dynamic shared
// memory), then its exponential is cached back into the same buffer. This
// removes the former `O(head_dim)` redundancy where pass 3 recomputed the full
// dot product for every owned output dimension. The arithmetic and the
// per-thread accumulation order are unchanged from the recompute form.
__device__ __forceinline__ void OnlineSoftmaxTile(
    const AttentionParams& p, const ComputeType* q_row, const ComputeType* k,
    const ComputeType* v, int kv_len, int kvh, long long qpos, float scale,
    ComputeType* out_row, float* stats_row, float* scratch) {
  const int dim = p.head_dim;
  const int kv_heads = KvHeadCountDev(p);
  const int tid = threadIdx.x;

  // Pass 1: compute the score once per key, cache it, and reduce the maximum.
  float local_max = -CUDART_INF_F;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    if (!KeyAllowedDev(p, qpos, j)) continue;
    const ComputeType* k_row =
        k + (static_cast<long long>(j) * kv_heads + kvh) * dim;
    const float score = scale * AttentionDot(q_row, k_row, dim);
    scratch[j] = score;
    local_max = fmaxf(local_max, score);
  }
  const float row_max = BlockReduceMax(local_max);
  if (row_max == -CUDART_INF_F) {
    // No visible key (an empty sliding window); the CPU writes zeros and a
    // zeroed statistic pair.
    for (int d = tid; d < dim; d += blockDim.x) {
      out_row[d] = ToComputeDev(0.0f);
    }
    if (tid == 0) {
      stats_row[0] = 0.0f;
      stats_row[1] = 0.0f;
    }
    return;
  }

  // Pass 2: exponentiate the cached scores in place and sum them.
  float local_sum = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    if (!KeyAllowedDev(p, qpos, j)) continue;
    const float weight = expf(scratch[j] - row_max);
    scratch[j] = weight;
    local_sum += weight;
  }
  const float sum_exp = BlockReduceSum(local_sum);
  const float inv = 1.0f / sum_exp;

  // Pass 3: the unnormalised output, per owned dimension, reading the cached
  // exponentials. No dot product is recomputed.
  for (int d = tid; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < kv_len; ++j) {
      if (!KeyAllowedDev(p, qpos, j)) continue;
      const ComputeType* v_row =
          v + (static_cast<long long>(j) * kv_heads + kvh) * dim;
      acc += scratch[j] * AsFloatDev(v_row[d]);
    }
    out_row[d] = ToComputeDev(acc * inv);
  }
  if (tid == 0) {
    stats_row[0] = row_max;
    stats_row[1] = sum_exp;
  }
}

// Softmax gradient tile: the backward pass for one query row. It forms the
// probabilities from the saved statistics and applies the
// `p * (dp - sum(p * dp))` softmax Jacobian, matching the CPU reference.
//
// The probability `p_j` and the value-weighted `dp_j = dout . v_j` are
// computed once per key and cached in the caller-provided `scratch` buffer (at
// least `2 * kv_len` floats of dynamic shared memory, laid out as `kv_len`
// probabilities followed by `kv_len` dp values). Pass 2 reads them instead of
// recomputing the dot product and `dout . v` for every owned dimension, which
// removes the former `O(head_dim)` redundancy. The arithmetic and the
// per-thread accumulation order are unchanged from the recompute form.
//
// The key/value gradients are atomic because grouped-query attention shares
// them across query heads; `dk`/`dv` point at the batch's base (already zeroed
// by the entry point).
__device__ __forceinline__ void SoftmaxGradTile(
    const AttentionParams& p, const ComputeType* q_row, const ComputeType* k,
    const ComputeType* v, int kv_len, int kvh, long long qpos, float scale,
    float row_max, float sum_exp, const ComputeType* dout_row,
    ComputeType* dq_row, ComputeType* dk, ComputeType* dv, float* scratch) {
  const int dim = p.head_dim;
  const int kv_heads = KvHeadCountDev(p);
  const int tid = threadIdx.x;
  if (sum_exp <= 0.0f) return;  // no visible key; dq/dk/dv stay zero
  const float inv = 1.0f / sum_exp;
  float* p_scratch = scratch;
  float* dp_scratch = scratch + kv_len;

  // Pass 1: compute `p_j` and `dp_j` once per key, cache both, and accumulate
  // the probability-weighted `dp`.
  float local_wdp = 0.0f;
  for (int j = tid; j < kv_len; j += blockDim.x) {
    if (!KeyAllowedDev(p, qpos, j)) continue;
    const ComputeType* k_row =
        k + (static_cast<long long>(j) * kv_heads + kvh) * dim;
    const ComputeType* v_row =
        v + (static_cast<long long>(j) * kv_heads + kvh) * dim;
    const float pj =
        expf(scale * AttentionDot(q_row, k_row, dim) - row_max) * inv;
    float dp = 0.0f;
    for (int d = 0; d < dim; ++d) {
      dp += AsFloatDev(dout_row[d]) * AsFloatDev(v_row[d]);
    }
    p_scratch[j] = pj;
    dp_scratch[j] = dp;
    local_wdp += pj * dp;
  }
  const float weighted_dp = BlockReduceSum(local_wdp);

  // Pass 2: accumulate the query, key, and value gradients from the cached
  // values. Each thread owns dimensions `tid, tid + blockDim, ...` and scans
  // every key, so `dq_row[d]` is private to its owning thread while the
  // key/value gradients use atomics (the grouped-query group shares them
  // across blocks).
  for (int d = tid; d < dim; d += blockDim.x) {
    float dq_acc = 0.0f;
    for (int j = 0; j < kv_len; ++j) {
      if (!KeyAllowedDev(p, qpos, j)) continue;
      const ComputeType* k_row =
          k + (static_cast<long long>(j) * kv_heads + kvh) * dim;
      const float pj = p_scratch[j];
      const float dp = dp_scratch[j];
      const float dscale = pj * (dp - weighted_dp) * scale;
      ComputeType* dk_row =
          dk + (static_cast<long long>(j) * kv_heads + kvh) * dim;
      ComputeType* dv_row =
          dv + (static_cast<long long>(j) * kv_heads + kvh) * dim;
      dq_acc += dscale * AsFloatDev(k_row[d]);
      AtomicAddDev(&dk_row[d], dscale * AsFloatDev(q_row[d]));
      AtomicAddDev(&dv_row[d], pj * AsFloatDev(dout_row[d]));
    }
    dq_row[d] = ToComputeDev(AsFloatDev(dq_row[d]) + dq_acc);
  }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

// Row maximum and sum of exponentials of the softcapped logits, reduced across
// the calling block. Only the first `n` columns are read; the padded tail is
// never touched. The two outputs are broadcast to every thread.
__device__ __forceinline__ void RowLogSumExp(const ComputeType* row, int n,
                                             float cap, float* out_max,
                                             float* out_sum) {
  float local_max = -CUDART_INF_F;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    local_max = fmaxf(local_max, cap * tanhf(AsFloatDev(row[j]) / cap));
  }
  const float row_max = BlockReduceMax(local_max);
  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    local_sum += expf(cap * tanhf(AsFloatDev(row[j]) / cap) - row_max);
  }
  *out_max = row_max;
  *out_sum = BlockReduceSum(local_sum);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

// row[d] += value[d] for the dimensions this thread owns. Used by the
// embedding scatter-add; the row must already have been zeroed.
__device__ __forceinline__ void ScatterAddRow(ComputeType* row, int dim,
                                              const ComputeType* value) {
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    AtomicAddDev(&row[d], AsFloatDev(value[d]));
  }
}

// Host-side power-of-two block size for a row of `dim` elements. Rows shorter
// than a warp still get a full warp so BlockReduceSum stays valid; long rows
// cap the block and stride.
inline int BlockSizeForDim(int dim) {
  int block = 32;
  while (block < dim && block < 256) block <<= 1;
  return block;
}

}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_DEVICE_UTILS_CUH_
