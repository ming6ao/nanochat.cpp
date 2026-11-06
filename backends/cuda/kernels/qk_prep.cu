// CUDA QkPrep family: fused RMSNorm -> RoPE -> scale (1.2 by default) on the
// query and key projections. One row is one (batch, seq, head): the row is
// normalized, then each (d, d + head_dim/2) pair is rotated by the fp32 rotary
// table and scaled, exactly like the CPU reference. A row runs on one block
// for every width, and on one warp at `head_dim` 128 (the model shape). Both
// schedules compute the same values in the same order; only the mapping of a
// row onto the block changes.
//
// The write is in place unless the caller sets `QkPrepParams::q_head_major`,
// which sends the prepared query to `q_out` in `[batch, heads, seq, dim]` and
// keeps every read in the model stream. That layout is what the batched
// attention products read, so the query transpose leaves the attention path.
//
// Backward contract (frozen seam): the q/k buffers handed to QkPrepBackward
// hold the *saved pre-norm projection outputs* on entry and are overwritten
// with the gradient with respect to those outputs. The upstream dq/dk are the
// gradients of the final (RoPE'd, scaled) activation.
//
// Both kernels carry the composition's arithmetic operation for operation, and
// not the reverse: `ops::QkPrepForward`/`ops::QkPrepBackward` in src/ops.cc
// stay the definition. The backward accumulates its two row statistics in
// double; both kernels derive `rstd` as `1.0f / std::sqrt(...)`, which is the
// `kRsqrt` pointwise code; and both round every product of the rotation
// separately.
//
// See docs/kernels.md and backends/cuda/kernels/device_utils.cuh.

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockSizeForDim;
using cuda_kernels::RowReduceSumSq;
using cuda_kernels::ToComputeDev;

// The forward's pair rotation, with every float operation rounded to nearest.
// The composition is the `kRotatePair` pointwise code, whose definition is
// `alpha * (x1 * c + x2 * s)` and `alpha * (-x1 * s + x2 * c)` with every
// multiply and the add rounded separately (spec/pointwise_ops.md section 3.2).
// nvcc contracts `x1 * c + x2 * s` into an FMA by default, which leaves one
// product unrounded, and that alone puts the fused forward an ulp from the
// composition at every pair. The ulp is not cosmetic: AdamW's
// `adam_eps` amplifies one ulp on `backout_lambda` into 1.23e-4 of parameter
// error, over the 1e-4 trajectory tolerance of `//tests:oracle_cuda_test`. The
// intrinsics hold the source form to one rounding per operation, because the
// composition is the definition and the kernel moves to it. The scale multiply
// stays at the caller, exactly where the composition applies it. The shared
// `RotatePairInPlace` keeps its plain expressions: the other rotary callers
// have not been measured and this family does not move them.
__device__ __forceinline__ void RotateForwardPair(float& x1, float& x2, float c,
                                                  float s) {
  const float t1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
  const float t2 = __fsub_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  x1 = t1;
  x2 = t2;
}

// The backward's rotation and combine, with every float operation rounded to
// nearest. `QkPrepBackwardKernel` reproduces the composition's arithmetic, and
// the composition rounds each float operation: the transposed rotation of
// `kRotatePair` is `alpha * (g1 * c - g2 * s)` with the two products rounded
// before the subtract. nvcc contracts the same expression into an FMA by
// default, which keeps the product unrounded, and that alone leaves the fused
// pair 1 or 2 ulp from the composition at every element of the model shape. The
// intrinsics hold the source form to the composition's rounding. The forward's
// rotation has its own helper for the same reason (see `RotateForwardPair`
// above).
__device__ __forceinline__ float RotateBackwardFirst(float g1, float c,
                                                     float g2, float s,
                                                     float scale) {
  return __fmul_rn(scale, __fsub_rn(__fmul_rn(g1, c), __fmul_rn(g2, s)));
}

__device__ __forceinline__ float RotateBackwardSecond(float g1, float s,
                                                      float g2, float c,
                                                      float scale) {
  return __fmul_rn(scale, __fadd_rn(__fmul_rn(g1, s), __fmul_rn(g2, c)));
}

__device__ __forceinline__ float CombineBackward(float r, float grad,
                                                 float coeff, float x) {
  return __fsub_rn(__fmul_rn(r, grad), __fmul_rn(coeff, x));
}

// The row base of the model stream, or of the head-major query when
// `head_major` is set. `row` is the model-stream row index
// `(b * seq + t) * heads + h`.
__device__ __forceinline__ std::size_t QueryRowBase(int row, int heads, int seq,
                                                    int dim, bool head_major) {
  if (!head_major) return static_cast<std::size_t>(row) * dim;
  const int h = row % heads;
  const int t = (row / heads) % seq;
  const int b = row / (seq * heads);
  return ((static_cast<std::size_t>(b) * heads + h) * seq + t) * dim;
}

// The base of one row of a head-major buffer `[batch, heads, seq, dim]`.
__device__ __forceinline__ std::size_t HeadMajorBase(int b, int t, int h,
                                                     int heads, int seq,
                                                     int dim) {
  return ((static_cast<std::size_t>(b) * heads + h) * seq + t) * dim;
}

// The destination heads of key/value head `kvh` when the prepared key is
// widened to the query head count: contiguous grouping, ascending, the map
// `KvHeadDev` reads and the transpose fold keeps.
__device__ __forceinline__ void HeadGroup(int kvh, int kv_heads, int dst_heads,
                                          int* first, int* last) {
  const int group = kv_heads > 0 ? dst_heads / kv_heads : 1;
  *first = kvh * group;
  *last = kvh == kv_heads - 1 ? dst_heads : *first + group;
}

// One row is (batch, seq, head); `heads` selects the tensor's head count. The
// row position feeds the shared rotary table. `out` is null for the in-place
// write and carries the head-major destination when the model asks for it.
__global__ void QkPrepForwardKernel(const QkPrepParams params,
                                    const float* __restrict__ cos,
                                    const float* __restrict__ sin,
                                    ComputeType* __restrict__ tensor,
                                    ComputeType* __restrict__ out, int heads,
                                    int dst_heads) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  ComputeType* rowp = tensor + static_cast<std::size_t>(row) * dim;
  const int b = row / (params.seq * heads);
  const int t = (row / heads) % params.seq;
  const int h = row % heads;
  int first = h;
  int last = h + 1;
  if (out != nullptr) HeadGroup(h, heads, dst_heads, &first, &last);

  const float sum_sq = RowReduceSumSq(rowp, dim);
  // The composition's `rstd`, not `rsqrtf`. `NormForward` runs the `kRsqrt`
  // pointwise code, which is `1.0f / std::sqrt` (backends/cpu/kernels.cc), and
  // it multiplies by the row's `1.0f / dim` before it adds `eps`; `RsqrtScale`
  // divides instead and calls `rsqrtf`. Each operation here is held to one
  // rounding so the value cannot move with the compiler's contraction choice.
  // Checked in the emitted PTX:
  // this form is `rcp.rn` / `mul.rn` / `add.rn` / `sqrt.rn` / `rcp.rn`, where
  // the old form was `rsqrt.approx.f32`.
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(__fadd_rn(__fmul_rn(sum_sq, inv_dim), params.eps));

  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    // Round the normalized value to the compute type *before* the rotation,
    // because the composition does: its RMSNorm writes a compute-type tensor
    // and the rotation reads that one. Rounding after the rotation, or keeping
    // the value in fp32 across it, is the more accurate arithmetic, but the
    // fused pair is held to the composition's arithmetic and not the other way
    // round.
    float x1 = AsFloatDev(ToComputeDev(AsFloatDev(rowp[d]) * r));
    float x2 = AsFloatDev(ToComputeDev(AsFloatDev(rowp[half + d]) * r));
    RotateForwardPair(x1, x2, crow[d], srow[d]);
    for (int dh = first; dh < last; ++dh) {
      ComputeType* dst =
          out != nullptr
              ? out + HeadMajorBase(b, t, dh, dst_heads, params.seq, dim)
              : rowp;
      dst[d] = ToComputeDev(x1 * params.scale);
      dst[half + d] = ToComputeDev(x2 * params.scale);
    }
  }
}

__global__ void QkPrepBackwardKernel(const QkPrepParams params,
                                     const float* __restrict__ cos,
                                     const float* __restrict__ sin,
                                     const ComputeType* __restrict__ grad,
                                     ComputeType* __restrict__ tensor,
                                     int heads, bool head_major) {
  const int row = blockIdx.x;
  const int dim = params.head_dim;
  const int half = dim / 2;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  ComputeType* rowp = tensor + base;
  const ComputeType* grow =
      grad + QueryRowBase(row, heads, params.seq, dim, head_major);

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * half;
  const float* srow = sin + static_cast<std::size_t>(t) * half;

  // Pull the upstream gradient back through the scale and the rotation, and
  // accumulate the two statistics the RMSNorm backward needs. The norm scale is
  // `1.0f / std::sqrt` of the float sum and the reciprocal is multiplied in
  // rather than divided, which is how the composition's `kRsqrt` step reads: a
  // `rsqrtf` would leave `r` one ulp away on every row, and AdamW's `adam_eps`
  // amplifies that past the trajectory tolerance.
  //
  // The two accumulators stay double here. The composition reduces through
  // `RowReduce`, whose device form accumulates in float, so the two agree to
  // within the declared bound rather than bit for bit; keeping double on the
  // fused side is the more accurate of the two and stays inside that bound. The
  // rotation itself stays float and rounded to nearest, as `kRotatePair`
  // defines it.
  double sum_sq = 0.0;
  double dot = 0.0;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = RotateBackwardFirst(g1, c, g2, s, params.scale);
    const float gd2 = RotateBackwardSecond(g1, s, g2, c, params.scale);
    sum_sq += static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
    dot += static_cast<double>(x1) * gd + static_cast<double>(x2) * gd2;
  }
  sum_sq = cuda_kernels::BlockReduceSum(sum_sq);
  dot = cuda_kernels::BlockReduceSum(dot);

  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
  const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;

  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float c = crow[d];
    const float s = srow[d];
    const float x1 = AsFloatDev(rowp[d]);
    const float x2 = AsFloatDev(rowp[half + d]);
    const float g1 = AsFloatDev(grow[d]);
    const float g2 = AsFloatDev(grow[half + d]);
    const float gd = RotateBackwardFirst(g1, c, g2, s, params.scale);
    const float gd2 = RotateBackwardSecond(g1, s, g2, c, params.scale);
    rowp[d] = ToComputeDev(CombineBackward(r, gd, coeff, x1));
    rowp[half + d] = ToComputeDev(CombineBackward(r, gd2, coeff, x2));
  }
}

// --- The warp-per-row schedule for `head_dim` 128 --------------------------
//
// The block kernels above run one block per row and take a block-wide
// reduction, which is two barriers at every level of the tree. At
// `head_dim` 128 that block is 128 threads and the pair loop uses only the low
// 64 of them, so half the block is idle and the row costs about nine barriers
// in the forward and twice that in the backward. The model shape pays that
// 163,840 times per layer.
//
// This schedule keeps the arithmetic and changes only the mapping. Lane `j`
// owns the elements `j`, `j + 32`, `j + 64` and `j + 96`, so its two rotation
// pairs are `(j, j + 64)` and `(j + 32, j + 96)`. Those are exactly the two
// pairs the 128-lane block reduction combines at its first two levels: level
// 64 pairs `i` with `i + 64`, and level 32 pairs the result for `i` with the
// result for `i + 32`. A lane sums its four values in that order, and the five
// warp shuffles are the tree's remaining five levels. The forward therefore
// moves no value and changes no rounding; it only replaces two shared-memory
// levels and their barriers with registers.
//
// The backward's block kernel runs its pair loop on the low 64 threads, so
// lanes 64 to 127 hold zero and the tree's level 64 adds only zeros. The warp
// schedule starts one level later: a lane holds the pairs `j` and `j + 32`,
// which is the level-32 combine, and the five shuffles are levels 16 to 1.
// Adding zero is exact, so dropping that level moves no value either.

constexpr int kQkWarpHeadDim = 128;
constexpr int kQkWarpHalf = kQkWarpHeadDim / 2;
constexpr int kQkWarpRowsPerBlock = 4;
constexpr int kQkWarpThreads = 32 * kQkWarpRowsPerBlock;

// Warp-wide sum over the 32 lanes. Each level adds `value + other`, the order
// the block reduction's `shared[i] += shared[i + offset]` keeps, and the
// butterfly leaves the same result in every lane.
template <typename T>
__device__ __forceinline__ T QkWarpSum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

int QkWarpGrid(int rows) {
  return (rows + kQkWarpRowsPerBlock - 1) / kQkWarpRowsPerBlock;
}

// `QkPrepForwardKernel` at `head_dim` 128, one warp per row. The squares use
// the round-to-nearest multiply, because the composition rounds each product
// on its own and a contraction would keep one of the two unrounded.
__global__ void __launch_bounds__(kQkWarpThreads) QkPrepForwardWarpKernel(
    const QkPrepParams params, const float* __restrict__ cos,
    const float* __restrict__ sin, ComputeType* __restrict__ tensor,
    ComputeType* __restrict__ out, int heads, int dst_heads) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kQkWarpRowsPerBlock + (threadIdx.x >> 5);
  if (row >= params.batch * params.seq * heads) return;
  const int dim = params.head_dim;
  ComputeType* rowp = tensor + static_cast<std::size_t>(row) * dim;
  const int b = row / (params.seq * heads);
  const int t = (row / heads) % params.seq;
  const int h = row % heads;
  int first = h;
  int last = h + 1;
  if (out != nullptr) HeadGroup(h, heads, dst_heads, &first, &last);
  const float* crow = cos + static_cast<std::size_t>(t) * kQkWarpHalf;
  const float* srow = sin + static_cast<std::size_t>(t) * kQkWarpHalf;

  // The four elements of this lane, read before any write so the two pairs of
  // a row can be transformed in either order.
  const float v0 = AsFloatDev(rowp[lane]);
  const float v1 = AsFloatDev(rowp[lane + kQkWarpHalf]);
  const float v2 = AsFloatDev(rowp[lane + 32]);
  const float v3 = AsFloatDev(rowp[lane + 32 + kQkWarpHalf]);

  const float sum_sq =
      QkWarpSum(__fadd_rn(__fadd_rn(__fmul_rn(v0, v0), __fmul_rn(v1, v1)),
                          __fadd_rn(__fmul_rn(v2, v2), __fmul_rn(v3, v3))));
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(__fadd_rn(__fmul_rn(sum_sq, inv_dim), params.eps));

  float x1 = AsFloatDev(ToComputeDev(v0 * r));
  float x2 = AsFloatDev(ToComputeDev(v1 * r));
  RotateForwardPair(x1, x2, crow[lane], srow[lane]);
  float x3 = AsFloatDev(ToComputeDev(v2 * r));
  float x4 = AsFloatDev(ToComputeDev(v3 * r));
  RotateForwardPair(x3, x4, crow[lane + 32], srow[lane + 32]);
  for (int dh = first; dh < last; ++dh) {
    ComputeType* dst = out != nullptr ? out + HeadMajorBase(b, t, dh, dst_heads,
                                                            params.seq, dim)
                                      : rowp;
    dst[lane] = ToComputeDev(x1 * params.scale);
    dst[lane + kQkWarpHalf] = ToComputeDev(x2 * params.scale);
    dst[lane + 32] = ToComputeDev(x3 * params.scale);
    dst[lane + 32 + kQkWarpHalf] = ToComputeDev(x4 * params.scale);
  }
}

// `QkPrepBackwardKernel` at `head_dim` 128, one warp per row. The pair values
// stay in registers across the reductions, so the second pass reads no saved
// row and no upstream gradient a second time.
__global__ void __launch_bounds__(kQkWarpThreads) QkPrepBackwardWarpKernel(
    const QkPrepParams params, const float* __restrict__ cos,
    const float* __restrict__ sin, const ComputeType* __restrict__ grad,
    ComputeType* __restrict__ tensor, int heads, bool head_major) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kQkWarpRowsPerBlock + (threadIdx.x >> 5);
  if (row >= params.batch * params.seq * heads) return;
  const int dim = params.head_dim;
  const std::size_t base = static_cast<std::size_t>(row) * dim;
  ComputeType* rowp = tensor + base;
  const ComputeType* grow =
      grad + QueryRowBase(row, heads, params.seq, dim, head_major);

  const int t = (row / heads) % params.seq;
  const float* crow = cos + static_cast<std::size_t>(t) * kQkWarpHalf;
  const float* srow = sin + static_cast<std::size_t>(t) * kQkWarpHalf;

  const float x1 = AsFloatDev(rowp[lane]);
  const float x2 = AsFloatDev(rowp[lane + kQkWarpHalf]);
  const float x3 = AsFloatDev(rowp[lane + 32]);
  const float x4 = AsFloatDev(rowp[lane + 32 + kQkWarpHalf]);
  const float g1 = AsFloatDev(grow[lane]);
  const float g2 = AsFloatDev(grow[lane + kQkWarpHalf]);
  const float g3 = AsFloatDev(grow[lane + 32]);
  const float g4 = AsFloatDev(grow[lane + 32 + kQkWarpHalf]);

  const float c0 = crow[lane];
  const float s0 = srow[lane];
  const float c1 = crow[lane + 32];
  const float s1 = srow[lane + 32];
  const float gd1 = RotateBackwardFirst(g1, c0, g2, s0, params.scale);
  const float gd2 = RotateBackwardSecond(g1, s0, g2, c0, params.scale);
  const float gd3 = RotateBackwardFirst(g3, c1, g4, s1, params.scale);
  const float gd4 = RotateBackwardSecond(g3, s1, g4, c1, params.scale);

  // The pair this lane owns, then the pair at `lane + 32`, which is the block
  // tree's level-32 combine. The double accumulators keep the composition's
  // precision argument (see the block kernel above).
  double sum_sq = static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
  double dot = static_cast<double>(x1) * gd1 + static_cast<double>(x2) * gd2;
  sum_sq += static_cast<double>(x3) * x3 + static_cast<double>(x4) * x4;
  dot += static_cast<double>(x3) * gd3 + static_cast<double>(x4) * gd4;
  sum_sq = QkWarpSum(sum_sq);
  dot = QkWarpSum(dot);

  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float r =
      1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
  const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;

  rowp[lane] = ToComputeDev(CombineBackward(r, gd1, coeff, x1));
  rowp[lane + kQkWarpHalf] = ToComputeDev(CombineBackward(r, gd2, coeff, x2));
  rowp[lane + 32] = ToComputeDev(CombineBackward(r, gd3, coeff, x3));
  rowp[lane + 32 + kQkWarpHalf] =
      ToComputeDev(CombineBackward(r, gd4, coeff, x4));
}

void LaunchQkPrepForward(const QkPrepParams& params, const float* cos,
                         const float* sin, ComputeType* tensor,
                         ComputeType* out, int heads, int dst_heads, int rows) {
  if (params.head_dim == kQkWarpHeadDim) {
    cuda_backend::Launch(QkPrepForwardWarpKernel, dim3(QkWarpGrid(rows)),
                         dim3(kQkWarpThreads), 0, params, cos, sin, tensor, out,
                         heads, dst_heads);
    return;
  }
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(QkPrepForwardKernel, dim3(rows), dim3(block), 0, params,
                       cos, sin, tensor, out, heads, dst_heads);
}

void LaunchQkPrepBackward(const QkPrepParams& params, const float* cos,
                          const float* sin, const ComputeType* grad,
                          ComputeType* tensor, int heads, int rows,
                          bool head_major) {
  if (params.head_dim == kQkWarpHeadDim) {
    cuda_backend::Launch(QkPrepBackwardWarpKernel, dim3(QkWarpGrid(rows)),
                         dim3(kQkWarpThreads), 0, params, cos, sin, grad,
                         tensor, heads, head_major);
    return;
  }
  const int block = BlockSizeForDim(params.head_dim);
  cuda_backend::Launch(QkPrepBackwardKernel, dim3(rows), dim3(block), 0, params,
                       cos, sin, grad, tensor, heads, head_major);
}

}  // namespace

void QkPrepForward(const QkPrepParams& params, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  const bool q_heads = params.q_head_major && params.q_out != nullptr;
  const int q_rows = params.batch * params.seq * params.num_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    LaunchQkPrepForward(params, cos, sin, q, q_heads ? params.q_out : nullptr,
                        params.num_heads, params.num_heads, q_rows);
  }
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (k_rows > 0 && params.head_dim > 0) {
    const bool k_heads = params.k_out != nullptr && params.num_heads > 0;
    LaunchQkPrepForward(params, cos, sin, k, k_heads ? params.k_out : nullptr,
                        params.num_kv_heads, params.num_heads, k_rows);
  }
}

void QkPrepBackward(const QkPrepParams& params, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  const int q_rows = params.batch * params.seq * params.num_heads;
  if (q_rows > 0 && params.head_dim > 0) {
    LaunchQkPrepBackward(params, cos, sin, dq, q, params.num_heads, q_rows,
                         params.q_head_major);
  }
  const int k_rows = params.batch * params.seq * params.num_kv_heads;
  if (k_rows > 0 && params.head_dim > 0) {
    LaunchQkPrepBackward(params, cos, sin, dk, k, params.num_kv_heads, k_rows,
                         false);
  }
}

}  // namespace kernels
}  // namespace nanochat
