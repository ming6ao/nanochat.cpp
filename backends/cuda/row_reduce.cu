// CUDA RowReduce: the per-segment coalesced scan of the primitive tier
// (`include/nanochat/kernels.h`, seam 2, and spec/decisions.md D-001).
//
// One block per output scalar. Segment `s` is `n_out` segments away from the
// first, each `count` elements wide with an element `stride`, so the descriptor
// covers both shapes its three consumers need:
//
//   * RmsNorm   — one contiguous row: `count` = row width, `stride` = 1,
//                 `n_out` = rows.
//   * QkPrep    — a `head_dim` slice of a wider, densely packed row:
//                 `count` = head_dim, `stride` = 1, `n_out` = rows * heads
//                 (the packed heads make the segments consecutive).
//   * Classifier — one row of the vocabulary slice with an offset:
//                 `count` = vocab slice, `stride` = 1, `offset` = the slice's
//                 first column, `n_out` = rows.
//
// A non-unit `stride` is the interleaved case the descriptor also allows, and
// it is the only thing that makes the two shapes above different code.
//
// The block reads its segment with `threadIdx.x + k * blockDim.x`, so
// consecutive threads read consecutive elements when `stride` is one: the scan
// is coalesced and the segment stays inside one CTA, which is what
// `spec/decisions.md` D-001 means by "a per-segment reduce is a coalesced scan
// inside one work group".
//
// This is the highest-risk line in the seam: its result feeds a divisor
// (RmsNorm, QkPrep) and a log-sum-exp (Classifier), so a wrong index or a wrong
// combine is a silently wrong loss. The arrangement follows
// backends/cpu/kernels.cc: the working type is float, and the CPU reference
// accumulates in double, which is well inside the fp32 oracle tolerance
// (docs/testing.md). Nothing here writes `out` except the block that owns it,
// so the outputs are written once and never accumulated.
//
// See docs/kernels.md ("Reductions") and spec/capability_bits.md bit 9.

#include <cstdint>

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

// The value a segment with no element reduces to, per combine op. A sum of
// nothing is zero; a maximum of nothing is minus infinity, and so is a
// log-sum-exp of nothing (`log(0)`), which is the answer a caller can test for
// instead of receiving a NaN from `-inf - -inf`.
__device__ __forceinline__ float EmptySegmentValue(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSumSq:
    case ReduceOp::kSum:
      return 0.0f;
    case ReduceOp::kMax:
    case ReduceOp::kLogSumExp:
      return -CUDART_INF_F;
  }
  return 0.0f;
}

// The element `j` of segment `seg`, from the descriptor's three fields: a base
// pointer, a stride, and a length.
__device__ __forceinline__ const ComputeType* SegmentElement(
    const ComputeType* a, const RowReduceParams& p, int seg, int j) {
  const std::int64_t base =
      p.offset + static_cast<std::int64_t>(seg) * p.count * p.stride;
  return a + base + static_cast<std::int64_t>(j) * p.stride;
}

__global__ void RowReduceKernel(RowReduceParams p,
                                const ComputeType* __restrict__ a,
                                float* __restrict__ out) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;

  if (p.count <= 0 || a == nullptr) {
    // No element to combine. Every thread takes the same branch, so the block
    // reduction helpers below are never entered half-way.
    if (tid == 0) out[seg] = EmptySegmentValue(p.op);
    return;
  }

  switch (p.op) {
    case ReduceOp::kSumSq: {
      float sum = 0.0f;
      for (int j = tid; j < p.count; j += blockDim.x) {
        const float v = AsFloatDev(*SegmentElement(a, p, seg, j));
        sum += v * v;
      }
      const float total = BlockReduceSum(sum);
      if (tid == 0) out[seg] = total;
      return;
    }
    case ReduceOp::kSum: {
      float sum = 0.0f;
      for (int j = tid; j < p.count; j += blockDim.x) {
        sum += AsFloatDev(*SegmentElement(a, p, seg, j));
      }
      const float total = BlockReduceSum(sum);
      if (tid == 0) out[seg] = total;
      return;
    }
    case ReduceOp::kMax: {
      float local = -CUDART_INF_F;
      for (int j = tid; j < p.count; j += blockDim.x) {
        local = fmaxf(local, AsFloatDev(*SegmentElement(a, p, seg, j)));
      }
      const float total = BlockReduceMax(local);
      if (tid == 0) out[seg] = total;
      return;
    }
    case ReduceOp::kLogSumExp: {
      // Max-stable, in two passes: the segment maximum, then the sum of the
      // exponentials shifted by it. The shift is what keeps a row of large
      // logits finite, and the Classifier's log-sum-exp is exactly this.
      float local = -CUDART_INF_F;
      for (int j = tid; j < p.count; j += blockDim.x) {
        local = fmaxf(local, AsFloatDev(*SegmentElement(a, p, seg, j)));
      }
      const float row_max = BlockReduceMax(local);
      if (row_max == -CUDART_INF_F) {
        // Every element is -inf (or the segment is empty): no finite maximum to
        // shift by, and the answer is -inf rather than a NaN.
        if (tid == 0) out[seg] = -CUDART_INF_F;
        return;
      }
      float sum = 0.0f;
      for (int j = tid; j < p.count; j += blockDim.x) {
        sum += expf(AsFloatDev(*SegmentElement(a, p, seg, j)) - row_max);
      }
      const float total = BlockReduceSum(sum);
      if (tid == 0) out[seg] = row_max + logf(total);
      return;
    }
  }
}

}  // namespace

void RowReduce(const RowReduceParams& params, const ComputeType* a,
               float* out) {
  if (params.n_out <= 0 || out == nullptr) return;
  if (params.count > 0 && a == nullptr) return;
  // One block per segment. The block size follows the segment width: a short
  // row still gets a full warp (the block reduction helpers require a
  // power-of-two block, and a warp is the smallest useful one), and a long row
  // caps the block so the grid keeps occupancy.
  const int block =
      cuda_kernels::BlockSizeForDim(params.count > 0 ? params.count : 1);
  cuda_backend::Launch(RowReduceKernel, dim3(params.n_out), dim3(block), 0,
                       params, a, out);
}

}  // namespace kernels
}  // namespace nanochat
