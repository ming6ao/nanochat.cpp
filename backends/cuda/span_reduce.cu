// CUDA SpanReduce: the device-wide ragged reduction of the primitive tier
// (`include/nanochat/kernels.h`, seam 2, spec/decisions.md D-001).
//
// One scalar out of a *ragged* list: `num_spans` segments whose lengths differ,
// where the segment pointers and the lengths both live in device memory, so the
// caller builds the list without a host round trip. `other` is a second list of
// the same ragged shape; when it is not null the reduction is the per-segment
// dot product of the two, which is what makes one entry point serve a plain sum
// (`GradNorm`'s spans) and a dot (`ScalarDot`'s operands).
//
// This is deliberately a different kernel from `row_reduce.cu`: a per-segment
// reduce is a coalesced scan inside one work group, and this is a device-wide
// reduction over a list the device reads itself. The two are different
// algorithms with different correctness risks, and a device may run them on
// different units (docs/model-and-device-bringup.md section 7.2). Merging them
// would put the strict per-row gate and the tolerant cross-span gate on one
// code path.
//
// Shape of the launch, mirroring `GradNorm` in kernels/grad_norm.cu:
//
//   1. A partial kernel, launched with `blockIdx.y` selecting the span, so the
//      whole list is one launch. A span wider than the x grid is covered by the
//      grid-stride loop; the surplus blocks of a narrow span contribute zero.
//   2. A one-thread finalize kernel that applies `scale`, honours `accumulate`,
//      publishes `*out`, and resets the accumulator for the next call.
//
// The accumulator is a `__device__` global rather than a fresh allocation so
// the entry point needs no scratch buffer and no host round trip. The backend
// runs everything on one stream, so the launches are ordered and a single
// accumulator is race-free.
//
// The order across spans is unspecified, so this is not the reduction a
// bit-exact gate rides on (`RowReduce` is). The working type is double, which
// is the type the CPU reference accumulates in, so the two agree to float
// rounding and the gate can stay a tolerance rather than a byte comparison.
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
using cuda_kernels::BlockReduceSum;

// The cross-span accumulator. Named for this entry point alone, so `GradNorm`'s
// and `ScalarDot`'s accumulators (kernels/grad_norm.cu) cannot interfere with
// it even when a caller interleaves the three.
__device__ double g_span_reduce_sum = 0.0;

// `spans`/`other` point at the first span of the chunk and `counts` at its
// first count; the kernel reads nothing from the host.
__global__ void SpanReducePartialKernel(
    const ComputeType* const* __restrict__ spans,
    const ComputeType* const* __restrict__ other,
    const int* __restrict__ counts) {
  const int span = blockIdx.y;
  const int n = counts[span];
  const ComputeType* segment = spans[span];
  // A zero-length span is legal (a parameter with no element, a disabled
  // branch), and it contributes zero. Every thread of the block takes the same
  // branch because `n` and `segment` are uniform, so the block reduction below
  // is never entered half-way.
  if (segment == nullptr || n <= 0) return;
  const ComputeType* second = other != nullptr ? other[span] : nullptr;

  double local = 0.0;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double v = AsFloatDev(segment[i]);
    local +=
        second != nullptr ? v * static_cast<double>(AsFloatDev(second[i])) : v;
  }
  const double block_sum = BlockReduceSum(local);
  if (threadIdx.x == 0) atomicAdd(&g_span_reduce_sum, block_sum);
}

__global__ void SpanReduceFinalizeKernel(float* __restrict__ out, float scale,
                                         bool accumulate) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float value = scale * static_cast<float>(g_span_reduce_sum);
  out[0] = accumulate ? out[0] + value : value;
  // Reset for the next invocation; the stream ordering makes this safe.
  g_span_reduce_sum = 0.0;
}

}  // namespace

void SpanReduce(const SpanReduceParams& params, const ComputeType* const* spans,
                const ComputeType* const* other, float* out) {
  if (params.num_spans <= 0 || spans == nullptr || params.counts == nullptr ||
      out == nullptr) {
    return;
  }
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  // The reduction is memory-bound, so a handful of blocks per span already
  // saturates the device; a block count that tracked the span count would only
  // add launch and atomics overhead. The budget is spread over the chunk
  // instead, exactly as `GradNorm` does it.
  constexpr int kBlockBudget = 4096;
  // `gridDim.y` is 16-bit, so a list larger than that is reduced in chunks,
  // each accumulating into the same device accumulator.
  constexpr int kMaxSpanChunk = 32768;

  for (int start = 0; start < params.num_spans; start += kMaxSpanChunk) {
    const int remaining = params.num_spans - start;
    const int chunk = remaining < kMaxSpanChunk ? remaining : kMaxSpanChunk;
    int grid = kBlockBudget / chunk;
    if (grid > kMaxBlocks) grid = kMaxBlocks;
    if (grid < 1) grid = 1;
    cuda_backend::Launch(SpanReducePartialKernel, dim3(grid, chunk, 1),
                         dim3(kThreads), 0, spans + start,
                         other != nullptr ? other + start : nullptr,
                         params.counts + start);
  }
  cuda_backend::Launch(SpanReduceFinalizeKernel, dim3(1), dim3(1), 0, out,
                       params.scale, params.accumulate);
}

}  // namespace kernels
}  // namespace nanochat
