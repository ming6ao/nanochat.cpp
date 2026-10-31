// CUDA Pointwise family: the nine op codes of spec/pointwise_ops.md -- scale,
// scale-add, gate-mul, relu^2, softcap, exp, rsqrt, log, and the row broadcast.
// Both entry points take the `PointwiseParams` block and one op code drives the
// forward and the backward, so the pair cannot drift apart. Grid-stride
// elementwise; the op switch mirrors the CPU reference
// (backends/cpu/kernels.cc) arm for arm, with no `default` arm, because the
// enum is closed and a forgotten code must be a compile error.
//
// `kRowScale` is the one code whose index expression is not `i`: `a` holds one
// scalar per row and `b`/`out` hold `count` values, so
// `out[i] = a[i / row_width] * b[i]` (spec/pointwise_ops.md section 3.1). Its
// backward is a per-row reduction (`da[r] = sum_c dy[r,c] * b[r,c]`) beside an
// elementwise `db[i] = dy[i] * a[i / row_width]`, so it gets its own kernel:
// one block per row, the row scanned by a block reduction. A `row_width` of 0,
// a width larger than the count, or a count that is not a multiple of it is a
// declared error, rejected on the host with a message that names the count and
// the width -- a wrong width silently reads the wrong row and every number
// downstream still looks plausible.
//
// Accuracy (spec/pointwise_ops.md section 4). The transcendental codes
// (`kSoftcap`, `kExp`, `kRsqrt`, `kLog`) carry a declared absolute bound per
// element rather than a bit pattern: a device `expf`/`logf`/`sqrtf` and a host
// libm are both correct and disagree in the last place. `kGateMul` is *bounded*
// too, at 4e-6 over `abs(a) <= 40, abs(b) <= 8`, even though the seam header
// lumps it in with the bit-exact arithmetic codes: it contains a sigmoid, so a
// device and a host `std::exp` can differ in the last place, and this
// implementation is bounded by the spec's number and never claims bit equality.
//
// See docs/kernels.md.

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::ToComputeDev;

__device__ __forceinline__ float SigmoidDev(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The `kRowScale` guard (spec/pointwise_ops.md section 3.1). A width of 0, a
// width larger than the count, and a count that is not a multiple of it are
// declared errors. The seam has no error return, so the rejection is a message
// naming the count and the width plus no write at all: a loud no-op beats a
// plausible wrong answer, and the caller-side rejection belongs to the
// composition tier. Mirrors the CPU backend so the two devices agree.
bool ValidRowScaleWidth(int count, int row_width) {
  return row_width > 0 && row_width <= count && count % row_width == 0;
}

void RejectRowScaleWidth(const char* entry, int count, int row_width) {
  std::fprintf(stderr,
               "pointwise: %s: kRowScale needs row_width > 0, "
               "row_width <= count and count %% row_width == 0 "
               "(count %d, row_width %d)\n",
               entry, count, row_width);
}

// `out[i] = f(a[i])` for the eight flat codes. `kRowScale` is unreachable here:
// the row broadcast is a per-row index expression and is dispatched on the host
// to its own kernel, but the arm is named so the switch stays exhaustive with
// no `default`.
__global__ void PointwiseForwardKernel(PointwiseParams p,
                                       const ComputeType* __restrict__ a,
                                       const ComputeType* __restrict__ b,
                                       ComputeType* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  const float av = AsFloatDev(a[i]);
  float result = 0.0f;
  switch (p.op) {
    case PointwiseOp::kScale:
      result = p.alpha * av;
      break;
    case PointwiseOp::kScaleAdd:
      result = p.alpha * av + p.beta * AsFloatDev(b[i]);
      break;
    case PointwiseOp::kGateMul:
      result = SigmoidDev(av) * AsFloatDev(b[i]);
      break;
    case PointwiseOp::kReluSquare: {
      const float relu = av > 0.0f ? av : 0.0f;
      result = relu * relu;
      break;
    }
    case PointwiseOp::kSoftcap: {
      const float cap = p.alpha;
      result = cap * tanhf(av / cap);
      break;
    }
    case PointwiseOp::kExp:
      result = expf(av);
      break;
    case PointwiseOp::kRsqrt:
      result = 1.0f / sqrtf(av);
      break;
    case PointwiseOp::kLog:
      result = logf(av);
      break;
    case PointwiseOp::kRowScale:
      // Unreachable: dispatched to PointwiseRowScaleForwardKernel on the host.
      break;
  }
  out[i] = ToComputeDev(result);
}

// The backward of the eight flat codes. It reads `a` and `b` only and never
// `out`: a composition's saved slot for a pointwise leg is its input `a`, so a
// fused override and a composition can share one workspace
// (spec/pointwise_ops.md section 2). `da` and `db` may be null, which means
// "this operand's gradient is not wanted".
__global__ void PointwiseBackwardKernel(PointwiseParams p,
                                        const ComputeType* __restrict__ a,
                                        const ComputeType* __restrict__ b,
                                        const ComputeType* __restrict__ dy,
                                        ComputeType* __restrict__ da,
                                        ComputeType* __restrict__ db) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  const float av = AsFloatDev(a[i]);
  const float dyv = AsFloatDev(dy[i]);
  switch (p.op) {
    case PointwiseOp::kScale:
      if (da != nullptr) da[i] = ToComputeDev(p.alpha * dyv);
      break;
    case PointwiseOp::kScaleAdd:
      if (da != nullptr) da[i] = ToComputeDev(p.alpha * dyv);
      if (db != nullptr) db[i] = ToComputeDev(p.beta * dyv);
      break;
    case PointwiseOp::kGateMul: {
      const float sig = SigmoidDev(av);
      const float bv = AsFloatDev(b[i]);
      if (da != nullptr) da[i] = ToComputeDev(dyv * bv * sig * (1.0f - sig));
      if (db != nullptr) db[i] = ToComputeDev(dyv * sig);
      break;
    }
    case PointwiseOp::kReluSquare:
      if (da != nullptr) {
        da[i] = ToComputeDev(av > 0.0f ? dyv * 2.0f * av : 0.0f);
      }
      break;
    case PointwiseOp::kSoftcap: {
      if (da != nullptr) {
        const float cap = p.alpha;
        const float t = tanhf(av / cap);
        da[i] = ToComputeDev(dyv * (1.0f - t * t));
      }
      break;
    }
    case PointwiseOp::kExp:
      // d/da exp(a) = exp(a), recomputed from `a` rather than read from `out`.
      if (da != nullptr) da[i] = ToComputeDev(dyv * expf(av));
      break;
    case PointwiseOp::kRsqrt: {
      // d/da a^(-1/2) = -0.5 * a^(-3/2), recomputed from `a`.
      if (da != nullptr) {
        const float r = 1.0f / sqrtf(av);
        da[i] = ToComputeDev(-0.5f * dyv * r * r * r);
      }
      break;
    }
    case PointwiseOp::kLog:
      if (da != nullptr) da[i] = ToComputeDev(dyv / av);
      break;
    case PointwiseOp::kRowScale:
      // Unreachable: dispatched to PointwiseRowScaleBackwardKernel on the host.
      break;
  }
}

// `out[i] = a[i / row_width] * b[i]`: one flat loop over the dense operand,
// with the row scalar read through the index expression, so the broadcast needs
// no materialized `rows x width` copy of `a` and no second pass. It reads
// neither `alpha` nor `beta` (spec/pointwise_ops.md section 3). `out` may not
// alias `a` here: `a` is the `count / row_width` row scalars, not the dense
// operand, so an in-place call would overwrite the scalars of later rows. That
// is the CPU reference's behavior too.
__global__ void PointwiseRowScaleForwardKernel(
    int count, int row_width, const ComputeType* __restrict__ a,
    const ComputeType* __restrict__ b, ComputeType* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = ToComputeDev(AsFloatDev(a[i / row_width]) * AsFloatDev(b[i]));
}

// The `kRowScale` backward: one block per row. The block scans the row once,
// writing `db` elementwise and reducing `dy * b` into `da[r]`, which is the
// per-row reduction of spec/pointwise_ops.md section 3.1 (the same closed shape
// the RowReduce primitive carries with a dot combine). A row is at most `count`
// wide and the scan is coalesced, so consecutive threads read consecutive
// elements.
//
// The empty-`da` case is a uniform branch across the block, as in
// `row_reduce.cu`: every thread of the block takes it, so a block reduction is
// never entered half-way.
__global__ void PointwiseRowScaleBackwardKernel(
    int row_width, const ComputeType* __restrict__ a,
    const ComputeType* __restrict__ b, const ComputeType* __restrict__ dy,
    ComputeType* __restrict__ da, ComputeType* __restrict__ db) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const std::int64_t base = static_cast<std::int64_t>(row) * row_width;
  const float scalar = AsFloatDev(a[row]);
  float sum = 0.0f;
  for (int c = tid; c < row_width; c += blockDim.x) {
    const std::int64_t i = base + c;
    const float dyv = AsFloatDev(dy[i]);
    sum += dyv * AsFloatDev(b[i]);
    if (db != nullptr) db[i] = ToComputeDev(dyv * scalar);
  }
  const float total = BlockReduceSum(sum);
  if (da != nullptr && tid == 0) da[row] = ToComputeDev(total);
}

inline int GridSize(int n, int block) {
  return n > 0 ? (n + block - 1) / block : 1;
}

}  // namespace

void PointwiseForward(const PointwiseParams& params, const ComputeType* a,
                      const ComputeType* b, ComputeType* out) {
  const int n = params.count;
  if (n <= 0 || a == nullptr || out == nullptr) return;
  constexpr int kThreads = 256;

  if (params.op == PointwiseOp::kRowScale) {
    const int row_width = params.row_width;
    if (!ValidRowScaleWidth(n, row_width)) {
      RejectRowScaleWidth("PointwiseForward", n, row_width);
      return;
    }
    if (b == nullptr) return;
    cuda_backend::Launch(PointwiseRowScaleForwardKernel,
                         dim3(GridSize(n, kThreads)), dim3(kThreads), 0, n,
                         row_width, a, b, out);
    return;
  }

  cuda_backend::Launch(PointwiseForwardKernel, dim3(GridSize(n, kThreads)),
                       dim3(kThreads), 0, params, a, b, out);
}

void PointwiseBackward(const PointwiseParams& params, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy,
                       ComputeType* da, ComputeType* db) {
  const int n = params.count;
  if (n <= 0 || a == nullptr || dy == nullptr) return;
  constexpr int kThreads = 256;

  if (params.op == PointwiseOp::kRowScale) {
    const int row_width = params.row_width;
    if (!ValidRowScaleWidth(n, row_width)) {
      RejectRowScaleWidth("PointwiseBackward", n, row_width);
      return;
    }
    if (b == nullptr) return;
    // One block per row: the row sum needs every column, and the rows are the
    // grid, so a row never spans two blocks.
    cuda_backend::Launch(PointwiseRowScaleBackwardKernel, dim3(n / row_width),
                         dim3(kThreads), 0, row_width, a, b, dy, da, db);
    return;
  }

  cuda_backend::Launch(PointwiseBackwardKernel, dim3(GridSize(n, kThreads)),
                       dim3(kThreads), 0, params, a, b, dy, da, db);
}

}  // namespace kernels
}  // namespace nanochat
