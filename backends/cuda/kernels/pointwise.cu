// CUDA Pointwise family: the twelve op codes of spec/pointwise_ops.md -- scale,
// scale-add, gate-mul, relu^2, softcap, exp, rsqrt, log, the row broadcast, the
// scalar affine, the sigmoid, and the half-split pair rotation.
// Both entry points take the `PointwiseParams` block and one op code drives the
// forward and the backward, so the pair cannot drift apart. Grid-stride
// elementwise; the op switch mirrors the CPU reference
// (backends/cpu/kernels.cc) arm for arm, with no `default` arm, because the
// enum is closed and a forgotten code must be a compile error.
//
// `kRowScale` and `kRotatePair` are the two codes whose index expressions are
// not `i`, so each gets its own kernel and a host-side dispatch arm.
// `kRowScale` is the broadcast: `a` holds one scalar per row and `b`/`out` hold
// `count` values, so `out[i] = a[i / row_width] * b[i]`
// (spec/pointwise_ops.md section 3.1). Its backward is a per-row reduction
// (`da[r] = sum_c dy[r,c] * b[r,c]`) beside an elementwise
// `db[i] = dy[i] * a[i / row_width]`, so it runs one block per row, the row
// scanned by a block reduction. `kRotatePair` is the half-split pair rotation:
// one thread owns both elements of a pair, so `out` may alias `a`, and the
// rotation's multiplier comes from a table selected by the row's sequence
// position (spec/pointwise_ops.md section 3.2). A `row_width` of 0, a width
// larger than the count, or a count that is not a multiple of it is a declared
// error for both codes, rejected on the host with a message that names the
// count and the width -- a wrong width silently reads the wrong row and every
// number downstream still looks plausible.
//
// Accuracy (spec/pointwise_ops.md section 4). The transcendental codes
// (`kSoftcap`, `kExp`, `kRsqrt`, `kLog`, `kSigmoid`) carry a declared absolute
// bound per element rather than a bit pattern: a device `expf`/`logf`/`sqrtf`
// and a host libm are both correct and disagree in the last place. `kGateMul`
// is *bounded* too, at 4e-6 over `abs(a) <= 40, abs(b) <= 8`, even though the
// seam header lumps it in with the bit-exact arithmetic codes: it contains a
// sigmoid, so a device and a host `std::exp` can differ in the last place, and
// this implementation is bounded by the spec's number and never claims bit
// equality. `kRotatePair` pins its rounding with the intrinsics for the reason
// `backends/cuda/kernels/qk_prep.cu` does: nvcc contracts `x1 * c + x2 * s`
// into an FMA, which leaves one product unrounded and moves the frozen
// qk_prep trace by an ulp per pair.
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
using cuda_kernels::ComputeVector;
using cuda_kernels::FloatVector;
using cuda_kernels::kFloatPerVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorGrid;
using cuda_kernels::VectorReady;

__device__ __forceinline__ float SigmoidDev(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// True when the op code reads the second operand `b`. The backward of the same
// code is the one that produces a gradient for `b`, so one predicate serves the
// forward's read and the backward's write.
__host__ __device__ __forceinline__ bool ReadsSecondOperand(PointwiseOp op) {
  return op == PointwiseOp::kScaleAdd || op == PointwiseOp::kGateMul;
}

// The forward arithmetic of one element, arm for arm with the CPU reference
// (backends/cpu/kernels.cc). Both the scalar kernel and the vector kernel call
// this, so the two paths cannot drift; the two index-specific codes are
// dispatched on the host and are unreachable here.
__device__ __forceinline__ float ForwardValueDev(PointwiseOp op, float alpha,
                                                 float beta, float av,
                                                 float bv) {
  switch (op) {
    case PointwiseOp::kScale:
      return alpha * av;
    case PointwiseOp::kScaleAdd:
      return alpha * av + beta * bv;
    case PointwiseOp::kGateMul:
      return SigmoidDev(av) * bv;
    case PointwiseOp::kReluSquare: {
      const float relu = av > 0.0f ? av : 0.0f;
      return relu * relu;
    }
    case PointwiseOp::kSoftcap: {
      const float cap = alpha;
      return cap * tanhf(av / cap);
    }
    case PointwiseOp::kExp:
      return expf(av);
    case PointwiseOp::kRsqrt:
      return 1.0f / sqrtf(av);
    case PointwiseOp::kLog:
      return logf(av);
    case PointwiseOp::kRowScale:
      break;
    case PointwiseOp::kAffine:
      return alpha * (av - beta);
    case PointwiseOp::kSigmoid:
      return SigmoidDev(av);
    case PointwiseOp::kRotatePair:
      break;
  }
  return 0.0f;
}

// The backward arithmetic of one element. `da` is always produced; `db` is
// produced only for the two codes that read `b`, and the caller decides whether
// to store it. The `kScale` arm returns `alpha * dy` for `da` and `beta * dy`
// for `db` only through `kScaleAdd`, which is the CPU reference's split.
struct BackwardPair {
  float da;
  float db;
};

__device__ __forceinline__ BackwardPair BackwardValuesDev(PointwiseOp op,
                                                          float alpha,
                                                          float beta, float av,
                                                          float bv, float dyv) {
  BackwardPair pair{0.0f, 0.0f};
  switch (op) {
    case PointwiseOp::kScale:
      pair.da = alpha * dyv;
      break;
    case PointwiseOp::kScaleAdd:
      pair.da = alpha * dyv;
      pair.db = beta * dyv;
      break;
    case PointwiseOp::kGateMul: {
      const float sig = SigmoidDev(av);
      pair.da = dyv * bv * sig * (1.0f - sig);
      pair.db = dyv * sig;
      break;
    }
    case PointwiseOp::kReluSquare:
      pair.da = av > 0.0f ? dyv * 2.0f * av : 0.0f;
      break;
    case PointwiseOp::kSoftcap: {
      const float cap = alpha;
      const float t = tanhf(av / cap);
      pair.da = dyv * (1.0f - t * t);
      break;
    }
    case PointwiseOp::kExp:
      // d/da exp(a) = exp(a), recomputed from `a` rather than read from `out`.
      pair.da = dyv * expf(av);
      break;
    case PointwiseOp::kRsqrt: {
      // d/da a^(-1/2) = -0.5 * a^(-3/2), recomputed from `a`.
      const float r = 1.0f / sqrtf(av);
      pair.da = -0.5f * dyv * r * r * r;
      break;
    }
    case PointwiseOp::kLog:
      pair.da = dyv / av;
      break;
    case PointwiseOp::kRowScale:
      break;
    case PointwiseOp::kAffine:
      pair.da = alpha * dyv;
      break;
    case PointwiseOp::kSigmoid: {
      const float sig = SigmoidDev(av);
      pair.da = dyv * sig * (1.0f - sig);
      break;
    }
    case PointwiseOp::kRotatePair:
      break;
  }
  return pair;
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

// The `kRotatePair` guard (spec/pointwise_ops.md section 3.2). The head width
// must be even and must tile the count, and the head rows must tile the table
// positions, because a wrong `heads` or `seq` reads the wrong table position
// for every row and the activation still looks plausible. Mirrors the CPU
// backend so the two devices agree.
bool ValidRotate(int count, const PointwiseParams& p) {
  if (p.row_width < 2 || p.row_width % 2 != 0 || p.row_width > count) {
    return false;
  }
  if (count % p.row_width != 0) return false;
  if (p.heads <= 0 || p.seq <= 0) return false;
  return (count / p.row_width) % p.heads == 0;
}

void RejectRotate(const char* entry, const PointwiseParams& p) {
  std::fprintf(stderr,
               "pointwise: %s: kRotatePair needs an even row_width in "
               "[2, count], count %% row_width == 0, (count / row_width) %% "
               "heads == 0, and heads and seq both positive "
               "(count %d, row_width %d, heads %d, seq %d)\n",
               entry, p.count, p.row_width, p.heads, p.seq);
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
  // One load per thread from a uniform address, which the hardware broadcasts.
  const float alpha = p.alpha.host();
  const float beta = p.beta.host();
  const float av = AsFloatDev(a[i]);
  const float bv = ReadsSecondOperand(p.op) ? AsFloatDev(b[i]) : 0.0f;
  out[i] = ToComputeDev(ForwardValueDev(p.op, alpha, beta, av, bv));
}

// The same arithmetic, one aligned 16-byte vector per thread. `out` may alias
// `a`, so each thread reads its whole vector before it writes anything.
__global__ void PointwiseForwardVectorKernel(
    PointwiseParams p, long long vectors, const ComputeVector* __restrict__ a,
    const ComputeVector* __restrict__ b, ComputeVector* __restrict__ out) {
  // A grid stride, because the shared grid helper caps the block count for a
  // large tensor and a single-shot launch would leave the tail untouched.
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const float alpha = p.alpha.host();
    const float beta = p.beta.host();
    const ComputeVector av = a[v];
    ComputeVector bv{};
    if (ReadsSecondOperand(p.op)) bv = b[v];
    ComputeVector result;
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      result.e[k] = ToComputeDev(ForwardValueDev(
          p.op, alpha, beta, AsFloatDev(av.e[k]), AsFloatDev(bv.e[k])));
    }
    out[v] = result;
  }
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
  const float alpha = p.alpha.host();
  const float beta = p.beta.host();
  const float av = AsFloatDev(a[i]);
  const float dyv = AsFloatDev(dy[i]);
  const float bv = ReadsSecondOperand(p.op) ? AsFloatDev(b[i]) : 0.0f;
  const BackwardPair pair = BackwardValuesDev(p.op, alpha, beta, av, bv, dyv);
  if (da != nullptr) da[i] = ToComputeDev(pair.da);
  if (db != nullptr && ReadsSecondOperand(p.op)) {
    db[i] = ToComputeDev(pair.db);
  }
}

// The same arithmetic, one aligned 16-byte vector per thread.
__global__ void PointwiseBackwardVectorKernel(
    PointwiseParams p, long long vectors, const ComputeVector* __restrict__ a,
    const ComputeVector* __restrict__ b, const ComputeVector* __restrict__ dy,
    ComputeVector* __restrict__ da, ComputeVector* __restrict__ db) {
  // A grid stride, so a capped grid still covers a large tensor.
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const float alpha = p.alpha.host();
    const float beta = p.beta.host();
    const ComputeVector av = a[v];
    const ComputeVector dyv = dy[v];
    ComputeVector bv{};
    if (ReadsSecondOperand(p.op)) bv = b[v];
    ComputeVector dv{};
    ComputeVector dbv{};
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      const BackwardPair pair =
          BackwardValuesDev(p.op, alpha, beta, AsFloatDev(av.e[k]),
                            AsFloatDev(bv.e[k]), AsFloatDev(dyv.e[k]));
      dv.e[k] = ToComputeDev(pair.da);
      dbv.e[k] = ToComputeDev(pair.db);
    }
    if (da != nullptr) da[v] = dv;
    if (db != nullptr && ReadsSecondOperand(p.op)) db[v] = dbv;
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

// The same broadcast, one aligned 16-byte vector per thread. A vector never
// crosses a row boundary because the host requires `row_width` to be a whole
// number of vectors, so every element of the vector shares one row scalar.
__global__ void PointwiseRowScaleForwardVectorKernel(
    long long vectors, int row_width, const ComputeType* __restrict__ a,
    const ComputeVector* __restrict__ b, ComputeVector* __restrict__ out) {
  // A grid stride, so a capped grid still covers a large tensor.
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const long long first = v * kVectorElems;
    const float scalar = AsFloatDev(a[first / row_width]);
    const ComputeVector bv = b[v];
    ComputeVector result;
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      result.e[k] = ToComputeDev(scalar * AsFloatDev(bv.e[k]));
    }
    out[v] = result;
  }
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

// The table position of head row `r` (spec/pointwise_ops.md section 3.2): head
// rows are grouped `heads` at a time and the group index repeats every `seq`
// rows, so a batch dimension stacked on top of `seq` reads the same table
// again.
__device__ __forceinline__ int TablePosition(int row, int heads, int seq) {
  return (row / heads) % seq;
}

// The forward's pair rotation, with every float operation rounded to nearest.
// The composition's definition is `alpha * (x1 * c + x2 * s)` with each product
// rounded before the add, and nvcc contracts that expression into an FMA by
// default. The intrinsics hold the source form to one rounding per operation,
// for the reason `backends/cuda/kernels/qk_prep.cu` states: the fused qk_prep
// family is gated against this arithmetic, and an unrounded product moves it by
// an ulp per pair.
__device__ __forceinline__ void RotatePairForward(float& x1, float& x2, float c,
                                                  float s, float alpha) {
  const float t1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
  const float t2 = __fadd_rn(__fmul_rn(x2, c), -__fmul_rn(x1, s));
  x1 = __fmul_rn(alpha, t1);
  x2 = __fmul_rn(alpha, t2);
}

// The transposed rotation, with the same per-operation rounding.
__device__ __forceinline__ void RotatePairBackward(float g1, float g2, float c,
                                                   float s, float alpha,
                                                   float* d1, float* d2) {
  *d1 = __fmul_rn(alpha, __fsub_rn(__fmul_rn(g1, c), __fmul_rn(g2, s)));
  *d2 = __fmul_rn(alpha, __fadd_rn(__fmul_rn(g1, s), __fmul_rn(g2, c)));
}

// One thread per pair. `j` selects head row `j / half` and column `j % half`,
// and the thread touches that pair's two elements and nothing else, which is
// what keeps `out` allowed to alias `a`: an element-flat grid would let one
// thread overwrite the pair half that another thread is still reading.
__global__ void PointwiseRotatePairForwardKernel(
    PointwiseParams p, const ComputeType* __restrict__ a,
    ComputeType* __restrict__ out) {
  const int width = p.row_width;
  const int half = width / 2;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= p.count / 2) return;
  const int row = j / half;
  const int d = j % half;
  const std::int64_t base = static_cast<std::int64_t>(row) * width;
  const std::int64_t table_base =
      static_cast<std::int64_t>(TablePosition(row, p.heads, p.seq)) * half;
  const float c = p.cos_table[table_base + d];
  const float s = p.sin_table[table_base + d];
  float x1 = AsFloatDev(a[base + d]);
  float x2 = AsFloatDev(a[base + half + d]);
  RotatePairForward(x1, x2, c, s, p.alpha.host());
  out[base + d] = ToComputeDev(x1);
  out[base + half + d] = ToComputeDev(x2);
}

// The transposed rotation. The rotation is linear, so this reads `dy` and never
// `a` (spec/pointwise_ops.md section 3.2). One thread per pair, so an in-place
// call is safe here for the same reason the forward's is.
__global__ void PointwiseRotatePairBackwardKernel(
    PointwiseParams p, const ComputeType* __restrict__ dy,
    ComputeType* __restrict__ da) {
  const int width = p.row_width;
  const int half = width / 2;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= p.count / 2) return;
  const int row = j / half;
  const int d = j % half;
  const std::int64_t base = static_cast<std::int64_t>(row) * width;
  const std::int64_t table_base =
      static_cast<std::int64_t>(TablePosition(row, p.heads, p.seq)) * half;
  const float c = p.cos_table[table_base + d];
  const float s = p.sin_table[table_base + d];
  const float g1 = AsFloatDev(dy[base + d]);
  const float g2 = AsFloatDev(dy[base + half + d]);
  float d1 = 0.0f;
  float d2 = 0.0f;
  RotatePairBackward(g1, g2, c, s, p.alpha.host(), &d1, &d2);
  da[base + d] = ToComputeDev(d1);
  da[base + half + d] = ToComputeDev(d2);
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
    if (row_width % kVectorElems == 0 && VectorReady(n, {a, b, out})) {
      const long long vectors = n / kVectorElems;
      cuda_backend::Launch(PointwiseRowScaleForwardVectorKernel,
                           dim3(VectorGrid(vectors, kThreads)), dim3(kThreads),
                           0, vectors, row_width, a,
                           reinterpret_cast<const ComputeVector*>(b),
                           reinterpret_cast<ComputeVector*>(out));
      return;
    }
    cuda_backend::Launch(PointwiseRowScaleForwardKernel,
                         dim3(GridSize(n, kThreads)), dim3(kThreads), 0, n,
                         row_width, a, b, out);
    return;
  }

  if (params.op == PointwiseOp::kRotatePair) {
    if (params.cos_table == nullptr || params.sin_table == nullptr) return;
    if (!ValidRotate(n, params)) {
      RejectRotate("PointwiseForward", params);
      return;
    }
    cuda_backend::Launch(PointwiseRotatePairForwardKernel,
                         dim3(GridSize(n / 2, kThreads)), dim3(kThreads), 0,
                         params, a, out);
    return;
  }

  if (VectorReady(n, {a, b, out})) {
    const long long vectors = n / kVectorElems;
    cuda_backend::Launch(PointwiseForwardVectorKernel,
                         dim3(VectorGrid(vectors, kThreads)), dim3(kThreads), 0,
                         params, vectors,
                         reinterpret_cast<const ComputeVector*>(a),
                         reinterpret_cast<const ComputeVector*>(b),
                         reinterpret_cast<ComputeVector*>(out));
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

  if (params.op == PointwiseOp::kRotatePair) {
    if (params.cos_table == nullptr || params.sin_table == nullptr) return;
    if (!ValidRotate(n, params)) {
      RejectRotate("PointwiseBackward", params);
      return;
    }
    if (da == nullptr) return;
    cuda_backend::Launch(PointwiseRotatePairBackwardKernel,
                         dim3(GridSize(n / 2, kThreads)), dim3(kThreads), 0,
                         params, dy, da);
    return;
  }

  if (VectorReady(n, {a, b, dy, da, db})) {
    const long long vectors = n / kVectorElems;
    cuda_backend::Launch(PointwiseBackwardVectorKernel,
                         dim3(VectorGrid(vectors, kThreads)), dim3(kThreads), 0,
                         params, vectors,
                         reinterpret_cast<const ComputeVector*>(a),
                         reinterpret_cast<const ComputeVector*>(b),
                         reinterpret_cast<const ComputeVector*>(dy),
                         reinterpret_cast<ComputeVector*>(da),
                         reinterpret_cast<ComputeVector*>(db));
    return;
  }
  cuda_backend::Launch(PointwiseBackwardKernel, dim3(GridSize(n, kThreads)),
                       dim3(kThreads), 0, params, a, b, dy, da, db);
}

}  // namespace kernels
}  // namespace nanochat
