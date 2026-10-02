// CUDA Pointwise family: relu^2, scale-add, gate-mul, scale, and softcap, all
// driven by the same op code for forward and backward so the pair cannot drift
// apart. Grid-stride elementwise; the op switch mirrors the CPU reference.
//
// See docs/kernels.md.

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::ToComputeDev;

__device__ __forceinline__ float SigmoidDev(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void PointwiseForwardKernel(const PointwiseOp op, int n,
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
      result = SigmoidDev(av) * AsFloatDev(b[i]);
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

__global__ void PointwiseBackwardKernel(const PointwiseOp op, int n,
                                        const ComputeType* __restrict__ a,
                                        const ComputeType* __restrict__ b,
                                        const ComputeType* __restrict__ dy,
                                        float alpha, float beta,
                                        ComputeType* __restrict__ da,
                                        ComputeType* __restrict__ db) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float av = AsFloatDev(a[i]);
  const float dyv = AsFloatDev(dy[i]);
  switch (op) {
    case PointwiseOp::kScale:
      if (da != nullptr) da[i] = ToComputeDev(alpha * dyv);
      break;
    case PointwiseOp::kScaleAdd:
      if (da != nullptr) da[i] = ToComputeDev(alpha * dyv);
      if (db != nullptr) db[i] = ToComputeDev(beta * dyv);
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
        const float cap = alpha;
        const float t = tanhf(av / cap);
        da[i] = ToComputeDev(dyv * (1.0f - t * t));
      }
      break;
    }
  }
}

inline int GridSize(int n, int block) {
  return n > 0 ? (n + block - 1) / block : 1;
}

}  // namespace

void PointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                      const ComputeType* b, float alpha, float beta,
                      ComputeType* out) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  cuda_backend::Launch(PointwiseForwardKernel, dim3(GridSize(n, kThreads)),
                       dim3(kThreads), 0, op, n, a, b, alpha, beta, out);
}

void PointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy, float alpha,
                       float beta, ComputeType* da, ComputeType* db) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  cuda_backend::Launch(PointwiseBackwardKernel, dim3(GridSize(n, kThreads)),
                       dim3(kThreads), 0, op, n, a, b, dy, alpha, beta, da, db);
}

}  // namespace kernels
}  // namespace nanochat
