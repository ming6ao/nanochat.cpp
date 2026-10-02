// CUDA AdamW family: the stateful pointwise update over a flat parameter
// block (embedding, head, and the scalar parameters). The bias-correction
// factors, the decoupled weight decay, and the two moment buffers follow
// backends/cpu/kernels.cc exactly: the first and second moments and the
// optimizer state stay fp32, and the parameter is written back through
// ComputeType.
//
// One grid-stride kernel keeps the whole update in a single pass, so the
// moments and the parameter are read and written once. See docs/kernels.md and
// docs/optimizer.md.

#include <cmath>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::ToComputeDev;

__global__ void AdamWUpdateKernel(long long n, float lr, float beta1,
                                  float beta2, float eps, float weight_decay,
                                  float step_size, float bias2,
                                  ComputeType* __restrict__ p,
                                  const ComputeType* __restrict__ g,
                                  float* __restrict__ m,
                                  float* __restrict__ v) {
  const long long stride =
      static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       i < n; i += stride) {
    const float grad = AsFloatDev(g[i]);
    // Decoupled weight decay, applied to the parameter before the update.
    float pi = AsFloatDev(p[i]) * (1.0f - lr * weight_decay);
    const float mi = m[i] + (1.0f - beta1) * (grad - m[i]);
    const float vi = v[i] + (1.0f - beta2) * (grad * grad - v[i]);
    m[i] = mi;
    v[i] = vi;
    const float denom = sqrtf(vi / bias2) + eps;
    pi -= step_size * (mi / denom);
    p[i] = ToComputeDev(pi);
  }
}

}  // namespace

void AdamWUpdate(int n, const AdamWParams& params, ComputeType* p,
                 const ComputeType* g, float* m, float* v) {
  if (n <= 0) return;
  // Bias correction is computed once on the host in fp32, matching the CPU
  // reference (1 - beta^step). `step` is 1-based.
  const float bias1 = 1.0f - powf(params.beta1, static_cast<float>(params.step));
  const float bias2 = 1.0f - powf(params.beta2, static_cast<float>(params.step));
  const float step_size = params.lr / bias1;

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  const long long blocks =
      (static_cast<long long>(n) + kThreads - 1) / kThreads;
  const int grid =
      blocks < kMaxBlocks ? static_cast<int>(blocks) : kMaxBlocks;
  cuda_backend::Launch(AdamWUpdateKernel, dim3(grid), dim3(kThreads), 0,
                       static_cast<long long>(n), params.lr, params.beta1,
                       params.beta2, params.eps, params.weight_decay,
                       step_size, bias2, p, g, m, v);
}

}  // namespace kernels
}  // namespace nanochat
