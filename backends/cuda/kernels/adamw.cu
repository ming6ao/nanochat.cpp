// CUDA AdamW family: the stateful pointwise update over a flat parameter
// block (embedding, head, and the scalar parameters). The bias-correction
// factors, the decoupled weight decay, and the two moment buffers follow
// backends/cpu/kernels.cc exactly: the first and second moments and the
// optimizer state stay fp32, and the parameter is written back through
// ComputeType.
//
// One grid-stride kernel keeps the whole update in a single pass, so the
// moments and the parameter are read and written once. See docs/kernels.md and
// docs/model.md.

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::ComputeVector;
using cuda_kernels::FloatVector;
using cuda_kernels::kFloatPerVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorGrid;
using cuda_kernels::VectorReady;

__global__ void AdamWUpdateVectorKernel(
    long long vectors, float lr, float beta1, float beta2, float eps,
    float weight_decay, float step_size, float bias2, ScalarArg grad_scale,
    FloatVector* __restrict__ master, ComputeVector* __restrict__ value,
    const FloatVector* __restrict__ g, FloatVector* __restrict__ moment1,
    FloatVector* __restrict__ moment2) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const float scale = grad_scale.host();
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const FloatVector* gbase = g + v * kFloatPerVector;
    FloatVector* mbase = moment1 + v * kFloatPerVector;
    FloatVector* vbase = moment2 + v * kFloatPerVector;
    FloatVector* pbase = master + v * kFloatPerVector;
    ComputeVector result;
#pragma unroll
    for (int j = 0; j < kFloatPerVector; ++j) {
      const FloatVector gv = gbase[j];
      const FloatVector mv = mbase[j];
      const FloatVector vv = vbase[j];
      const FloatVector pv = pbase[j];
      FloatVector next_m;
      FloatVector next_v;
      FloatVector next_p;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        // The clip scale rides on the gradient read, so the decoupled weight
        // decay above stays unscaled (one when the clip is inactive).
        const float grad = gv.e[s] * scale;
        // Decoupled weight decay, applied to the fp32 master before the update.
        float pi = pv.e[s] * (1.0f - lr * weight_decay);
        const float mi = mv.e[s] + (1.0f - beta1) * (grad - mv.e[s]);
        const float vi = vv.e[s] + (1.0f - beta2) * (grad * grad - vv.e[s]);
        next_m.e[s] = mi;
        next_v.e[s] = vi;
        const float denom = sqrtf(vi / bias2) + eps;
        pi -= step_size * (mi / denom);
        next_p.e[s] = pi;
        result.e[4 * j + s] = ToComputeDev(pi);
      }
      mbase[j] = next_m;
      vbase[j] = next_v;
      pbase[j] = next_p;
    }
    value[v] = result;
  }
}

__global__ void AdamWUpdateKernel(
    long long n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float step_size, float bias2, ScalarArg grad_scale,
    float* __restrict__ master, ComputeType* __restrict__ value,
    const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const float scale = grad_scale.host();
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    // The clip scale rides on the gradient read, so the decoupled weight decay
    // above stays unscaled (one when the clip is inactive).
    const float grad = g[i] * scale;
    // Decoupled weight decay, applied to the fp32 master before the update.
    float pi = master[i] * (1.0f - lr * weight_decay);
    const float mi = m[i] + (1.0f - beta1) * (grad - m[i]);
    const float vi = v[i] + (1.0f - beta2) * (grad * grad - v[i]);
    m[i] = mi;
    v[i] = vi;
    const float denom = sqrtf(vi / bias2) + eps;
    pi -= step_size * (mi / denom);
    master[i] = pi;
    value[i] = ToComputeDev(pi);
  }
}

}  // namespace

void AdamWUpdate(int n, const AdamWParams& params, float* master,
                 ComputeType* value, const float* g, float* m, float* v) {
  if (n <= 0) return;
  // Bias correction is computed once on the host in fp32, matching the CPU
  // reference (1 - beta^step). `step` is 1-based.
  const float bias1 =
      1.0f - powf(params.beta1, static_cast<float>(params.step));
  const float bias2 =
      1.0f - powf(params.beta2, static_cast<float>(params.step));
  const float step_size = params.lr / bias1;

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 512;
  const long long blocks =
      (static_cast<long long>(n) + kThreads - 1) / kThreads;
  if (VectorReady(n, {master, value, g, m, v})) {
    const long long vectors = static_cast<long long>(n) / kVectorElems;
    const long long vector_blocks = (vectors + kThreads - 1) / kThreads;
    constexpr long long kMaxVectorBlocks = 4096;
    const int grid = static_cast<int>(
        vector_blocks < kMaxVectorBlocks ? vector_blocks : kMaxVectorBlocks);
    cuda_backend::Launch(
        AdamWUpdateVectorKernel, dim3(grid), dim3(kThreads), 0, vectors,
        params.lr, params.beta1, params.beta2, params.eps, params.weight_decay,
        step_size, bias2, params.grad_scale,
        reinterpret_cast<FloatVector*>(master),
        reinterpret_cast<ComputeVector*>(value),
        reinterpret_cast<const FloatVector*>(g),
        reinterpret_cast<FloatVector*>(m), reinterpret_cast<FloatVector*>(v));
    return;
  }
  const int grid = blocks < kMaxBlocks ? static_cast<int>(blocks) : kMaxBlocks;
  // The clip scale is either the host value or a device scalar the reduction
  // wrote; the kernel dereferences the device one per pass, so a step that
  // keeps the norm on the device reads the host never.
  cuda_backend::Launch(AdamWUpdateKernel, dim3(grid), dim3(kThreads), 0,
                       static_cast<long long>(n), params.lr, params.beta1,
                       params.beta2, params.eps, params.weight_decay, step_size,
                       bias2, params.grad_scale, master, value, g, m, v);
}

}  // namespace kernels
}  // namespace nanochat
