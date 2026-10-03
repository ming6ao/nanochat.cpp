// CUDA ResFormer value gate: `v += 3*sigmoid(h[:12] @ gate_w^T) * ve`, forward
// and backward. This used to run as a host loop with a device round trip; it
// is now a per-row device kernel so the whole block forward and backward stay
// on the GPU. See docs/kernels.md.
//
// One block per row. The forward computes the per-head gate in a few threads
// (the reduction is only over `kVeGateChannels`), broadcasts it through shared
// memory, and updates every value dimension. The backward reduces
// `dot(dv, ve)` per head with the block reduction helper, then accumulates the
// gate-weight gradient (atomics, because every row contributes) and the input
// gradient.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::AtomicAddDev;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::ToComputeDev;

constexpr int kGateChannels = 12;

__global__ void ValueGateForwardKernel(int rows, int hidden, int kv_heads,
                                       int head_dim,
                                       const ComputeType* __restrict__ h,
                                       const ComputeType* __restrict__ ve,
                                       const ComputeType* __restrict__ gate_w,
                                       ComputeType* __restrict__ v,
                                       ComputeType* __restrict__ gate_out) {
  extern __shared__ float gates[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid < kv_heads) {
    const ComputeType* hrow = h + static_cast<std::int64_t>(row) * hidden;
    const ComputeType* wrow = gate_w + tid * kGateChannels;
    float pre = 0.0f;
    for (int j = 0; j < kGateChannels; ++j) {
      pre += AsFloatDev(hrow[j]) * AsFloatDev(wrow[j]);
    }
    const float gate = 3.0f / (1.0f + expf(-pre));
    gates[tid] = gate;
    gate_out[static_cast<std::int64_t>(row) * kv_heads + tid] =
        ToComputeDev(gate);
  }
  __syncthreads();

  const std::int64_t base =
      static_cast<std::int64_t>(row) * kv_heads * head_dim;
  for (int kh = 0; kh < kv_heads; ++kh) {
    const float g = gates[kh];
    const std::int64_t off = base + static_cast<std::int64_t>(kh) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      v[off + d] =
          ToComputeDev(AsFloatDev(v[off + d]) + g * AsFloatDev(ve[off + d]));
    }
  }
}

__global__ void ValueGateBackwardKernel(
    int rows, int hidden, int kv_heads, int head_dim,
    const ComputeType* __restrict__ h, const ComputeType* __restrict__ ve,
    const ComputeType* __restrict__ gate_w,
    const ComputeType* __restrict__ gate, const ComputeType* __restrict__ dv,
    ComputeType* __restrict__ gate_w_grad, ComputeType* __restrict__ dh,
    ComputeType* __restrict__ dve) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const ComputeType* hrow = h + static_cast<std::int64_t>(row) * hidden;
  ComputeType* dhrow = dh + static_cast<std::int64_t>(row) * hidden;
  const std::int64_t base =
      static_cast<std::int64_t>(row) * kv_heads * head_dim;

  for (int kh = 0; kh < kv_heads; ++kh) {
    const std::int64_t off = base + static_cast<std::int64_t>(kh) * head_dim;
    const float g =
        AsFloatDev(gate[static_cast<std::int64_t>(row) * kv_heads + kh]);
    float local_dgate = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      const float dvd = AsFloatDev(dv[off + d]);
      local_dgate += dvd * AsFloatDev(ve[off + d]);
      dve[off + d] = ToComputeDev(g * dvd);
    }
    const float dgate = BlockReduceSum(local_dgate);
    // gate = 3*sigmoid(pre) => d(gate)/d(pre) = gate*(1 - gate/3).
    const float dpre = dgate * g * (1.0f - g / 3.0f);
    if (tid < kGateChannels) {
      AtomicAddDev(&gate_w_grad[kh * kGateChannels + tid],
                   dpre * AsFloatDev(hrow[tid]));
      dhrow[tid] =
          ToComputeDev(AsFloatDev(dhrow[tid]) +
                       dpre * AsFloatDev(gate_w[kh * kGateChannels + tid]));
    }
  }
}

}  // namespace

void ValueGateForward(int rows, int hidden, int num_kv_heads, int head_dim,
                      const ComputeType* h, const ComputeType* ve,
                      const ComputeType* gate_w, ComputeType* v,
                      ComputeType* gate_out) {
  if (rows <= 0 || hidden < kGateChannels || num_kv_heads <= 0 ||
      head_dim <= 0) {
    return;
  }
  const int block = cuda_kernels::BlockSizeForDim(head_dim);
  const std::size_t shared =
      sizeof(float) * static_cast<std::size_t>(num_kv_heads);
  cuda_backend::Launch(ValueGateForwardKernel, dim3(rows), dim3(block), shared,
                       rows, hidden, num_kv_heads, head_dim, h, ve, gate_w, v,
                       gate_out);
}

void ValueGateBackward(int rows, int hidden, int num_kv_heads, int head_dim,
                       const ComputeType* h, const ComputeType* ve,
                       const ComputeType* gate_w, const ComputeType* gate,
                       const ComputeType* dv, ComputeType* gate_w_grad,
                       ComputeType* dh, ComputeType* dve) {
  if (rows <= 0 || hidden < kGateChannels || num_kv_heads <= 0 ||
      head_dim <= 0) {
    return;
  }
  const int block = cuda_kernels::BlockSizeForDim(head_dim);
  cuda_backend::Launch(ValueGateBackwardKernel, dim3(rows), dim3(block), 0,
                       rows, hidden, num_kv_heads, head_dim, h, ve, gate_w,
                       gate, dv, gate_w_grad, dh, dve);
}

}  // namespace kernels
}  // namespace nanochat
