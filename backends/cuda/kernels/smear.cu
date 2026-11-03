// CUDA smear: the learned bigram residual on the token embedding.
//
//   gate[row] = lambda * sigmoid(dot(emb_norm[row, :channels], gate_w))
//   x0[row]   = emb_norm[row] + gate[row] * emb_norm[row - 1]
//
// Position 0 of a sequence has no previous token, so its gate is zero and its
// row is a copy. Only the leading `channels` feed the gate; the whole row
// moves.
//
// Both directions are ONE launch. Every row owns its output row and one halo
// row of the input, so the only state that crosses rows is the pair of
// parameter gradients: `gate_w_grad` (channels values) and `lambda_grad` (one).
// Those are reduced the way a training megakernel will reduce them: each block
// writes its own partial, and the last block to arrive combines the partials in
// ascending block order. No atomic touches the data, and the order is fixed by
// the block index, so two runs of one job produce identical bits.
//
// That combine is a phase boundary, not a kernel, which is why the backward is
// one launch instead of a row pass plus a reduce pass.
//
// The family used to be a host loop with a device round trip of the whole
// activation. See docs/kernels.md.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::ToComputeDev;

constexpr int kSmearThreads = 256;

// One partial slot per gate weight, one for the lambda gradient, rounded up to
// a warp so the combine reads a block's partials in whole coalesced lines.
int PartialStride(int channels) { return ((channels + 1 + 31) / 32) * 32; }

__global__ void SmearForwardKernel(SmearParams p,
                                   const ComputeType* __restrict__ emb_norm,
                                   const ComputeType* __restrict__ gate_w,
                                   ComputeType* __restrict__ x0,
                                   float* __restrict__ sig) {
  const int row = blockIdx.x;
  const bool first = row % p.seq == 0;
  const std::int64_t base = static_cast<std::int64_t>(row) * p.hidden;
  const ComputeType* cur = emb_norm + base;

  __shared__ float shared_gate;
  if (threadIdx.x == 0) {
    // The gate is `channels` products: cheaper to compute once than to reduce.
    float pre = 0.0f;
    if (!first) {
      for (int j = 0; j < p.channels; ++j) {
        pre += AsFloatDev(cur[j]) * AsFloatDev(gate_w[j]);
      }
    }
    const float s = first ? 0.0f : 1.0f / (1.0f + expf(-pre));
    if (sig != nullptr) sig[row] = s;
    shared_gate = p.lambda * s;
  }
  __syncthreads();

  const float gate = shared_gate;
  const ComputeType* prev = first ? cur : cur - p.hidden;
  ComputeType* dst = x0 + base;
  for (int i = threadIdx.x; i < p.hidden; i += blockDim.x) {
    dst[i] = ToComputeDev(AsFloatDev(cur[i]) + gate * AsFloatDev(prev[i]));
  }
}

__global__ void SmearBackwardKernel(
    SmearParams p, const ComputeType* __restrict__ emb_norm,
    const ComputeType* __restrict__ gate_w, const float* __restrict__ sig,
    const ComputeType* __restrict__ d_x0, float* __restrict__ gate_w_grad,
    float* __restrict__ lambda_grad, ComputeType* __restrict__ de,
    float* __restrict__ partials, int stride, float* __restrict__ dgate_rows,
    unsigned int* __restrict__ done) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const bool first = row % p.seq == 0;
  const bool tail = row % p.seq == p.seq - 1;
  const std::int64_t base = static_cast<std::int64_t>(row) * p.hidden;
  const float s = first ? 0.0f : sig[row];
  // The gate the NEXT row applies to this row's gradient. Past the end of a
  // sequence there is no such row, and the term is zero.
  const float gate_next = tail ? 0.0f : p.lambda * sig[row + 1];

  // de starts as d_x0, then takes this row's gate term and, last, the gate term
  // the following row points back with. Reading the next row, instead of
  // contributing into the previous one, keeps one writer per output row.
  for (int i = tid; i < p.hidden; i += blockDim.x) {
    de[base + i] = d_x0[base + i];
  }

  // The gate gradient is one dot over the row, and it is the whole source of
  // arithmetic freedom in this family: the reference accumulates it in double,
  // in ascending column order. Keeping that order here costs one lane per row
  // (the parallelism is across rows) and makes the device gradient identical to
  // the reference, which is what lets the committed fixtures and the numeric
  // gate read the tree's own noise floor instead of this kernel's ordering.
  __shared__ float shared_dgate;
  float dgate_f = 0.0f;
  if (!first) {
    if (tid == 0) {
      double dgate = 0.0;
      for (int i = 0; i < p.hidden; ++i) {
        dgate += static_cast<double>(AsFloatDev(d_x0[base + i])) *
                 static_cast<double>(AsFloatDev(emb_norm[base - p.hidden + i]));
      }
      shared_dgate = static_cast<float>(dgate);
      dgate_rows[row] = shared_dgate;
    }
    __syncthreads();
    dgate_f = shared_dgate;
  } else if (tid == 0) {
    dgate_rows[row] = 0.0f;  // position 0 of a sequence carries no gate
  }
  const float dpre = first ? 0.0f : dgate_f * p.lambda * s * (1.0f - s);
  for (int j = tid; j < p.channels; j += blockDim.x) {
    partials[static_cast<std::int64_t>(row) * stride + j] =
        dpre * AsFloatDev(emb_norm[base + j]);
    if (!first) {
      de[base + j] =
          ToComputeDev(AsFloatDev(de[base + j]) + dpre * AsFloatDev(gate_w[j]));
    }
  }
  if (!tail) {
    for (int i = tid; i < p.hidden; i += blockDim.x) {
      de[base + i] =
          ToComputeDev(AsFloatDev(de[base + i]) +
                       gate_next * AsFloatDev(d_x0[base + p.hidden + i]));
    }
  }

  // Phase boundary: the last block combines every block's partial in ascending
  // block order, which fixes the summation order across rows.
  __threadfence();
  __shared__ bool shared_last;
  if (tid == 0) {
    shared_last = atomicAdd(done, 1u) == static_cast<unsigned>(gridDim.x - 1);
  }
  __syncthreads();
  if (!shared_last) return;

  if (tid < p.channels) {
    float total = 0.0f;
    for (unsigned int b = 0; b < gridDim.x; ++b) {
      total += partials[static_cast<std::int64_t>(b) * stride + tid];
    }
    gate_w_grad[tid] += total;
  } else if (tid == p.channels) {
    // One lane, ascending rows, double: the reference's own summation.
    double total = 0.0;
    for (unsigned int b = 0; b < gridDim.x; ++b) {
      total += static_cast<double>(dgate_rows[b]) * sig[b];
    }
    *lambda_grad += static_cast<float>(total);
  }
}

}  // namespace

void SmearForward(const SmearParams& params, const ComputeType* emb_norm,
                  const ComputeType* gate_w, ComputeType* x0, float* sig) {
  const int rows = params.rows();
  if (rows <= 0 || params.hidden <= 0 || params.channels <= 0) return;
  if (params.channels > params.hidden) return;
  cuda_backend::Launch(SmearForwardKernel, dim3(rows), dim3(kSmearThreads), 0,
                       params, emb_norm, gate_w, x0, sig);
}

void SmearBackward(const SmearParams& params, const ComputeType* emb_norm,
                   const ComputeType* gate_w, const float* sig,
                   const ComputeType* d_x0, float* gate_w_grad,
                   float* lambda_grad, ComputeType* de) {
  const int rows = params.rows();
  if (rows <= 0 || params.hidden <= 0 || params.channels <= 0) return;
  if (params.channels > params.hidden) return;
  const int stride = PartialStride(params.channels);
  // The scratch holds one partial row per block, the per-row gate values the
  // combine needs, and one arrival counter. It lives for one launch and is
  // reused across steps, so the steady state issues no cudaMalloc and no
  // cudaFree. One process owns one device (`src/ops.cc`, `Fused`), so a
  // file-scope cache is the whole concurrency story.
  static float* scratch = nullptr;
  static std::size_t capacity = 0;
  const std::size_t partial_words =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(stride);
  const std::size_t wanted = partial_words + static_cast<std::size_t>(rows) + 1;
  if (wanted > capacity) {
    if (scratch != nullptr) Free(scratch);
    scratch = static_cast<float*>(Alloc(wanted * sizeof(float)));
    capacity = wanted;
  }
  float* dgate_rows = scratch + partial_words;
  unsigned int* done = reinterpret_cast<unsigned int*>(
      dgate_rows + static_cast<std::size_t>(rows));
  Memset(done, 0, sizeof(unsigned int));
  cuda_backend::Launch(SmearBackwardKernel, dim3(rows), dim3(kSmearThreads), 0,
                       params, emb_norm, gate_w, sig, d_x0, gate_w_grad,
                       lambda_grad, de, scratch, stride, dgate_rows, done);
}

}  // namespace kernels
}  // namespace nanochat
