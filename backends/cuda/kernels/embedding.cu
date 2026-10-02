// CUDA Embedding family: an indexed gather forward and a scatter-add backward
// over a persistent dense gradient buffer. The token ids are `int` on the host
// (the frozen seam's convention), so the entry points stage them into device
// memory.
//
// The backward contract is the CPU reference's: zero only the rows this batch
// touches, then add the output gradients. Duplicate ids are safe because the
// zeroing pass runs in its own kernel before the scatter-add pass, and the
// scatter-add uses atomics for the (possible) duplicate rows.
//
// See docs/kernels.md and docs/model.md.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::ScatterAddRow;
using cuda_kernels::ToComputeDev;

// Stage the host-side token ids into device memory for the kernel's lifetime.
struct DeviceIds {
  int* ptr = nullptr;
  DeviceIds(const int* host, int count) {
    if (count > 0 && host != nullptr) {
      ptr = static_cast<int*>(Alloc(sizeof(int) * static_cast<std::size_t>(count)));
      Memcpy(ptr, host, sizeof(int) * static_cast<std::size_t>(count),
             CopyDir::kHostToDevice);
    }
  }
  ~DeviceIds() {
    if (ptr != nullptr) {
      Synchronize();
      Free(ptr);
    }
  }
  DeviceIds(const DeviceIds&) = delete;
  DeviceIds& operator=(const DeviceIds&) = delete;
};

__global__ void EmbeddingForwardKernel(int tokens, int dim,
                                       const int* __restrict__ ids,
                                       const ComputeType* __restrict__ table,
                                       ComputeType* __restrict__ out) {
  const long long n = static_cast<long long>(tokens) * dim;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       idx < n; idx += stride) {
    const long long token = idx / dim;
    const int d = static_cast<int>(idx - token * dim);
    out[idx] = table[static_cast<long long>(ids[token]) * dim + d];
  }
}

// Pass 1: zero the rows the batch touches. Duplicate ids simply write zero
// more than once.
__global__ void EmbeddingZeroRowsKernel(int dim, const int* __restrict__ ids,
                                        ComputeType* __restrict__ dtable) {
  const int token = blockIdx.x;
  ComputeType* row =
      dtable + static_cast<long long>(ids[token]) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    row[d] = ToComputeDev(0.0f);
  }
}

// Pass 2: scatter-add each row's output gradient. Atomics keep duplicate ids
// correct.
__global__ void EmbeddingScatterAddKernel(int dim, const int* __restrict__ ids,
                                          const ComputeType* __restrict__ dout,
                                          ComputeType* __restrict__ dtable) {
  const int token = blockIdx.x;
  ComputeType* row = dtable + static_cast<long long>(ids[token]) * dim;
  const ComputeType* value =
      dout + static_cast<long long>(token) * dim;
  ScatterAddRow(row, dim, value);
}

}  // namespace

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out) {
  if (tokens <= 0 || dim <= 0) return;
  const DeviceIds device_ids(ids, tokens);
  constexpr int kThreads = 256;
  const long long n = static_cast<long long>(tokens) * dim;
  const int blocks =
      static_cast<int>((n + kThreads - 1) / kThreads);
  cuda_backend::Launch(EmbeddingForwardKernel, dim3(blocks), dim3(kThreads), 0,
                       tokens, dim, device_ids.ptr, table, out);
}

void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, ComputeType* dtable) {
  if (tokens <= 0 || dim <= 0) return;
  const DeviceIds device_ids(ids, tokens);
  const int block = cuda_kernels::BlockSizeForDim(dim);
  cuda_backend::Launch(EmbeddingZeroRowsKernel, dim3(tokens), dim3(block), 0,
                       dim, device_ids.ptr, dtable);
  cuda_backend::Launch(EmbeddingScatterAddKernel, dim3(tokens), dim3(block), 0,
                       dim, device_ids.ptr, dout, dtable);
}

}  // namespace kernels
}  // namespace nanochat
