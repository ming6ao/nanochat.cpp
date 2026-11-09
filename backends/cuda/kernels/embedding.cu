// CUDA Embedding family: an indexed gather forward and a scatter-add backward
// over a persistent dense gradient buffer. The token ids are `int` on the host
// (the frozen seam's convention), so the entry points stage them through the
// one reusable per-thread buffer of `device_utils.cuh`. A launch may still read
// that buffer when the entry point returns, which is safe: the next staging
// copy is synchronous and waits for every queued launch before it writes.
//
// The backward contract is the CPU reference's: add each token row to the
// gradient buffer. The caller clears the buffer once with ZeroGrad, so repeated
// backward calls sum. A duplicate id adds every occurrence through the atomic
// scatter-add, so no zeroing pass is necessary.
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

using cuda_kernels::AsFloatDev;
using cuda_kernels::AtomicAddDev;
using cuda_kernels::ComputeVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorReady;

__global__ void EmbeddingForwardKernel(int tokens, int dim,
                                       const int* __restrict__ ids,
                                       const ComputeType* __restrict__ table,
                                       ComputeType* __restrict__ out) {
  const long long n = static_cast<long long>(tokens) * dim;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long idx =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const long long token = idx / dim;
    const int d = static_cast<int>(idx - token * dim);
    out[idx] = table[static_cast<long long>(ids[token]) * dim + d];
  }
}

// The same gather, one aligned 16-byte vector per thread. The copy is the
// scalar copy: a whole vector moves from one row of the table to one row of
// the output, so no value is read or written in a different order. The row
// must divide by the vector width and both pointers must be aligned, which the
// entry point checks (docs/kernels.md Part 4 item 1).
__global__ void EmbeddingForwardVectorKernel(
    int tokens, int vectors_per_row, const int* __restrict__ ids,
    const ComputeVector* __restrict__ table, ComputeVector* __restrict__ out) {
  const long long n = static_cast<long long>(tokens) * vectors_per_row;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long idx =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const long long token = idx / vectors_per_row;
    const long long d = idx - token * vectors_per_row;
    out[idx] = table[static_cast<long long>(ids[token]) * vectors_per_row + d];
  }
}

// Scatter-add each row's output gradient. Atomics keep duplicate ids correct.
__global__ void EmbeddingScatterAddKernel(int dim, const int* __restrict__ ids,
                                          const ComputeType* __restrict__ dout,
                                          float* __restrict__ dtable) {
  const int token = blockIdx.x;
  float* row = dtable + static_cast<long long>(ids[token]) * dim;
  const ComputeType* value = dout + static_cast<long long>(token) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    AtomicAddDev(&row[d], AsFloatDev(value[d]));
  }
}

}  // namespace

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out) {
  if (tokens <= 0 || dim <= 0) return;
  const int* device_ids = cuda_kernels::StageIndices(ids, tokens);
  constexpr int kThreads = 256;
  const long long n = static_cast<long long>(tokens) * dim;
  if (dim % kVectorElems == 0 && VectorReady(n, {table, out})) {
    const int vectors_per_row = dim / kVectorElems;
    const long long vectors = static_cast<long long>(tokens) * vectors_per_row;
    const int blocks = static_cast<int>((vectors + kThreads - 1) / kThreads);
    cuda_backend::Launch(EmbeddingForwardVectorKernel, dim3(blocks),
                         dim3(kThreads), 0, tokens, vectors_per_row, device_ids,
                         reinterpret_cast<const ComputeVector*>(table),
                         reinterpret_cast<ComputeVector*>(out));
    return;
  }
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  cuda_backend::Launch(EmbeddingForwardKernel, dim3(blocks), dim3(kThreads), 0,
                       tokens, dim, device_ids, table, out);
}

void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, float* dtable) {
  if (tokens <= 0 || dim <= 0) return;
  const int* device_ids = cuda_kernels::StageIndices(ids, tokens);
  const int block = cuda_kernels::BlockSizeForDim(dim);
  cuda_backend::Launch(EmbeddingScatterAddKernel, dim3(tokens), dim3(block), 0,
                       dim, device_ids, dout, dtable);
}

}  // namespace kernels
}  // namespace nanochat
