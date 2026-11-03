// CUDA Classifier family: logit softcap + cross-entropy over the real
// vocabulary slice, forward and backward. `logits` is
// `[rows, padded_vocab_size]`; only the first `vocab_size` columns are read and
// the backward zeroes the padded tail, matching the CPU reference.
//
// The `targets` are `int` on the host (the frozen seam's convention), so the
// entry point stages them through the one reusable per-thread buffer of
// `device_utils.cuh`. A launch may still read that buffer when the entry point
// returns, which is safe: the next staging copy is synchronous and waits for
// every queued launch before it writes. One block per row, reduced
// cooperatively through the shared RowLogSumExp helper.
//
// See docs/kernels.md.

#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::RowLogSumExp;
using cuda_kernels::ToComputeDev;

// The staged target ids are read by a launch, so that kernel may still be in
// flight when the entry point returns. That is correct: the next staging copy
// is synchronous, and it waits for every queued launch before it writes.

__host__ __device__ inline int PaddedVocab(const ClassifierParams& params) {
  return params.padded_vocab_size > 0 ? params.padded_vocab_size
                                      : params.vocab_size;
}

__global__ void ClassifierForwardKernel(const ClassifierParams params,
                                        const ComputeType* __restrict__ logits,
                                        const int* __restrict__ targets,
                                        float* __restrict__ losses) {
  const int row = blockIdx.x;
  const int target = targets[row];
  if (target == params.ignore_index) {
    if (threadIdx.x == 0) losses[row] = 0.0f;
    return;
  }
  const int padded = PaddedVocab(params);
  const ComputeType* lrow = logits + static_cast<long long>(row) * padded;
  float row_max = 0.0f;
  float sum_exp = 0.0f;
  RowLogSumExp(lrow, params.vocab_size, params.softcap, &row_max, &sum_exp);
  if (threadIdx.x == 0) {
    // An out-of-range target contributes a zero, which is the behavior the
    // composition defines (`src/ops.cc`, `ClassifierForward`): the caller
    // produces the ids, and one defensive branch is cheaper than a device-side
    // fault. Without it a target outside `[0, vocab_size)` read past the row,
    // which is what the RL stage hit.
    float z_target = 0.0f;
    if (target >= 0 && target < params.vocab_size) {
      z_target =
          params.softcap * tanhf(AsFloatDev(lrow[target]) / params.softcap);
    }
    losses[row] = logf(sum_exp) + row_max - z_target;
  }
}

__global__ void ClassifierBackwardKernel(const ClassifierParams params,
                                         const ComputeType* __restrict__ logits,
                                         const int* __restrict__ targets,
                                         ComputeType* __restrict__ dlogits) {
  const int row = blockIdx.x;
  const int vocab = params.vocab_size;
  const int padded = PaddedVocab(params);
  const long long base = static_cast<long long>(row) * padded;
  const int target = targets[row];
  if (target == params.ignore_index) {
    for (int j = threadIdx.x; j < padded; j += blockDim.x) {
      dlogits[base + j] = ToComputeDev(0.0f);
    }
    return;
  }
  // A non-null row scale multiplies the whole row after the softcap chain
  // rule. A null pointer keeps the unweighted arithmetic.
  const float row_scale =
      params.row_scale != nullptr ? params.row_scale[row] : 1.0f;
  const ComputeType* lrow = logits + base;
  float row_max = 0.0f;
  float sum_exp = 0.0f;
  RowLogSumExp(lrow, vocab, params.softcap, &row_max, &sum_exp);
  const float inv = 1.0f / sum_exp;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
    const float t = tanhf(AsFloatDev(lrow[j]) / params.softcap);
    const float sech2 = 1.0f - t * t;
    const float z = params.softcap * t;
    const float p = expf(z - row_max) * inv;
    const float onehot = (j == target) ? 1.0f : 0.0f;
    // dL/d(raw logit) = (softmax - onehot) * d(softcap)/d(raw).
    dlogits[base + j] = ToComputeDev((p - onehot) * sech2 * row_scale);
  }
  for (int j = vocab + threadIdx.x; j < padded; j += blockDim.x) {
    dlogits[base + j] = ToComputeDev(0.0f);
  }
}

}  // namespace

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       float* losses) {
  if (params.rows <= 0 || params.vocab_size <= 0) return;
  const int* device_targets = cuda_kernels::StageIndices(targets, params.rows);
  constexpr int kThreads = 256;
  cuda_backend::Launch(ClassifierForwardKernel, dim3(params.rows),
                       dim3(kThreads), 0, params, logits, device_targets,
                       losses);
}

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits) {
  if (params.rows <= 0 || params.vocab_size <= 0) return;
  const int* device_targets = cuda_kernels::StageIndices(targets, params.rows);
  constexpr int kThreads = 256;
  cuda_backend::Launch(ClassifierBackwardKernel, dim3(params.rows),
                       dim3(kThreads), 0, params, logits, device_targets,
                       dlogits);
}

}  // namespace kernels
}  // namespace nanochat
