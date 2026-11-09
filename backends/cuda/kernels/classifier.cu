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
using cuda_kernels::ComputeVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::RowLogSumExp;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorReady;

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
                                        float* __restrict__ losses,
                                        float* __restrict__ stats) {
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
    // The pair the backward would rebuild from the same row (docs/kernels.md
    // Part 4). It is the helper's own output, so the two paths are identical
    // and not merely close.
    if (stats != nullptr) {
      stats[2 * row] = row_max;
      stats[2 * row + 1] = sum_exp;
    }
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
                                         ComputeType* __restrict__ dlogits,
                                         const float* __restrict__ stats,
                                         float dlogit_scale) {
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
  if (stats != nullptr) {
    // The forward's own pair, so this pass reads the logits once: for the
    // softcapped value it writes. `RowLogSumExp` would read the row twice more.
    row_max = stats[2 * row];
    sum_exp = stats[2 * row + 1];
  } else {
    RowLogSumExp(lrow, vocab, params.softcap, &row_max, &sum_exp);
  }
  const float inv = 1.0f / sum_exp;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
    const float t = tanhf(AsFloatDev(lrow[j]) / params.softcap);
    const float sech2 = 1.0f - t * t;
    const float z = params.softcap * t;
    const float p = expf(z - row_max) * inv;
    const float onehot = (j == target) ? 1.0f : 0.0f;
    // dL/d(raw logit) = (softmax - onehot) * d(softcap)/d(raw).
    //
    // The store keeps the two roundings a separate scale pass had: the value
    // rounds to the compute type, and the normalization multiplies that
    // rounded value. A bf16 build therefore keeps the bits it shipped with.
    const float value = (p - onehot) * sech2 * row_scale;
    dlogits[base + j] =
        ToComputeDev(AsFloatDev(ToComputeDev(value)) * dlogit_scale);
  }
  for (int j = vocab + threadIdx.x; j < padded; j += blockDim.x) {
    dlogits[base + j] = ToComputeDev(0.0f);
  }
}

// The same gradient write, one aligned 16-byte vector per thread. The
// reduction stays `RowLogSumExp`, so `row_max` and `sum_exp` keep their order
// and every element is computed from its own logit and those two values. The
// packed path therefore cannot move a value, and only the traffic changes
// (docs/kernels.md Part 4 item 1). The padded tail keeps its scalar zeroing.
__global__ void ClassifierBackwardVectorKernel(
    const ClassifierParams params, const ComputeType* __restrict__ logits,
    const int* __restrict__ targets, ComputeType* __restrict__ dlogits,
    const float* __restrict__ stats, float dlogit_scale, int vectors) {
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
  const float row_scale =
      params.row_scale != nullptr ? params.row_scale[row] : 1.0f;
  const ComputeType* lrow = logits + base;
  ComputeVector* dv = reinterpret_cast<ComputeVector*>(dlogits + base);
  const ComputeVector* lv = reinterpret_cast<const ComputeVector*>(lrow);
  float row_max = 0.0f;
  float sum_exp = 0.0f;
  if (stats != nullptr) {
    row_max = stats[2 * row];
    sum_exp = stats[2 * row + 1];
  } else {
    RowLogSumExp(lrow, vocab, params.softcap, &row_max, &sum_exp);
  }
  const float inv = 1.0f / sum_exp;
  for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
    const ComputeVector in = lv[v];
    ComputeVector result;
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      const int j = v * kVectorElems + k;
      const float t = tanhf(AsFloatDev(in.e[k]) / params.softcap);
      const float sech2 = 1.0f - t * t;
      const float z = params.softcap * t;
      const float p = expf(z - row_max) * inv;
      const float onehot = (j == target) ? 1.0f : 0.0f;
      // The same expression and the same two roundings as the scalar kernel.
      const float value = (p - onehot) * sech2 * row_scale;
      result.e[k] =
          ToComputeDev(AsFloatDev(ToComputeDev(value)) * dlogit_scale);
    }
    dv[v] = result;
  }
  for (int j = vocab + threadIdx.x; j < padded; j += blockDim.x) {
    dlogits[base + j] = ToComputeDev(0.0f);
  }
}

}  // namespace

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       float* losses, float* stats) {
  if (params.rows <= 0 || params.vocab_size <= 0) return;
  const int* device_targets = cuda_kernels::StageIndices(targets, params.rows);
  constexpr int kThreads = 256;
  cuda_backend::Launch(ClassifierForwardKernel, dim3(params.rows),
                       dim3(kThreads), 0, params, logits, device_targets,
                       losses, stats);
}

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits, const float* stats,
                        float dlogit_scale) {
  if (params.rows <= 0 || params.vocab_size <= 0) return;
  const int* device_targets = cuda_kernels::StageIndices(targets, params.rows);
  constexpr int kThreads = 256;
  const int padded = PaddedVocab(params);
  if (params.vocab_size % kVectorElems == 0 &&
      VectorReady(padded, {logits, dlogits})) {
    cuda_backend::Launch(ClassifierBackwardVectorKernel, dim3(params.rows),
                         dim3(kThreads), 0, params, logits, device_targets,
                         dlogits, stats, dlogit_scale,
                         params.vocab_size / kVectorElems);
    return;
  }
  cuda_backend::Launch(ClassifierBackwardKernel, dim3(params.rows),
                       dim3(kThreads), 0, params, logits, device_targets,
                       dlogits, stats, dlogit_scale);
}

}  // namespace kernels
}  // namespace nanochat
