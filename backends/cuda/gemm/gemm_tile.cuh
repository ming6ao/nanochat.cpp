#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_

// The staged operand tile of the hand tensor-core GEMM
// (docs/kernels.md Part 3, docs/tensor-core-gemm-plan.md Part B).
//
// A block walks the k extent in slices of `Plan::kTileK`. Each slice is staged
// into shared memory in the layout `wmma` reads, and the staging is two
// buffered: slice `n + 1` is copied while slice `n` is used, so the global
// latency of one slice hides behind the arithmetic of the one before it.
//
// A staged row is `kTileK` wide for A and `kTileN` wide for B, in the layout
// the fragments read, and the pad after it only breaks the bank the row stride
// would repeat. The pad is never read.
//
// Two copy paths, and the call decides which:
//
//   * an operand whose k index steps by one element is contiguous, so a staged
//     row moves in sixteen-byte chunks through `__pipeline_memcpy_async`;
//   * any other operand moves one element at a time.
//
// A chunk also needs both ends aligned to sixteen bytes. A row whose global end
// is not falls back to the element path, which is what the 5 by 12 Muon group
// takes and what a training step with a multiple-of-eight hidden width does
// not.
//
// Both paths zero the part of a row outside the operand, so a k tail and a
// ragged edge never add a product.
//
// A bf16 `wmma` needs a tensor core with bf16, so this header exists only in a
// bf16 or fp8 build. The fp32 build keeps cuBLAS.

#if defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {
namespace gemm_tile {

// The element the tile stages and the fragments hold. The vendor fragment is
// specialized for `__nv_bfloat16`, and the seam's `ComputeType` is the tree's
// `Bf16`, so the tile names the vendor type and views the operand memory
// through it. The emulator maps the two to one type.
using TileElement = __nv_bfloat16;

//: Elements of one staged chunk, which is sixteen bytes.
constexpr int kChunk = 8;
//: A micro-tile edge, the only shape `wmma` 16 by 16 by 16 covers here.
constexpr int kMicro = 16;

// --- The asynchronous copy -------------------------------------------------

// The pipeline needs `sm_80`. Above it the copy is asynchronous and the wait
// orders it; below it, and on the emulator, the copy completes at the call and
// the two ordering calls do nothing. One header covers both, so the kernel has
// no second body for Turing (docs/tensor-core-gemm-plan.md section A.3).
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void CopyChunk(TileElement* dst,
                                          const TileElement* src, int bytes) {
  __pipeline_memcpy_async(dst, src, static_cast<std::size_t>(bytes));
}

__device__ __forceinline__ void CommitStage() { __pipeline_commit(); }

// Waits until at most `in_flight` staged groups are still moving.
__device__ __forceinline__ void WaitStage(int in_flight) {
  __pipeline_wait_prior(static_cast<std::size_t>(in_flight));
}
#else
__device__ __forceinline__ void CopyChunk(TileElement* dst,
                                          const TileElement* src, int bytes) {
  const int elements = bytes / static_cast<int>(sizeof(TileElement));
  for (int i = 0; i < elements; ++i) dst[i] = src[i];
}

__device__ __forceinline__ void CommitStage() {}

__device__ __forceinline__ void WaitStage(int /*in_flight*/) {}
#endif

// True when a chunk of `count` elements at `src` can move as one asynchronous
// sixteen-byte copy.
__device__ __forceinline__ bool ChunkAligned(const TileElement* src,
                                             int count) {
  return count == kChunk && (reinterpret_cast<std::uintptr_t>(src) % 16u) == 0u;
}

// --- The staged slice ------------------------------------------------------

// One staged row of A, or the zero that stands in for it. `contiguous` states
// that `stride_k` is one; the branch is uniform across the block, because it
// reads only the call's flags, so the two ordering calls stay uniform too.
template <typename PlanT>
__device__ __forceinline__ void StageARow(const TileElement* base, int stride_i,
                                          int stride_k, bool contiguous, int m0,
                                          int m, int k0, int k, int i,
                                          TileElement* row) {
  if (!contiguous || (m0 + i) >= m) {
    const bool in_range = contiguous ? false : ((m0 + i) < m);
    for (int kk = 0; kk < PlanT::kTileK; ++kk) {
      row[kk] = (in_range && (k0 + kk) < k)
                    ? base[i * stride_i + (k0 + kk) * stride_k]
                    : __float2bfloat16(0.0f);
    }
    return;
  }
  const int remaining = k - k0;
  for (int kk = 0; kk < PlanT::kTileK; kk += kChunk) {
    const int left = remaining - kk;
    const int count = left >= kChunk ? kChunk : (left > 0 ? left : 0);
    const TileElement* src = base + i * stride_i + k0 + kk;
    if (ChunkAligned(src, count)) {
      CopyChunk(row + kk, src, kChunk * static_cast<int>(sizeof(TileElement)));
    } else {
      for (int e = 0; e < kChunk; ++e) {
        row[kk + e] = (e < count) ? src[e] : __float2bfloat16(0.0f);
      }
    }
  }
}

// One staged row of B, with the same two paths. A B whose n index steps by one
// element is the contiguous form.
template <typename PlanT>
__device__ __forceinline__ void StageBRow(const TileElement* base, int stride_k,
                                          int stride_j, bool contiguous, int n0,
                                          int n, int k0, int k, int kk,
                                          TileElement* row) {
  if (!contiguous || (k0 + kk) >= k) {
    const bool in_range = contiguous ? false : ((k0 + kk) < k);
    for (int j = 0; j < PlanT::kTileN; ++j) {
      row[j] = (in_range && (n0 + j) < n)
                   ? base[(k0 + kk) * stride_k + j * stride_j]
                   : __float2bfloat16(0.0f);
    }
    return;
  }
  const int remaining = n - n0;
  for (int j = 0; j < PlanT::kTileN; j += kChunk) {
    const int left = remaining - j;
    const int count = left >= kChunk ? kChunk : (left > 0 ? left : 0);
    const TileElement* src = base + (k0 + kk) * stride_k + j;
    if (ChunkAligned(src, count)) {
      CopyChunk(row + j, src, kChunk * static_cast<int>(sizeof(TileElement)));
    } else {
      for (int e = 0; e < kChunk; ++e) {
        row[j + e] = (e < count) ? src[e] : __float2bfloat16(0.0f);
      }
    }
  }
}

// Stages one k slice: every A row and every B row of the slice, in whatever
// copy path the call allows. The thread grid walks the rows and the rows walk
// the chunks, so both paths cover the slice without a nested loop.
template <typename PlanT>
__device__ __forceinline__ void StageSlice(
    const TileElement* a_base, int a_stride_i, int a_stride_k,
    const TileElement* b_base, int b_stride_k, int b_stride_j, int m0, int n0,
    int m, int n, int k0, int k, TileElement* stage_a, TileElement* stage_b) {
  const int stride = static_cast<int>(blockDim.x);
  const bool a_contiguous = a_stride_k == 1;
  const bool b_contiguous = b_stride_j == 1;
  for (int i = static_cast<int>(threadIdx.x); i < PlanT::kTileM; i += stride) {
    StageARow<PlanT>(a_base, a_stride_i, a_stride_k, a_contiguous, m0, m, k0, k,
                     i, stage_a + i * PlanT::kLeadA);
  }
  for (int kk = static_cast<int>(threadIdx.x); kk < PlanT::kTileK;
       kk += stride) {
    StageBRow<PlanT>(b_base, b_stride_k, b_stride_j, b_contiguous, n0, n, k0, k,
                     kk, stage_b + kk * PlanT::kLeadB);
  }
}

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_
