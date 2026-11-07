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
// **One staged row walks the contiguous axis of the operand.** The element the
// fragment reads as its leading dimension is the staged row, and the staged
// columns are the axis the operand memory runs through. A transposed call
// therefore stages the same way as a plain one, from the other axis:
//
//   * A plain, A is `m` by `k` with `k` contiguous: rows are `m`, columns `k`,
//     and the fragment reads row major;
//   * A transposed, A is `k` by `m` with `m` contiguous: rows are `k`, columns
//     `m`, and the fragment reads column major;
//   * B plain, B is `k` by `n` with `n` contiguous: rows are `k`, columns `n`,
//     row major;
//   * B transposed, B is `n` by `k` with `k` contiguous: rows are `n`, columns
//     `k`, column major.
//
// The pad after a staged row only breaks the bank the row stride would repeat.
// The pad is never read.
//
// **One copy path, in chunks.** A staged row moves in sixteen-byte chunks
// through `__pipeline_memcpy_async`, and the thread index walks the (row,
// chunk) grid in row-major order, so threads that are next to each other read
// addresses that are next to each other. A chunk that is partly outside the
// operand, or whose ends are not sixteen-byte aligned, takes a per-element
// loop instead. The zero fill of a k tail and of a ragged edge is part of that
// loop, so an operand edge never adds a product.
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

// True when a whole chunk at `src` and `dst` can move as one asynchronous
// sixteen-byte copy.
__device__ __forceinline__ bool ChunkAligned(const TileElement* dst,
                                             const TileElement* src) {
  return ((reinterpret_cast<std::uintptr_t>(src) % 16u) == 0u) &&
         ((reinterpret_cast<std::uintptr_t>(dst) % 16u) == 0u);
}

// --- The staged slice ------------------------------------------------------

// One axis of the staged tile: `rows` staged rows of `columns` elements, from
// `base` with `row_stride` elements between two of them. `valid_rows` and
// `valid_columns` are the part of the tile the operand covers; everything
// outside is the zero that a k tail and a ragged edge need.
struct StagedTile {
  const TileElement* base;
  TileElement* dst;
  int row_stride;
  int lead;
  int rows;
  int valid_rows;
  int valid_columns;
};

// Stages one operand of one k slice. `Columns` is a compile-time extent, so the
// thread grid divides its flat index by a constant and the two paths carry no
// runtime division. Consecutive threads take consecutive chunks of one row,
// which is what makes the read coalesced.
template <int Columns>
__device__ __forceinline__ void StageOperand(const StagedTile& tile) {
  constexpr int kChunksPerRow = Columns / kChunk;
  const int total = tile.rows * kChunksPerRow;
  const int stride = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += stride) {
    // Rows are a multiple of the chunk, so a shift is exact and cheap.
    const int row = index / kChunksPerRow;
    const int column = (index - row * kChunksPerRow) * kChunk;
    TileElement* dst = tile.dst + row * tile.lead + column;
    const bool row_valid = row < tile.valid_rows;
    const int left = tile.valid_columns - column;
    const int count = left >= kChunk ? kChunk : (left > 0 ? left : 0);
    if (row_valid && count == kChunk) {
      const TileElement* src = tile.base + row * tile.row_stride + column;
      if (ChunkAligned(dst, src)) {
        CopyChunk(dst, src, kChunk * static_cast<int>(sizeof(TileElement)));
        continue;
      }
    }
    for (int e = 0; e < kChunk; ++e) {
      dst[e] = (row_valid && e < count)
                   ? tile.base[row * tile.row_stride + column + e]
                   : __float2bfloat16(0.0f);
    }
  }
}

// Stages one k slice of A. `transpose_a` states that A is stored `k` by `m`, so
// the m axis is the contiguous one and the staged rows are the k axis.
template <typename PlanT>
__device__ __forceinline__ void StageA(const TileElement* a, int lda,
                                       bool transpose_a, int m0, int m, int k0,
                                       int k, TileElement* stage) {
  if (transpose_a) {
    const StagedTile tile{a + static_cast<long long>(k0) * lda + m0,
                          stage,
                          lda,
                          PlanT::kLeadACol,
                          PlanT::kTileK,
                          k - k0,
                          m - m0};
    StageOperand<PlanT::kTileM>(tile);
    return;
  }
  const StagedTile tile{a + static_cast<long long>(m0) * lda + k0,
                        stage,
                        lda,
                        PlanT::kLeadARow,
                        PlanT::kTileM,
                        m - m0,
                        k - k0};
  StageOperand<PlanT::kTileK>(tile);
}

// Stages one k slice of B. `transpose_b` states that B is stored `n` by `k`, so
// the k axis is the contiguous one and the staged rows are the n axis.
template <typename PlanT>
__device__ __forceinline__ void StageB(const TileElement* b, int ldb,
                                       bool transpose_b, int n0, int n, int k0,
                                       int k, TileElement* stage) {
  if (transpose_b) {
    const StagedTile tile{b + static_cast<long long>(n0) * ldb + k0,
                          stage,
                          ldb,
                          PlanT::kLeadBCol,
                          PlanT::kTileN,
                          n - n0,
                          k - k0};
    StageOperand<PlanT::kTileK>(tile);
    return;
  }
  const StagedTile tile{b + static_cast<long long>(k0) * ldb + n0,
                        stage,
                        ldb,
                        PlanT::kLeadBRow,
                        PlanT::kTileK,
                        k - k0,
                        n - n0};
  StageOperand<PlanT::kTileN>(tile);
}

// Stages both operands of one k slice. Every thread calls it, so the two
// ordering calls that follow are uniform across the block.
template <typename PlanT>
__device__ __forceinline__ void StageSlice(
    const TileElement* a, int lda, bool transpose_a, const TileElement* b,
    int ldb, bool transpose_b, int m0, int n0, int m, int n, int k0, int k,
    TileElement* stage_a, TileElement* stage_b) {
  StageA<PlanT>(a, lda, transpose_a, m0, m, k0, k, stage_a);
  StageB<PlanT>(b, ldb, transpose_b, n0, n, k0, k, stage_b);
}

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_
