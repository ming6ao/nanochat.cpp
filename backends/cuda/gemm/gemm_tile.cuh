#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_
#define NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_

// The block tile of the hand-written tensor-core GEMM
// (docs/kernels.md Part 3, docs/tensor-core-gemm-plan.md Part B).
//
//   grid    ceil(n / kTileN) by ceil(m / kTileM) by batch
//   block   kThreads threads, kWarpsM by kWarpsN warps
//   warp    kWarpM by kWarpN of C, kFragsM by kFragsN fragments
//
// One shared window serves both uses: the operand stage during the k-loop, and
// the fp32 epilogue scratch after it. The scratch is the larger, so one buffer
// holds either.
//
// The copy path is a synchronous gather. The asynchronous 16-byte path of
// `__pipeline_memcpy_async` lands with the measurement in section B.4; this
// layout is what it must preserve.
//
// A bf16 `wmma` needs a tensor core with bf16, so this header exists only in a
// bf16 or fp8 build. The fp32 build keeps cuBLAS.

#if defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

// Rows and columns of C one block owns, and the depth of one k slice.
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 128;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = kThreads / 32 / kWarpsM;
constexpr int kWarpM = kTileM / kWarpsM;
constexpr int kWarpN = kTileN / kWarpsN;
constexpr int kFragsM = kWarpM / 16;
constexpr int kFragsN = kWarpN / 16;

// Pad on a staged row, for the bank the row stride would otherwise repeat.
constexpr int kPad = 8;
constexpr int kLeadA = kTileK + kPad;
constexpr int kLeadB = kTileN + kPad;

constexpr int kStageAElements = kTileM * kLeadA;
constexpr int kStageBElements = kTileK * kLeadB;
constexpr int kStageElements = kStageAElements + kStageBElements;
// The epilogue scratch: one float per element of C, held as two tile elements.
constexpr int kScratchElements = kTileM * kTileN * 2;
constexpr int kSmemElements =
    kStageElements > kScratchElements ? kStageElements : kScratchElements;

constexpr int kMicro = 16;

static_assert(sizeof(TileElement) == sizeof(ComputeType),
              "the operand memory is viewed as the vendor bf16, not converted");
static_assert(kTileM % (kWarpsM * kMicro) == 0 &&
                  kTileN % (kWarpsN * kMicro) == 0,
              "the warp grid must cover the tile in whole fragments");
static_assert(kTileK % kMicro == 0, "one k slice must split into whole tiles");
static_assert(kStageElements <= kScratchElements,
              "one shared window holds either use, so the scratch dominates");

// The resolved operands of one call. Every seam default is settled on the host,
// by the same rules `GemmImpl` in `backends/cuda/gemm.cu` applies.
struct GemmTileArgs {
  int m = 0;
  int n = 0;
  int k = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  long long stride_a = 0;
  long long stride_b = 0;
  long long stride_c = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool transpose_a = false;
  bool transpose_b = false;
};

// Stages one A slice: `stage[i * kLeadA + kk] = op(A)[m0 + i][k0 + kk]`, and
// zero outside the operand. A plain A is `base = a + m0 * lda`,
// `stride_i = lda`, `stride_k = 1`; a transposed A is `base = a + m0`,
// `stride_i = 1`, `stride_k = lda`.
__device__ __forceinline__ void StageASlice(const TileElement* base,
                                            int stride_i, int stride_k, int m0,
                                            int m, int k0, int k,
                                            TileElement* stage) {
  const int stride = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < kTileM * kTileK;
       index += stride) {
    const int i = index / kTileK;
    const int kk = index - i * kTileK;
    const bool inside = (m0 + i) < m && (k0 + kk) < k;
    stage[i * kLeadA + kk] = inside ? base[i * stride_i + (k0 + kk) * stride_k]
                                    : __float2bfloat16(0.0f);
  }
}

// Stages one B slice: `stage[kk * kLeadB + j] = op(B)[k0 + kk][n0 + j]`, and
// zero outside the operand. A plain B is `base = b + n0`, `stride_k = ldb`,
// `stride_j = 1`; a transposed B is `base = b + n0 * ldb`, `stride_k = 1`,
// `stride_j = ldb`.
__device__ __forceinline__ void StageBSlice(const TileElement* base,
                                            int stride_k, int stride_j, int n0,
                                            int n, int k0, int k,
                                            TileElement* stage) {
  const int stride = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < kTileK * kTileN;
       index += stride) {
    const int kk = index / kTileN;
    const int j = index - kk * kTileN;
    const bool inside = (n0 + j) < n && (k0 + kk) < k;
    stage[kk * kLeadB + j] = inside ? base[(k0 + kk) * stride_k + j * stride_j]
                                    : __float2bfloat16(0.0f);
  }
}

}  // namespace gemm_tile
}  // namespace kernels
}  // namespace nanochat

#endif  // defined(NANOCHAT_PRECISION) && (NANOCHAT_PRECISION != 0)

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_GEMM_TILE_CUH_
