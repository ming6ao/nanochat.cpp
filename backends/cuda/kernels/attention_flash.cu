// The flash attention family of `docs/attention.md` Part 7: one forward tile
// and two backward kernels, all on `mma.sync.aligned.m16n8k16` with bf16
// operands and fp32 accumulators.
//
// The forward tile is one query block of 128 rows against key tiles of 64 rows,
// eight warps of 16 query rows each. It keeps the online-softmax state and the
// output accumulator in registers, skips every key tile that no query row in
// the block can see, and never materialises the `seq x seq` score matrix.
//
// Two properties of this file are what the family design buys over the `wmma`
// tile of `attention_bf16.cu`:
//   * the `m16n8` accumulator layout is documented, so the tile reads the row
//     of a score in registers and needs no fp32 scratch tile in shared memory;
//   * two adjacent 16 by 8 accumulators are the 16 by 16 A operand of the value
//     product, so the probability operand stays in registers between the two
//     products.
// The row maximum, the row sum, and the rescale factor are therefore a shuffle
// over the four lanes of a quad, and the probability tile of Part 6 leaves the
// kernel.
//
// Layouts. The query tile reads `[batch, heads, seq, dim]` when
// `q_head_major` is set and `[batch, seq, heads, dim]` otherwise, so one index
// function serves both and a head-major query is not a reason to leave the
// family. The key, value, output, and both gradients keep the model stream
// `[batch, seq, heads, dim]`.
//
// The backward is two kernels, one per direction, so no element is written
// twice and no atomic is needed:
//   * `FlashAttentionQueryGradKernel` owns one query block and writes `dq`
//     once per query row;
//   * `FlashAttentionKeyValueGradKernel` owns one key block of one key/value
//     head and writes `dk` and `dv` once per key row, folding the two query
//     heads of a group into the same registers.
// The forward keeps no probability matrix, so both kernels recompute the score
// from the saved statistics. The query kernel runs two passes over its key
// tiles: the first accumulates the per-row dot product that the second needs.
// `docs/attention.md` Part 7, section 4.4 states the issued-work arithmetic.
//
// The arithmetic mirrors `backends/cuda/kernels/testing/sequence_ref.h`: the
// row maximum over the visible keys, the causal plus sliding-window mask
// through `KeyAllowedDev`, and the `(0, 0)` statistics for an empty window. The
// exponential is base two, as in the reference kernel: the softmax scale
// carries `log2(e)`, so one fused multiply-add and one exponential serve each
// score, and the saved row maximum is converted back to natural units. Only the
// two products round to bf16, and so does the saved probability operand of each
// product.

// The family stages its operands in sixteen-byte copies of two-byte elements,
// so it needs a bf16 or fp8 build, and it needs the CUDA compiler, because
// `mma.sync` and `cp.async` do not compile on the host pass. The build answers
// both: it compiles this file only in a bf16 or fp8 build, and an fp32 build
// and the emulator link `attention_flash_absent.cc` instead.

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/attention_flash.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using CudaBf16 = __nv_bfloat16;

// --- Geometry -------------------------------------------------------------

//: Query rows per forward block, and per query-gradient block. Eight warps,
//: one 16-row fragment set per warp.
constexpr int kFlashBr = 128;
//: Key and value rows per key tile of the forward, and per key block of the
//: key and value gradient kernel.
constexpr int kFlashBc = 64;
//: Threads per block: eight warps.
constexpr int kFlashThreads = 256;
constexpr int kFlashWarps = kFlashThreads / 32;
constexpr int kFlashRowsPerWarp = kFlashBr / kFlashWarps;
//: Resident blocks per streaming multiprocessor. The register budget decides
//: it: one block of 256 threads at about 180 registers each.
constexpr int kFlashMinBlocks = 1;
//: The wider of the two head widths the family covers, and the one its shared
//: memory requests are sized for.
constexpr int kFlashMaxDim = 128;
//: Eight-key n-tiles of one key tile, and eight-dimension n-tiles of one head
//: row of width `kDim`.
constexpr int kFlashKeyTiles = kFlashBc / 8;
//: Keys per 16 by 16 A operand of the value product, which is also the mma `k`
//: step of both products.
constexpr int kFlashKeysPerChunk = 16;
constexpr int kFlashChunks = kFlashBc / kFlashKeysPerChunk;
//: Query rows per staged tile of the key and value gradient kernel. The tile
//: is 32 rows because that kernel holds `dk`, `dv`, the score, and the score
//: gradient of 16 key rows at once; 64 query rows spill registers.
constexpr int kFlashBackwardBq = 32;
//: Key rows per block of the key and value gradient kernel: one 16-row
//: fragment set per warp, so every warp owns work. A key block of the
//: forward's 64 rows would leave half the warps without a row.
constexpr int kFlashKvGradBc = kFlashWarps * kFlashRowsPerWarp;
//: Sixteen-byte copy granularity in two-byte elements.
constexpr int kFlashChunkElems = 8;

//: The base-two form of the exponential. `exp2` is one instruction where
//: `exp` is a range reduction around it, so the softmax scale carries
//: `log2(e)` and the saved row maximum is converted back to natural units.
constexpr float kFlashLog2e = 1.4426950408889634f;
constexpr float kFlashLn2 = 0.6931471805599453f;

//: One staged row of a key, value, query, or output-gradient tile: the head
//: width plus eight elements of bank padding. At width 128 the stride is 272
//: bytes, and at width 64 it is 144 bytes; both are multiples of sixteen, so
//: every staged segment stays aligned for a sixteen-byte copy.
template <int kDim>
__host__ __device__ constexpr int FlashStride() {
  return kDim + 8;
}

template <int kDim>
__host__ __device__ constexpr int FlashQueryElems() {
  return kFlashBr * FlashStride<kDim>();
}

template <int kDim>
__host__ __device__ constexpr int FlashKvElems() {
  return kFlashBc * FlashStride<kDim>();
}

template <int kDim>
__host__ __device__ constexpr int FlashBackwardQueryElems() {
  return kFlashBackwardBq * FlashStride<kDim>();
}

template <int kDim>
__host__ __device__ constexpr int FlashKvGradElems() {
  return kFlashKvGradBc * FlashStride<kDim>();
}

//: The forward tile: the resident query tile, two key stages, two value
//: stages, and no probability tile and no score scratch.
template <int kDim>
__host__ __device__ constexpr std::size_t FlashSharedBytes() {
  return (FlashQueryElems<kDim>() + 4 * FlashKvElems<kDim>()) *
         sizeof(CudaBf16);
}

//: The query-gradient kernel: the resident query and output-gradient tiles and
//: the two key and value stages.
template <int kDim>
__host__ __device__ constexpr std::size_t FlashQueryGradSharedBytes() {
  return (2 * FlashQueryElems<kDim>() + 4 * FlashKvElems<kDim>()) *
         sizeof(CudaBf16);
}

//: The key and value gradient kernel: one key tile, one value tile, one staged
//: query tile, one staged output-gradient tile, and the three per-row values
//: the staged query rows carry.
template <int kDim>
__host__ __device__ constexpr std::size_t FlashKeyValueGradSharedBytes() {
  return (2 * FlashKvGradElems<kDim>() + 2 * FlashBackwardQueryElems<kDim>()) *
             sizeof(CudaBf16) +
         3 * kFlashBackwardBq * sizeof(float);
}

// --- The micro-tile -------------------------------------------------------

// Packs two fp32 values into one register of two bf16 elements, low half
// first, which is the operand order of `mma.sync`.
__device__ __forceinline__ std::uint32_t PackBf16Pair(float lo, float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&pair);
}

// `d = a * b + d`, one 16 by 8 by 16 micro-tile with bf16 operands and an fp32
// accumulator. `a` is the 16 by 16 row-major A operand in four registers, `b`
// the 16 by 8 B operand in two, and `c` the 16 by 8 accumulator in four.
// The fragment ownership is the documented layout of the instruction:
//
//   a[0] = A[g][2t], A[g][2t+1]        a[1] = A[g+8][2t], A[g+8][2t+1]
//   a[2] = A[g][2t+8], A[g][2t+9]      a[3] = A[g+8][2t+8], A[g+8][2t+9]
//   b[0] = B[2t][g], B[2t+1][g]        b[1] = B[2t+8][g], B[2t+9][g]
//   c[0..1] = C[g][2t], C[g][2t+1]     c[2..3] = C[g+8][2t], C[g+8][2t+1]
//
// with `g = lane / 4` and `t = lane % 4`. Every loader below fills exactly one
// of those shapes.
__device__ __forceinline__ void MmaBf16(const std::uint32_t (&a)[4],
                                        const std::uint32_t* b, float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// --- The shared-memory operand loads --------------------------------------
//
// Every operand leaves the shared tile through `ldmatrix`, which is the load
// the reference kernel uses (the `SmemCopyAtom` of flash-attention's
// `kernel_traits.h`). One instruction fills the four registers of an A operand,
// or the two registers each of two B operands, where a scalar loader needs one
// instruction per register and a merge per pair. The staged rows are padded to
// a multiple of sixteen bytes, which is the only alignment `ldmatrix` asks for,
// and the padding of eight elements also puts the eight rows of one matrix on
// distinct banks, so no swizzle is needed.
//
// `ldmatrix` reads one sixteen-byte segment per lane, so lane `l` supplies the
// address of the segment its own matrix needs. Matrix `l / 8` of the four
// starts `(l / 8) % 2 * 8 + l % 8` rows down the tile and `(l / 16) * 8`
// columns along it.

// The address of a shared-memory segment, in the window `ldmatrix` reads.
__device__ __forceinline__ std::uint32_t SegmentAddress(const CudaBf16* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

// The four-register A operand of a 16 by 16 row-major tile whose `k` dimension
// is contiguous: registers 0 and 1 are rows 0-7 and rows 8-15 of columns 0-7,
// registers 2 and 3 the same two row groups of columns 8-15. `base` points at
// row 0, column 0 of the tile.
__device__ __forceinline__ void LoadRowMajorA(std::uint32_t (&a)[4],
                                              const CudaBf16* base,
                                              int stride) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int row = ((matrix & 1) << 3) + (lane & 7);
  const int col = (matrix >> 1) << 3;
  const std::uint32_t address = SegmentAddress(base + row * stride + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(address));
}

// The B operands of sixteen keys, one pair of registers for each 8-key n-tile,
// from a staged row-major tile whose `k` dimension is contiguous. Registers 0
// and 1 are the low and the high half of `k` for keys 0-7, registers 2 and 3
// the same halves for keys 8-15. `base` points at key 0, column 0.
__device__ __forceinline__ void LoadRowB(std::uint32_t (&b)[4],
                                         const CudaBf16* base, int stride) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int row = ((matrix >> 1) << 3) + (lane & 7);
  const int col = (matrix & 1) << 3;
  const std::uint32_t address = SegmentAddress(base + row * stride + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
      : "r"(address));
}

// The same two B operands when the `k` dimension of the product is the staged
// row and the `n` dimension is contiguous. `ldmatrix.trans` transposes each
// eight-row segment as it loads, so one instruction replaces the four half-word
// reads and two merges a scalar loader needs for every fragment. Registers 0
// and 1 are the `n`-columns 0-7 for `k` rows 0-7 and 8-15, registers 2 and 3
// the `n`-columns 8-15 for the same rows.
__device__ __forceinline__ void LoadStridedB(std::uint32_t (&b)[4],
                                             const CudaBf16* base, int stride) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int row = ((matrix & 1) << 3) + (lane & 7);
  const int col = (matrix >> 1) << 3;
  const std::uint32_t address = SegmentAddress(base + row * stride + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
      : "r"(address));
}

// --- Staging --------------------------------------------------------------

// Copies one key and one value tile into stage `stage`. Rows at or past
// `kv_len` are zeroed with plain stores, because an asynchronous copy cannot
// write a constant.
template <int kDim>
__device__ __forceinline__ void StageKeyValue(const AttentionParams& params,
                                              int tile, int stage, int b,
                                              int kvh, const ComputeType* k,
                                              const ComputeType* v,
                                              CudaBf16* ks, CudaBf16* vs) {
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kChunks = kDim / kFlashChunkElems;
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = cuda_kernels::KvHeadCountDev(params);

  CudaBf16* kdst = ks + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();
  CudaBf16* vdst = vs + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();

  for (int idx = threadIdx.x; idx < kFlashBc * kChunks; idx += kFlashThreads) {
    const int row = idx / kChunks;
    const int chunk = idx - row * kChunks;
    const int j = tile * kFlashBc + row;
    CudaBf16* kout = kdst + row * kStride + chunk * kFlashChunkElems;
    CudaBf16* vout = vdst + row * kStride + chunk * kFlashChunkElems;
    if (j < kv_len) {
      const long long offset =
          ((static_cast<long long>(b) * kv_len + j) * kv_heads + kvh) * kDim +
          chunk * kFlashChunkElems;
      __pipeline_memcpy_async(kout, reinterpret_cast<const uint4*>(k + offset),
                              sizeof(uint4));
      __pipeline_memcpy_async(vout, reinterpret_cast<const uint4*>(v + offset),
                              sizeof(uint4));
    } else {
      *reinterpret_cast<uint4*>(kout) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(vout) = make_uint4(0, 0, 0, 0);
    }
  }
}

// Copies the query rows of this block into a resident or staged tile, in the
// layout `q_head_major` selects. An invalid row is zeroed so its products stay
// finite; the epilogue never writes it.
template <int kDim>
__device__ __forceinline__ void StageQueryRows(const AttentionParams& params,
                                               int tq0, int rows, int b, int h,
                                               const ComputeType* q,
                                               CudaBf16* qs) {
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kChunks = kDim / kFlashChunkElems;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const bool head_major = params.q_head_major;

  for (int idx = threadIdx.x; idx < rows * kChunks; idx += kFlashThreads) {
    const int row = idx / kChunks;
    const int chunk = idx - row * kChunks;
    const int tq = tq0 + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (tq < seq) {
      const long long offset =
          head_major
              ? ((static_cast<long long>(b) * heads + h) * seq + tq) * kDim +
                    chunk * kFlashChunkElems
              : ((static_cast<long long>(b) * seq + tq) * heads + h) * kDim +
                    chunk * kFlashChunkElems;
      value = *reinterpret_cast<const uint4*>(q + offset);
    }
    *reinterpret_cast<uint4*>(qs + row * kStride + chunk * kFlashChunkElems) =
        value;
  }
}

// Copies `rows` output-gradient rows of one `(batch, head)` pair into a staged
// tile. The output gradient always keeps the model stream layout.
template <int kDim>
__device__ __forceinline__ void StageOutputGradRows(
    const AttentionParams& params, int tq0, int rows, int b, int h,
    const ComputeType* dout, CudaBf16* ds) {
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kChunks = kDim / kFlashChunkElems;
  const int seq = params.seq;
  const int heads = params.num_heads;

  for (int idx = threadIdx.x; idx < rows * kChunks; idx += kFlashThreads) {
    const int row = idx / kChunks;
    const int chunk = idx - row * kChunks;
    const int tq = tq0 + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (tq < seq) {
      const long long offset =
          ((static_cast<long long>(b) * seq + tq) * heads + h) * kDim +
          chunk * kFlashChunkElems;
      value = *reinterpret_cast<const uint4*>(dout + offset);
    }
    *reinterpret_cast<uint4*>(ds + row * kStride + chunk * kFlashChunkElems) =
        value;
  }
}

// True when every key of `[j_lo, j_hi]` is visible to the query row at `qpos`,
// so the tile loop can skip the per-element mask. A width that leaves the key
// range, a causal bound, and either window bound are all checked here.
__device__ __forceinline__ bool TileFullyVisible(const AttentionParams& params,
                                                 long long qpos, int j_lo,
                                                 int j_hi, int kv_len) {
  if (j_lo < 0 || j_hi >= kv_len) return false;
  if (params.causal && j_hi > qpos) return false;
  if (params.window_left >= 0 && qpos - j_lo > params.window_left) return false;
  if (params.window_right >= 0 && j_hi - qpos > params.window_right) {
    return false;
  }
  return true;
}

// The two statistics of one query row, in the layout `AttentionStatsCount`
// states: `[batch, num_heads, seq, 2]`, with `stats_row_stride` rows per head
// when the caller processes a sub-block.
__device__ __forceinline__ std::size_t StatsBase(const AttentionParams& params,
                                                 int stats_stride, int b, int h,
                                                 int tq) {
  return (static_cast<std::size_t>(b) * params.num_heads + h) * stats_stride +
         tq;
}

// --- The forward tile ------------------------------------------------------

// One thread block owns `kFlashBr` query rows of one `(batch, head)` pair. Warp
// `w` owns query rows `w * 16 .. w * 16 + 15`; inside the warp, lane `l` keeps
// the online state of rows `l / 4` and `l / 4 + 8`, and the four lanes of a
// quad share that state through a butterfly shuffle.
template <int kDim>
__global__ void __launch_bounds__(kFlashThreads, kFlashMinBlocks)
    FlashAttentionForwardKernel(const AttentionParams params,
                                const ComputeType* __restrict__ q,
                                const ComputeType* __restrict__ k,
                                const ComputeType* __restrict__ v,
                                ComputeType* __restrict__ out,
                                float* __restrict__ stats) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kDimTiles = kDim / 8;

  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const float scale =
      params.scale > 0.0f ? params.scale : rsqrtf(static_cast<float>(kDim));
  const float scale_log2 = scale * kFlashLog2e;

  const int t0 = blockIdx.x * kFlashBr;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int kvh = cuda_kernels::KvHeadDev(h, heads, params.num_kv_heads);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int quad_row = lane >> 2;
  const int quad_col = lane & 3;
  const int row_base = warp * kFlashRowsPerWarp;
  const int col = 2 * quad_col;

  // The query stream position of the first and the last row of this block. A
  // decode step has `seq < kv_len`; the offset keeps the mask in model
  // positions instead of local ones.
  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kFlashBr - 1;

  // The visible keys of the block are one contiguous range: the left window
  // bounds it from below, and causality or the right window bounds it from
  // above. The block therefore skips a run of leading tiles and a run of
  // trailing tiles, and it tests nothing inside the range.
  const int num_key_tiles = (kv_len + kFlashBc - 1) / kFlashBc;
  int first_key = 0;
  if (params.window_left >= 0) first_key = qpos_min - params.window_left;
  const int first_tile = first_key > 0 ? first_key / kFlashBc : 0;
  int last_key = kv_len - 1;
  if (params.causal && qpos_max < last_key) last_key = qpos_max;
  if (params.window_right >= 0 && qpos_max + params.window_right < last_key) {
    last_key = qpos_max + params.window_right;
  }
  int last_tile = last_key < 0 ? -1 : last_key / kFlashBc;
  if (last_tile > num_key_tiles - 1) last_tile = num_key_tiles - 1;

  extern __shared__ __align__(16) char tile_smem[];
  CudaBf16* qs = reinterpret_cast<CudaBf16*>(tile_smem);
  CudaBf16* ks = qs + FlashQueryElems<kDim>();
  CudaBf16* vs = ks + 2 * FlashKvElems<kDim>();

  // The online-softmax state of the two rows this lane holds, and the output
  // accumulator of the sixteen rows the warp owns.
  float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float row_sum[2] = {0.0f, 0.0f};
  float oacc[kDimTiles][4];
#pragma unroll
  for (int t = 0; t < kDimTiles; ++t) {
    oacc[t][0] = 0.0f;
    oacc[t][1] = 0.0f;
    oacc[t][2] = 0.0f;
    oacc[t][3] = 0.0f;
  }

  StageQueryRows<kDim>(params, t0, kFlashBr, b, h, q, qs);
  __syncthreads();

  const int visible = last_tile - first_tile + 1;

  // Two stages alternate. Iteration `i` stages tile `i + 1` before it waits for
  // tile `i`, so the copy overlaps the arithmetic. Tile `i + 1` takes the stage
  // that tile `i - 1` released, so a stage is never refilled while a warp still
  // reads it.
  if (visible > 0) {
    StageKeyValue<kDim>(params, first_tile, 0, b, kvh, k, v, ks, vs);
    __pipeline_commit();
  }

  for (int i = 0; i < visible; ++i) {
    if (i + 1 < visible) {
      StageKeyValue<kDim>(params, first_tile + i + 1, (i + 1) & 1, b, kvh, k, v,
                          ks, vs);
      __pipeline_commit();
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();

    const int stage = i & 1;
    const int tile = first_tile + i;
    const int j_base = tile * kFlashBc;
    const CudaBf16* k_tile =
        ks + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();
    const CudaBf16* v_tile =
        vs + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();

    // S = Q K^T, as eight 16 by 8 sub-tiles of eight keys each, unscaled. The
    // scale folds into the exponential below, where it costs one fused
    // multiply-add per score instead of a separate product here.
    float sacc[kFlashKeyTiles][4];
#pragma unroll
    for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
      sacc[nt][0] = 0.0f;
      sacc[nt][1] = 0.0f;
      sacc[nt][2] = 0.0f;
      sacc[nt][3] = 0.0f;
    }
    for (int d = 0; d < kDim; d += 16) {
      std::uint32_t a[4];
      LoadRowMajorA(a, qs + row_base * kStride + d, kStride);
#pragma unroll
      for (int nt = 0; nt < kFlashKeyTiles; nt += 2) {
        std::uint32_t bf[4];
        LoadRowB(bf, k_tile + nt * 8 * kStride + d, kStride);
        MmaBf16(a, bf, sacc[nt]);
        MmaBf16(a, bf + 2, sacc[nt + 1]);
      }
    }

    // The mask, the row maximum, and P = exp2(scale_log2 * S - max). Lane `l`
    // owns rows `l / 4` and `l / 4 + 8`, and the key columns `2 * (l % 4)` and
    // `2 * (l % 4) + 1` of each eight-key sub-tile.
    const int tq0 = t0 + row_base;
    const long long qpos[2] = {kv_len - seq + tq0 + quad_row,
                               kv_len - seq + tq0 + quad_row + 8};
    const bool full = TileFullyVisible(params, qpos[0], j_base,
                                       j_base + kFlashBc - 1, kv_len) &&
                      TileFullyVisible(params, qpos[1], j_base,
                                       j_base + kFlashBc - 1, kv_len);

    float local_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int j = j_base + nt * 8 + col + e;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          // The mask writes the sentinel into the score instead of a branch in
          // the probability pass: its exponent is then zero, and `exp2(-inf)`
          // needs no guard.
          float s = sacc[nt][2 * r + e];
          if (!full && (j >= kv_len ||
                        !cuda_kernels::KeyAllowedDev(params, qpos[r], j))) {
            s = -CUDART_INF_F;
          }
          sacc[nt][2 * r + e] = s;
          local_max[r] = fmaxf(local_max[r], s);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      local_max[r] =
          fmaxf(local_max[r], __shfl_xor_sync(0xffffffffu, local_max[r], 1));
      local_max[r] =
          fmaxf(local_max[r], __shfl_xor_sync(0xffffffffu, local_max[r], 2));
    }

    // The running maximum is in base-two units, so the rescale factor of the
    // output accumulator is one exponential of the maximum difference.
    float alpha[2] = {1.0f, 1.0f};
    float max_exp[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      // A row whose whole key tile is hidden keeps the step out of the running
      // maximum, so the two sentinels never meet in a subtraction.
      if (local_max[r] == -CUDART_INF_F) continue;
      const float previous = row_max[r];
      const float updated = fmaxf(previous, local_max[r] * scale_log2);
      alpha[r] = (previous == -CUDART_INF_F) ? 0.0f : exp2f(previous - updated);
      row_max[r] = updated;
      max_exp[r] = updated;
    }

    float local_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float p0 = exp2f(sacc[nt][2 * r] * scale_log2 - max_exp[r]);
        const float p1 = exp2f(sacc[nt][2 * r + 1] * scale_log2 - max_exp[r]);
        sacc[nt][2 * r] = p0;
        sacc[nt][2 * r + 1] = p1;
        local_sum[r] += p0 + p1;
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      local_sum[r] += __shfl_xor_sync(0xffffffffu, local_sum[r], 1);
      local_sum[r] += __shfl_xor_sync(0xffffffffu, local_sum[r], 2);
      row_sum[r] = alpha[r] * row_sum[r] + local_sum[r];
    }

    // Rescale the output accumulator. The factor of a row is the same in all
    // four lanes of its quad, so no shared tile carries it.
#pragma unroll
    for (int t = 0; t < kDimTiles; ++t) {
      oacc[t][0] *= alpha[0];
      oacc[t][1] *= alpha[0];
      oacc[t][2] *= alpha[1];
      oacc[t][3] *= alpha[1];
    }

    // Round P to bf16 in registers. Two adjacent 16 by 8 accumulators are the
    // 16 by 16 A operand of the value product, so the probability operand
    // never leaves the register file.
    std::uint32_t pa[kFlashChunks][4];
#pragma unroll
    for (int c = 0; c < kFlashChunks; ++c) {
      pa[c][0] = PackBf16Pair(sacc[2 * c][0], sacc[2 * c][1]);
      pa[c][1] = PackBf16Pair(sacc[2 * c][2], sacc[2 * c][3]);
      pa[c][2] = PackBf16Pair(sacc[2 * c + 1][0], sacc[2 * c + 1][1]);
      pa[c][3] = PackBf16Pair(sacc[2 * c + 1][2], sacc[2 * c + 1][3]);
    }

    // O += P V, one probability chunk of sixteen keys in the A operand and
    // every pair of eight-dimension sub-tiles in the B operand.
#pragma unroll
    for (int c = 0; c < kFlashChunks; ++c) {
      const CudaBf16* vbase = v_tile + c * kFlashKeysPerChunk * kStride;
#pragma unroll
      for (int t = 0; t < kDimTiles; t += 2) {
        std::uint32_t bf[4];
        LoadStridedB(bf, vbase + t * 8, kStride);
        MmaBf16(pa[c], bf, oacc[t]);
        MmaBf16(pa[c], bf + 2, oacc[t + 1]);
      }
    }

    __syncthreads();
  }

  // The epilogue: an output row is scaled by `1 / row_sum`, and the lane whose
  // first column it owns publishes the row maximum and the row sum.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float inverse = row_sum[r] > 0.0f ? 1.0f / row_sum[r] : 0.0f;
    const int tq = t0 + row_base + quad_row + r * 8;
    if (tq >= seq) continue;
    const std::size_t base =
        (static_cast<std::size_t>(b) * seq + tq) * heads * kDim +
        static_cast<std::size_t>(h) * kDim;
#pragma unroll
    for (int t = 0; t < kDimTiles; ++t) {
      out[base + t * 8 + col] =
          cuda_kernels::ToComputeDev(oacc[t][2 * r] * inverse);
      out[base + t * 8 + col + 1] =
          cuda_kernels::ToComputeDev(oacc[t][2 * r + 1] * inverse);
    }
    if (quad_col != 0) continue;
    const std::size_t stat_base = (StatsBase(params, seq, b, h, tq)) * 2;
    const bool empty = row_sum[r] <= 0.0f;
    // The running maximum is base two; the saved statistic is natural units.
    stats[stat_base] = empty ? 0.0f : row_max[r] * kFlashLn2;
    stats[stat_base + 1] = empty ? 0.0f : row_sum[r];
  }
#endif
}

// --- The query gradient ----------------------------------------------------

// One thread block owns `kFlashBr` query rows of one `(batch, head)` pair and
// writes their gradient once. It loops over the visible key tiles twice,
// because the row dot product that the score gradient needs is only complete
// after the last key tile:
//
//   pass 0:  S = scale Q K^T,  dP = dO V^T,  row_dot += sum(P * dP)
//   pass 1:  S = scale Q K^T,  dP = dO V^T,  dQ += scale * (P * (dP - row_dot))
//   K
//
// The first pass writes `row_dot` to the shared scratch the key and value
// gradient kernel reads. `docs/attention.md` Part 7, section 4.4 counts the
// issued products of both kernels.
template <int kDim>
__global__ void __launch_bounds__(kFlashThreads, kFlashMinBlocks)
    FlashAttentionQueryGradKernel(
        const AttentionParams params, const ComputeType* __restrict__ q,
        const ComputeType* __restrict__ k, const ComputeType* __restrict__ v,
        const float* __restrict__ stats, const ComputeType* __restrict__ dout,
        ComputeType* __restrict__ dq, float* __restrict__ row_dot_out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kDimTiles = kDim / 8;

  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const int stats_stride =
      params.stats_row_stride > 0 ? params.stats_row_stride : seq;
  const float scale =
      params.scale > 0.0f ? params.scale : rsqrtf(static_cast<float>(kDim));
  const float scale_log2 = scale * kFlashLog2e;

  const int t0 = blockIdx.x * kFlashBr;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int kvh = cuda_kernels::KvHeadDev(h, heads, params.num_kv_heads);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int quad_row = lane >> 2;
  const int quad_col = lane & 3;
  const int row_base = warp * kFlashRowsPerWarp;
  const int col = 2 * quad_col;

  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kFlashBr - 1;

  const int num_key_tiles = (kv_len + kFlashBc - 1) / kFlashBc;
  int first_key = 0;
  if (params.window_left >= 0) first_key = qpos_min - params.window_left;
  const int first_tile = first_key > 0 ? first_key / kFlashBc : 0;
  int last_key = kv_len - 1;
  if (params.causal && qpos_max < last_key) last_key = qpos_max;
  if (params.window_right >= 0 && qpos_max + params.window_right < last_key) {
    last_key = qpos_max + params.window_right;
  }
  int last_tile = last_key < 0 ? -1 : last_key / kFlashBc;
  if (last_tile > num_key_tiles - 1) last_tile = num_key_tiles - 1;

  extern __shared__ __align__(16) char grad_smem[];
  CudaBf16* qs = reinterpret_cast<CudaBf16*>(grad_smem);
  CudaBf16* dos = qs + FlashQueryElems<kDim>();
  CudaBf16* ks = dos + FlashQueryElems<kDim>();
  CudaBf16* vs = ks + 2 * FlashKvElems<kDim>();

  const int tq0 = t0 + row_base;
  const long long qpos[2] = {kv_len - seq + tq0 + quad_row,
                             kv_len - seq + tq0 + quad_row + 8};

  // The saved statistics of the two rows this lane holds.
  float row_max[2] = {0.0f, 0.0f};
  float row_inv[2] = {0.0f, 0.0f};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int tq = tq0 + quad_row + r * 8;
    if (tq >= seq) continue;
    const std::size_t stat_base = StatsBase(params, stats_stride, b, h, tq) * 2;
    // The saved maximum is in natural units; the exponential is base two.
    row_max[r] = stats[stat_base] * kFlashLog2e;
    const float sum_exp = stats[stat_base + 1];
    row_inv[r] = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
  }

  float row_dot[2] = {0.0f, 0.0f};
  float dqacc[kDimTiles][4];
#pragma unroll
  for (int t = 0; t < kDimTiles; ++t) {
    dqacc[t][0] = 0.0f;
    dqacc[t][1] = 0.0f;
    dqacc[t][2] = 0.0f;
    dqacc[t][3] = 0.0f;
  }

  StageQueryRows<kDim>(params, t0, kFlashBr, b, h, q, qs);
  StageOutputGradRows<kDim>(params, t0, kFlashBr, b, h, dout, dos);
  __syncthreads();

  const int visible = last_tile - first_tile + 1;

  for (int pass = 0; pass < 2; ++pass) {
    float dot_acc[2] = {0.0f, 0.0f};

    if (visible > 0) {
      StageKeyValue<kDim>(params, first_tile, 0, b, kvh, k, v, ks, vs);
      __pipeline_commit();
    }

    for (int i = 0; i < visible; ++i) {
      if (i + 1 < visible) {
        StageKeyValue<kDim>(params, first_tile + i + 1, (i + 1) & 1, b, kvh, k,
                            v, ks, vs);
        __pipeline_commit();
        __pipeline_wait_prior(1);
      } else {
        __pipeline_wait_prior(0);
      }
      __syncthreads();

      const int stage = i & 1;
      const int tile = first_tile + i;
      const int j_base = tile * kFlashBc;
      const CudaBf16* k_tile =
          ks + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();
      const CudaBf16* v_tile =
          vs + static_cast<std::size_t>(stage) * FlashKvElems<kDim>();

      // S = scale * Q K^T and dP = dO V^T, both over (query, key).
      float sacc[kFlashKeyTiles][4];
      float dpacc[kFlashKeyTiles][4];
#pragma unroll
      for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
        sacc[nt][0] = 0.0f;
        sacc[nt][1] = 0.0f;
        sacc[nt][2] = 0.0f;
        sacc[nt][3] = 0.0f;
        dpacc[nt][0] = 0.0f;
        dpacc[nt][1] = 0.0f;
        dpacc[nt][2] = 0.0f;
        dpacc[nt][3] = 0.0f;
      }
      for (int d = 0; d < kDim; d += 16) {
        std::uint32_t aq[4];
        std::uint32_t ad[4];
        LoadRowMajorA(aq, qs + row_base * kStride + d, kStride);
        LoadRowMajorA(ad, dos + row_base * kStride + d, kStride);
#pragma unroll
        for (int nt = 0; nt < kFlashKeyTiles; nt += 2) {
          std::uint32_t bk[4];
          std::uint32_t bv[4];
          LoadRowB(bk, k_tile + nt * 8 * kStride + d, kStride);
          LoadRowB(bv, v_tile + nt * 8 * kStride + d, kStride);
          MmaBf16(aq, bk, sacc[nt]);
          MmaBf16(ad, bv, dpacc[nt]);
          MmaBf16(aq, bk + 2, sacc[nt + 1]);
          MmaBf16(ad, bv + 2, dpacc[nt + 1]);
        }
      }

      // P = exp2(scale_log2 * S - row_max) / row_sum. The mask writes the
      // sentinel into the score, so the pass below is one fused multiply-add,
      // one exponential, and one product per element with no branch; a row
      // whose sum is zero contributes nothing.
      const bool full = TileFullyVisible(params, qpos[0], j_base,
                                         j_base + kFlashBc - 1, kv_len) &&
                        TileFullyVisible(params, qpos[1], j_base,
                                         j_base + kFlashBc - 1, kv_len);
#pragma unroll
      for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int j = j_base + nt * 8 + col + e;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            if (!full && (j >= kv_len ||
                          !cuda_kernels::KeyAllowedDev(params, qpos[r], j))) {
              sacc[nt][2 * r + e] = -CUDART_INF_F;
            }
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float p0 =
              exp2f(sacc[nt][2 * r] * scale_log2 - row_max[r]) * row_inv[r];
          const float p1 =
              exp2f(sacc[nt][2 * r + 1] * scale_log2 - row_max[r]) * row_inv[r];
          sacc[nt][2 * r] = p0;
          sacc[nt][2 * r + 1] = p1;
        }
      }

      if (pass == 0) {
        // The per-row dot product that the score gradient subtracts.
#pragma unroll
        for (int nt = 0; nt < kFlashKeyTiles; ++nt) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            dot_acc[r] += sacc[nt][2 * r] * dpacc[nt][2 * r] +
                          sacc[nt][2 * r + 1] * dpacc[nt][2 * r + 1];
          }
        }
        __syncthreads();
        continue;
      }

      // dS = P * (dP - row_dot), rounded to bf16 as the A operand of the query
      // gradient product.
      std::uint32_t dsa[kFlashChunks][4];
#pragma unroll
      for (int c = 0; c < kFlashChunks; ++c) {
        dsa[c][0] =
            PackBf16Pair(sacc[2 * c][0] * (dpacc[2 * c][0] - row_dot[0]),
                         sacc[2 * c][1] * (dpacc[2 * c][1] - row_dot[0]));
        dsa[c][1] =
            PackBf16Pair(sacc[2 * c][2] * (dpacc[2 * c][2] - row_dot[1]),
                         sacc[2 * c][3] * (dpacc[2 * c][3] - row_dot[1]));
        dsa[c][2] = PackBf16Pair(
            sacc[2 * c + 1][0] * (dpacc[2 * c + 1][0] - row_dot[0]),
            sacc[2 * c + 1][1] * (dpacc[2 * c + 1][1] - row_dot[0]));
        dsa[c][3] = PackBf16Pair(
            sacc[2 * c + 1][2] * (dpacc[2 * c + 1][2] - row_dot[1]),
            sacc[2 * c + 1][3] * (dpacc[2 * c + 1][3] - row_dot[1]));
      }

      // dQ += dS K, one key chunk of sixteen keys at a time and every pair of
      // eight-dimension sub-tiles.
#pragma unroll
      for (int c = 0; c < kFlashChunks; ++c) {
        const CudaBf16* kbase = k_tile + c * kFlashKeysPerChunk * kStride;
#pragma unroll
        for (int t = 0; t < kDimTiles; t += 2) {
          std::uint32_t bk[4];
          LoadStridedB(bk, kbase + t * 8, kStride);
          MmaBf16(dsa[c], bk, dqacc[t]);
          MmaBf16(dsa[c], bk + 2, dqacc[t + 1]);
        }
      }

      __syncthreads();
    }

    if (pass == 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        dot_acc[r] += __shfl_xor_sync(0xffffffffu, dot_acc[r], 1);
        dot_acc[r] += __shfl_xor_sync(0xffffffffu, dot_acc[r], 2);
        row_dot[r] = dot_acc[r];
        const int tq = tq0 + quad_row + r * 8;
        if (quad_col != 0 || tq >= seq) continue;
        row_dot_out[StatsBase(params, stats_stride, b, h, tq)] = row_dot[r];
      }
    }
  }

  // dQ = scale * sum_k dS K, written once per query row in the layout the
  // caller's query uses.
  const bool head_major = params.q_head_major;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int tq = tq0 + quad_row + r * 8;
    if (tq >= seq) continue;
    const std::size_t base =
        head_major
            ? ((static_cast<std::size_t>(b) * heads + h) * seq + tq) * kDim
            : (static_cast<std::size_t>(b) * seq + tq) * heads * kDim +
                  static_cast<std::size_t>(h) * kDim;
#pragma unroll
    for (int t = 0; t < kDimTiles; ++t) {
      dq[base + t * 8 + col] =
          cuda_kernels::ToComputeDev(scale * dqacc[t][2 * r]);
      dq[base + t * 8 + col + 1] =
          cuda_kernels::ToComputeDev(scale * dqacc[t][2 * r + 1]);
    }
  }
#endif
}

// --- The key and value gradient --------------------------------------------

// One thread block owns `kFlashKvGradBc` key rows of one `(batch, key/value
// head)` pair and writes `dk` and `dv` once per key row. Warp `w` owns key
// rows `w * 16 .. w * 16 + 15`, so every one of the eight warps owns work, and
// the accumulator of each direction is the transpose of the loop: rows are
// keys, columns are the head dimension.
//
// The loop runs over the query tiles whose window contains this key block,
// and inside it over the query heads that read this key/value head. The two
// heads of a group accumulate into the same fp32 registers, so grouped-query
// attention needs no fold kernel.
template <int kDim>
__global__ void __launch_bounds__(kFlashThreads, kFlashMinBlocks)
    FlashAttentionKeyValueGradKernel(
        const AttentionParams params, const ComputeType* __restrict__ q,
        const ComputeType* __restrict__ k, const ComputeType* __restrict__ v,
        const float* __restrict__ stats, const ComputeType* __restrict__ dout,
        const float* __restrict__ row_dot_in, ComputeType* __restrict__ dk,
        ComputeType* __restrict__ dv) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kStride = FlashStride<kDim>();
  constexpr int kDimTiles = kDim / 8;
  constexpr int kQueryTiles = kFlashBackwardBq / 8;
  constexpr int kQueryChunks = kFlashBackwardBq / kFlashKeysPerChunk;

  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = cuda_kernels::KvHeadCountDev(params);
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const int stats_stride =
      params.stats_row_stride > 0 ? params.stats_row_stride : seq;
  const float scale =
      params.scale > 0.0f ? params.scale : rsqrtf(static_cast<float>(kDim));
  const float scale_log2 = scale * kFlashLog2e;

  const int k_lo = blockIdx.x * kFlashKvGradBc;
  const int kvh = blockIdx.y;
  const int b = blockIdx.z;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int quad_row = lane >> 2;
  const int quad_col = lane & 3;
  const int key_base = warp * kFlashRowsPerWarp;
  const int col = 2 * quad_col;

  // The query rows whose window meets this key block. Causality bounds them
  // from below through the first key, the left window from above, and the
  // right window from below again.
  const int k_hi = k_lo + kFlashKvGradBc - 1;
  int qpos_lo = k_lo;
  if (!params.causal && params.window_right >= 0) {
    qpos_lo = k_lo - params.window_right;
  }
  long long qpos_hi = params.window_left >= 0
                          ? static_cast<long long>(k_hi) + params.window_left
                          : kv_len - 1;
  if (qpos_hi > kv_len - 1) qpos_hi = kv_len - 1;
  const int offset = kv_len - seq;
  int row_lo = qpos_lo - offset;
  int row_hi = static_cast<int>(qpos_hi) - offset;
  if (row_lo < 0) row_lo = 0;
  if (row_hi > seq - 1) row_hi = seq - 1;

  extern __shared__ __align__(16) char kv_smem[];
  CudaBf16* ks = reinterpret_cast<CudaBf16*>(kv_smem);
  CudaBf16* vs = ks + FlashKvGradElems<kDim>();
  CudaBf16* qs = vs + FlashKvGradElems<kDim>();
  CudaBf16* dos = qs + FlashBackwardQueryElems<kDim>();
  float* row_max_stage =
      reinterpret_cast<float*>(dos + FlashBackwardQueryElems<kDim>());
  float* row_inv_stage = row_max_stage + kFlashBackwardBq;
  float* row_dot_stage = row_inv_stage + kFlashBackwardBq;

  float dkacc[kDimTiles][4];
  float dvacc[kDimTiles][4];
#pragma unroll
  for (int t = 0; t < kDimTiles; ++t) {
    dkacc[t][0] = 0.0f;
    dkacc[t][1] = 0.0f;
    dkacc[t][2] = 0.0f;
    dkacc[t][3] = 0.0f;
    dvacc[t][0] = 0.0f;
    dvacc[t][1] = 0.0f;
    dvacc[t][2] = 0.0f;
    dvacc[t][3] = 0.0f;
  }

  // The key and value rows of this block, staged once.
  {
    constexpr int kChunks = kDim / kFlashChunkElems;
    for (int idx = tid; idx < kFlashKvGradBc * kChunks; idx += kFlashThreads) {
      const int row = idx / kChunks;
      const int chunk = idx - row * kChunks;
      const int j = k_lo + row;
      uint4 kvalue = make_uint4(0, 0, 0, 0);
      uint4 vvalue = make_uint4(0, 0, 0, 0);
      if (j < kv_len) {
        const long long offset_j =
            ((static_cast<long long>(b) * kv_len + j) * kv_heads + kvh) * kDim +
            chunk * kFlashChunkElems;
        kvalue = *reinterpret_cast<const uint4*>(k + offset_j);
        vvalue = *reinterpret_cast<const uint4*>(v + offset_j);
      }
      *reinterpret_cast<uint4*>(ks + row * kStride + chunk * kFlashChunkElems) =
          kvalue;
      *reinterpret_cast<uint4*>(vs + row * kStride + chunk * kFlashChunkElems) =
          vvalue;
    }
  }

  const int group = heads / kv_heads > 0 ? heads / kv_heads : 1;
  const int head_first = kvh * group;
  const int head_last = kvh == kv_heads - 1 ? heads : (kvh + 1) * group;

  const long long key_pos[2] = {k_lo + key_base + quad_row,
                                k_lo + key_base + quad_row + 8};

  for (int tq0 = row_lo >= 0 ? (row_lo / kFlashBackwardBq) * kFlashBackwardBq
                             : 0;
       tq0 <= row_hi; tq0 += kFlashBackwardBq) {
    for (int h = head_first; h < head_last; ++h) {
      StageQueryRows<kDim>(params, tq0, kFlashBackwardBq, b, h, q, qs);
      StageOutputGradRows<kDim>(params, tq0, kFlashBackwardBq, b, h, dout, dos);
      for (int idx = tid; idx < kFlashBackwardBq; idx += kFlashThreads) {
        const int tq = tq0 + idx;
        float m = 0.0f;
        float inv = 0.0f;
        float dot = 0.0f;
        if (tq < seq) {
          const std::size_t stat_base =
              StatsBase(params, stats_stride, b, h, tq) * 2;
          m = stats[stat_base] * kFlashLog2e;
          const float sum_exp = stats[stat_base + 1];
          inv = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
          dot = row_dot_in[StatsBase(params, stats_stride, b, h, tq)];
        }
        row_max_stage[idx] = m;
        row_inv_stage[idx] = inv;
        row_dot_stage[idx] = dot;
      }
      __syncthreads();

      // S^T = scale * K Q^T, one 16 by 8 sub-tile of eight queries at a time.
      float sacc[kQueryTiles][4];
      float dpacc[kQueryTiles][4];
#pragma unroll
      for (int nt = 0; nt < kQueryTiles; ++nt) {
        sacc[nt][0] = 0.0f;
        sacc[nt][1] = 0.0f;
        sacc[nt][2] = 0.0f;
        sacc[nt][3] = 0.0f;
        dpacc[nt][0] = 0.0f;
        dpacc[nt][1] = 0.0f;
        dpacc[nt][2] = 0.0f;
        dpacc[nt][3] = 0.0f;
      }
      for (int d = 0; d < kDim; d += 16) {
        std::uint32_t ak[4];
        std::uint32_t av[4];
        LoadRowMajorA(ak, ks + key_base * kStride + d, kStride);
        LoadRowMajorA(av, vs + key_base * kStride + d, kStride);
#pragma unroll
        for (int nt = 0; nt < kQueryTiles; nt += 2) {
          std::uint32_t bq[4];
          std::uint32_t bd[4];
          LoadRowB(bq, qs + nt * 8 * kStride + d, kStride);
          LoadRowB(bd, dos + nt * 8 * kStride + d, kStride);
          MmaBf16(ak, bq, sacc[nt]);
          MmaBf16(av, bd, dpacc[nt]);
          MmaBf16(ak, bq + 2, sacc[nt + 1]);
          MmaBf16(av, bd + 2, dpacc[nt + 1]);
        }
      }

      // P^T = exp2(scale_log2 * S^T - row_max) / row_sum, and dS^T = P^T *
      // (dP^T - row_dot). Both are per query column, so the row maximum, the
      // inverse sum, and the row dot are read by column. The mask writes the
      // sentinel into the score, so neither pass below needs a branch.
#pragma unroll
      for (int nt = 0; nt < kQueryTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int query = nt * 8 + col + e;
          if (tq0 + query >= seq) {
            sacc[nt][e] = -CUDART_INF_F;
            sacc[nt][2 + e] = -CUDART_INF_F;
            continue;
          }
          const long long qpos_q = offset + tq0 + query;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            if (!cuda_kernels::KeyAllowedDev(params, qpos_q, key_pos[r])) {
              sacc[nt][2 * r + e] = -CUDART_INF_F;
            }
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < kQueryTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int query = nt * 8 + col + e;
          const float max_exp = row_max_stage[query];
          const float inv = row_inv_stage[query];
          const float dot = row_dot_stage[query];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const float p =
                exp2f(sacc[nt][2 * r + e] * scale_log2 - max_exp) * inv;
            sacc[nt][2 * r + e] = p;
            dpacc[nt][2 * r + e] = p * (dpacc[nt][2 * r + e] - dot);
          }
        }
      }

      // Round P^T and dS^T to bf16 in registers: two adjacent 16 by 8
      // accumulators are one 16 by 16 A operand.
      std::uint32_t pt[kQueryChunks][4];
      std::uint32_t dst[kQueryChunks][4];
#pragma unroll
      for (int c = 0; c < kQueryChunks; ++c) {
        pt[c][0] = PackBf16Pair(sacc[2 * c][0], sacc[2 * c][1]);
        pt[c][1] = PackBf16Pair(sacc[2 * c][2], sacc[2 * c][3]);
        pt[c][2] = PackBf16Pair(sacc[2 * c + 1][0], sacc[2 * c + 1][1]);
        pt[c][3] = PackBf16Pair(sacc[2 * c + 1][2], sacc[2 * c + 1][3]);
        dst[c][0] = PackBf16Pair(dpacc[2 * c][0], dpacc[2 * c][1]);
        dst[c][1] = PackBf16Pair(dpacc[2 * c][2], dpacc[2 * c][3]);
        dst[c][2] = PackBf16Pair(dpacc[2 * c + 1][0], dpacc[2 * c + 1][1]);
        dst[c][3] = PackBf16Pair(dpacc[2 * c + 1][2], dpacc[2 * c + 1][3]);
      }

      // dV += P^T dO and dK += dS^T Q, one query chunk of sixteen rows at a
      // time and every pair of eight-dimension sub-tiles.
#pragma unroll
      for (int c = 0; c < kQueryChunks; ++c) {
        const CudaBf16* dbase = dos + c * kFlashKeysPerChunk * kStride;
        const CudaBf16* qbase = qs + c * kFlashKeysPerChunk * kStride;
#pragma unroll
        for (int t = 0; t < kDimTiles; t += 2) {
          std::uint32_t bd[4];
          std::uint32_t bq[4];
          LoadStridedB(bd, dbase + t * 8, kStride);
          LoadStridedB(bq, qbase + t * 8, kStride);
          MmaBf16(pt[c], bd, dvacc[t]);
          MmaBf16(dst[c], bq, dkacc[t]);
          MmaBf16(pt[c], bd + 2, dvacc[t + 1]);
          MmaBf16(dst[c], bq + 2, dkacc[t + 1]);
        }
      }

      __syncthreads();
    }
  }

  // The gradient rows, written once per key row of this key/value head.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int j = k_lo + key_base + quad_row + r * 8;
    if (j >= kv_len) continue;
    const std::size_t base =
        (static_cast<std::size_t>(b) * kv_len + j) * kv_heads * kDim +
        static_cast<std::size_t>(kvh) * kDim;
#pragma unroll
    for (int t = 0; t < kDimTiles; ++t) {
      dk[base + t * 8 + col] =
          cuda_kernels::ToComputeDev(scale * dkacc[t][2 * r]);
      dk[base + t * 8 + col + 1] =
          cuda_kernels::ToComputeDev(scale * dkacc[t][2 * r + 1]);
      dv[base + t * 8 + col] = cuda_kernels::ToComputeDev(dvacc[t][2 * r]);
      dv[base + t * 8 + col + 1] =
          cuda_kernels::ToComputeDev(dvacc[t][2 * r + 1]);
    }
  }
#endif
}

// The per-row dot product of the backward, in the layout the two backward
// kernels read. It is `batch * heads * seq` floats, held in one cached
// allocation per device thread.
float* CachedRowDot(std::size_t count) {
  static thread_local float* buffer = nullptr;
  static thread_local std::size_t capacity = 0;
  if (count > capacity) {
    if (buffer != nullptr) Free(buffer);
    buffer = static_cast<float*>(Alloc(count * sizeof(float)));
    capacity = count;
  }
  return buffer;
}

// Requests the opt-in shared-memory limit for both head widths once, then
// launches the tile. `//tests:cuda_sim_test` fails a launch above 48 KiB
// without the matching `cudaFuncSetAttribute`.
void ConfigureFlashSharedMemory() {
  static const bool configured = [] {
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionForwardKernel<64>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashSharedBytes<64>()));
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionForwardKernel<128>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashSharedBytes<128>()));
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionQueryGradKernel<64>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashQueryGradSharedBytes<64>()));
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionQueryGradKernel<128>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashQueryGradSharedBytes<128>()));
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionKeyValueGradKernel<64>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashKeyValueGradSharedBytes<64>()));
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(FlashAttentionKeyValueGradKernel<128>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FlashKeyValueGradSharedBytes<128>()));
    return true;
  }();
  (void)configured;
}

// The scratch the query-gradient kernel writes and the key and value gradient
// kernel reads. The two kernels run back to back on one stream, so one buffer
// serves both and no host read is needed.
float* RowDotScratch(const AttentionParams& params) {
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.num_heads * params.seq;
  return CachedRowDot(count);
}

}  // namespace

// Whether this device takes the family for this shape. The shape conditions
// are in `attention_dispatch.h`; this adds the two device answers: tensor
// cores, and the opt-in shared-memory budget the query-gradient kernel
// requests. A card whose limit is below that budget keeps the cuBLAS path
// instead of failing the launch, which is what an `sm_86` card would do.
bool UseFusedAttentionFlash(const AttentionParams& p) {
  static const bool ready = [] {
    const Caps caps = GetCaps();
    if (!caps.has_tensor_cores) return false;
    int optin = 0;
    const cudaError_t status = cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, caps.device_index);
    if (status != cudaSuccess) return false;
    return static_cast<std::size_t>(optin) >=
           FlashQueryGradSharedBytes<kFlashMaxDim>();
  }();
  return ready && FusedAttentionFlashCovers(p, /*tensor_cores=*/true);
}

void LaunchFlashAttention(const AttentionParams& params, const ComputeType* q,
                          const ComputeType* k, const ComputeType* v,
                          ComputeType* out, float* stats) {
  ConfigureFlashSharedMemory();
  const int query_tiles = (params.seq + kFlashBr - 1) / kFlashBr;
  const dim3 grid(query_tiles, params.num_heads, params.batch);
  const dim3 block(kFlashThreads);
  if (params.head_dim == 64) {
    cuda_backend::Launch(FlashAttentionForwardKernel<64>, grid, block,
                         FlashSharedBytes<64>(), params, q, k, v, out, stats);
    return;
  }
  cuda_backend::Launch(FlashAttentionForwardKernel<128>, grid, block,
                       FlashSharedBytes<128>(), params, q, k, v, out, stats);
}

void LaunchFlashAttentionBackward(const AttentionParams& params,
                                  const ComputeType* q, const ComputeType* k,
                                  const ComputeType* v, const float* stats,
                                  const ComputeType* dout, ComputeType* dq,
                                  ComputeType* dk, ComputeType* dv) {
  ConfigureFlashSharedMemory();
  float* row_dot = RowDotScratch(params);
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = cuda_kernels::KvHeadCountDev(params);

  const dim3 query_grid((params.seq + kFlashBr - 1) / kFlashBr,
                        params.num_heads, params.batch);
  const dim3 key_grid((kv_len + kFlashKvGradBc - 1) / kFlashKvGradBc, kv_heads,
                      params.batch);
  const dim3 block(kFlashThreads);

  if (params.head_dim == 64) {
    cuda_backend::Launch(FlashAttentionQueryGradKernel<64>, query_grid, block,
                         FlashQueryGradSharedBytes<64>(), params, q, k, v,
                         stats, dout, dq, row_dot);
    cuda_backend::Launch(FlashAttentionKeyValueGradKernel<64>, key_grid, block,
                         FlashKeyValueGradSharedBytes<64>(), params, q, k, v,
                         stats, dout, row_dot, dk, dv);
    return;
  }
  cuda_backend::Launch(FlashAttentionQueryGradKernel<128>, query_grid, block,
                       FlashQueryGradSharedBytes<128>(), params, q, k, v, stats,
                       dout, dq, row_dot);
  cuda_backend::Launch(FlashAttentionKeyValueGradKernel<128>, key_grid, block,
                       FlashKeyValueGradSharedBytes<128>(), params, q, k, v,
                       stats, dout, row_dot, dk, dv);
}

}  // namespace kernels
}  // namespace nanochat
