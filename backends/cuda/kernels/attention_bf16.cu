// The bf16 tensor-core tile for the windowed attention forward. It is the
// third branch of `AttentionForward`, beside the fp32 tile and the cuBLAS
// decomposition, and it runs only in a bf16 build on a device with tensor
// cores at head width 128. See `docs/attention.md` Part 6.
//
// The tile keeps the online-softmax state and the output accumulator in
// registers, skips every key tile that no query row in the block can see, and
// never materialises the `seq x seq` score matrix. That is what turns the
// sliding window into saved work in a bf16 build, where the cuBLAS path runs
// at the tensor-core rate and still pays for every masked pair.
//
// Two microkernels do the arithmetic, both through the `wmma` interface with
// 16 by 16 by 16 fragments of bf16 and fp32 accumulators:
//   * the score product reads the key tile as a column-major operand, which is
//     the layout the model stream already gives it, so no transpose is needed;
//   * the value product reads the probability tile and the value tile as
//     row-major operands, which is again the staged layout.
// The `wmma` accumulator layout is not part of that interface, so the tile
// cannot read the row of one accumulator element. It therefore stages its four
// score fragments into a per-warp fp32 scratch tile and processes one row per
// pair of lanes. `docs/attention.md` Part 6, section 4.4 bounds that traffic.
//
// The arithmetic mirrors `backends/cuda/kernels/testing/sequence_ref.h` and the
// fp32 tile: the row maximum over the visible keys, the causal plus sliding
// window mask through `KeyAllowedDev`, the `(0, 0)` statistics for an empty
// window, and `expf` in fp32. Only the two products round to bf16.

// The tile stages its operands in sixteen-byte copies of two-byte elements, so
// it needs a bf16 or fp8 build. It also needs the CUDA compiler, because the
// `wmma` interface and `cp.async` do not compile on the host pass. The build
// answers both: the build compiles this file only in a bf16 or fp8 build, and
// an fp32 build and the emulator link `attention_bf16_absent.cc` instead.

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/attention_bf16.h"
#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

// --- Geometry -------------------------------------------------------------

//: Query rows per block. Four warps, one warp per sixteen rows.
constexpr int kBf16TileBr = 64;
//: Key and value rows per key tile.
constexpr int kBf16TileBc = 64;
//: The only head width the tile covers (`docs/attention.md` Part 6).
constexpr int kBf16TileDim = 128;
//: Threads per block: four warps.
constexpr int kBf16TileThreads = 128;
constexpr int kBf16TileWarps = kBf16TileThreads / 32;
constexpr int kBf16RowsPerWarp = kBf16TileBr / kBf16TileWarps;
//: One staged row of the key, value, or query tile: the head width plus eight
//: elements of bank padding. The stride is 272 bytes, which is a multiple of
//: sixteen, so every staged segment stays aligned for a sixteen-byte copy.
constexpr int kBf16TileStride = kBf16TileDim + 8;
//: One row of the per-warp probability tile.
constexpr int kBf16ProbStride = kBf16TileBc + 8;
//: Sixteen-key sub-tiles of one key tile, and sixteen-element sub-tiles of the
//: head width.
constexpr int kBf16SubTiles = kBf16TileBc / 16;
constexpr int kBf16DimTiles = kBf16TileDim / 16;
//: Resident blocks per streaming multiprocessor.
constexpr int kBf16TileMinBlocks = 2;
//: Sixteen-byte copy granularity of the staged rows.
constexpr int kBf16ChunkElems = 8;

constexpr std::size_t kBf16QueryElems =
    static_cast<std::size_t>(kBf16TileBr) * kBf16TileStride;
constexpr std::size_t kBf16KvElems =
    static_cast<std::size_t>(kBf16TileBc) * kBf16TileStride;
constexpr std::size_t kBf16ProbElems =
    static_cast<std::size_t>(kBf16TileWarps) * kBf16RowsPerWarp *
    kBf16ProbStride;
constexpr std::size_t kBf16ScratchElems =
    static_cast<std::size_t>(kBf16TileWarps) * kBf16RowsPerWarp * kBf16TileBc;

//: The dynamic shared memory the tile requests. `tests/cuda_sim_test.cc`
//: refuses a launch above 48 KiB without the matching `cudaFuncSetAttribute`.
constexpr std::size_t kBf16TileSharedBytes =
    (kBf16QueryElems + 4 * kBf16KvElems + kBf16ProbElems) *
        sizeof(__nv_bfloat16) +
    kBf16ScratchElems * sizeof(float);

// --- Staging --------------------------------------------------------------

namespace bf16_tile_detail {

// Copies one key and one value tile into the stage `stage` buffers. Rows at or
// past `kv_len` are zeroed with plain stores, because an asynchronous copy
// cannot write a constant.
__device__ __forceinline__ void StageKeyValue(const AttentionParams& params,
                                              int tile, int stage, int b,
                                              int kvh, const ComputeType* k,
                                              const ComputeType* v,
                                              __nv_bfloat16* ks,
                                              __nv_bfloat16* vs) {
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = cuda_kernels::KvHeadCountDev(params);
  const int dim = params.head_dim;
  const int chunks = kBf16TileDim / kBf16ChunkElems;

  __nv_bfloat16* kdst = ks + static_cast<std::size_t>(stage) * kBf16KvElems;
  __nv_bfloat16* vdst = vs + static_cast<std::size_t>(stage) * kBf16KvElems;

  for (int idx = threadIdx.x; idx < kBf16TileBc * chunks;
       idx += kBf16TileThreads) {
    const int row = idx / chunks;
    const int chunk = idx - row * chunks;
    const int j = tile * kBf16TileBc + row;
    __nv_bfloat16* kout =
        kdst + row * kBf16TileStride + chunk * kBf16ChunkElems;
    __nv_bfloat16* vout =
        vdst + row * kBf16TileStride + chunk * kBf16ChunkElems;
    if (j < kv_len) {
      const long long offset =
          ((static_cast<long long>(b) * kv_len + j) * kv_heads + kvh) * dim +
          chunk * kBf16ChunkElems;
      const uint4* ksrc = reinterpret_cast<const uint4*>(k + offset);
      const uint4* vsrc = reinterpret_cast<const uint4*>(v + offset);
      __pipeline_memcpy_async(kout, ksrc, sizeof(uint4));
      __pipeline_memcpy_async(vout, vsrc, sizeof(uint4));
    } else {
      *reinterpret_cast<uint4*>(kout) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(vout) = make_uint4(0, 0, 0, 0);
    }
  }
}

// Copies the query rows of this block into the resident query tile. An invalid
// row is zeroed so its products stay finite; the epilogue never writes it.
__device__ __forceinline__ void StageQuery(const AttentionParams& params,
                                           int t0, int b, int h,
                                           const ComputeType* q,
                                           __nv_bfloat16* qs) {
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int dim = params.head_dim;
  const int chunks = kBf16TileDim / kBf16ChunkElems;
  for (int idx = threadIdx.x; idx < kBf16TileBr * chunks;
       idx += kBf16TileThreads) {
    const int row = idx / chunks;
    const int chunk = idx - row * chunks;
    const int tq = t0 + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (tq < seq) {
      const long long offset =
          (static_cast<long long>(b) * seq + tq) * heads * dim +
          static_cast<long long>(h) * dim + chunk * kBf16ChunkElems;
      value = *reinterpret_cast<const uint4*>(q + offset);
    }
    *reinterpret_cast<uint4*>(qs + row * kBf16TileStride +
                              chunk * kBf16ChunkElems) = value;
  }
}

}  // namespace bf16_tile_detail

// --- The tile -------------------------------------------------------------

// One thread block owns `kBf16TileBr` query rows of one `(batch, head)` pair.
// Warp `w` owns query rows `w * 16 .. w * 16 + 15`; inside the warp, lane `l`
// keeps the online state of row `l & 15`, and the two lanes of a row share it
// through a butterfly reduction over the 64 keys of a tile.
static __global__ void __launch_bounds__(kBf16TileThreads, kBf16TileMinBlocks)
    Bf16AttentionForwardKernel(const AttentionParams params,
                               const ComputeType* __restrict__ q,
                               const ComputeType* __restrict__ k,
                               const ComputeType* __restrict__ v,
                               ComputeType* __restrict__ out,
                               float* __restrict__ stats) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  namespace wmma = nvcuda::wmma;
  using CudaBf16 = __nv_bfloat16;

  const int dim = params.head_dim;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const float scale =
      params.scale > 0.0f ? params.scale : rsqrtf(static_cast<float>(dim));

  const int t0 = blockIdx.x * kBf16TileBr;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int kvh = cuda_kernels::KvHeadDev(h, heads, params.num_kv_heads);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // The query stream position of the first and the last row of this block.
  // A decode step has `seq < kv_len`; the offset keeps the mask in model
  // positions instead of local ones.
  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kBf16TileBr - 1;

  // The visible keys of the block are one contiguous range: the left window
  // bounds it from below, and causality or the right window bounds it from
  // above. The block therefore skips a run of leading tiles and a run of
  // trailing tiles, and it tests nothing inside the range.
  const int num_key_tiles = (kv_len + kBf16TileBc - 1) / kBf16TileBc;
  int first_key = 0;
  if (params.window_left >= 0) first_key = qpos_min - params.window_left;
  const int first_tile = first_key > 0 ? first_key / kBf16TileBc : 0;
  int last_key = kv_len - 1;
  if (params.causal && qpos_max < last_key) last_key = qpos_max;
  if (params.window_right >= 0 && qpos_max + params.window_right < last_key) {
    last_key = qpos_max + params.window_right;
  }
  int last_tile = last_key < 0 ? -1 : last_key / kBf16TileBc;
  if (last_tile > num_key_tiles - 1) last_tile = num_key_tiles - 1;

  // Shared memory: the resident query tile, the two key stages, the two value
  // stages, the per-warp probability tile, and the per-warp score scratch.
  extern __shared__ __align__(16) char tile_smem[];
  CudaBf16* qs = reinterpret_cast<CudaBf16*>(tile_smem);
  CudaBf16* ks = qs + kBf16QueryElems;
  CudaBf16* vs = ks + 2 * kBf16KvElems;
  CudaBf16* ps = vs + 2 * kBf16KvElems;
  float* ss = reinterpret_cast<float*>(ps + kBf16ProbElems);

  CudaBf16* prob = ps + warp * kBf16RowsPerWarp * kBf16ProbStride;
  float* scratch = ss + warp * kBf16RowsPerWarp * kBf16TileBc;

  // The online-softmax state of the one row this lane keeps, and the output
  // accumulator of the sixteen rows the warp owns.
  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kBf16DimTiles];
#pragma unroll
  for (int t = 0; t < kBf16DimTiles; ++t) wmma::fill_fragment(acc[t], 0.0f);

  bf16_tile_detail::StageQuery(params, t0, b, h, q, qs);
  __syncthreads();

  const int visible = last_tile - first_tile + 1;

  // Two stages alternate. Iteration `i` stages tile `i + 1` before it waits for
  // tile `i`, so the copy overlaps the arithmetic. Tile `i + 1` takes the stage
  // that tile `i - 1` released, so a stage is never refilled while a warp still
  // reads it.
  if (visible > 0) {
    bf16_tile_detail::StageKeyValue(params, first_tile, 0, b, kvh, k, v, ks,
                                    vs);
    __pipeline_commit();
  }

  for (int i = 0; i < visible; ++i) {
    if (i + 1 < visible) {
      bf16_tile_detail::StageKeyValue(params, first_tile + i + 1, (i + 1) & 1,
                                      b, kvh, k, v, ks, vs);
      __pipeline_commit();
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();

    const int stage = i & 1;
    const int tile = first_tile + i;
    const CudaBf16* k_tile =
        ks + static_cast<std::size_t>(stage) * kBf16KvElems;
    const CudaBf16* v_tile =
        vs + static_cast<std::size_t>(stage) * kBf16KvElems;

    // S = scale * Q K^T, as four sub-tiles of sixteen keys. The query fragment
    // is loaded once per head-dimension step and reused by every sub-tile.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score[kBf16SubTiles];
#pragma unroll
    for (int s = 0; s < kBf16SubTiles; ++s) wmma::fill_fragment(score[s], 0.0f);
    for (int d = 0; d < kBf16TileDim; d += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, CudaBf16, wmma::row_major> a;
      wmma::load_matrix_sync(
          a, qs + (warp * kBf16RowsPerWarp) * kBf16TileStride + d,
          kBf16TileStride);
#pragma unroll
      for (int s = 0; s < kBf16SubTiles; ++s) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, CudaBf16, wmma::col_major>
            bf;
        wmma::load_matrix_sync(bf, k_tile + (s * 16) * kBf16TileStride + d,
                               kBf16TileStride);
        wmma::mma_sync(score[s], a, bf, score[s]);
      }
    }
#pragma unroll
    for (int s = 0; s < kBf16SubTiles; ++s) {
      wmma::store_matrix_sync(scratch + s * 16, score[s], kBf16TileBc,
                              wmma::mem_row_major);
    }
    __syncwarp();

    // The mask, the row maximum, and P = exp(S - row_max). Lane `l` owns row
    // `l & 15` and one half of its 64 keys, so the butterfly of distance 16
    // joins the two halves of a row.
    const int row = lane & 15;
    const int col0 = (lane >> 4) * 32;
    const int tq = t0 + warp * kBf16RowsPerWarp + row;
    const int qpos = kv_len - seq + tq;
    float* srow = scratch + row * kBf16TileBc + col0;

    float local_max = -CUDART_INF_F;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int j = tile * kBf16TileBc + col0 + c;
      float sv = srow[c] * scale;
      if (j >= kv_len || !cuda_kernels::KeyAllowedDev(params, qpos, j)) {
        sv = -CUDART_INF_F;
      }
      srow[c] = sv;
      local_max = fmaxf(local_max, sv);
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 16));

    float alpha = 1.0f;
    if (local_max != -CUDART_INF_F) {
      const float previous = row_max;
      const float updated = fmaxf(previous, local_max);
      row_max = updated;
      alpha = (previous == -CUDART_INF_F) ? 0.0f : expf(previous - updated);
    }

    float local_sum = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      float p = 0.0f;
      if (local_max != -CUDART_INF_F) p = expf(srow[c] - row_max);
      local_sum += p;
      prob[row * kBf16ProbStride + col0 + c] = __float2bfloat16(p);
    }
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 16);
    row_sum = alpha * row_sum + local_sum;

    // Publish the rescale of this key tile. The score scratch is dead now, so
    // its first sixteen columns hold one alpha per row and load back as an
    // accumulator fragment in the same layout as the output accumulator.
    if (lane < 16) {
#pragma unroll
      for (int c = 0; c < 16; ++c) scratch[lane * kBf16TileBc + c] = alpha;
    }
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rescale;
    wmma::load_matrix_sync(rescale, scratch, kBf16TileBc, wmma::mem_row_major);
#pragma unroll
    for (int t = 0; t < kBf16DimTiles; ++t) {
#pragma unroll
      for (int e = 0; e < rescale.num_elements; ++e) {
        acc[t].x[e] *= rescale.x[e];
      }
    }

    // O += P V, one probability sub-tile loaded per sixteen keys and reused by
    // every head-dimension sub-tile.
    for (int s = 0; s < kBf16SubTiles; ++s) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, CudaBf16, wmma::row_major> pa;
      wmma::load_matrix_sync(pa, prob + s * 16, kBf16ProbStride);
#pragma unroll
      for (int t = 0; t < kBf16DimTiles; ++t) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, CudaBf16, wmma::row_major>
            vb;
        wmma::load_matrix_sync(vb, v_tile + (s * 16) * kBf16TileStride + t * 16,
                               kBf16TileStride);
        wmma::mma_sync(acc[t], pa, vb, acc[t]);
      }
    }

    __syncthreads();
  }

  // The epilogue. An output row is scaled by `1 / row_sum`, and the lane that
  // holds that row publishes the inverse past the sixteen by sixteen store
  // region. The statistics need no shared memory: the same lane holds them.
  constexpr int kInvSlot = kBf16RowsPerWarp * 16;
  if (lane < 16) {
    scratch[kInvSlot + lane] = row_sum <= 0.0f ? 0.0f : 1.0f / row_sum;
  }
  __syncwarp();

  for (int t = 0; t < kBf16DimTiles; ++t) {
    wmma::store_matrix_sync(scratch, acc[t], 16, wmma::mem_row_major);
    __syncwarp();
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int idx = lane + n * 32;
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tq = t0 + warp * kBf16RowsPerWarp + r;
      if (tq >= seq) continue;
      const float value = scratch[idx] * scratch[kInvSlot + r];
      const std::size_t base =
          (static_cast<std::size_t>(b) * seq + tq) * heads * kBf16TileDim +
          static_cast<std::size_t>(h) * kBf16TileDim + t * 16 + c;
      out[base] = cuda_kernels::ToComputeDev(value);
    }
    __syncwarp();
  }

  if (lane < 16) {
    const int tq = t0 + warp * kBf16RowsPerWarp + lane;
    if (tq < seq) {
      const bool empty = row_sum <= 0.0f;
      const std::size_t stat_base =
          ((static_cast<std::size_t>(b) * heads + h) * seq + tq) * 2;
      stats[stat_base] = empty ? 0.0f : row_max;
      stats[stat_base + 1] = empty ? 0.0f : row_sum;
    }
  }
#endif
}

// Whether this device takes the tile for this shape. The shape conditions are
// in `attention_dispatch.h`; this adds the device answer. `attention.cu` asks
// this function instead of stating the conditions itself, so the compiled and
// the emulated answers cannot disagree. See `docs/attention.md` Part 6,
// section 5.
//
// The tile replaces the dense cuBLAS score matrix with the tensor cores, so a
// bf16 build on a card without them keeps cuBLAS. The fp32 tile in
// `attention.cu` is the mirror case: 34.6 ms for the 13 windowed layers of the
// `big` profile, against a 1.06 ms roofline bound for all 20.
bool UseFusedAttentionBf16(const AttentionParams& p) {
  static const bool tensor_cores = GetCaps().has_tensor_cores;
  return FusedAttentionBf16Covers(p, tensor_cores);
}

// Requests the opt-in shared-memory limit once, then launches the tile.
void LaunchBf16Attention(const AttentionParams& params, const ComputeType* q,
                         const ComputeType* k, const ComputeType* v,
                         ComputeType* out, float* stats) {
  static const bool configured = [] {
    cudaFuncSetAttribute(
        reinterpret_cast<const void*>(Bf16AttentionForwardKernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kBf16TileSharedBytes));
    return true;
  }();
  (void)configured;
  const int query_tiles = (params.seq + kBf16TileBr - 1) / kBf16TileBr;
  const dim3 grid(query_tiles, params.num_heads, params.batch);
  cuda_backend::Launch(Bf16AttentionForwardKernel, grid, dim3(kBf16TileThreads),
                       kBf16TileSharedBytes, params, q, k, v, out, stats);
}

}  // namespace kernels
}  // namespace nanochat
