// The Hopper attention forward tile: the query tile in registers, both key and
// value rings filled by the copy engine, one producer warpgroup, and two
// consumer warpgroups that issue every product and wait on no global load
// (docs/attention.md Part 8).
//
// **The geometry.** One block owns `kBr` query rows of one `(batch, head)`
// pair: two consumer warpgroups of `kAtomM` rows each, so both of them share
// the one staged key tile of a step and the arithmetic of one staged byte is
// doubled by the second warpgroup. The key tile is `kAtomN` keys, which is the
// `n` extent of `wgmma.mma_async.m64n128k16`, and the head width is the `k`
// extent. Two stages of both rings is what the shared window holds beside the
// two query tiles: 224 of the 227 KiB a block may ask for.
//
// **The two rings.** The key ring is staged in the swizzled panel form the
// copy engine fills, which is the form the score product reads with a layout
// one descriptor. The value ring is staged in the same form and then
// transposed in shared memory into the tile the value product reads, because a
// press of 1,594 stagings, transpose bits, atom steps, and descriptor offsets
// found exactly one form the instruction reads and it is the transposed
// staging. So the copies carry both rings and the transposition stays a pass.
//
// **The roles.** A producer that issues the copies and no product and
// consumers that issue every product and no copy hand a stage over through two
// `mbarrier` objects, so neither side waits for the other beyond the tile in
// flight. `setmaxnreg` moves the register file between them: the producer gives
// back what it does not need and the consumers take it.
//
// **The measured rate.** The probe of `dev/kernels/wgmma_bench.cu` states this
// geometry at 395.6 TFLOP/s dense and 390.9 causal, against 71.9 for the
// `mma.sync` family of Part 7 on the card, which is the number this file
// exists to move. The probe's rows also state the two levers that remain and
// are not here: the value transposition off the producer's path, and the
// intensity of the grouped-query pair.
//
// **What this file does not do.** It covers the forward of `head_dim` 128
// only, because the tile's key width is the instruction's and the head width is
// its pass depth. It writes the output and both statistics of the family
// unchanged, so the backward of Part 7 reads them as it does today.
//
// The tile is architecture conditional: the target states `archs = ["sm_90a"]`,
// which replaces the command line for that target alone, so no build command
// carries an arch flag and the default build stays at the tree's architecture.
// The tripwire below refuses a device build that names another architecture,
// and every build that does not compile this file links
// `attention_hopper_absent.cc` instead.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/attention_dispatch.h"
#include "backends/cuda/kernels/attention_hopper.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "backends/cuda/kernels/hopper_ops.cuh"
#include "nanochat/kernels.h"

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error \
    "attention_hopper.cu needs sm_90a; its target states archs = [\"sm_90a\"]"
#endif

namespace nanochat {
namespace kernels {
namespace {

using CudaBf16 = __nv_bfloat16;
namespace hop = cuda_kernels::hopper;

// --- The geometry -----------------------------------------------------------

//: The head width, which is the `k` extent of both products and the `n` extent
//: of the value product. The forward of any other width keeps the family.
constexpr int kDim = 128;

//: Query rows of one consumer warpgroup, keys of one staged tile, and the `k`
//: extent of one atom. The first is the `m` extent of the instruction.
constexpr int kAtomM = 64;
constexpr int kAtomN = 128;
constexpr int kAtomK = 16;

//: The atoms of one product: over the head width for the score, over the key
//: width for the value.
constexpr int kAtoms = kDim / kAtomK;
constexpr int kKeyAtoms = kAtomN / kAtomK;

//: The consumer warpgroups of a block and the query rows it owns, the ring
//: depth, and the threads of the block: one producer warpgroup beside the
//: consumers.
constexpr int kConsumers = 2;
constexpr int kBr = kConsumers * kAtomM;
constexpr int kStages = 2;
constexpr int kThreads = (1 + kConsumers) * hop::kWarpgroupThreads;

//: The accumulator slots of one thread. The score tile is `kAtomM` rows by
//: `kAtomN` keys and the output tile is the same rows by the head width, so one
//: column loop reads both.
constexpr int kScoreSlots = kAtomM * kAtomN / hop::kWarpgroupThreads;
constexpr int kOutputSlots = kAtomM * kDim / hop::kWarpgroupThreads;
static_assert(kScoreSlots == kOutputSlots,
              "one slot loop must cover the score and the output tile");

//: The shared footprint of one tile of each ring, and of the whole block: the
//: query tile of every consumer, the key ring, the value ring the step reads,
//: and the raw ring the copy fills before the transposition.
constexpr int kQueryBytes = kAtomM * kDim * sizeof(CudaBf16);
constexpr int kKeyBytes = kAtomN * kDim * sizeof(CudaBf16);
constexpr int kValueBytes = kAtomN * kDim * sizeof(CudaBf16);
constexpr int kPanelBytes = kAtomN * 64 * sizeof(CudaBf16);
constexpr int kSharedBytes = kConsumers * kQueryBytes +
                             kStages * (kKeyBytes + kValueBytes + kValueBytes);

//: The largest dynamic shared-memory request the card accepts, which this block
//: must stay under. One block per multiprocessor is the schedule.
constexpr int kSmemCeiling = 227 * 1024;
static_assert(kSharedBytes <= kSmemCeiling,
              "one block of this tile must fit the shared window");

//: The bytes one stage of both rings states to its barrier: four boxes, two
//: panels of the key tile and two of the value tile.
constexpr int kStageBytes = 4 * kPanelBytes;

//: The shared tile the epilogue stages: one row of the head width plus eight
//: elements, so the four lanes that hold one row do not all address one bank.
//: The last step drains the key ring, so the epilogue reuses it and asks for no
//: shared memory of its own.
constexpr int kEpilogueStride = kDim + 8;
constexpr int kEpilogueBytes = kAtomM * kEpilogueStride * sizeof(CudaBf16);
static_assert(kConsumers * kEpilogueBytes <= kStages * kKeyBytes,
              "the epilogue stages into the key ring, which the last step "
              "frees");

//: The register budget of the two roles. The producer holds addresses, loop
//: counters, and a staging slice; the consumers hold two accumulators, the
//: probability tile, and the online-softmax state. The total over a block of
//: 384 threads stays inside the 65,536 registers of one multiprocessor:
//: 128 by 32 and 256 by 224 is 61,440.
constexpr int kProducerRegisters = 32;
constexpr int kConsumerRegisters = 224;

//: The interleaved core-matrix form: one core matrix is eight rows of eight
//: elements, sixteen bytes to a row, so the two halves of the `k` axis of one
//: row group are one leading offset apart and the next row group follows at one
//: stride offset.
constexpr int kElementsPerGroup = 8;
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 2 * kLboBytes;

//: The descriptor of a swizzled key tile, in sixteen-byte units: one core
//: matrix group is eight rows of a hundred and twenty-eight bytes, so the atom
//: step is two units and the panel is a kilobyte of them.
constexpr int kSwizzledAtomUnits = 2;
constexpr int kPanelStepUnits = 1024;
//: The descriptor's two offsets, in bytes: the leading offset is the sixteen
//: bytes of one half of a core matrix, and the stride offset is the one
//: kilobyte of one atom of eight rows. The atom step of the *instruction* is
//: not the panel step of the *copy*: four atoms fill one panel, so the panel
//: step is a kilobyte of sixteen-byte units and the descriptor's stride is a
//: kilobyte of bytes.
constexpr int kSwizzledLeadingBytes = 16;
constexpr int kSwizzledStrideBytes = 1024;

//: The base-two form of the exponential. `exp2` is one instruction where `exp`
//: is a range reduction around it, so the softmax scale carries `log2(e)` and
//: every row maximum this tile states is in base-two units.
constexpr float kLog2e = 1.4426950408889634f;
//: The natural-unit scale the family saves beside the running maximum.
constexpr float kLn2 = 0.6931471805599453f;
//: The identity of a row maximum and the sentinel of the running one. It is a
//: finite value, so the first rescale of a row underflows to zero instead of
//: forming an infinity.
constexpr float kNegInf = -3.402823466e38f;

// --- The staging ------------------------------------------------------------

//: One `k` step of a descriptor, in sixteen-byte units, for a tile of `rows`.
__host__ __device__ constexpr int AtomStep(int rows) {
  return (rows / 8) * kSboBytes / 16;
}
static_assert(kAtoms * AtomStep(kAtomM) * 16 == kQueryBytes,
              "the query tile must fill its interleaved form");
static_assert(kAtoms * AtomStep(kDim) * 16 == kValueBytes,
              "the value tile must fill its interleaved form");

//: The interleaved offset of one element of a row-major source tile.
__device__ __forceinline__ int InterleavedOffset(int row, int k, int rows) {
  const int block = k / kAtomK;
  const int half = (k % kAtomK) / kElementsPerGroup;
  const int element = k % kElementsPerGroup;
  const int group = row / 8;
  return block * AtomStep(rows) * 16 + group * kSboBytes + half * kLboBytes +
         (row % 8) * 16 + element * 2;
}

//: Stages `kAtomM` query rows of one consumer warpgroup in the interleaved form
//: the score product reads, one sixteen-byte group of a row a thread.
//:
//: Sixteen bytes is the unit the interleaved form keeps contiguous, so the pass
//: is one vector load and one vector store an iteration. A pass that moves
//: sixteen bits an access pays a memory latency an element instead, and that is
//: the whole floor of a block (`docs/attention.md` Part 8, section 9.9).
//:
//: A row past the sequence is zeroed, and the epilogue skips it, so a partial
//: last block reads no memory it does not own.
__device__ __forceinline__ void StageQueryTile(char* stage,
                                               const CudaBf16* source,
                                               int row_stride, int valid_rows,
                                               int thread) {
  constexpr int kGroups = kDim / kElementsPerGroup;
  constexpr int kUnits = kAtomM * kGroups;
  for (int unit = thread; unit < kUnits; unit += hop::kWarpgroupThreads) {
    const int row = unit / kGroups;
    const int k = (unit - row * kGroups) * kElementsPerGroup;
    const uint4 value =
        row < valid_rows
            ? *reinterpret_cast<const uint4*>(
                  source + static_cast<std::size_t>(row) * row_stride + k)
            : make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(stage + InterleavedOffset(row, k, kAtomM)) =
        value;
  }
}

//: The transposition of one staged value tile: from the swizzled panel form the
//: copy engine fills into the interleaved tile of `dim` rows by `kAtomN` keys
//: that the value product reads.
//:
//: The swizzle permutes whole sixteen-byte chunks and is its own inverse, so
//: one read of the source states where a core matrix sits: the panel of the
//: head-width block, the row of the key, and the chunk exclusive-ored by the
//: row of its atom. One thread reads eight keys of one head-width block, which
//: is coalesced across the warp, and writes the eight sixteen-byte pieces the
//: interleaved form keeps adjacent. The store order is the straight one: moving
//: the piece each lane writes by the lane's own index spreads the banks, and it
//: measured a third of the rate because the eight rows then leave the register
//: file (section 9.8 of Part 8).
__device__ __forceinline__ void StageTransposeValue(char* stage,
                                                    const char* source,
                                                    int thread) {
  constexpr int kUnits = (kAtomN / 8) * (kDim / 16) * 2;
  for (int unit = thread; unit < kUnits; unit += hop::kWarpgroupThreads) {
    const int half = unit % 2;
    const int block = (unit / 2) % (kDim / 16);
    const int group = unit / (2 * (kDim / 16));
    const int column = block * 16 + half * 8;
    const int panel = column / 64;
    const int chunk = ((column % 64) / 8);
    char* to =
        stage + InterleavedOffset((block * 2 + half) * 8, group * 8, kDim);
    std::uint32_t rows[8][4];
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      const int key = group * 8 + row;
      const int offset =
          panel * kPanelBytes + key * 128 + ((chunk ^ (key & 7)) * 16);
      const uint4 loaded = *reinterpret_cast<const uint4*>(source + offset);
      rows[row][0] = loaded.x;
      rows[row][1] = loaded.y;
      rows[row][2] = loaded.z;
      rows[row][3] = loaded.w;
    }
#pragma unroll
    for (int column_out = 0; column_out < 8; ++column_out) {
      const int word = column_out / 2;
      const std::uint32_t selector = (column_out % 2) != 0 ? 0x7632u : 0x5410u;
      uint4 out;
      out.x = __byte_perm(rows[0][word], rows[1][word], selector);
      out.y = __byte_perm(rows[2][word], rows[3][word], selector);
      out.z = __byte_perm(rows[4][word], rows[5][word], selector);
      out.w = __byte_perm(rows[6][word], rows[7][word], selector);
      *reinterpret_cast<uint4*>(to + column_out * 16) = out;
    }
  }
}

// --- One step of the tile ---------------------------------------------------

//: The four numbers the mask of a step reads. They are scalars of the block and
//: not the whole `AttentionParams`, because the mask reads them once a slot of
//: a tile and the parameter block is larger than the register file wants to
//: hold across a step.
struct MaskBounds {
  long long kv_len;
  int causal;
  int window_left;
  int window_right;
};

//: Whether one query position may read one key position, which is the rule of
//: `cuda_kernels::KeyAllowedDev` over the scalars above.
__device__ __forceinline__ bool MaskAllows(const MaskBounds& mask,
                                           long long qpos, long long key_pos) {
  // A key past the sequence is hidden whatever the mask says, because the copy
  // of the last tile reads a box wider than the rows that exist. The engine
  // fills the part of the box past the tensor with zeros, and the rows between
  // the sequence and the next head are hidden here.
  if (key_pos >= mask.kv_len) return false;
  if (mask.causal != 0 && key_pos > qpos) return false;
  if (mask.window_left >= 0 && qpos - key_pos > mask.window_left) return false;
  if (mask.window_right >= 0 && key_pos - qpos > mask.window_right) {
    return false;
  }
  return true;
}

// One step: the score product over the whole head width, the mask, the row
// maximum, the exponentials and the row sum, the rescale of the output
// accumulator, the rounding of the probability tile, and the value product
// with that tile as its first operand in registers.
//
// `kMask` states a tile that is not fully visible for every row of the
// warpgroup: the causal boundary, the window boundary, or a last key tile that
// runs past the sequence. A row whose whole tile is hidden keeps the step out
// of its running state, so the two sentinels never meet in a subtraction. A
// fully visible tile compiles without the branch, the compare, and the select,
// and `kMask` false is the common case of a dense band.
//
// `qpos_base` is the stream position of the first query row of the calling
// warpgroup and `key_base` the position of the first key of the tile, because
// the mask is stated in model positions and not in tile offsets.
template <bool kMask>
__device__ __forceinline__ void FlashStep(
    hop::MatrixDescriptor query, hop::MatrixDescriptor key,
    hop::MatrixDescriptor value, float log2_scale, int lane, int qpos_base,
    int key_base, const MaskBounds& mask, float (&running_max)[2],
    float (&running_sum)[2], float (&output)[kOutputSlots]) {
  // The score product. The first atom overwrites, so the accumulator carries no
  // score of the step before it.
  float score[kScoreSlots];
#pragma unroll
  for (int i = 0; i < kScoreSlots; ++i) score[i] = 0.0f;
  hop::FenceAccumulator<kScoreSlots>(score);
  hop::Fence();
#pragma unroll
  for (int atom = 0; atom < kAtoms; ++atom) {
    // A swizzled key tile is two panels of sixty-four columns, so four atoms
    // sit in one panel and the fifth jumps a panel.
    const int key_units =
        (atom / 4) * kPanelStepUnits + (atom % 4) * kSwizzledAtomUnits;
    hop::ScoreAtom<kAtomN>(
        hop::MatrixDescriptor{query.bits + atom * AtomStep(kAtomM)},
        hop::MatrixDescriptor{key.bits + key_units}, score, atom != 0);
  }
  hop::Commit();
  hop::Wait<0>();
  hop::FenceAccumulator<kScoreSlots>(score);

  // The mask, and the row maximum in base-two units. The maximum is monotone
  // under a positive scale, so the raw score is reduced and the one reduced
  // value is scaled.
  float tile_max[2] = {kNegInf, kNegInf};
#pragma unroll
  for (int i = 0; i < kScoreSlots; ++i) {
    if constexpr (kMask) {
      const long long qpos = qpos_base + hop::SlotRow(lane, i);
      const long long key_pos = key_base + hop::SlotColumn(lane, i);
      // A key past the sequence is hidden whatever the mask says, because the
      // copy of the last tile reads a box wider than the rows that exist. The
      // engine fills the part of the box past the tensor with zeros, and the
      // rows between the sequence and the next head are masked here.
      if (!MaskAllows(mask, qpos, key_pos)) score[i] = kNegInf;
    }
    const int slot = (i / 2) % 2;
    tile_max[slot] = fmaxf(tile_max[slot], score[i]);
  }

  bool active[2] = {true, true};
  float new_max[2] = {running_max[0], running_max[1]};
  float rescale[2] = {1.0f, 1.0f};
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    tile_max[slot] =
        fmaxf(tile_max[slot], __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 1));
    tile_max[slot] =
        fmaxf(tile_max[slot], __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 2));
    if constexpr (kMask) active[slot] = tile_max[slot] != kNegInf;
    tile_max[slot] *= log2_scale;
    new_max[slot] = fmaxf(running_max[slot], tile_max[slot]);
    // A row maximum that did not move gives a factor of exactly 1, which is
    // the state of a hidden tile and of a row that has seen no key yet.
    rescale[slot] = cuda_kernels::FastExp2(running_max[slot] - new_max[slot]);
  }

  // The exponentials and the row sum, in the same two reductions. A hidden row
  // contributes nothing, so its sentinel never reaches the exponential pipe.
  float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kScoreSlots; ++i) {
    const int slot = (i / 2) % 2;
    float probability =
        cuda_kernels::FastExp2(fmaf(score[i], log2_scale, -new_max[slot]));
    if constexpr (kMask) {
      if (!active[slot]) probability = 0.0f;
    }
    score[i] = probability;
    tile_sum[slot] += probability;
  }
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 1);
    tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 2);
    running_sum[slot] = running_sum[slot] * rescale[slot] + tile_sum[slot];
    running_max[slot] = new_max[slot];
  }

  // The output accumulator holds the earlier steps under the earlier maximum,
  // so every slot is rescaled before this step's value product adds to it.
#pragma unroll
  for (int i = 0; i < kOutputSlots; ++i) {
    output[i] *= rescale[(i / 2) % 2];
  }

  // The probability operand: two keys a register, from the score slots of this
  // same lane, low half of the register first. One conversion a pair, which is
  // half what two scalar forms and an or cost. The map is a bijection, so no
  // probability leaves the lane that holds its score.
  std::uint32_t probability[kKeyAtoms][4];
#pragma unroll
  for (int atom = 0; atom < kKeyAtoms; ++atom) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      const int index = 8 * atom + 4 * (reg / 2) + 2 * (reg % 2);
      const __nv_bfloat162 pair =
          __floats2bfloat162_rn(score[index], score[index + 1]);
      std::memcpy(&probability[atom][reg], &pair, sizeof(std::uint32_t));
    }
  }
  hop::FencePacked<kKeyAtoms * 4>(&probability[0][0]);
  hop::Fence();
#pragma unroll
  for (int atom = 0; atom < kKeyAtoms; ++atom) {
    // The value tile is the transposed staging of `dim` rows by `kAtomN` keys,
    // so one atom of the product steps it by one pass of the head width.
    hop::ValueAtom(probability[atom],
                   hop::MatrixDescriptor{value.bits + atom * AtomStep(kDim)},
                   output);
  }
  hop::Commit();
  hop::Wait<0>();
}

//: Whether every key of a tile is visible to one query row, which is the same
//: question the family asks of its own tile.
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

// --- The tile ---------------------------------------------------------------

// One block of `kBr` query rows of one `(batch, head)` pair. The producer
// warpgroup issues four tiled copies a stage and runs the transposition of the
// value tile; the two consumer warpgroups each own `kAtomM` query rows and run
// one step a staged key tile.
//
// The four maps are the two panels of the key tensor and the two of the value
// tensor, each a rank-four map over `(head width, key, kv head, batch)` in the
// swizzled form. They are built per call, as the CUTLASS arm of the GEMM builds
// its own, so a pointer that moves between layers needs no cache.
__global__ __launch_bounds__(kThreads, 1) void HopperForwardKernel(
    const AttentionParams params, const CudaBf16* __restrict__ q,
    const __grid_constant__ CUtensorMap key_low,
    const __grid_constant__ CUtensorMap key_high,
    const __grid_constant__ CUtensorMap value_low,
    const __grid_constant__ CUtensorMap value_high,
    ComputeType* __restrict__ out, float* __restrict__ stats) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* keys = query + kConsumers * kQueryBytes;
  char* values = keys + kStages * kKeyBytes;
  char* raw = values + kStages * kValueBytes;

  __shared__ __align__(8) std::uint64_t filled[kStages];
  __shared__ __align__(8) std::uint64_t ready[kStages];
  __shared__ __align__(8) std::uint64_t released[kStages];

  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_len = params.kv_len > 0 ? params.kv_len : seq;
  const int t0 = static_cast<int>(blockIdx.x) * kBr;
  const int h = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int kvh = cuda_kernels::KvHeadDev(h, heads, params.num_kv_heads);
  const int warpgroup = static_cast<int>(threadIdx.x) / hop::kWarpgroupThreads;
  const int lane = static_cast<int>(threadIdx.x) % hop::kWarpgroupThreads;

  // The visible keys of the block are one contiguous range: the left window
  // bounds it from below, and causality or the right window bounds it from
  // above. The block skips a run of leading tiles and a run of trailing tiles
  // and tests nothing inside the range.
  const int qpos_min = kv_len - seq + t0;
  const int qpos_max = qpos_min + kBr - 1;
  int first_key = 0;
  if (params.window_left >= 0) first_key = qpos_min - params.window_left;
  const int first_tile = first_key > 0 ? first_key / kAtomN : 0;
  int last_key = kv_len - 1;
  if (params.causal && qpos_max < last_key) last_key = qpos_max;
  if (params.window_right >= 0 && qpos_max + params.window_right < last_key) {
    last_key = qpos_max + params.window_right;
  }
  const int key_tiles = (kv_len + kAtomN - 1) / kAtomN;
  int last_tile = last_key < 0 ? -1 : last_key / kAtomN;
  if (last_tile > key_tiles - 1) last_tile = key_tiles - 1;
  const int visible_tiles = last_tile - first_tile + 1;

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      // One arrival the copies complete, one the producer states when the stage
      // reads, and one a consumer thread of the block.
      hop::BarrierInit(&filled[stage], 1);
      hop::BarrierInit(&ready[stage], 1);
      hop::BarrierInit(&released[stage], kConsumers * hop::kWarpgroupThreads);
    }
  }

  // The query tile of every consumer warpgroup, staged by the threads of the
  // producer warpgroup before the roles split. The stream layout of the model
  // and the head-major layout of the fused preparation differ in one stride, so
  // both are staged by the same pass.
  if (warpgroup == 0) {
    const int q_row_stride = params.q_head_major ? kDim : heads * kDim;
    for (int tile = 0; tile < kConsumers; ++tile) {
      const int first_row = t0 + tile * kAtomM;
      const std::size_t base =
          params.q_head_major
              ? (static_cast<std::size_t>(b) * heads + h) * seq + first_row
              : (static_cast<std::size_t>(b) * seq + first_row) * heads + h;
      const int valid_rows =
          first_row < seq
              ? (seq - first_row < kAtomM ? seq - first_row : kAtomM)
              : 0;
      StageQueryTile(query + tile * kQueryBytes, q + base * kDim, q_row_stride,
                     valid_rows, static_cast<int>(threadIdx.x));
    }
  }
  __syncthreads();

  if (warpgroup == 0) {
    // The producer: every copy, no product.
    hop::RegistersGive<kProducerRegisters>();
    for (int tile = 0; tile < visible_tiles; ++tile) {
      const int stage = tile % kStages;
      const unsigned phase = static_cast<unsigned>((tile / kStages) & 1);
      if (threadIdx.x == 0) {
        if (tile >= kStages) {
          hop::BarrierWait(&released[stage], static_cast<int>(phase ^ 1u));
        }
        hop::BarrierExpectTx(&filled[stage], kStageBytes);
        const int coordinates[4] = {0, kvh, (first_tile + tile) * kAtomN, b};
        char* key_stage = keys + stage * kKeyBytes;
        char* value_stage = raw + stage * kValueBytes;
        hop::TmaLoad<4>(key_stage, key_low, coordinates, &filled[stage]);
        hop::TmaLoad<4>(key_stage + kPanelBytes, key_high, coordinates,
                        &filled[stage]);
        hop::TmaLoad<4>(value_stage, value_low, coordinates, &filled[stage]);
        hop::TmaLoad<4>(value_stage + kPanelBytes, value_high, coordinates,
                        &filled[stage]);
        hop::BarrierWait(&filled[stage], static_cast<int>(phase));
      }
      // The whole producer warpgroup waits the copies and runs the
      // transposition, so the consumers issue no staging instruction and never
      // wait on a global load.
      hop::NamedBarrier<hop::kWarpgroupThreads>(1);
      StageTransposeValue(values + stage * kValueBytes,
                          raw + stage * kValueBytes,
                          static_cast<int>(threadIdx.x));
      hop::NamedBarrier<hop::kWarpgroupThreads>(1);
      if (threadIdx.x == 0) hop::BarrierArrive(&ready[stage]);
    }
    return;
  }

  // A consumer: every product, no copy.
  const int consumer = warpgroup - 1;
  hop::RegistersTake<kConsumerRegisters>();
  const float scale =
      params.scale > 0.0f ? params.scale : rsqrtf(static_cast<float>(kDim));
  const float log2_scale = scale * kLog2e;
  const int qpos_base = qpos_min + consumer * kAtomM;
  const long long stats_stride = seq;
  const MaskBounds mask{static_cast<long long>(kv_len), params.causal ? 1 : 0,
                        params.window_left, params.window_right};

  float running_max[2] = {kNegInf, kNegInf};
  float running_sum[2] = {0.0f, 0.0f};
  float output[kOutputSlots];
#pragma unroll
  for (int i = 0; i < kOutputSlots; ++i) output[i] = 0.0f;
  hop::FenceAccumulator<kOutputSlots>(output);

  const hop::MatrixDescriptor query_tile =
      hop::MakeMatrixDescriptor(query + consumer * kQueryBytes, kLboBytes,
                                kSboBytes, hop::OperandLayout::kInterleaved);
  hop::DeclareSharedOperand(query + consumer * kQueryBytes, kQueryBytes);

  for (int tile = 0; tile < visible_tiles; ++tile) {
    const int stage = tile % kStages;
    const int key_base = (first_tile + tile) * kAtomN;
    hop::BarrierWait(&ready[stage], (tile / kStages) & 1);
    hop::FenceAsyncShared();
    // The swizzled key tile is the copy engine's form and the descriptor states
    // it with a leading offset of sixteen bytes and a stride of one kilobyte.
    const hop::MatrixDescriptor key_tile = hop::MakeMatrixDescriptor(
        keys + stage * kKeyBytes, kSwizzledLeadingBytes, kSwizzledStrideBytes,
        hop::OperandLayout::kSwizzled128);
    const hop::MatrixDescriptor value_tile =
        hop::MakeMatrixDescriptor(values + stage * kValueBytes, kLboBytes,
                                  kSboBytes, hop::OperandLayout::kInterleaved);
    const bool full = TileFullyVisible(params, qpos_base, key_base,
                                       key_base + kAtomN - 1, kv_len) &&
                      TileFullyVisible(params, qpos_base + kAtomM - 1, key_base,
                                       key_base + kAtomN - 1, kv_len);
    if (full) {
      FlashStep<false>(query_tile, key_tile, value_tile, log2_scale, lane,
                       qpos_base, key_base, mask, running_max, running_sum,
                       output);
    } else {
      FlashStep<true>(query_tile, key_tile, value_tile, log2_scale, lane,
                      qpos_base, key_base, mask, running_max, running_sum,
                      output);
    }
    hop::BarrierArrive(&released[stage]);
  }

  // The epilogue: an output row is scaled by `1 / row_sum`, and the lane that
  // owns its first column publishes the two statistics of the family.
  //
  // The store stages the tile through shared memory and leaves as one
  // sixteen-byte copy a group, because the accumulator map scatters a row over
  // four lanes and eight columns. A pass that moves sixteen bits an access pays
  // a memory latency an element, which is the floor of a block
  // (docs/attention.md Part 8, section 9.9). The key ring is free here, because
  // the last step drained it.
  hop::FenceAccumulator<kOutputSlots>(output);
  CudaBf16* staged =
      reinterpret_cast<CudaBf16*>(keys + consumer * kEpilogueBytes);
  const int row_first = hop::SlotRow(lane, 0);
  const int quad_column = 2 * (lane % 4);
  const long long tq_base = t0 + consumer * kAtomM + row_first;
  float inverse[2];
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    inverse[slot] = running_sum[slot] > 0.0f ? 1.0f / running_sum[slot] : 0.0f;
  }
#pragma unroll
  for (int group = 0; group < kOutputSlots / 4; ++group) {
    const int offset = 8 * group + quad_column;
#pragma unroll
    for (int half = 0; half < 4; ++half) {
      const int slot = half / 2;
      staged[(row_first + 8 * slot) * kEpilogueStride + offset + (half % 2)] =
          cuda_kernels::ToComputeDev(output[4 * group + half] * inverse[slot]);
    }
  }
  // One consumer warpgroup at a time, on its own numbered barrier.
  hop::NamedBarrier<hop::kWarpgroupThreads>(2 + consumer);
  constexpr int kEpilogueGroups = kDim / kElementsPerGroup;
  constexpr int kEpilogueUnits = kAtomM * kEpilogueGroups;
  for (int unit = lane; unit < kEpilogueUnits; unit += hop::kWarpgroupThreads) {
    const int row = unit / kEpilogueGroups;
    const int column = (unit - row * kEpilogueGroups) * kElementsPerGroup;
    const long long tq = t0 + consumer * kAtomM + row;
    if (tq >= seq) continue;
    ComputeType* destination =
        out + (static_cast<std::size_t>(b) * seq + tq) * heads * kDim +
        static_cast<std::size_t>(h) * kDim + column;
    *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(
        staged + row * kEpilogueStride + column);
  }
  // The first column of a row belongs to one lane of its quad, which is the
  // lane that states the two statistics. The running maximum is base two and
  // the saved statistic is natural units; an empty row saves the pair (0, 0).
  if (quad_column != 0) return;
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const long long tq = tq_base + 8 * slot;
    if (tq >= seq) continue;
    const bool empty = running_sum[slot] <= 0.0f;
    const std::size_t base =
        (static_cast<std::size_t>(b) * heads + h) * stats_stride + tq;
    stats[base * 2] = empty ? 0.0f : running_max[slot] * kLn2;
    stats[base * 2 + 1] = empty ? 0.0f : running_sum[slot];
  }
}

// --- The host side ----------------------------------------------------------

//: The one driver entry this file takes. Nothing else of the driver interface
//: is needed, so this translation unit links no driver library and asks the
//: runtime for the symbol instead.
using EncodeTiledFunction =
    CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                 const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFunction EncodeTiled() {
  static EncodeTiledFunction function = nullptr;
  if (function == nullptr) {
    void* pointer = nullptr;
    const cudaError_t status = cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &pointer, cudaEnableDefault);
    if (status != cudaSuccess || pointer == nullptr) {
      std::fprintf(stderr, "fatal: no cuTensorMapEncodeTiled: %s\n",
                   cudaGetErrorString(status));
      std::exit(1);
    }
    function = reinterpret_cast<EncodeTiledFunction>(pointer);
  }
  return function;
}

// Reports a refused descriptor. A map that cannot be built is a programming
// error and not a shape the dispatch can hand to another path, because
// `UseHopperAttention` has already answered yes, so this stops the run rather
// than writing an output no step computed.
bool MakePanelMap(CUtensorMap* map, const void* source, int panel, int kv_len,
                  int kv_heads, int batch) {
  //: The `dim` axis is the map's innermost one, so a panel of sixty-four
  //: elements is the hundred and twenty-eight bytes the 128-byte swizzle
  //: spans, and the second panel is the same map over a base one panel on.
  const void* base =
      static_cast<const char*>(source) + static_cast<std::size_t>(panel) * 128;
  //: The axes are the tensor's own, innermost first, and the encoder requires
  //: a stride to rise from one axis to the next: the head is the innermost
  //: axis, then the key-value head, then the key, then the batch. A key tile is
  //: a box of one key-value head and one batch, so two of the four axes state
  //: one and the coordinate of each is the block's own.
  const cuuint64_t dims[4] = {64, static_cast<cuuint64_t>(kv_heads),
                              static_cast<cuuint64_t>(kv_len),
                              static_cast<cuuint64_t>(batch)};
  const cuuint64_t strides[3] = {
      static_cast<cuuint64_t>(kDim) * sizeof(CudaBf16),
      static_cast<cuuint64_t>(kv_heads) * kDim * sizeof(CudaBf16),
      static_cast<cuuint64_t>(kv_len) * kv_heads * kDim * sizeof(CudaBf16)};
  const cuuint32_t box[4] = {64, 1, kAtomN, 1};
  const cuuint32_t element[4] = {1, 1, 1, 1};
  const CUresult status = EncodeTiled()(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), dims,
      strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "fatal: the attention value or key map was refused (%d)\n",
                 static_cast<int>(status));
    std::exit(1);
  }
  return true;
}

//: Whether a shipped run lets the tile take a shape the family would have
//: taken. **A measurement promotes it and not a build.**
//:
//: The rule is the family's own: an arm takes a shape when a run of its report
//: beats the family on every production row, at one protocol and one revision.
//: The tile's floor is what decides that, and section 9.9 of Part 8 states it:
//: a block of two key tiles is a latency and not a rate, so every staging and
//: store path of this file moves sixteen bytes an access.
//:
//: The runs of one session, at one protocol and one revision, read the tile at
//: 0.0215 ms against the family's 0.0426 on `d8_s512`, 0.0190 against 0.0297 on
//: the quarter-context window, 0.0210 against 0.0416 on the grouped-query
//: shape, and 0.1138 against 0.2003 at `b4_s2048_h8_win512`. The tile is
//: therefore ahead on every production row by 1.56 to 1.98 times, and
//: `kPromoted` is what that measurement sets. The absolute gate of the phase,
//: 200 TFLOP/s on the forward row, is still unmet by both paths and is not what
//: this turns on.
constexpr bool kPromoted = true;

//: The measured paths, spelled the way `NANOCHAT_GEMM_LIB` spells the GEMM's.
//: `kAuto` is what a shipped run leaves unset, so a build never changes which
//: tile a step takes; `kHopper` is what a measurement names.
enum class TileChoice { kAuto = 0, kHopper = 1, kFamily = 2 };

TileChoice TileChoiceFromEnv() {
  static const TileChoice choice = [] {
    const char* setting = std::getenv("NANOCHAT_ATTENTION_LIB");
    if (setting == nullptr) return TileChoice::kAuto;
    if (std::strcmp(setting, "hopper") == 0) return TileChoice::kHopper;
    if (std::strcmp(setting, "family") == 0) return TileChoice::kFamily;
    return TileChoice::kAuto;
  }();
  return choice;
}

// Requests the opt-in shared-memory budget of the tile once, then launches it.
void ConfigureHopperSharedMemory() {
  static const bool configured = [] {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(HopperForwardKernel),
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSharedBytes);
    return true;
  }();
  (void)configured;
}

}  // namespace

bool UseHopperAttention(const AttentionParams& p) {
  static const bool ready = [] {
    const Caps caps = GetCaps();
    if (!caps.has_tensor_cores) return false;
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               caps.device_index) != cudaSuccess) {
      return false;
    }
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               caps.device_index) != cudaSuccess) {
      return false;
    }
    // The target compiles `sm_90a` and no other architecture, so a card of
    // another generation keeps the family: the cubin of this build would not
    // load there, and the family is the answer that cannot fail.
    if (major != 9 || minor != 0) return false;
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               caps.device_index) != cudaSuccess) {
      return false;
    }
    return static_cast<std::size_t>(optin) >=
           static_cast<std::size_t>(kSharedBytes);
  }();
  const TileChoice choice = TileChoiceFromEnv();
  if (choice == TileChoice::kFamily) return false;
  if (choice == TileChoice::kAuto && !kPromoted) return false;
  return ready && FusedAttentionHopperCovers(p, /*tensor_cores=*/true);
}

void LaunchHopperAttention(const AttentionParams& params, const ComputeType* q,
                           const ComputeType* k, const ComputeType* v,
                           ComputeType* out, float* stats) {
  ConfigureHopperSharedMemory();
  const int kv_len = params.kv_len > 0 ? params.kv_len : params.seq;
  const int kv_heads = cuda_kernels::KvHeadCountDev(params);
  const CudaBf16* key = reinterpret_cast<const CudaBf16*>(k);
  const CudaBf16* value = reinterpret_cast<const CudaBf16*>(v);

  CUtensorMap key_low{};
  CUtensorMap key_high{};
  CUtensorMap value_low{};
  CUtensorMap value_high{};
  MakePanelMap(&key_low, key, /*panel=*/0, kv_len, kv_heads, params.batch);
  MakePanelMap(&key_high, key, /*panel=*/1, kv_len, kv_heads, params.batch);
  MakePanelMap(&value_low, value, /*panel=*/0, kv_len, kv_heads, params.batch);
  MakePanelMap(&value_high, value, /*panel=*/1, kv_len, kv_heads, params.batch);

  const dim3 grid((params.seq + kBr - 1) / kBr, params.num_heads, params.batch);
  cuda_backend::Launch(HopperForwardKernel, grid, dim3(kThreads), kSharedBytes,
                       params, reinterpret_cast<const CudaBf16*>(q), key_low,
                       key_high, value_low, value_high, out, stats);
}

}  // namespace kernels
}  // namespace nanochat
