// The facts the Hopper attention tile needs before it can be written: the two
// operand maps of the warpgroup instruction, the operand form it reads, and the
// rate it reaches when its key tile is staged by a copy path.
//
// **The form.** The instruction reads each shared-memory operand as a run of
// core matrices, and a core matrix is eight rows of eight elements with the
// rows sixteen bytes apart. A row-major tile is not readable: the instruction
// then reads one element out of every eight along each axis. An earlier version
// of this probe staged row-major tiles and did not notice for three runs,
// because its check compared the **sum** of a product, and that quantity is
// invariant under the row and column permutation a wrong form induces. Every
// check below therefore compares one element at a time.
//
// **The accumulator map.** The accumulator of `wgmma.mma_async.m64n128k16` is
// 64 by 128 fp32 values over 128 threads, so each thread holds 64. This file
// measures which (row, column) each of those 64 registers holds, in two ways,
// and prints both: a one-hot query row against a one-hot key row says where one
// element lands, and two separable patterns decode the whole table so that a
// reader can check the map against the counts. The published row and column
// formulas of Part 8 section 5 are then checked against that decode.
//
// **The register operand.** The value product takes the probability tile as its
// first operand in registers, and that element order is not a descriptor. The
// same one-hot method measures it, and the order it reports is what decides
// whether the probability of a score tile reaches the value product from the
// score accumulator of the same lane with no shuffle.
//
// **The transposed value operand.** The value product's second transpose bit is
// the one operand form Part 8 section 5 records as unmeasured, because its long
// axis is the head width and not the key index. A one-hot probability element
// against a one-hot value element leaves one result, and that result names the
// form and the atom stride that read the staged value tile.
//
// **The staged rows.** A real attention tile reads a new key tile every step,
// so the last section times the same score product three ways: from a resident
// tile, from a key tile that a copy path stages into a ring, and from the
// copies alone. The rate of such a row is the bandwidth of its copy path times
// the arithmetic that one staged byte buys, so the rows sweep that arithmetic:
// the warpgroups of one block and the query tiles of each, from one to two and
// one to three, which state 64 to 384 FLOP a staged byte. The copy rows sweep
// the depth of the ring, the thread count, and two orders of the staged stores.
// The staged product is compared with the resident product before it is timed,
// so a wrong staging fails instead of measuring. The query tiles of every
// warpgroup share one ring, so a block of two warpgroups doubles the arithmetic
// of one copy and, with it, the rate the copies allow.
//
// **One binary, three row sets, one document.** The resident section carries
// the ceiling rows of the Phase 0 probe this file absorbed, so one build and
// one card job replace two. The `attention-wgmma:` row names mark the H1 rows
// of docs/attention.md Part 8, the `wgmma:` row names mark the Phase 0 rows of
// docs/kernels.md Part 3, and the `attention-flash:` row names mark the flash
// tile of the H2 prototype. `--rows phase0|h1|flash|all` selects the set, `all`
// is the default, and the row names do not move with the selection, so every
// baselines entry stays reproducible from one run. The first two sets differ in
// pass depth, eight atoms against four, and in occupancy, one warpgroup per SM
// against two, which is why neither divides into the other.
//
// **The flash rows.** A real attention tile reads a new key block and a new
// value block every step, so the staged rows above price the score product
// alone. The flash rows price the whole tile instead: one resident query block,
// the score product, the base-two softmax in the registers, the value product
// from the probability operand the score accumulator already holds in the lane
// that produced it, and the running maximum, sum, and rescale. The row that
// carries no copy path is `attention-flash:resident`, which is the softmax and
// the second product priced against the score-only ceiling of
// `attention-wgmma:resident`. Every row is checked against the host reference
// of the family, `seqref::AttentionForward`, one element at a time, so a wrong
// probability operand or a wrong value form fails before anything is timed.
// That check is also the first end-to-end test of the register operand: the H1
// rows measured its element order with one-hot elements and never carried a
// full product through it.
//
// The tile is a chain and not a pipeline. The score product must retire before
// the softmax can read it, and the softmax must finish before the value product
// can issue, so a flash row states the cost of that chain. The overlap of
// decision H4 is what later hides it. The mask, the staged ring, and the
// transposed value staging are the phases that follow this one.
//
// **The report.** Every timed row goes through `BenchReport`, so the program
// states one `nanochat.bench.v3` document with `--json`. The lines this program
// used to print are diagnostics: they stay on stdout for a human run and are
// dropped under `--json`, so the document stays pipeable.
//
// **The metric.** The rate is the necessary work over the elapsed time, in
// GFLOP/s, and `mfu` is that rate over the card's declared bf16 peak. That pair
// is the metric of every row: no row divides by a peak by hand, and the report
// states the denominator it used. `AddTiming` of `bench_protocol.h` writes two
// rows from one measurement, the minimum under the row's own name and the
// median beside it, so a Phase 0 or H1 gate reads the statistic it named.
//
// **The protocol.** The timing is the one of
// `backends/cuda/kernels/testing/bench_protocol.h`, which
// `backends/cuda/gemm/gemm_benchmark.cc` also uses, so a row of this program
// and a row of that one state the same statistic. One call is one launch.
// `--iters` calls fill one window and `--rounds` windows give the samples.
// A flush of the second-level cache runs before every timed call, because a
// staged window of 512 key tiles is 16 MiB and stays warm across calls. The
// flush cost is measured once and removed from the window.
//
// The two numbers the card's peak does not state stay beside their row as
// extras. The per-SM share is the row's own rate over the peak the report
// states, divided by the multiprocessor count, so it cannot disagree with the
// row it sits beside. A copy row issues no arithmetic, so it states no rate and
// the bytes per second is its extra.
//
// The target states `archs = ["sm_90a"]` for itself, so no command line carries
// a flag, and the tripwire below refuses a build that names another
// architecture. The emulation engine maps no warpgroup instruction, so the
// probe is device only: it carries the `gpu` and `manual` tags and its gate is
// the H100 leg.
//
//   modal run tools/modal/app.py::card_job --name attention_wgmma

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "backends/cuda/kernels/testing/bench_protocol.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "nanochat/bench.h"
#include "nanochat/device_profile.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error \
    "dev/kernels/wgmma_bench.cu needs sm_90a; its target states archs = [\"sm_90a\"]"
#endif

namespace {

using Bf16 = __nv_bfloat16;

// --- The tile ---------------------------------------------------------------

//: Threads of one warpgroup, which is the width the instruction spans.
constexpr int kWarpgroupThreads = 128;

//: Query rows, key rows, and the head width: the `m`, `n`, and `k` extents of
//: one score product.
constexpr int kAtomM = 64;
constexpr int kAtomN = 128;
constexpr int kDim = 128;
//: The `k` extent of one atom, the atoms of the whole tile, and the registers
//: one thread holds.
constexpr int kAtomK = 16;
constexpr int kAtoms = kDim / kAtomK;
constexpr int kRegisters = kAtomM * kAtomN / kWarpgroupThreads;

//: The pass depth of the Phase 0 rows of docs/kernels.md Part 3, in atoms of
//: sixteen. Eight atoms is the head width and the attention geometry; four is
//: the 64-deep pass whose gate is stated against the per-SM share of the card
//: peak on a one-block row. Both depths live here so that one binary reproduces
//: every recorded row of both parts.
constexpr int kHalfAtoms = kAtoms / 2;

//: The key blocks of one tile: one atom a block, so the value product walks the
//: same eight and a key or value tile of one step is `kKeyBytes` of operand.
constexpr int kKeyAtoms = kAtomN / kAtomK;

//: The interleaved core-matrix form. One core matrix is eight rows of eight
//: elements, sixteen bytes to a row, so the two halves of the `k` axis of one
//: row group are one leading offset apart and the next row group follows at one
//: stride offset.
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 2 * kLboBytes;

//: The ring of staged key tiles, and the warpgroups and query tiles that share
//: one staged key tile. Two query tiles is the geometry of Part 8 section 3,
//: and every warpgroup of the block shares the ring, so the arithmetic one
//: staged byte buys is `64 * warpgroups * query_tiles` FLOP: a staged key
//: element is reused by one query row for each of those rows.
constexpr int kStages = 4;
constexpr int kMaxWarpgroups = 2;
constexpr int kMaxQueryTiles = 3;
//: The query tiles one block holds at the deepest geometry.
constexpr int kMaxTiles = kMaxWarpgroups * kMaxQueryTiles;

constexpr int kQueryBytes = kAtomM * kDim * sizeof(Bf16);
constexpr int kKeyBytes = kAtomN * kDim * sizeof(Bf16);
//: A value tile is a key tile of the same shape, and the flash tile stages
//: both.
constexpr int kValueBytes = kAtomN * kDim * sizeof(Bf16);

//: The staged shared memory of one geometry: the query tile of every
//: warpgroup, then the ring of key tiles that all of them read.
constexpr int StagedBytes(int warpgroups, int query_tiles, int stages) {
  return warpgroups * query_tiles * kQueryBytes + stages * kKeyBytes;
}

//: The largest dynamic shared-memory request the card accepts, which one block
//: per multiprocessor must stay under.
constexpr int kSmemCeiling = 227 * 1024;
static_assert(StagedBytes(kMaxWarpgroups, kMaxQueryTiles, kStages) <=
                  kSmemCeiling,
              "the staged ring must fit one block per multiprocessor");

//: The shared region of the resident flash tile: one query tile a warpgroup,
//: one key tile, and one value tile. The set is the first step towards the ring
//: of docs/attention.md Part 8 section 3, which adds
//: `stages * (kKeyBytes + kValueBytes)` beside the same query region.
constexpr int FlashResidentBytes(int warpgroups) {
  return warpgroups * kQueryBytes + kKeyBytes + kValueBytes;
}
static_assert(FlashResidentBytes(kMaxWarpgroups) <= kSmemCeiling,
              "the resident flash tile must fit one block per multiprocessor");

//: The base-two form of the exponential. `exp2` is one instruction where `exp`
//: is a range reduction around it, so the softmax scale carries `log2(e)` and
//: every row maximum this kernel states is in base-two units.
constexpr float kLog2e = 1.4426950408889634f;

//: The identity of a row maximum and the sentinel of the running one. It is a
//: finite value, so the first rescale of a row underflows to zero instead of
//: forming an infinity.
constexpr float kFlashNegInf = -3.402823466e38f;

//: The softmax scale of the reference shape of the check, which is the one
//: `AttentionParams` states when it carries no scale of its own.
constexpr float kFlashScale = 0.08838834764831845f;

//: One sixteen-byte group of a row-major source is the unit of both staging
//: paths: the interleaved form keeps eight adjacent elements of one `k` block
//: of one row adjacent, so one group of a source row is one store of a stage.
constexpr int kElementsPerGroup = 8;
constexpr int kGroupsPerRow = kDim / kElementsPerGroup;

//: One `k` step of a descriptor, in sixteen-byte units, for a tile of `rows`.
//: The two operands differ because their tiles have different heights.
__host__ __device__ constexpr int AtomStep(int rows) {
  return (rows / 8) * kSboBytes / 16;
}
static_assert(kAtoms * AtomStep(kAtomM) * 16 == kQueryBytes,
              "the query tile must fill its interleaved form");
static_assert(kAtoms * AtomStep(kAtomN) * 16 == kKeyBytes,
              "the key tile must fill its interleaved form");

//: The row and the column of one accumulator slot of `m64n128k16`, from the map
//: Part 8 section 5 publishes and this file measured one element at a time. Two
//: readers use them: the host decodes a slot of a read-back tile, and the flash
//: kernel reduces a row over the four lanes of the quad that hold it.
__host__ __device__ constexpr int SlotRow(int lane, int index) {
  return 16 * (lane / 32) + (lane % 32) / 4 + 8 * ((index / 2) % 2);
}
__host__ __device__ constexpr int SlotColumn(int lane, int index) {
  return 8 * (index / 4) + 2 * (lane % 4) + (index % 2);
}

//: The reference peak of `backends/cuda/caps.json` for the h100 entry. It is
//: the fallback denominator of a card the table does not name, and no share is
//: computed from it: every share divides by the peak the report states.
constexpr const char* kReferenceDevice = "h100";

//: The necessary work of one score product over `atoms` atoms of sixteen k
//: steps. One function, so a row cannot state a count that disagrees with the
//: instruction loop it timed.
constexpr double FlopsPerPass(int atoms) {
  return 2.0 * kAtomM * kAtomN * kAtomK * atoms;
}

//: The slots one product leaves, over the warpgroup.
constexpr std::size_t kProductSlots =
    static_cast<std::size_t>(kWarpgroupThreads) * kRegisters;

// --- The staging ------------------------------------------------------------

// The interleaved offset of one element of a row-major source tile.
__device__ __forceinline__ int InterleavedOffset(int row, int k, int rows) {
  const int block = k / kAtomK;
  const int half = (k % kAtomK) / kElementsPerGroup;
  const int element = k % kElementsPerGroup;
  const int group = row / 8;
  const int within_row = row % 8;
  return block * AtomStep(rows) * 16 + group * kSboBytes + half * kLboBytes +
         within_row * 16 + element * 2;
}

// Stages one operand of the whole tile element by element. This is the resident
// path and the query path; the staged key tiles use the copy engine below. The
// stride is the thread count of the caller, because a block of two warpgroups
// stages its shared region with both of them.
__device__ __forceinline__ void StageInterleaved(char* stage,
                                                 const Bf16* source, int rows,
                                                 int threads) {
  const int total = rows * kDim;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += threads) {
    const int row = index / kDim;
    const int k = index - row * kDim;
    *reinterpret_cast<Bf16*>(stage + InterleavedOffset(row, k, rows)) =
        source[index];
  }
}

// One sixteen-byte copy from global memory into shared memory, in flight until
// the group that holds it completes.
__device__ __forceinline__ void CopyAsync16(void* destination,
                                            const void* source) {
  const std::uint32_t shared =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared),
               "l"(source));
}

__device__ __forceinline__ void CopyCommit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Waits until at most `Pending` copy groups of this thread are still in flight.
template <int Pending>
__device__ __forceinline__ void CopyWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending) : "memory");
}

// Stages one key tile with the copy engine, in the interleaved form: one group
// of a source row is one sixteen-byte copy, so the staged form costs exactly
// the bytes of the source and no shuffle.
//
// `row_fast` picks which field of an element the adjacent threads vary, and the
// two orders are a bank-conflict question. The interleaved form puts the eight
// rows of a core matrix sixteen bytes apart, so an order that varies the row of
// that matrix makes sixteen lanes write one run of a hundred and twenty-eight
// bytes. The other order varies the `k` block first, whose stride is a multiple
// of the shared-memory period, so sixteen lanes then share four banks. The copy
// rows measure both, and they agree to within two percent, so the store order
// is not what limits this path.
//
// `threads` is the thread count of the block: two warpgroups issue the copies
// of one key tile together, so a thread of either warpgroup copies half of what
// a single warpgroup copies.
__device__ __forceinline__ void StageCopy(char* stage, const Bf16* source,
                                          bool row_fast, int threads) {
  const int groups = kAtomN * kGroupsPerRow;
  for (int unit = static_cast<int>(threadIdx.x); unit < groups;
       unit += threads) {
    int row = 0;
    int k = 0;
    if (row_fast) {
      // One unit is (row group, k block, half, row of the matrix), and the row
      // of the matrix varies fastest.
      const int matrix = unit / (kGroupsPerRow * 8);
      const int within = unit - matrix * (kGroupsPerRow * 8);
      const int block = within / (2 * 8);
      const int rest = within - block * (2 * 8);
      const int half = rest / 8;
      row = matrix * 8 + (rest - half * 8);
      k = block * kAtomK + half * kElementsPerGroup;
    } else {
      row = unit / kGroupsPerRow;
      k = (unit - row * kGroupsPerRow) * kElementsPerGroup;
    }
    CopyAsync16(stage + InterleavedOffset(row, k, kAtomN),
                source + row * kDim + k);
  }
}

// Stages the value tile transposed, as a tile of `dim` rows by `kAtomN` keys.
// The value product wants `D = P V`, whose `k` axis is the key index, and the
// proven operand form is the interleaved one with the transpose bits clear. So
// the transpose happens at the copy and not in the instruction: one thread
// reads eight keys of one head-width row, which is coalesced across the warp,
// and writes the sixteen bytes that the interleaved form keeps adjacent.
//
// This stage keeps plain stores where `StageCopy` uses the copy engine, because
// a transposition is not a contiguous sixteen-byte move and no path copies one.
// `threads` is the thread count of the block, as in `StageInterleaved`.
__device__ __forceinline__ void StageTransposed(char* stage, const Bf16* source,
                                                int threads) {
  constexpr int kKeyGroups = kAtomN / kElementsPerGroup;
  for (int unit = static_cast<int>(threadIdx.x); unit < kKeyGroups * kDim;
       unit += threads) {
    const int key = (unit / kDim) * kElementsPerGroup;
    const int row = unit % kDim;
    Bf16 packed[kElementsPerGroup];
#pragma unroll
    for (int j = 0; j < kElementsPerGroup; ++j) {
      packed[j] = source[(key + j) * kDim + row];
    }
    *reinterpret_cast<uint4*>(stage + InterleavedOffset(row, key, kDim)) =
        *reinterpret_cast<uint4*>(packed);
  }
}

// --- The descriptor ---------------------------------------------------------

// The six fields of the shared-memory matrix descriptor: the address in sixteen
// byte units, the leading offset, the stride offset, base offset zero, and
// layout type zero for the interleaved form.
__device__ __forceinline__ std::uint64_t MakeDescriptor(const void* base) {
  const std::uint32_t address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(base));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((address >> 4) & 0x3FFFu);
  desc |= static_cast<std::uint64_t>((kLboBytes >> 4) & 0x3FFFu) << 16;
  desc |= static_cast<std::uint64_t>((kSboBytes >> 4) & 0x3FFFu) << 32;
  return desc;
}

// --- The instruction --------------------------------------------------------

// D += A * B^T over one 64 by 128 by 16 atom. Both operands come from shared
// memory, and the scale-D predicate is false for the first atom of a pass.
__device__ __forceinline__ void WgmmaM64N128K16(std::uint64_t desc_a,
                                                std::uint64_t desc_b, float* d,
                                                bool accumulate) {
  const std::uint32_t scale = accumulate ? 1u : 0u;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %66, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31, "
      "   %32, %33, %34, %35, %36, %37, %38, %39, "
      "   %40, %41, %42, %43, %44, %45, %46, %47, "
      "   %48, %49, %50, %51, %52, %53, %54, %55, "
      "   %56, %57, %58, %59, %60, %61, %62, %63}, "
      "  %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),
        "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
        "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
        "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale));
}

// The same atom as `WgmmaM64N128K16` with the second transpose bit set, which
// is the form the value product needs: its `k` axis is the key index and its
// `n` axis is the head width, so the staged value tile of `[key][dim]` rows is
// the `k`-major operand. Part 8 section 5 records this form as unmeasured, so
// the section below measures it one element at a time.
__device__ __forceinline__ void WgmmaM64N128K16TransposedB(std::uint64_t desc_a,
                                                           std::uint64_t desc_b,
                                                           float* d,
                                                           bool accumulate) {
  const std::uint32_t scale = accumulate ? 1u : 0u;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %66, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31, "
      "   %32, %33, %34, %35, %36, %37, %38, %39, "
      "   %40, %41, %42, %43, %44, %45, %46, %47, "
      "   %48, %49, %50, %51, %52, %53, %54, %55, "
      "   %56, %57, %58, %59, %60, %61, %62, %63}, "
      "  %64, %65, p, 1, 1, 0, 1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),
        "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
        "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
        "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale));
}

__device__ __forceinline__ void WgmmaFence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// The same atom with the first operand in registers, which is the form the
// value product needs for the probability tile: four `b16x2` registers a
// thread, eight elements over the warpgroup, sixty-four query rows by sixteen
// keys. This form has no descriptor for the first operand, so its element order
// is a separate fact and the section below measures it.
__device__ __forceinline__ void WgmmaM64N128K16OperandA(
    const std::uint32_t (&a)[4], std::uint64_t desc_b, float* d) {
  asm volatile(
      "{\n"
      "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31, "
      "   %32, %33, %34, %35, %36, %37, %38, %39, "
      "   %40, %41, %42, %43, %44, %45, %46, %47, "
      "   %48, %49, %50, %51, %52, %53, %54, %55, "
      "   %56, %57, %58, %59, %60, %61, %62, %63}, "
      "  {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),
        "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
        "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
        "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b));
}

template <int Count>
__device__ __forceinline__ void WgmmaFenceOperand(float* d) {
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    asm volatile("" : "+f"(d[i]) : : "memory");
  }
}

// The same fence over the packed `b16x2` registers of a probability operand.
// The value product reads them through the async proxy, so the region needs the
// same bound the accumulator fence gives the fp32 slots of the score tile.
template <int Count>
__device__ __forceinline__ void WgmmaFencePacked(std::uint32_t* a) {
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    asm volatile("" : "+r"(a[i]) : : "memory");
  }
}

__device__ __forceinline__ void WgmmaCommit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void WgmmaWait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(Pending) : "memory");
}

__device__ __forceinline__ void FenceAsyncShared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// --- The kernels ------------------------------------------------------------

// One warpgroup, one resident query tile and one resident key tile in the
// interleaved form, and `passes` score products over them. `Atoms` is the k
// extent of one pass: `kAtoms` for the attention geometry, `kHalfAtoms` for the
// Phase 0 geometry. The accumulator is 64 by 128 whatever the pass depth is, so
// only the loop bound and the flop count follow `Atoms`. The registers leave
// raw, and every reader below decodes them.
template <int Atoms = kAtoms>
__global__ void ResidentKernel(const Bf16* __restrict__ q,
                               const Bf16* __restrict__ k,
                               float* __restrict__ out, int passes) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* key = smem + kQueryBytes;
  StageInterleaved(query, q, kAtomM, kWarpgroupThreads);
  StageInterleaved(key, k, kAtomN, kWarpgroupThreads);
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_query = MakeDescriptor(query);
  const std::uint64_t base_key = MakeDescriptor(key);
  float d[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(d);

  for (int pass = 0; pass < passes; ++pass) {
#pragma unroll
    for (int atom = 0; atom < Atoms; ++atom) {
      WgmmaM64N128K16(base_query + atom * AtomStep(kAtomM),
                      base_key + atom * AtomStep(kAtomN), d,
                      pass != 0 || atom != 0);
    }
    WgmmaCommit();
    WgmmaWait<1>();
  }
  WgmmaWait<0>();
  WgmmaFenceOperand<kRegisters>(d);

  float* destination = out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = d[i];
}

// The flash tile with no copy path: `Warpgroups` warpgroups of one resident
// query tile each, one resident key tile, one resident value tile, and the
// chain of docs/attention.md Part 8 section 3. Each step issues the score
// product over the whole head width, carries the result through the base-two
// softmax in the registers, packs the probability operand out of the score
// slots of the same lane, and issues the value product against the transposed
// value tile. The running maximum, the running sum, and the rescale of the
// output accumulator carry from one step to the next, so `tiles` steps are one
// online softmax and not `tiles` independent products.
//
// The score accumulator and the output accumulator are both 64 by 128 over the
// same lane map, so the two share `SlotRow` and `SlotColumn`. The probability
// operand of atom `atom` and register `reg` holds the score slots
// `8 * atom + 4 * (reg / 2) + 2 * (reg % 2) + half`, which is the measured
// register map written the other way round: a register holds the keys
// `8 * (reg / 2) + 2 * (lane % 4) + half`, and the score slot of a key is
// `4 * (key / 8) + 2 * ((row / 8) % 2) + (key % 2)`.
//
// The step is a chain: the score product retires before the softmax reads it,
// and the softmax finishes before the value product issues. The value group is
// drained at the end of a step because the next step rescales the same output
// registers, so a row of this kernel states the cost of the serialized chain
// and not of an overlapped one. Decision H4 of Part 8 section 2 is what later
// hides it.
//
// The kernel leaves the output unnormalised. One division by the row sum is a
// pass over the output at the end of a real block and not part of a step, so a
// timed row prices the step alone and the check applies the division itself.
template <int Warpgroups>
__global__ void ResidentFlashKernel(const Bf16* __restrict__ q,
                                    const Bf16* __restrict__ k,
                                    const Bf16* __restrict__ v,
                                    float* __restrict__ out,
                                    float* __restrict__ row_sums, int tiles,
                                    float softmax_scale) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* key = query + Warpgroups * kQueryBytes;
  char* value = key + kKeyBytes;

  const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;

  // Every thread of the block stages the whole region, as the staged kernel
  // does, because the barrier below is the only one before the loop.
  for (int tile = 0; tile < Warpgroups; ++tile) {
    StageInterleaved(query + tile * kQueryBytes,
                     q + static_cast<std::size_t>(tile) * kAtomM * kDim, kAtomM,
                     kThreads);
  }
  StageInterleaved(key, k, kAtomN, kThreads);
  StageTransposed(value, v, kThreads);
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_query =
      MakeDescriptor(query + warpgroup * kQueryBytes);
  const std::uint64_t base_key = MakeDescriptor(key);
  const std::uint64_t base_value = MakeDescriptor(value);

  // The online state of the two rows this thread holds. A row of the tile is
  // the four lanes of one quad, so both reductions below are two shuffles and
  // never cross a warp.
  float running_max[2] = {kFlashNegInf, kFlashNegInf};
  float running_sum[2] = {0.0f, 0.0f};

  float output[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) output[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(output);

  const float log2_scale = softmax_scale * kLog2e;
  std::uint32_t probability[kKeyAtoms][4];

  for (int tile = 0; tile < tiles; ++tile) {
    // The score product over the whole head width. The first atom overwrites,
    // so the accumulator carries no score of the step before it.
    float score[kRegisters];
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) score[i] = 0.0f;
    WgmmaFenceOperand<kRegisters>(score);
    WgmmaFence();
#pragma unroll
    for (int atom = 0; atom < kAtoms; ++atom) {
      WgmmaM64N128K16(base_query + atom * AtomStep(kAtomM),
                      base_key + atom * AtomStep(kAtomN), score, atom != 0);
    }
    WgmmaCommit();
    WgmmaWait<0>();
    WgmmaFenceOperand<kRegisters>(score);

    // The scaled score in base-two units, then the row maximum: the thread's
    // own thirty-two columns of each of its two rows, then the quad.
    float tile_max[2] = {kFlashNegInf, kFlashNegInf};
    float new_max[2];
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) {
      score[i] *= log2_scale;
      const int slot = (i / 2) % 2;
      tile_max[slot] = fmaxf(tile_max[slot], score[i]);
    }
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      tile_max[slot] = fmaxf(tile_max[slot],
                             __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 1));
      tile_max[slot] = fmaxf(tile_max[slot],
                             __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 2));
      new_max[slot] = fmaxf(running_max[slot], tile_max[slot]);
    }

    // The exponentials and the row sum, in the same two reductions. The rescale
    // of the running state is the factor the output accumulator needs as well.
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) {
      const int slot = (i / 2) % 2;
      score[i] = exp2f(score[i] - new_max[slot]);
      tile_sum[slot] += score[i];
    }
    float rescale[2];
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 1);
      tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 2);
      rescale[slot] = exp2f(running_max[slot] - new_max[slot]);
      running_sum[slot] = running_sum[slot] * rescale[slot] + tile_sum[slot];
      running_max[slot] = new_max[slot];
    }

    // The output accumulator holds the earlier steps under the earlier maximum,
    // so every slot is rescaled before this step's value product adds to it.
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) {
      output[i] *= rescale[(i / 2) % 2];
    }

    // The probability operand, two keys a register from the score slots of this
    // same lane, low half of the register first.
#pragma unroll
    for (int atom = 0; atom < kKeyAtoms; ++atom) {
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int index = 8 * atom + 4 * (reg / 2) + 2 * (reg % 2);
        const std::uint32_t low = static_cast<std::uint32_t>(
            __bfloat16_as_ushort(__float2bfloat16_rn(score[index])));
        const std::uint32_t high = static_cast<std::uint32_t>(
            __bfloat16_as_ushort(__float2bfloat16_rn(score[index + 1])));
        probability[atom][reg] = low | (high << 16);
      }
    }

    // The value product. The first operand is in registers, so it needs the
    // operand fence in place of a descriptor, and the value tile is the
    // transposed staging with both transpose bits clear.
    WgmmaFencePacked<kKeyAtoms * 4>(&probability[0][0]);
    WgmmaFence();
#pragma unroll
    for (int atom = 0; atom < kKeyAtoms; ++atom) {
      WgmmaM64N128K16OperandA(probability[atom],
                              base_value + atom * AtomStep(kDim), output);
    }
    WgmmaCommit();
    WgmmaWait<0>();
  }

  WgmmaFenceOperand<kRegisters>(output);
  float* destination = out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = output[i];

  // The row sums, one write a row: the four lanes of a quad agree, so the first
  // of them states it. A timed row passes no buffer and pays nothing.
  if (row_sums != nullptr && (lane % 4) == 0) {
    const int row = 16 * (lane / 32) + (lane % 32) / 4;
    row_sums[warpgroup * kAtomM + row] = running_sum[0];
    row_sums[warpgroup * kAtomM + row + 8] = running_sum[1];
  }
}

//: The two experiment switches of the staged kernel. `kPlainStage` stages the
//: ring with plain stores instead of the asynchronous copies, so a wrong staged
//: product separates into a wrong copy and a wrong ring. `kRowFast` picks the
//: store order of the bank-conflict note on `StageCopy`.
constexpr int kPlainStage = 1;
constexpr int kRowFast = 2;

// One block of `Warpgroups` warpgroups, `QueryTiles` resident query tiles for
// each of them, and a ring of `Stages` key tiles that the copy engine fills.
// The ring is one region of the block: a warpgroup reads the tile that every
// other warpgroup reads, so the copies of one key tile are shared and the
// arithmetic that one staged byte buys is multiplied by the warpgroup count.
//
// `kProducts` false is the copy row: the same bytes, the same ring, and no
// instruction.
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages>
__global__ void StagedKernel(const Bf16* __restrict__ q,
                             const Bf16* __restrict__ k, int source_tiles,
                             float* __restrict__ out, int tiles, int options) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  constexpr int kTiles = Warpgroups * QueryTiles;
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* stages = smem + kTiles * kQueryBytes;

  // The warpgroup owns the tile it multiplies: its `QueryTiles` tiles are one
  // run of the query region, because the descriptor of a tile is its address.
  const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;

  // Every thread of the block stages the whole query region, because every
  // warpgroup reads it once the barrier below opens.
  for (int tile = 0; tile < kTiles; ++tile) {
    StageInterleaved(query + tile * kQueryBytes,
                     q + static_cast<std::size_t>(tile) * kAtomM * kDim, kAtomM,
                     kThreads);
  }
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  std::uint64_t base_query[QueryTiles];
#pragma unroll
  for (int tile = 0; tile < QueryTiles; ++tile) {
    base_query[tile] =
        MakeDescriptor(query + (warpgroup * QueryTiles + tile) * kQueryBytes);
  }

  float d[QueryTiles * kRegisters];
  if constexpr (kProducts) {
#pragma unroll
    for (int i = 0; i < QueryTiles * kRegisters; ++i) d[i] = 0.0f;
    WgmmaFenceOperand<QueryTiles * kRegisters>(d);
  }

  const bool plain_stage = (options & kPlainStage) != 0;
  const bool row_fast = (options & kRowFast) != 0;

  // The prologue fills all but one stage, so the first product waits on a copy
  // that is already in flight. It fills the stage even when the run is shorter
  // than the ring: the wait below counts committed groups, so a short prologue
  // would let the first product read a tile that is still in flight. The extra
  // copies read the source ring, so they are harmless.
  for (int tile = 0; tile < Stages - 1; ++tile) {
    const Bf16* source =
        k + static_cast<std::size_t>(tile % source_tiles) * kAtomN * kDim;
    if (plain_stage) {
      StageInterleaved(stages + tile * kKeyBytes, source, kAtomN, kThreads);
    } else {
      StageCopy(stages + tile * kKeyBytes, source, row_fast, kThreads);
    }
    CopyCommit();
  }

  for (int tile = 0; tile < tiles; ++tile) {
    // Stages - 1 groups are in flight here, so waiting for all but Stages - 2
    // of them completes this tile's copy and no later one.
    CopyWait<Stages - 2>();
    __syncthreads();
    FenceAsyncShared();

    if constexpr (kProducts) {
      const std::uint64_t base_key =
          MakeDescriptor(stages + (tile % Stages) * kKeyBytes);
#pragma unroll
      for (int slot = 0; slot < QueryTiles; ++slot) {
#pragma unroll
        for (int atom = 0; atom < kAtoms; ++atom) {
          WgmmaM64N128K16(base_query[slot] + atom * AtomStep(kAtomM),
                          base_key + atom * AtomStep(kAtomN),
                          d + slot * kRegisters, true);
        }
      }
      WgmmaCommit();
      // One group stays in flight, so the next tile's product starts before
      // this one drains; the wait still retires the previous tile's reads of
      // the stage that the next copy fills.
      WgmmaWait<1>();
    }

    // The copy that refills the stage this tile occupies is issued only after
    // the previous tile's product has retired, because that stage is the one it
    // read.
    const int next = tile + Stages - 1;
    if (next < tiles) {
      const Bf16* source =
          k + static_cast<std::size_t>(next % source_tiles) * kAtomN * kDim;
      if (plain_stage) {
        StageInterleaved(stages + (next % Stages) * kKeyBytes, source, kAtomN,
                         kThreads);
      } else {
        StageCopy(stages + (next % Stages) * kKeyBytes, source, row_fast,
                  kThreads);
      }
    }
    CopyCommit();
  }

  if constexpr (kProducts) {
    WgmmaWait<0>();
    WgmmaFenceOperand<QueryTiles * kRegisters>(d);
    const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;
#pragma unroll
    for (int slot = 0; slot < QueryTiles; ++slot) {
      // One tile of the output per query tile of the block, in tile order, so
      // the caller reads tile `warpgroup * QueryTiles + slot`.
      float* destination =
          out + (static_cast<std::size_t>(warpgroup * QueryTiles + slot) *
                     kWarpgroupThreads +
                 static_cast<std::size_t>(lane)) *
                    kRegisters;
#pragma unroll
      for (int i = 0; i < kRegisters; ++i) {
        destination[i] = d[slot * kRegisters + i];
      }
    }
  }
}

// One warpgroup and a staged key tile, with the first operand in registers: one
// element of one thread's eight is set, so the single non-zero result names the
// query row and the key index of that element. The key tile holds 2^j in
// dimension j of row zero and zero elsewhere, so the result of the one-hot
// element is 2^j and the two fields of the element separate in one launch.
__global__ void RegisterOperandKernel(const Bf16* __restrict__ k,
                                      float* __restrict__ out, int lane,
                                      int reg, int half) {
  extern __shared__ __align__(1024) char smem[];
  char* key = smem;
  StageInterleaved(key, k, kAtomN, kWarpgroupThreads);
  __syncthreads();
  FenceAsyncShared();

  std::uint32_t a[4] = {0u, 0u, 0u, 0u};
  if (static_cast<int>(threadIdx.x) == lane) {
    constexpr std::uint32_t kOne = 0x3F80u;  // bf16 1.0
    a[reg] = half == 0 ? kOne : (kOne << 16);
  }
  float d[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(d);
  WgmmaFence();

  WgmmaM64N128K16OperandA(a, MakeDescriptor(key), d);
  WgmmaCommit();
  WgmmaWait<0>();
  WgmmaFenceOperand<kRegisters>(d);

  float* destination = out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = d[i];
}

// One warpgroup, the value product: the probability tile as the first operand
// and the staged value tile as the second, with its transpose bit under test.
// `atom_step` is the sixteen-byte stride between two atoms along the key axis,
// which is the row axis of the staged value tile; the caller sweeps it, because
// that offset is what the transposed form changes.
__global__ void ValueOperandKernel(const Bf16* __restrict__ p,
                                   const Bf16* __restrict__ v, int atom_step,
                                   int trans_b, int transpose_stage,
                                   float* __restrict__ out) {
  extern __shared__ __align__(1024) char smem[];
  char* probability = smem;
  char* value = smem + kQueryBytes;
  StageInterleaved(probability, p, kAtomM, kWarpgroupThreads);
  if (transpose_stage != 0) {
    StageTransposed(value, v, kWarpgroupThreads);
  } else {
    StageInterleaved(value, v, kAtomN, kWarpgroupThreads);
  }
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_probability = MakeDescriptor(probability);
  const std::uint64_t base_value = MakeDescriptor(value);
  float d[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(d);

  for (int atom = 0; atom < kKeyAtoms; ++atom) {
    const std::uint64_t desc_p = base_probability + atom * AtomStep(kAtomM);
    const std::uint64_t desc_v = base_value + atom * atom_step;
    if (trans_b != 0) {
      WgmmaM64N128K16TransposedB(desc_p, desc_v, d, atom != 0);
    } else {
      WgmmaM64N128K16(desc_p, desc_v, d, atom != 0);
    }
  }
  WgmmaCommit();
  WgmmaWait<0>();
  WgmmaFenceOperand<kRegisters>(d);

  float* destination = out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = d[i];
}

// --- The host side ----------------------------------------------------------

std::vector<float> ReadBack(const float* device, std::size_t count) {
  std::vector<float> host(count, 0.0f);
  cudaMemcpy(host.data(), device, count * sizeof(float),
             cudaMemcpyDeviceToHost);
  return host;
}

// The published map of Part 8 section 5: which (row, column) the accumulator
// slot of one lane and one register holds.
std::pair<int, int> SlotOf(int lane, int index) {
  return {SlotRow(lane, index), SlotColumn(lane, index)};
}

// The measured register-operand formula, the counterpart of `SlotOf`: which
// (row, key) of a 64 by 16 tile one element of one thread's eight holds.
std::pair<int, int> RegisterSlotOf(int lane, int reg, int half) {
  const int row = 16 * (lane / 32) + (lane % 32) / 4 + 8 * (reg % 2);
  const int key = 8 * (reg / 2) + 2 * (lane % 4) + half;
  return {row, key};
}

// The result of one resident pass, for the checks below.
std::vector<float> RunResident(const Bf16* device_q, const Bf16* device_k,
                               float* device_out) {
  ResidentKernel<kAtoms><<<1, kWarpgroupThreads, kQueryBytes + kKeyBytes>>>(
      device_q, device_k, device_out, 1);
  cudaDeviceSynchronize();
  return ReadBack(device_out, kProductSlots);
}

// The one opt-in dynamic shared-memory request, stated for each instantiation
// the file launches. A templated kernel has no single function address, so the
// attribute is set once per pass depth rather than once per file.
template <int Atoms>
void SetResidentSharedMemory() {
  cudaFuncSetAttribute(reinterpret_cast<const void*>(ResidentKernel<Atoms>),
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kQueryBytes + kKeyBytes);
}

// One launch of the resident kernel: `blocks` warpgroups, `Atoms` atoms per
// pass, and `steps` passes of the loop inside the kernel. The protocol around
// it owns the warm-up, the rounds, and the statistic, so this function states
// the call and nothing else.
template <int Atoms>
void LaunchResident(int blocks, const Bf16* device_q, const Bf16* device_k,
                    float* device_out, int steps) {
  ResidentKernel<Atoms><<<blocks, kWarpgroupThreads, kQueryBytes + kKeyBytes>>>(
      device_q, device_k, device_out, steps);
}

// The opt-in dynamic shared-memory request of a resident flash instantiation,
// and its one launch: `blocks` blocks of `Warpgroups` warpgroups, `tiles` key
// tiles every block, and no copy path in the loop. A templated kernel has no
// single function address, so the attribute is set once per instantiation and
// outside any timed window.
template <int Warpgroups>
void SetFlashSharedMemory() {
  constexpr int kShared = FlashResidentBytes(Warpgroups);
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(ResidentFlashKernel<Warpgroups>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

template <int Warpgroups>
void LaunchResidentFlash(int blocks, const Bf16* device_q, const Bf16* device_k,
                         const Bf16* device_v, float* device_out,
                         float* device_row_sums, int tiles, float scale) {
  constexpr int kShared = FlashResidentBytes(Warpgroups);
  ResidentFlashKernel<Warpgroups>
      <<<blocks, Warpgroups * kWarpgroupThreads, kShared>>>(
          device_q, device_k, device_v, device_out, device_row_sums, tiles,
          scale);
}

// One step of the flash tile over one block, read back: the accumulator slots
// of every warpgroup in thread order, and the row sums beside them.
template <int Warpgroups>
std::vector<float> RunResidentFlash(const Bf16* device_q, const Bf16* device_k,
                                    const Bf16* device_v, float* device_out,
                                    float* device_row_sums) {
  LaunchResidentFlash<Warpgroups>(1, device_q, device_k, device_v, device_out,
                                  device_row_sums, 1, kFlashScale);
  cudaDeviceSynchronize();
  return ReadBack(device_out, Warpgroups * kProductSlots);
}

// The one opt-in dynamic shared-memory request of a staged instantiation. A
// templated kernel has no single function address, so the attribute is set once
// per instantiation and outside any timed window.
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages>
void SetStagedSharedMemory() {
  constexpr int kShared = StagedBytes(Warpgroups, QueryTiles, Stages);
  static_assert(kShared <= kSmemCeiling,
                "the ring must fit one block per multiprocessor");
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          StagedKernel<Warpgroups, QueryTiles, kProducts, Stages>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

// One launch of the staged kernel: `tiles` steps of the ring, over `blocks`
// blocks, with the copy order of `options`.
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages>
void LaunchStaged(int blocks, int tiles, int source_tiles, const Bf16* device_q,
                  const Bf16* device_k, float* device_out, int options = 0) {
  constexpr int kShared = StagedBytes(Warpgroups, QueryTiles, Stages);
  StagedKernel<Warpgroups, QueryTiles, kProducts, Stages>
      <<<blocks, Warpgroups * kWarpgroupThreads, kShared>>>(
          device_q, device_k, source_tiles, device_out, tiles, options);
}

// One geometry of the staged ladder, dispatched to its own instantiation. The
// kernel carries the two counts as template parameters, so a sweep over them is
// one branch per geometry, and this is the one place that states the pairs.
template <typename Visitor>
void VisitGeometry(int warpgroups, int query_tiles, Visitor&& visit) {
  if (warpgroups == 1 && query_tiles == 1) {
    visit(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  } else if (warpgroups == 1 && query_tiles == 2) {
    visit(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  } else if (warpgroups == 1 && query_tiles == 3) {
    visit(std::integral_constant<int, 1>{}, std::integral_constant<int, 3>{});
  } else if (warpgroups == 2 && query_tiles == 1) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  } else if (warpgroups == 2 && query_tiles == 2) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  } else if (warpgroups == 2 && query_tiles == 3) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{});
  } else {
    std::fprintf(
        stderr,
        "the probe has no geometry of %d warpgroups and %d query tiles\n",
        warpgroups, query_tiles);
    std::exit(2);
  }
}

// One geometry's staged check: every query tile of every warpgroup must agree
// with the resident product of that same tile, because all of them multiply
// their own query tile by the same key tile. The answer is the count of
// elements outside the tolerance and the worst difference, both as the caller
// reports them.
template <int Warpgroups, int QueryTiles>
std::pair<std::size_t, double> CheckStagedGeometry(
    const Bf16* device_queries, const Bf16* device_k, float* device_out,
    int source_tiles, int options,
    const std::vector<std::vector<float>>& reference_tiles, double scale) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  constexpr int kTiles = Warpgroups * QueryTiles;
  constexpr int kShared = StagedBytes(Warpgroups, QueryTiles, kStages);
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(StagedKernel<Warpgroups, QueryTiles, true>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  cudaMemset(device_out, 0, kTiles * kProductSlots * sizeof(float));
  StagedKernel<Warpgroups, QueryTiles, true><<<1, kThreads, kShared>>>(
      device_queries, device_k, source_tiles, device_out, 1, options);
  cudaDeviceSynchronize();
  const std::vector<float> staged =
      ReadBack(device_out, kTiles * kProductSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (int tile = 0; tile < kTiles; ++tile) {
    for (std::size_t slot = 0; slot < kProductSlots; ++slot) {
      const double difference = std::fabs(
          static_cast<double>(
              staged[static_cast<std::size_t>(tile) * kProductSlots + slot]) -
          static_cast<double>(reference_tiles[tile][slot]));
      if (difference > 1e-3 * scale) ++bad;
      worst = std::max(worst, difference);
    }
  }
  return {bad, worst / scale};
}

void Upload(const std::vector<nanochat::Bf16>& source, Bf16* device) {
  cudaMemcpy(device, source.data(), source.size() * sizeof(nanochat::Bf16),
             cudaMemcpyHostToDevice);
}

// A short linear congruential generator, so a run reproduces its own operands.
struct Random {
  std::uint32_t state = 12345u;
  float Next() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f * 2.0f - 1.0f;
  }
};

// The verbose lines of a human run, and nothing when the caller asked for the
// document. `BenchReport::Print` writes the rows, so this class carries only
// the per-element detail a table cannot hold, and `--json` therefore stays
// pipeable.
class Diagnostics {
 public:
  explicit Diagnostics(bool enabled) : enabled_(enabled) {}

  void operator()(const char* format, ...) const {
    if (!enabled_) return;
    va_list arguments;
    va_start(arguments, format);
    std::vprintf(format, arguments);
    va_end(arguments);
  }

 private:
  bool enabled_;
};

// The three row sets this one binary prints. `phase0` is the ceiling rows of
// docs/kernels.md Part 3, `h1` is the H1 rows and the maps of docs/attention.md
// Part 8, and `flash` is the flash tile of the H2 prototype. `all` runs every
// set, so a caller that names no set loses no row.
enum class RowSet { kAll, kPhase0, kH1, kFlash };

// The parsed command line.
struct ProbeOptions {
  nanochat::BenchOptions bench;
  RowSet rows = RowSet::kAll;
  // Flushes the second-level cache before every timed call.
  bool flush = true;

  bool phase0() const { return rows != RowSet::kH1 && rows != RowSet::kFlash; }
  bool h1() const { return rows != RowSet::kPhase0 && rows != RowSet::kFlash; }
  bool flash() const { return rows == RowSet::kAll || rows == RowSet::kFlash; }
  const char* name() const {
    switch (rows) {
      case RowSet::kPhase0:
        return "phase0";
      case RowSet::kH1:
        return "h1";
      case RowSet::kFlash:
        return "flash";
      case RowSet::kAll:
        break;
    }
    return "all";
  }
};

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--rows phase0|h1|flash|all] [--iters N] "
               "[--rounds N] [--warmup N] [--no-flush] [--json] [--out PATH]\n",
               program);
}

ProbeOptions ParseOptions(int argc, char** argv) {
  ProbeOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--json") {
      options.bench.json = true;
    } else if (argument == "--rows" && i + 1 < argc) {
      const std::string value(argv[++i]);
      if (value == "phase0") {
        options.rows = RowSet::kPhase0;
      } else if (value == "h1") {
        options.rows = RowSet::kH1;
      } else if (value == "flash") {
        options.rows = RowSet::kFlash;
      } else if (value == "all") {
        options.rows = RowSet::kAll;
      } else {
        PrintUsage(argv[0]);
        std::exit(2);
      }
    } else if (argument == "--no-flush") {
      options.flush = false;
    } else if (argument == "--iters" && i + 1 < argc) {
      options.bench.iters = std::atoi(argv[++i]);
    } else if (argument == "--rounds" && i + 1 < argc) {
      options.bench.rounds = std::atoi(argv[++i]);
    } else if (argument == "--warmup" && i + 1 < argc) {
      options.bench.warmup = std::atoi(argv[++i]);
    } else if (argument == "--out" && i + 1 < argc) {
      options.bench.out = argv[++i];
    } else {
      PrintUsage(argv[0]);
      std::exit(2);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");
  const ProbeOptions options = ParseOptions(argc, argv);
  const Diagnostics note(!options.bench.json);

  // The one writer of the document. The denominator is the card's declared
  // bf16 peak, so every row carries `mfu`, which is the same quantity the old
  // line rows called `fraction_of_peak`. A card the table does not name falls
  // back to the reference entry the baselines cite, so the denominator cannot
  // silently disappear.
  nanochat::BenchReport report;
  report.SetPrecision(nanochat::BuildPrecision::kBf16);
  // The extra names the selection, and it must not be called `rows`: the
  // document's own `rows` array closes before the extras, so a second key of
  // that name replaces the measurements with a string for every strict reader.
  report.SetExtraText("rows_selected", options.name());
  // A run that names no check row cannot have run the checks, so the status
  // alone would overstate what this document covers.
  std::string checks_run;
  if (options.h1()) checks_run = "h1";
  if (options.flash()) {
    if (!checks_run.empty()) checks_run += "+";
    checks_run += "flash";
  }
  report.SetExtraText("checks_run", checks_run.empty() ? "none" : checks_run);

  note(
      "attention wgmma probe: the accumulator map, the operand form, the "
      "staged "
      "rate, and the flash tile, %d by %d by %d\n",
      kAtomM, kAtomN, kDim);

  int device = 0;
  cudaDeviceProp properties{};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  const int sm_count = properties.multiProcessorCount;
  if (!(report.UseDevicePeak(properties.name, nanochat::BuildPrecision::kBf16) >
        0.0)) {
    // A card the table does not name takes the declaration of the reference
    // entry the baselines cite, so the share of every row stays a share of a
    // declared rate and this program holds no rate of its own
    // (docs/precision.md).
    report.SetPeakFlops(nanochat::PeakFlopsFor(
        kReferenceDevice, nanochat::BuildPrecision::kBf16));
    note("the table names no rate for this card; %s stands in\n",
         kReferenceDevice);
  }
  note("card: %s, %d SMs, %d kHz clock\n", properties.name, sm_count,
       properties.clockRate / 1000);
  note("%s\n", report.PeakCaption().c_str());

  cudaFuncSetAttribute(reinterpret_cast<const void*>(ResidentKernel<kAtoms>),
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kQueryBytes + kKeyBytes);
  SetResidentSharedMemory<kHalfAtoms>();

  std::vector<nanochat::Bf16> host_q(static_cast<std::size_t>(kAtomM) * kDim);
  std::vector<nanochat::Bf16> host_k(static_cast<std::size_t>(kAtomN) * kDim);
  // The flash check reads one value tile, and the flash rows stage it every
  // step, so it is a device buffer from the first line of the set.
  std::vector<nanochat::Bf16> host_v(static_cast<std::size_t>(kAtomN) * kDim);
  // The query buffer holds one tile for every query tile of the deepest
  // geometry, and every row of the ladder reads a prefix of it.
  std::vector<nanochat::Bf16> host_queries(static_cast<std::size_t>(kMaxTiles) *
                                           kAtomM * kDim);
  Bf16* device_q = nullptr;
  Bf16* device_queries = nullptr;
  Bf16* device_k = nullptr;
  Bf16* device_value = nullptr;
  float* device_out = nullptr;
  float* device_row_sums = nullptr;
  constexpr int kSourceTiles = 2048;
  const std::size_t key_elements =
      static_cast<std::size_t>(kSourceTiles) * kAtomN * kDim;
  cudaMalloc(reinterpret_cast<void**>(&device_q), host_q.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_queries),
             host_queries.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_k), key_elements * sizeof(Bf16));
  cudaMalloc(
      reinterpret_cast<void**>(&device_out),
      std::max(kProductSlots, kMaxTiles * kProductSlots) * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&device_value),
             host_v.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_row_sums),
             static_cast<std::size_t>(kMaxWarpgroups) * kAtomM * sizeof(float));

  // The map and form sections are the H1 rows of Part 8, so `--rows phase0`
  // skips them and reports the ceiling rows alone.
  bool ok = true;
  if (options.h1()) {
    // --- One element at a time
    // ------------------------------------------------ A one-hot query row
    // against a one-hot key row leaves one non-zero value in the product, and
    // the slot that holds it is read off without any assumption about the order
    // of the registers.
    struct Pixel {
      int m;
      int n;
    };
    const Pixel pixels[] = {{0, 0},  {0, 1},  {0, 2},   {0, 7},    {0, 8},
                            {1, 0},  {2, 0},  {4, 0},   {8, 0},    {16, 0},
                            {32, 0}, {63, 0}, {0, 127}, {63, 127}, {17, 33}};
    int found = 0;
    for (const Pixel& pixel : pixels) {
      std::fill(host_q.begin(), host_q.end(), nanochat::Bf16FromFloat(0.0f));
      std::fill(host_k.begin(), host_k.end(), nanochat::Bf16FromFloat(0.0f));
      host_q[static_cast<std::size_t>(pixel.m) * kDim] =
          nanochat::Bf16FromFloat(1.0f);
      host_k[static_cast<std::size_t>(pixel.n) * kDim] =
          nanochat::Bf16FromFloat(1.0f);
      Upload(host_q, device_q);
      Upload(host_k, device_k);
      const std::vector<float> out =
          RunResident(device_q, device_k, device_out);
      int lane = -1;
      int index = -1;
      for (int l = 0; l < kWarpgroupThreads; ++l) {
        for (int i = 0; i < kRegisters; ++i) {
          if (std::fabs(out[static_cast<std::size_t>(l) * kRegisters + i]) >
              0.5f) {
            lane = l;
            index = i;
          }
        }
      }
      if (lane >= 0) ++found;
      note("attention-wgmma:pixel m=%d n=%d lane=%d index=%d\n", pixel.m,
           pixel.n, lane, index);
    }
    const int pixel_probed =
        static_cast<int>(sizeof(pixels) / sizeof(pixels[0]));
    note("attention-wgmma:pixel_summary probed=%d found=%d\n", pixel_probed,
         found);
    report.SetExtraInt("pixel:probed", pixel_probed);
    report.SetExtraInt("pixel:found", found);

    // --- The whole table
    // ------------------------------------------------------ Two products
    // decode the map: the first depends on the row only and the second on the
    // column only, and both patterns are exact in bf16, so a decoded value is
    // an index and not an estimate.
    std::vector<nanochat::Bf16> row_q(host_q.size());
    std::vector<nanochat::Bf16> row_k(host_k.size());
    std::vector<nanochat::Bf16> col_q(host_q.size());
    std::vector<nanochat::Bf16> col_k(host_k.size());
    for (int m = 0; m < kAtomM; ++m) {
      for (int k = 0; k < kDim; ++k) {
        row_q[static_cast<std::size_t>(m) * kDim + k] =
            nanochat::Bf16FromFloat(1.0f + static_cast<float>(m) / 64.0f);
        col_q[static_cast<std::size_t>(m) * kDim + k] =
            nanochat::Bf16FromFloat(1.0f);
      }
    }
    for (int n = 0; n < kAtomN; ++n) {
      for (int k = 0; k < kDim; ++k) {
        row_k[static_cast<std::size_t>(n) * kDim + k] =
            nanochat::Bf16FromFloat(1.0f);
        col_k[static_cast<std::size_t>(n) * kDim + k] =
            nanochat::Bf16FromFloat(1.0f + static_cast<float>(n) / 128.0f);
      }
    }
    Upload(row_q, device_q);
    Upload(row_k, device_k);
    const std::vector<float> rows = RunResident(device_q, device_k, device_out);
    Upload(col_q, device_q);
    Upload(col_k, device_k);
    const std::vector<float> cols = RunResident(device_q, device_k, device_out);

    auto row_of = [](float value) {
      return static_cast<int>(std::lround((value / 128.0 - 1.0) * 64.0));
    };
    auto col_of = [](float value) {
      return static_cast<int>(std::lround((value / 128.0 - 1.0) * 128.0));
    };
    std::vector<int> seen(static_cast<std::size_t>(kAtomM) * kAtomN, 0);
    int outside = 0;
    int formula_mismatch = 0;
    for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
      for (int index = 0; index < kRegisters; ++index) {
        const std::size_t slot = static_cast<std::size_t>(lane) * kRegisters +
                                 static_cast<std::size_t>(index);
        const int row = row_of(rows[slot]);
        const int col = col_of(cols[slot]);
        if (row < 0 || row >= kAtomM || col < 0 || col >= kAtomN) {
          ++outside;
          continue;
        }
        seen[static_cast<std::size_t>(row) * kAtomN + col] += 1;
        const std::pair<int, int> published = SlotOf(lane, index);
        if (published.first != row || published.second != col) {
          ++formula_mismatch;
        }
      }
    }
    int missing = 0;
    int repeated = 0;
    for (int count : seen) {
      if (count == 0) ++missing;
      if (count > 1) ++repeated;
    }
    note(
        "attention-wgmma:map_summary outside=%d missing=%d repeated=%d "
        "formula_mismatch=%d of %d\n",
        outside, missing, repeated, formula_mismatch, kAtomM * kAtomN);
    report.SetExtraInt("map:outside", outside);
    report.SetExtraInt("map:missing", missing);
    report.SetExtraInt("map:repeated", repeated);
    report.SetExtraInt("map:formula_mismatch", formula_mismatch);
    report.SetExtraInt("map:elements", kAtomM * kAtomN);
    report.SetExtraText("map:row_formula",
                        "16*(lane/32) + (lane%32)/4 + "
                        "8*((index/2)%2)");
    report.SetExtraText("map:col_formula",
                        "8*(index/4) + 2*(lane%4) + (index%2)");
    for (int lane : {0, 1, 4, 32}) {
      note("attention-wgmma:map lane=%d", lane);
      for (int index = 0; index < kRegisters; ++index) {
        const std::size_t slot = static_cast<std::size_t>(lane) * kRegisters +
                                 static_cast<std::size_t>(index);
        note(" %d:%d", row_of(rows[slot]), col_of(cols[slot]));
      }
      note("\n");
    }

    // --- The operand form
    // ----------------------------------------------------- The check the sum
    // cannot make: one element at a time, against a host reference built from
    // the same operands. A row-major stage reads one element of every eight, so
    // it fails here and passed the sum.
    Random random;
    for (nanochat::Bf16& value : host_q) {
      value = nanochat::Bf16FromFloat(random.Next());
    }
    for (nanochat::Bf16& value : host_k) {
      value = nanochat::Bf16FromFloat(random.Next());
    }
    std::vector<double> reference(static_cast<std::size_t>(kAtomM) * kAtomN,
                                  0.0);
    double scale = 0.0;
    for (int m = 0; m < kAtomM; ++m) {
      for (int n = 0; n < kAtomN; ++n) {
        double sum = 0.0;
        for (int k = 0; k < kDim; ++k) {
          sum += static_cast<double>(nanochat::Bf16ToFloat(
                     host_q[static_cast<std::size_t>(m) * kDim + k])) *
                 static_cast<double>(nanochat::Bf16ToFloat(
                     host_k[static_cast<std::size_t>(n) * kDim + k]));
        }
        reference[static_cast<std::size_t>(m) * kAtomN + n] = sum;
        scale = std::max(scale, std::fabs(sum));
      }
    }
    Upload(host_q, device_q);
    Upload(host_k, device_k);
    const std::vector<float> resident =
        RunResident(device_q, device_k, device_out);
    double worst = 0.0;
    for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
      for (int index = 0; index < kRegisters; ++index) {
        const std::pair<int, int> slot = SlotOf(lane, index);
        const double expected =
            reference[static_cast<std::size_t>(slot.first) * kAtomN +
                      slot.second];
        const double actual =
            resident[static_cast<std::size_t>(lane) * kRegisters + index];
        worst = std::max(worst, std::fabs(actual - expected));
      }
    }
    const double form_error = worst / scale;
    note("attention-wgmma:form_check error=%.3e scale=%.3f status=%s\n",
         form_error, scale, form_error < 3e-2 ? "ok" : "failed");
    report.SetExtra("form_check:error", form_error);
    report.SetExtra("form_check:scale", scale);
    report.SetExtraText("form_check:status",
                        form_error < 3e-2 ? "ok" : "failed");
    ok = ok && form_error < 3e-2;

    // --- The staged path
    // ------------------------------------------------------ The staged product
    // of every geometry must agree with the resident product of its own query
    // tile before either is timed. Every query tile is a different operand, so
    // a warpgroup that reads the wrong tile of the query region fails here, and
    // so does a wrong ring, a wrong barrier, or a wrong stage offset.
    std::vector<std::vector<float>> reference_tiles(kMaxTiles);
    for (int tile = 0; tile < kMaxTiles; ++tile) {
      const std::size_t offset = static_cast<std::size_t>(tile) * host_q.size();
      for (std::size_t element = 0; element < host_q.size(); ++element) {
        host_queries[offset + element] = nanochat::Bf16FromFloat(random.Next());
        host_q[element] = host_queries[offset + element];
      }
      Upload(host_q, device_q);
      reference_tiles[static_cast<std::size_t>(tile)] =
          RunResident(device_q, device_k, device_out);
    }
    Upload(host_queries, device_queries);
    Upload(host_k, device_k);
    bool staged_ok = true;
    auto check_geometry = [&](auto warpgroups, auto query_tiles) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kQueryTiles = decltype(query_tiles)::value;
      constexpr int kTiles = kWarpgroups * kQueryTiles;
      for (int options : {0, kPlainStage}) {
        const std::pair<std::size_t, double> result =
            CheckStagedGeometry<kWarpgroups, kQueryTiles>(
                device_queries, device_k, device_out, kSourceTiles, options,
                reference_tiles, scale);
        note(
            "attention-wgmma:staged_check warpgroups=%d query_tiles=%d "
            "options=%d error=%.3e bad=%zu of %d status=%s\n",
            kWarpgroups, kQueryTiles, options, result.second, result.first,
            kTiles * static_cast<int>(kProductSlots),
            result.second < 1e-6 ? "ok" : "failed");
        // The key names the geometry and the staging path, because the loop
        // runs once per pair and a repeated key would state the same field
        // twice in one document.
        const std::string prefix =
            "staged_check:" +
            std::string(options == kPlainStage ? "plain" : "async") +
            ":warpgroups" + std::to_string(kWarpgroups) + ":query_tiles" +
            std::to_string(kQueryTiles) + ":";
        report.SetExtra(prefix + "error", result.second);
        report.SetExtraInt(prefix + "bad",
                           static_cast<long long>(result.first));
        if (options == kPlainStage) {
          staged_ok = staged_ok && result.second < 1e-6;
        } else {
          staged_ok = result.second < 1e-6;
        }
      }
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int query_tiles = 1; query_tiles <= kMaxQueryTiles; ++query_tiles) {
        VisitGeometry(warpgroups, query_tiles, check_geometry);
      }
    }
    ok = ok && staged_ok;

    // --- The register operand
    // ------------------------------------------------- The value product's
    // probability tile is the first operand, and unlike the two staged operands
    // its element order is not a descriptor. One element at a time against a
    // key tile whose row zero encodes the key index names both fields of every
    // element, and the table below states the whole order.
    {
      std::fill(host_k.begin(), host_k.end(), nanochat::Bf16FromFloat(0.0f));
      for (int j = 0; j < kAtomK; ++j) {
        host_k[j] = nanochat::Bf16FromFloat(static_cast<float>(1 << j));
      }
      Upload(host_k, device_k);
      cudaFuncSetAttribute(reinterpret_cast<const void*>(RegisterOperandKernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kKeyBytes);
      std::vector<int> seen(static_cast<std::size_t>(kAtomM) * kAtomK, 0);
      int register_outside = 0;
      for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
        for (int reg = 0; reg < 4; ++reg) {
          for (int half = 0; half < 2; ++half) {
            RegisterOperandKernel<<<1, kWarpgroupThreads, kKeyBytes>>>(
                device_k, device_out, lane, reg, half);
            cudaDeviceSynchronize();
            const std::vector<float> out = ReadBack(device_out, kProductSlots);
            int at_lane = -1;
            int at_index = -1;
            float at_value = 0.0f;
            for (int thread = 0; thread < kWarpgroupThreads; ++thread) {
              for (int index = 0; index < kRegisters; ++index) {
                const float value =
                    out[static_cast<std::size_t>(thread) * kRegisters + index];
                if (std::fabs(value) > std::fabs(at_value) &&
                    std::fabs(value) > 0.5f) {
                  at_lane = thread;
                  at_index = index;
                  at_value = value;
                }
              }
            }
            int row = -1;
            int key = -1;
            if (at_lane >= 0) {
              row = SlotOf(at_lane, at_index).first;
              key = static_cast<int>(std::lround(std::log2(at_value)));
            }
            if (row >= 0 && row < kAtomM && key >= 0 && key < kAtomK) {
              seen[static_cast<std::size_t>(row) * kAtomK + key] += 1;
            } else {
              ++register_outside;
            }
            if (lane == 0 || lane == 1 || lane == 4 || lane == 32) {
              note(
                  "attention-wgmma:a_operand lane=%d reg=%d half=%d row=%d "
                  "key=%d value=%.0f\n",
                  lane, reg, half, row, key, static_cast<double>(at_value));
            }
          }
        }
      }
      int register_missing = 0;
      int register_repeated = 0;
      for (int count : seen) {
        if (count == 0) ++register_missing;
        if (count > 1) ++register_repeated;
      }
      note(
          "attention-wgmma:a_operand_summary outside=%d missing=%d repeated=%d "
          "of %d\n",
          register_outside, register_missing, register_repeated,
          kAtomM * kAtomK);
      report.SetExtraInt("register_operand:outside", register_outside);
      report.SetExtraInt("register_operand:missing", register_missing);
      report.SetExtraInt("register_operand:repeated", register_repeated);
      report.SetExtraInt("register_operand:elements", kAtomM * kAtomK);
      report.SetExtraText("register_operand:row_formula",
                          "16*(lane/32) + (lane%32)/4 + 8*(reg%2)");
      report.SetExtraText("register_operand:key_formula",
                          "8*(reg/2) + 2*(lane%4) + half");
      ok = ok && register_outside == 0 && register_missing == 0 &&
           register_repeated == 0;
    }

    // --- The probability permutation
    // ------------------------------------------ The value product takes the
    // probability tile as its first operand in registers, so the score
    // accumulator of a tile has to reach that operand. Section 5 of Part 8
    // states that no shuffle is needed, which holds only if a score element and
    // its probability share one lane. This checks that on the host from the two
    // measured formulas: every (row, key) of a 64 by 128 score tile must have
    // exactly one accumulator slot and one register slot, and the two must name
    // the same lane.
    {
      std::vector<int> accumulator_lane(
          static_cast<std::size_t>(kAtomM) * kAtomN, -1);
      std::vector<int> register_lane(static_cast<std::size_t>(kAtomM) * kAtomN,
                                     -1);
      int accumulator_shared = 0;
      int register_shared = 0;
      for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
        for (int index = 0; index < kRegisters; ++index) {
          const std::pair<int, int> slot = SlotOf(lane, index);
          int& owner =
              accumulator_lane[static_cast<std::size_t>(slot.first) * kAtomN +
                               slot.second];
          if (owner >= 0) ++accumulator_shared;
          owner = lane;
        }
      }
      // The value product walks the score tile in eight 16-key atoms, and one
      // atom is what a thread's eight registers hold.
      constexpr int kKeyAtoms = kAtomN / kAtomK;
      for (int atom = 0; atom < kKeyAtoms; ++atom) {
        for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
          for (int reg = 0; reg < 4; ++reg) {
            for (int half = 0; half < 2; ++half) {
              const std::pair<int, int> slot = RegisterSlotOf(lane, reg, half);
              const std::size_t flat =
                  static_cast<std::size_t>(slot.first) * kAtomN +
                  atom * kAtomK + slot.second;
              int& owner = register_lane[flat];
              if (owner >= 0) ++register_shared;
              owner = lane;
            }
          }
        }
      }
      int same_lane = 0;
      int cross_lane = 0;
      for (int row = 0; row < kAtomM; ++row) {
        for (int key = 0; key < kAtomN; ++key) {
          const std::size_t flat = static_cast<std::size_t>(row) * kAtomN + key;
          if (accumulator_lane[flat] < 0 || register_lane[flat] < 0) continue;
          if (accumulator_lane[flat] == register_lane[flat]) {
            ++same_lane;
          } else {
            ++cross_lane;
          }
        }
      }
      const bool permutation_ok = same_lane == kAtomM * kAtomN &&
                                  cross_lane == 0 && accumulator_shared == 0 &&
                                  register_shared == 0;
      note(
          "attention-wgmma:probability_permutation same_lane=%d cross_lane=%d "
          "accumulator_shared=%d register_shared=%d status=%s\n",
          same_lane, cross_lane, accumulator_shared, register_shared,
          permutation_ok ? "ok" : "failed");
      report.SetExtraInt("probability:same_lane", same_lane);
      report.SetExtraInt("probability:cross_lane", cross_lane);
      report.SetExtraInt("probability:accumulator_shared", accumulator_shared);
      report.SetExtraInt("probability:register_shared", register_shared);
      ok = ok && permutation_ok;
    }

    // --- The transposed value operand
    // ----------------------------------------- Part 8 section 5 records the
    // second transpose bit as unmeasured. One element at a time again: a
    // one-hot probability element at (row, key) and a one-hot value element at
    // (key, dim) leave one non-zero result, and it must land at (row, dim) for
    // the form and the atom stride to be right.
    {
      struct Triple {
        int row;
        int key;
        int dim;
      };
      const Triple triples[] = {{0, 0, 0},      {3, 5, 7},  {17, 33, 65},
                                {63, 127, 127}, {8, 64, 0}, {0, 127, 64}};
      std::vector<nanochat::Bf16> host_p(static_cast<std::size_t>(kAtomM) *
                                         kDim);
      std::vector<nanochat::Bf16> host_v(static_cast<std::size_t>(kAtomN) *
                                         kDim);
      Bf16* device_p = nullptr;
      Bf16* device_v = nullptr;
      cudaMalloc(reinterpret_cast<void**>(&device_p),
                 host_p.size() * sizeof(nanochat::Bf16));
      cudaMalloc(reinterpret_cast<void**>(&device_v),
                 host_v.size() * sizeof(nanochat::Bf16));
      cudaFuncSetAttribute(reinterpret_cast<const void*>(ValueOperandKernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kQueryBytes + kKeyBytes);
      bool value_ok = false;
      int value_configs_ok = 0;
      int value_best_found = 0;
      for (int transpose_stage : {0, 1}) {
        for (int trans_b : {0, 1}) {
          for (int atom_step :
               {2 * kSboBytes / 16, AtomStep(kAtomM), AtomStep(kAtomN)}) {
            int found = 0;
            for (const Triple& triple : triples) {
              std::fill(host_p.begin(), host_p.end(),
                        nanochat::Bf16FromFloat(0.0f));
              std::fill(host_v.begin(), host_v.end(),
                        nanochat::Bf16FromFloat(0.0f));
              host_p[static_cast<std::size_t>(triple.row) * kDim + triple.key] =
                  nanochat::Bf16FromFloat(1.0f);
              host_v[static_cast<std::size_t>(triple.key) * kDim + triple.dim] =
                  nanochat::Bf16FromFloat(1.0f);
              Upload(host_p, device_p);
              Upload(host_v, device_v);
              ValueOperandKernel<<<1, kWarpgroupThreads,
                                   kQueryBytes + kKeyBytes>>>(
                  device_p, device_v, atom_step, trans_b, transpose_stage,
                  device_out);
              cudaDeviceSynchronize();
              const std::vector<float> out =
                  ReadBack(device_out, kProductSlots);
              int one = 0;
              bool at_expected = false;
              for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
                for (int index = 0; index < kRegisters; ++index) {
                  const float value =
                      out[static_cast<std::size_t>(lane) * kRegisters + index];
                  if (std::fabs(value) > 0.5f) {
                    ++one;
                    const std::pair<int, int> slot = SlotOf(lane, index);
                    if (slot.first == triple.row && slot.second == triple.dim) {
                      at_expected = true;
                    }
                  }
                }
              }
              if (one == 1 && at_expected) ++found;
            }
            const int expected = sizeof(triples) / sizeof(triples[0]);
            const bool this_ok = found == expected;
            value_ok = value_ok || this_ok;
            if (this_ok) ++value_configs_ok;
            value_best_found = std::max(value_best_found, found);
            note(
                "attention-wgmma:value_operand transpose_stage=%d trans_b=%d "
                "atom_step=%d found=%d of=%d status=%s\n",
                transpose_stage, trans_b, atom_step, found, expected,
                this_ok ? "ok" : "failed");
          }
        }
      }
      ok = ok && value_ok;
      report.SetExtraInt(
          "value_operand:triples",
          static_cast<long long>(sizeof(triples) / sizeof(triples[0])));
      report.SetExtraInt("value_operand:configs_ok", value_configs_ok);
      report.SetExtraInt("value_operand:best_found", value_best_found);
      report.SetExtraText("value_operand:status", value_ok ? "ok" : "failed");
      cudaFree(device_p);
      cudaFree(device_v);
    }
  }  // options.h1(): the map and form sections

  // --- The flash tile
  // The check runs before a rate is timed. One block, one warpgroup, one query
  // tile, one key tile, and one value tile, compared with the host reference of
  // the family one element at a time: a wrong probability operand, a wrong
  // transposed value staging, a wrong row reduction, or a wrong rescale all
  // fail here instead of measuring. This is also the first end-to-end test of
  // the register operand, whose element order the H1 rows measured with one-hot
  // elements and never carried a full product through.
  if (options.flash()) {
    SetFlashSharedMemory<1>();
    SetFlashSharedMemory<2>();

    Random flash_random;
    std::vector<nanochat::Bf16> flash_q(host_q.size());
    std::vector<nanochat::Bf16> flash_k(host_k.size());
    std::vector<nanochat::Bf16> flash_v(host_v.size());
    for (nanochat::Bf16& value : flash_q) {
      value = nanochat::Bf16FromFloat(flash_random.Next());
    }
    for (nanochat::Bf16& value : flash_k) {
      value = nanochat::Bf16FromFloat(flash_random.Next());
    }
    for (nanochat::Bf16& value : flash_v) {
      value = nanochat::Bf16FromFloat(flash_random.Next());
    }
    // The reference reads the operands the device reads, so it is built from
    // the bf16 values and not from the draws that produced them.
    std::vector<float> reference_q(flash_q.size());
    std::vector<float> reference_k(flash_k.size());
    std::vector<float> reference_v(flash_v.size());
    for (std::size_t i = 0; i < flash_q.size(); ++i) {
      reference_q[i] = nanochat::Bf16ToFloat(flash_q[i]);
    }
    for (std::size_t i = 0; i < flash_k.size(); ++i) {
      reference_k[i] = nanochat::Bf16ToFloat(flash_k[i]);
      reference_v[i] = nanochat::Bf16ToFloat(flash_v[i]);
    }
    // One head, one batch, and every key visible: the check states the softmax
    // and the two products, and the mask is a phase of its own.
    nanochat::AttentionParams params;
    params.batch = 1;
    params.seq = kAtomM;
    params.num_heads = 1;
    params.num_kv_heads = 1;
    params.head_dim = kDim;
    params.causal = false;
    params.window_right = -1;
    params.kv_len = kAtomN;
    params.scale = kFlashScale;
    std::vector<float> reference_stats;
    const std::vector<float> reference =
        nanochat::dev::seqref::AttentionForward(
            params, reference_q, reference_k, reference_v, &reference_stats);

    Upload(flash_q, device_q);
    Upload(flash_k, device_k);
    Upload(flash_v, device_value);
    const std::vector<float> slots = RunResidentFlash<1>(
        device_q, device_k, device_value, device_out, device_row_sums);
    const std::vector<float> row_sums = ReadBack(device_row_sums, kAtomM);

    // One element at a time, as the form check above does. The kernel leaves
    // its output unnormalised, so the division by the row sum is the caller's
    // and it is what makes a wrong maximum or a wrong sum visible.
    double scale_seen = 0.0;
    for (float value : reference) {
      scale_seen = std::max(scale_seen, std::fabs(static_cast<double>(value)));
    }
    double worst = 0.0;
    std::size_t bad = 0;
    for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
      for (int index = 0; index < kRegisters; ++index) {
        const int row = SlotRow(lane, index);
        const int column = SlotColumn(lane, index);
        const double expected =
            reference[static_cast<std::size_t>(row) * kDim + column];
        const double sum = row_sums[static_cast<std::size_t>(row)];
        const double actual =
            sum > 0.0 ? static_cast<double>(
                            slots[static_cast<std::size_t>(lane) * kRegisters +
                                  index]) /
                            sum
                      : 0.0;
        const double difference = std::fabs(actual - expected);
        if (difference > 3e-2 * scale_seen) ++bad;
        worst = std::max(worst, difference);
      }
    }
    // The row sums against the statistics of the reference, which states the
    // same quantity in natural units.
    double sum_worst = 0.0;
    for (int row = 0; row < kAtomM; ++row) {
      const double expected =
          reference_stats[static_cast<std::size_t>(row) * 2 + 1];
      const double actual = row_sums[static_cast<std::size_t>(row)];
      if (expected > 0.0) {
        sum_worst =
            std::max(sum_worst, std::fabs(actual - expected) / expected);
      }
    }
    const double flash_error = scale_seen > 0.0 ? worst / scale_seen : 0.0;
    const bool flash_ok = flash_error < 3e-2 && sum_worst < 3e-2;
    note(
        "attention-flash:reference_check error=%.3e sum_error=%.3e "
        "scale=%.3f bad=%zu of %d status=%s\n",
        flash_error, sum_worst, scale_seen, bad, kAtomM * kDim,
        flash_ok ? "ok" : "failed");
    report.SetExtra("flash_check:error", flash_error);
    report.SetExtra("flash_check:sum_error", sum_worst);
    report.SetExtra("flash_check:scale", scale_seen);
    report.SetExtraInt("flash_check:bad", static_cast<long long>(bad));
    report.SetExtraInt("flash_check:elements", kAtomM * kDim);
    report.SetExtraText("flash_check:status", flash_ok ? "ok" : "failed");
    ok = ok && flash_ok;
  }

  // --- The rate -------------------------------------------------------------
  // The score product of the whole tile, from a resident tile, in four rows.
  //
  // Eight atoms is the head width and the H1 geometry of docs/attention.md
  // Part 8. Four atoms is the Phase 0 geometry of docs/kernels.md Part 3, whose
  // gate reads the per-SM share of the one-block row. `--rows` selects the set,
  // and the row names do not move with the selection.
  const int passes = 20000;

  // The protocol of `bench_protocol.h`: one call is one launch, `--iters` calls
  // fill one window, `--rounds` windows give the samples, and a flush of the
  // second-level cache runs before every timed call. The flush cost is measured
  // once and removed, so a window states the call and not the flush.
  nanochat::dev::Harness harness(options.bench, options.flush);
  if (options.flush && !harness.Init()) {
    note("the flush buffer did not fit; the rows run without a flush\n");
  }
  report.SetExtraInt("flush_bytes",
                     static_cast<long long>(harness.flush_bytes()));
  report.SetExtra("flush_ms", harness.flush_ms());
  report.SetExtraInt("iters", options.bench.iters);
  report.SetExtraInt("rounds", options.bench.rounds);
  note("flush %zu B, %.4f ms a call; %d calls a window, %d windows\n",
       harness.flush_bytes(), harness.flush_ms(), options.bench.iters,
       options.bench.rounds);

  // The identity of one row: the family, the phase of Part 8 or Part 3, and the
  // shape the row names. `AddTiming` writes the minimum row and its median
  // companion from it, so no row states a statistic of its own.
  auto make_row = [&](const char* name, const std::string& shape,
                      const char* phase, int occupancy = 0) {
    nanochat::BenchRow row;
    row.name = name;
    row.shape = shape;
    row.family = "wgmma";
    row.phase = phase;
    row.occupancy = occupancy;
    return row;
  };

  // One timed row: the two rows of `AddTiming`, and the rate the diagnostics
  // print. The rate is the necessary work over the elapsed time, and the report
  // turns it into the share of the card's peak, so nothing here divides by a
  // peak by hand.
  auto add_rate_row = [&](const nanochat::BenchRow& base,
                          const nanochat::dev::Timing& timing, double flops) {
    nanochat::dev::AddTiming(&report, base, timing, flops);
    return nanochat::dev::Gflops(flops, timing.min_ms);
  };

  // The per-SM share of the card peak: the row's own rate over the peak the
  // report states. It is the report's `mfu` times the multiprocessor count, so
  // the extra cannot disagree with the row beside it.
  auto sm_share = [&](double flops, double milliseconds) {
    const double peak = report.peak_flops();
    if (!(peak > 0.0) || sm_count <= 0) return 0.0;
    return nanochat::dev::Gflops(flops, milliseconds) * 1e9 /
           (peak / static_cast<double>(sm_count));
  };

  std::fill(host_q.begin(), host_q.end(), nanochat::Bf16FromFloat(0.5f));
  std::fill(host_k.begin(), host_k.end(), nanochat::Bf16FromFloat(0.5f));
  Upload(host_q, device_q);
  Upload(host_queries, device_queries);
  // The staged source is kSourceTiles key tiles; the whole buffer is filled
  // once, so the copies read the same bytes every pass.
  {
    std::vector<nanochat::Bf16> source(key_elements,
                                       nanochat::Bf16FromFloat(0.5f));
    cudaMemcpy(device_k, source.data(), key_elements * sizeof(Bf16),
               cudaMemcpyHostToDevice);
  }
  // The Phase 0 gate divides its one-block row by the per-SM share of the peak,
  // which no row field states, so each one-block row also carries that share as
  // an extra. The key names the block count, because two rows share a name.
  auto add_resident = [&](const char* name, int atoms, int blocks,
                          int occupancy) {
    const std::string shape = "blocks=" + std::to_string(blocks) +
                              " passes=" + std::to_string(passes) +
                              " atoms=" + std::to_string(atoms);
    const double flops =
        FlopsPerPass(atoms) * static_cast<double>(passes) * blocks;
    const nanochat::dev::Timing timing =
        atoms == kAtoms ? harness.Time([&] {
          LaunchResident<kAtoms>(blocks, device_q, device_k, device_out,
                                 passes);
        })
                        : harness.Time([&] {
                            LaunchResident<kHalfAtoms>(
                                blocks, device_q, device_k, device_out, passes);
                          });
    const double rate = add_rate_row(
        make_row(name, shape, atoms == kAtoms ? "h1" : "phase0", occupancy),
        timing, flops);
    const double share = sm_share(flops, timing.min_ms);
    report.SetExtra(
        name + std::string(":blocks") + std::to_string(blocks) + ":sm_share",
        share);
    note("%s %s min=%.3f ms median=%.3f ms tflops=%.1f sm_share=%.3f\n", name,
         shape.c_str(), timing.min_ms, timing.median_ms, rate / 1e12, share);
  };
  if (options.h1()) {
    add_resident("attention-wgmma:resident", kAtoms, 1, 1);
    add_resident("attention-wgmma:resident", kAtoms, sm_count, 1);
  }
  if (options.phase0()) {
    add_resident("wgmma:1block", kHalfAtoms, 1, 1);
    add_resident("wgmma:grid", kHalfAtoms, 2 * sm_count, 2);
  }

  if (options.h1()) {
    // The staged rows. The tile window states which source the copies read: 512
    // key tiles is 16 MiB and stays in the second-level cache, and 2048 is 64
    // MiB and does not. `copy_order` carries the order of the bank-conflict
    // note on `StageCopy`, so the two rows of a pair differ in the order alone.
    // The ladder over the warpgroups of a block and the query tiles of each is
    // the arithmetic that one staged byte buys, which is the whole of the
    // rate's account.
    const int product_tiles = 4096;
    const int copy_tiles = 8192;
    // The opt-in shared-memory request of each instantiation this section
    // launches, stated once here so no timed window charges a driver call.
    SetStagedSharedMemory<1, 1, true>();
    SetStagedSharedMemory<1, 2, true>();
    SetStagedSharedMemory<1, 3, true>();
    SetStagedSharedMemory<2, 1, true>();
    SetStagedSharedMemory<2, 2, true>();
    SetStagedSharedMemory<2, 3, true>();
    SetStagedSharedMemory<1, 1, false>();
    SetStagedSharedMemory<2, 1, false>();
    SetStagedSharedMemory<1, kMaxQueryTiles, false>();
    SetStagedSharedMemory<1, kMaxQueryTiles, false, 2>();
    SetStagedSharedMemory<1, kMaxQueryTiles, false, 5>();

    // One timed product row of one geometry. The work is the score product of
    // every query tile of every warpgroup over every staged key tile, so the
    // rate and the arithmetic per staged byte cannot disagree.
    auto time_geometry = [&](auto warpgroups, auto query_tiles) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kQueryTiles = decltype(query_tiles)::value;
      for (int window : {512, kSourceTiles}) {
        for (int copy_order : {0, kRowFast}) {
          const nanochat::dev::Timing timing = harness.Time([&] {
            LaunchStaged<kWarpgroups, kQueryTiles, true>(
                sm_count, product_tiles, window, device_queries, device_k,
                device_out, copy_order);
          });
          const double flops = 2.0 * kAtomM * kAtomN * kDim *
                               static_cast<double>(product_tiles) *
                               static_cast<double>(kWarpgroups) *
                               static_cast<double>(kQueryTiles) *
                               static_cast<double>(sm_count);
          const std::string shape =
              "warpgroups=" + std::to_string(kWarpgroups) +
              " query_tiles=" + std::to_string(kQueryTiles) +
              " window=" + std::to_string(window) +
              " copy_order=" + std::to_string(copy_order) +
              " blocks=" + std::to_string(sm_count) +
              " tiles=" + std::to_string(product_tiles);
          const double rate = add_rate_row(
              make_row("attention-wgmma:staged", shape, "h1"), timing, flops);
          // The arithmetic one staged byte buys, stated beside the rate it
          // earns, so a reader divides nothing by hand.
          report.SetExtra("staged:" + shape + ":flop_per_staged_byte",
                          64.0 * static_cast<double>(kWarpgroups) *
                              static_cast<double>(kQueryTiles));
          note(
              "attention-wgmma:staged %s min=%.3f ms tflops=%.1f "
              "flop_per_staged_byte=%.0f\n",
              shape.c_str(), timing.min_ms, rate / 1e12,
              64.0 * static_cast<double>(kWarpgroups) *
                  static_cast<double>(kQueryTiles));
        }
      }
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int query_tiles = 1; query_tiles <= kMaxQueryTiles; ++query_tiles) {
        VisitGeometry(warpgroups, query_tiles, time_geometry);
      }
    }

    // The copies alone, at the same thread counts the product rows run at, so a
    // reader compares the cap of the path with the rate of the products. The
    // copy row stages one query tile, because a copy reads no query tile.
    auto time_copies = [&](auto warpgroups, auto) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      for (int copy_order : {0, kRowFast}) {
        for (int window : {512, kSourceTiles}) {
          for (int blocks : {1, sm_count}) {
            const nanochat::dev::Timing timing = harness.Time([&] {
              LaunchStaged<kWarpgroups, 1, false>(blocks, copy_tiles, window,
                                                  device_queries, device_k,
                                                  device_out, copy_order);
            });
            const double bytes = static_cast<double>(kKeyBytes) *
                                 static_cast<double>(copy_tiles) *
                                 static_cast<double>(blocks);
            const double bandwidth = bytes / (timing.min_ms * 1e-3);
            const std::string shape =
                "warpgroups=" + std::to_string(kWarpgroups) +
                " window=" + std::to_string(window) +
                " copy_order=" + std::to_string(copy_order) +
                " blocks=" + std::to_string(blocks) +
                " tiles=" + std::to_string(copy_tiles);
            // A copy reads and writes no operand of the product, so the row
            // states no FLOP and no rate. The byte rate it reached is the one
            // number beside it, and the table prints `0.0` in the rate column
            // and `-` in the two share columns.
            add_rate_row(make_row("attention-wgmma:copies", shape, "h1"),
                         timing, /*flops=*/0.0);
            report.SetExtra("copies:" + shape + ":tbps", bandwidth / 1e12);
            note("attention-wgmma:copies %s min=%.3f ms tbps=%.2f\n",
                 shape.c_str(), timing.min_ms, bandwidth / 1e12);
          }
        }
      }
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      VisitGeometry(warpgroups, 1, time_copies);
    }

    // The depth of the ring sets how many copy groups one block keeps in
    // flight, and the copies are the limit of a staged row. The deepest ring
    // the 227 KiB opt-in limit allows is five key tiles beside three query
    // tiles, which is the deepest geometry without a second warpgroup.
    {
      const nanochat::dev::Timing two_stages = harness.Time([&] {
        LaunchStaged<1, kMaxQueryTiles, false, 2>(sm_count, copy_tiles,
                                                  kSourceTiles, device_queries,
                                                  device_k, device_out, 0);
      });
      const nanochat::dev::Timing four_stages = harness.Time([&] {
        LaunchStaged<1, kMaxQueryTiles, false, kStages>(
            sm_count, copy_tiles, kSourceTiles, device_queries, device_k,
            device_out, 0);
      });
      const nanochat::dev::Timing five_stages = harness.Time([&] {
        LaunchStaged<1, kMaxQueryTiles, false, 5>(sm_count, copy_tiles,
                                                  kSourceTiles, device_queries,
                                                  device_k, device_out, 0);
      });
      for (const std::pair<int, const nanochat::dev::Timing*>& depth :
           {std::pair<int, const nanochat::dev::Timing*>{2, &two_stages},
            std::pair<int, const nanochat::dev::Timing*>{kStages, &four_stages},
            std::pair<int, const nanochat::dev::Timing*>{5, &five_stages}}) {
        const double bytes = static_cast<double>(kKeyBytes) *
                             static_cast<double>(copy_tiles) *
                             static_cast<double>(sm_count);
        const double bandwidth = bytes / (depth.second->min_ms * 1e-3);
        const std::string shape =
            "warpgroups=1 window=" + std::to_string(kSourceTiles) +
            " blocks=" + std::to_string(sm_count) +
            " stages=" + std::to_string(depth.first) +
            " tiles=" + std::to_string(copy_tiles);
        add_rate_row(make_row("attention-wgmma:copy_depth", shape, "h1"),
                     *depth.second, /*flops=*/0.0);
        report.SetExtra("copy_depth:warpgroups1:stages" +
                            std::to_string(depth.first) + ":tbps",
                        bandwidth / 1e12);
        note("attention-wgmma:copy_depth %s min=%.3f ms tbps=%.2f\n",
             shape.c_str(), depth.second->min_ms, bandwidth / 1e12);
      }
    }
  }  // options.h1(): the staged rows

  // The flash rows: the whole tile from a resident operand set, so a row prices
  // the base-two softmax, the second product, and the chain between them
  // against the score-only ceiling of `attention-wgmma:resident`.
  if (options.flash()) {
    const int flash_tiles = 5000;
    auto add_flash_resident = [&](int blocks) {
      const std::string shape = "blocks=" + std::to_string(blocks) +
                                " tiles=" + std::to_string(flash_tiles) +
                                " warpgroups=1 atoms=8+8";
      const double flops = 2.0 * FlopsPerPass(kAtoms) *
                           static_cast<double>(flash_tiles) *
                           static_cast<double>(blocks);
      const nanochat::dev::Timing timing = harness.Time([&] {
        LaunchResidentFlash<1>(blocks, device_queries, device_k, device_value,
                               device_out, /*device_row_sums=*/nullptr,
                               flash_tiles, kFlashScale);
      });
      const double rate =
          add_rate_row(make_row("attention-flash:resident", shape, "flash", 1),
                       timing, flops);
      const double share = sm_share(flops, timing.min_ms);
      report.SetExtra(std::string("flash_resident:blocks") +
                          std::to_string(blocks) + ":sm_share",
                      share);
      note(
          "attention-flash:resident %s min=%.3f ms median=%.3f ms "
          "tflops=%.1f sm_share=%.3f\n",
          shape.c_str(), timing.min_ms, timing.median_ms, rate / 1e12, share);
    };
    add_flash_resident(1);
    add_flash_resident(sm_count);
  }

  constexpr int kDeepestStagedBytes =
      StagedBytes(kMaxWarpgroups, kMaxQueryTiles, kStages);
  note(
      "precision=%s smem_bytes=%d lbo=%d sbo=%d stages=%d "
      "query_atom_step=%d key_atom_step=%d\n",
      report.precision().c_str(), kDeepestStagedBytes, kLboBytes, kSboBytes,
      kStages, AtomStep(kAtomM), AtomStep(kAtomN));
  report.SetExtraInt("smem:staged_bytes", kDeepestStagedBytes);
  report.SetExtraInt("smem:leading_byte_offset", kLboBytes);
  report.SetExtraInt("smem:stride_byte_offset", kSboBytes);
  report.SetExtraInt("smem:key_stages", kStages);
  report.SetExtraInt("smem:query_atom_step", AtomStep(kAtomM));
  report.SetExtraInt("smem:key_atom_step", AtomStep(kAtomN));
  report.SetExtraInt("smem:query_bytes", kQueryBytes);
  report.SetExtraInt("smem:key_bytes", kKeyBytes);
  report.SetExtraText("status", ok ? "ok" : "failed");
  note("attention-wgmma:status %s\n", ok ? "ok" : "failed");

  cudaFree(device_q);
  cudaFree(device_queries);
  cudaFree(device_k);
  cudaFree(device_value);
  cudaFree(device_out);
  cudaFree(device_row_sums);
  report.Print(options.bench);
  return ok ? 0 : 1;
}
