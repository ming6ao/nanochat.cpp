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
// copies alone. The staged rows run at one and at two query tiles, because the
// arithmetic one staged byte buys decides whether the copies or the products
// are the limit, and the copy rows sweep the depth of the ring and two orders
// of the staged stores. The staged product is compared with the resident
// product before it is timed, so a wrong staging fails instead of measuring.
//
// **One binary, two row sets, one document.** The resident section carries the
// ceiling rows of the Phase 0 probe this file absorbed, so one build and one
// card job replace two. The `attention-wgmma:` row names mark the H1 rows of
// docs/attention.md Part 8, and the `wgmma:` row names mark the Phase 0 rows of
// docs/kernels.md Part 3. `--rows phase0|h1|all` selects the set, `all` is the
// default, and the row names do not move with the selection, so both baselines
// entries stay reproducible from one run. The two sets differ in pass depth,
// eight atoms against four, and in occupancy, one warpgroup per SM against two,
// which is why neither divides into the other.
//
// **The report.** Every timed row goes through `BenchReport`, so the program
// states one `nanochat.bench.v3` document with `--json`, and `mfu` on each row
// is that row's share of the card's declared bf16 peak. The lines this program
// used to print are diagnostics: they stay on stdout for a human run and are
// dropped under `--json`, so the document stays pipeable.
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
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nanochat/bench.h"
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

//: The interleaved core-matrix form. One core matrix is eight rows of eight
//: elements, sixteen bytes to a row, so the two halves of the `k` axis of one
//: row group are one leading offset apart and the next row group follows at one
//: stride offset.
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 2 * kLboBytes;

//: The ring of staged key tiles, and the query tiles that share one staged key
//: tile. Two query tiles is the geometry of Part 8 section 3: it doubles the
//: arithmetic one staged byte buys and moves the tile off the copy cap.
constexpr int kStages = 4;
constexpr int kQueryTiles = 2;

constexpr int kQueryBytes = kAtomM * kDim * sizeof(Bf16);
constexpr int kKeyBytes = kAtomN * kDim * sizeof(Bf16);
constexpr int kStagedBytes = kQueryTiles * kQueryBytes + kStages * kKeyBytes;

//: The largest dynamic shared-memory request the card accepts, which one block
//: per multiprocessor must stay under.
constexpr int kSmemCeiling = 227 * 1024;
static_assert(kStagedBytes <= kSmemCeiling,
              "the staged ring must fit one block per multiprocessor");

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

//: The reference peak of `backends/cuda/caps.json` for the h100 entry.
constexpr double kPeakFlops = 989e12;

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
// path and the query path; the staged key tiles use the copy engine below.
__device__ __forceinline__ void StageInterleaved(char* stage,
                                                 const Bf16* source, int rows) {
  const int total = rows * kDim;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += kWarpgroupThreads) {
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
__device__ __forceinline__ void StageCopy(char* stage, const Bf16* source,
                                          bool row_fast) {
  const int groups = kAtomN * kGroupsPerRow;
  for (int unit = static_cast<int>(threadIdx.x); unit < groups;
       unit += kWarpgroupThreads) {
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
__device__ __forceinline__ void StageTransposed(char* stage,
                                                const Bf16* source) {
  constexpr int kKeyGroups = kAtomN / kElementsPerGroup;
  for (int unit = static_cast<int>(threadIdx.x); unit < kKeyGroups * kDim;
       unit += kWarpgroupThreads) {
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
  StageInterleaved(query, q, kAtomM);
  StageInterleaved(key, k, kAtomN);
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

//: The two experiment switches of the staged kernel. `kPlainStage` stages the
//: ring with plain stores instead of the asynchronous copies, so a wrong staged
//: product separates into a wrong copy and a wrong ring. `kRowFast` picks the
//: store order of the bank-conflict note on `StageCopy`.
constexpr int kPlainStage = 1;
constexpr int kRowFast = 2;

// One warpgroup, `QueryTiles` resident query tiles, and a ring of `kStages`
// key tiles that the copy engine fills. `kProducts` false is the copy row: the
// same bytes, the same ring, and no instruction.
template <int QueryTiles, bool kProducts, int Stages = kStages>
__global__ void StagedKernel(const Bf16* __restrict__ q,
                             const Bf16* __restrict__ k, int source_tiles,
                             float* __restrict__ out, int tiles, int options) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* stages = smem + QueryTiles * kQueryBytes;

#pragma unroll
  for (int tile = 0; tile < QueryTiles; ++tile) {
    StageInterleaved(query + tile * kQueryBytes,
                     q + static_cast<std::size_t>(tile) * kAtomM * kDim,
                     kAtomM);
  }
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  std::uint64_t base_query[QueryTiles];
#pragma unroll
  for (int tile = 0; tile < QueryTiles; ++tile) {
    base_query[tile] = MakeDescriptor(query + tile * kQueryBytes);
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
      StageInterleaved(stages + tile * kKeyBytes, source, kAtomN);
    } else {
      StageCopy(stages + tile * kKeyBytes, source, row_fast);
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
        StageInterleaved(stages + (next % Stages) * kKeyBytes, source, kAtomN);
      } else {
        StageCopy(stages + (next % Stages) * kKeyBytes, source, row_fast);
      }
    }
    CopyCommit();
  }

  if constexpr (kProducts) {
    WgmmaWait<0>();
    WgmmaFenceOperand<QueryTiles * kRegisters>(d);
#pragma unroll
    for (int slot = 0; slot < QueryTiles; ++slot) {
      float* destination =
          out + (static_cast<std::size_t>(slot) * kWarpgroupThreads +
                 static_cast<std::size_t>(threadIdx.x)) *
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
  StageInterleaved(key, k, kAtomN);
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
  StageInterleaved(probability, p, kAtomM);
  if (transpose_stage != 0) {
    StageTransposed(value, v);
  } else {
    StageInterleaved(value, v, kAtomN);
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

  constexpr int kKeyAtoms = kAtomN / kAtomK;
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
  const int row = 16 * (lane / 32) + (lane % 32) / 4 + 8 * ((index / 2) % 2);
  const int column = 8 * (index / 4) + 2 * (lane % 4) + (index % 2);
  return {row, column};
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

// One timed resident row: `blocks` warpgroups, `Atoms` atoms per pass, one
// warm-up launch and one measured launch. Returns the elapsed milliseconds and
// nothing else, because the caller owns the row and the report owns the table.
template <int Atoms>
double TimeResident(int blocks, const Bf16* device_q, const Bf16* device_k,
                    float* device_out, int passes) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  ResidentKernel<Atoms><<<blocks, kWarpgroupThreads, kQueryBytes + kKeyBytes>>>(
      device_q, device_k, device_out, 8);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  ResidentKernel<Atoms><<<blocks, kWarpgroupThreads, kQueryBytes + kKeyBytes>>>(
      device_q, device_k, device_out, passes);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(milliseconds);
}

template <int QueryTiles, bool kProducts, int Stages = kStages>
float TimeStaged(int blocks, int tiles, int source_tiles, const Bf16* device_q,
                 const Bf16* device_k, float* device_out, int options = 0) {
  constexpr int kShared = QueryTiles * kQueryBytes + Stages * kKeyBytes;
  static_assert(kShared <= kSmemCeiling,
                "the ring must fit one block per multiprocessor");
  cudaFuncSetAttribute(reinterpret_cast<const void*>(
                           StagedKernel<QueryTiles, kProducts, Stages>),
                       cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  StagedKernel<QueryTiles, kProducts, Stages>
      <<<blocks, kWarpgroupThreads, kShared>>>(device_q, device_k, source_tiles,
                                               device_out, 8, options);
  cudaDeviceSynchronize();
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  StagedKernel<QueryTiles, kProducts, Stages>
      <<<blocks, kWarpgroupThreads, kShared>>>(device_q, device_k, source_tiles,
                                               device_out, tiles, options);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
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

// The two row sets this one binary prints. `phase0` is the ceiling rows of
// docs/kernels.md Part 3, `h1` is the H1 rows and the maps of docs/attention.md
// Part 8, and `all` runs both. `all` is the default, so a caller that names no
// set loses no row.
enum class RowSet { kAll, kPhase0, kH1 };

// The parsed command line.
struct ProbeOptions {
  nanochat::BenchOptions bench;
  RowSet rows = RowSet::kAll;

  bool phase0() const { return rows != RowSet::kH1; }
  bool h1() const { return rows != RowSet::kPhase0; }
  const char* name() const {
    return rows == RowSet::kPhase0 ? "phase0"
                                   : (rows == RowSet::kH1 ? "h1" : "all");
  }
};

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--rows phase0|h1|all] [--json] [--out PATH]\n",
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
      } else if (value == "all") {
        options.rows = RowSet::kAll;
      } else {
        PrintUsage(argv[0]);
        std::exit(2);
      }
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
  report.SetExtraText("rows", options.name());
  // A run that names no H1 row cannot have run the form checks, so the status
  // alone would overstate what this document covers.
  report.SetExtraText("checks_run", options.h1() ? "h1" : "none");

  note(
      "attention wgmma probe: the accumulator map, the operand form, and the "
      "staged rate, %d by %d by %d\n",
      kAtomM, kAtomN, kDim);

  int device = 0;
  cudaDeviceProp properties{};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  const int sm_count = properties.multiProcessorCount;
  if (!(report.UseDevicePeak(properties.name, nanochat::BuildPrecision::kBf16) >
        0.0)) {
    report.SetPeakFlops(kPeakFlops);
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
  // The query buffer holds kQueryTiles tiles, which the two-query-tile rows
  // read.
  std::vector<nanochat::Bf16> host_queries(
      static_cast<std::size_t>(kQueryTiles) * kAtomM * kDim);
  Bf16* device_q = nullptr;
  Bf16* device_queries = nullptr;
  Bf16* device_k = nullptr;
  float* device_out = nullptr;
  constexpr int kSourceTiles = 2048;
  const std::size_t key_elements =
      static_cast<std::size_t>(kSourceTiles) * kAtomN * kDim;
  cudaMalloc(reinterpret_cast<void**>(&device_q), host_q.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_queries),
             host_queries.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_k), key_elements * sizeof(Bf16));
  cudaMalloc(
      reinterpret_cast<void**>(&device_out),
      std::max(kProductSlots, kQueryTiles * kProductSlots) * sizeof(float));

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
    // must agree with the resident product before either is timed, and the two
    // query tiles must agree with each other.
    for (int tile = 0; tile < kQueryTiles; ++tile) {
      std::copy(host_q.begin(), host_q.end(),
                host_queries.begin() + tile * host_q.size());
    }
    Upload(host_queries, device_queries);
    Upload(host_k, device_k);
    cudaMemset(device_out, 0, kQueryTiles * kProductSlots * sizeof(float));
    bool staged_ok = true;
    for (int options : {0, kPlainStage}) {
      constexpr int kShared = kQueryTiles * kQueryBytes + kStages * kKeyBytes;
      cudaFuncSetAttribute(
          reinterpret_cast<const void*>(StagedKernel<kQueryTiles, true>),
          cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
      cudaMemset(device_out, 0, kQueryTiles * kProductSlots * sizeof(float));
      StagedKernel<kQueryTiles, true><<<1, kWarpgroupThreads, kShared>>>(
          device_queries, device_k, kSourceTiles, device_out, 1, options);
      cudaDeviceSynchronize();
      const std::vector<float> staged =
          ReadBack(device_out, kQueryTiles * kProductSlots);
      double staged_worst = 0.0;
      double staged_sum = 0.0;
      std::size_t staged_bad = 0;
      std::size_t worst_slot = 0;
      for (int tile = 0; tile < kQueryTiles; ++tile) {
        for (std::size_t slot = 0; slot < kProductSlots; ++slot) {
          const double difference = std::fabs(
              static_cast<double>(
                  staged[static_cast<std::size_t>(tile) * kProductSlots +
                         slot]) -
              static_cast<double>(resident[slot]));
          staged_sum += difference;
          if (difference > 1e-3 * scale) ++staged_bad;
          if (difference > staged_worst) {
            staged_worst = difference;
            worst_slot = slot;
          }
        }
      }
      const double staged_error = staged_worst / scale;
      const double staged_mean = staged_sum / (kQueryTiles * kProductSlots);
      note(
          "attention-wgmma:staged_check query_tiles=%d options=%d error=%.3e "
          "mean=%.3e bad=%zu worst_slot=%zu worst=%.4f expected=%.4f "
          "status=%s\n",
          kQueryTiles, options, staged_error, staged_mean / scale, staged_bad,
          worst_slot, static_cast<double>(staged[worst_slot]),
          static_cast<double>(resident[worst_slot]),
          staged_error < 1e-6 ? "ok" : "failed");
      // The key names the staging path, because the loop runs twice and a
      // repeated key would state the same field twice in one document.
      const std::string prefix = std::string("staged_check:") +
                                 (options == kPlainStage ? "plain" : "async") +
                                 ":";
      report.SetExtra(prefix + "error", staged_error);
      report.SetExtra(prefix + "mean", staged_mean / scale);
      report.SetExtraInt(prefix + "bad", static_cast<long long>(staged_bad));
      if (options == kPlainStage) {
        staged_ok = staged_ok && staged_error < 1e-6;
      } else {
        staged_ok = staged_error < 1e-6;
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

  // --- The rate -------------------------------------------------------------
  // The score product of the whole tile, from a resident tile, in four rows.
  //
  // Eight atoms is the head width and the H1 geometry of docs/attention.md
  // Part 8. Four atoms is the Phase 0 geometry of docs/kernels.md Part 3, whose
  // gate reads the per-SM share of the one-block row. `--rows` selects the set,
  // and the row names do not move with the selection.
  const int passes = 20000;

  // One timed row. The rate is the necessary work over the elapsed time, and
  // the report turns it into `mfu` from the card's declared peak, so nothing
  // here divides by a peak by hand.
  auto add_rate_row = [&](const char* name, const std::string& shape,
                          double milliseconds, double flops, bool h1,
                          int occupancy = 0) {
    nanochat::BenchRow row;
    row.name = name;
    row.shape = shape;
    row.ms = milliseconds;
    row.gflops = flops > 0.0 && milliseconds > 0.0
                     ? flops / (milliseconds * 1e-3) * 1e-9
                     : 0.0;
    row.occupancy = occupancy;
    row.family = "wgmma";
    row.phase = h1 ? "h1" : "phase0";
    report.Add(row);
    return milliseconds > 0.0 ? flops / (milliseconds * 1e-3) : 0.0;
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
    const double milliseconds =
        atoms == kAtoms ? TimeResident<kAtoms>(blocks, device_q, device_k,
                                               device_out, passes)
                        : TimeResident<kHalfAtoms>(blocks, device_q, device_k,
                                                   device_out, passes);
    const double rate = add_rate_row(name, shape, milliseconds, flops,
                                     atoms == kAtoms, occupancy);
    const double share = rate / (kPeakFlops / static_cast<double>(sm_count));
    report.SetExtra(
        name + std::string(":blocks") + std::to_string(blocks) + ":sm_share",
        share);
    note("%s %s ms=%.3f tflops=%.1f sm_share=%.3f\n", name, shape.c_str(),
         milliseconds, rate / 1e12, share);
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
    const int product_tiles = 4096;
    const int copy_tiles = 8192;
    for (int copy_order : {0, kRowFast}) {
      for (int window : {512, kSourceTiles}) {
        for (int query_tiles = 1; query_tiles <= kQueryTiles; ++query_tiles) {
          const double milliseconds =
              query_tiles == 1
                  ? TimeStaged<1, true>(sm_count, product_tiles, window,
                                        device_queries, device_k, device_out,
                                        copy_order)
                  : TimeStaged<2, true>(sm_count, product_tiles, window,
                                        device_queries, device_k, device_out,
                                        copy_order);
          const double flops = 2.0 * kAtomM * kAtomN * kDim *
                               static_cast<double>(product_tiles) *
                               static_cast<double>(query_tiles) *
                               static_cast<double>(sm_count);
          const std::string shape =
              "query_tiles=" + std::to_string(query_tiles) +
              " window=" + std::to_string(window) +
              " copy_order=" + std::to_string(copy_order) +
              " blocks=" + std::to_string(sm_count) +
              " tiles=" + std::to_string(product_tiles);
          const double rate = add_rate_row("attention-wgmma:staged", shape,
                                           milliseconds, flops, true);
          note("attention-wgmma:staged %s ms=%.3f tflops=%.1f\n", shape.c_str(),
               milliseconds, rate / 1e12);
        }
      }
      for (int window : {512, kSourceTiles}) {
        for (int blocks : {1, sm_count}) {
          const double milliseconds =
              TimeStaged<2, false>(blocks, copy_tiles, window, device_queries,
                                   device_k, device_out, copy_order);
          const double bytes = static_cast<double>(kKeyBytes) *
                               static_cast<double>(copy_tiles) *
                               static_cast<double>(blocks);
          const double bandwidth = bytes / (milliseconds * 1e-3);
          const std::string shape =
              "window=" + std::to_string(window) +
              " copy_order=" + std::to_string(copy_order) +
              " blocks=" + std::to_string(blocks) +
              " tiles=" + std::to_string(copy_tiles);
          // A copy reads and writes no operand of the product, so the row
          // states no rate and the byte rate it reached is an extra beside it.
          add_rate_row("attention-wgmma:copies", shape, milliseconds, 0.0,
                       true);
          report.SetExtra("copies:" + shape + ":tbps", bandwidth / 1e12);
          note("attention-wgmma:copies %s ms=%.3f tbps=%.2f\n", shape.c_str(),
               milliseconds, bandwidth / 1e12);
        }
      }
    }

    // The depth of the ring sets how many copy groups one block keeps in
    // flight, and the copies are the limit of a staged row. The deepest ring
    // the 227 KiB opt-in limit allows is five key tiles beside two query tiles.
    for (int window : {kSourceTiles}) {
      const float two_stages = TimeStaged<kQueryTiles, false, 2>(
          sm_count, copy_tiles, window, device_queries, device_k, device_out,
          0);
      const float four_stages = TimeStaged<kQueryTiles, false, kStages>(
          sm_count, copy_tiles, window, device_queries, device_k, device_out,
          0);
      const float five_stages = TimeStaged<kQueryTiles, false, 5>(
          sm_count, copy_tiles, window, device_queries, device_k, device_out,
          0);
      for (const std::pair<int, float>& depth :
           {std::pair<int, float>{2, two_stages},
            std::pair<int, float>{kStages, four_stages},
            std::pair<int, float>{5, five_stages}}) {
        const double bytes = static_cast<double>(kKeyBytes) *
                             static_cast<double>(copy_tiles) *
                             static_cast<double>(sm_count);
        const double bandwidth =
            bytes / (static_cast<double>(depth.second) * 1e-3);
        const std::string shape = "window=" + std::to_string(window) +
                                  " blocks=" + std::to_string(sm_count) +
                                  " stages=" + std::to_string(depth.first) +
                                  " tiles=" + std::to_string(copy_tiles);
        add_rate_row("attention-wgmma:copy_depth", shape,
                     static_cast<double>(depth.second), 0.0, true);
        report.SetExtra(
            "copy_depth:stages" + std::to_string(depth.first) + ":tbps",
            bandwidth / 1e12);
        note("attention-wgmma:copy_depth %s ms=%.3f tbps=%.2f\n", shape.c_str(),
             static_cast<double>(depth.second), bandwidth / 1e12);
      }
    }
  }  // options.h1(): the staged rows

  note(
      "precision=%s smem_bytes=%d lbo=%d sbo=%d stages=%d "
      "query_atom_step=%d key_atom_step=%d\n",
      report.precision().c_str(), kStagedBytes, kLboBytes, kSboBytes, kStages,
      AtomStep(kAtomM), AtomStep(kAtomN));
  report.SetExtraInt("smem:staged_bytes", kStagedBytes);
  report.SetExtraInt("smem:leading_byte_offset", kLboBytes);
  report.SetExtraInt("smem:stride_byte_offset", kSboBytes);
  report.SetExtraInt("smem:key_stages", kStages);
  report.SetExtraInt("smem:query_atom_step", AtomStep(kAtomM));
  report.SetExtraInt("smem:key_atom_step", AtomStep(kAtomN));
  report.SetExtraText("status", ok ? "ok" : "failed");
  note("attention-wgmma:status %s\n", ok ? "ok" : "failed");

  cudaFree(device_q);
  cudaFree(device_queries);
  cudaFree(device_k);
  cudaFree(device_out);
  report.Print(options.bench);
  return ok ? 0 : 1;
}
