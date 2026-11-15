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
// **The key width.** The instruction holds `m64n128k16` and `m64n64k16`, and
// the narrow form is the one lever on a staged row that costs no copy
// bandwidth: half the key width halves the accumulator, so one register file
// then holds twice the query rows behind one staged key tile. The narrow rows
// state the instruction's own ceiling (`attention-wgmma:resident_narrow`), the
// copy rate of the narrow ring, and the rate of the `score64:2x4` shape, and
// two checks stand behind them: `form_check_narrow` measures the narrow
// accumulator map one element at a time against a host reference, and
// `staged_check keys=64` compares the narrow staged product with the narrow
// resident product of the same query tile.
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
// **One binary, four row sets, one document.** The resident section carries
// the ceiling rows of the Phase 0 probe this file absorbed, so one build and
// one card job replace two. The `attention-wgmma:` row names mark the H1 rows
// of docs/attention.md Part 8, the `wgmma:` row names mark the Phase 0 rows of
// docs/kernels.md Part 3, the `attention-flash:` row names mark the flash tile
// of the H2 prototype, and the `attention-special:` row names mark the
// warp-specialized block of decision H3. `--rows phase0|h1|flash|h3|all`
// selects the set, `all` is the default, and the row names do not move with the
// selection, so every baselines entry stays reproducible from one run. The
// first two sets differ in pass depth, eight atoms against four, and in
// occupancy, one warpgroup per SM against two, which is why neither divides
// into the other.
//
// **The warp-specialized rows.** Decision H3 of Part 8 section 2 splits one
// block into two roles. A producer issues every copy and issues no product, and
// the consumer warpgroups issue every product and wait on no global load: a
// stage passes between them through two `mbarrier` objects, so neither role
// waits for the other beyond the tile that is in flight. The producer widths
// the set prices are one warp, which is the first phase of the decision, and
// one warpgroup, which is the width `setmaxnreg` spans: the warpgroup gives
// back the registers it does not need and the consumers take them. The
// arithmetic one staged byte buys does not move with any of this, so the rows
// price what the separation of the issue paths is worth and nothing else.
//
// **The flash rows.** A real attention tile reads a new key block and a new
// value block every step, so the staged rows above price the score product
// alone. The flash rows price the whole tile: the score product, the base-two
// softmax in the registers, the value product from the probability operand the
// score lane already holds, and the running maximum, sum, and rescale. The step
// is a chain, because the score must retire before the softmax reads it and the
// softmax must finish before the value product issues, so a row states the cost
// of that chain. Decision H4 of Part 8 section 2 is what later hides it.
//
// The set prices the tile twice: `attention-flash:resident` with no copy path,
// and `attention-flash:staged` behind a ring of key tiles and a ring of value
// tiles. `attention-flash:copies` states the cap of each ring, because the copy
// engine fills the key ring and the transposed plain staging fills the value
// ring, and no copy engine carries a transposition. Every row is checked
// against `seqref::AttentionForward` one element at a time, and every staged
// geometry against the resident step of the same operands, so a wrong operand
// form fails before anything is timed.
//
// Two masks run: dense, and the causal boundary tile, where the query block and
// the key tile start at the same offset so a row sees the keys up to its own
// index. That tile issues a full key tile for every query row and needs the
// triangle of it, so its row states `work_ratio` as 256 over the query rows
// plus one.
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
// **The shape space.** A staged row reaches at most the copy rate of its path
// times the arithmetic that one staged byte buys, and that figure is the count
// of query rows behind one staged key tile. The counts that state it are the
// warpgroups of a block, the query tiles of each, the key width, and the ring
// depth. This file names those counts in one table and prints the table beside
// the rows it prices: an `attention-wgmma:shape` line states the threads, the
// query rows, the accumulator registers, the registers the compiler reported,
// the shared bytes, the intensity, the cap, and a verdict. A shape the register
// file or the shared window refuses is a row with its reason, and it never
// launches, so the shape a reader asks for is a row of one document. `--shapes`
// prints the table and exits, which needs no card and no measurement.
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
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "backends/cuda/kernels/device_utils.cuh"
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

// --- The Tensor Memory Accelerator ------------------------------------------
//
// Decision H2b of docs/attention.md Part 8 section 2: the copies leave the
// threads and become one instruction a stage. The map is the whole question. A
// tiled map writes a box from global memory in the box's own dimension order,
// so the axes of the map are the axes of the form it must land in: this file's
// interleaved form is one permutation of the global tile, and its five axes --
// eight elements, eight rows of one core matrix, the two halves of a sixteen
// wide block, the key groups, and the head-width blocks -- state it exactly.
// The shared order of a tiled copy is the map's own dimension order, so the map
// below fills the tile `StageCopy` fills, byte for byte, and no operand form
// changes with the copy path.

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

//: The fp32 accumulator of one score tile of `key_width` keys, in registers a
//: thread, and the slots it leaves over the warpgroup. The instruction holds
//: `kAtomM` rows by the key width, so the key width is the one lever on the
//: accumulator of a score tile, and the shape space of the section below prices
//: it. `kRegisters` above is the recorded width of this file.
constexpr int KeyRegisters(int key_width) {
  return kAtomM * key_width / kWarpgroupThreads;
}
constexpr int KeySlots(int key_width) { return kAtomM * key_width; }

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

//: The narrow key tile the shape space states beside the recorded one, and the
//: query tiles of each warpgroup that only the narrow tile can hold: half the
//: key width is half the accumulator, so one more query tile fits a tile of the
//: same register budget.
constexpr int kNarrowKeyWidth = kAtomN / 2;
constexpr int kNarrowQueryTiles = 4;

//: The query tiles one block holds at the deepest geometry of the shape space.
constexpr int kMaxTiles = kMaxWarpgroups * kNarrowQueryTiles;

constexpr int kQueryBytes = kAtomM * kDim * sizeof(Bf16);

//: A key or value tile of `key_width` keys: that many rows of the head width.
//: The tile of this file is the recorded width, and the shape space of the
//: staged rows also states the narrower one, which is a register lever.
constexpr int KeyBytes(int key_width) {
  return key_width * kDim * sizeof(Bf16);
}

constexpr int kKeyBytes = KeyBytes(kAtomN);
//: A value tile is a key tile of the same shape, and the flash tile stages
//: both.
constexpr int kValueBytes = KeyBytes(kAtomN);

//: The shared window of the value-operand sweep. The sweep launches descriptor
//: forms the instruction cannot read, so a bad form must fail on values and not
//: by addressing past the window: every read a form can issue stays inside this
//: one, which is why it is four key tiles and not one.
constexpr int kProbeSharedBytes = kQueryBytes + 4 * kKeyBytes;

//: The staged shared memory of one geometry: the query tile of every
//: warpgroup, then the ring of key tiles that all of them read.
constexpr int StagedBytes(int warpgroups, int query_tiles, int stages,
                          int key_width = kAtomN) {
  return warpgroups * query_tiles * kQueryBytes + stages * KeyBytes(key_width);
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
constexpr int FlashResidentBytes(int warpgroups, int key_width = kAtomN) {
  return warpgroups * kQueryBytes + KeyBytes(key_width) + KeyBytes(key_width);
}
static_assert(FlashResidentBytes(kMaxWarpgroups) <= kSmemCeiling,
              "the resident flash tile must fit one block per multiprocessor");

//: The ring contents a copy row can stage. A key tile alone prices the copy
//: engine, and the value tile beside it prices the transposed plain staging
//: that no copy engine carries, so the two contents are two rows and not one.
constexpr bool kRingKeyOnly = false;
constexpr bool kRingKeyValue = true;

//: The shared region of the staged flash tile: one query tile a warpgroup, a
//: ring of key tiles, and a ring of value tiles. A copy row that stages the key
//: tile alone keeps no value ring.
constexpr int FlashStagedBytes(int warpgroups, int stages, bool stage_value,
                               int key_width = kAtomN) {
  return warpgroups * kQueryBytes + stages * KeyBytes(key_width) +
         (stage_value ? stages * KeyBytes(key_width) : 0);
}
static_assert(FlashStagedBytes(kMaxWarpgroups, 3, kRingKeyValue) <=
                  kSmemCeiling,
              "the staged flash tile must fit one block per multiprocessor");

//: The shared region of the staged flash tile whose value ring is a copy engine
//: fill and a shared transposition: the query region, the key ring, the ring
//: the step reads, and the raw ring the copy fills. The transposition needs
//: both rings, so this shape holds two stages of both and not three.
constexpr int FlashSharedValueBytes(int warpgroups, int stages) {
  return warpgroups * kQueryBytes + stages * kKeyBytes +
         2 * stages * kValueBytes;
}
static_assert(FlashSharedValueBytes(kMaxWarpgroups, 2) <= kSmemCeiling,
              "the shared-transpose value ring must fit one block");
static_assert(FlashSharedValueBytes(kMaxWarpgroups, 3) > kSmemCeiling,
              "a third stage of the shared-transpose value ring cannot fit, so "
              "the shape space states two");

//: The shared region of the engine-fed tile: the query region, the swizzled key
//: ring, the raw value ring the engine fills, and the transposed value ring the
//: step reads. It is the shared-transpose tile's window, because the engine
//: replaces the staging threads and not the buffers.
constexpr int EngineFlashBytes(int consumers, int stages) {
  return consumers * kQueryBytes + stages * kKeyBytes +
         2 * stages * kValueBytes;
}
static_assert(EngineFlashBytes(2, 2) <= kSmemCeiling,
              "the engine-fed tile must fit one block per multiprocessor");

//: The shared region of the warp-specialized tile: the query tile of every
//: consumer warpgroup, a ring of key tiles, and a ring of value tiles. The
//: producer's own registers hold no tile, so the region is the same one the
//: staged flash tile asks for at the same counts.
constexpr int SpecializedBytes(int consumers, int stages, bool stage_value) {
  return consumers * kQueryBytes + stages * kKeyBytes +
         (stage_value ? stages * kValueBytes : 0);
}
static_assert(SpecializedBytes(2, 2, kRingKeyValue) <= kSmemCeiling,
              "the warp-specialized ring must fit one block per "
              "multiprocessor");
static_assert(SpecializedBytes(2, 3, kRingKeyValue) <= kSmemCeiling,
              "the third stage of both rings is the deepest this window "
              "holds");
static_assert(SpecializedBytes(2, 4, kRingKeyValue) > kSmemCeiling,
              "a fourth stage of both rings must not fit, or the shape space "
              "states a depth this file cannot launch");

//: The consumer warpgroups and the ring depths the warp-specialized rows time.
//: Two consumers is the recorded tile of Part 8 section 3, one is the first
//: phase of the decision, and a depth of two or three is every ring the shared
//: window of that tile holds.
constexpr int kSpecialOneConsumer = 1;
constexpr int kSpecialTwoConsumers = 2;
constexpr int kSpecialStages = 2;
constexpr int kSpecialDeepStages = 3;

//: The two producer widths the rows compare: one warpgroup, which `setmaxnreg`
//: spans, and one warp, which is the first phase of the decision.
constexpr int kSpecialWarpgroupProducer = kWarpgroupThreads;
constexpr int kSpecialWarpProducer = 32;

// --- The shape space --------------------------------------------------------
//
// A shape is the tile a row stages: the warpgroups of its block, the query
// tiles of each, the width of its key tile, and the depth of its ring. Those
// four counts state the whole of the rate a staged row can reach, so they are
// named in one table here, and the table is printed beside the rows it prices.
// The question the bench answers is therefore one table lookup a shape, and the
// shape that cannot launch is a row of the table and not a silence.
//
// **The intensity.** A staged key tile of `key_width` keys holds
// `key_width * kDim` elements, and the arithmetic that reads it is
// `2 * query_rows * key_width * kDim`. One staged key byte therefore buys
// `query_rows` FLOP at any key width, and the cap of a staged shape is the copy
// rate of its path times that figure.
//
// **The register identity.** One query tile is an fp32 accumulator of `kAtomM`
// rows by the key width, which is `query_rows * key_width` words for the block.
// One multiprocessor holds 65,536 registers, so `query_rows * key_width <=
// 65536` closes the whole space, and the key width is the lever that moves it:
// half the width doubles the query rows one flat register file holds.
//
// **The flash tile holds three accumulators.** The score, the output, and the
// packed probability operand. It therefore closes one step lower than the score
// product, and at the recorded 128 wide key tile the deepest full tile is two
// warpgroups of one query tile each. The narrower key tile of the candidates is
// what admits a third warpgroup, because it is the score and the probability
// that shrink with it.
//
// **Measured, not estimated.** The register count of a shape this file
// instantiates is the one its own build states, read with
// `cudaFuncGetAttributes`. A shape with no instantiation is charged
// `kScoreStagingRegisters` or `kFlashStagingRegisters` over its accumulators,
// which are the largest overheads one build of this file measures with
// `cuobjdump -res-usage` over the twelve instantiations: 40 to 62 registers
// over the accumulators of a staged score tile, and 8 to 37 over those of a
// staged flash tile.

//: The fp32 registers one multiprocessor holds, and the threads one block may
//: have. Both are properties of the card and not of a tile.
constexpr int kRegistersPerMultiprocessor = 65536;
constexpr int kMaxThreadsPerBlock = 1024;

//: The registers a thread spends on the staging path, the descriptors, and the
//: online state of a shape that holds no instantiation, over its accumulators.
//: Measured, as above: one build reads 40 to 62 over the accumulators of the
//: staged score tile and 8 to 37 over those of the staged flash tile, and a
//: prediction that no compiler has met is charged the largest of its family.
constexpr int kScoreStagingRegisters = 62;
constexpr int kFlashStagingRegisters = 37;

//: `nanochat::dev::Gflops` states a rate in GFLOP/s, so a diagnostic that
//: prints TFLOP/s multiplies by this. A division by 1e12 states a TFLOP/s
//: figure that is a thousand million times too small, which is what the first
//: card run of this table did to every escalation below.
constexpr double kTflopsPerGflops = 1e-3;

//: The copy rate the sixteen-byte store path measures in the recorded run, 2.36
//: TB/s, from the `attention-wgmma:copies` rows of the `attention_wgmma_probe`
//: entry of docs/baselines.json. A staged shape's cap is that figure times its
//: intensity, and the row states the share of it the shape reached.
constexpr double kRecordedCopyBytesPerSecond = 2.36e12;

enum class ShapeKind { kScore, kFlash, kFlashResident, kSpecial };

const char* ShapeKindName(ShapeKind kind) {
  switch (kind) {
    case ShapeKind::kScore:
      return "score";
    case ShapeKind::kFlash:
      return "flash";
    case ShapeKind::kFlashResident:
      return "resident";
    case ShapeKind::kSpecial:
      return "special";
  }
  return "unknown";
}

//: The fp32 accumulator of one query tile, in registers a thread: the
//: instruction holds `kAtomM` rows by the key width over one warpgroup.
constexpr int ScoreAccumulators(int key_width) {
  return KeyRegisters(key_width);
}

//: The output accumulator of one flash tile, which is as wide as the head.
constexpr int OutputAccumulators() { return kAtomM * kDim / kWarpgroupThreads; }

//: The probability operand of one flash tile, in packed pairs: one word holds
//: two keys, and the tile holds `key_width / kAtomK` blocks of four words.
constexpr int ProbabilityRegisters(int key_width) {
  return (key_width / kAtomK) * 4;
}

//: Every accumulator one query tile of `kind` holds, in registers a thread.
constexpr int TileAccumulators(ShapeKind kind, int key_width) {
  const int score = ScoreAccumulators(key_width);
  if (kind == ShapeKind::kScore) return score;
  return score + OutputAccumulators() + ProbabilityRegisters(key_width);
}

// One shape of the sweep.
struct ShapeSpec {
  //: The name the table and the document's extras use.
  const char* name;
  ShapeKind kind;
  int warpgroups;
  //: The query tiles of each warpgroup. A flash tile holds one.
  int query_tiles;
  int key_width;
  //: The depth of the ring, and one for a resident shape, which holds none.
  int stages;
  //: Whether this file instantiates the shape, so its register count is the
  //: compiler's own.
  bool instantiated;
  //: What the row is for, when it is not a shape the sweep launches.
  const char* note;
  //: The threads of the producer role of a warp-specialized shape, and zero for
  //: a shape whose own threads issue the copies.
  int producer_threads = 0;
  //: Whether the value ring is a copy engine fill and a shared transposition
  //: instead of the transposing plain staging.
  bool shared_value = false;
};

//: The shape space. The thirteen rows down to `score64:2x4` are the geometries
//: the sweep launches, three of them at the narrow key tile. The three
//: candidates below them are the shapes the register identity of this section
//: opens and closes, and they are stated here because a reader of the table is
//: asking which shape should be written next.
constexpr ShapeSpec kShapeSpace[] = {
    {"score:1x1", ShapeKind::kScore, 1, 1, kAtomN, kStages, true, nullptr},
    {"score:1x2", ShapeKind::kScore, 1, 2, kAtomN, kStages, true, nullptr},
    {"score:1x3", ShapeKind::kScore, 1, 3, kAtomN, kStages, true, nullptr},
    {"score:2x1", ShapeKind::kScore, 2, 1, kAtomN, kStages, true, nullptr},
    {"score:2x2", ShapeKind::kScore, 2, 2, kAtomN, kStages, true, nullptr},
    {"score:2x3", ShapeKind::kScore, 2, 3, kAtomN, kStages, true, nullptr},
    {"flash:1:2", ShapeKind::kFlash, 1, 1, kAtomN, 2, true, nullptr},
    {"flash:1:3", ShapeKind::kFlash, 1, 1, kAtomN, 3, true, nullptr},
    {"flash:2:2", ShapeKind::kFlash, 2, 1, kAtomN, 2, true, nullptr},
    {"flash:2:3", ShapeKind::kFlash, 2, 1, kAtomN, 3, true, nullptr},
    {"resident:1", ShapeKind::kFlashResident, 1, 1, kAtomN, 1, true, nullptr},
    {"resident:2", ShapeKind::kFlashResident, 2, 1, kAtomN, 1, true, nullptr},
    {"score64:2x4", ShapeKind::kScore, kMaxWarpgroups, kNarrowQueryTiles,
     kNarrowKeyWidth, kStages, true, nullptr},
    {"cand:score64-4x4", ShapeKind::kScore, 4, 4, kNarrowKeyWidth, kStages,
     false,
     "the same intensity over four warpgroups, which the register file closes"},
    {"cand:flash64-3", ShapeKind::kFlash, 3, 1, kNarrowKeyWidth, 3, false,
     "three warpgroups of one query tile, the deepest full tile that fits"},
    {"cand:flash128-3", ShapeKind::kFlash, 3, 1, kAtomN, 3, false,
     "the same depth at the recorded key width, which the register file "
     "closes"},
    {"special:p32c1:2", ShapeKind::kSpecial, kSpecialOneConsumer, 1, kAtomN,
     kSpecialStages, true, nullptr, kSpecialWarpProducer},
    {"special:p32c1:3", ShapeKind::kSpecial, kSpecialOneConsumer, 1, kAtomN,
     kSpecialDeepStages, true, nullptr, kSpecialWarpProducer},
    {"special:p32c2:2", ShapeKind::kSpecial, kSpecialTwoConsumers, 1, kAtomN,
     kSpecialStages, true, nullptr, kSpecialWarpProducer},
    {"special:p32c2:3", ShapeKind::kSpecial, kSpecialTwoConsumers, 1, kAtomN,
     kSpecialDeepStages, true, nullptr, kSpecialWarpProducer},
    {"special:p128c1:2", ShapeKind::kSpecial, kSpecialOneConsumer, 1, kAtomN,
     kSpecialStages, true, nullptr, kSpecialWarpgroupProducer},
    {"special:p128c1:3", ShapeKind::kSpecial, kSpecialOneConsumer, 1, kAtomN,
     kSpecialDeepStages, true, nullptr, kSpecialWarpgroupProducer},
    {"special:p128c2:2", ShapeKind::kSpecial, kSpecialTwoConsumers, 1, kAtomN,
     kSpecialStages, true, nullptr, kSpecialWarpgroupProducer},
    {"special:p128c2:3", ShapeKind::kSpecial, kSpecialTwoConsumers, 1, kAtomN,
     kSpecialDeepStages, true, nullptr, kSpecialWarpgroupProducer},
    {"flash:2:2:shared", ShapeKind::kFlash, kMaxWarpgroups, 1, kAtomN, 2, true,
     nullptr, 0, true},
};

//: The shared bytes one shape requests, from the tile functions above, so the
//: table and the launch agree on the window by construction.
constexpr int SharedBytesOf(const ShapeSpec& shape) {
  if (shape.kind == ShapeKind::kFlashResident) {
    return FlashResidentBytes(shape.warpgroups, shape.key_width);
  }
  if (shape.kind == ShapeKind::kFlash) {
    if (shape.shared_value) {
      return FlashSharedValueBytes(shape.warpgroups, shape.stages);
    }
    return FlashStagedBytes(shape.warpgroups, shape.stages, kRingKeyValue,
                            shape.key_width);
  }
  if (shape.kind == ShapeKind::kSpecial) {
    return SpecializedBytes(shape.warpgroups, shape.stages, kRingKeyValue);
  }
  return StagedBytes(shape.warpgroups, shape.query_tiles, shape.stages,
                     shape.key_width);
}

// What one shape implies, and whether it holds its place.
struct ShapeFacts {
  int threads = 0;
  //: The query rows behind one staged key tile, which is the intensity.
  int query_rows = 0;
  int accumulator_registers = 0;
  int registers = 0;
  //: Whether `registers` is the compiler's count or the arithmetic estimate.
  bool registers_from_compiler = false;
  int registers_per_multiprocessor = 0;
  int shared_bytes = 0;
  int intensity = 0;
  double copy_cap_tflops = 0.0;
  bool fits = false;
  //: The limit that refused the shape, and `fits` when none did.
  const char* reason = "fits";
};

// The facts of one shape. `compiler_registers` is the count the shape's own
// build states, and zero falls back to the staging charge of the tile's family
// over its accumulators.
ShapeFacts FactsOf(const ShapeSpec& shape, int compiler_registers) {
  ShapeFacts facts;
  facts.threads = shape.kind == ShapeKind::kSpecial
                      ? (1 + shape.warpgroups) * kWarpgroupThreads
                      : shape.warpgroups * kWarpgroupThreads;
  facts.query_rows = shape.warpgroups * shape.query_tiles * kAtomM;
  facts.accumulator_registers =
      shape.query_tiles * TileAccumulators(shape.kind, shape.key_width);
  facts.registers_from_compiler = compiler_registers > 0;
  facts.registers =
      facts.registers_from_compiler
          ? compiler_registers
          : facts.accumulator_registers + (shape.kind == ShapeKind::kScore
                                               ? kScoreStagingRegisters
                                               : kFlashStagingRegisters);
  facts.registers_per_multiprocessor = facts.threads * facts.registers;
  facts.shared_bytes = SharedBytesOf(shape);
  facts.intensity =
      shape.kind == ShapeKind::kFlashResident ? 0 : facts.query_rows;
  facts.copy_cap_tflops =
      static_cast<double>(facts.intensity) * kRecordedCopyBytesPerSecond / 1e12;
  if (facts.threads > kMaxThreadsPerBlock) {
    facts.reason = "threads";
  } else if (facts.registers_per_multiprocessor > kRegistersPerMultiprocessor) {
    facts.reason = "registers";
  } else if (facts.shared_bytes > kSmemCeiling) {
    facts.reason = "shared_memory";
  } else if (shape.kind != ShapeKind::kFlashResident && shape.stages < 2) {
    facts.reason = "ring";
  } else {
    facts.fits = true;
  }
  return facts;
}

// The row of the shape space that matches one geometry. A geometry the table
// does not hold is a shape no reader can audit, so it stops the run instead of
// timing quietly.
const ShapeSpec& FindShape(ShapeKind kind, int warpgroups, int query_tiles,
                           int key_width, int stages,
                           bool shared_value = false) {
  for (const ShapeSpec& shape : kShapeSpace) {
    if (shape.kind == kind && shape.warpgroups == warpgroups &&
        shape.query_tiles == query_tiles && shape.key_width == key_width &&
        shape.stages == stages && shape.shared_value == shared_value) {
      return shape;
    }
  }
  std::fprintf(stderr,
               "the shape space holds no %s geometry of %d warpgroups, %d "
               "query tiles, %d keys, and %d stages\n",
               ShapeKindName(kind), warpgroups, query_tiles, key_width, stages);
  std::exit(2);
}

// The registers one instantiation spends a thread, as its own build states it.
// Zero when the query fails, which makes the table state its estimate instead.
// The entry is a `const void*` because the address of a kernel that a template
// parameter names does not deduce a type in a lambda this file instantiates.
int CompilerRegisters(const void* kernel) {
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) return 0;
  return attributes.numRegs;
}

// The shape space of one run: the facts of every shape the run states, the
// compiler's register count where the shape holds an instantiation, and the
// fastest rate the sweep measured for it. `Print` writes the table once, at the
// end of the run, so the shape that reached the highest rate and the shape that
// cannot launch are one document.
class ShapeTable {
 public:
  // States one shape and answers whether it may launch. The caller states a
  // refusal through its own diagnostic line, so this class holds no output of
  // its own and the document stays the one writer of a machine-readable run.
  ShapeFacts Observe(const ShapeSpec& shape, int compiler_registers) {
    Entry* entry = Find(shape);
    if (entry == nullptr) {
      entries_.push_back(Entry{shape, compiler_registers, 0.0});
      entry = &entries_.back();
    } else if (compiler_registers > 0) {
      entry->compiler_registers = compiler_registers;
    }
    return FactsOf(entry->shape, entry->compiler_registers);
  }

  // The fastest rate one shape stated, in TFLOP/s. A shape the sweep times
  // twice keeps the faster row, because that is the row a decision reads.
  void Measure(const ShapeSpec& shape, double tflops) {
    Entry* entry = Find(shape);
    if (entry == nullptr) {
      entries_.push_back(Entry{shape, 0, tflops});
      return;
    }
    entry->tflops = std::max(entry->tflops, tflops);
  }

  // The table the document carries, one extra a field, so a reader of the
  // document compares two shapes without a second run.
  void Write(nanochat::BenchReport* report) const {
    int instantiated = 0;
    int refused = 0;
    const Entry* best = nullptr;
    for (const Entry& entry : entries_) {
      const ShapeFacts facts = FactsOf(entry.shape, entry.compiler_registers);
      const std::string prefix = std::string("shape:") + entry.shape.name + ":";
      report->SetExtraInt(prefix + "intensity_flop_per_byte", facts.intensity);
      report->SetExtraInt(prefix + "accumulator_registers",
                          facts.accumulator_registers);
      report->SetExtraInt(prefix + "registers", facts.registers);
      report->SetExtraInt(prefix + "registers_per_multiprocessor",
                          facts.registers_per_multiprocessor);
      report->SetExtraInt(prefix + "shared_bytes", facts.shared_bytes);
      report->SetExtra(prefix + "copy_cap_tflops", facts.copy_cap_tflops);
      report->SetExtraInt(prefix + "fits", facts.fits ? 1 : 0);
      if (!facts.fits) {
        report->SetExtraText(prefix + "verdict", facts.reason);
        ++refused;
      }
      if (entry.shape.instantiated) ++instantiated;
      if (entry.tflops > 0.0) {
        report->SetExtra(prefix + "tflops", entry.tflops);
        if (facts.intensity > 0) {
          report->SetExtra(prefix + "copy_rate_used_tbps",
                           entry.tflops * 1e12 / facts.intensity / 1e12);
        }
        if (facts.copy_cap_tflops > 0.0) {
          report->SetExtra(prefix + "copy_path_used",
                           entry.tflops / facts.copy_cap_tflops);
        }
        if (best == nullptr || entry.tflops > best->tflops) best = &entry;
      }
    }
    report->SetExtraInt("shape:count", static_cast<long long>(entries_.size()));
    report->SetExtraInt("shape:instantiated", instantiated);
    report->SetExtraInt("shape:refused", refused);
    if (best != nullptr) {
      report->SetExtraText("shape:best", best->shape.name);
      report->SetExtra("shape:best_tflops", best->tflops);
    }
  }

  // The table a human run reads: one line a shape, in the order the run stated
  // them, then the count and the fastest row. `peak_tflops` is the declared
  // peak rate the run measured against, and zero means the run named none, as a
  // `--shapes` run does: the compute share is then left out and no share is
  // stated against a rate the run did not use.
  void Print(std::FILE* out, double peak_tflops) const {
    int refused = 0;
    const Entry* best = nullptr;
    for (const Entry& entry : entries_) {
      const ShapeFacts facts = FactsOf(entry.shape, entry.compiler_registers);
      std::fprintf(out,
                   "attention-wgmma:shape name=%s kind=%s warpgroups=%d "
                   "query_tiles=%d keys=%d stages=%d threads=%d query_rows=%d "
                   "producer_threads=%d accumulators=%d registers=%d "
                   "registers_source=%s registers_per_sm=%d smem_bytes=%d "
                   "intensity_flop_per_byte=%d copy_cap_tflops=%.1f",
                   entry.shape.name, ShapeKindName(entry.shape.kind),
                   entry.shape.warpgroups, entry.shape.query_tiles,
                   entry.shape.key_width, entry.shape.stages, facts.threads,
                   facts.query_rows, entry.shape.producer_threads,
                   facts.accumulator_registers, facts.registers,
                   facts.registers_from_compiler ? "compiler" : "estimate",
                   facts.registers_per_multiprocessor, facts.shared_bytes,
                   facts.intensity, facts.copy_cap_tflops);
      if (entry.tflops > 0.0) {
        // The rate, then the two resources it is a share of. `copy_rate_used`
        // is the rate divided by the intensity, which is the copy bytes a
        // second the row consumed, and `copy_path_used` is that over the
        // measured copy rate. The two are one number twice, because the cap is
        // the intensity times that rate. `compute_used` is the rate over the
        // declared peak, and it is the same share the document's `mfu` states.
        const double copy_rate_used = entry.tflops * 1e12 / facts.intensity;
        const double copy_path_used = facts.copy_cap_tflops > 0.0
                                          ? entry.tflops / facts.copy_cap_tflops
                                          : 0.0;
        std::fprintf(out,
                     " tflops=%.1f copy_rate_used_tbps=%.2f "
                     "copy_path_used=%.3f",
                     entry.tflops, copy_rate_used / 1e12, copy_path_used);
        if (peak_tflops > 0.0) {
          std::fprintf(out, " compute_used=%.3f", entry.tflops / peak_tflops);
        }
      }
      std::fprintf(out, " verdict=%s", facts.fits ? "fits" : facts.reason);
      if (entry.shape.note != nullptr) {
        std::fprintf(out, " note=\"%s\"", entry.shape.note);
      }
      std::fprintf(out, "\n");
      if (!facts.fits) ++refused;
      if (entry.tflops > 0.0 &&
          (best == nullptr || entry.tflops > best->tflops)) {
        best = &entry;
      }
    }
    std::fflush(out);
    if (best != nullptr) {
      std::fprintf(out,
                   "attention-wgmma:shape_summary shapes=%zu refused=%d "
                   "best=%s best_tflops=%.1f\n",
                   entries_.size(), refused, best->shape.name, best->tflops);
    } else {
      std::fprintf(out,
                   "attention-wgmma:shape_summary shapes=%zu refused=%d "
                   "best=none best_tflops=0.0\n",
                   entries_.size(), refused);
    }
    std::fflush(out);
  }

 private:
  struct Entry {
    ShapeSpec shape;
    int compiler_registers = 0;
    double tflops = 0.0;
  };

  Entry* Find(const ShapeSpec& shape) {
    for (Entry& entry : entries_) {
      if (std::strcmp(entry.shape.name, shape.name) == 0) return &entry;
    }
    return nullptr;
  }

  std::vector<Entry> entries_;
};

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
static_assert(kAtoms * AtomStep(kNarrowKeyWidth) * 16 ==
                  KeyBytes(kNarrowKeyWidth),
              "the narrow key tile must fill its interleaved form");
//: The accumulator map of `m64n128k16` covers its tile exactly, and the narrow
//: tile is that map cut off at half the columns: 128 lanes of 32 registers are
//: the 64 by 64 words of one `m64n64k16`, with the same row formula and a
//: column that stops at 63. The check on the card settles it one element at a
//: time.
static_assert(kWarpgroupThreads * KeyRegisters(kNarrowKeyWidth) ==
                  kAtomM * kNarrowKeyWidth,
              "the narrow accumulator must cover its tile");

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
constexpr double FlopsPerPass(int atoms, int key_width = kAtomN) {
  return 2.0 * kAtomM * key_width * kAtomK * atoms;
}

//: The slots one product of the recorded key tile leaves, over the warpgroup.
constexpr std::size_t kProductSlots =
    static_cast<std::size_t>(KeySlots(kAtomN));

// --- The staging ------------------------------------------------------------

// The interleaved offset of one element of a row-major source tile. The host
// states it too, because a check compares a staged tile with the tile the form
// names and the copy engine's map states the same five axes.
__host__ __device__ __forceinline__ int InterleavedOffset(int row, int k,
                                                          int rows) {
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
                                          bool row_fast, int threads,
                                          int rows) {
  const int groups = rows * kGroupsPerRow;
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
    CopyAsync16(stage + InterleavedOffset(row, k, rows),
                source + row * kDim + k);
  }
}

// The shared address of one byte offset of a row-major tile with the chunk
// swizzle of the layout applied. Whole sixteen-byte chunks are permuted, so the
// copy engine can target the result and no transpose is needed.
__device__ __forceinline__ int SwizzleBytes(int offset, int layout) {
  if (layout == 1) return offset ^ (((offset >> 7) & 7) << 4);
  if (layout == 2) return offset ^ (((offset >> 6) & 7) << 4);
  if (layout == 3) return offset ^ (((offset >> 7) & 3) << 4);
  // The canonical 128-byte swizzle of a Tensor Memory Accelerator tile: the
  // atom is eight rows of a hundred and twenty-eight bytes, so the chunk index
  // within the atom is exclusive-ored with the row of the atom and the atom
  // itself is a kilobyte. A tile wider than one atom holds several side by
  // side, which is what separates this layout from the three above.
  if (layout == 4) {
    const int atom = offset / 1024;
    const int within = offset - atom * 1024;
    return atom * 1024 + (within ^ (((within >> 7) & 7) << 4));
  }
  return offset;
}

// Stages a tile in its own row order with that swizzle, which is the layout a
// swizzled descriptor reads. One sixteen-byte group of a source row is one
// store, so the swizzle costs only the address and the pass stays a copy.
__device__ __forceinline__ void StageSwizzled(char* stage, const Bf16* source,
                                              int rows, int threads,
                                              int layout) {
  const int groups = rows * kGroupsPerRow;
  for (int unit = static_cast<int>(threadIdx.x); unit < groups;
       unit += threads) {
    const int row = unit / kGroupsPerRow;
    const int group = (unit - row * kGroupsPerRow) * kElementsPerGroup;
    // The panel form of a swizzled tile: the engine's swizzle spans a hundred
    // and twenty-eight bytes, so a row wider than that is stored as panels side
    // by side, each eight rows of sixteen chunks with the chunk exclusive-ored
    // by the row of its atom.
    const int offset =
        layout == 5
            ? (group / 64) * (rows * 128) + row * 128 +
                  ((((group % 64) / kElementsPerGroup) ^ (row & 7)) * 16)
            : SwizzleBytes((row * kDim + group) * 2, layout);
    *reinterpret_cast<uint4*>(stage + offset) =
        *reinterpret_cast<const uint4*>(source + row * kDim + group);
  }
}

// The same sixteen-byte groups the copy engine moves, written by ordinary
// stores. The control for the value ring: if this reads the key ring's bytes at
// the value ring's rate, then the transpose is innocent and the plain store
// path is the cost.
__device__ __forceinline__ void StageCopyPlain(char* stage, const Bf16* source,
                                               bool row_fast, int threads,
                                               int rows) {
  const int groups = rows * kGroupsPerRow;
#pragma unroll 4
  for (int unit = static_cast<int>(threadIdx.x); unit < groups;
       unit += threads) {
    int row = 0;
    int k = 0;
    if (row_fast) {
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
    *reinterpret_cast<uint4*>(stage + InterleavedOffset(row, k, rows)) =
        *reinterpret_cast<const uint4*>(source + row * kDim + k);
  }
}

// The same transposed staging read the other way round: one sixteen-byte group
// of a source row per thread, which is coalesced across the warp, and the eight
// two-byte elements it holds written to eight rows of the interleaved tile. The
// strided form above does the opposite and its loads are the strided ones, so
// this pass separates a load-bound transpose from a store-bound one.
__device__ __forceinline__ void StageTransposedVector(char* stage,
                                                      const Bf16* source,
                                                      int threads) {
  const int groups = kAtomN * kGroupsPerRow;
#pragma unroll 4
  for (int unit = static_cast<int>(threadIdx.x); unit < groups;
       unit += threads) {
    const int key = unit / kGroupsPerRow;
    const int dim = (unit - key * kGroupsPerRow) * kElementsPerGroup;
    const uint4 loaded =
        *reinterpret_cast<const uint4*>(source + key * kDim + dim);
    const Bf16* elements = reinterpret_cast<const Bf16*>(&loaded);
#pragma unroll
    for (int j = 0; j < kElementsPerGroup; ++j) {
      *reinterpret_cast<Bf16*>(stage + InterleavedOffset(dim + j, key, kDim)) =
          elements[j];
    }
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
#pragma unroll 4
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

// The same transposition, from one shared tile into another. One thread takes
// one core matrix of the staged value tile -- eight keys by eight head-width
// elements, a contiguous hundred and twenty-eight bytes -- and writes its
// transpose into the tile the value product reads. So the pass costs shared
// traffic and no global read, and the tile a copy engine fills stays exactly
// the tile the key ring fills.
//
// The two tiles are one permutation of each other at the granularity of the
// core matrix: the raw tile keeps eight keys of one head-width block together,
// and the transposed tile keeps eight head-width elements of one key block
// together. `InterleavedOffset` states both, so this pass is the mapping
// between them and one eight by eight transpose inside each unit.
__device__ __forceinline__ void StageTransposeShared(char* stage,
                                                     const char* source,
                                                     int threads) {
  // One unit is eight keys of one key group and eight elements of one
  // head-width half: sixteen key groups, eight head-width blocks, and two
  // halves a tile.
  constexpr int kUnits = (kAtomN / 8) * (kDim / 16) * 2;
  for (int unit = static_cast<int>(threadIdx.x); unit < kUnits;
       unit += threads) {
    const int half = unit % 2;
    const int block = (unit / 2) % (kDim / 16);
    const int group = unit / (2 * (kDim / 16));
    const char* from =
        source + InterleavedOffset(group * 8, block * 16 + half * 8, kAtomN);
    char* to =
        stage + InterleavedOffset((block * 2 + half) * 8, group * 8, kDim);
    // The source rows are keys and its columns are head-width elements, so the
    // destination row of one element gathers one element of every source row.
    // Two adjacent destination elements are one packed pair, taken from two
    // source rows by `__byte_perm`, and the half of the source word and the
    // selector follow the destination column.
    std::uint32_t rows[8][4];
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      const uint4 loaded = *reinterpret_cast<const uint4*>(from + row * 16);
      rows[row][0] = loaded.x;
      rows[row][1] = loaded.y;
      rows[row][2] = loaded.z;
      rows[row][3] = loaded.w;
    }
#pragma unroll
    for (int column = 0; column < 8; ++column) {
      const int word = column / 2;
      const std::uint32_t selector = (column % 2) != 0 ? 0x7632u : 0x5410u;
      uint4 out;
      out.x = __byte_perm(rows[0][word], rows[1][word], selector);
      out.y = __byte_perm(rows[2][word], rows[3][word], selector);
      out.z = __byte_perm(rows[4][word], rows[5][word], selector);
      out.w = __byte_perm(rows[6][word], rows[7][word], selector);
      *reinterpret_cast<uint4*>(to + column * 16) = out;
    }
  }
}

// The same transposition, with the raw tile in the swizzled panel form the copy
// engine fills. The swizzle permutes whole sixteen-byte chunks and is its own
// inverse, so one read of the source states where a core matrix sits: the panel
// of the head-width block, the row of the key, and the chunk exclusive-ored by
// the row of its atom.
// `kRotate` states the order the pieces of one core matrix leave a lane. A lane
// holds eight keys of one group, so its eight destination pieces are 128
// contiguous bytes, and every lane's run starts on a 128-byte boundary, which
// is one full bank sweep: 32 lanes then put their 16 bytes on 4 banks, which is
// an 8-way conflict. Moving the piece a lane writes by the lane's own index
// spreads the same 32 lanes over 8 distinct 16-byte offsets and halves the
// store time. It states the same tile, so the rows that check it are the same
// rows.
template <bool kRotate = false>
__device__ __forceinline__ void StageTransposeSwizzled(char* stage,
                                                       const char* source,
                                                       int threads,
                                                       int index = -1) {
  constexpr int kUnits = (kAtomN / 8) * (kDim / 16) * 2;
  constexpr int kPanelBytes = kAtomN * 64 * sizeof(Bf16);
  // `index` is the caller's place in the pass, which is the thread's own index
  // unless a subset of the block runs it.
  const int me = index >= 0 ? index : static_cast<int>(threadIdx.x);
  for (int unit = me; unit < kUnits; unit += threads) {
    const int half = unit % 2;
    const int block = (unit / 2) % (kDim / 16);
    const int group = unit / (2 * (kDim / 16));
    const int column = block * 16 + half * 8;
    const int panel = column / 64;
    const int chunk = ((column % 64) / 8);
    char* to =
        stage + InterleavedOffset((block * 2 + half) * 8, group * 8, kDim);
    // One source row is one key: eight head-width elements, sixteen bytes, at
    // the chunk its atom's row names.
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
    for (int step = 0; step < 8; ++step) {
      const int out_column = kRotate ? ((step + me) & 7) : step;
      const int word = out_column / 2;
      const std::uint32_t selector = (out_column % 2) != 0 ? 0x7632u : 0x5410u;
      uint4 out;
      out.x = __byte_perm(rows[0][word], rows[1][word], selector);
      out.y = __byte_perm(rows[2][word], rows[3][word], selector);
      out.z = __byte_perm(rows[4][word], rows[5][word], selector);
      out.w = __byte_perm(rows[6][word], rows[7][word], selector);
      *reinterpret_cast<uint4*>(to + out_column * 16) = out;
    }
  }
}

// The same transposition by warp instructions: one `ldmatrix.trans` reads four
// core matrices and hands every lane the transposed fragment, and four stores a
// lane write those rows back. A lane holds row (lane / 4) and the pair
// (lane % 4) of each matrix, so one four-byte store a matrix states the row.
__device__ __forceinline__ void StageTransposeSwizzledWarp(char* stage,
                                                           const char* source,
                                                           int threads) {
  constexpr int kUnits = (kAtomN / 8) * (kDim / 16) * 2;
  constexpr int kPanelBytes = kAtomN * 64 * sizeof(Bf16);
  const int warp = static_cast<int>(threadIdx.x) % threads / 32;
  const int warps = threads / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  for (int group = warp; group < kUnits / 4; group += warps) {
    // The unit this lane helps load: the lane names the matrix and its row.
    const int unit = group * 4 + lane / 8;
    const int half = unit % 2;
    const int block = (unit / 2) % (kDim / 16);
    const int keygroup = unit / (2 * (kDim / 16));
    const int column = block * 16 + half * 8;
    const int panel = column / 64;
    const int chunk = (column % 64) / 8;
    const int key = keygroup * 8 + (lane % 8);
    const char* address =
        source + panel * kPanelBytes + key * 128 + ((chunk ^ (key & 7)) * 16);
    const std::uint32_t shared =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
    std::uint32_t r0 = 0;
    std::uint32_t r1 = 0;
    std::uint32_t r2 = 0;
    std::uint32_t r3 = 0;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(shared));
    const int offset = (lane / 4) * 16 + (lane % 4) * 4;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const std::uint32_t value = part == 0   ? r0
                                  : part == 1 ? r1
                                  : part == 2 ? r2
                                              : r3;
      const int unit_of_part = group * 4 + part;
      const int part_half = unit_of_part % 2;
      const int part_block = (unit_of_part / 2) % (kDim / 16);
      const int part_group = unit_of_part / (2 * (kDim / 16));
      char* destination =
          stage + InterleavedOffset((part_block * 2 + part_half) * 8,
                                    part_group * 8, kDim);
      *reinterpret_cast<std::uint32_t*>(destination + offset) = value;
    }
  }
}

// --- The descriptor ---------------------------------------------------------

// The six fields of the shared-memory matrix descriptor: the address in sixteen
// byte units, the leading offset, the stride offset, base offset zero, and
// layout type zero for the interleaved form.
// The same fields with the leading offset, the stride offset, and the layout
// type stated by the caller, so a sweep can try another operand form against
// the same staged tile. Layout zero is the interleaved form this file stages;
// layout one is the 128-byte swizzle, which a sixteen-byte copy can still
// target because the swizzle permutes whole chunks.
__device__ __forceinline__ std::uint64_t MakeDescriptorFull(const void* base,
                                                            int lbo_bytes,
                                                            int sbo_bytes,
                                                            int layout) {
  const std::uint32_t address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(base));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((address >> 4) & 0x3FFFu);
  desc |= static_cast<std::uint64_t>((lbo_bytes >> 4) & 0x3FFFu) << 16;
  desc |= static_cast<std::uint64_t>((sbo_bytes >> 4) & 0x3FFFu) << 32;
  desc |= static_cast<std::uint64_t>(layout & 0x3) << 62;
  return desc;
}

__device__ __forceinline__ std::uint64_t MakeDescriptorBytes(const void* base,
                                                             int lbo_bytes,
                                                             int sbo_bytes) {
  return MakeDescriptorFull(base, lbo_bytes, sbo_bytes, /*layout=*/0);
}

__device__ __forceinline__ std::uint64_t MakeDescriptor(const void* base) {
  return MakeDescriptorBytes(base, kLboBytes, kSboBytes);
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

// The same atom over a key tile of 64 keys. `m64n64k16` holds thirty-two fp32
// accumulators a thread where `m64n128k16` holds sixty-four, and every other
// field of the instruction is the same, so the narrow key tile is exactly the
// halving the register identity of the shape space states. One core matrix is
// still eight rows of eight elements, so the descriptor form and the atom step
// are the ones the wide tile uses, read at half the rows.
//
// The accumulator map is the wide one with its column field stopped at 63,
// which is what `SlotRow` and `SlotColumn` state when the index runs to
// thirty-two. That is a claim about the instruction, so `form_check_narrow`
// measures it one element at a time before any narrow row is timed.
__device__ __forceinline__ void WgmmaM64N64K16(std::uint64_t desc_a,
                                               std::uint64_t desc_b, float* d,
                                               bool accumulate) {
  const std::uint32_t scale = accumulate ? 1u : 0u;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %34, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31}, "
      "  %32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale));
}

// One atom of the score product over a key tile of `KeyWidth` keys, which is
// the one place the two instruction forms differ in this file.
template <int KeyWidth>
__device__ __forceinline__ void ScoreAtom(std::uint64_t desc_a,
                                          std::uint64_t desc_b, float* d,
                                          bool accumulate) {
  static_assert(KeyWidth == kNarrowKeyWidth || KeyWidth == kAtomN,
                "the score atom reads a key tile of 64 or 128 keys");
  if constexpr (KeyWidth == kAtomN) {
    WgmmaM64N128K16(desc_a, desc_b, d, accumulate);
  } else {
    WgmmaM64N64K16(desc_a, desc_b, d, accumulate);
  }
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

// --- The handoff of a warp-specialized block --------------------------------
//
// A producer that issues the copies and consumers that issue the products do
// not share a barrier of the block, because a `__syncthreads` there makes the
// slower side wait for the faster one every step. They hand a stage over
// through two `mbarrier` objects instead: `ready` says the stage holds what the
// step reads, and `released` says every consumer warpgroup has drained it.

// The shared address of one object, in the form the PTX below takes.
__device__ __forceinline__ std::uint32_t SharedWord(const void* address) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
}

// Arms one barrier for `arrivals` arrivals. A stage whose value tile is staged
// arms two arrivals a producer thread and one that stages the key tile alone
// arms one, so the count is the caller's.
__device__ __forceinline__ void BarrierInit(std::uint64_t* barrier,
                                            int arrivals) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(SharedWord(barrier)),
      "r"(arrivals));
}

// Arrives once from this thread, with the release that carries the stores made
// before it to the thread that waits.
__device__ __forceinline__ void BarrierArrive(std::uint64_t* barrier) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(SharedWord(barrier))
      : "memory");
}

// States that this thread's asynchronous copies complete the barrier, so the
// producer issues a stage and moves to the next one instead of waiting for its
// own copy. `.noinc` raises no pending count, because the arm already holds
// this arrival.
__device__ __forceinline__ void BarrierArriveOnCopies(std::uint64_t* barrier) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(
                   SharedWord(barrier))
               : "memory");
}

// Waits for the completion of the phase whose parity is `phase`. The poll is a
// `try_wait` in a C++ loop and not a PTX label, so the wait may stand at many
// inlined sites and no two labels meet.
__device__ __forceinline__ void BarrierWait(std::uint64_t* barrier, int phase) {
  std::uint32_t complete = 0;
  do {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(complete)
        : "r"(SharedWord(barrier)), "r"(phase)
        : "memory");
  } while (complete == 0);
}

// The register budget of the two roles, in registers a thread. `setmaxnreg`
// moves registers between warpgroups of one block, so the producer gives back
// what it does not need and the consumers take it. The instruction spans a
// warpgroup, so every thread of one runs it and no other thread does.
template <int Count>
__device__ __forceinline__ void WarpgroupRegistersGive() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Count) : "memory");
}

// The other direction: the calling warpgroup takes `Count` registers a thread
// from the pool its block released.
template <int Count>
__device__ __forceinline__ void WarpgroupRegistersTake() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Count) : "memory");
}

//: The descriptor of a swizzled key tile, in sixteen-byte units: one core
//: matrix group is eight rows of a hundred and twenty-eight bytes, so the atom
//: step is two units and the panel is a kilobyte of them. Both are the fields
//: the engine's reading was measured with.
constexpr int kSwizzledAtomUnits = 2;
constexpr int kPanelStepUnits = 1024;
constexpr int kSwizzledLeadingBytes = 16;
constexpr int kSwizzledStrideBytes = 1024;

//: The registers the producer warpgroup of a warp-specialized block keeps, and
//: the registers a consumer warpgroup asks for. The producer holds addresses,
//: loop counters, and a staging slice, so it needs little. The two counts are a
//: multiple of eight, which the instruction requires, and their total over a
//: block of 384 threads stays inside the 65,536 registers of one
//: multiprocessor: 128 by 32 and 256 by 224 is 61,440.
constexpr int kProducerRegisters = 32;
constexpr int kConsumerRegisters = 224;

// One stage of a tile by the Tensor Memory Accelerator, and the tile leaves for
// global memory so the caller can compare it with the staging the threads do.
// One thread issues the copy; the barrier states the bytes it will land.
__global__ void TmaStageKernel(const __grid_constant__ CUtensorMap map,
                               int group, int bytes, float* destination) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) std::uint64_t ready;
  if (threadIdx.x == 0) {
    BarrierInit(&ready, 1);
    const std::uint32_t barrier = SharedWord(&ready);
    const std::uint32_t target = SharedWord(smem);
    const std::uint64_t map_address = reinterpret_cast<std::uint64_t>(&map);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            barrier),
        "r"(bytes)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(
            target),
        "l"(map_address), "r"(0), "r"(0), "r"(0), "r"(group), "r"(0),
        "r"(barrier)
        : "memory");
  }
  __syncthreads();
  BarrierWait(&ready, 0);
  const int words = bytes / 4;
  for (int word = static_cast<int>(threadIdx.x); word < words;
       word += static_cast<int>(blockDim.x)) {
    destination[word] =
        __int_as_float(reinterpret_cast<const int*>(smem)[word]);
  }
}

// A ring of tiles filled by the Tensor Memory Accelerator, with no product and
// no thread staging: the bytes a second the copy engine delivers on its own.
// One thread issues every copy and waits for the stage it is about to reuse.
template <int Stages>
__global__ void TmaRingKernel(const __grid_constant__ CUtensorMap map,
                              int tiles, int group_step, int bytes) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) std::uint64_t ready[Stages];
  if (threadIdx.x != 0) return;
  for (int stage = 0; stage < Stages; ++stage) {
    BarrierInit(&ready[stage], 1);
  }
  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % Stages;
    if (tile >= Stages) {
      BarrierWait(&ready[stage], ((tile / Stages) - 1) & 1);
    }
    const std::uint32_t barrier = SharedWord(&ready[stage]);
    const std::uint32_t target = SharedWord(smem + stage * bytes);
    const std::uint64_t map_address = reinterpret_cast<std::uint64_t>(&map);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            barrier),
        "r"(bytes)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(
            target),
        "l"(map_address), "r"(0), "r"(0), "r"(0), "r"(tile * group_step),
        "r"(0), "r"(barrier)
        : "memory");
  }
}

// The same ring, filled with a 128-byte swizzle: two panels a tile, each a
// two-dimensional box the engine likes, and one barrier for the bytes of both.
template <int Stages>
__global__ void TmaSwizzledRingKernel(
    const __grid_constant__ CUtensorMap panel_low,
    const __grid_constant__ CUtensorMap panel_high, int tiles, int rows) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) std::uint64_t ready[Stages];
  constexpr int kBytes = kAtomN * kDim * sizeof(Bf16);
  constexpr int kPanelBytes = kAtomN * 64 * sizeof(Bf16);
  if (threadIdx.x != 0) return;
  for (int stage = 0; stage < Stages; ++stage) {
    BarrierInit(&ready[stage], 1);
  }
  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % Stages;
    if (tile >= Stages) {
      BarrierWait(&ready[stage], ((tile / Stages) - 1) & 1);
    }
    const std::uint32_t barrier = SharedWord(&ready[stage]);
    const std::uint32_t first = SharedWord(smem + stage * kBytes);
    const std::uint32_t second = first + kPanelBytes;
    const std::uint64_t low = reinterpret_cast<std::uint64_t>(&panel_low);
    const std::uint64_t high = reinterpret_cast<std::uint64_t>(&panel_high);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            barrier),
        "r"(kBytes)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(first),
        "l"(low), "r"(0), "r"(tile * rows), "r"(barrier)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(second),
        "l"(high), "r"(0), "r"(tile * rows), "r"(barrier)
        : "memory");
  }
}

// --- The kernels ------------------------------------------------------------

// One warpgroup, one resident query tile and one resident key tile in the
// interleaved form, and `passes` score products over them. `Atoms` is the k
// extent of one pass: `kAtoms` for the attention geometry, `kHalfAtoms` for the
// Phase 0 geometry. The accumulator is 64 by 128 whatever the pass depth is, so
// only the loop bound and the flop count follow `Atoms`. The registers leave
// raw, and every reader below decodes them.
template <int Atoms = kAtoms, int KeyWidth = kAtomN>
__global__ void ResidentKernel(const Bf16* __restrict__ q,
                               const Bf16* __restrict__ k,
                               float* __restrict__ out, int passes) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* key = smem + kQueryBytes;
  StageInterleaved(query, q, kAtomM, kWarpgroupThreads);
  StageInterleaved(key, k, KeyWidth, kWarpgroupThreads);
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_query = MakeDescriptor(query);
  const std::uint64_t base_key = MakeDescriptor(key);
  constexpr int kTileRegisters = KeyRegisters(KeyWidth);
  float d[kTileRegisters];
#pragma unroll
  for (int i = 0; i < kTileRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand<kTileRegisters>(d);

  for (int pass = 0; pass < passes; ++pass) {
#pragma unroll
    for (int atom = 0; atom < Atoms; ++atom) {
      ScoreAtom<KeyWidth>(base_query + atom * AtomStep(kAtomM),
                          base_key + atom * AtomStep(KeyWidth), d,
                          pass != 0 || atom != 0);
    }
    WgmmaCommit();
    WgmmaWait<1>();
  }
  WgmmaWait<0>();
  WgmmaFenceOperand<kTileRegisters>(d);

  float* destination =
      out + static_cast<std::size_t>(threadIdx.x) * kTileRegisters;
#pragma unroll
  for (int i = 0; i < kTileRegisters; ++i) destination[i] = d[i];
}

// One step of the flash tile, and the one place the step is written: the
// resident row and the staged row both call it, so neither can state another
// chain. The probability operand of atom `atom` and register `reg` takes the
// score slots `8 * atom + 4 * (reg / 2) + 2 * (reg % 2) + half`, which is the
// measured register map written the other way round.
//
// `kMask` states the causal boundary tile: the query block and the key tile
// start at the same offset, so a row sees the keys up to its own index and no
// later one. `row_base` is the first query row of the calling warpgroup.
//
// `kSoftmax` false is the ceiling row of the chain: the two products, the
// conversion, and nothing between them. The difference from a row that states
// the real step is what the softmax and its serialized place in the chain cost.
//
// `kTrim` and `kGuard` are the two ways to make the same step cheaper without
// moving it, and each states one row beside the untrimmed one so the two are
// one run of one build. `kTrim` states the exponential as 1 instruction and
// folds the score scale into it, and `kGuard` skips the rescale of the output
// accumulator for a step whose row maximum did not move.
template <bool kMask, bool kSoftmax = true, bool kSwizzledKey = false,
          bool kTrim = false, bool kGuard = false>
__device__ __forceinline__ void FlashStep(
    std::uint64_t base_query, std::uint64_t base_key, std::uint64_t base_value,
    float log2_scale, int row_base, int lane, float (&running_max)[2],
    float (&running_sum)[2], float (&output)[kRegisters]) {
  // The score product over the whole head width. The first atom overwrites, so
  // the accumulator carries no score of the step before it.
  float score[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) score[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(score);
  WgmmaFence();
#pragma unroll
  for (int atom = 0; atom < kAtoms; ++atom) {
    // A swizzled key tile is two panels of sixty-four columns, so four atoms
    // sit in one panel and the fifth jumps a panel.
    const std::uint64_t key_atom =
        kSwizzledKey ? base_key + (atom / 4) * kPanelStepUnits +
                           (atom % 4) * kSwizzledAtomUnits
                     : base_key + atom * AtomStep(kAtomN);
    WgmmaM64N128K16(base_query + atom * AtomStep(kAtomM), key_atom, score,
                    atom != 0);
  }
  WgmmaCommit();
  WgmmaWait<0>();
  WgmmaFenceOperand<kRegisters>(score);

  // The scaled score in base-two units, then the row maximum: the thread's own
  // thirty-two columns of each of its two rows, then the quad.
  float tile_max[2] = {kFlashNegInf, kFlashNegInf};
  float new_max[2];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) {
    // The maximum is monotone under a positive scale, so a trimmed step reads
    // it from the raw score and scales the 1 reduced value instead of every
    // slot.
    if constexpr (!kTrim) score[i] *= log2_scale;
    if constexpr (kMask) {
      // One select a score, so the masked row prices the mask itself. A masked
      // score is the sentinel the running maximum already uses, and its
      // exponential then underflows to zero.
      if (SlotColumn(lane, i) > row_base + SlotRow(lane, i)) {
        score[i] = kFlashNegInf;
      }
    }
    if constexpr (kSoftmax) {
      const int slot = (i / 2) % 2;
      tile_max[slot] = fmaxf(tile_max[slot], score[i]);
    }
  }
  if constexpr (kSoftmax) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      tile_max[slot] = fmaxf(tile_max[slot],
                             __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 1));
      tile_max[slot] = fmaxf(tile_max[slot],
                             __shfl_xor_sync(0xFFFFFFFFu, tile_max[slot], 2));
      // The running maximum is in base-two units, so the raw tile maximum
      // joins it after 1 scale and not before.
      if constexpr (kTrim) tile_max[slot] *= log2_scale;
      new_max[slot] = fmaxf(running_max[slot], tile_max[slot]);
    }

    // The exponentials and the row sum, in the same two reductions. The rescale
    // of the running state is the factor the output accumulator needs as well.
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) {
      const int slot = (i / 2) % 2;
      if constexpr (kTrim) {
        // 1 fused multiply-add and 1 instruction of the exponential pipe.
        score[i] = nanochat::cuda_kernels::FastExp2(
            fmaf(score[i], log2_scale, -new_max[slot]));
      } else {
        score[i] = exp2f(score[i] - new_max[slot]);
      }
      tile_sum[slot] += score[i];
    }
    float rescale[2];
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 1);
      tile_sum[slot] += __shfl_xor_sync(0xFFFFFFFFu, tile_sum[slot], 2);
      // A row maximum that did not move gives a factor of exactly 1, under
      // either exponential, because both forms are exact at zero.
      rescale[slot] = kTrim ? nanochat::cuda_kernels::FastExp2(
                                  running_max[slot] - new_max[slot])
                            : exp2f(running_max[slot] - new_max[slot]);
      running_sum[slot] = running_sum[slot] * rescale[slot] + tile_sum[slot];
      running_max[slot] = new_max[slot];
    }

    // The output accumulator holds the earlier steps under the earlier maximum,
    // so every slot is rescaled before this step's value product adds to it.
    // A warp whose rows all kept their maximum skips the pass; `__any_sync`
    // needs every lane, and the block above is convergent here.
    if constexpr (kGuard) {
      const bool unmoved = rescale[0] == 1.0f && rescale[1] == 1.0f;
      if (__any_sync(0xFFFFFFFFu, !unmoved)) {
#pragma unroll
        for (int i = 0; i < kRegisters; ++i) {
          output[i] *= rescale[(i / 2) % 2];
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kRegisters; ++i) {
        output[i] *= rescale[(i / 2) % 2];
      }
    }
  }

  // The probability operand, two keys a register from the score slots of this
  // same lane, low half of the register first. One `cvt.rn.bf16x2.f32` a pair,
  // which is half the conversion instructions two scalar forms and an or cost.
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

// The flash tile with no copy path: `Warpgroups` warpgroups of one resident
// query tile each, one resident key tile, one resident value tile, and one
// `FlashStep` a tile, so `tiles` steps are one online softmax and not `tiles`
// independent products. The output leaves unnormalised: one division by the row
// sum belongs to the end of a real block, and the check applies it.
template <int Warpgroups, bool kMask, bool kSoftmax, bool kTrim = false,
          bool kGuard = false>
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
  const int row_base = warpgroup * kAtomM;

  for (int tile = 0; tile < tiles; ++tile) {
    FlashStep<kMask, kSoftmax, /*kSwizzledKey=*/false, kTrim, kGuard>(
        base_query, base_key, base_value, log2_scale, row_base, lane,
        running_max, running_sum, output);
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

// The flash tile with a ring: `Warpgroups` warpgroups of one resident query
// tile each, a ring of `Stages` key tiles that the copy engine fills, and a
// ring of `Stages` value tiles that the transposed plain staging fills. Every
// warpgroup of the block reads the same rings, so the arithmetic that one
// staged key byte buys is `64 * Warpgroups` FLOP. `kProducts` false is a copy
// row, and `kStageValue` false stages the key ring alone. A step is
// `FlashStep`, so a staged row and a resident row measure the same chain.
template <int Warpgroups, int Stages, bool kProducts, bool kStageValue,
          bool kMask, bool kSoftmax, bool kVectorValue = false,
          bool kPlainKey = false, bool kSkipKey = false,
          bool kSharedValue = false, bool kTranspose = false>
__global__ void StagedFlashKernel(const Bf16* __restrict__ q,
                                  const Bf16* __restrict__ k,
                                  const Bf16* __restrict__ v, int source_tiles,
                                  float* __restrict__ out,
                                  float* __restrict__ row_sums, int tiles,
                                  float softmax_scale) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* keys = query + Warpgroups * kQueryBytes;
  [[maybe_unused]] char* values = keys + Stages * kKeyBytes;
  // The shared-transpose path keeps the tile a copy engine fills in its own
  // ring, and transposes one stage of it into the ring the step reads.
  [[maybe_unused]] char* raw_values =
      values + (kSharedValue ? Stages * kValueBytes : 0);

  const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;

  // Every thread of the block stages the whole query region, because every
  // warpgroup reads it once the barrier below opens.
  for (int tile = 0; tile < Warpgroups; ++tile) {
    StageInterleaved(query + tile * kQueryBytes,
                     q + static_cast<std::size_t>(tile) * kAtomM * kDim, kAtomM,
                     kThreads);
  }

  // One stage of both rings: the key tile through the copy engine and the value
  // tile through the transposed plain staging. The copy is committed as one
  // group by the caller, so the wait counts key tiles and not value tiles.
  auto fill = [&](int stage, int source) {
    const std::size_t offset =
        static_cast<std::size_t>(source % source_tiles) * kAtomN * kDim;
    if constexpr (!kSkipKey) {
      if constexpr (kPlainKey) {
        StageCopyPlain(keys + stage * kKeyBytes, k + offset, /*row_fast=*/true,
                       kThreads, kAtomN);
      } else {
        StageCopy(keys + stage * kKeyBytes, k + offset, /*row_fast=*/true,
                  kThreads, kAtomN);
      }
    }
    if constexpr (kStageValue) {
      if constexpr (kSharedValue) {
        // The tile a copy engine fills is the key ring's own tile: rows of the
        // key index and a contiguous head width.
        StageCopy(raw_values + stage * kValueBytes, v + offset,
                  /*row_fast=*/true, kThreads, kAtomN);
      } else if constexpr (kVectorValue) {
        StageTransposedVector(values + stage * kValueBytes, v + offset,
                              kThreads);
      } else {
        StageTransposed(values + stage * kValueBytes, v + offset, kThreads);
      }
    }
  };

  // The prologue fills every stage but one, so the first step waits on a copy
  // that is already in flight. It fills the stage even when the run is shorter
  // than the ring, as the staged score rows do, because the wait below counts
  // committed groups and the extra copies read the source ring.
  for (int stage = 0; stage < Stages - 1; ++stage) {
    fill(stage, stage);
    CopyCommit();
  }
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  auto refill = [&](int tile) {
    const int next = tile + Stages - 1;
    if (next < tiles) fill(next % Stages, next);
    CopyCommit();
  };

  if constexpr (kProducts) {
    // The online state of the two rows this thread holds, carried across the
    // steps of the ring.
    float running_max[2] = {kFlashNegInf, kFlashNegInf};
    float running_sum[2] = {0.0f, 0.0f};
    float output[kRegisters];
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) output[i] = 0.0f;
    WgmmaFenceOperand<kRegisters>(output);

    const std::uint64_t base_query =
        MakeDescriptor(query + warpgroup * kQueryBytes);
    const float log2_scale = softmax_scale * kLog2e;

    for (int tile = 0; tile < tiles; ++tile) {
      // Stages - 1 groups are in flight here, so waiting for all but Stages - 2
      // of them completes this step's copy and no later one. The barrier also
      // orders the plain value stores of the stage, and the fence carries both
      // to the async proxy the instruction reads through.
      CopyWait<Stages - 2>();
      __syncthreads();
      if constexpr (kSharedValue && kTranspose) {
        // The transposition leaves the copy path here: the copy engine fills
        // the raw stage and this pass turns it into the tile the value product
        // reads.
        StageTransposeShared(values + (tile % Stages) * kValueBytes,
                             raw_values + (tile % Stages) * kValueBytes,
                             kThreads);
        __syncthreads();
      }
      FenceAsyncShared();

      FlashStep<kMask, kSoftmax>(
          base_query, MakeDescriptor(keys + (tile % Stages) * kKeyBytes),
          MakeDescriptor(values + (tile % Stages) * kValueBytes), log2_scale,
          warpgroup * kAtomM, lane, running_max, running_sum, output);

      // The stage this step released is the one the copy refills, and the step
      // has drained its value group, so nothing reads it now.
      refill(tile);
    }

    WgmmaFenceOperand<kRegisters>(output);
    float* destination =
        out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) destination[i] = output[i];

    if (row_sums != nullptr && (lane % 4) == 0) {
      const int row = 16 * (lane / 32) + (lane % 32) / 4;
      row_sums[warpgroup * kAtomM + row] = running_sum[0];
      row_sums[warpgroup * kAtomM + row + 8] = running_sum[1];
    }
  } else {
    // The copy row: the same bytes, the same rings, and no instruction.
    for (int tile = 0; tile < tiles; ++tile) {
      CopyWait<Stages - 2>();
      __syncthreads();
      if constexpr (kSharedValue && kTranspose) {
        StageTransposeShared(values + (tile % Stages) * kValueBytes,
                             raw_values + (tile % Stages) * kValueBytes,
                             kThreads);
        __syncthreads();
      }
      FenceAsyncShared();
      refill(tile);
    }
  }
}

// --- The engine-fed tile
// ------------------------------------------------------
//
// Both rings moved by the copy engine and no thread staging them: the key ring
// in the swizzled panel form the engine fills, and the value ring in the same
// form and then transposed in shared memory into the tile the value product
// reads. A producer warpgroup issues the copies and runs the transposition, so
// the consumers issue no staging instruction and never wait on a global load.
template <int Consumers, int Stages, bool kMask, bool kSoftmax = true,
          bool kWarpTranspose = false, bool kConsumerTranspose = false,
          bool kTrim = false, bool kGuard = false, bool kRotateStore = false>
__global__ __launch_bounds__(
    (1 + Consumers) * kWarpgroupThreads,
    1) void EngineFlashKernel(const Bf16* __restrict__ q,
                              const __grid_constant__ CUtensorMap key_low,
                              const __grid_constant__ CUtensorMap key_high,
                              const __grid_constant__ CUtensorMap value_low,
                              const __grid_constant__ CUtensorMap value_high,
                              float* __restrict__ out, int tiles,
                              float softmax_scale) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* keys = query + Consumers * kQueryBytes;
  char* values = keys + Stages * kKeyBytes;
  char* raw = values + Stages * kValueBytes;

  __shared__ __align__(8) std::uint64_t filled[Stages];
  __shared__ __align__(8) std::uint64_t ready[Stages];
  __shared__ __align__(8) std::uint64_t released[Stages];

  constexpr int kPanelBytes = kAtomN * 64 * sizeof(Bf16);
  constexpr int kStageBytes = 2 * kKeyBytes;
  const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < Stages; ++stage) {
      BarrierInit(&filled[stage], 1);
      BarrierInit(&ready[stage], 1);
      BarrierInit(&released[stage], Consumers * kWarpgroupThreads);
    }
  }
  if (static_cast<int>(threadIdx.x) < kWarpgroupThreads) {
    for (int tile = 0; tile < Consumers; ++tile) {
      StageInterleaved(query + tile * kQueryBytes,
                       q + static_cast<std::size_t>(tile) * kAtomM * kDim,
                       kAtomM, kWarpgroupThreads);
    }
  }
  __syncthreads();

  if (warpgroup == 0) {
    WarpgroupRegistersGive<kProducerRegisters>();
    for (int tile = 0; tile < tiles; ++tile) {
      const int stage = tile % Stages;
      const unsigned phase = static_cast<unsigned>((tile / Stages) & 1);
      if (threadIdx.x == 0) {
        if (tile >= Stages) {
          BarrierWait(&released[stage], static_cast<int>((phase ^ 1u)));
        }
        const std::uint32_t barrier = SharedWord(&filled[stage]);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                barrier),
            "r"(kStageBytes)
            : "memory");
        const std::uint64_t key_low_map =
            reinterpret_cast<std::uint64_t>(&key_low);
        const std::uint64_t key_high_map =
            reinterpret_cast<std::uint64_t>(&key_high);
        const std::uint64_t value_low_map =
            reinterpret_cast<std::uint64_t>(&value_low);
        const std::uint64_t value_high_map =
            reinterpret_cast<std::uint64_t>(&value_high);
        const std::uint32_t key_first = SharedWord(keys + stage * kKeyBytes);
        const std::uint32_t key_second = key_first + kPanelBytes;
        const std::uint32_t value_first = SharedWord(raw + stage * kValueBytes);
        const std::uint32_t value_second = value_first + kPanelBytes;
        const int row = tile * kAtomN;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(key_first),
            "l"(key_low_map), "r"(0), "r"(row), "r"(barrier)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                key_second),
            "l"(key_high_map), "r"(0), "r"(row), "r"(barrier)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                value_first),
            "l"(value_low_map), "r"(0), "r"(row), "r"(barrier)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                value_second),
            "l"(value_high_map), "r"(0), "r"(row), "r"(barrier)
            : "memory");
        BarrierWait(&filled[stage], static_cast<int>(phase));
      }
      // The whole producer warpgroup waits the copies, and runs the
      // transposition unless the consumers were given it.
      asm volatile("bar.sync 1, %0;" ::"n"(kWarpgroupThreads) : "memory");
      if constexpr (!kConsumerTranspose) {
        if constexpr (kWarpTranspose) {
          StageTransposeSwizzledWarp(values + stage * kValueBytes,
                                     raw + stage * kValueBytes,
                                     kWarpgroupThreads);
        } else {
          StageTransposeSwizzled<kRotateStore>(values + stage * kValueBytes,
                                               raw + stage * kValueBytes,
                                               kWarpgroupThreads);
        }
        asm volatile("bar.sync 1, %0;" ::"n"(kWarpgroupThreads) : "memory");
      }
      if (threadIdx.x == 0) {
        BarrierArrive(&ready[stage]);
      }
    }
    return;
  }

  const int consumer = warpgroup - 1;
  WarpgroupRegistersTake<kConsumerRegisters>();
  float running_max[2] = {kFlashNegInf, kFlashNegInf};
  float running_sum[2] = {0.0f, 0.0f};
  float output[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) output[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(output);

  const std::uint64_t base_query =
      MakeDescriptor(query + consumer * kQueryBytes);
  const float log2_scale = softmax_scale * kLog2e;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % Stages;
    BarrierWait(&ready[stage], (tile / Stages) & 1);
    if constexpr (kConsumerTranspose) {
      // The pass moves here: every consumer warpgroup transposes its share of
      // the stage, and the named barrier holds the step until both have.
      StageTransposeSwizzled(values + stage * kValueBytes,
                             raw + stage * kValueBytes,
                             Consumers * kWarpgroupThreads,
                             static_cast<int>(threadIdx.x) - kWarpgroupThreads);
      asm volatile("bar.sync 2, %0;" ::"n"(Consumers * kWarpgroupThreads)
                   : "memory");
    }
    FenceAsyncShared();
    FlashStep<kMask, kSoftmax, /*kSwizzledKey=*/true, kTrim, kGuard>(
        base_query,
        MakeDescriptorFull(keys + stage * kKeyBytes, kSwizzledLeadingBytes,
                           kSwizzledStrideBytes, /*layout=*/1),
        MakeDescriptor(values + stage * kValueBytes), log2_scale,
        consumer * kAtomM, lane, running_max, running_sum, output);
    BarrierArrive(&released[stage]);
  }

  WgmmaFenceOperand<kRegisters>(output);
  const std::size_t slot =
      static_cast<std::size_t>(consumer) * kWarpgroupThreads + lane;
  float* destination = out + slot * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = output[i];
}

// --- The warp-specialized tile ----------------------------------------------
//
// Decision H3 of docs/attention.md Part 8 section 2: one producer warpgroup
// issues the copies and never computes, and the consumer warpgroups issue the
// products and never wait on a global load. The two roles hand stages over
// through `ready` and `released` barriers, so neither waits for the other
// beyond the tile that is actually in flight, and `setmaxnreg` moves registers
// from the producer to the consumers.
//
// `ProducerThreads` is the width of the producer's staging: 128 threads is the
// whole warpgroup, and 32 is the single warp of the first phase of the
// decision. The producer role is the whole first warpgroup either way, because
// `setmaxnreg` and the warpgroup instruction of the consumers both span 128
// consecutive threads: a block whose roles did not start on a warpgroup
// boundary cannot issue either. The consumer counts, the ring depth, and the
// two products follow the staged flash tile, so a step here and a step there
// are one `FlashStep`.
template <int ProducerThreads, int Consumers, int Stages, bool kProducts,
          bool kStageValue, bool kMask, bool kSoftmax = true>
__global__ void __launch_bounds__((1 + Consumers) * kWarpgroupThreads, 1)
    SpecializedFlashKernel(const Bf16* __restrict__ q,
                           const Bf16* __restrict__ k,
                           const Bf16* __restrict__ v, int source_tiles,
                           float* __restrict__ out,
                           float* __restrict__ row_sums, int tiles,
                           float softmax_scale) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* keys = query + Consumers * kQueryBytes;
  [[maybe_unused]] char* values = keys + Stages * kKeyBytes;

  __shared__ __align__(8) std::uint64_t ready[Stages];
  __shared__ __align__(8) std::uint64_t released[Stages];

  // A copy row stages what the step reads, so the two rows state one ring.
  constexpr bool kStageValues = kStageValue;
  constexpr int kProducerArrivals = ProducerThreads * (kStageValues ? 2 : 1);
  constexpr int kConsumerArrivals = Consumers * kWarpgroupThreads;
  //: The producer role is the first warpgroup, so the consumers start on the
  //: warpgroup boundary the instruction and the register move both need.
  constexpr int kProducerRole = 0;
  const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < Stages; ++stage) {
      BarrierInit(&ready[stage], kProducerArrivals);
      BarrierInit(&released[stage], kConsumerArrivals);
    }
  }

  // The producer stages every query tile of the block before the first step,
  // because one consumer warpgroup reads each of them and the barrier below is
  // the only one before the loop.
  if (static_cast<int>(threadIdx.x) < ProducerThreads) {
    for (int tile = 0; tile < Consumers; ++tile) {
      StageInterleaved(query + tile * kQueryBytes,
                       q + static_cast<std::size_t>(tile) * kAtomM * kDim,
                       kAtomM, ProducerThreads);
    }
  }
  __syncthreads();

  if (warpgroup == kProducerRole) {
    WarpgroupRegistersGive<kProducerRegisters>();
    if (static_cast<int>(threadIdx.x) < ProducerThreads) {
      // The producer runs ahead of the consumers by `Stages - 1` stages and
      // never waits on a copy it issued: the barrier of a stage completes when
      // the thread's own copies land, so the issue of the next stage does not
      // wait for the last one to arrive.
      for (int tile = 0; tile < tiles + Stages - 1; ++tile) {
        const int stage = tile % Stages;
        if (tile >= Stages) {
          BarrierWait(&released[stage], ((tile / Stages) - 1) & 1);
        }
        const std::size_t offset =
            static_cast<std::size_t>(tile % source_tiles) * kAtomN * kDim;
        StageCopy(keys + stage * kKeyBytes, k + offset, /*row_fast=*/true,
                  ProducerThreads, kAtomN);
        BarrierArriveOnCopies(&ready[stage]);
        if constexpr (kStageValues) {
          StageTransposed(values + stage * kValueBytes, v + offset,
                          ProducerThreads);
          BarrierArrive(&ready[stage]);
        }
      }
    }
    return;
  }

  // The consumers: one query tile a warpgroup, the same `FlashStep` as the
  // staged flash tile, and no staging instruction at all.
  const int consumer = warpgroup - 1;
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;
  WarpgroupRegistersTake<kConsumerRegisters>();

  if constexpr (kProducts) {
    float running_max[2] = {kFlashNegInf, kFlashNegInf};
    float running_sum[2] = {0.0f, 0.0f};
    float output[kRegisters];
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) output[i] = 0.0f;
    WgmmaFenceOperand<kRegisters>(output);

    const std::uint64_t base_query =
        MakeDescriptor(query + consumer * kQueryBytes);
    const float log2_scale = softmax_scale * kLog2e;

    for (int tile = 0; tile < tiles; ++tile) {
      const int stage = tile % Stages;
      BarrierWait(&ready[stage], (tile / Stages) & 1);
      // The wait carries the producer's stores to this thread; this fence
      // carries them to the async proxy the instruction reads through.
      FenceAsyncShared();

      FlashStep<kMask, kSoftmax>(
          base_query, MakeDescriptor(keys + stage * kKeyBytes),
          MakeDescriptor(values + stage * kValueBytes), log2_scale,
          consumer * kAtomM, lane, running_max, running_sum, output);

      // The step has drained its value product, so nothing reads this stage.
      BarrierArrive(&released[stage]);
    }

    WgmmaFenceOperand<kRegisters>(output);
    const std::size_t slot =
        static_cast<std::size_t>(consumer) * kWarpgroupThreads + lane;
    float* destination = out + slot * kRegisters;
#pragma unroll
    for (int i = 0; i < kRegisters; ++i) destination[i] = output[i];

    if (row_sums != nullptr && (lane % 4) == 0) {
      const int row = 16 * (lane / 32) + (lane % 32) / 4;
      row_sums[consumer * kAtomM + row] = running_sum[0];
      row_sums[consumer * kAtomM + row + 8] = running_sum[1];
    }
  } else {
    // The copy row: the same block, the same ring, and no instruction, so the
    // bytes a second of the producer standing alone are one number.
    for (int tile = 0; tile < tiles; ++tile) {
      const int stage = tile % Stages;
      BarrierWait(&ready[stage], (tile / Stages) & 1);
      FenceAsyncShared();
      BarrierArrive(&released[stage]);
    }
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
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages,
          int KeyWidth = kAtomN>
__global__ void StagedKernel(const Bf16* __restrict__ q,
                             const Bf16* __restrict__ k, int source_tiles,
                             float* __restrict__ out, int tiles, int options) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  constexpr int kTiles = Warpgroups * QueryTiles;
  // The accumulator of one query tile, and the bytes of one staged key tile.
  // Both follow the key width, so a narrow ring costs half the shared memory
  // and half the registers a tile.
  constexpr int kTileRegisters = KeyRegisters(KeyWidth);
  constexpr int kStageBytes = KeyBytes(KeyWidth);
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

  float d[QueryTiles * kTileRegisters];
  if constexpr (kProducts) {
#pragma unroll
    for (int i = 0; i < QueryTiles * kTileRegisters; ++i) d[i] = 0.0f;
    WgmmaFenceOperand<QueryTiles * kTileRegisters>(d);
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
        k + static_cast<std::size_t>(tile % source_tiles) * KeyWidth * kDim;
    if (plain_stage) {
      StageInterleaved(stages + tile * kStageBytes, source, KeyWidth, kThreads);
    } else {
      StageCopy(stages + tile * kStageBytes, source, row_fast, kThreads,
                KeyWidth);
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
          MakeDescriptor(stages + (tile % Stages) * kStageBytes);
#pragma unroll
      for (int slot = 0; slot < QueryTiles; ++slot) {
#pragma unroll
        for (int atom = 0; atom < kAtoms; ++atom) {
          ScoreAtom<KeyWidth>(base_query[slot] + atom * AtomStep(kAtomM),
                              base_key + atom * AtomStep(KeyWidth),
                              d + slot * kTileRegisters, true);
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
          k + static_cast<std::size_t>(next % source_tiles) * KeyWidth * kDim;
      if (plain_stage) {
        StageInterleaved(stages + (next % Stages) * kStageBytes, source,
                         KeyWidth, kThreads);
      } else {
        StageCopy(stages + (next % Stages) * kStageBytes, source, row_fast,
                  kThreads, KeyWidth);
      }
    }
    CopyCommit();
  }

  if constexpr (kProducts) {
    WgmmaWait<0>();
    WgmmaFenceOperand<QueryTiles * kTileRegisters>(d);
    const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;
#pragma unroll
    for (int slot = 0; slot < QueryTiles; ++slot) {
      // One tile of the output per query tile of the block, in tile order, so
      // the caller reads tile `warpgroup * QueryTiles + slot`.
      float* destination =
          out + (static_cast<std::size_t>(warpgroup * QueryTiles + slot) *
                     kWarpgroupThreads +
                 static_cast<std::size_t>(lane)) *
                    kTileRegisters;
#pragma unroll
      for (int i = 0; i < kTileRegisters; ++i) {
        destination[i] = d[slot * kTileRegisters + i];
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
__global__ void OperandSweepKernel(
    const Bf16* __restrict__ p, const Bf16* __restrict__ v, int atom_step,
    int trans_b, int transpose_stage, int lbo, int sbo, int layout,
    int panel_atoms, int panel_step,
    const __grid_constant__ CUtensorMap panel_low,
    const __grid_constant__ CUtensorMap panel_high, float* __restrict__ out) {
  extern __shared__ __align__(1024) char smem[];
  char* probability = smem;
  char* value = smem + kQueryBytes;
  StageInterleaved(probability, p, kAtomM, kWarpgroupThreads);
  if (transpose_stage == 3) {
    // The copy engine fills the two panels of the swizzled tile, and one
    // barrier states the bytes of both.
    __shared__ __align__(8) std::uint64_t engine_ready;
    if (threadIdx.x == 0) {
      BarrierInit(&engine_ready, 1);
      const std::uint32_t barrier = SharedWord(&engine_ready);
      constexpr int kBytes = kAtomN * kDim * sizeof(Bf16);
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
              barrier),
          "r"(kBytes)
          : "memory");
      const std::uint64_t low = reinterpret_cast<std::uint64_t>(&panel_low);
      const std::uint64_t high = reinterpret_cast<std::uint64_t>(&panel_high);
      const std::uint32_t first = SharedWord(value);
      const std::uint32_t second =
          SharedWord(value + kAtomN * 64 * sizeof(Bf16));
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
          "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(first),
          "l"(low), "r"(0), "r"(0), "r"(barrier)
          : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
          "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(second),
          "l"(high), "r"(0), "r"(0), "r"(barrier)
          : "memory");
    }
    __syncthreads();
    BarrierWait(&engine_ready, 0);
  } else if (transpose_stage == 2) {
    StageSwizzled(value, v, kAtomN, kWarpgroupThreads, layout);
  } else if (transpose_stage == 1) {
    StageTransposed(value, v, kWarpgroupThreads);
  } else {
    StageInterleaved(value, v, kAtomN, kWarpgroupThreads);
  }
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_probability = MakeDescriptor(probability);
  const std::uint64_t base_value = MakeDescriptorFull(value, lbo, sbo, layout);
  float d[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand<kRegisters>(d);

  for (int atom = 0; atom < kKeyAtoms; ++atom) {
    const std::uint64_t desc_p = base_probability + atom * AtomStep(kAtomM);
    const int panel = panel_atoms > 0 ? atom / panel_atoms : 0;
    const int within = panel_atoms > 0 ? atom % panel_atoms : atom;
    const std::uint64_t desc_v =
        base_value + within * atom_step + panel * panel_step;
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

// The resident shared window of one key width: the query tile and the key tile.
template <int KeyWidth = kAtomN>
constexpr int ResidentBytes() {
  return kQueryBytes + KeyBytes(KeyWidth);
}

// The result of one resident pass, for the checks below.
template <int KeyWidth = kAtomN>
std::vector<float> RunResident(const Bf16* device_q, const Bf16* device_k,
                               float* device_out) {
  ResidentKernel<kAtoms, KeyWidth>
      <<<1, kWarpgroupThreads, ResidentBytes<KeyWidth>()>>>(device_q, device_k,
                                                            device_out, 1);
  cudaDeviceSynchronize();
  return ReadBack(device_out, static_cast<std::size_t>(KeySlots(KeyWidth)));
}

// The one opt-in dynamic shared-memory request, stated for each instantiation
// the file launches. A templated kernel has no single function address, so the
// attribute is set once per pass depth rather than once per file.
template <int Atoms, int KeyWidth = kAtomN>
void SetResidentSharedMemory() {
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(ResidentKernel<Atoms, KeyWidth>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, ResidentBytes<KeyWidth>());
}

// One launch of the resident kernel: `blocks` warpgroups, `Atoms` atoms per
// pass, and `steps` passes of the loop inside the kernel. The protocol around
// it owns the warm-up, the rounds, and the statistic, so this function states
// the call and nothing else.
template <int Atoms, int KeyWidth = kAtomN>
void LaunchResident(int blocks, const Bf16* device_q, const Bf16* device_k,
                    float* device_out, int steps) {
  static_assert(ResidentBytes<KeyWidth>() <= kSmemCeiling,
                "the resident tile must fit one block per multiprocessor");
  ResidentKernel<Atoms, KeyWidth>
      <<<blocks, kWarpgroupThreads, ResidentBytes<KeyWidth>()>>>(
          device_q, device_k, device_out, steps);
}

// Fails the run when the device holds an error. A launch the device refuses
// leaves the error pending and an event pair that times a few microseconds, so
// a row whose kernel never ran would otherwise state a rate. The first card run
// of the narrow copy row did exactly that, at 0.0006 ms for 268 MB of staged
// bytes, because its shared-memory request had no opt-in attribute.
void CheckDevice(const std::string& what) {
  const cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "fatal: %s: %s\n", what.c_str(),
               cudaGetErrorString(status));
  std::fflush(stderr);
  std::exit(1);
}

// A checked sync that names the phase it waited for. An unchecked sync reports
// the failure at the next checked call of the harness, which names nothing.
void SyncFlash(const std::string& what) {
  // A launch that the device refused, such as one above the default dynamic
  // shared-memory window, reports nothing from the synchronizing call. It
  // states itself only as the last error, so both are read here: a silent
  // refusal otherwise leaves the buffer of a previous launch to be compared
  // or timed as if it were this one.
  const cudaError_t status = cudaDeviceSynchronize();
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t failure = status != cudaSuccess ? status : launch;
  if (failure == cudaSuccess) return;
  std::fprintf(stderr, "fatal: flash %s: %s\n", what.c_str(),
               cudaGetErrorString(failure));
  std::fflush(stderr);
  std::exit(1);
}

// The opt-in dynamic shared-memory request of a resident flash instantiation,
// and its one launch. A templated kernel has no single function address, so the
// attribute is set once per instantiation and outside any timed window.
template <int Warpgroups, bool kMask, bool kSoftmax = true, bool kTrim = false,
          bool kGuard = false>
void SetFlashSharedMemory() {
  constexpr int kShared = FlashResidentBytes(Warpgroups);
  static_assert(
      kShared <= kSmemCeiling,
      "the resident flash tile must fit one block per multiprocessor");
  static_assert(Warpgroups * kWarpgroupThreads <= kMaxThreadsPerBlock,
                "one block of warpgroups must fit the thread limit");
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          ResidentFlashKernel<Warpgroups, kMask, kSoftmax, kTrim, kGuard>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

template <int Warpgroups, bool kMask, bool kSoftmax = true, bool kTrim = false,
          bool kGuard = false>
void LaunchResidentFlash(int blocks, const Bf16* device_q, const Bf16* device_k,
                         const Bf16* device_v, float* device_out,
                         float* device_row_sums, int tiles, float scale) {
  constexpr int kShared = FlashResidentBytes(Warpgroups);
  ResidentFlashKernel<Warpgroups, kMask, kSoftmax, kTrim, kGuard>
      <<<blocks, Warpgroups * kWarpgroupThreads, kShared>>>(
          device_q, device_k, device_v, device_out, device_row_sums, tiles,
          scale);
}

// One step of the flash tile over one block, read back: the accumulator slots
// of every warpgroup in thread order, and the row sums beside them. `tiles` is
// the number of steps of the online softmax, which the ring checks set above
// the depth of the ring so the ring wraps.
template <int Warpgroups, bool kMask, bool kSoftmax = true, bool kTrim = false,
          bool kGuard = false>
std::vector<float> RunResidentFlash(const Bf16* device_q, const Bf16* device_k,
                                    const Bf16* device_v, float* device_out,
                                    float* device_row_sums, int tiles = 1) {
  LaunchResidentFlash<Warpgroups, kMask, kSoftmax, kTrim, kGuard>(
      1, device_q, device_k, device_v, device_out, device_row_sums, tiles,
      kFlashScale);
  SyncFlash("resident warpgroups=" + std::to_string(Warpgroups) +
            (kMask ? " causal" : " dense"));
  return ReadBack(device_out, Warpgroups * kProductSlots);
}

// The opt-in dynamic shared-memory request of a staged flash instantiation, and
// its one launch. A templated kernel has no single function address, so the
// attribute is set once per instantiation and outside any timed window.
template <int Warpgroups, int Stages, bool kProducts, bool kStageValue,
          bool kMask, bool kSoftmax = true, bool kVectorValue = false,
          bool kPlainKey = false, bool kSkipKey = false,
          bool kSharedValue = false, bool kTranspose = false>
void SetStagedFlashSharedMemory() {
  constexpr int kShared =
      kSharedValue ? FlashSharedValueBytes(Warpgroups, Stages)
                   : FlashStagedBytes(Warpgroups, Stages, kStageValue);
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          StagedFlashKernel<Warpgroups, Stages, kProducts, kStageValue, kMask,
                            kSoftmax, kVectorValue, kPlainKey, kSkipKey,
                            kSharedValue, kTranspose>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

template <int Warpgroups, int Stages, bool kProducts, bool kStageValue,
          bool kMask, bool kSoftmax = true, bool kVectorValue = false,
          bool kPlainKey = false, bool kSkipKey = false,
          bool kSharedValue = false, bool kTranspose = false>
void LaunchStagedFlash(int blocks, int tiles, int source_tiles,
                       const Bf16* device_q, const Bf16* device_k,
                       const Bf16* device_v, float* device_out,
                       float* device_row_sums) {
  constexpr int kShared =
      kSharedValue ? FlashSharedValueBytes(Warpgroups, Stages)
                   : FlashStagedBytes(Warpgroups, Stages, kStageValue);
  StagedFlashKernel<Warpgroups, Stages, kProducts, kStageValue, kMask, kSoftmax,
                    kVectorValue, kPlainKey, kSkipKey, kSharedValue, kTranspose>
      <<<blocks, Warpgroups * kWarpgroupThreads, kShared>>>(
          device_q, device_k, device_v, source_tiles, device_out,
          device_row_sums, tiles, kFlashScale);
}

// One step of a staged geometry against the resident step of the same operands:
// the ring, its barriers, the stage offsets, and the value staging all fail
// here instead of measuring.
template <int Warpgroups, int Stages, bool kMask>
std::pair<std::size_t, double> CheckStagedFlash(
    const Bf16* device_queries, const Bf16* device_k, const Bf16* device_v,
    float* device_out, float* device_row_sums,
    const std::vector<float>& reference, double scale) {
  LaunchStagedFlash<Warpgroups, Stages, true, kRingKeyValue, kMask>(
      1, /*tiles=*/1, /*source_tiles=*/1, device_queries, device_k, device_v,
      device_out, device_row_sums);
  SyncFlash("staged warpgroups=" + std::to_string(Warpgroups) + " stages=" +
            std::to_string(Stages) + (kMask ? " causal" : " dense"));
  const std::vector<float> staged =
      ReadBack(device_out, Warpgroups * kProductSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t slot = 0; slot < staged.size(); ++slot) {
    const double difference = std::fabs(static_cast<double>(staged[slot]) -
                                        static_cast<double>(reference[slot]));
    if (difference > 1e-6 * scale) ++bad;
    worst = std::max(worst, difference);
  }
  return {bad, worst / scale};
}

// One step of the staged tile whose value ring is a copy engine fill and a
// shared transposition, against the resident step of the same operands: a wrong
// unit mapping, a wrong element order inside a unit, or a stage offset fails
// here instead of measuring.
template <int Warpgroups, int Stages, bool kMask>
std::pair<std::size_t, double> CheckSharedValue(
    const Bf16* device_queries, const Bf16* device_k, const Bf16* device_v,
    float* device_out, float* device_row_sums,
    const std::vector<float>& reference, double scale) {
  SetStagedFlashSharedMemory<Warpgroups, Stages, true, kRingKeyValue, kMask,
                             /*kSoftmax=*/true, /*kVectorValue=*/false,
                             /*kPlainKey=*/false, /*kSkipKey=*/false,
                             /*kSharedValue=*/true, /*kTranspose=*/true>();
  LaunchStagedFlash<Warpgroups, Stages, true, kRingKeyValue, kMask,
                    /*kSoftmax=*/true, /*kVectorValue=*/false,
                    /*kPlainKey=*/false, /*kSkipKey=*/false,
                    /*kSharedValue=*/true, /*kTranspose=*/true>(
      1, /*tiles=*/1, /*source_tiles=*/1, device_queries, device_k, device_v,
      device_out, device_row_sums);
  SyncFlash("shared-value warpgroups=" + std::to_string(Warpgroups) +
            " stages=" + std::to_string(Stages) +
            (kMask ? " causal" : " dense"));
  const std::vector<float> staged =
      ReadBack(device_out, Warpgroups * kProductSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t slot = 0; slot < staged.size(); ++slot) {
    const double difference = std::fabs(static_cast<double>(staged[slot]) -
                                        static_cast<double>(reference[slot]));
    if (difference > 1e-6 * scale) ++bad;
    worst = std::max(worst, difference);
  }
  return {bad, worst / scale};
}

// The opt-in shared-memory request, the launch, and the registers of the
// engine-fed tile.
template <int Consumers, int Stages, bool kMask, bool kSoftmax = true,
          bool kWarpTranspose = false, bool kConsumerTranspose = false,
          bool kTrim = false, bool kGuard = false, bool kRotateStore = false>
void SetEngineFlashSharedMemory() {
  constexpr int kShared = EngineFlashBytes(Consumers, Stages);
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          EngineFlashKernel<Consumers, Stages, kMask, kSoftmax, kWarpTranspose,
                            kConsumerTranspose, kTrim, kGuard, kRotateStore>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

template <int Consumers, int Stages, bool kMask, bool kSoftmax = true,
          bool kWarpTranspose = false, bool kConsumerTranspose = false,
          bool kTrim = false, bool kGuard = false, bool kRotateStore = false>
void LaunchEngineFlash(int blocks, int tiles, const Bf16* device_q,
                       const CUtensorMap& key_low, const CUtensorMap& key_high,
                       const CUtensorMap& value_low,
                       const CUtensorMap& value_high, float* device_out) {
  constexpr int kShared = EngineFlashBytes(Consumers, Stages);
  EngineFlashKernel<Consumers, Stages, kMask, kSoftmax, kWarpTranspose,
                    kConsumerTranspose, kTrim, kGuard, kRotateStore>
      <<<blocks, (1 + Consumers) * kWarpgroupThreads, kShared>>>(
          device_q, key_low, key_high, value_low, value_high, device_out, tiles,
          kFlashScale);
}

template <int Consumers, int Stages, bool kMask, bool kSoftmax = true,
          bool kWarpTranspose = false, bool kConsumerTranspose = false,
          bool kTrim = false, bool kGuard = false, bool kRotateStore = false>
int EngineFlashRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      EngineFlashKernel<Consumers, Stages, kMask, kSoftmax, kWarpTranspose,
                        kConsumerTranspose, kTrim, kGuard, kRotateStore>));
}

// One step of the engine-fed tile against the resident step of the same
// operands: a wrong swizzled key reading, a wrong panel step, or a wrong
// transposition fails here instead of measuring.
template <int Consumers, int Stages, bool kMask, bool kWarpTranspose = false,
          bool kConsumerTranspose = false, bool kTrim = false,
          bool kGuard = false, bool kRotateStore = false>
std::pair<std::size_t, double> CheckEngineFlash(
    const Bf16* device_queries, const CUtensorMap& key_low,
    const CUtensorMap& key_high, const CUtensorMap& value_low,
    const CUtensorMap& value_high, float* device_out,
    const std::vector<float>& reference, double scale) {
  SetEngineFlashSharedMemory<Consumers, Stages, kMask, /*kSoftmax=*/true,
                             kWarpTranspose, kConsumerTranspose, kTrim, kGuard,
                             kRotateStore>();
  LaunchEngineFlash<Consumers, Stages, kMask, /*kSoftmax=*/true, kWarpTranspose,
                    kConsumerTranspose, kTrim, kGuard, kRotateStore>(
      1, /*tiles=*/1, device_queries, key_low, key_high, value_low, value_high,
      device_out);
  SyncFlash("engine-fed tile consumers=" + std::to_string(Consumers));
  const std::vector<float> staged =
      ReadBack(device_out, Consumers * kProductSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t slot = 0; slot < staged.size(); ++slot) {
    const double difference = std::fabs(static_cast<double>(staged[slot]) -
                                        static_cast<double>(reference[slot]));
    if (difference > 1e-6 * scale) ++bad;
    worst = std::max(worst, difference);
  }
  return {bad, worst / scale};
}

// The registers of the shared-transpose value instantiation, read from the
// build, so the shape table states the compiler's count and not an estimate.
template <int Warpgroups, int Stages, bool kMask>
int SharedValueRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      StagedFlashKernel<Warpgroups, Stages, true, kRingKeyValue, kMask,
                        /*kSoftmax=*/true, /*kVectorValue=*/false,
                        /*kPlainKey=*/false, /*kSkipKey=*/false,
                        /*kSharedValue=*/true, /*kTranspose=*/true>));
}

// One geometry of the flash ladder, dispatched to its own instantiation. The
// warpgroup count and the ring depth are template parameters, so a sweep over
// them is one branch per geometry, and this is the one place that states the
// pairs.
template <typename Visitor>
void VisitFlashGeometry(int warpgroups, int stages, Visitor&& visit) {
  if (warpgroups == 1 && stages == 2) {
    visit(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  } else if (warpgroups == 1 && stages == 3) {
    visit(std::integral_constant<int, 1>{}, std::integral_constant<int, 3>{});
  } else if (warpgroups == 2 && stages == 2) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  } else if (warpgroups == 2 && stages == 3) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 3>{});
  } else {
    std::fprintf(stderr,
                 "the flash rows have no geometry of %d warpgroups and %d "
                 "stages\n",
                 warpgroups, stages);
    std::exit(2);
  }
}

// The opt-in dynamic shared-memory request of a warp-specialized
// instantiation, and its one launch. A templated kernel has no single function
// address, so the attribute is set once per instantiation and outside any timed
// window.
template <int ProducerThreads, int Consumers, int Stages, bool kProducts,
          bool kStageValue, bool kMask, bool kSoftmax = true>
void SetSpecializedSharedMemory() {
  constexpr int kShared = SpecializedBytes(Consumers, Stages, kStageValue);
  static_assert(kShared <= kSmemCeiling,
                "the warp-specialized ring must fit one block per "
                "multiprocessor");
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          SpecializedFlashKernel<ProducerThreads, Consumers, Stages, kProducts,
                                 kStageValue, kMask, kSoftmax>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

template <int ProducerThreads, int Consumers, int Stages, bool kProducts,
          bool kStageValue, bool kMask, bool kSoftmax = true>
void LaunchSpecialized(int blocks, int tiles, int source_tiles,
                       const Bf16* device_q, const Bf16* device_k,
                       const Bf16* device_v, float* device_out,
                       float* device_row_sums) {
  constexpr int kShared = SpecializedBytes(Consumers, Stages, kStageValue);
  constexpr int kThreads = (1 + Consumers) * kWarpgroupThreads;
  SpecializedFlashKernel<ProducerThreads, Consumers, Stages, kProducts,
                         kStageValue, kMask, kSoftmax>
      <<<blocks, kThreads, kShared>>>(device_q, device_k, device_v,
                                      source_tiles, device_out, device_row_sums,
                                      tiles, kFlashScale);
}

// One step of a warp-specialized geometry against the resident step of the same
// operands, as the staged ladder is checked: a wrong barrier, stage offset,
// value staging, or role boundary fails here instead of measuring.
template <int ProducerThreads, int Consumers, int Stages, bool kMask>
std::pair<std::size_t, double> CheckSpecialized(
    const Bf16* device_queries, const Bf16* device_k, const Bf16* device_v,
    float* device_out, float* device_row_sums,
    const std::vector<float>& reference, double scale, int tiles = 1) {
  SetSpecializedSharedMemory<ProducerThreads, Consumers, Stages, true,
                             kRingKeyValue, kMask>();
  LaunchSpecialized<ProducerThreads, Consumers, Stages, true, kRingKeyValue,
                    kMask>(1, tiles, /*source_tiles=*/1, device_queries,
                           device_k, device_v, device_out, device_row_sums);
  SyncFlash("specialized producer=" + std::to_string(ProducerThreads) +
            " consumers=" + std::to_string(Consumers) + " stages=" +
            std::to_string(Stages) + (kMask ? " causal" : " dense"));
  const std::vector<float> staged =
      ReadBack(device_out, Consumers * kProductSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t slot = 0; slot < staged.size(); ++slot) {
    const double difference = std::fabs(static_cast<double>(staged[slot]) -
                                        static_cast<double>(reference[slot]));
    if (difference > 1e-6 * scale) ++bad;
    worst = std::max(worst, difference);
  }
  return {bad, worst / scale};
}

// The registers of one warp-specialized instantiation, read from the build. The
// block holds two roles in one kernel, so this is the count the compiler gave
// the wider path, and the redistribution above moves the runtime allocation
// around it.
template <int ProducerThreads, int Consumers, int Stages, bool kMask,
          bool kSoftmax = true>
int SpecializedRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      SpecializedFlashKernel<ProducerThreads, Consumers, Stages, true,
                             kRingKeyValue, kMask, kSoftmax>));
}

// One producer width of the warp-specialized ladder, dispatched to its own
// instantiation, so this is the one place that states the widths.
template <typename Visitor>
void VisitSpecializedProducer(int producer_threads, Visitor&& visit) {
  if (producer_threads == kSpecialWarpProducer) {
    visit(std::integral_constant<int, kSpecialWarpProducer>{});
  } else if (producer_threads == kSpecialWarpgroupProducer) {
    visit(std::integral_constant<int, kSpecialWarpgroupProducer>{});
  } else {
    std::fprintf(stderr,
                 "the warp-specialized rows have no producer of %d threads\n",
                 producer_threads);
    std::exit(2);
  }
}

// One consumer count of the warp-specialized ladder, dispatched to its own
// instantiation.
template <typename Visitor>
void VisitSpecializedConsumers(int consumers, Visitor&& visit) {
  if (consumers == kSpecialOneConsumer) {
    visit(std::integral_constant<int, kSpecialOneConsumer>{});
  } else if (consumers == kSpecialTwoConsumers) {
    visit(std::integral_constant<int, kSpecialTwoConsumers>{});
  } else {
    std::fprintf(stderr,
                 "the warp-specialized rows have no block of %d consumer "
                 "warpgroups\n",
                 consumers);
    std::exit(2);
  }
}

// One ring depth of the warp-specialized ladder, dispatched to its own
// instantiation.
template <typename Visitor>
void VisitSpecializedRing(int stages, Visitor&& visit) {
  if (stages == kSpecialStages) {
    visit(std::integral_constant<int, kSpecialStages>{});
  } else if (stages == kSpecialDeepStages) {
    visit(std::integral_constant<int, kSpecialDeepStages>{});
  } else {
    std::fprintf(stderr,
                 "the warp-specialized rows have no ring of %d stages\n",
                 stages);
    std::exit(2);
  }
}

// The one opt-in dynamic shared-memory request of a staged instantiation. A
// templated kernel has no single function address, so the attribute is set once
// per instantiation and outside any timed window.
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages,
          int KeyWidth = kAtomN>
void SetStagedSharedMemory() {
  constexpr int kShared = StagedBytes(Warpgroups, QueryTiles, Stages, KeyWidth);
  static_assert(kShared <= kSmemCeiling,
                "the ring must fit one block per multiprocessor");
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          StagedKernel<Warpgroups, QueryTiles, kProducts, Stages, KeyWidth>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
}

// One launch of the staged kernel: `tiles` steps of the ring, over `blocks`
// blocks, with the copy order of `options`.
template <int Warpgroups, int QueryTiles, bool kProducts, int Stages = kStages,
          int KeyWidth = kAtomN>
void LaunchStaged(int blocks, int tiles, int source_tiles, const Bf16* device_q,
                  const Bf16* device_k, float* device_out, int options = 0) {
  constexpr int kShared = StagedBytes(Warpgroups, QueryTiles, Stages, KeyWidth);
  StagedKernel<Warpgroups, QueryTiles, kProducts, Stages, KeyWidth>
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

// One geometry of the narrow key tile, dispatched to its own instantiation. The
// narrow tile holds twice the accumulator words of a tile the wide one can
// carry, so its geometries are the ones the wide ladder stops short of, and
// this is the one place that states them.
template <typename Visitor>
void VisitNarrowGeometry(int warpgroups, int query_tiles, Visitor&& visit) {
  if (warpgroups == 2 && query_tiles == 4) {
    visit(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  } else {
    std::fprintf(stderr,
                 "the narrow key tile has no geometry of %d warpgroups and %d "
                 "query tiles\n",
                 warpgroups, query_tiles);
    std::exit(2);
  }
}

// One geometry's staged check: every query tile of every warpgroup must agree
// with the resident product of that same tile, because all of them multiply
// their own query tile by the same key tile. The answer is the count of
// elements outside the tolerance and the worst difference, both as the caller
// reports them.
template <int Warpgroups, int QueryTiles, int KeyWidth = kAtomN>
std::pair<std::size_t, double> CheckStagedGeometry(
    const Bf16* device_queries, const Bf16* device_k, float* device_out,
    int source_tiles, int options,
    const std::vector<std::vector<float>>& reference_tiles, double scale) {
  constexpr int kThreads = Warpgroups * kWarpgroupThreads;
  constexpr int kTiles = Warpgroups * QueryTiles;
  constexpr std::size_t kSlots = static_cast<std::size_t>(KeySlots(KeyWidth));
  constexpr int kShared =
      StagedBytes(Warpgroups, QueryTiles, kStages, KeyWidth);
  cudaFuncSetAttribute(
      reinterpret_cast<const void*>(
          StagedKernel<Warpgroups, QueryTiles, true, kStages, KeyWidth>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  cudaMemset(device_out, 0, kTiles * kSlots * sizeof(float));
  StagedKernel<Warpgroups, QueryTiles, true, kStages, KeyWidth>
      <<<1, kThreads, kShared>>>(device_queries, device_k, source_tiles,
                                 device_out, 1, options);
  cudaDeviceSynchronize();
  const std::vector<float> staged = ReadBack(device_out, kTiles * kSlots);
  double worst = 0.0;
  std::size_t bad = 0;
  for (int tile = 0; tile < kTiles; ++tile) {
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
      const double difference = std::fabs(
          static_cast<double>(
              staged[static_cast<std::size_t>(tile) * kSlots + slot]) -
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
    // A fatal device error leaves through `std::exit`, which does not flush.
    std::fflush(stdout);
  }

 private:
  bool enabled_;
};

// The four row sets this one binary prints. `phase0` is the ceiling rows of
// docs/kernels.md Part 3, `h1` is the H1 rows and the maps of docs/attention.md
// Part 8, `flash` is the flash tile of the H2 prototype, and `h3` is the
// warp-specialized block of decision H3. `all` runs every set, so a caller that
// names no set loses no row.
enum class RowSet { kAll, kPhase0, kH1, kFlash, kH3 };

// The parsed command line.
struct ProbeOptions {
  nanochat::BenchOptions bench;
  RowSet rows = RowSet::kAll;
  // Flushes the second-level cache before every timed call.
  bool flush = true;
  // States the shape space and exits, so the table is readable with no card and
  // no measurement.
  bool shapes_only = false;

  bool phase0() const {
    return rows == RowSet::kAll || rows == RowSet::kPhase0;
  }
  bool h1() const { return rows == RowSet::kAll || rows == RowSet::kH1; }
  bool flash() const { return rows == RowSet::kAll || rows == RowSet::kFlash; }
  bool h3() const { return rows == RowSet::kAll || rows == RowSet::kH3; }
  const char* name() const {
    switch (rows) {
      case RowSet::kPhase0:
        return "phase0";
      case RowSet::kH1:
        return "h1";
      case RowSet::kFlash:
        return "flash";
      case RowSet::kH3:
        return "h3";
      case RowSet::kAll:
        break;
    }
    return "all";
  }
};

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--rows phase0|h1|flash|h3|all] [--iters N] "
               "[--rounds N] [--warmup N] [--no-flush] [--shapes] [--json] "
               "[--out PATH]\n",
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
      } else if (value == "h3") {
        options.rows = RowSet::kH3;
      } else if (value == "all") {
        options.rows = RowSet::kAll;
      } else {
        PrintUsage(argv[0]);
        std::exit(2);
      }
    } else if (argument == "--no-flush") {
      options.flush = false;
    } else if (argument == "--shapes") {
      options.shapes_only = true;
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

// The registers of one staged score instantiation, read from the build. The
// instantiation is named inside a template function because a lambda that names
// a kernel template-id with defaulted arguments does not resolve it.
template <int Warpgroups, int QueryTiles, int KeyWidth = kAtomN>
int StagedRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      StagedKernel<Warpgroups, QueryTiles, true, kStages, KeyWidth>));
}

// The registers of one resident flash instantiation, read from the build.
template <int Warpgroups, bool kMask, bool kSoftmax, bool kTrim = false,
          bool kGuard = false>
int ResidentFlashRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      ResidentFlashKernel<Warpgroups, kMask, kSoftmax, kTrim, kGuard>));
}

// The registers of one staged flash instantiation, read from the build. Both
// rings are staged, because that is the shape a product row times.
template <int Warpgroups, int Stages, bool kMask, bool kSoftmax = true>
int StagedFlashRegisters() {
  return CompilerRegisters(reinterpret_cast<const void*>(
      StagedFlashKernel<Warpgroups, Stages, true, kRingKeyValue, kMask,
                        kSoftmax>));
}

// States every row of the shape space: the twelve geometries this file
// instantiates, with the register count its own build states, and the four
// candidates, with the estimate the identity of the shape space gives them.
// This is the table `--shapes` prints and the table a full run ends with, so
// the shape that reached the highest rate and the shape that cannot launch are
// one document and the second does not depend on the first having run.
// One operand sweep: the staging a descriptor can name, the transpose bit, the
// step to the next atom, and the two offsets. One list states the
// configurations and one function runs them, so the score product's operand and
// the value product's operand are swept by one code path.
struct OperandConfig {
  int transpose_stage;
  int trans_b;
  int atom_step;
  int lbo;
  int sbo;
  int layout;
  //: The atoms one panel holds, and the step to the next panel, in sixteen-byte
  //: units. A swizzled tile of a hundred and twenty-eight columns is two panels
  //: of sixty-four, so the atom step changes once and this pair states where.
  int panel_atoms = 0;
  int panel_step = 0;
};

// One one-hot triple: a query row, the key the product of the left operand
// carries, and the head-width element of the right operand.
struct OperandTriple {
  int row;
  int key;
  int dim;
};

// One sweep's answer: how many configurations read, and which one that did not
// came closest.
struct OperandSweep {
  int configs = 0;
  int configs_ok = 0;
  int best_found = 0;
  int best_missed_found = 0;
  OperandConfig best{0, 0, 0, 0, 0, 0};
  OperandConfig best_missed{0, 0, 0, 0, 0, 0};
  std::string pass_configs;
};

// The configuration a diagnostic line and a baselines entry state.
std::string Describe(const OperandConfig& config) {
  return "transpose_stage=" + std::to_string(config.transpose_stage) +
         " trans_b=" + std::to_string(config.trans_b) +
         " atom_step=" + std::to_string(config.atom_step) +
         " lbo=" + std::to_string(config.lbo) +
         " sbo=" + std::to_string(config.sbo) +
         " layout=" + std::to_string(config.layout) +
         (config.panel_atoms > 0
              ? " panel_atoms=" + std::to_string(config.panel_atoms) +
                    " panel_step=" + std::to_string(config.panel_step)
              : std::string());
}

// Runs one sweep and returns its answer. `key_product` states which product is
// swept, and with it where the one-hot of the left operand sits: the
// probability tile's `k` axis is the key index and the query tile's is the head
// width, and the column the product must land in follows.
// The one driver entry this file takes. Nothing else of the driver API is
// needed, so the probe links no driver library and asks the runtime for the
// symbol instead.
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

// The map of the interleaved form over `tiles` key or value tiles of a
// row-major source: five axes, four of them with the stride the source gives
// them, and the shared order of the tile is the order of the axes.
bool MakeInterleavedMap(CUtensorMap* map, const void* source, int tiles,
                        int rows, int columns) {
  // Axis 0 is the eight elements of one core matrix, so its stride is the two
  // bytes of an element and the encoder does not take it. The four strides the
  // encoder takes are the ones of axis 1 to 4: the eight-row step of a group
  // (one row of the source), the half of a sixteen-wide block, the key group,
  // and the head-width block.
  const cuuint64_t dims[5] = {
      8, 8, 2,
      static_cast<cuuint64_t>(rows) * static_cast<cuuint64_t>(tiles) / 8,
      static_cast<cuuint64_t>(columns) / 16};
  const cuuint64_t strides[4] = {
      static_cast<cuuint64_t>(columns) * 2, 16,
      static_cast<cuuint64_t>(8) * static_cast<cuuint64_t>(columns) * 2, 32};
  const cuuint32_t box[5] = {8, 8, 2, static_cast<cuuint32_t>(rows) / 8,
                             static_cast<cuuint32_t>(columns) / 16};
  const cuuint32_t element[5] = {1, 1, 1, 1, 1};
  const CUresult status = EncodeTiled()(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void*>(source), dims,
      strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "fatal: the interleaved map was refused (%d)\n",
                 static_cast<int>(status));
    return false;
  }
  return true;
}

// The map of one panel of a swizzled tile: sixty-four elements of a row are a
// hundred and twenty-eight bytes, which is the span the engine's swizzle
// covers, so a row of a hundred and twenty-eight elements is two panels and two
// maps.
bool MakeSwizzledPanelMap(CUtensorMap* map, const void* source, int rows,
                          int box_rows, int panel) {
  const void* base =
      static_cast<const char*>(source) + static_cast<std::size_t>(panel) * 128;
  const cuuint64_t dims[2] = {64, static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(kDim) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element[2] = {1, 1};
  const CUresult status = EncodeTiled()(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims,
      strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "fatal: the swizzled panel map was refused (%d)\n",
                 static_cast<int>(status));
    return false;
  }
  return true;
}

OperandSweep RunOperandSweep(
    const std::string& name, const std::vector<OperandConfig>& configs,
    const std::vector<OperandTriple>& triples, bool key_product,
    std::vector<nanochat::Bf16>* host_left,
    std::vector<nanochat::Bf16>* host_right, Bf16* device_left,
    Bf16* device_right, const CUtensorMap& panel_low,
    const CUtensorMap& panel_high, float* device_out, const Diagnostics& note) {
  OperandSweep sweep;
  sweep.configs = static_cast<int>(configs.size());
  for (const OperandConfig& config : configs) {
    int found = 0;
    for (const OperandTriple& triple : triples) {
      std::fill(host_left->begin(), host_left->end(),
                nanochat::Bf16FromFloat(0.0f));
      std::fill(host_right->begin(), host_right->end(),
                nanochat::Bf16FromFloat(0.0f));
      const int left_axis = key_product ? triple.dim : triple.key;
      (*host_left)[static_cast<std::size_t>(triple.row) * kDim + left_axis] =
          nanochat::Bf16FromFloat(1.0f);
      (*host_right)[static_cast<std::size_t>(triple.key) * kDim + triple.dim] =
          nanochat::Bf16FromFloat(1.0f);
      Upload(*host_left, device_left);
      Upload(*host_right, device_right);
      OperandSweepKernel<<<1, kWarpgroupThreads, kProbeSharedBytes>>>(
          device_left, device_right, config.atom_step, config.trans_b,
          config.transpose_stage, config.lbo, config.sbo, config.layout,
          config.panel_atoms, config.panel_step, panel_low, panel_high,
          device_out);
      SyncFlash(name + " " + Describe(config));
      const std::vector<float> out = ReadBack(device_out, kProductSlots);
      const int expected_column = key_product ? triple.key : triple.dim;
      int one = 0;
      bool at_expected = false;
      for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
        for (int index = 0; index < kRegisters; ++index) {
          const float value =
              out[static_cast<std::size_t>(lane) * kRegisters + index];
          if (std::fabs(value) > 0.5f) {
            ++one;
            const std::pair<int, int> slot = SlotOf(lane, index);
            if (slot.first == triple.row && slot.second == expected_column) {
              at_expected = true;
            }
          }
        }
      }
      if (one == 1 && at_expected) ++found;
    }
    const int expected = static_cast<int>(triples.size());
    const bool this_ok = found == expected;
    if (this_ok) ++sweep.configs_ok;
    if (found > sweep.best_found) {
      sweep.best_found = found;
      sweep.best = config;
    }
    if (!this_ok && found > sweep.best_missed_found) {
      sweep.best_missed_found = found;
      sweep.best_missed = config;
    }
    // A near miss is the useful line when nothing passes: it names the fields
    // the reading does agree with.
    if (found >= 2 && !this_ok) {
      note("attention-wgmma:%s_near %s found=%d\n", name.c_str(),
           Describe(config).c_str(), found);
    }
    if (this_ok) {
      sweep.pass_configs += Describe(config) + "; ";
      note("attention-wgmma:%s %s found=%d of=%d\n", name.c_str(),
           Describe(config).c_str(), found, expected);
    }
  }
  return sweep;
}

// The document's fields of one sweep, under the row-name prefix its subject
// states.
void WriteOperandSweep(nanochat::BenchReport* report, const std::string& name,
                       const OperandSweep& sweep) {
  report->SetExtraInt(name + ":configs", sweep.configs);
  report->SetExtraText(name + ":pass_config", sweep.pass_configs.empty()
                                                  ? "none"
                                                  : sweep.pass_configs);
  report->SetExtraInt(name + ":triples", 6);
  report->SetExtraInt(name + ":configs_ok", sweep.configs_ok);
  report->SetExtraInt(name + ":best_found", sweep.best_found);
  report->SetExtraText(name + ":best_config", Describe(sweep.best));
  report->SetExtraInt(name + ":best_missed_found", sweep.best_missed_found);
  report->SetExtraText(name + ":best_missed_config",
                       Describe(sweep.best_missed));
  report->SetExtraText(name + ":status",
                       sweep.configs_ok > 0 ? "ok" : "failed");
}

void StateShapeSpace(ShapeTable* shapes) {
  for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
    for (int query_tiles = 1; query_tiles <= kMaxQueryTiles; ++query_tiles) {
      VisitGeometry(warpgroups, query_tiles, [&](auto count, auto tiles) {
        constexpr int kWarpgroups = decltype(count)::value;
        constexpr int kQueryTiles = decltype(tiles)::value;
        shapes->Observe(FindShape(ShapeKind::kScore, kWarpgroups, kQueryTiles,
                                  kAtomN, kStages),
                        StagedRegisters<kWarpgroups, kQueryTiles>());
      });
    }
  }
  for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
    for (int stages = 2; stages <= 3; ++stages) {
      VisitFlashGeometry(warpgroups, stages, [&](auto count, auto ring) {
        constexpr int kWarpgroups = decltype(count)::value;
        constexpr int kRing = decltype(ring)::value;
        shapes->Observe(
            FindShape(ShapeKind::kFlash, kWarpgroups, 1, kAtomN, kRing),
            StagedFlashRegisters<kWarpgroups, kRing, false>());
      });
    }
  }
  VisitNarrowGeometry(
      kMaxWarpgroups, kNarrowQueryTiles, [&](auto count, auto tiles) {
        constexpr int kWarpgroups = decltype(count)::value;
        constexpr int kQueryTiles = decltype(tiles)::value;
        shapes->Observe(
            FindShape(ShapeKind::kScore, kWarpgroups, kQueryTiles,
                      kNarrowKeyWidth, kStages),
            StagedRegisters<kWarpgroups, kQueryTiles, kNarrowKeyWidth>());
      });
  shapes->Observe(FindShape(ShapeKind::kFlashResident, 1, 1, kAtomN, 1),
                  ResidentFlashRegisters<1, false, true>());
  shapes->Observe(FindShape(ShapeKind::kFlashResident, 2, 1, kAtomN, 1),
                  ResidentFlashRegisters<2, false, true>());
  shapes->Observe(FindShape(ShapeKind::kFlash, kMaxWarpgroups, 1, kAtomN, 2,
                            /*shared_value=*/true),
                  SharedValueRegisters<kMaxWarpgroups, 2, /*kMask=*/false>());
  // The warp-specialized shapes, at the producer width each one names: the
  // register count is the one the build gives a block of two roles, which is
  // the count the `setmaxnreg` of the producer warpgroup redistributes.
  for (const ShapeSpec& shape : kShapeSpace) {
    if (shape.kind != ShapeKind::kSpecial || !shape.instantiated) continue;
    VisitSpecializedProducer(shape.producer_threads, [&](auto producer) {
      constexpr int kProducer = decltype(producer)::value;
      VisitSpecializedConsumers(shape.warpgroups, [&](auto consumers) {
        constexpr int kConsumers = decltype(consumers)::value;
        VisitSpecializedRing(shape.stages, [&](auto ring) {
          constexpr int kRing = decltype(ring)::value;
          shapes->Observe(shape,
                          SpecializedRegisters<kProducer, kConsumers, kRing,
                                               /*kMask=*/false>());
        });
      });
    });
  }
  // The candidates hold no instantiation of this file, so the arithmetic of the
  // shape space is the count they state.
  for (const ShapeSpec& shape : kShapeSpace) {
    if (!shape.instantiated) shapes->Observe(shape, 0);
  }
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
  if (options.h3()) {
    if (!checks_run.empty()) checks_run += "+";
    checks_run += "h3";
  }
  report.SetExtraText("checks_run", checks_run.empty() ? "none" : checks_run);

  note(
      "attention wgmma probe: the accumulator map, the operand form, the "
      "staged "
      "rate, and the flash tile, %d by %d by %d\n",
      kAtomM, kAtomN, kDim);

  // The shape space of the staged and flash rows: the tile of every geometry
  // this file instantiates and the four candidates the register identity opens
  // and closes. A product row states its shape here before it launches, and a
  // shape the table refuses never runs, so the register file, the shared
  // window, and the thread count are checked before a measurement instead of
  // after it. `--shapes` prints the space and exits, which needs no card.
  ShapeTable shapes;
  StateShapeSpace(&shapes);
  if (options.shapes_only) {
    // The table alone: a document with the shape space and no row, or the human
    // table. Neither needs a card, a clock, or a measurement.
    shapes.Write(&report);
    if (options.bench.json) {
      report.Print(options.bench);
    } else {
      // No peak is declared yet, so the table states no compute share.
      shapes.Print(stdout, /*peak_tflops=*/0.0);
    }
    return 0;
  }

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
  Bf16* device_special_queries = nullptr;
  Bf16* device_special_keys = nullptr;
  Bf16* device_special_values = nullptr;
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
             key_elements * sizeof(Bf16));
  // The warp-specialized rows hold their own operands, because a run of every
  // set would otherwise stage the flash rows from the set that follows them.
  cudaMalloc(reinterpret_cast<void**>(&device_special_queries),
             static_cast<std::size_t>(kSpecialTwoConsumers) * kAtomM * kDim *
                 sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_special_keys),
             key_elements * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_special_values),
             key_elements * sizeof(Bf16));
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
    Upload(host_q, device_q);
    Upload(host_k, device_k);
    // The scale both checks divide by. It is the largest element of the wide
    // reference, and the wide check runs first, so the narrow check reads the
    // scale of the same operand set.
    double scale = 0.0;
    // One key width, one element at a time. The first column of the key tile
    // holds the narrow product of the same operands, so the two checks state
    // the wide and the narrow instruction over one input.
    auto check_form = [&](auto width_constant) {
      constexpr int kWidth = decltype(width_constant)::value;
      constexpr int kTileRegisters = KeyRegisters(kWidth);
      std::vector<double> reference(static_cast<std::size_t>(kAtomM) * kWidth,
                                    0.0);
      for (int m = 0; m < kAtomM; ++m) {
        for (int n = 0; n < kWidth; ++n) {
          double sum = 0.0;
          for (int k = 0; k < kDim; ++k) {
            sum += static_cast<double>(nanochat::Bf16ToFloat(
                       host_q[static_cast<std::size_t>(m) * kDim + k])) *
                   static_cast<double>(nanochat::Bf16ToFloat(
                       host_k[static_cast<std::size_t>(n) * kDim + k]));
          }
          reference[static_cast<std::size_t>(m) * kWidth + n] = sum;
          scale = std::max(scale, std::fabs(sum));
        }
      }
      const std::vector<float> resident =
          RunResident<kWidth>(device_q, device_k, device_out);
      double worst = 0.0;
      for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
        for (int index = 0; index < kTileRegisters; ++index) {
          const std::pair<int, int> slot = SlotOf(lane, index);
          const double expected =
              reference[static_cast<std::size_t>(slot.first) * kWidth +
                        slot.second];
          const double actual =
              resident[static_cast<std::size_t>(lane) * kTileRegisters + index];
          worst = std::max(worst, std::fabs(actual - expected));
        }
      }
      const double form_error = worst / scale;
      const std::string name =
          kWidth == kAtomN ? "form_check" : "form_check_narrow";
      note("attention-wgmma:%s keys=%d error=%.3e scale=%.3f status=%s\n",
           name.c_str(), kWidth, form_error, scale,
           form_error < 3e-2 ? "ok" : "failed");
      report.SetExtra(name + ":error", form_error);
      report.SetExtra(name + ":scale", scale);
      report.SetExtraText(name + ":status",
                          form_error < 3e-2 ? "ok" : "failed");
      ok = ok && form_error < 3e-2;
    };
    check_form(std::integral_constant<int, kAtomN>{});
    check_form(std::integral_constant<int, kNarrowKeyWidth>{});

    // --- The staged path
    // ------------------------------------------------------ The staged product
    // of every geometry must agree with the resident product of its own query
    // tile before either is timed. Every query tile is a different operand, so
    // a warpgroup that reads the wrong tile of the query region fails here, and
    // so does a wrong ring, a wrong barrier, or a wrong stage offset.
    std::vector<std::vector<float>> reference_tiles(kMaxTiles);
    std::vector<std::vector<float>> narrow_reference(kMaxTiles);
    for (int tile = 0; tile < kMaxTiles; ++tile) {
      const std::size_t offset = static_cast<std::size_t>(tile) * host_q.size();
      for (std::size_t element = 0; element < host_q.size(); ++element) {
        host_queries[offset + element] = nanochat::Bf16FromFloat(random.Next());
        host_q[element] = host_queries[offset + element];
      }
      Upload(host_q, device_q);
      reference_tiles[static_cast<std::size_t>(tile)] =
          RunResident(device_q, device_k, device_out);
      // The narrow check reads the narrow instruction over the same query tile,
      // so it compares against the narrow resident product of that tile.
      narrow_reference[static_cast<std::size_t>(tile)] =
          RunResident<kNarrowKeyWidth>(device_q, device_k, device_out);
    }
    Upload(host_queries, device_queries);
    Upload(host_k, device_k);
    bool staged_ok = true;
    auto check_shape = [&](auto width_constant, auto warpgroups,
                           auto query_tiles) {
      constexpr int kWidth = decltype(width_constant)::value;
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kQueryTiles = decltype(query_tiles)::value;
      constexpr int kTiles = kWarpgroups * kQueryTiles;
      const std::vector<std::vector<float>>& reference =
          kWidth == kAtomN ? reference_tiles : narrow_reference;
      for (int options : {0, kPlainStage}) {
        const std::pair<std::size_t, double> result =
            CheckStagedGeometry<kWarpgroups, kQueryTiles, kWidth>(
                device_queries, device_k, device_out, kSourceTiles, options,
                reference, scale);
        note(
            "attention-wgmma:staged_check keys=%d warpgroups=%d query_tiles=%d "
            "options=%d error=%.3e bad=%zu of %d status=%s\n",
            kWidth, kWarpgroups, kQueryTiles, options, result.second,
            result.first, kTiles * KeySlots(kWidth),
            result.second < 1e-6 ? "ok" : "failed");
        // The key names the key width, the geometry, and the staging path,
        // because the loop runs once per pair and a repeated key would state
        // the same field twice in one document.
        const std::string prefix =
            "staged_check:" +
            std::string(options == kPlainStage ? "plain" : "async") + ":keys" +
            std::to_string(kWidth) + ":warpgroups" +
            std::to_string(kWarpgroups) + ":query_tiles" +
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
    auto check_geometry = [&](auto warpgroups, auto query_tiles) {
      check_shape(std::integral_constant<int, kAtomN>{}, warpgroups,
                  query_tiles);
    };
    auto check_narrow = [&](auto warpgroups, auto query_tiles) {
      check_shape(std::integral_constant<int, kNarrowKeyWidth>{}, warpgroups,
                  query_tiles);
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int query_tiles = 1; query_tiles <= kMaxQueryTiles; ++query_tiles) {
        VisitGeometry(warpgroups, query_tiles, check_geometry);
      }
    }
    VisitNarrowGeometry(kMaxWarpgroups, kNarrowQueryTiles, check_narrow);
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

    // --- The value operand
    // Part 8 section 5 records the second transpose bit as unmeasured. One
    // element at a time: a one-hot probability element at (row, key) and a
    // one-hot value element at (key, dim) leave one non-zero result, and it
    // must land at (row, dim) for the form, the step, and the offsets to be
    // right.
    {
      const std::vector<OperandTriple> triples = {{0, 0, 0},    {3, 5, 7},
                                                  {17, 33, 65}, {63, 127, 127},
                                                  {8, 64, 0},   {0, 127, 64}};
      std::vector<nanochat::Bf16> host_left(static_cast<std::size_t>(kAtomM) *
                                            kDim);
      std::vector<nanochat::Bf16> host_right(static_cast<std::size_t>(kAtomN) *
                                             kDim);
      Bf16* device_left = nullptr;
      Bf16* device_right = nullptr;
      cudaMalloc(reinterpret_cast<void**>(&device_left),
                 host_left.size() * sizeof(nanochat::Bf16));
      cudaMalloc(reinterpret_cast<void**>(&device_right),
                 host_right.size() * sizeof(nanochat::Bf16));
      cudaFuncSetAttribute(reinterpret_cast<const void*>(OperandSweepKernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kProbeSharedBytes);
      std::vector<OperandConfig> configs;
      // The transposed staging with both bits clear is the form the value
      // product reads today, so it is one entry of every list.
      configs.push_back(
          {1, 0, AtomStep(kAtomN), kLboBytes, kSboBytes, /*layout=*/0});
      // The staging a copy engine can fill: the value tile in the interleaved
      // form of the key tile, whose rows are the key index and whose contiguous
      // axis is the head width. Read with the transpose bit, that is the value
      // operand, and the tile then costs a copy and no transpose. Neither the
      // step along the key axis nor the two offsets is published for that
      // reading, so the sweep walks every multiple of one core-matrix group
      // over a whole tile, with the two offsets in both orders.
      for (int trans_b : {0, 1}) {
        for (int atom_step = 8; atom_step <= 512; atom_step += 8) {
          for (int lbo : {kLboBytes, kSboBytes}) {
            for (int sbo : {kLboBytes, kSboBytes}) {
              configs.push_back(
                  {0, trans_b, atom_step, lbo, sbo, /*layout=*/0});
            }
          }
        }
      }
      // The transposed staging over the transpose bit it is not known to need.
      configs.push_back({1, 1, AtomStep(kAtomN), kLboBytes, kSboBytes, 0});
      // The swizzled stagings, which are the ones a Tensor Memory Accelerator
      // descriptor reads: the stand-in 128-byte swizzle and the canonical one,
      // both transpose bits. A swizzled tile steps to the next atom by tens of
      // bytes and not by the interleaved form's kilobytes, so the steps are
      // stated in units of sixteen bytes from one.
      for (int layout : {1, 4}) {
        for (int trans_b : {0, 1}) {
          for (int atom_step : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
            for (int lbo : {16, 32, kLboBytes, 256, 1024}) {
              for (int sbo : {16, 32, kLboBytes, 256, 1024, 2048}) {
                configs.push_back({2, trans_b, atom_step, lbo, sbo, layout});
              }
            }
          }
        }
      }
      const CUtensorMap no_map{};
      const OperandSweep sweep = RunOperandSweep(
          "value_operand", configs, triples, /*key_product=*/false, &host_left,
          &host_right, device_left, device_right, no_map, no_map, device_out,
          note);
      WriteOperandSweep(&report, "value_operand", sweep);
      ok = ok && sweep.configs_ok > 0;
      cudaFree(device_left);
      cudaFree(device_right);
    }

    // --- The key operand
    // Decision H2b stages the key tile with the Tensor Memory Accelerator,
    // which writes a swizzled row-major tile and not the interleaved form the
    // staged rows of this file fill. The same one-element check states which
    // swizzle and which descriptor fields read it: a one-hot query element at
    // (row, dim) and a one-hot key element at (key, dim) must leave one result,
    // and it must land at (row, key).
    {
      const std::vector<OperandTriple> triples = {{0, 0, 0},    {3, 5, 7},
                                                  {17, 33, 65}, {63, 127, 127},
                                                  {8, 64, 0},   {0, 127, 64}};
      std::vector<nanochat::Bf16> host_left(static_cast<std::size_t>(kAtomM) *
                                            kDim);
      std::vector<nanochat::Bf16> host_right(static_cast<std::size_t>(kAtomN) *
                                             kDim);
      Bf16* device_left = nullptr;
      Bf16* device_right = nullptr;
      cudaMalloc(reinterpret_cast<void**>(&device_left),
                 host_left.size() * sizeof(nanochat::Bf16));
      cudaMalloc(reinterpret_cast<void**>(&device_right),
                 host_right.size() * sizeof(nanochat::Bf16));
      cudaFuncSetAttribute(reinterpret_cast<const void*>(OperandSweepKernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kProbeSharedBytes);
      std::vector<OperandConfig> configs;
      // The interleaved form the staged score rows fill, as the control: it
      // must read, or the sweep itself is wrong.
      configs.push_back({0, 0, AtomStep(kAtomN), kLboBytes, kSboBytes, 0});
      for (int layout : {1, 4}) {
        for (int atom_step : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
          for (int lbo : {16, 32, kLboBytes, 256, 1024}) {
            for (int sbo : {16, 32, kLboBytes, 256, 1024, 2048}) {
              configs.push_back({2, 0, atom_step, lbo, sbo, layout});
            }
          }
        }
      }
      // The swizzled tile the copy engine fills, in its two panels: the
      // stand-in this file writes, and the engine itself. The descriptor fields
      // are the ones that layout leaves open, so both families sweep them.
      // A hundred and twenty-eight columns are two panels of sixty-four, so the
      // atom step changes once and the panel pair below states where; every
      // other field is the descriptor's.
      for (int stage : {2, 3}) {
        // The stand-in writes the panels and the descriptor names swizzle 1
        // through the same field; the engine fills the same tile, so its
        // descriptor states the swizzle and no staging.
        const int layout = stage == 2 ? 5 : 1;
        for (int atom_step : {1, 2, 4, 8, 16}) {
          for (int panel_atoms : {4, 8}) {
            for (int panel_step : {512, 1024, 2048}) {
              for (int lbo : {16, 32, kLboBytes, 256}) {
                for (int sbo : {16, 32, 64, kLboBytes, 256, 1024}) {
                  configs.push_back({stage, 0, atom_step, lbo, sbo, layout,
                                     panel_atoms, panel_step});
                }
              }
            }
          }
        }
      }
      CUtensorMap panel_low{};
      CUtensorMap panel_high{};
      const bool panels =
          MakeSwizzledPanelMap(&panel_low, device_right, kAtomN, kAtomN,
                               /*panel=*/0) &&
          MakeSwizzledPanelMap(&panel_high, device_right, kAtomN, kAtomN,
                               /*panel=*/1);
      if (!panels) ok = false;
      const OperandSweep sweep =
          RunOperandSweep("key_operand", configs, triples, /*key_product=*/true,
                          &host_left, &host_right, device_left, device_right,
                          panel_low, panel_high, device_out, note);
      WriteOperandSweep(&report, "key_operand", sweep);
      ok = ok && sweep.configs_ok > 0;
      cudaFree(device_left);
      cudaFree(device_right);
    }
    // --- The copy engine
    // Decision H2b: one instruction a stage instead of a thread a sixteen-byte
    // group. The map states the interleaved form as five axes, so the check is
    // the one a wrong form fails: the tile the copy lands is the tile the
    // threads stage, byte for byte.
    {
      std::vector<nanochat::Bf16> host_tile(static_cast<std::size_t>(kAtomN) *
                                            kDim);
      Random tma_random;
      for (nanochat::Bf16& value : host_tile) {
        value = nanochat::Bf16FromFloat(tma_random.Next());
      }
      Bf16* device_tile = nullptr;
      cudaMalloc(reinterpret_cast<void**>(&device_tile),
                 host_tile.size() * sizeof(nanochat::Bf16));
      Upload(host_tile, device_tile);
      CUtensorMap map{};
      const bool built =
          MakeInterleavedMap(&map, device_tile, /*tiles=*/1, kAtomN, kDim);
      bool tma_ok = built;
      std::size_t tma_bad = 0;
      if (built) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(TmaStageKernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kKeyBytes);
        cudaMemset(device_out, 0, kKeyBytes);
        TmaStageKernel<<<1, kWarpgroupThreads, kKeyBytes>>>(
            map, /*group=*/0, kKeyBytes, device_out);
        SyncFlash("tma stage");
        const std::vector<float> staged = ReadBack(device_out, kKeyBytes / 4);
        for (int row = 0; row < kAtomN; ++row) {
          for (int k = 0; k < kDim; k += 2) {
            const int offset = InterleavedOffset(row, k, kAtomN);
            const std::uint16_t low =
                host_tile[static_cast<std::size_t>(row) * kDim + k].bits;
            const std::uint16_t high =
                host_tile[static_cast<std::size_t>(row) * kDim + k + 1].bits;
            const std::uint32_t expected =
                static_cast<std::uint32_t>(low) |
                (static_cast<std::uint32_t>(high) << 16);
            float actual = 0.0f;
            std::memcpy(&actual, &staged[static_cast<std::size_t>(offset / 4)],
                        sizeof(actual));
            std::uint32_t bits = 0;
            std::memcpy(&bits, &actual, sizeof(bits));
            if (bits != expected) ++tma_bad;
          }
        }
        tma_ok = tma_bad == 0;
      }
      note("attention-wgmma:tma_stage bytes=%d bad=%zu status=%s\n", kKeyBytes,
           tma_bad, tma_ok ? "ok" : "failed");
      report.SetExtraInt("tma_stage:bad", static_cast<long long>(tma_bad));
      report.SetExtraInt("tma_stage:words", kKeyBytes / 4);
      report.SetExtraText("tma_stage:status", tma_ok ? "ok" : "failed");
      ok = ok && tma_ok;
      cudaFree(device_tile);
    }

  }  // options.h1(): the map and form sections

  // --- The flash tile
  // The checks run before a rate is timed. The resident step is compared with
  // the host reference of the family one element at a time, so a wrong
  // probability operand, value staging, row reduction, or rescale fails here.
  // Every staged geometry is then compared with that resident step.
  if (options.flash()) {
    SetFlashSharedMemory<1, false>();
    SetFlashSharedMemory<2, false>();
    SetFlashSharedMemory<1, true>();
    SetFlashSharedMemory<2, true>();
    // The window belongs to one instantiation, so the trimmed step of the same
    // geometry states it again before its own check.
    SetFlashSharedMemory<1, false, /*kSoftmax=*/true, /*kTrim=*/true,
                         /*kGuard=*/true>();
    SetFlashSharedMemory<1, true, /*kSoftmax=*/true, /*kTrim=*/true,
                         /*kGuard=*/true>();
    SetFlashSharedMemory<2, false, /*kSoftmax=*/true, /*kTrim=*/true,
                         /*kGuard=*/true>();
    SetFlashSharedMemory<2, true, /*kSoftmax=*/true, /*kTrim=*/true,
                         /*kGuard=*/true>();

    Random flash_random;
    auto draw = [&](std::vector<nanochat::Bf16>* values) {
      for (nanochat::Bf16& value : *values) {
        value = nanochat::Bf16FromFloat(flash_random.Next());
      }
    };
    // The reference reads the operands the device reads, so it is built from
    // the bf16 values and not from the draws that produced them.
    auto widen = [](const std::vector<nanochat::Bf16>& values) {
      std::vector<float> out(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = nanochat::Bf16ToFloat(values[i]);
      }
      return out;
    };
    std::vector<nanochat::Bf16> flash_q(host_q.size());
    std::vector<nanochat::Bf16> flash_k(host_k.size());
    std::vector<nanochat::Bf16> flash_v(host_v.size());
    draw(&flash_q);
    draw(&flash_k);
    draw(&flash_v);
    const std::vector<float> reference_q = widen(flash_q);
    const std::vector<float> reference_k = widen(flash_k);
    const std::vector<float> reference_v = widen(flash_v);
    // The dense step and the causal boundary tile are each compared with the
    // host reference of the family, one element at a time. The kernel leaves
    // its output unnormalised, so the division by the row sum is the caller's,
    // and it is what makes a wrong maximum or a wrong sum visible.
    Upload(flash_q, device_q);
    Upload(flash_k, device_k);
    Upload(flash_v, device_value);
    auto check_mask = [&](auto mask_constant, auto trim_constant,
                          auto guard_constant) {
      constexpr bool kMask = decltype(mask_constant)::value;
      constexpr bool kTrim = decltype(trim_constant)::value;
      constexpr bool kGuard = decltype(guard_constant)::value;
      // One head and one batch. The dense check sees every key. The causal
      // check sees the keys up to the row's own index, which is the boundary
      // tile of a query block that starts at the key tile.
      nanochat::AttentionParams params;
      params.batch = 1;
      params.seq = kAtomM;
      params.num_heads = 1;
      params.num_kv_heads = 1;
      params.head_dim = kDim;
      params.causal = kMask;
      params.window_right = kMask ? 0 : -1;
      params.kv_len = kMask ? kAtomM : kAtomN;
      params.scale = kFlashScale;
      std::vector<float> reference_stats;
      const std::vector<float> reference =
          nanochat::dev::seqref::AttentionForward(
              params, reference_q, reference_k, reference_v, &reference_stats);
      const std::vector<float> slots =
          RunResidentFlash<1, kMask, /*kSoftmax=*/true, kTrim, kGuard>(
              device_q, device_k, device_value, device_out, device_row_sums);
      const std::vector<float> row_sums = ReadBack(device_row_sums, kAtomM);
      double scale_seen = 0.0;
      for (float value : reference) {
        scale_seen =
            std::max(scale_seen, std::fabs(static_cast<double>(value)));
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
              sum > 0.0
                  ? static_cast<double>(
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
      const double error = scale_seen > 0.0 ? worst / scale_seen : 0.0;
      const bool passed = error < 3e-2 && sum_worst < 3e-2;
      note(
          "attention-flash:reference_check mask=%s trim=%s error=%.3e "
          "sum_error=%.3e scale=%.3f bad=%zu of %d status=%s\n",
          kMask ? "causal" : "dense", kTrim ? "on" : "off", error, sum_worst,
          scale_seen, bad, kAtomM * kDim, passed ? "ok" : "failed");
      const std::string key = std::string("flash_check:") +
                              (kMask ? "causal:" : "dense:") +
                              (kTrim ? "trim:" : "") + (kGuard ? "guard:" : "");
      report.SetExtra(key + "error", error);
      report.SetExtra(key + "sum_error", sum_worst);
      report.SetExtra(key + "scale", scale_seen);
      report.SetExtraInt(key + "bad", static_cast<long long>(bad));
      report.SetExtraInt(key + "elements", kAtomM * kDim);
      report.SetExtraText(key + "status", passed ? "ok" : "failed");
      return passed;
    };
    ok = ok && check_mask(std::integral_constant<bool, false>{},
                          std::integral_constant<bool, false>{},
                          std::integral_constant<bool, false>{});
    ok = ok && check_mask(std::integral_constant<bool, true>{},
                          std::integral_constant<bool, false>{},
                          std::integral_constant<bool, false>{});
    // The same 2 checks on the trimmed step, because 3 of its 4 changes touch
    // the arithmetic the reference states: the flushed exponential, the folded
    // scale, and the skipped rescale.
    ok = ok && check_mask(std::integral_constant<bool, false>{},
                          std::integral_constant<bool, true>{},
                          std::integral_constant<bool, true>{});
    ok = ok && check_mask(std::integral_constant<bool, true>{},
                          std::integral_constant<bool, true>{},
                          std::integral_constant<bool, true>{});

    // The staged rows read one query tile a warpgroup from `device_queries`, so
    // the check states a distinct tile for every warpgroup of the deepest
    // geometry and then compares each staged geometry with the resident step of
    // the same operands.
    std::vector<nanochat::Bf16> flash_queries(
        static_cast<std::size_t>(kMaxWarpgroups) * flash_q.size());
    for (int tile = 0; tile < kMaxWarpgroups; ++tile) {
      for (nanochat::Bf16& value : flash_q) {
        value = nanochat::Bf16FromFloat(flash_random.Next());
      }
      std::copy(flash_q.begin(), flash_q.end(),
                flash_queries.begin() +
                    static_cast<std::size_t>(tile) * flash_q.size());
    }
    Upload(flash_queries, device_queries);

    bool staged_ok = true;
    auto check_geometry = [&](auto warpgroups, auto stages) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kRing = decltype(stages)::value;
      auto check_one = [&](auto mask_constant) {
        constexpr bool kMask = decltype(mask_constant)::value;
        SetStagedFlashSharedMemory<kWarpgroups, kRing, true, kRingKeyValue,
                                   kMask>();
        const std::vector<float> resident =
            RunResidentFlash<kWarpgroups, kMask>(device_queries, device_k,
                                                 device_value, device_out,
                                                 device_row_sums);
        const std::pair<std::size_t, double> result =
            CheckStagedFlash<kWarpgroups, kRing, kMask>(
                device_queries, device_k, device_value, device_out,
                device_row_sums, resident, /*scale=*/1.0);
        const std::string key = std::string("flash_staged_check:warpgroups") +
                                std::to_string(kWarpgroups) + ":stages" +
                                std::to_string(kRing) +
                                (kMask ? ":causal:" : ":dense:");
        note(
            "attention-flash:staged_check warpgroups=%d stages=%d mask=%s "
            "error=%.3e bad=%zu of %d status=%s\n",
            kWarpgroups, kRing, kMask ? "causal" : "dense", result.second,
            result.first, kWarpgroups * static_cast<int>(kProductSlots),
            result.second < 1e-6 ? "ok" : "failed");
        report.SetExtra(key + "error", result.second);
        report.SetExtraInt(key + "bad", static_cast<long long>(result.first));
        staged_ok = staged_ok && result.second < 1e-6;
      };
      check_one(std::integral_constant<bool, false>{});
      check_one(std::integral_constant<bool, true>{});
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int stages = 2; stages <= 3; ++stages) {
        VisitFlashGeometry(warpgroups, stages, check_geometry);
      }
    }

    // The same ring with the value tile carried by the copy engine and
    // transposed in shared memory, against the resident step of the same
    // operands. The element order inside a unit and the unit mapping are what
    // this check covers.
    auto check_shared_value = [&](auto mask_constant) {
      constexpr bool kMask = decltype(mask_constant)::value;
      const std::vector<float> resident =
          RunResidentFlash<kMaxWarpgroups, kMask>(device_queries, device_k,
                                                  device_value, device_out,
                                                  device_row_sums);
      const std::pair<std::size_t, double> result =
          CheckSharedValue<kMaxWarpgroups, 2, kMask>(
              device_queries, device_k, device_value, device_out,
              device_row_sums, resident, /*scale=*/1.0);
      const std::string key = std::string("flash_shared_value_check:") +
                              (kMask ? "causal:" : "dense:");
      note(
          "attention-flash:shared_value_check warpgroups=%d stages=2 mask=%s "
          "error=%.3e bad=%zu of %d status=%s\n",
          kMaxWarpgroups, kMask ? "causal" : "dense", result.second,
          result.first, kMaxWarpgroups * static_cast<int>(kProductSlots),
          result.second < 1e-6 ? "ok" : "failed");
      report.SetExtra(key + "error", result.second);
      report.SetExtraInt(key + "bad", static_cast<long long>(result.first));
      staged_ok = staged_ok && result.second < 1e-6;
    };
    check_shared_value(std::integral_constant<bool, false>{});
    check_shared_value(std::integral_constant<bool, true>{});
    ok = ok && staged_ok;

    // The engine-fed tile, against the resident step of the same operands. The
    // key operand changes form here and the value operand is transposed from a
    // swizzled stage, so this check is the one that says both are right.
    {
      CUtensorMap key_low{};
      CUtensorMap key_high{};
      CUtensorMap value_low{};
      CUtensorMap value_high{};
      const int rows = kAtomN * kSourceTiles;
      const bool built =
          MakeSwizzledPanelMap(&key_low, device_k, rows, kAtomN, /*panel=*/0) &&
          MakeSwizzledPanelMap(&key_high, device_k, rows, kAtomN,
                               /*panel=*/1) &&
          MakeSwizzledPanelMap(&value_low, device_value, rows, kAtomN,
                               /*panel=*/0) &&
          MakeSwizzledPanelMap(&value_high, device_value, rows, kAtomN,
                               /*panel=*/1);
      bool engine_ok = built;
      if (built) {
        auto check_mask = [&](auto mask_constant) {
          constexpr bool kMask = decltype(mask_constant)::value;
          const std::vector<float> resident =
              RunResidentFlash<kMaxWarpgroups, kMask>(device_queries, device_k,
                                                      device_value, device_out,
                                                      device_row_sums);
          // Both transpositions, because the rows time both.
          auto check_transpose = [&](auto warp_constant) {
            constexpr bool kWarp = decltype(warp_constant)::value;
            const std::pair<std::size_t, double> result =
                CheckEngineFlash<kMaxWarpgroups, 2, kMask, kWarp>(
                    device_queries, key_low, key_high, value_low, value_high,
                    device_out, resident, /*scale=*/1.0);
            const std::string key = std::string("flash_engine_check:") +
                                    (kWarp ? "warp:" : "thread:") +
                                    (kMask ? "causal:" : "dense:");
            note(
                "attention-flash:engine_check consumers=%d stages=2 "
                "transpose=%s mask=%s error=%.3e bad=%zu of %d status=%s\n",
                kMaxWarpgroups, kWarp ? "warp" : "thread",
                kMask ? "causal" : "dense", result.second, result.first,
                kMaxWarpgroups * static_cast<int>(kProductSlots),
                result.second < 1e-6 ? "ok" : "failed");
            report.SetExtra(key + "error", result.second);
            report.SetExtraInt(key + "bad",
                               static_cast<long long>(result.first));
            engine_ok = engine_ok && result.second < 1e-6;
          };
          check_transpose(std::integral_constant<bool, false>{});
          check_transpose(std::integral_constant<bool, true>{});
          {
            // The trimmed step against the trimmed resident step of the same
            // operands, so the ring, the swizzled key reading, and the
            // transposition are still checked exactly under the arithmetic the
            // timed trimmed row states.
            const std::vector<float> trimmed_resident =
                RunResidentFlash<kMaxWarpgroups, kMask, /*kSoftmax=*/true,
                                 /*kTrim=*/true, /*kGuard=*/true>(
                    device_queries, device_k, device_value, device_out,
                    device_row_sums);
            const std::pair<std::size_t, double> trimmed =
                CheckEngineFlash<kMaxWarpgroups, 2, kMask,
                                 /*kWarpTranspose=*/false,
                                 /*kConsumerTranspose=*/false,
                                 /*kTrim=*/true, /*kGuard=*/true>(
                    device_queries, key_low, key_high, value_low, value_high,
                    device_out, trimmed_resident, /*scale=*/1.0);
            // The rotated store states the same tile, so the same reference
            // checks it.
            const std::pair<std::size_t, double> rotated =
                CheckEngineFlash<kMaxWarpgroups, 2, kMask,
                                 /*kWarpTranspose=*/false,
                                 /*kConsumerTranspose=*/false,
                                 /*kTrim=*/true, /*kGuard=*/true,
                                 /*kRotateStore=*/true>(
                    device_queries, key_low, key_high, value_low, value_high,
                    device_out, trimmed_resident, /*scale=*/1.0);
            const std::string rotated_key =
                std::string("flash_engine_check:rotated:") +
                (kMask ? "causal:" : "dense:");
            note(
                "attention-flash:engine_check consumers=%d stages=2 "
                "store=rotated mask=%s error=%.3e bad=%zu of %d status=%s\n",
                kMaxWarpgroups, kMask ? "causal" : "dense", rotated.second,
                rotated.first, kMaxWarpgroups * static_cast<int>(kProductSlots),
                rotated.second < 1e-6 ? "ok" : "failed");
            report.SetExtra(rotated_key + "error", rotated.second);
            report.SetExtraInt(rotated_key + "bad",
                               static_cast<long long>(rotated.first));
            engine_ok = engine_ok && rotated.second < 1e-6;
            const std::string trimmed_key =
                std::string("flash_engine_check:trim:") +
                (kMask ? "causal:" : "dense:");
            note(
                "attention-flash:engine_check consumers=%d stages=2 "
                "trim=on mask=%s error=%.3e bad=%zu of %d status=%s\n",
                kMaxWarpgroups, kMask ? "causal" : "dense", trimmed.second,
                trimmed.first, kMaxWarpgroups * static_cast<int>(kProductSlots),
                trimmed.second < 1e-6 ? "ok" : "failed");
            report.SetExtra(trimmed_key + "error", trimmed.second);
            report.SetExtraInt(trimmed_key + "bad",
                               static_cast<long long>(trimmed.first));
            engine_ok = engine_ok && trimmed.second < 1e-6;

            const std::pair<std::size_t, double> result =
                CheckEngineFlash<kMaxWarpgroups, 2, kMask,
                                 /*kWarpTranspose=*/false,
                                 /*kConsumerTranspose=*/true>(
                    device_queries, key_low, key_high, value_low, value_high,
                    device_out, resident, /*scale=*/1.0);
            const std::string key =
                std::string("flash_engine_check:consumer:") +
                (kMask ? "causal:" : "dense:");
            note(
                "attention-flash:engine_check consumers=%d stages=2 "
                "transpose=consumer mask=%s error=%.3e bad=%zu of %d "
                "status=%s\n",
                kMaxWarpgroups, kMask ? "causal" : "dense", result.second,
                result.first, kMaxWarpgroups * static_cast<int>(kProductSlots),
                result.second < 1e-6 ? "ok" : "failed");
            report.SetExtra(key + "error", result.second);
            report.SetExtraInt(key + "bad",
                               static_cast<long long>(result.first));
            engine_ok = engine_ok && result.second < 1e-6;
          }
        };
        check_mask(std::integral_constant<bool, false>{});
        check_mask(std::integral_constant<bool, true>{});
      }
      ok = ok && engine_ok;
    }
  }

  // --- The warp-specialized block
  // Decision H3 of Part 8 section 2: one producer issues the copies and never
  // computes, and the consumer warpgroups issue the products and never wait on
  // a global load. The two producer widths are the first phase of the decision,
  // one warp, and the full warpgroup that `setmaxnreg` spans. Every geometry is
  // checked against the resident step of the same operands before it is timed,
  // so a wrong barrier, stage offset, or role boundary fails here instead of
  // measuring.
  std::vector<nanochat::Bf16> special_queries(
      static_cast<std::size_t>(kSpecialTwoConsumers) * kAtomM * kDim);
  std::vector<nanochat::Bf16> special_keys(
      static_cast<std::size_t>(kSourceTiles) * kAtomN * kDim);
  std::vector<nanochat::Bf16> special_values(special_keys.size());
  if (options.h3()) {
    Random special_random;
    for (nanochat::Bf16& value : special_queries) {
      value = nanochat::Bf16FromFloat(special_random.Next());
    }
    // One key tile and one value tile, repeated over the source ring, so every
    // step of a timed row reads defined operands instead of whatever the
    // allocator left behind.
    const std::size_t tile_elements = static_cast<std::size_t>(kAtomN) * kDim;
    std::vector<nanochat::Bf16> key_tile(tile_elements);
    std::vector<nanochat::Bf16> value_tile(tile_elements);
    for (nanochat::Bf16& value : key_tile) {
      value = nanochat::Bf16FromFloat(special_random.Next());
    }
    for (nanochat::Bf16& value : value_tile) {
      value = nanochat::Bf16FromFloat(special_random.Next());
    }
    for (int tile = 0; tile < kSourceTiles; ++tile) {
      std::copy(key_tile.begin(), key_tile.end(),
                special_keys.begin() +
                    static_cast<std::size_t>(tile) * tile_elements);
      std::copy(value_tile.begin(), value_tile.end(),
                special_values.begin() +
                    static_cast<std::size_t>(tile) * tile_elements);
    }
    Upload(special_queries, device_special_queries);
    Upload(special_keys, device_special_keys);
    Upload(special_values, device_special_values);
  }

  if (options.h3()) {
    SetFlashSharedMemory<kSpecialOneConsumer, false>();
    SetFlashSharedMemory<kSpecialOneConsumer, true>();
    SetFlashSharedMemory<kSpecialTwoConsumers, false>();
    SetFlashSharedMemory<kSpecialTwoConsumers, true>();
    bool special_ok = true;
    auto check_producer = [&](auto producer) {
      constexpr int kProducer = decltype(producer)::value;
      auto check_consumers = [&](auto consumers) {
        constexpr int kConsumers = decltype(consumers)::value;
        auto check_ring = [&](auto ring) {
          constexpr int kRing = decltype(ring)::value;
          auto check_mask = [&](auto mask_constant) {
            constexpr bool kMask = decltype(mask_constant)::value;
            // Two tile counts: one step, and enough steps for the ring to wrap
            // twice, so the release waits and the stage reuse are checked and
            // not only the first fill.
            constexpr int kWrapTiles = 2 * kRing + 1;
            auto check_tiles = [&](int tiles) {
              const std::vector<float> resident =
                  RunResidentFlash<kConsumers, kMask>(
                      device_special_queries, device_special_keys,
                      device_special_values, device_out, device_row_sums,
                      tiles);
              const std::pair<std::size_t, double> result =
                  CheckSpecialized<kProducer, kConsumers, kRing, kMask>(
                      device_special_queries, device_special_keys,
                      device_special_values, device_out, device_row_sums,
                      resident, /*scale=*/1.0, tiles);
              const std::string key = std::string("special_check:producer") +
                                      std::to_string(kProducer) + ":consumers" +
                                      std::to_string(kConsumers) + ":stages" +
                                      std::to_string(kRing) +
                                      (kMask ? ":causal:" : ":dense:") +
                                      "tiles" + std::to_string(tiles) + ":";
              note(
                  "attention-special:check producer=%d consumers=%d stages=%d "
                  "mask=%s tiles=%d error=%.3e bad=%zu of %d status=%s\n",
                  kProducer, kConsumers, kRing, kMask ? "causal" : "dense",
                  tiles, result.second, result.first,
                  kConsumers * static_cast<int>(kProductSlots),
                  result.second < 1e-6 ? "ok" : "failed");
              report.SetExtra(key + "error", result.second);
              report.SetExtraInt(key + "bad",
                                 static_cast<long long>(result.first));
              special_ok = special_ok && result.second < 1e-6;
            };
            check_tiles(1);
            check_tiles(kWrapTiles);
          };
          check_mask(std::integral_constant<bool, false>{});
          check_mask(std::integral_constant<bool, true>{});
        };
        VisitSpecializedRing(kSpecialStages, check_ring);
        VisitSpecializedRing(kSpecialDeepStages, check_ring);
      };
      VisitSpecializedConsumers(kSpecialOneConsumer, check_consumers);
      VisitSpecializedConsumers(kSpecialTwoConsumers, check_consumers);
    };
    VisitSpecializedProducer(kSpecialWarpProducer, check_producer);
    VisitSpecializedProducer(kSpecialWarpgroupProducer, check_producer);
    ok = ok && special_ok;
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
    // Outside the timed window on purpose: the check names the row whose launch
    // the device refused, and it costs the measurement nothing.
    CheckDevice("row " + base.name);
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

  // One staged product row: the launch, the work it issues, and the arithmetic
  // one staged byte buys. The row is the caller's, because a boundary tile
  // states a work ratio and a plain tile does not, and this tail is the one
  // place both row sets state a product.
  auto add_product_rate = [&](const nanochat::BenchRow& row,
                              const std::string& extra_prefix,
                              const std::string& shape, double flops,
                              double flop_per_staged_byte, auto&& launch) {
    const nanochat::dev::Timing timing = harness.Time(launch);
    const double rate = add_rate_row(row, timing, flops);
    report.SetExtra(extra_prefix + shape + ":flop_per_staged_byte",
                    flop_per_staged_byte);
    note("%s %s min=%.3f ms tflops=%.1f flop_per_staged_byte=%.0f\n",
         row.name.c_str(), shape.c_str(), timing.min_ms,
         rate * kTflopsPerGflops, flop_per_staged_byte);
    return rate;
  };

  // One copy row: the bytes a launch staged, and the rate it reached. A copy
  // reads and writes no operand of a product, so the row states no FLOP and the
  // byte rate is its one number. The share columns of the document print `0.0`
  // and `-` for it.
  auto add_copy_rate =
      [&](const nanochat::BenchRow& row, const std::string& extra_prefix,
          const std::string& shape, double bytes, auto&& launch) {
        const nanochat::dev::Timing timing = harness.Time(launch);
        const double bandwidth = bytes / (timing.min_ms * 1e-3);
        add_rate_row(row, timing, /*flops=*/0.0);
        report.SetExtra(extra_prefix + shape + ":tbps", bandwidth / 1e12);
        note("%s %s min=%.3f ms tbps=%.2f\n", row.name.c_str(), shape.c_str(),
             timing.min_ms, bandwidth / 1e12);
        return bandwidth;
      };

  // States one shape before the row that prices it. The facts of the shape are
  // the ones the table already holds, so a row can never launch a geometry the
  // printed table does not state.
  auto admit = [&](const ShapeSpec& shape, int compiler_registers) {
    const ShapeFacts facts = shapes.Observe(shape, compiler_registers);
    if (!facts.fits) {
      note(
          "attention-wgmma:shape_refused name=%s warpgroups=%d query_tiles=%d "
          "keys=%d stages=%d threads=%d registers=%d registers_per_sm=%d "
          "smem_bytes=%d reason=%s\n",
          shape.name, shape.warpgroups, shape.query_tiles, shape.key_width,
          shape.stages, facts.threads, facts.registers,
          facts.registers_per_multiprocessor, facts.shared_bytes, facts.reason);
    }
    return facts.fits;
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
    // The flash rings read the same source window, so the value ring is the
    // whole ring and not the single tile the check stages. A run that names no
    // flash row stages no value tile, so it copies none.
    if (options.flash()) {
      cudaMemcpy(device_value, source.data(), key_elements * sizeof(Bf16),
                 cudaMemcpyHostToDevice);
    }
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
         shape.c_str(), timing.min_ms, timing.median_ms,
         rate * kTflopsPerGflops, share);
  };
  if (options.h1()) {
    // The copy engine's own rate: a ring of key tiles that one thread fills and
    // no thread stages, so the bytes a second are the path's and not a
    // thread's.
    {
      CUtensorMap map{};
      const bool built =
          MakeInterleavedMap(&map, device_k, kSourceTiles, kAtomN, kDim);
      bool ring_ok = built;
      if (built) {
        constexpr int kRing = 3;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(TmaRingKernel<kRing>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, kRing * kKeyBytes);
        const int tiles = 8192;
        const double bytes = static_cast<double>(kKeyBytes) *
                             static_cast<double>(tiles) *
                             static_cast<double>(sm_count);
        const double bandwidth = add_copy_rate(
            make_row("attention-wgmma:tma_copies",
                     "stages=" + std::to_string(kRing) +
                         " blocks=" + std::to_string(sm_count) +
                         " tiles=" + std::to_string(tiles),
                     "h1"),
            "tma_copies:", "ring", bytes, [&] {
              TmaRingKernel<kRing><<<sm_count, 32, kRing * kKeyBytes>>>(
                  map, tiles, /*group_step=*/kAtomN / 8, kKeyBytes);
            });
        report.SetExtra("tma_copies:bytes_per_second_per_block",
                        bandwidth / static_cast<double>(sm_count));
        ring_ok = bandwidth > 0.0;
      }
      report.SetExtraText("tma_copies:status", ring_ok ? "ok" : "failed");
      ok = ok && ring_ok;
    }

    // The engine's own rate on the box it likes: a swizzled panel, two a tile,
    // which is the form the fast row of the plan records at 6.03 TB/s.
    {
      CUtensorMap low{};
      CUtensorMap high{};
      const bool built =
          MakeSwizzledPanelMap(&low, device_k, kAtomN * kSourceTiles, kAtomN,
                               /*panel=*/0) &&
          MakeSwizzledPanelMap(&high, device_k, kAtomN * kSourceTiles, kAtomN,
                               /*panel=*/1);
      bool ring_ok = built;
      if (built) {
        constexpr int kRing = 3;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(TmaSwizzledRingKernel<kRing>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, kRing * kKeyBytes);
        const int tiles = 8192;
        const double bytes = static_cast<double>(kKeyBytes) *
                             static_cast<double>(tiles) *
                             static_cast<double>(sm_count);
        const double bandwidth = add_copy_rate(
            make_row("attention-wgmma:tma_swizzled_copies",
                     "stages=" + std::to_string(kRing) +
                         " blocks=" + std::to_string(sm_count) +
                         " tiles=" + std::to_string(tiles) + " swizzle=128b",
                     "h1"),
            "tma_swizzled_copies:", "ring", bytes, [&] {
              TmaSwizzledRingKernel<kRing><<<sm_count, 32, kRing * kKeyBytes>>>(
                  low, high, tiles, kAtomN);
            });
        report.SetExtra("tma_swizzled_copies:bytes_per_second_per_block",
                        bandwidth / static_cast<double>(sm_count));
        ring_ok = bandwidth > 0.0;
      }
      report.SetExtraText("tma_swizzled_copies:status",
                          ring_ok ? "ok" : "failed");
      ok = ok && ring_ok;
    }

    add_resident("attention-wgmma:resident", kAtoms, 1, 1);
    add_resident("attention-wgmma:resident", kAtoms, sm_count, 1);
    // The ceiling of the narrow instruction: the same one-warpgroup tile and
    // the same grid, over a key tile of half the width. The staged row divides
    // the released rate by this figure, so a narrow key tile that costs the
    // instruction nothing is stated and not assumed.
    SetResidentSharedMemory<kAtoms, kNarrowKeyWidth>();
    const std::string narrow_shape = "blocks=" + std::to_string(sm_count) +
                                     " passes=" + std::to_string(passes) +
                                     " atoms=" + std::to_string(kAtoms) +
                                     " keys=" + std::to_string(kNarrowKeyWidth);
    const double narrow_flops = FlopsPerPass(kAtoms, kNarrowKeyWidth) *
                                static_cast<double>(passes) *
                                static_cast<double>(sm_count);
    const nanochat::dev::Timing narrow_timing = harness.Time([&] {
      LaunchResident<kAtoms, kNarrowKeyWidth>(sm_count, device_q, device_k,
                                              device_out, passes);
    });
    const double narrow_rate = add_rate_row(
        make_row("attention-wgmma:resident_narrow", narrow_shape, "h1", 1),
        narrow_timing, narrow_flops);
    report.SetExtra("resident_narrow:sm_share",
                    sm_share(narrow_flops, narrow_timing.min_ms));
    note(
        "attention-wgmma:resident_narrow %s min=%.3f ms median=%.3f ms "
        "tflops=%.1f sm_share=%.3f\n",
        narrow_shape.c_str(), narrow_timing.min_ms, narrow_timing.median_ms,
        narrow_rate * kTflopsPerGflops,
        sm_share(narrow_flops, narrow_timing.min_ms));
  }
  if (options.phase0()) {
    // One row, because the full grid answers the question on its own: a card at
    // its declared peak carries no room above it, so no one-block row can state
    // a fraction the grid row does not already state. The one-warpgroup claim
    // stays measured by the `attention-wgmma:resident` row of the H1 set, which
    // launches one block over the eight-atom tile and reports its share of the
    // per-SM peak.
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

    // One timed product row of one geometry and one key width. The work is the
    // score product of every query tile of every warpgroup over every staged
    // key tile, so the rate and the arithmetic per staged byte cannot disagree.
    auto time_shape = [&](auto width_constant, auto warpgroups,
                          auto query_tiles) {
      constexpr int kWidth = decltype(width_constant)::value;
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kQueryTiles = decltype(query_tiles)::value;
      const ShapeSpec& shape_row = FindShape(ShapeKind::kScore, kWarpgroups,
                                             kQueryTiles, kWidth, kStages);
      if (!admit(shape_row,
                 StagedRegisters<kWarpgroups, kQueryTiles, kWidth>())) {
        return;
      }
      // The launch of this instantiation, and its opt-in window beside it: a
      // geometry above the 48 KiB default request fails at the launch and not
      // at the attribute, so the two are stated together.
      SetStagedSharedMemory<kWarpgroups, kQueryTiles, true, kStages, kWidth>();
      for (int window : {512, kSourceTiles}) {
        for (int copy_order : {0, kRowFast}) {
          const double flops = 2.0 * kAtomM * kWidth * kDim *
                               static_cast<double>(product_tiles) *
                               static_cast<double>(kWarpgroups) *
                               static_cast<double>(kQueryTiles) *
                               static_cast<double>(sm_count);
          const std::string shape =
              "keys=" + std::to_string(kWidth) +
              " warpgroups=" + std::to_string(kWarpgroups) +
              " query_tiles=" + std::to_string(kQueryTiles) +
              " window=" + std::to_string(window) +
              " copy_order=" + std::to_string(copy_order) +
              " blocks=" + std::to_string(sm_count) +
              " tiles=" + std::to_string(product_tiles);
          // The arithmetic one staged byte buys is the query rows of the block,
          // at any key width.
          const double per_byte = 64.0 * static_cast<double>(kWarpgroups) *
                                  static_cast<double>(kQueryTiles);
          const double rate = add_product_rate(
              make_row("attention-wgmma:staged", shape, "h1"), "staged:", shape,
              flops, per_byte, [&] {
                LaunchStaged<kWarpgroups, kQueryTiles, true, kStages, kWidth>(
                    sm_count, product_tiles, window, device_queries, device_k,
                    device_out, copy_order);
              });
          // The fastest window and order of the shape is the row a decision
          // reads, so the table keeps the best rate a shape states.
          shapes.Measure(shape_row, rate * kTflopsPerGflops);
        }
      }
    };
    auto time_geometry = [&](auto warpgroups, auto query_tiles) {
      time_shape(std::integral_constant<int, kAtomN>{}, warpgroups,
                 query_tiles);
    };
    auto time_narrow = [&](auto warpgroups, auto query_tiles) {
      time_shape(std::integral_constant<int, kNarrowKeyWidth>{}, warpgroups,
                 query_tiles);
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int query_tiles = 1; query_tiles <= kMaxQueryTiles; ++query_tiles) {
        VisitGeometry(warpgroups, query_tiles, time_geometry);
      }
    }
    VisitNarrowGeometry(kMaxWarpgroups, kNarrowQueryTiles, time_narrow);

    // The copies alone, at the same thread counts the product rows run at, so a
    // reader compares the cap of the path with the rate of the products. The
    // copy row stages one query tile, because a copy reads no query tile.
    auto time_copies = [&](auto warpgroups, auto) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      for (int copy_order : {0, kRowFast}) {
        for (int window : {512, kSourceTiles}) {
          for (int blocks : {1, sm_count}) {
            const std::string shape =
                "warpgroups=" + std::to_string(kWarpgroups) +
                " window=" + std::to_string(window) +
                " copy_order=" + std::to_string(copy_order) +
                " blocks=" + std::to_string(blocks) +
                " tiles=" + std::to_string(copy_tiles);
            const double bytes = static_cast<double>(kKeyBytes) *
                                 static_cast<double>(copy_tiles) *
                                 static_cast<double>(blocks);
            SetStagedSharedMemory<kWarpgroups, 1, false>();
            add_copy_rate(make_row("attention-wgmma:copies", shape, "h1"),
                          "copies:", shape, bytes, [&] {
                            LaunchStaged<kWarpgroups, 1, false>(
                                blocks, copy_tiles, window, device_queries,
                                device_k, device_out, copy_order);
                          });
          }
        }
      }
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      VisitGeometry(warpgroups, 1, time_copies);
    }

    // The copies of the narrow ring. The narrow key tile stages half the bytes
    // of a wide one at the same path, so a product row of the narrow shape has
    // the cap of its own measurement and not the cap of the wide tile's.
    {
      for (int window : {512, kSourceTiles}) {
        for (int copy_order : {0, kRowFast}) {
          const std::string shape =
              "keys=" + std::to_string(kNarrowKeyWidth) +
              " window=" + std::to_string(window) +
              " copy_order=" + std::to_string(copy_order) +
              " blocks=" + std::to_string(sm_count) +
              " tiles=" + std::to_string(copy_tiles);
          const double bytes = static_cast<double>(KeyBytes(kNarrowKeyWidth)) *
                               static_cast<double>(copy_tiles) *
                               static_cast<double>(sm_count);
          SetStagedSharedMemory<kMaxWarpgroups, 1, false, kStages,
                                kNarrowKeyWidth>();
          add_copy_rate(make_row("attention-wgmma:copies", shape, "h1"),
                        "copies:", shape, bytes, [&] {
                          LaunchStaged<kMaxWarpgroups, 1, false, kStages,
                                       kNarrowKeyWidth>(
                              sm_count, copy_tiles, window, device_queries,
                              device_k, device_out, copy_order);
                        });
        }
      }
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
    // One resident geometry: no copy path, `Warpgroups` warpgroups of one query
    // tile each, and one `FlashStep` a tile. The warpgroup count is the overlap
    // the step has, because two warpgroups on one multiprocessor interleave
    // their chains, so this is the row that separates the cost of the chain
    // from the cost of the copies. The untrimmed row and the trimmed row beside
    // it are the same geometry and the same build, so their difference is the
    // price of the 3 levers of `FlashStep` and not of the card.
    auto add_flash_resident = [&](auto warpgroups, auto mask_constant,
                                  auto softmax_constant, int blocks,
                                  auto trim_constant, auto guard_constant) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr bool kMask = decltype(mask_constant)::value;
      constexpr bool kSoftmax = decltype(softmax_constant)::value;
      constexpr bool kTrim = decltype(trim_constant)::value;
      constexpr bool kGuard = decltype(guard_constant)::value;
      const ShapeSpec& shape_row =
          FindShape(ShapeKind::kFlashResident, kWarpgroups, 1, kAtomN, 1);
      if (!admit(shape_row, ResidentFlashRegisters<kWarpgroups, kMask, kSoftmax,
                                                   kTrim, kGuard>())) {
        return;
      }
      SetFlashSharedMemory<kWarpgroups, kMask, kSoftmax, kTrim, kGuard>();
      const std::string shape =
          "blocks=" + std::to_string(blocks) +
          " tiles=" + std::to_string(flash_tiles) +
          " warpgroups=" + std::to_string(kWarpgroups) +
          " atoms=8+8 mask=" + (kMask ? "causal" : "dense") +
          " softmax=" + (kSoftmax ? "on" : "off") +
          (kTrim ? " trim=on" : " trim=off") +
          (kGuard ? " guard=on" : " guard=off");
      const double flops =
          2.0 * FlopsPerPass(kAtoms) * static_cast<double>(flash_tiles) *
          static_cast<double>(blocks) * static_cast<double>(kWarpgroups);
      const nanochat::dev::Timing timing = harness.Time([&] {
        LaunchResidentFlash<kWarpgroups, kMask, kSoftmax, kTrim, kGuard>(
            blocks, device_queries, device_k, device_value, device_out,
            /*device_row_sums=*/nullptr, flash_tiles, kFlashScale);
      });
      nanochat::BenchRow row =
          make_row("attention-flash:resident", shape, "flash", kWarpgroups);
      // The boundary tile issues a full 128-key tile for every query row and
      // needs the triangle of it, so the ratio is issued over necessary work.
      // Each warpgroup of the block holds its own boundary tile, so the ratio
      // is the one of the whole block of query rows.
      row.work_ratio = kMask ? 256.0 / (kWarpgroups * kAtomM + 1) : 1.0;
      const double rate = add_rate_row(row, timing, flops);
      const double share = sm_share(flops, timing.min_ms);
      report.SetExtra(
          std::string("flash_resident:warpgroups") +
              std::to_string(kWarpgroups) + ":blocks" + std::to_string(blocks) +
              (kMask ? ":causal" : ":dense") + (kSoftmax ? ":on" : ":off") +
              (kTrim ? ":trim" : ":plain") + (kGuard ? ":guard" : ":noguard") +
              ":sm_share",
          share);
      // The chain row is the one rate of the shape, because the softmax-off row
      // beside it prices the step without the softmax and not the step. A
      // trimmed row states its rate alone, so the shape table keeps the row the
      // recorded baselines hold.
      if (kSoftmax && !kTrim) {
        shapes.Measure(shape_row, rate * kTflopsPerGflops);
      }
      note(
          "attention-flash:resident %s min=%.3f ms median=%.3f ms "
          "tflops=%.1f sm_share=%.3f\n",
          shape.c_str(), timing.min_ms, timing.median_ms,
          rate * kTflopsPerGflops, share);
    };
    const std::integral_constant<int, 1> kOneWarpgroup{};
    const std::integral_constant<int, 2> kTwoWarpgroups{};
    const std::integral_constant<bool, false> kDense{};
    const std::integral_constant<bool, true> kCausal{};
    const std::integral_constant<bool, true> kChain{};
    const std::integral_constant<bool, false> kNoChain{};
    const std::integral_constant<bool, false> kPlain{};
    const std::integral_constant<bool, false> kUnguarded{};
    add_flash_resident(kOneWarpgroup, kDense, kChain, 1, kPlain, kUnguarded);
    add_flash_resident(kOneWarpgroup, kDense, kChain, sm_count, kPlain,
                       kUnguarded);
    add_flash_resident(kOneWarpgroup, kCausal, kChain, sm_count, kPlain,
                       kUnguarded);
    // The ceiling of the chain: the two products and the conversion, with no
    // softmax between them. The gap to the row above is what the chain costs.
    add_flash_resident(kOneWarpgroup, kDense, kNoChain, sm_count, kPlain,
                       kUnguarded);
    // The same three rows at the overlap two warpgroups give, so a reader reads
    // how much of the chain's cost the interleave hides before any copy path
    // holds a byte.
    add_flash_resident(kTwoWarpgroups, kDense, kChain, sm_count, kPlain,
                       kUnguarded);
    add_flash_resident(kTwoWarpgroups, kCausal, kChain, sm_count, kPlain,
                       kUnguarded);
    add_flash_resident(kTwoWarpgroups, kDense, kNoChain, sm_count, kPlain,
                       kUnguarded);
    // The same rows with the 3 levers of `FlashStep`: the exponential as 1
    // instruction, the score scale folded into it, and the rescale of the
    // output accumulator skipped when a row maximum did not move. The guard is
    // 1 row on its own, because its saving is data dependent and a timed row
    // repeats 1 key tile.
    const std::integral_constant<bool, true> kTrimmed{};
    const std::integral_constant<bool, true> kGuarded{};
    add_flash_resident(kTwoWarpgroups, kDense, kChain, sm_count, kTrimmed,
                       kUnguarded);
    add_flash_resident(kTwoWarpgroups, kDense, kChain, sm_count, kTrimmed,
                       kGuarded);
    add_flash_resident(kTwoWarpgroups, kCausal, kChain, sm_count, kTrimmed,
                       kGuarded);
    add_flash_resident(kOneWarpgroup, kDense, kChain, sm_count, kTrimmed,
                       kGuarded);
    // The trimmed chain against the untrimmed ceiling, so a reader separates
    // what the scalar pipes cost from what they must cost.
    add_flash_resident(kTwoWarpgroups, kDense, kNoChain, sm_count, kTrimmed,
                       kGuarded);

    // The copies alone, so a reader divides a staged rate by the cap of the
    // path that carries it. The two ring contents are two rows, because the
    // copy engine fills the key ring and the plain transposed staging fills the
    // value ring.
    const int flash_copy_tiles = 8192;
    auto add_copy_row = [&](auto warpgroups, auto stages, auto stage_value,
                            auto vector_constant, auto plain_key,
                            auto skip_key) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kRing = decltype(stages)::value;
      constexpr bool kStageValue = decltype(stage_value)::value;
      constexpr bool kVector = decltype(vector_constant)::value;
      constexpr bool kPlainKey = decltype(plain_key)::value;
      constexpr bool kSkipKey = decltype(skip_key)::value;
      SetStagedFlashSharedMemory<kWarpgroups, kRing, false, kStageValue,
                                 /*kMask=*/false, /*kSoftmax=*/true, kVector,
                                 kPlainKey, kSkipKey>();
      const std::string shape =
          "warpgroups=" + std::to_string(kWarpgroups) +
          " stages=" + std::to_string(kRing) + " content=" +
          (kSkipKey ? "value" : (kStageValue ? "key+value" : "key")) +
          " value_staging=" +
          (kStageValue ? (kVector ? "vector" : "strided") : "none") +
          " key_staging=" + (kPlainKey ? "plain" : "engine") +
          " window=" + std::to_string(kSourceTiles) +
          " blocks=" + std::to_string(sm_count) +
          " tiles=" + std::to_string(flash_copy_tiles);
      const double bytes =
          static_cast<double>((kSkipKey ? 0 : kKeyBytes) +
                              (kStageValue ? kValueBytes : 0)) *
          static_cast<double>(flash_copy_tiles) * static_cast<double>(sm_count);
      const double bandwidth = add_copy_rate(
          make_row("attention-flash:copies", shape, "flash"),
          "flash_copies:", shape, bytes, [&] {
            LaunchStagedFlash<kWarpgroups, kRing, false, kStageValue,
                              /*kMask=*/false, /*kSoftmax=*/true, kVector,
                              kPlainKey, kSkipKey>(
                sm_count, flash_copy_tiles, kSourceTiles, device_queries,
                device_k, device_value, device_out,
                /*device_row_sums=*/nullptr);
          });
      // One block of the ring reads these bytes a second, which is the figure a
      // ring of a card is sized from. The console line states the card rate and
      // the document carries this one beside it.
      report.SetExtra("flash_copies:" + shape + ":bytes_per_second_per_block",
                      bandwidth / static_cast<double>(sm_count));
    };
    auto copy_visitor = [&](auto warpgroups, auto stages) {
      constexpr bool kNo = false;
      constexpr bool kYes = true;
      // The key ring alone, through the copy engine and through ordinary
      // stores, so the two paths compare on their own bytes.
      add_copy_row(warpgroups, stages,
                   std::integral_constant<bool, kRingKeyOnly>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{});
      add_copy_row(warpgroups, stages,
                   std::integral_constant<bool, kRingKeyOnly>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kYes>{},
                   std::integral_constant<bool, kNo>{});
      // The value ring alone, which separates its pass from the key pass.
      add_copy_row(warpgroups, stages,
                   std::integral_constant<bool, kRingKeyValue>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kYes>{});
      // Both rings, with each reading of the transpose.
      add_copy_row(warpgroups, stages,
                   std::integral_constant<bool, kRingKeyValue>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{});
      add_copy_row(warpgroups, stages,
                   std::integral_constant<bool, kRingKeyValue>{},
                   std::integral_constant<bool, kYes>{},
                   std::integral_constant<bool, kNo>{},
                   std::integral_constant<bool, kNo>{});
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int stages = 2; stages <= 3; ++stages) {
        VisitFlashGeometry(warpgroups, stages, copy_visitor);
      }
    }

    // The resident step with the rings in the way. The arithmetic one staged
    // byte buys is `64 * warpgroups`, so the rate follows the geometry.
    const int flash_product_tiles = 4096;
    auto add_staged_row = [&](auto warpgroups, auto stages) {
      constexpr int kWarpgroups = decltype(warpgroups)::value;
      constexpr int kRing = decltype(stages)::value;
      const ShapeSpec& shape_row =
          FindShape(ShapeKind::kFlash, kWarpgroups, 1, kAtomN, kRing);
      // The dense instantiation states the register count of the shape. The
      // causal one reads two registers above it in the recorded build, which is
      // a difference no verdict of the table turns on.
      if (!admit(shape_row,
                 StagedFlashRegisters<kWarpgroups, kRing, false>())) {
        return;
      }
      auto add_one = [&](auto mask_constant) {
        constexpr bool kMask = decltype(mask_constant)::value;
        SetStagedFlashSharedMemory<kWarpgroups, kRing, true, kRingKeyValue,
                                   kMask>();
        const double flops = 2.0 * FlopsPerPass(kAtoms) *
                             static_cast<double>(flash_product_tiles) *
                             static_cast<double>(kWarpgroups) *
                             static_cast<double>(sm_count);
        const std::string shape =
            "warpgroups=" + std::to_string(kWarpgroups) +
            " stages=" + std::to_string(kRing) +
            " window=" + std::to_string(kSourceTiles) +
            " blocks=" + std::to_string(sm_count) +
            " tiles=" + std::to_string(flash_product_tiles) +
            " mask=" + (kMask ? "causal" : "dense");
        nanochat::BenchRow row =
            make_row("attention-flash:staged", shape, "flash");
        row.work_ratio = kMask ? 256.0 / (kWarpgroups * kAtomM + 1) : 1.0;
        // Both rings are staged, so one staged byte buys the query rows of the
        // step: one query tile a warpgroup.
        const double rate = add_product_rate(
            row, "flash_staged:", shape, flops,
            64.0 * static_cast<double>(kWarpgroups), [&] {
              LaunchStagedFlash<kWarpgroups, kRing, true, kRingKeyValue, kMask>(
                  sm_count, flash_product_tiles, kSourceTiles, device_queries,
                  device_k, device_value, device_out,
                  /*device_row_sums=*/nullptr);
            });
        // The chain row is the rate the shape states: the softmax-off row
        // beside it prices the products alone and not the step.
        shapes.Measure(shape_row, rate * kTflopsPerGflops);
      };
      add_one(std::integral_constant<bool, false>{});
      add_one(std::integral_constant<bool, true>{});
    };
    for (int warpgroups = 1; warpgroups <= kMaxWarpgroups; ++warpgroups) {
      for (int stages = 2; stages <= 3; ++stages) {
        VisitFlashGeometry(warpgroups, stages, add_staged_row);
      }
    }

    // The ceiling of the chain behind the rings: the same ring and the same two
    // products, with no softmax between them.
    {
      constexpr int kWarpgroups = 2;
      constexpr int kRing = 2;
      const ShapeSpec& shape_row =
          FindShape(ShapeKind::kFlash, kWarpgroups, 1, kAtomN, kRing);
      const int registers =
          StagedFlashRegisters<kWarpgroups, kRing, false, false>();
      if (admit(shape_row, registers)) {
        SetStagedFlashSharedMemory<kWarpgroups, kRing, true, kRingKeyValue,
                                   false,
                                   /*kSoftmax=*/false>();
        const nanochat::dev::Timing timing = harness.Time([&] {
          LaunchStagedFlash<kWarpgroups, kRing, true, kRingKeyValue, false,
                            /*kSoftmax=*/false>(
              sm_count, flash_product_tiles, kSourceTiles, device_queries,
              device_k, device_value, device_out, /*device_row_sums=*/nullptr);
        });
        const double flops = 2.0 * FlopsPerPass(kAtoms) *
                             static_cast<double>(flash_product_tiles) *
                             static_cast<double>(kWarpgroups) *
                             static_cast<double>(sm_count);
        const std::string shape =
            "warpgroups=" + std::to_string(kWarpgroups) +
            " stages=" + std::to_string(kRing) +
            " window=" + std::to_string(kSourceTiles) +
            " blocks=" + std::to_string(sm_count) +
            " tiles=" + std::to_string(flash_product_tiles) +
            " mask=dense softmax=off";
        const double rate = add_rate_row(
            make_row("attention-flash:staged", shape, "flash"), timing, flops);
        note("attention-flash:staged %s min=%.3f ms tflops=%.1f\n",
             shape.c_str(), timing.min_ms, rate * kTflopsPerGflops);
      }
    }
  }

  // The value ring of section 5's second direction: the copy engine fills the
  // tile a key ring fills, and one shared pass transposes it into the tile the
  // value product reads. So the transposition leaves the global path, and the
  // rows state what that is worth: the copies alone, the copies with the
  // transposition, and the step that reads them.
  if (options.flash()) {
    constexpr int kWarpgroups = kMaxWarpgroups;
    constexpr int kRing = 2;
    const ShapeSpec& shape_row =
        FindShape(ShapeKind::kFlash, kWarpgroups, 1, kAtomN, kRing,
                  /*shared_value=*/true);
    const int registers =
        SharedValueRegisters<kWarpgroups, kRing, /*kMask=*/false>();
    if (admit(shape_row, registers)) {
      const int copy_tiles = 8192;
      auto add_shared_copies = [&](auto transpose_constant) {
        constexpr bool kTranspose = decltype(transpose_constant)::value;
        SetStagedFlashSharedMemory<kWarpgroups, kRing, /*kProducts=*/false,
                                   kRingKeyValue, /*kMask=*/false,
                                   /*kSoftmax=*/true, /*kVectorValue=*/false,
                                   /*kPlainKey=*/false, /*kSkipKey=*/false,
                                   /*kSharedValue=*/true, kTranspose>();
        const std::string shape =
            "warpgroups=" + std::to_string(kWarpgroups) +
            " stages=" + std::to_string(kRing) +
            " window=" + std::to_string(kSourceTiles) +
            " blocks=" + std::to_string(sm_count) +
            " tiles=" + std::to_string(copy_tiles) +
            " content=key+value value_staging=shared key_staging=engine"
            " transpose=" +
            (kTranspose ? std::string("on") : std::string("off"));
        const double bytes = static_cast<double>(kKeyBytes + kValueBytes) *
                             static_cast<double>(copy_tiles) *
                             static_cast<double>(sm_count);
        const double bandwidth = add_copy_rate(
            make_row("attention-flash:copies", shape, "flash"),
            "flash_copies:", shape, bytes, [&] {
              LaunchStagedFlash<kWarpgroups, kRing, /*kProducts=*/false,
                                kRingKeyValue, /*kMask=*/false,
                                /*kSoftmax=*/true, /*kVectorValue=*/false,
                                /*kPlainKey=*/false, /*kSkipKey=*/false,
                                /*kSharedValue=*/true, kTranspose>(
                  sm_count, copy_tiles, kSourceTiles, device_queries, device_k,
                  device_value, device_out, /*device_row_sums=*/nullptr);
            });
        report.SetExtra("flash_copies:" + shape + ":bytes_per_second_per_block",
                        bandwidth / static_cast<double>(sm_count));
      };
      add_shared_copies(std::integral_constant<bool, false>{});
      add_shared_copies(std::integral_constant<bool, true>{});

      const int product_tiles = 4096;
      auto add_shared_step = [&](auto mask_constant, auto softmax_constant) {
        constexpr bool kMask = decltype(mask_constant)::value;
        constexpr bool kSoftmax = decltype(softmax_constant)::value;
        SetStagedFlashSharedMemory<kWarpgroups, kRing, /*kProducts=*/true,
                                   kRingKeyValue, kMask, kSoftmax,
                                   /*kVectorValue=*/false,
                                   /*kPlainKey=*/false, /*kSkipKey=*/false,
                                   /*kSharedValue=*/true,
                                   /*kTranspose=*/true>();
        const double flops =
            2.0 * FlopsPerPass(kAtoms) * static_cast<double>(product_tiles) *
            static_cast<double>(kWarpgroups) * static_cast<double>(sm_count);
        const std::string shape = "warpgroups=" + std::to_string(kWarpgroups) +
                                  " stages=" + std::to_string(kRing) +
                                  " window=" + std::to_string(kSourceTiles) +
                                  " blocks=" + std::to_string(sm_count) +
                                  " tiles=" + std::to_string(product_tiles) +
                                  " mask=" + (kMask ? "causal" : "dense") +
                                  " softmax=" + (kSoftmax ? "on" : "off") +
                                  " value_staging=shared";
        nanochat::BenchRow row =
            make_row("attention-flash:staged", shape, "flash");
        row.work_ratio = kMask ? 256.0 / (kWarpgroups * kAtomM + 1) : 1.0;
        const double rate = add_product_rate(
            row, "flash_staged:", shape, flops,
            64.0 * static_cast<double>(kWarpgroups), [&] {
              LaunchStagedFlash<kWarpgroups, kRing, /*kProducts=*/true,
                                kRingKeyValue, kMask, kSoftmax,
                                /*kVectorValue=*/false,
                                /*kPlainKey=*/false, /*kSkipKey=*/false,
                                /*kSharedValue=*/true, /*kTranspose=*/true>(
                  sm_count, product_tiles, kSourceTiles, device_queries,
                  device_k, device_value, device_out,
                  /*device_row_sums=*/nullptr);
            });
        if (kSoftmax) shapes.Measure(shape_row, rate * kTflopsPerGflops);
      };
      add_shared_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{});
      add_shared_step(std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{});
      add_shared_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, false>{});
    }
  }

  // The engine-fed tile: both rings moved by the copy engine in the swizzled
  // panel form, the value ring transposed in shared memory, and a producer
  // warpgroup that issues the copies and the transposition. The rows state what
  // the engine and the role split are worth together.
  if (options.flash()) {
    constexpr int kWarpgroups = kMaxWarpgroups;
    constexpr int kRing = 2;
    const ShapeSpec& shape_row =
        FindShape(ShapeKind::kFlash, kWarpgroups, 1, kAtomN, kRing,
                  /*shared_value=*/true);
    const int registers =
        EngineFlashRegisters<kWarpgroups, kRing, /*kMask=*/false>();
    CUtensorMap key_low{};
    CUtensorMap key_high{};
    CUtensorMap value_low{};
    CUtensorMap value_high{};
    const int rows = kAtomN * kSourceTiles;
    const bool built =
        MakeSwizzledPanelMap(&key_low, device_k, rows, kAtomN, /*panel=*/0) &&
        MakeSwizzledPanelMap(&key_high, device_k, rows, kAtomN, /*panel=*/1) &&
        MakeSwizzledPanelMap(&value_low, device_value, rows, kAtomN,
                             /*panel=*/0) &&
        MakeSwizzledPanelMap(&value_high, device_value, rows, kAtomN,
                             /*panel=*/1);
    if (built && admit(shape_row, registers)) {
      const int engine_tiles = 4096;
      auto add_engine_step = [&](auto mask_constant, auto softmax_constant,
                                 auto warp_constant, auto trim_constant,
                                 auto guard_constant, auto rotate_constant) {
        constexpr bool kMask = decltype(mask_constant)::value;
        constexpr bool kSoftmax = decltype(softmax_constant)::value;
        constexpr bool kWarp = decltype(warp_constant)::value;
        constexpr bool kTrim = decltype(trim_constant)::value;
        constexpr bool kGuard = decltype(guard_constant)::value;
        constexpr bool kRotate = decltype(rotate_constant)::value;
        SetEngineFlashSharedMemory<kWarpgroups, kRing, kMask, kSoftmax, kWarp,
                                   /*kConsumerTranspose=*/false, kTrim, kGuard,
                                   kRotate>();
        const double flops =
            2.0 * FlopsPerPass(kAtoms) * static_cast<double>(engine_tiles) *
            static_cast<double>(kWarpgroups) * static_cast<double>(sm_count);
        const std::string shape =
            "consumers=" + std::to_string(kWarpgroups) +
            " stages=" + std::to_string(kRing) +
            " blocks=" + std::to_string(sm_count) +
            " tiles=" + std::to_string(engine_tiles) +
            " mask=" + (kMask ? "causal" : "dense") +
            " softmax=" + (kSoftmax ? "on" : "off") +
            " staging=engine producers=1warpgroup" +
            " transpose=" + (kWarp ? "warp" : "thread") +
            (kTrim ? " trim=on" : " trim=off") +
            (kGuard ? " guard=on" : " guard=off") +
            (kRotate ? " store=rotated" : " store=straight");
        nanochat::BenchRow row =
            make_row("attention-flash:staged", shape, "flash");
        row.work_ratio = kMask ? 256.0 / (kWarpgroups * kAtomM + 1) : 1.0;
        const double rate = add_product_rate(
            row, "flash_staged:", shape, flops,
            128.0 * static_cast<double>(kWarpgroups) / 2.0, [&] {
              LaunchEngineFlash<kWarpgroups, kRing, kMask, kSoftmax, kWarp,
                                /*kConsumerTranspose=*/false, kTrim, kGuard,
                                kRotate>(sm_count, engine_tiles, device_queries,
                                         key_low, key_high, value_low,
                                         value_high, device_out);
            });
        if (kSoftmax && !kTrim) {
          shapes.Measure(shape_row, rate * kTflopsPerGflops);
        }
      };
      auto add_both = [&](auto mask_constant, auto softmax_constant) {
        add_engine_step(mask_constant, softmax_constant,
                        std::integral_constant<bool, false>{},
                        std::integral_constant<bool, false>{},
                        std::integral_constant<bool, false>{},
                        std::integral_constant<bool, false>{});
        add_engine_step(mask_constant, softmax_constant,
                        std::integral_constant<bool, true>{},
                        std::integral_constant<bool, false>{},
                        std::integral_constant<bool, false>{},
                        std::integral_constant<bool, false>{});
      };
      auto add_consumer = [&](auto mask_constant) {
        constexpr bool kMask = decltype(mask_constant)::value;
        SetEngineFlashSharedMemory<kWarpgroups, kRing, kMask,
                                   /*kSoftmax=*/true, /*kWarpTranspose=*/false,
                                   /*kConsumerTranspose=*/true>();
        const double flops =
            2.0 * FlopsPerPass(kAtoms) * static_cast<double>(engine_tiles) *
            static_cast<double>(kWarpgroups) * static_cast<double>(sm_count);
        const std::string shape =
            "consumers=" + std::to_string(kWarpgroups) +
            " stages=" + std::to_string(kRing) +
            " blocks=" + std::to_string(sm_count) +
            " tiles=" + std::to_string(engine_tiles) +
            " mask=" + (kMask ? "causal" : "dense") +
            " softmax=on staging=engine producers=1warpgroup"
            " transpose=consumer";
        nanochat::BenchRow row =
            make_row("attention-flash:staged", shape, "flash");
        row.work_ratio = kMask ? 256.0 / (kWarpgroups * kAtomM + 1) : 1.0;
        add_product_rate(row, "flash_staged:", shape, flops,
                         128.0 * static_cast<double>(kWarpgroups) / 2.0, [&] {
                           LaunchEngineFlash<kWarpgroups, kRing, kMask,
                                             /*kSoftmax=*/true,
                                             /*kWarpTranspose=*/false,
                                             /*kConsumerTranspose=*/true>(
                               sm_count, engine_tiles, device_queries, key_low,
                               key_high, value_low, value_high, device_out);
                         });
      };
      add_consumer(std::integral_constant<bool, false>{});
      add_consumer(std::integral_constant<bool, true>{});
      add_both(std::integral_constant<bool, false>{},
               std::integral_constant<bool, true>{});
      add_both(std::integral_constant<bool, true>{},
               std::integral_constant<bool, true>{});
      add_both(std::integral_constant<bool, false>{},
               std::integral_constant<bool, false>{});
      // The engine-fed tile with the 3 levers of `FlashStep`. The two
      // untrimmed rows above are its control, and the row with the softmax off
      // is the ceiling it is measured against.
      add_engine_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{});
      add_engine_step(std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{});
      add_engine_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{});
      // The store order of the transposition, priced on the same tile. A lane's
      // 8 destination pieces are 128 bytes, so every lane starts on a bank
      // boundary and 32 lanes put their 16 bytes on 4 banks; the rotated store
      // moves the piece a lane writes by the lane's own index.
      add_engine_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{});
      add_engine_step(std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{});
      add_engine_step(std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, false>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{},
                      std::integral_constant<bool, true>{});
    }
  }

  // The warp-specialized rows of decision H3: the same tile and the same ring
  // as the staged flash row above, with the copies on a producer that never
  // computes. The two producer widths are one warp, which is the first phase of
  // the decision, and one warpgroup, which `setmaxnreg` spans. A copy row
  // states the bytes a second the producer sustains on its own, so a reader
  // compares the step with the path under it.
  if (options.h3()) {
    const int special_tiles = 4096;
    const int special_copy_tiles = 8192;
    auto add_special = [&](auto producer) {
      constexpr int kProducer = decltype(producer)::value;
      auto add_consumers = [&](auto consumers) {
        constexpr int kConsumers = decltype(consumers)::value;
        auto add_ring = [&](auto ring) {
          constexpr int kRing = decltype(ring)::value;
          const ShapeSpec& shape_row =
              FindShape(ShapeKind::kSpecial, kConsumers, 1, kAtomN, kRing);
          const int registers =
              SpecializedRegisters<kProducer, kConsumers, kRing,
                                   /*kMask=*/false>();
          if (!admit(shape_row, registers)) return;
          // The copies alone, so the producer's own rate is one number.
          SetSpecializedSharedMemory<kProducer, kConsumers, kRing,
                                     /*kProducts=*/false, kRingKeyValue,
                                     /*kMask=*/false>();
          const std::string copy_shape =
              "producer=" + std::to_string(kProducer) +
              " consumers=" + std::to_string(kConsumers) +
              " stages=" + std::to_string(kRing) +
              " window=" + std::to_string(kSourceTiles) +
              " blocks=" + std::to_string(sm_count) +
              " tiles=" + std::to_string(special_copy_tiles);
          const double bytes = static_cast<double>(kKeyBytes + kValueBytes) *
                               static_cast<double>(special_copy_tiles) *
                               static_cast<double>(sm_count);
          const double bandwidth = add_copy_rate(
              make_row("attention-special:copies", copy_shape, "h3"),
              "special_copies:", copy_shape, bytes, [&] {
                LaunchSpecialized<kProducer, kConsumers, kRing,
                                  /*kProducts=*/false, kRingKeyValue,
                                  /*kMask=*/false>(
                    sm_count, special_copy_tiles, kSourceTiles,
                    device_special_queries, device_special_keys,
                    device_special_values, device_out,
                    /*device_row_sums=*/nullptr);
              });
          report.SetExtra(
              "special_copies:" + copy_shape + ":bytes_per_second_per_block",
              bandwidth / static_cast<double>(sm_count));
          auto add_step = [&](auto mask_constant, auto softmax_constant) {
            constexpr bool kMask = decltype(mask_constant)::value;
            constexpr bool kSoftmax = decltype(softmax_constant)::value;
            SetSpecializedSharedMemory<kProducer, kConsumers, kRing,
                                       /*kProducts=*/true, kRingKeyValue, kMask,
                                       kSoftmax>();
            const double flops = 2.0 * FlopsPerPass(kAtoms) *
                                 static_cast<double>(special_tiles) *
                                 static_cast<double>(kConsumers) *
                                 static_cast<double>(sm_count);
            const std::string shape =
                "producer=" + std::to_string(kProducer) +
                " consumers=" + std::to_string(kConsumers) +
                " stages=" + std::to_string(kRing) +
                " window=" + std::to_string(kSourceTiles) +
                " blocks=" + std::to_string(sm_count) +
                " tiles=" + std::to_string(special_tiles) +
                " mask=" + (kMask ? "causal" : "dense") +
                " softmax=" + (kSoftmax ? "on" : "off");
            nanochat::BenchRow row =
                make_row("attention-special:step", shape, "h3", kConsumers);
            row.work_ratio = kMask ? 256.0 / (kConsumers * kAtomM + 1) : 1.0;
            const double rate = add_product_rate(
                row, "special_step:", shape, flops,
                64.0 * static_cast<double>(kConsumers), [&] {
                  LaunchSpecialized<kProducer, kConsumers, kRing,
                                    /*kProducts=*/true, kRingKeyValue, kMask,
                                    kSoftmax>(
                      sm_count, special_tiles, kSourceTiles,
                      device_special_queries, device_special_keys,
                      device_special_values, device_out,
                      /*device_row_sums=*/nullptr);
                });
            // The chain row is the rate the shape states: the softmax-off row
            // beside it prices the products alone and not the step.
            if (kSoftmax) shapes.Measure(shape_row, rate * kTflopsPerGflops);
          };
          const std::integral_constant<bool, false> kDense{};
          const std::integral_constant<bool, true> kCausal{};
          const std::integral_constant<bool, true> kChain{};
          const std::integral_constant<bool, false> kNoChain{};
          add_step(kDense, kChain);
          add_step(kCausal, kChain);
          add_step(kDense, kNoChain);
        };
        VisitSpecializedRing(kSpecialStages, add_ring);
        VisitSpecializedRing(kSpecialDeepStages, add_ring);
      };
      VisitSpecializedConsumers(kSpecialOneConsumer, add_consumers);
      VisitSpecializedConsumers(kSpecialTwoConsumers, add_consumers);
    };
    VisitSpecializedProducer(kSpecialWarpProducer, add_special);
    VisitSpecializedProducer(kSpecialWarpgroupProducer, add_special);
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

  // The shape space the run stated, beside the rows it priced. The document
  // carries one extra a field, so a reader compares two shapes without a second
  // run, and the human table prints the same rows in the order the run met
  // them.
  shapes.Write(&report);
  if (!options.bench.json) {
    shapes.Print(stdout, report.peak_flops() / 1e12);
  }

  cudaFree(device_q);
  cudaFree(device_queries);
  cudaFree(device_k);
  cudaFree(device_value);
  cudaFree(device_special_queries);
  cudaFree(device_special_keys);
  cudaFree(device_special_values);
  cudaFree(device_out);
  cudaFree(device_row_sums);
  report.Print(options.bench);
  return ok ? 0 : 1;
}
