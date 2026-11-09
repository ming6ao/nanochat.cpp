#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_CANDIDATES_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_CANDIDATES_H_

// The candidate registry of the Hopper GEMM (docs/kernels.md section 11.4).
//
// One candidate is one kernel body: a tile, a cluster, a stage count, a kernel
// schedule, and an epilogue schedule. The table below names four, and one
// source file, `cutlass_candidate.cu`, holds the body of every one of them. The
// build compiles that file once for each index, so one compiler process never
// holds two candidates' instantiations. `dev/cutlass/README.md` measures one
// instantiation at 19 s and 1.2 GiB, and a candidate is eight instantiations,
// so the split is what keeps the memory peak inside the development host's
// budget.
//
// **The table is the one home of the data.** It is `constexpr`, because a
// candidate's translation unit turns its own row into template arguments: the
// tile, the cluster, and the stage count. The tile name, the shared-memory
// estimate, and the sweep's report all derive from these fields, so a row
// states its tile once.
//
// The four are the four shape families the one model issues, and no single
// kernel is right for all of them:
//
//   * `wide-n` is the square tile of `dev/cutlass/README.md` section 3. Its 256
//     wide N suits the classifier, whose N is 32768.
//   * `balanced` halves N, which halves the accumulator.
//   * `deep-k` doubles the k slice and halves the stage count, for a long k
//     loop with few tiles.
//   * `tall-m` doubles M, for the weight gradient, whose N is short and whose M
//     is the token count.
//
// A candidate that spills its accumulator, or that fails the check of section
// 11.4 item 3, leaves the registry before a sweep. The gates are
// `tools/gemm_spill_gate.py` and `gemm_benchmark`'s own comparison.
//
// This header is precision free and device free: no CUDA type, no vendor type.
// `cutlass_candidates.cc` holds the lookups, so a host test reads the registry
// with no card and no CUTLASS header.

#include <cstddef>

namespace nanochat {
namespace kernels {

// One candidate.
struct GemmCandidateSpec {
  // The report name and the value `NANOCHAT_GEMM_CANDIDATE` takes.
  const char* name;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int cluster_m = 1;
  int cluster_n = 1;
  // The pipeline depth. `dev/cutlass/README.md` section 3 measures the count
  // that fits the window for each tile, and this is it.
  int stages = 0;
  // The kernel schedule and the epilogue schedule, as the CUTLASS names spell
  // them. A report prints them, so a reader compares two candidates without
  // reading the source.
  const char* kernel_schedule;
  const char* epilogue_schedule;
  // The shape family the candidate is meant for, in the words of section 11.6.
  const char* use;
};

// The candidates, in registry order. A row is one candidate, and its position
// in this table is its index.
inline constexpr GemmCandidateSpec kGemmCandidates[] = {
    {"wide-n", 128, 256, 64, 2, 1, 3, "KernelTmaWarpSpecializedCooperative",
     "TmaWarpSpecializedCooperative",
     "the classifier forward, whose N is 32768 wide"},
    {"balanced", 128, 128, 64, 2, 1, 5, "KernelTmaWarpSpecializedCooperative",
     "TmaWarpSpecializedCooperative",
     "the Muon square and the linear forward, where the accumulator of a 256 "
     "wide tile does not fit the register file"},
    {"deep-k", 128, 128, 128, 2, 1, 2, "KernelTmaWarpSpecializedCooperative",
     "TmaWarpSpecializedCooperative",
     "a long k loop with few tiles, where one slice is 128 deep"},
    {"tall-m", 256, 128, 64, 2, 1, 3, "KernelTmaWarpSpecializedCooperative",
     "TmaWarpSpecializedCooperative",
     "the weight gradient, whose M is the token count and whose N is short"},
};

// The count of candidates. Section 11.4 bounds it by the build cost, because
// one candidate is one compiler process with eight instantiations.
inline constexpr int kGemmCandidateCount =
    static_cast<int>(sizeof(kGemmCandidates) / sizeof(kGemmCandidates[0]));

// The indices, as the one list of them. One source file is compiled once for
// each index, and a linker needs a distinct name for each, so
// `cutlass_launch.h` declares one entry set for each index and
// `cutlass_bf16.cu` builds its call table from them. The assert below ties the
// list to the table, so a new candidate is one row above and one line here.
#define NANOCHAT_FOR_EACH_CANDIDATE(ENTRY) \
  ENTRY(0)                                 \
  ENTRY(1)                                 \
  ENTRY(2)                                 \
  ENTRY(3)

// The fp8 arm's tiles (docs/kernels.md section 10.4): the forward form's row
// and the deeper slice the two backward forms need.
//
// It is one row and not a registry entry. An fp8 axis over the four bf16 rows
// would be sixty-four instantiations, and `dev/cutlass/README.md` measures one
// at 19 s and 1.2 GiB, so the build budget bounds the count.
//
// The forward tile is the probe's `balanced` row. The first build ran `wide-n`
// and spilled, because the fp8 accumulation path keeps two fp32 accumulators.
//
// The backward tiles take the k slice of the probe's `deep-k` row, and they
// take it for a different reason. The two shipped backward forms are not
// K-major in both operands, so CUTLASS stages the awkward operand through its
// transposing collective, and that path refuses a k slice of 64: it needs more
// than two multiply instructions. A 128 deep slice gives four.
struct Fp8TileSpec {
  const char* name;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int cluster_m = 1;
  int cluster_n = 1;
  int stages = 0;
};

inline constexpr Fp8TileSpec kFp8Tile{"fp8-balanced", 128, 128, 64, 2, 1, 5};
inline constexpr Fp8TileSpec kFp8BackwardTile{
    "fp8-deep-k", 128, 128, 128, 2, 1, 2};

namespace cutlass_detail {

// The count the list states, which the assert below compares with the table.
constexpr int ListedCandidateCount() {
  int count = 0;
#define NANOCHAT_COUNT_ONE_CANDIDATE(index) count += 1;
  NANOCHAT_FOR_EACH_CANDIDATE(NANOCHAT_COUNT_ONE_CANDIDATE)
#undef NANOCHAT_COUNT_ONE_CANDIDATE
  return count;
}

}  // namespace cutlass_detail

static_assert(cutlass_detail::ListedCandidateCount() == kGemmCandidateCount,
              "NANOCHAT_FOR_EACH_CANDIDATE must name one index for each row of "
              "kGemmCandidates");

// One candidate, or an empty specification for an index outside the table.
const GemmCandidateSpec& GemmCandidateAt(int index);

// The count of candidates, in the form a loop reads. It is the constant above,
// so a caller pays no call and the two cannot disagree.
inline constexpr int GemmCandidateCount() { return kGemmCandidateCount; }

// The index `name` spells, or -1. The comparison is exact, because a report and
// a command line must name one candidate and not two.
int GemmCandidateIndexFromName(const char* name);

// The index `NANOCHAT_GEMM_CANDIDATE` names, or -1 when the variable is unset,
// empty, or names no candidate. It is the benchmark-only override of section
// 11.4 item 5, and it is what lets one binary sweep the registry.
int GemmCandidateFromEnv();

// The count of kernel instantiations one candidate holds: the four operand
// orientations of `GemmParams` times the two store types. A host test states
// the matrix is complete against this number.
int GemmCandidateInstantiations();

// The tile of a candidate as a report spells it: `128x256x64`, built from the
// candidate's own fields.
const char* GemmCandidateTileName(int index);

// The bytes of shared memory a candidate's staged operands need, from its own
// fields. The probe's window is 227 KiB, and the epilogue takes its own share
// of it on top of this, so the number is a necessary condition and not the
// total. `candidate_registry_test.cc` states that every row leaves room for the
// epilogue's share of the measurement in `dev/cutlass/README.md` section 3.
std::size_t GemmCandidateSharedBytes(int index);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_CANDIDATES_H_
