// One candidate's kernel body: the registry row that `NANOCHAT_CANDIDATE_INDEX`
// names (docs/kernels.md section 11.4).
//
// **This is one file, not four.** The build compiles it once for each index, so
// a compiler process holds one candidate's eight instantiations and never two.
// `dev/cutlass/README.md` measures one instantiation at 19 s and 1.2 GiB, and
// four candidates in one process would exceed the development host's budget;
// that build cost is the only reason the candidates are separate translation
// units. The source is shared, so the tile, the cluster, and the pipeline depth
// come from the registry and not from a copy of it.
//
// **The tile is a template argument, so the row must be `constexpr`.** The
// registry of `cutlass_candidates.h` is, and this file reads its own row from
// it. A candidate therefore states neither a tile nor a stage count: it states
// a row.
//
// **The names are generated.** One source file compiled four times cannot
// export one name four times, so each set of entries takes the index as a
// suffix. `cutlass_launch.h` declares them from the same list, and
// `cutlass_bf16.cu` builds its call table from it, so no caller spells an
// index.
//
// See docs/kernels.md Part 3 section 11 and backends/cuda/gemm/README.md.

#ifndef NANOCHAT_CANDIDATE_INDEX
#error \
    "state -DNANOCHAT_CANDIDATE_INDEX=<n>; BUILD.bazel does it once for each candidate"
#endif

#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/cutlass_kernel.cuh"

namespace nanochat {
namespace kernels {

namespace {

// The registry row this translation unit instantiates.
constexpr int kIndex = NANOCHAT_CANDIDATE_INDEX;
static_assert(kIndex >= 0 && kIndex < kGemmCandidateCount,
              "the build names a candidate index of the registry");
constexpr GemmCandidateSpec kSpec = kGemmCandidates[kIndex];

// The row, as the template arguments the kernel body takes. A cluster of one
// model in the third mode is the batch, so the third extent is one.
using TileShape = cute::Shape<cute::Int<kSpec.tile_m>, cute::Int<kSpec.tile_n>,
                              cute::Int<kSpec.tile_k>>;
using ClusterShape = cute::Shape<cute::Int<kSpec.cluster_m>,
                                 cute::Int<kSpec.cluster_n>, cute::_1>;

}  // namespace

#define NANOCHAT_JOIN_INNER(left, right) left##right
#define NANOCHAT_JOIN(left, right) NANOCHAT_JOIN_INNER(left, right)
#define NANOCHAT_CANDIDATE_ENTRY(name) \
  NANOCHAT_JOIN(name, NANOCHAT_CANDIDATE_INDEX)

// The seam's element. A forward call and an input gradient store the compute
// type; a weight gradient stores fp32.
bool NANOCHAT_CANDIDATE_ENTRY(LaunchGemmCutlassCandidate)(
    const ResolvedGemm& gemm, const ComputeType* a, const ComputeType* b,
    ComputeType* c) {
  return cutlass_detail::LaunchOriented<TileShape, ClusterShape, kSpec.stages,
                                        Bf16>(gemm, a, b, c);
}

bool NANOCHAT_CANDIDATE_ENTRY(LaunchGemmCutlassCandidateToFp32)(
    const ResolvedGemm& gemm, const ComputeType* a, const ComputeType* b,
    float* c) {
  return cutlass_detail::LaunchOriented<TileShape, ClusterShape, kSpec.stages,
                                        float>(gemm, a, b, c);
}

int NANOCHAT_CANDIDATE_ENTRY(GemmCutlassCandidateOccupancy)() {
  return cutlass_detail::OccupancyOf<TileShape, ClusterShape, kSpec.stages>();
}

#undef NANOCHAT_CANDIDATE_ENTRY
#undef NANOCHAT_JOIN
#undef NANOCHAT_JOIN_INNER

}  // namespace kernels
}  // namespace nanochat
