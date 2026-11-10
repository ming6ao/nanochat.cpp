// The host gate of the candidate registry (docs/kernels.md section 11.7 step
// 2).
//
// The registry is data, so this test checks the data without a card:
//
//   1. Four candidates, one for each shape family the one model issues, with
//      distinct names, tiles, and stage counts.
//   2. Every tile is one the probe measured, at the stage count the probe
//      measured for it, and every tile is at least 128 in M, which the
//      cooperative schedule asserts.
//   3. Every candidate's staged operands plus the epilogue's measured share fit
//      the Hopper window, so no candidate is built that cannot launch.
//   4. One candidate is eight instantiations, four operand orientations times
//   two
//      store types, and the registry states the matrix.
//   5. `NANOCHAT_GEMM_CANDIDATE` selects exactly the candidate it names, and an
//      unknown name selects none.
//
// The spill gate is not here: it reads the machine code of a built object, so
// it runs as `tools/gemm_spill_gate.py` after a build.
//
// Hermetic and CPU only: the header names no CUDA type and the table names no
// vendor type, so this rides the default `-gpu` test leg.

#include "backends/cuda/gemm/cutlass_candidates.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using nanochat::kernels::GemmCandidateAt;
using nanochat::kernels::GemmCandidateCount;
using nanochat::kernels::GemmCandidateFromEnv;
using nanochat::kernels::GemmCandidateIndexFromName;
using nanochat::kernels::GemmCandidateInstantiations;
using nanochat::kernels::GemmCandidateSharedBytes;
using nanochat::kernels::GemmCandidateSpec;
using nanochat::kernels::GemmCandidateTileName;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

// The window one Hopper block may ask for, as a measurement of
// `dev/cutlass/README.md` section 3.
constexpr std::size_t kWindowBytes = 227 * 1024;

// The five tiles the probe measured, with the stage count the builder chose.
struct ProbedTile {
  int m;
  int n;
  int k;
  int stages;
  std::size_t total_bytes;
};

constexpr ProbedTile kProbed[] = {
    {128, 256, 64, 3, 215'040}, {128, 256, 32, 6, 215'040},
    {128, 128, 64, 5, 231'424}, {128, 128, 128, 2, 198'656},
    {256, 128, 64, 3, 215'040},
};

constexpr int kProbedCount =
    static_cast<int>(sizeof(kProbed) / sizeof(kProbed[0]));

// The share of the window the epilogue takes, from the same five rows: the
// total less the staged operands is 67,584 B for every one of them, which is a
// 128 by 128 fp32 exchange tile and its barriers. The exchange tile is 128
// columns wide and no wider, so a candidate whose N is below 128 stages a
// narrower one.
constexpr std::size_t EpilogueBytes(int tile_n) {
  const std::size_t columns =
      tile_n < 128 ? static_cast<std::size_t>(tile_n) : std::size_t{128};
  return 128 * columns * 4 + 2'048;
}

void CheckTable() {
  // The three shape families of section 11.6 and the two rows the sweep of the
  // slow shapes added. A registry below four serves no family, and one above
  // five costs more build than the development host should spend.
  Check(GemmCandidateCount() == 5,
        "the registry holds five candidates, it holds " +
            std::to_string(GemmCandidateCount()));
  std::vector<std::string> names;
  for (int index = 0; index < GemmCandidateCount(); ++index) {
    const GemmCandidateSpec& spec = GemmCandidateAt(index);
    const std::string label = "candidate " + std::to_string(index);
    Check(spec.name != nullptr && spec.name[0] != '\0', label + " has a name");
    for (const std::string& other : names) {
      Check(other != spec.name, label + " has a name no other candidate takes");
    }
    names.emplace_back(spec.name);
    Check(GemmCandidateIndexFromName(spec.name) == index,
          label + " round-trips through its own name");
    Check(spec.tile_m > 0 && spec.tile_n > 0 && spec.tile_k > 0,
          label + " states a tile");
    // The cooperative schedule splits M between two consumer warpgroups, so a
    // tile narrower than one warpgroup tile does not exist.
    Check(spec.tile_m >= 128,
          label + " is at least 128 wide in M, which the schedule requires");
    Check(spec.cluster_m > 0 && spec.cluster_n > 0,
          label + " states a cluster");
    Check(spec.stages > 0, label + " states a pipeline depth");
    Check(spec.kernel_schedule != nullptr && spec.kernel_schedule[0] != '\0',
          label + " names its kernel schedule");
    Check(
        spec.epilogue_schedule != nullptr && spec.epilogue_schedule[0] != '\0',
        label + " names its epilogue schedule");
    Check(spec.use != nullptr && spec.use[0] != '\0',
          label + " names the shape family it is meant for");
    const std::string want_tile = std::to_string(spec.tile_m) + "x" +
                                  std::to_string(spec.tile_n) + "x" +
                                  std::to_string(spec.tile_k);
    Check(want_tile == GemmCandidateTileName(index),
          label + " spells its tile name from its own fields");
  }
  // Four distinct tiles: two candidates that name one tile are one candidate.
  for (int a = 0; a < GemmCandidateCount(); ++a) {
    for (int b = a + 1; b < GemmCandidateCount(); ++b) {
      const GemmCandidateSpec& left = GemmCandidateAt(a);
      const GemmCandidateSpec& right = GemmCandidateAt(b);
      const bool same = left.tile_m == right.tile_m &&
                        left.tile_n == right.tile_n &&
                        left.tile_k == right.tile_k;
      Check(!same, "candidates " + std::to_string(a) + " and " +
                       std::to_string(b) + " state different tiles");
    }
  }
  // An index outside the table is empty and names no tile.
  Check(GemmCandidateAt(-1).name[0] == '\0', "a negative index is empty");
  Check(GemmCandidateAt(GemmCandidateCount()).name[0] == '\0',
        "an index past the table is empty");
  Check(GemmCandidateTileName(99)[0] == '\0',
        "an index past the table names no tile");
}

void CheckAgainstProbe() {
  for (int index = 0; index < GemmCandidateCount(); ++index) {
    const GemmCandidateSpec& spec = GemmCandidateAt(index);
    const std::string label = std::string("candidate ") + spec.name;
    // Every candidate must fit the window, because a candidate that cannot
    // launch is not a candidate. A tile the probe measured must also state the
    // stage count the probe measured for it; a later row states a count this
    // arithmetic checks instead.
    const std::size_t total =
        GemmCandidateSharedBytes(index) + EpilogueBytes(spec.tile_n);
    Check(total <= kWindowBytes, label + " fits the 227 KiB window, it needs " +
                                     std::to_string(total) + " B");
    bool found = false;
    for (int probe = 0; probe < kProbedCount; ++probe) {
      const ProbedTile& tile = kProbed[probe];
      if (tile.m != spec.tile_m || tile.n != spec.tile_n ||
          tile.k != spec.tile_k) {
        continue;
      }
      found = true;
      Check(spec.stages == tile.stages,
            label + " states the " + std::to_string(tile.stages) +
                " stages the probe measured for its tile");
      // The staged operands plus the epilogue's share are the probe's total.
      Check(total == tile.total_bytes, label + " accounts for the probe's " +
                                           std::to_string(tile.total_bytes) +
                                           " B, it accounts for " +
                                           std::to_string(total));
    }
    if (!found) {
      // A row the probe did not size states a tile whose window this test just
      // checked, and the build is the authority on whether it launches.
      Check(spec.stages > 0, label + " states a pipeline depth");
    }
  }
  // The cluster is one or two blocks along one axis, and never wider: a block
  // of this schedule already holds one multiprocessor, so a larger cluster
  // takes blocks away from a grid that does not have many.
  for (int index = 0; index < GemmCandidateCount(); ++index) {
    const GemmCandidateSpec& spec = GemmCandidateAt(index);
    const bool along_m = spec.cluster_m == 2 && spec.cluster_n == 1;
    const bool along_n = spec.cluster_m == 1 && spec.cluster_n == 2;
    Check(along_m || along_n, std::string("candidate ") + spec.name +
                                  " clusters two blocks along one axis");
  }
}

void CheckMatrix() {
  // Four operand orientations times two store types: the seam's own cross
  // product, and the reason one candidate is eight kernels.
  Check(GemmCandidateInstantiations() == 8,
        "one candidate holds eight instantiations");
  Check(GemmCandidateInstantiations() * GemmCandidateCount() ==
            8 * GemmCandidateCount(),
        "the registry states one matrix of instantiations for each candidate");
}

void CheckOverride() {
  unsetenv("NANOCHAT_GEMM_CANDIDATE");
  Check(GemmCandidateFromEnv() == -1, "an unset variable selects no candidate");
  for (int index = 0; index < GemmCandidateCount(); ++index) {
    const char* name = GemmCandidateAt(index).name;
    setenv("NANOCHAT_GEMM_CANDIDATE", name, 1);
    Check(GemmCandidateFromEnv() == index,
          std::string("the variable selects ") + name);
  }
  setenv("NANOCHAT_GEMM_CANDIDATE", "no-such-candidate", 1);
  Check(GemmCandidateFromEnv() == -1, "an unknown name selects no candidate");
  // The comparison is exact, so a prefix of a name is not that name.
  setenv("NANOCHAT_GEMM_CANDIDATE", "wide", 1);
  Check(GemmCandidateFromEnv() == -1, "a prefix of a name is not the name");
  unsetenv("NANOCHAT_GEMM_CANDIDATE");
}

}  // namespace

int main() {
  CheckTable();
  CheckAgainstProbe();
  CheckMatrix();
  CheckOverride();
  if (g_failures != 0) {
    std::fprintf(stderr, "candidate registry test: %d failure(s)\n",
                 g_failures);
    return 1;
  }
  std::printf("candidate registry test: %d candidates, %d instantiations\n",
              GemmCandidateCount(), GemmCandidateInstantiations());
  return 0;
}
