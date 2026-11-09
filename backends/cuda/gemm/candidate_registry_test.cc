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

// The window one Hopper block may ask for, and the share of it the epilogue
// takes. Both are measurements of `dev/cutlass/README.md` section 3: every tile
// there states a total, and the total less the staged operands is the same
// 67,584 B for all five.
constexpr std::size_t kWindowBytes = 227 * 1024;
constexpr std::size_t kEpilogueBytes = 67'584;

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

void CheckTable() {
  Check(GemmCandidateCount() == 4,
        "the registry holds four candidates, it holds " +
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
      Check(
          GemmCandidateSharedBytes(index) + kEpilogueBytes == tile.total_bytes,
          label + " accounts for the probe's " +
              std::to_string(tile.total_bytes) + " B, it accounts for " +
              std::to_string(GemmCandidateSharedBytes(index) + kEpilogueBytes));
      Check(tile.total_bytes <= kWindowBytes,
            label + " fits the 227 KiB window");
    }
    Check(found, label + " states a tile the probe measured");
  }
  // The cluster is the probe's 2 by 1 for every candidate, because a 2 by 2
  // cluster halves the block count of a tile that already holds one per
  // multiprocessor.
  for (int index = 0; index < GemmCandidateCount(); ++index) {
    Check(GemmCandidateAt(index).cluster_m == 2 &&
              GemmCandidateAt(index).cluster_n == 1,
          std::string("candidate ") + GemmCandidateAt(index).name +
              " keeps the measured 2 by 1 cluster");
  }
}

void CheckMatrix() {
  // Four operand orientations times two store types: the seam's own cross
  // product, and the reason one candidate is eight kernels.
  Check(GemmCandidateInstantiations() == 8,
        "one candidate holds eight instantiations");
  Check(GemmCandidateInstantiations() * GemmCandidateCount() == 32,
        "the registry holds 32 instantiations");
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
