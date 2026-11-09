// The candidate registry's lookups (docs/kernels.md section 11.4).
//
// The table itself is `kGemmCandidates` in the header, because a candidate's
// translation unit reads its own row as a compile-time constant. What is left
// here is the lookups a report, a command line, and a test use. No vendor
// header and no CUDA type: the table is data, so a host test reads it.

#include "backends/cuda/gemm/cutlass_candidates.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

namespace nanochat {
namespace kernels {

namespace {

// The candidate an empty answer states, so a caller that reads a name past the
// table reads an empty string and not a stale candidate.
const GemmCandidateSpec kEmpty{"", 0, 0, 0, 0, 0, 0, "", "", ""};

// One tile name for each candidate, built once from the candidate's own fields.
const std::array<std::string, kGemmCandidateCount>& TileNames() {
  static const auto kNames = [] {
    std::array<std::string, kGemmCandidateCount> names;
    for (int index = 0; index < kGemmCandidateCount; ++index) {
      const GemmCandidateSpec& spec = kGemmCandidates[index];
      names[static_cast<std::size_t>(index)] =
          std::to_string(spec.tile_m) + "x" + std::to_string(spec.tile_n) +
          "x" + std::to_string(spec.tile_k);
    }
    return names;
  }();
  return kNames;
}

}  // namespace

const GemmCandidateSpec& GemmCandidateAt(int index) {
  if (index < 0 || index >= kGemmCandidateCount) return kEmpty;
  return kGemmCandidates[index];
}

int GemmCandidateIndexFromName(const char* name) {
  if (name == nullptr) return -1;
  for (int index = 0; index < kGemmCandidateCount; ++index) {
    if (std::strcmp(name, kGemmCandidates[index].name) == 0) return index;
  }
  return -1;
}

int GemmCandidateFromEnv() {
  return GemmCandidateIndexFromName(std::getenv("NANOCHAT_GEMM_CANDIDATE"));
}

int GemmCandidateInstantiations() {
  // Four operand orientations times two store types.
  return 8;
}

const char* GemmCandidateTileName(int index) {
  if (index < 0 || index >= kGemmCandidateCount) return "";
  return TileNames()[static_cast<std::size_t>(index)].c_str();
}

std::size_t GemmCandidateSharedBytes(int index) {
  const GemmCandidateSpec& spec = GemmCandidateAt(index);
  if (spec.stages <= 0) return 0;
  // The two staged operands of one slice, at the seam's two-byte element, times
  // the pipeline depth. The epilogue's own share of the window is not in it.
  const std::size_t one_stage =
      (static_cast<std::size_t>(spec.tile_m) * spec.tile_k +
       static_cast<std::size_t>(spec.tile_n) * spec.tile_k) *
      2;
  return one_stage * static_cast<std::size_t>(spec.stages);
}

}  // namespace kernels
}  // namespace nanochat
