// The Hopper GEMM's entry point: the candidate the seam asks for, and the
// device gate in front of it (docs/kernels.md Part 3 section 11.4).
//
// **What this file is.** The dispatcher of a registry of four kernel bodies.
// One source file, `cutlass_candidate.cu`, holds every body and is compiled
// once for each index, because a candidate is eight instantiations and one
// compiler process holds one candidate at a time. What this file adds is the
// call table, the choice of candidate, the device test, and the two seam-facing
// entries.
//
// **The call table is the registry.** One row for each index, built from the
// same list that declares the entries, so a new candidate adds no branch here.
//
// **The choice is offline, and it has one override.** A call takes the
// candidate `NANOCHAT_GEMM_CANDIDATE` names, or the candidate the decision
// table of `cutlass_rules.h` selects for its shape, or candidate 0, which is
// the tile the adapter shipped before this registry existed. The variable is
// the benchmark-only override of section 11.4, as `NANOCHAT_GEMM_SCHEDULE` is
// for the hand tile. Nothing here times anything.
//
// **The seam decides whether to call at all.** `TryGemmCutlass` in `gemm.cu`
// reads the same table through `CutlassCandidateFor`, so a shape the table does
// not promote keeps the hand tile and then cuBLAS. This file therefore never
// refuses a call for being unpromoted: it serves the candidate it is given.
//
// **This kernel has no emulated leg.** The emulation engine maps no Tensor
// Memory Accelerator and no warpgroup instruction (docs/simulator.md
// section 9.2), so the off-device gate covers the hand tile and not this file.
// The hand tile therefore stays, as the reference and as the host leg.
//
// See docs/kernels.md Part 3 sections 10 and 11, and
// backends/cuda/gemm/README.md.

#include <cstddef>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/cutlass_kernel.cuh"
#include "backends/cuda/gemm/cutlass_launch.h"
#include "backends/cuda/gemm/cutlass_rules.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

namespace {

// The candidate this call runs.
//
// The environment wins, because a sweep sets it and one binary must measure
// every candidate. Without it the decision table answers, and a shape the table
// does not hold takes candidate 0: the launcher still serves the call, and
// `TryGemmCutlass` is what keeps an unpromoted shape away from it.
int CandidateFor(const ResolvedGemm& gemm) {
  const int forced = GemmCandidateFromEnv();
  if (forced >= 0) return forced;
  const int promoted = CutlassCandidateFor(gemm);
  return promoted >= 0 ? promoted : 0;
}

// One candidate's entries, so a dispatch is a table read and a bounds test and
// not a switch.
struct CandidateEntries {
  bool (*launch)(const ResolvedGemm&, const ComputeType*, const ComputeType*,
                 ComputeType*);
  bool (*launch_fp32)(const ResolvedGemm&, const ComputeType*,
                      const ComputeType*, float*);
  int (*occupancy)();
  CutlassGrid (*grid)(const ResolvedGemm&);
};

constexpr CandidateEntries kCandidateEntries[] = {
#define NANOCHAT_CANDIDATE_ENTRY(index)                      \
  CandidateEntries{&LaunchGemmCutlassCandidate##index,       \
                   &LaunchGemmCutlassCandidateToFp32##index, \
                   &GemmCutlassCandidateOccupancy##index,    \
                   &GemmCutlassCandidateGrid##index},
    NANOCHAT_FOR_EACH_CANDIDATE(NANOCHAT_CANDIDATE_ENTRY)
#undef NANOCHAT_CANDIDATE_ENTRY
};

static_assert(sizeof(kCandidateEntries) / sizeof(kCandidateEntries[0]) ==
                  static_cast<std::size_t>(kGemmCandidateCount),
              "the call table holds one row for each candidate of the "
              "registry");

}  // namespace

// --- The registry -----------------------------------------------------------

int GemmCutlassKernelCount() {
  return kGemmCandidateCount * GemmCandidateInstantiations();
}

const char* GemmCutlassTileName() { return GemmCandidateTileName(0); }

bool LaunchGemmCutlassCandidate(int candidate, const ResolvedGemm& gemm,
                                const ComputeType* a, const ComputeType* b,
                                ComputeType* c) {
  if (candidate < 0 || candidate >= kGemmCandidateCount) return false;
  return kCandidateEntries[candidate].launch(gemm, a, b, c);
}

bool LaunchGemmCutlassCandidateToFp32(int candidate, const ResolvedGemm& gemm,
                                      const ComputeType* a,
                                      const ComputeType* b, float* c) {
  if (candidate < 0 || candidate >= kGemmCandidateCount) return false;
  return kCandidateEntries[candidate].launch_fp32(gemm, a, b, c);
}

int GemmCutlassCandidateOccupancy(int candidate) {
  if (candidate < 0 || candidate >= kGemmCandidateCount) return 0;
  return kCandidateEntries[candidate].occupancy();
}

CutlassGrid GemmCutlassCandidateGrid(int candidate, const ResolvedGemm& gemm) {
  if (candidate < 0 || candidate >= kGemmCandidateCount) return CutlassGrid{};
  return kCandidateEntries[candidate].grid(gemm);
}

int GemmCutlassRefusalCount() { return cutlass_detail::RefusalCount(); }

void ResetGemmCutlassRefusalCount() { cutlass_detail::RefusalCount() = 0; }

// --- The seam entries -------------------------------------------------------

bool LaunchGemmCutlass(const ResolvedGemm& gemm, const ComputeType* a,
                       const ComputeType* b, ComputeType* c) {
  if (!cutlass_detail::Facts().hopper) return false;
  const int candidate = CandidateFor(gemm);
  if constexpr (!kComputeIsBf16) {
    return false;
  } else {
    return LaunchGemmCutlassCandidate(candidate, gemm, a, b, c);
  }
}

bool LaunchGemmCutlassToFp32(const ResolvedGemm& gemm, const ComputeType* a,
                             const ComputeType* b, float* c) {
  if (!cutlass_detail::Facts().hopper) return false;
  const int candidate = CandidateFor(gemm);
  if constexpr (!kComputeIsBf16) {
    return false;
  } else {
    return LaunchGemmCutlassCandidateToFp32(candidate, gemm, a, b, c);
  }
}

}  // namespace kernels
}  // namespace nanochat
