#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_

// The launcher of the Hopper mainloop that NVIDIA CUTLASS supplies
// (docs/kernels.md Part 3 section 5, dev/cutlass/README.md).
//
// Backend-internal: `backends/cuda/gemm.cu` keeps the seam entry and calls one
// of these when `JudgeCutlass` accepts the call. The frozen seam gains no
// symbol.
//
// Two entries, because `GemmMode` is the caller's label and not an operand:
// `Gemm` and `GemmWgrad` differ only in the type they store. That difference is
// not cosmetic here: the store type selects one template instantiation and one
// epilogue element, so eight kernels exist rather than four.
//
// Each returns false when the kernel did not take the call, which is not an
// error. A false happens on a device that cannot run the instruction, and the
// caller keeps the hand tile and then cuBLAS.
//
// The declarations are unconditional, name no vendor type, and include no CUDA
// header, so a host pass reads this file. Only a bf16 or fp8 build defines
// them. `gemm.cu` discards the call through `if constexpr (kComputeIsBf16)`,
// exactly as it does for the hand tile.
//
// `ResolvedGemm` is the argument and not `GemmParams`, because the caller has
// already resolved every default to decide the verdict. Two resolutions of one
// call could disagree, and the disagreement would be silent.

#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// `C = alpha * op(A) * op(B) + beta * C`, with C stored as `ComputeType`.
bool LaunchGemmCutlass(const ResolvedGemm& gemm, const ComputeType* a,
                       const ComputeType* b, ComputeType* c);

// The same call with C stored as fp32, which is the weight-gradient form. The
// accumulator is fp32 in both, so this changes the store and nothing else.
bool LaunchGemmCutlassToFp32(const ResolvedGemm& gemm, const ComputeType* a,
                             const ComputeType* b, float* c);

// How many kernel instantiations the adapter holds: four operand orientations
// times two store types, for each candidate of the registry. A host test reads
// it to state the matrix is complete, and the answer is a constant of the
// translation unit and not of the card.
int GemmCutlassKernelCount();

// The tile candidate 0 runs, as a name a report can print. It is the fallback
// when nothing selects another candidate.
const char* GemmCutlassTileName();

// --- The candidate registry (docs/kernels.md section 11.4) -------------------
//
// One candidate is a registry row: a tile, a cluster, a stage count, a kernel
// schedule, and an epilogue schedule. `cutlass_candidate.cu` is one source file
// compiled once for each index, so one build never holds two candidates'
// instantiations in one compiler process.
//
// **The index is the only thing a caller names.** `LaunchGemmCutlassCandidate`,
// `LaunchGemmCutlassCandidateToFp32`, and `GemmCutlassCandidateOccupancy` take
// an index and read a table, so a caller names no numbered entry and a new
// candidate adds no branch. A caller that has already resolved the candidate,
// which is the seam entry, calls them directly.
bool LaunchGemmCutlassCandidate(int candidate, const ResolvedGemm& gemm,
                                const ComputeType* a, const ComputeType* b,
                                ComputeType* c);
bool LaunchGemmCutlassCandidateToFp32(int candidate, const ResolvedGemm& gemm,
                                      const ComputeType* a,
                                      const ComputeType* b, float* c);

// The active blocks for each multiprocessor the driver answers for one
// candidate, or zero when the index names no candidate or the query failed. It
// is the occupancy column of the sweep report: the cooperative schedule keeps
// one block resident, and a reader must otherwise infer that from a timing.
int GemmCutlassCandidateOccupancy(int candidate);

// --- The per-candidate entries ---------------------------------------------
//
// One entry set for each index, declared from the registry's own list. The
// suffix states the index, because one source file compiled once for each
// candidate must export a distinct name each time. Nothing else in the tree
// spells one of these names.
#define NANOCHAT_DECLARE_CANDIDATE(index)                                   \
  bool LaunchGemmCutlassCandidate##index(                                   \
      const ResolvedGemm& gemm, const ComputeType* a, const ComputeType* b, \
      ComputeType* c);                                                      \
  bool LaunchGemmCutlassCandidateToFp32##index(                             \
      const ResolvedGemm& gemm, const ComputeType* a, const ComputeType* b, \
      float* c);                                                            \
  int GemmCutlassCandidateOccupancy##index();

NANOCHAT_FOR_EACH_CANDIDATE(NANOCHAT_DECLARE_CANDIDATE)

#undef NANOCHAT_DECLARE_CANDIDATE

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_LAUNCH_H_
