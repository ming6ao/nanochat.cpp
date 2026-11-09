#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_RULES_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_RULES_H_

// The decision table of the GEMM arms (docs/kernels.md section 11.4 item 5).
//
// **The order is candidate, then hand tile, then cuBLAS.** A row of the closed
// list of `dispatch.cc` names the arm the sweep chose for that shape, or states
// that neither arm took it. The seam reads this table and nothing else, so a
// training step pays one table lookup. The choice is offline: a run-time timing
// pass would put card work at the start of every step, and the shape rule would
// stop being a host-only check.
//
// **The table is data.** `tools/gemm_rules.py --write` writes the block between
// the markers below from a `gemm_bench --json` report of the production rows.
// `--check` compares the committed block against the report that produced it,
// so a hand edit fails the gate instead of changing a shipped decision.
//
// **Only the production rows may enter.** The measurement rows of
// `bench_shapes.h` answer a question about a kernel and not about a model, so
// they never select an arm for a production shape. They decide whether a
// candidate is fit to enter a sweep.
//
// **The hand column is a record, not a promotion.** A row whose route is
// `kHand` means cuBLAS took the shape after the CUTLASS candidates lost, and
// the hand tile's own promotion stays the enabled bit of `dispatch.cc`. This
// table never turns that bit on.
//
// A row's key is the extents, the batch, and the operand form. `GemmRuleFor`
// also requires the tight leading dimensions, so a call with padding falls
// through to the hand tile and then to cuBLAS instead of taking a shape's
// measurement it does not match.

#include <array>

#include "backends/cuda/gemm/gemm_shape.h"

namespace nanochat {
namespace kernels {

// Which arm a shape takes, in the order section 11.4 states.
enum class GemmRoute {
  // Neither the CUTLASS candidates nor the hand tile beat cuBLAS.
  kCublas = 0,
  // The hand tile took the shape and no CUTLASS candidate did. The seam refuses
  // the CUTLASS call, and the hand tile's own bit decides what serves it.
  kHand = 1,
  // A candidate of the registry took the shape.
  kCandidate = 2,
};

// `route` as a report or a log line spells it.
inline const char* GemmRouteName(GemmRoute route) {
  switch (route) {
    case GemmRoute::kHand:
      return "hand";
    case GemmRoute::kCandidate:
      return "candidate";
    case GemmRoute::kCublas:
    default:
      return "cublas";
  }
}

// One row of the decision table.
struct GemmRuleRow {
  int m;
  int n;
  int k;
  int batch_count;
  bool transpose_a;
  bool transpose_b;
  bool fp32_c;
  // The registry index, or -1 when the row names no candidate.
  int candidate;
  GemmRoute route;
  // The measured fraction of the cuBLAS time the chosen candidate read, and the
  // hand tile's fraction on the same operands. The fractions are the evidence
  // of the row, and a reader compares them without a second run.
  double of_cublas;
  double hand_of_cublas;
};

// --- BEGIN GENERATED RULES ---
// Written by `tools/gemm_rules.py --write` from a `gemm_bench --json`
// report of the production rows. A row promotes the arm that beat cuBLAS
// by the measured margin of section 11.4 item 5, and every other row
// keeps cuBLAS. The fractions are that run's, so the block is the
// evidence of the decision and not a summary of it.
inline constexpr std::array<GemmRuleRow, 28> kGemmRules{{
    // muon:5x12:A
    {
        5,
        5,
        12,
        10,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.7398,
    },
    // muon:5x12:A2
    {
        5,
        5,
        5,
        10,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.6259,
    },
    // muon:5x12:prod
    {
        5,
        12,
        5,
        10,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.6390,
    },
    // muon:640x1280:A
    {
        640,
        640,
        1280,
        40,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.4101,
    },
    // muon:640x1280:A2
    {
        640,
        640,
        640,
        40,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.4132,
    },
    // muon:640x1280:prod
    {
        640,
        1280,
        640,
        40,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3755,
    },
    // muon:1280x1280:A
    {
        1280,
        1280,
        1280,
        40,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3115,
    },
    // muon:1280x1280:A2+prod
    {
        1280,
        1280,
        1280,
        40,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3151,
    },
    // muon:1280x5120:A
    {
        1280,
        1280,
        5120,
        20,
        false,
        true,
        false,
        0,
        GemmRoute::kCandidate,
        1.0850,
        0.3454,
    },
    // muon:1280x5120:A2+A2
    {
        1280,
        1280,
        1280,
        20,
        false,
        false,
        false,
        0,
        GemmRoute::kCandidate,
        1.1270,
        0.3479,
    },
    // muon:1280x5120:prod
    {
        1280,
        5120,
        1280,
        20,
        false,
        false,
        false,
        0,
        GemmRoute::kCandidate,
        1.1757,
        0.3495,
    },
    // muon:5120x1280:A
    {
        1280,
        1280,
        5120,
        20,
        true,
        false,
        false,
        0,
        GemmRoute::kCandidate,
        1.1027,
        0.3348,
    },
    // muon:5120x1280:prod
    {
        5120,
        1280,
        1280,
        20,
        false,
        false,
        false,
        0,
        GemmRoute::kCandidate,
        1.1297,
        0.3278,
    },
    // linear:fwd:q+proj
    {
        16384,
        1280,
        1280,
        1,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3040,
    },
    // linear:dgrad:q+proj
    {
        16384,
        1280,
        1280,
        1,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3057,
    },
    // linear:wgrad:q+proj
    {
        1280,
        1280,
        16384,
        1,
        true,
        false,
        true,
        1,
        GemmRoute::kCandidate,
        1.1182,
        0.2536,
    },
    // linear:fwd:kv
    {
        16384,
        640,
        1280,
        1,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3168,
    },
    // linear:dgrad:kv
    {
        16384,
        1280,
        640,
        1,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.2907,
    },
    // linear:wgrad:kv
    {
        640,
        1280,
        16384,
        1,
        true,
        false,
        true,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.1470,
    },
    // linear:fwd:fc
    {
        16384,
        5120,
        1280,
        1,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.2942,
    },
    // linear:dgrad:fc
    {
        16384,
        1280,
        5120,
        1,
        false,
        false,
        false,
        0,
        GemmRoute::kCandidate,
        1.0484,
        0.3441,
    },
    // linear:wgrad:fc
    {
        5120,
        1280,
        16384,
        1,
        true,
        false,
        true,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3255,
    },
    // linear:fwd:cproj
    {
        16384,
        1280,
        5120,
        1,
        false,
        true,
        false,
        0,
        GemmRoute::kCandidate,
        1.0459,
        0.3474,
    },
    // linear:dgrad:cproj
    {
        16384,
        5120,
        1280,
        1,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.2745,
    },
    // linear:wgrad:cproj
    {
        1280,
        5120,
        16384,
        1,
        true,
        false,
        true,
        0,
        GemmRoute::kCandidate,
        1.0554,
        0.3524,
    },
    // linear:fwd:cls
    {
        16384,
        32768,
        1280,
        1,
        false,
        true,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.2876,
    },
    // linear:dgrad:cls
    {
        16384,
        1280,
        32768,
        1,
        false,
        false,
        false,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3181,
    },
    // linear:wgrad:cls
    {
        32768,
        1280,
        16384,
        1,
        true,
        false,
        true,
        -1,
        GemmRoute::kCublas,
        1.0000,
        0.3203,
    },
}};
// --- END GENERATED RULES ---

// The count of rows the table holds.
inline constexpr int kGemmRuleCount = static_cast<int>(kGemmRules.size());

// The rule of one call, or nullptr when the table does not hold the shape. A
// call whose leading dimensions are not the tight ones holds no row.
const GemmRuleRow* GemmRuleFor(const ResolvedGemm& gemm);

// The candidate the table selects for this shape, or -1 when no candidate does.
// It is what the seam asks before it takes a CUTLASS call, so a shape the table
// does not promote keeps the hand tile and then cuBLAS.
int CutlassCandidateFor(const ResolvedGemm& gemm);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_RULES_H_
