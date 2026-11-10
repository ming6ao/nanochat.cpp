#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_

// The closed shape list of the hand tensor-core GEMM, and the one verdict that
// decides whether a call runs on it (docs/tensor-core-gemm-plan.md section
// B.6).
//
// The list holds one row per shape of the closed list: thirteen Muon rows and
// fifteen Linear rows. Every row is inert, because the sweep measured each
// shape against cuBLAS and no shape reached the promotion gate
// (`backends/cuda/gemm/README.md` section 8). A shape takes the tile only after
// a committed measurement turns its bit on, so no run can change behaviour
// without that measurement. cuBLAS serves every other shape, and a shape the
// tile does not hold exactly stays on cuBLAS.
//
// `GemmTileMode` is the measurement switch. A process reads it from
// `NANOCHAT_GEMM_LIB`, and a gate passes it in directly.
//
// `NANOCHAT_GEMM_SCHEDULE` names one schedule of `gemm_plan.h` and overrides
// the row's own, which is how a card run sweeps one shape over the sweep.
//
// `NANOCHAT_GEMM_SWIZZLE` and `NANOCHAT_GEMM_RASTER` set the two launch
// arguments of the CUTLASS persistent scheduler. The sweep that reads them
// follows the protocol of `docs/kernels.md` section 11.4.

#include <string>

#include "backends/cuda/gemm/gemm_shape.h"

namespace nanochat {
namespace kernels {

enum class GemmTileMode {
  // The shape table decides. The shipped value.
  kAuto,
  // The Hopper mainloop of `cutlass_bf16.cu` takes every shape its own rule
  // accepts. An A/B measurement sets this, and it is the value a promotion
  // would leave on.
  kCutlass,
  // The hand tile takes every callable shape, for an A/B measurement. Nothing
  // but a benchmark sets this.
  kHand,
  // cuBLAS takes every call, for the other half of the same measurement.
  kCublas,
};

// `mode` as a report or a log line spells it.
const char* GemmTileModeName(GemmTileMode mode);

// The mode `NANOCHAT_GEMM_LIB` names: "cutlass", "hand", "cublas", or "auto"
// for anything else, including an unset variable.
GemmTileMode GemmTileModeFromEnv();

// The schedule this call takes, or -1 when cuBLAS serves it.
int GemmTileScheduleFor(const ResolvedGemm& gemm, GemmTileMode mode);

// True when `GemmTileScheduleFor` names a schedule.
bool GemmTileSelected(const ResolvedGemm& gemm, GemmTileMode mode);

// The sweep of `gemm_plan.h`, by name.
int GemmScheduleCount();
const char* GemmScheduleName(int index);

// The index `NANOCHAT_GEMM_SCHEDULE` names, or -1 when the variable is unset,
// empty, or names no schedule.
int GemmScheduleIndexFromEnv();

// The rows of the closed list, in table order. One row per shape, each with the
// name the benchmark reports and the key it matches.
int GemmTileShapeCount();
const char* GemmTileShapeName(int index);
const ResolvedGemm& GemmTileShapeGemm(int index);
bool GemmTileShapeEnabled(int index);
int GemmTileShapeSchedule(int index);

// Turns one row on or off. The shipped table is inert, so a campaign flips a
// bit here after a measurement and a gate flips one to prove the path. The
// row's schedule is already the one its measurement chose.
void SetGemmTileShapeEnabled(int index, bool enabled);

// --- The Hopper scheduler's two launch arguments -----------------------------
//
// The CUTLASS persistent scheduler walks the tile grid, and it can keep a group
// of tiles in the second-level cache. Both are launch arguments and not
// template parameters, so one built kernel serves every pair. This block names
// no CUDA type, so `dispatch_test` reads it on the host.

// The order the walk takes over the tile grid.
enum class CutlassRasterOrder {
  // The vendor chooses from the grid shape. The shipped value.
  kHeuristic = 0,
  // M first, so the resident blocks share a B panel.
  kAlongM,
  // N first, so they share an A panel.
  kAlongN,
};

// `order` as a report, a log line, or an environment value spells it.
const char* CutlassRasterOrderName(CutlassRasterOrder order);

// The order `name` spells, or `kHeuristic` for any other value.
CutlassRasterOrder CutlassRasterOrderFromName(const char* name);

// The two launch arguments of the persistent scheduler.
struct CutlassScheduler {
  // The widest tile group the walk keeps. One is the vendor default, and it
  // disables the group.
  int max_swizzle_size = 1;
  CutlassRasterOrder raster = CutlassRasterOrder::kHeuristic;
};

// True when `NANOCHAT_GEMM_SWIZZLE` or `NANOCHAT_GEMM_RASTER` states a value,
// which fills `*out`. A false leaves `*out` alone.
bool CutlassSchedulerFromEnv(CutlassScheduler* out);

// The tile counts of a call. They follow the kernel's own tile and not the
// shape alone, so the caller states them.
struct CutlassTiles {
  int m = 0;
  int n = 0;
};

// The smallest tile count that takes a group. The vendor ignores a request
// below six.
inline constexpr int kCutlassGroupMinTiles = 6;

// The multiple of sixteen that makes a group of eight cost no block.
//
// A group rounds both tile counts up to its own width times the cluster's, so a
// count off the multiple spends blocks that compute nothing. Sixteen covers a
// cluster of one or two blocks on either axis.
inline constexpr int kCutlassGroupMultiple = 16;

// True when a group of eight costs no block, which is the only case the rule
// takes.
bool CutlassGroupIsFree(const CutlassTiles& tiles);

// The scheduler a call takes. The benchmark variables override the rule.
CutlassScheduler CutlassSchedulerFor(const CutlassTiles& tiles);

// True when a candidate serves a call that stores this type. A candidate states
// its store, and `CutlassCandidateFor` reads this before it answers, so no call
// reaches a kernel that would fall back under the candidate's name. An index
// outside the registry serves nothing.
bool CutlassCandidateServesStore(int candidate, bool fp32_c);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_DISPATCH_H_
