#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_FUSED_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_FUSED_H_

// The fused Muon update: one phase-parameterised device function, two drivers,
// and the workspace both `MuonUpdate` paths share (docs/kernels.md
// sections 4 to 6). This header is backend-internal, like `rms_norm.h`: only
// code under backends/cuda includes it, and the frozen `nanochat/kernels.h`
// seam keeps its one `MuonUpdate` entry point and gains no capability bit.
//
// `backends/cuda/kernels/muon.cu` owns the seam entry. It selects a path by
// shape (`MuonFusedSelected`), sizes the one workspace with these helpers, and
// calls `MuonFusedUpdate` for the fused path; the multi-launch path stays in
// muon.cu and keeps `kernels::Gemm`, which is cuBLAS.
//
// The 28 phase executions of the fused path (16 phases, the Polar Express ones
// repeated `ns_steps` times) are:
//
//   1      momentum and the Nesterov combination
//   2, 3   partial pass and finish for the Frobenius norm
//   4      row equilibration, one block per (matrix, row)
//   5, 6   partial pass and finish for the divisor 1.01*||x|| + 1e-6
//   7      divisor scale
//   8      A = X^T X (tall) or X X^T (wide), batched GEMM
//   9      A.A, with the b = cb*a + cc*acc epilogue
//   10     X.b (tall) or b.X (wide), with the xnext = ca*x + prod epilogue
//   11-13  renormalisation: partials, finish, scale
//   14     NorMuon mean squares, per row or per column
//   15     the factored second moment and the per-index step
//   16     the NorMuon scale, with the cautious update as its epilogue
//
// Phases 1 to 7 run once, 8 to 10 run once per Polar Express iteration, and 11
// to 16 run once, which is 28 executions and therefore 27 grid barriers in the
// cooperative driver against the 39 launches the multi-launch path issues.
//
// See docs/kernels.md, docs/model.md, and docs/simulator.md.

#include <cstddef>

#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

// The blocks one matrix is reduced with: one per eight thousand elements, at
// least one and at most 32. The whole-matrix statistic passes of both
// `MuonUpdate` paths use this one answer, so the block count (and therefore the
// partial order, and therefore the statistic to the last bit) cannot drift
// between the paths.
int MuonPartialBlocks(long long mat);

// The working set the fused path needs, in `ComputeType` elements:
// `2 * total + 2 * min_total`. Two full-size matrices are the iterate and its
// ping-pong partner (phase 10 writes the combined iterate into the second one,
// which is why the `prod` buffer of the multi-launch path doubles as it), and
// two `min_dim` by `min_dim` buffers are `A` and the folded `b` (phase 9 writes
// `b` from the accumulator, which removes the separate `a2` buffer). The
// multi-launch path uses the same four slices.
std::size_t MuonFusedWorkspaceElements(int num_params, int rows, int cols);

// The `float` statistics scratch both paths share, in floats: the per-matrix
// statistic (`num_params`), then the NorMuon mean squares
// (`num_params * max(rows, cols)`), then the NorMuon global scales
// (`num_params`). The larger extent settles the size, so one allocation serves
// both `red_dim` values of a group.
std::size_t MuonFusedScratchFloats(int num_params, int rows, int cols);

// One region of that scratch, rounded up to a 16-byte group. Every region
// starts on a vector boundary, so the elementwise phases of the fused path can
// move a whole 16-byte group of the statistics and not only of the matrices: a
// region that began mid-vector would switch those phases to their scalar form
// for the whole group. Both paths slice the scratch with this one rule.
inline std::size_t MuonScratchOffset(std::size_t count) {
  return (count + 3) & ~static_cast<std::size_t>(3);
}

// The `double` partial buffer both paths share, in doubles:
// `num_params * MuonPartialBlocks(rows * cols)`.
std::size_t MuonFusedPartialDoubles(int num_params, int rows, int cols);

// Which driver runs the fused device function (section 4's table).
enum class MuonFusedDriver {
  // The cooperative launch when the device accepts one, the phase driver when
  // it does not or when the launch is refused. What `MuonUpdate` asks for.
  kAuto,
  // One launch per phase execution. The simulator's path, the fallback for a
  // card without a cooperative launch, and the phase-bisection tool.
  kPhase,
  // One launch for the whole group, with a grid barrier between executions, and
  // the phase driver as the fallback.
  kCooperative,
};

// `driver` as a report row spells it: "auto", "phase", or "cooperative".
const char* MuonFusedDriverName(MuonFusedDriver driver);

// How many phase executions one group runs, and how many phases there are. The
// gate and the benchmark report both numbers, and the cooperative driver's
// barrier count is `MuonFusedExecutions(ns_steps) - 1`.
int MuonFusedExecutions(int ns_steps);
int MuonFusedPhaseCount();

// The two knobs the gates and the benchmark need, and nothing a shipped call
// sets: which driver to ask for, and how far to inflate the persistent grid.
struct MuonFusedOptions {
  MuonFusedDriver driver = MuonFusedDriver::kAuto;
  // Multiplies the grid the resident-block query sizes. A value above one asks
  // for a grid the device cannot hold resident, which is how the gate forces
  // `cudaErrorCooperativeLaunchTooLarge` and checks that the fallback produces
  // the same answer as the phase driver. One is the production value.
  int grid_multiplier = 1;
};

// What one fused update did: the driver that produced the answer, and whether a
// refused cooperative launch sent it to the phase driver. The gate asserts
// both, because a fallback that is not observable is a fallback nobody has
// tested.
struct MuonFusedRun {
  MuonFusedDriver driver = MuonFusedDriver::kPhase;
  bool fell_back = false;
};

// One Muon step over one shape group, through the fused path.
//
// `workspace`, `scratch`, and `partials` are the caller's cached buffers, sized
// with the helpers above (muon.cu owns the allocation and reuses it across
// groups). The update reads `stacked_grads` and updates `stacked_master`,
// `stacked_values`, `buf1`, and `buf2`, exactly as the multi-launch path does.
//
// The fused schedule is the fp32 one (`ComputeType` floats, loaded as float4
// quads by its GEMM), so a bf16 build stays on the multi-launch cuBLAS path and
// `MuonFusedSelected` reports false there.
MuonFusedRun MuonFusedUpdate(
    const MuonParams& params, const float* stacked_grads, float* stacked_master,
    ComputeType* stacked_values, float* buf1, float* buf2,
    ComputeType* workspace, float* scratch, double* partials,
    const MuonFusedOptions& options = MuonFusedOptions());

// Which path one group takes (section 4: "keep the multi-launch path for groups
// where the bet is poor, and select by shape"). The threshold is the measured
// one recorded in backends/cuda/kernels/README.md: a group whose Polar Express
// arithmetic is small enough that the folded elementwise passes and the removed
// launch gaps pay for the hand GEMM's rate gap takes the fused path, and a
// group where the GEMM dominates stays on `kernels::Gemm`.
bool MuonFusedSelected(const MuonParams& params);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_MUON_FUSED_H_
