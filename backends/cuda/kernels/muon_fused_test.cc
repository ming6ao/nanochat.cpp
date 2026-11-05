// T1 GPU correctness gate for the fused Muon path (docs/muon-fused-kernel.md
// section 6, phases 2 to 4 of section 5).
//
// Three implementations of one Muon step run on one input and one seed:
//
//   * the host reference, `testing/optim_ref.h`, which is the specification;
//   * the shipped `MuonUpdate`, which is the multi-launch path and keeps cuBLAS
//     for the Polar Express products;
//   * the fused path (`muon_fused.cu`), run through the phase driver and
//   through
//     the cooperative driver, so both ends of section 4's table are covered.
//
// Every one of them is compared against the reference at the family's 1e-5
// tolerance, on the tall and wide branches, both `red_dim` values, Nesterov on
// and off, `ns_steps` of 0, 1, and 5, and `num_params` of 1 and 3. The stages
// that keep their arithmetic order (momentum, equilibration, both scales, the
// factored second moment, and the update) are therefore checked against the
// implementation they must stay bit exact with, and the GEMM's reduction order
// is the one difference the tolerance covers.
//
// The gates that are not a tolerance comparison:
//
//   * the two drivers must produce one answer, bit for bit: they run one device
//     body over one workspace, so only the launch geometry differs;
//   * the workspace check: the 16 phases and the three epilogue folds have to
//   fit
//     in `2 * total + 2 * min_total` elements, which is one `min_total` slice
//     less than the multi-launch path used to carry. The fused path is handed
//     exactly that many elements with a sentinel tail behind them, so a phase
//     that wrote past the end would show up as a changed sentinel;
//   * a forced `cudaErrorCooperativeLaunchTooLarge` -- the persistent grid
//     inflated past what the card can keep resident -- falls back to the phase
//     driver and produces the phase driver's answer.
//
// A bf16 build keeps the multi-launch cuBLAS path for every group, so the fused
// schedule (fp32 operands) is not run there and this test reports it.
//
// See docs/testing.md and backends/cuda/kernels/README.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/muon_fused.h"
#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/optim_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::MuonParams;
using nanochat::dev::C;
using nanochat::dev::Check;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::optimref::MuonUpdate;
using nanochat::kernels::MuonFusedDriver;
using nanochat::kernels::MuonFusedDriverName;
using nanochat::kernels::MuonFusedExecutions;
using nanochat::kernels::MuonFusedOptions;
using nanochat::kernels::MuonFusedPartialDoubles;
using nanochat::kernels::MuonFusedPhaseCount;
using nanochat::kernels::MuonFusedRun;
using nanochat::kernels::MuonFusedScratchFloats;
using nanochat::kernels::MuonFusedUpdate;
using nanochat::kernels::MuonFusedWorkspaceElements;

// The per-matrix size of buf2 for a `red_dim`, the rule the kernel, the
// multi-launch path, and the reference all apply.
int RedIndex(int rows, int cols, int red_dim) {
  const bool reduce_cols = red_dim == -1 || (red_dim != -2 && rows >= cols);
  return reduce_cols ? rows : cols;
}

std::vector<float> Zeros(int n) { return std::vector<float>(n, 0.0f); }

// True when every value is finite. A `CheckVectorClose` comparison against a
// NaN is false for every tolerance, so a divergent stage has to be caught here.
bool AllFinite(const std::vector<float>& values) {
  for (const float value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

// The parameters the tests run with, resolved the way `NanochatOptimizer::Step`
// resolves them for this family.
MuonParams MakeParams(int num_params, int rows, int cols, int red_dim,
                      bool nesterov, int ns_steps) {
  MuonParams params;
  params.num_params = num_params;
  params.rows = rows;
  params.cols = cols;
  params.lr = 0.02f;
  params.momentum = 0.95f;
  params.beta2 = 0.9f;
  params.weight_decay = 0.01f;
  params.ns_steps = ns_steps;
  params.red_dim = red_dim;
  params.nesterov = nesterov;
  return params;
}

// One implementation's device state: the fp32 master, the `ComputeType` copy
// the graph reads, the momentum buffer, and the factored second moment.
struct State {
  State(int total, const std::vector<float>& master, int buf2_len)
      : master(master),
        values(ToStorage(master)),
        buf1(Zeros(total)),
        buf2(Zeros(buf2_len)) {}

  DevBuf<float> master;
  DevBuf<ComputeType> values;
  DevBuf<float> buf1;
  DevBuf<float> buf2;
};

// The fused path's three cached buffers, at the sizes its own helpers state.
// The workspace is the iterate and its partner plus the two min-by-min
// operands; the scratch is the statistics, and the partials are the double
// reduction buffer.
struct FusedBuffers {
  FusedBuffers(int num_params, int rows, int cols)
      : workspace(static_cast<int>(
            MuonFusedWorkspaceElements(num_params, rows, cols))),
        scratch(
            static_cast<int>(MuonFusedScratchFloats(num_params, rows, cols))),
        partials(
            static_cast<int>(MuonFusedPartialDoubles(num_params, rows, cols))) {
  }

  DevBuf<ComputeType> workspace;
  DevBuf<float> scratch;
  DevBuf<double> partials;
};

void CheckState(const State& state, const std::vector<float>& master,
                const std::vector<float>& buf1, const std::vector<float>& buf2,
                double tol, const char* name, int step, const char* who) {
  char label[192];
  const std::vector<float> got_master = state.master.Download();
  const std::vector<float> got_buf1 = state.buf1.Download();
  const std::vector<float> got_buf2 = state.buf2.Download();
  const std::vector<float> got_values =
      nanochat::dev::FromStorage(state.values.Download());
  // `CheckVectorClose` compares a difference against a tolerance, and every
  // comparison against a NaN is false, so a divergent stage would pass it.
  // Every output has to be finite before it is compared.
  std::snprintf(label, sizeof(label), "%s step %d %s finite", name, step, who);
  Check(AllFinite(got_master) && AllFinite(got_buf1) && AllFinite(got_buf2) &&
            AllFinite(got_values),
        label);
  std::snprintf(label, sizeof(label), "%s step %d %s params", name, step, who);
  CheckVectorClose(got_master, master, tol, label);
  std::snprintf(label, sizeof(label), "%s step %d %s buf1", name, step, who);
  CheckVectorClose(got_buf1, buf1, tol, label);
  std::snprintf(label, sizeof(label), "%s step %d %s buf2", name, step, who);
  CheckVectorClose(got_buf2, buf2, tol, label);
  std::snprintf(label, sizeof(label), "%s step %d %s values", name, step, who);
  CheckVectorClose(got_values, master, tol, label);
}

// One family case: every implementation runs `steps` sequential steps from the
// same initial state, and each step is compared against the host reference.
void RunCase(int num_params, int rows, int cols, int red_dim, bool nesterov,
             int ns_steps, int steps, const char* name) {
  const int total = num_params * rows * cols;
  const int buf2_len = num_params * RedIndex(rows, cols, red_dim);
  const MuonParams params =
      MakeParams(num_params, rows, cols, red_dim, nesterov, ns_steps);

  Rng rng;
  const std::vector<float> grads_host = RandomVec(total, &rng);
  const std::vector<float> master_host = RandomVec(total, &rng);

  DevBuf<float> grads(grads_host);
  State shipped(total, master_host, buf2_len);
  State phase(total, master_host, buf2_len);
  State cooperative(total, master_host, buf2_len);
  FusedBuffers fused(num_params, rows, cols);

  MuonFusedOptions phase_options;
  phase_options.driver = MuonFusedDriver::kPhase;
  MuonFusedOptions cooperative_options;
  cooperative_options.driver = MuonFusedDriver::kCooperative;

  std::vector<float> master_ref = master_host;
  std::vector<float> buf1_ref = Zeros(total);
  std::vector<float> buf2_ref = Zeros(buf2_len);

  for (int step = 0; step < steps; ++step) {
    nanochat::kernels::MuonUpdate(params, grads.ptr, shipped.master.ptr,
                                  shipped.values.ptr, shipped.buf1.ptr,
                                  shipped.buf2.ptr);
    const MuonFusedRun phase_run =
        MuonFusedUpdate(params, grads.ptr, phase.master.ptr, phase.values.ptr,
                        phase.buf1.ptr, phase.buf2.ptr, fused.workspace.ptr,
                        fused.scratch.ptr, fused.partials.ptr, phase_options);
    const MuonFusedRun cooperative_run = MuonFusedUpdate(
        params, grads.ptr, cooperative.master.ptr, cooperative.values.ptr,
        cooperative.buf1.ptr, cooperative.buf2.ptr, fused.workspace.ptr,
        fused.scratch.ptr, fused.partials.ptr, cooperative_options);
    nanochat::kernels::Synchronize();
    MuonUpdate(params, grads_host, &master_ref, &buf1_ref, &buf2_ref);

    // The driver each call took. A refused cooperative launch is the one answer
    // that lands on the phase driver without the caller asking for it.
    Check(phase_run.driver == MuonFusedDriver::kPhase && !phase_run.fell_back,
          "the phase driver runs when it is asked for");
    Check(cooperative_run.driver == MuonFusedDriver::kCooperative ||
              cooperative_run.fell_back,
          "the cooperative driver either launches or falls back");

    const int one = step + 1;
    CheckState(shipped, master_ref, buf1_ref, buf2_ref, 1e-5, name, one,
               "multi-launch");
    CheckState(phase, master_ref, buf1_ref, buf2_ref, 1e-5, name, one,
               "fused-phase");
    CheckState(cooperative, master_ref, buf1_ref, buf2_ref, 1e-5, name, one,
               "fused-cooperative");

    // Both drivers run one device body over one workspace, so they must agree
    // exactly and not merely within the tolerance.
    char label[192];
    std::snprintf(label, sizeof(label), "%s step %d drivers agree", name, one);
    CheckVectorClose(cooperative.master.Download(), phase.master.Download(),
                     0.0, label);
    std::snprintf(label, sizeof(label), "%s step %d drivers agree buf1", name,
                  one);
    CheckVectorClose(cooperative.buf1.Download(), phase.buf1.Download(), 0.0,
                     label);
    std::snprintf(label, sizeof(label), "%s step %d drivers agree buf2", name,
                  one);
    CheckVectorClose(cooperative.buf2.Download(), phase.buf2.Download(), 0.0,
                     label);
  }
}

// The workspace check. The fused path is handed exactly
// `MuonFusedWorkspaceElements` elements with a sentinel tail behind them: a
// phase or a fold that reached past the last slice changes the tail, and the
// master it produces still has to match the reference.
void CheckWorkspace(int num_params, int rows, int cols) {
  const std::size_t needed = MuonFusedWorkspaceElements(num_params, rows, cols);
  const std::size_t total = static_cast<std::size_t>(num_params) * rows * cols;
  const std::size_t min_dim = static_cast<std::size_t>(std::min(rows, cols));
  const std::size_t min_total =
      static_cast<std::size_t>(num_params) * min_dim * min_dim;
  Check(needed == 2 * total + 2 * min_total,
        "the workspace is 2 * total + 2 * min_total");
  Check(needed < 2 * total + 3 * min_total,
        "the folds removed one min_total slice");

  constexpr int kGuard = 64;
  const float kSentinel = 1234.5f;
  const int guard_len = static_cast<int>(needed) + kGuard;
  std::vector<ComputeType> host(static_cast<std::size_t>(guard_len),
                                C(kSentinel));
  DevBuf<ComputeType> buffer(host);

  const int buf2_len = num_params * RedIndex(rows, cols, -1);
  const MuonParams params = MakeParams(num_params, rows, cols, -1, true, 5);
  Rng rng;
  const std::vector<float> grads_host =
      RandomVec(num_params * rows * cols, &rng);
  const std::vector<float> master_host =
      RandomVec(num_params * rows * cols, &rng);
  DevBuf<float> grads(grads_host);
  State state(num_params * rows * cols, master_host, buf2_len);
  DevBuf<float> scratch(
      static_cast<int>(MuonFusedScratchFloats(num_params, rows, cols)));
  DevBuf<double> partials(
      static_cast<int>(MuonFusedPartialDoubles(num_params, rows, cols)));

  MuonFusedOptions options;
  options.driver = MuonFusedDriver::kPhase;
  MuonFusedUpdate(params, grads.ptr, state.master.ptr, state.values.ptr,
                  state.buf1.ptr, state.buf2.ptr, buffer.ptr, scratch.ptr,
                  partials.ptr, options);
  nanochat::kernels::Synchronize();

  std::vector<float> master_ref = master_host;
  std::vector<float> buf1_ref = Zeros(num_params * rows * cols);
  std::vector<float> buf2_ref = Zeros(buf2_len);
  MuonUpdate(params, grads_host, &master_ref, &buf1_ref, &buf2_ref);

  char label[192];
  std::snprintf(label, sizeof(label), "guarded %dx%d batch %d guards", rows,
                cols, num_params);
  const std::vector<ComputeType> written = buffer.Download();
  bool intact = true;
  for (int i = 0; i < kGuard; ++i) {
    if (written[needed + static_cast<std::size_t>(i)] != C(kSentinel)) {
      intact = false;
    }
  }
  Check(intact, label);

  std::snprintf(label, sizeof(label), "guarded %dx%d batch %d params", rows,
                cols, num_params);
  CheckVectorClose(state.master.Download(), master_ref, 1e-5, label);
}

// A cooperative launch the device refuses -- the persistent grid inflated past
// what the card keeps resident -- has to fall back to the phase driver, and the
// answer has to be the phase driver's.
void CheckCooperativeFallback(int num_params, int rows, int cols) {
  const int total = num_params * rows * cols;
  const int buf2_len = num_params * RedIndex(rows, cols, -1);
  const MuonParams params = MakeParams(num_params, rows, cols, -1, true, 5);

  Rng rng;
  const std::vector<float> grads_host = RandomVec(total, &rng);
  const std::vector<float> master_host = RandomVec(total, &rng);
  DevBuf<float> grads(grads_host);
  State phase(total, master_host, buf2_len);
  State forced(total, master_host, buf2_len);
  FusedBuffers fused(num_params, rows, cols);

  MuonFusedOptions phase_options;
  phase_options.driver = MuonFusedDriver::kPhase;
  MuonFusedUpdate(params, grads.ptr, phase.master.ptr, phase.values.ptr,
                  phase.buf1.ptr, phase.buf2.ptr, fused.workspace.ptr,
                  fused.scratch.ptr, fused.partials.ptr, phase_options);

  MuonFusedOptions forced_options;
  forced_options.driver = MuonFusedDriver::kCooperative;
  // Eight times the resident grid: a cooperative launch of that grid cannot be
  // kept resident, which is the `cudaErrorCooperativeLaunchTooLarge` path.
  forced_options.grid_multiplier = 8;
  const MuonFusedRun run =
      MuonFusedUpdate(params, grads.ptr, forced.master.ptr, forced.values.ptr,
                      forced.buf1.ptr, forced.buf2.ptr, fused.workspace.ptr,
                      fused.scratch.ptr, fused.partials.ptr, forced_options);
  nanochat::kernels::Synchronize();

  char label[192];
  std::snprintf(label, sizeof(label), "forced refusal %dx%d batch %d fell back",
                rows, cols, num_params);
  Check(run.fell_back, label);
  std::snprintf(label, sizeof(label),
                "forced refusal %dx%d batch %d runs the phase driver", rows,
                cols, num_params);
  Check(run.driver == MuonFusedDriver::kPhase, label);
  std::snprintf(label, sizeof(label), "forced refusal %dx%d batch %d params",
                rows, cols, num_params);
  CheckVectorClose(forced.master.Download(), phase.master.Download(), 0.0,
                   label);
  std::snprintf(label, sizeof(label), "forced refusal %dx%d batch %d buf2",
                rows, cols, num_params);
  CheckVectorClose(forced.buf2.Download(), phase.buf2.Download(), 0.0, label);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("muon fused gpu test\n");

  if (nanochat::kComputeIsBf16) {
    std::printf(
        "muon fused: a bf16 build keeps the multi-launch cuBLAS path for every "
        "group, so the fp32 fused schedule is not run\n");
    return 0;
  }

  // The schedule section 4 states, which the barrier count and the workspace
  // both follow from.
  Check(MuonFusedPhaseCount() == 16, "16 phases");
  Check(MuonFusedExecutions(5) == 28, "28 executions at ns_steps 5");
  Check(MuonFusedExecutions(0) == 13, "13 executions at ns_steps 0");
  Check(MuonFusedExecutions(5) - 1 == 27, "27 barriers at ns_steps 5");
  Check(std::string(MuonFusedDriverName(MuonFusedDriver::kPhase)) == "phase",
        "the phase driver has a name");

  // Section 6's case set: both branches, both `red_dim` values, Nesterov on and
  // off, `ns_steps` of 0, 1, and 5, and `num_params` of 1 and 3. The 5x5 shape
  // is not a whole number of 16-byte groups, so it is the scalar elementwise
  // path; the others run the vector one.
  const int shapes[][2] = {{4, 3}, {3, 6}, {5, 5}};
  for (const auto& shape : shapes) {
    const int rows = shape[0];
    const int cols = shape[1];
    for (int red_dim : {-1, -2}) {
      for (bool nesterov : {true, false}) {
        for (int ns_steps : {0, 1, 5}) {
          for (int num_params : {1, 3}) {
            char name[128];
            std::snprintf(name, sizeof(name),
                          "%dx%d red=%d nesterov=%d ns=%d batch=%d", rows, cols,
                          red_dim, nesterov ? 1 : 0, ns_steps, num_params);
            RunCase(num_params, rows, cols, red_dim, nesterov, ns_steps, 2,
                    name);
          }
        }
      }
    }
  }

  // One mid-size group per branch, so the GEMM's 16-byte loaders, its whole 128
  // by 64 tiles, its 8-deep k slices, and a partial pass over several blocks
  // all run: the tiny shapes above only reach the ragged-tile scalar gather and
  // a single k slice.
  for (int ns_steps : {1, 5}) {
    char name[64];
    std::snprintf(name, sizeof(name), "tall whole tiles ns=%d", ns_steps);
    RunCase(2, 256, 128, -1, true, ns_steps, 1, name);
    std::snprintf(name, sizeof(name), "wide whole tiles ns=%d", ns_steps);
    RunCase(1, 128, 256, -2, true, ns_steps, 1, name);
  }

  CheckWorkspace(3, 4, 3);
  CheckWorkspace(1, 3, 6);
  CheckCooperativeFallback(2, 4, 3);
  CheckCooperativeFallback(1, 3, 6);

  if (Failures() != 0) {
    std::printf("muon fused: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("muon fused: all checks passed\n");
  return 0;
}
