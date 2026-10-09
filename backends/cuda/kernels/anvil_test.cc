// T1 GPU correctness test for the ANVIL family. Runs several sequential steps
// on the device and compares the stacked parameter, the twin-rail velocity, and
// the lane energy against the host reference in optim_ref.h. Covers the tall
// and wide cascade branches, both `red_dim` values (column and row lanes),
// square matrices, a short map count, both rail states (pre- and post-engage),
// and a no-op shape. Tiny shapes.
//
// See docs/optimizer-anvil-design.md and docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/optim_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AnvilParams;
using nanochat::ComputeType;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::optimref::AnvilUpdate;

// The device stores ComputeType, so the reference consumes the same stored
// values the device does and the tolerance measures only the kernel arithmetic,
// not the input conversion. fp32 keeps the tight 1e-5; fp16 stores the working
// matrix as half, and the recorded tolerance covers the half-ulp rounding
// (2^-11 ~ 4.9e-4) that the cascade and the equalizer propagate.
#if defined(NANOCHAT_PRECISION_FP16)
constexpr double kTol = 3e-3;
#else
constexpr double kTol = 1e-5;
#endif

// The per-matrix lane count implied by the CPU reference's `red_dim` rule.
int LaneCount(int rows, int cols, int red_dim) {
  const bool reduce_cols = red_dim == -1 || (red_dim != -2 && rows >= cols);
  return reduce_cols ? rows : cols;
}

// Runs `steps` ANVIL steps on device and host, keeping both in lockstep, and
// compares the parameter, the two velocity rails, and the lane energy after
// every step. The device inputs are the rounded storage values, and the host
// reference consumes the same floats the device does, so the recorded
// tolerance measures only the kernel arithmetic.
void RunCase(int num_params, int rows, int cols, int red_dim, int num_maps,
             float fast_beta, float fast_weight, bool nesterov,
             const char* name, int steps) {
  const int mat = rows * cols;
  const int total = num_params * mat;
  const int lane_count = LaneCount(rows, cols, red_dim);

  AnvilParams params;
  params.num_params = num_params;
  params.rows = rows;
  params.cols = cols;
  params.lr = 0.023f;
  params.momentum = 0.93f;
  params.fast_beta = fast_beta;
  params.slow_beta = 0.98f;
  params.fast_weight = fast_weight;
  params.beta2 = 0.9f;
  params.weight_decay = 0.02f;
  params.num_maps = num_maps;
  params.red_dim = red_dim;
  params.nesterov = nesterov;

  Rng rng;
  const std::vector<float> g_float = RandomVec(total, &rng);
  const std::vector<float> p_float = RandomVec(total, &rng);
  const std::vector<float> g0 = FromStorage(ToStorage(g_float));
  const std::vector<float> p0 = FromStorage(ToStorage(p_float));
  std::vector<float> velocity_ref(static_cast<std::size_t>(2) * total, 0.0f);
  std::vector<float> lane_ref(static_cast<std::size_t>(num_params) * lane_count,
                              0.0f);

  DevBuf<ComputeType> grads(ToStorage(g0));
  DevBuf<float> master(p0);
  DevBuf<ComputeType> params_dev(ToStorage(p0));
  DevBuf<float> velocity(velocity_ref);
  DevBuf<float> lane(lane_ref);

  std::vector<float> p_ref = p0;

  for (int s = 0; s < steps; ++s) {
    nanochat::kernels::AnvilUpdate(params, grads.ptr, master.ptr,
                                   params_dev.ptr, velocity.ptr, lane.ptr);
    nanochat::kernels::Synchronize();
    AnvilUpdate(params, g0, &p_ref, &velocity_ref, &lane_ref);

    char label[128];
    std::snprintf(label, sizeof(label), "%s step %d params", name, s + 1);
    CheckVectorClose(master.Download(), p_ref, kTol, label);
    std::snprintf(label, sizeof(label), "%s step %d velocity", name, s + 1);
    CheckVectorClose(velocity.Download(), velocity_ref, kTol, label);
    std::snprintf(label, sizeof(label), "%s step %d lane", name, s + 1);
    CheckVectorClose(lane.Download(), lane_ref, kTol, label);
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("anvil gpu test\n");

  // Post-engage rails: the fast rail holds the fixed fast beta and the blend
  // reads both rails.
  const float kFastBeta = 0.85f;
  const float kFastWeight = 0.4385f;
  // Tall matrices (rows > cols): A = X^T X, prod = X B.
  RunCase(2, 4, 3, -1, 6, kFastBeta, kFastWeight, true, "tall red=-1", 3);
  RunCase(2, 4, 3, -2, 6, kFastBeta, kFastWeight, true, "tall red=-2", 3);
  // Wide matrices (rows < cols): A = X X^T, prod = B X.
  RunCase(2, 3, 6, -1, 6, kFastBeta, kFastWeight, true, "wide red=-1", 3);
  RunCase(2, 3, 6, -2, 6, kFastBeta, kFastWeight, true, "wide red=-2", 3);
  // Square matrices take the wide branch with a square A.
  RunCase(1, 4, 4, -1, 6, kFastBeta, kFastWeight, true, "square", 2);
  // A short map count exercises the cascade boundary.
  RunCase(2, 4, 3, -1, 3, kFastBeta, kFastWeight, true, "maps=3", 2);
  RunCase(2, 4, 3, -1, 0, kFastBeta, kFastWeight, true, "maps=0", 2);
  // Pre-engage rails: the fast rail alone, on the scheduled beta, weight 1.
  RunCase(2, 4, 3, -1, 6, 0.89f, 1.0f, true, "pre-engage", 2);
  // Nesterov off.
  RunCase(2, 4, 3, -1, 6, kFastBeta, kFastWeight, false, "no-nesterov", 2);

  // rows/cols <= 0 is a no-op.
  {
    DevBuf<ComputeType> one(1);
    DevBuf<float> one_master(1);
    DevBuf<float> onef(1);
    DevBuf<float> onef2(2);
    AnvilParams bad;
    bad.rows = 0;
    bad.cols = 0;
    nanochat::kernels::AnvilUpdate(bad, one.ptr, one_master.ptr, one.ptr,
                                   onef2.ptr, onef.ptr);
    nanochat::kernels::Synchronize();
  }

  if (Failures() != 0) {
    std::printf("anvil: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("anvil: all checks passed\n");
  return 0;
}
