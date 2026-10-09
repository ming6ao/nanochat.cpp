// T1 GPU correctness test for the Muon family. Runs several sequential steps
// on the device and compares the stacked parameter, the momentum buffer, and
// the factored second moment against the host reference in optim_ref.h. Covers
// the tall and wide Polar Express branches, both `red_dim` values (column and
// row reduction), Nesterov on/off, and a short iteration count. Tiny shapes.
// See docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/optim_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::MuonParams;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::optimref::MuonUpdate;

// Returns the per-matrix size of buf2 for the CPU reference's `red_dim` rule.
int RedIndex(int rows, int cols, int red_dim) {
  const bool reduce_cols = red_dim == -1 || (red_dim != -2 && rows >= cols);
  return reduce_cols ? rows : cols;
}

void RunCase(int num_params, int rows, int cols, int red_dim, bool nesterov,
             int ns_steps, const char* name, int steps) {
  const int mat = rows * cols;
  const int total = num_params * mat;
  const int red_index = RedIndex(rows, cols, red_dim);

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

  Rng rng;
  const std::vector<float> g_float = RandomVec(total, &rng);
  const std::vector<float> g0 = FromStorage(ToStorage(g_float));
  const std::vector<float> p0 = RandomVec(total, &rng);
  std::vector<float> buf1_ref(total, 0.0f);
  std::vector<float> buf2_ref(num_params * red_index, 0.0f);

  DevBuf<ComputeType> grads(ToStorage(g0));
  DevBuf<float> master(p0);
  DevBuf<ComputeType> params_dev(ToStorage(p0));
  DevBuf<float> buf1(buf1_ref);
  DevBuf<float> buf2(buf2_ref);

  std::vector<float> p_ref = p0;

  for (int s = 0; s < steps; ++s) {
    nanochat::kernels::MuonUpdate(params, grads.ptr, master.ptr, params_dev.ptr,
                                  buf1.ptr, buf2.ptr);
    nanochat::kernels::Synchronize();
    MuonUpdate(params, g0, &p_ref, &buf1_ref, &buf2_ref);

    char label[128];
    std::snprintf(label, sizeof(label), "%s step %d params", name, s + 1);
    CheckVectorClose(master.Download(), p_ref, 1e-5, label);
    std::snprintf(label, sizeof(label), "%s step %d buf1", name, s + 1);
    CheckVectorClose(buf1.Download(), buf1_ref, 1e-5, label);
    std::snprintf(label, sizeof(label), "%s step %d buf2", name, s + 1);
    CheckVectorClose(buf2.Download(), buf2_ref, 1e-5, label);
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("muon gpu test\n");

  // Tall matrices (rows > cols): A = X^T X, prod = X B.
  RunCase(2, 4, 3, -1, true, 5, "tall red=-1", 3);
  RunCase(2, 4, 3, -2, true, 5, "tall red=-2", 3);
  // Wide matrices (rows < cols): A = X X^T, prod = B X.
  RunCase(2, 3, 6, -1, true, 5, "wide red=-1", 3);
  RunCase(2, 3, 6, -2, true, 5, "wide red=-2", 3);
  // Square matrices take the wide branch with a square A.
  RunCase(1, 4, 4, -1, true, 5, "square", 2);
  // Nesterov off and a short iteration count.
  RunCase(2, 4, 3, -1, false, 5, "no-nesterov", 2);
  RunCase(2, 3, 6, -1, true, 3, "ns-steps=3", 2);

  // rows/cols <= 0 is a no-op.
  {
    DevBuf<ComputeType> one(1);
    DevBuf<float> one_master(1);
    DevBuf<float> onef(1);
    MuonParams bad;
    bad.rows = 0;
    bad.cols = 0;
    nanochat::kernels::MuonUpdate(bad, one.ptr, one_master.ptr, one.ptr,
                                  onef.ptr, onef.ptr);
    nanochat::kernels::Synchronize();
  }

  if (Failures() != 0) {
    std::printf("muon: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("muon: all checks passed\n");
  return 0;
}
