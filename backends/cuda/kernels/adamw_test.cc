// T1 GPU correctness test for the AdamW family. Runs several sequential steps
// on the device and compares the parameter and both moment buffers against the
// host reference in optim_ref.h, exercising bias correction (varying `step`)
// and decoupled weight decay. Tiny shapes; one multi-block shape for the
// grid-stride path. See docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/optim_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::AdamWParams;
using nanochat::ComputeType;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::optimref::AdamWUpdate;

// Runs `steps` AdamW steps on device and host, keeping both in lockstep, and
// compares the parameter and the two moments after every step.
void RunCase(int n, const AdamWParams& base, const char* name, int steps) {
  Rng rng;
  const std::vector<float> p0 = RandomVec(n, &rng);
  const std::vector<float> g_float = RandomVec(n, &rng);
  const std::vector<float> g0 = FromStorage(ToStorage(g_float));
  std::vector<float> m0(n, 0.0f);
  std::vector<float> v0(n, 0.0f);

  DevBuf<float> master(p0);
  DevBuf<ComputeType> p(ToStorage(p0));
  DevBuf<ComputeType> g(ToStorage(g0));
  DevBuf<float> m(m0);
  DevBuf<float> v(v0);

  std::vector<float> p_ref = p0;
  std::vector<float> m_ref = m0;
  std::vector<float> v_ref = v0;

  for (int s = 0; s < steps; ++s) {
    AdamWParams params = base;
    params.step = s + 1;
    nanochat::kernels::AdamWUpdate(n, params, master.ptr, p.ptr, g.ptr, m.ptr,
                                   v.ptr);
    nanochat::kernels::Synchronize();
    AdamWUpdate(n, params, &p_ref, g0, &m_ref, &v_ref);

    char label[96];
    std::snprintf(label, sizeof(label), "%s step %d p", name, s + 1);
    CheckVectorClose(master.Download(), p_ref, 1e-5, label);
    std::snprintf(label, sizeof(label), "%s step %d m", name, s + 1);
    CheckVectorClose(m.Download(), m_ref, 1e-5, label);
    std::snprintf(label, sizeof(label), "%s step %d v", name, s + 1);
    CheckVectorClose(v.Download(), v_ref, 1e-5, label);
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("adamw gpu test\n");

  AdamWParams base;
  base.lr = 0.01f;
  base.beta1 = 0.9f;
  base.beta2 = 0.999f;
  base.eps = 1e-8f;
  base.weight_decay = 0.0f;
  base.step = 1;

  // Bias correction over a run of steps with no weight decay.
  RunCase(5, base, "plain", 4);

  // Decoupled weight decay.
  AdamWParams decayed = base;
  decayed.weight_decay = 0.1f;
  RunCase(7, decayed, "weight-decay", 3);

  // A shape that spans more than one 256-thread block, so the grid-stride loop
  // wraps.
  RunCase(1000, decayed, "multi-block", 2);

  // n == 0 is a no-op.
  DevBuf<ComputeType> empty(1);
  DevBuf<float> empty_master(1);
  DevBuf<float> emptyf(1);
  nanochat::kernels::AdamWUpdate(0, base, empty_master.ptr, empty.ptr,
                                 empty.ptr, emptyf.ptr, emptyf.ptr);
  nanochat::kernels::Synchronize();

  if (Failures() != 0) {
    std::printf("adamw: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("adamw: all checks passed\n");
  return 0;
}
