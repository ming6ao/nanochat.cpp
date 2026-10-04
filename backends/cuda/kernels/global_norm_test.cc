// T1 GPU correctness test for the GlobalNorm family: the global L2 norm and
// the in-place clip. Tiny and one multi-block shape; no backward. Compared
// against the host reference in row_ref.h. See docs/testing.md.

#include <cstdio>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/row_ref.h"

namespace {

using nanochat::ComputeType;
using nanochat::dev::CheckClose;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using namespace nanochat::dev::rowref;

void RunCase(const std::vector<float>& g0, float clip, const char* name) {
  const int n = static_cast<int>(g0.size());
  DevBuf<ComputeType> g(ToStorage(g0));
  DevBuf<float> norm(1);
  nanochat::kernels::GlobalNorm(n, clip, g.ptr, norm.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref = g0;
  const float ref_norm = GlobalNorm(n, clip, &ref);
  CheckClose(norm.Download()[0], ref_norm, 1e-4, name);
  CheckVectorClose(FromStorage(g.Download()), ref, 1e-4, name);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("global_norm gpu test\n");

  // {3, 4} has norm 5: clipped when the threshold is below it, untouched when
  // above it, and untouched when the threshold disables clipping.
  RunCase({3.0f, 4.0f}, 2.5f, "clip below");
  RunCase({3.0f, 4.0f}, 10.0f, "clip above");
  RunCase({3.0f, 4.0f}, 0.0f, "clip disabled");
  RunCase({3.0f, 4.0f}, -1.0f, "clip negative");

  // A multi-block reduction: 1000 elements span more than one 256-thread block.
  Rng rng;
  const std::vector<float> big = RandomVec(1000, &rng);
  RunCase(big, 1e-3f, "large clipped");
  RunCase(big, 1e9f, "large unclipped");

  // n == 0 is a no-op and must not launch or read the norm buffer.
  DevBuf<ComputeType> empty(1);
  DevBuf<float> norm(1);
  nanochat::kernels::GlobalNorm(0, 1.0f, empty.ptr, norm.ptr);
  nanochat::kernels::GlobalNorm(0, 1.0f, empty.ptr, nullptr);
  nanochat::kernels::Synchronize();

  if (Failures() != 0) {
    std::printf("global_norm: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("global_norm: all checks passed\n");
  return 0;
}
