// The fp32 stand-in of the two card jobs of the hand tensor-core GEMM
// (docs/tensor-core-gemm-plan.md sections B.4 and B.8).
//
// The tile needs bf16 operands, so an fp32 build has nothing to check against
// cuBLAS and nothing to measure. `backends/cuda/gemm/BUILD.bazel` compiles this
// file in place of `gemm_gpu_test.cc` and `gemm_benchmark.cc`, so both targets
// exist in every precision.

#include <cstdio>

int main() {
  std::printf("gemm: the hand tile needs a bf16 build, nothing to run here\n");
  return 0;
}
