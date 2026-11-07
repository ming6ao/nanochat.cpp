// The bf16 stand-in of the two `muon_fused_gemm` prototype jobs.
//
// The prototype takes fp32 operands, so its body names `float` buffers where
// the seam takes `ComputeType` and cannot compile in a bf16 build.
// `dev/kernels/BUILD.bazel` compiles this file in place of the test and the
// benchmark, so both targets build in every precision.

#include <cstdio>

int main() {
  std::printf("muon_fused_gemm: the prototype is fp32-only, nothing to run\n");
  return 0;
}
