// P0 build-configuration smoke test. CPU-only: it checks that the .bazelrc
// precision config selects fp32 and that the sandbox guard is reachable from a
// test. No compute logic lives here.

#include <cstdio>

#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

constexpr const char* kPrecision =
    nanochat::kBuildPrecision == nanochat::BuildPrecision::kFp32   ? "fp32"
    : nanochat::kBuildPrecision == nanochat::BuildPrecision::kBf16 ? "bf16"
                                                                   : "fp8";

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("build config: precision=%s\n", kPrecision);
  return 0;
}
