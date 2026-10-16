// P0 build-configuration smoke test. CPU-only: it checks that the .bazelrc
// precision config selects fp32 and that the sandbox guard is reachable from a
// test. No compute logic lives here.

#include <cstdio>

#include "nanochat/sandbox.h"

#if !defined(NANOCHAT_PRECISION_FP32)
#error "no precision selected; --config=fp32 sets it"
#endif

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("build config: precision=fp32\n");
  return 0;
}
