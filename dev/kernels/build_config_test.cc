// P0 build-configuration smoke test. CPU-only: it checks that the .bazelrc
// precision config selects exactly one precision and that the sandbox guard is
// reachable from a test. No compute logic lives here.

#include <cstdio>

#include "nanochat/sandbox.h"

#if defined(NANOCHAT_PRECISION_FP16) && defined(NANOCHAT_PRECISION_FP32)
#error "exactly one precision must be selected"
#endif
#if !defined(NANOCHAT_PRECISION_FP16) && !defined(NANOCHAT_PRECISION_FP32)
#error "no precision selected; --config=fp32 / --config=fp16 sets it"
#endif

int main() {
  nanochat::RequireSandboxOrDie("test");
#if defined(NANOCHAT_PRECISION_FP16)
  std::printf("build config: precision=fp16\n");
#else
  std::printf("build config: precision=fp32\n");
#endif
  return 0;
}
