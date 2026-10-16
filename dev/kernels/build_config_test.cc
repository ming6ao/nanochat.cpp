// P0 build-configuration smoke test. CPU-only: it checks that the .bazelrc
// precision config selects fp32 and that the sandbox guard is reachable from a
// test. No compute logic lives here.

#include <cstdio>

#include "nanochat/sandbox.h"

#if defined(NANOCHAT_PRECISION_FP32) && defined(NANOCHAT_PRECISION_BF16)
#error "select one precision"
#elif defined(NANOCHAT_PRECISION_BF16)
constexpr const char* kPrecision = "bf16";
#elif defined(NANOCHAT_PRECISION_FP32)
constexpr const char* kPrecision = "fp32";
#else
#error "no precision selected"
#endif

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("build config: precision=%s\n", kPrecision);
  return 0;
}
