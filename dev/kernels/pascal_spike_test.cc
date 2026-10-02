// T1 GPU correctness test for the P0 toolchain spike. Tiny shapes only, per
// docs/testing.md: this is a gate, not an iteration tool. Host code is
// vendor-free and relies on the wrapper in pascal_spike.cu.

#include <cmath>
#include <cstdio>

#include "nanochat/sandbox.h"
#include "pascal_spike.h"

#if defined(NANOCHAT_PRECISION_FP16) && defined(NANOCHAT_PRECISION_FP32)
#error "exactly one precision must be selected"
#endif
#if !defined(NANOCHAT_PRECISION_FP16) && !defined(NANOCHAT_PRECISION_FP32)
#error "no precision selected; --config=fp32 / --config=fp16 sets it"
#endif

int main() {
  nanochat::RequireSandboxOrDie("test");

#if defined(NANOCHAT_PRECISION_FP16)
  std::printf("pascal_spike: precision fp16\n");
#else
  std::printf("pascal_spike: precision fp32\n");
#endif

  int major = 0;
  int minor = 0;
  if (nanochat::dev::SpikeComputeCapability(&major, &minor) != 0) {
    std::fprintf(stderr, "pascal_spike: no CUDA device available\n");
    return 1;
  }
  std::printf("pascal_spike: device compute capability %d.%d\n", major, minor);

  constexpr int kCount = 64;
  float host_a[kCount];
  float host_b[kCount];
  float host_out[kCount];
  for (int i = 0; i < kCount; ++i) {
    host_a[i] = static_cast<float>(i % 7) - 3.0f;
    host_b[i] = static_cast<float>(i % 5) * 0.5f;
  }

  constexpr float kAlpha = 0.5f;
  constexpr float kBeta = 0.25f;
  const int status = nanochat::dev::SpikeScaleAddHost(
      host_a, host_b, kAlpha, kBeta, host_out, kCount);
  if (status != 0) {
    std::fprintf(stderr, "pascal_spike: kernel failed with CUDA error %d\n",
                 status);
    return 1;
  }

  for (int i = 0; i < kCount; ++i) {
    const float expected = kAlpha * host_a[i] + kBeta * host_b[i];
    if (std::fabs(host_out[i] - expected) > 1e-5f) {
      std::fprintf(stderr,
                   "pascal_spike: mismatch at %d: got %f expected %f\n", i,
                   host_out[i], expected);
      return 1;
    }
  }

  std::printf("pascal_spike: ok, %d elements verified\n", kCount);
  return 0;
}
