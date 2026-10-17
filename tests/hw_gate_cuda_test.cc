// tests/hw_gate_cuda_test.cc -- the standalone hardware gate
// (docs/correctness-gates.md section 4.6).
//
// The training path calls `RunHardwareCorrectnessGate` at the start of a fresh
// run. This target is the same check without a training run around it, so the
// gate can be exercised on a device outside the training loop.
//
// It is `gpu` and `manual`: `tools/nanochat test --gpu
// //tests:hw_gate_cuda_test` holds the broker and supplies a device. On a host
// with no accelerator it prints a skip line.

#include <cstdio>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/hw_gate.h"

int main() {
  nanochat::RequireSandboxOrDie("test");

  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  if (!caps.is_device) {
    std::printf("hw gate: skipped (no accelerator)\n");
    return 0;
  }

  const nanochat::HardwareGateResult gate =
      nanochat::RunHardwareCorrectnessGate();
  if (!gate.passed) {
    std::printf("hw gate: FAILED: %s\n", gate.first_failure.c_str());
    return 1;
  }
  std::printf("hw gate: passed (%d cases on %s)\n", gate.cases,
              caps.device_name);
  return 0;
}
