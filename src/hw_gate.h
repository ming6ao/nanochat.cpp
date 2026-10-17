#ifndef NANOCHAT_SRC_HW_GATE_H_
#define NANOCHAT_SRC_HW_GATE_H_

// The hardware correctness gate (docs/correctness-gates.md section 4).
//
// The simulator proves the fp8 graph on the CPU. It cannot prove the
// tensor-core arithmetic, the `cublasLt` result, the layout, or the fused
// attention tile. This gate runs those cases on the real device at the start
// of a run, against a host reference, and aborts on a mismatch.
//
// The gate is device-agnostic: it drives the frozen seam and compares against a
// host reference computed from `nanochat/fp8.h`. On the CPU reference it passes
// by construction, so a caller runs it only when the device reports
// `caps.is_device`.
//
// The shapes are tiny (B=2, T=8, hidden=64, vocab=128), so the gate costs under
// a second.

#include <string>

namespace nanochat {

struct HardwareGateResult {
  bool passed = true;
  int cases = 0;
  std::string first_failure;
};

// Runs every case and returns the result. A caller that wants a hard failure
// checks `passed`. The function never aborts; it returns the first failure, so
// the caller can log it and choose the exit code.
HardwareGateResult RunHardwareCorrectnessGate();

}  // namespace nanochat

#endif  // NANOCHAT_SRC_HW_GATE_H_
