#ifndef NANOCHAT_TESTS_PARAM_STAGE_H_
#define NANOCHAT_TESTS_PARAM_STAGE_H_

// Test-only helper for writing fixture parameters into a model.
//
// The canonical persistent weight is the fp32 master, and the compute copy is
// derived from it (docs/precision.md section 5). A fixture carries fp32 values,
// so a test writes the master and then asks the model to rebuild the compute
// copy with `Model::SyncCompute()`. In the fp32 build the master aliases the
// compute copy, so the write is the same write and the sync is a no-op.
//
// This header is part of the `//tests:oracle_fixture` library, so the fixture
// readers that need it already depend on it.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace testutil {

// True when this build stores compute tensors as fp32.
//
// A test that compares against a committed fp32 PyTorch golden runs only in
// this build. The bf16 build rounds every stored activation and weight to 7
// mantissa bits, so it cannot reproduce an fp32 golden. The bf16 parity gate is
// the separate bf16 numeric trace (docs/precision.md section 6, Stage 2).
#if defined(NANOCHAT_PRECISION_FP32)
inline constexpr bool kFp32Compute = true;
#else
inline constexpr bool kFp32Compute = false;
#endif

// Returns true when a golden-fixture parity test must stop without running.
// The skip is printed, so a bf16 run records why the leg was not exercised.
inline bool SkipFp32Golden(const char* test_name) {
  if (kFp32Compute) return false;
  std::printf(
      "%s: skipped (bf16 build; the committed fixture is an fp32 golden)\n",
      test_name);
  return true;
}

// Writes `values` (length `view.count`) into the parameter's fp32 master.
// The caller calls `model->SyncCompute()` once after the last parameter.
inline void StageParameter(const ParamView& view, const float* values) {
  if (view.master == nullptr || view.count <= 0 || values == nullptr) return;
  kernels::Memcpy(view.master, values,
                  static_cast<std::size_t>(view.count) * sizeof(float),
                  CopyDir::kHostToDevice);
}

}  // namespace testutil
}  // namespace nanochat

#endif  // NANOCHAT_TESTS_PARAM_STAGE_H_
