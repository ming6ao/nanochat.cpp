#ifndef NANOCHAT_SRC_FP8_POLICY_H_
#define NANOCHAT_SRC_FP8_POLICY_H_

#include <algorithm>
#include <cstdint>

// The fp8 eligibility policy: which matrix multiplies convert to fp8, and the
// scratch one conversion needs (docs/precision.md Part 1, sections 4.3 and
// 6.3, and
// docs/performance-model.md section 5.2).
//
// It is a policy, not a graph piece, so it lives beside the performance model
// that prices the same rule (`performance_model.cc`, `AssignMatrixPrecision`)
// rather than in `ops.h`. `ops.h` includes this header because the graph pieces
// pass the scratch through; the dependency runs one way, and nothing here knows
// about a graph, a tensor, or a kernel.

namespace nanochat {

// The fp8 scratch for one linear call: the two byte operands and the two
// dequantization scales. The workspace owns the memory. A caller that passes a
// null scratch, or a scratch whose pointers are null, uses the dense
// compute-type path.
struct Fp8Scratch {
  std::uint8_t* a_bytes = nullptr;  // the activation operand
  std::uint8_t* b_bytes = nullptr;  // the weight operand
  float* a_scale = nullptr;
  float* b_scale = nullptr;

  bool ready() const {
    return a_bytes != nullptr && b_bytes != nullptr && a_scale != nullptr &&
           b_scale != nullptr;
  }
};

// The fp8 eligibility rule: both dimensions are divisible by 16, and the
// smaller dimension is at least 128.
inline bool Fp8Eligible(std::int64_t rows, std::int64_t cols) {
  return rows > 0 && cols > 0 && rows % 16 == 0 && cols % 16 == 0 &&
         std::min(rows, cols) >= 128;
}

}  // namespace nanochat

#endif  // NANOCHAT_SRC_FP8_POLICY_H_
