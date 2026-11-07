#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_MMA_OPS_CUH_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_MMA_OPS_CUH_

// The fp8 matrix multiply-accumulate micro-tile
// (docs/tensor-core-gemm-plan.md section A.4).
//
// `wmma` has no fp8 fragment, and an `mma.sync` instruction has no host
// mapping, so the operation is wrapped here: backend-internal, included only
// from `backends/cuda`, and hidden from the frozen seam.
//
// One arm, the scalar reference, runs on the host and on the device. It decodes
// through `nanochat/fp8.h` and accumulates in fp32, so a card below `sm_89`
// still computes the right answer. The `mma.sync.aligned.m16n8k32` arm for
// `sm_89` and newer belongs to the fp8 kernel campaign, where a device test can
// check it.
//
// The per-tensor scale is not applied here: the seam's alpha and beta carry it.

#include <cstdint>

#include "nanochat/fp8.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace cuda_kernels {

// The 16 by 8 by 32 micro-tile, in shared-memory layout.
constexpr int kFp8MmaM = 16;
constexpr int kFp8MmaN = 8;
constexpr int kFp8MmaK = 32;

// A: (m, k) at `x[m * kFp8MmaK + k]`.
struct Fp8MicroA {
  std::uint8_t x[kFp8MmaM * kFp8MmaK];
};

// B: (k, n) at `x[k * kFp8MmaN + n]`.
struct Fp8MicroB {
  std::uint8_t x[kFp8MmaK * kFp8MmaN];
};

// The accumulator: (m, n) at `x[m * kFp8MmaN + n]`.
struct Fp8MicroC {
  float x[kFp8MmaM * kFp8MmaN];
};

inline float Fp8ByteToFloat(std::uint8_t bits, Fp8Format format) {
  return format == Fp8Format::kE4M3 ? Fp8E4M3Decode(bits) : Fp8E5M2Decode(bits);
}

// `d = a * b + c`, one format per operand, and `d` may be `c`.
inline void Fp8MmaTile(Fp8MicroC* d, const Fp8MicroA& a, const Fp8MicroB& b,
                       const Fp8MicroC& c, Fp8Format format_a,
                       Fp8Format format_b) {
  for (int i = 0; i < kFp8MmaM; ++i) {
    for (int j = 0; j < kFp8MmaN; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < kFp8MmaK; ++k) {
        sum += Fp8ByteToFloat(a.x[i * kFp8MmaK + k], format_a) *
               Fp8ByteToFloat(b.x[k * kFp8MmaN + j], format_b);
      }
      d->x[i * kFp8MmaN + j] = c.x[i * kFp8MmaN + j] + sum;
    }
  }
}

}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_MMA_OPS_CUH_
