#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_DISPATCH_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_DISPATCH_H_

// The dispatch predicates of the Attention forward, in one host-only header so
// that `attention.cu` and `attention_test.cc` answer the same question from the
// same code. `attention.cu` owns the kernels and the branch order; this header
// owns only the shape conditions. See `docs/attention.md` Part 4 and Part 6,
// section 5.
//
// Three paths exist:
//   1. the fp32 fused tile, on a build whose matmuls have no tensor cores to
//      lose to (Part 4);
//   2. the bf16 tensor-core tile, where the tensor-core rate makes the tile win
//      over the dense cuBLAS score matrix (Part 6);
//   3. the cuBLAS decomposition, for every other shape.
//
// Both tiles read the query in the model stream, so a head-major query keeps
// the batched path. Both cover only a windowed shape at head width 128.

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// Whether the fp32 tile covers this shape. A bf16 build never takes it: the
// scalar FMA tile loses to the tensor-core cuBLAS path there.
inline bool FusedAttentionFp32Covers(const AttentionParams& p) {
  return !kComputeIsBf16 && !p.q_head_major && p.window_left >= 0 &&
         p.head_dim == 128;
}

// Whether the bf16 tensor-core tile covers this shape. `tensor_cores` is the
// device answer (`Caps::has_tensor_cores`); the test passes it explicitly so
// that both device classes are covered without hardware.
inline bool FusedAttentionBf16Covers(const AttentionParams& p,
                                     bool tensor_cores) {
  return kComputeIsBf16 && tensor_cores && !p.q_head_major &&
         p.window_left >= 0 && p.head_dim == 128;
}

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_DISPATCH_H_
