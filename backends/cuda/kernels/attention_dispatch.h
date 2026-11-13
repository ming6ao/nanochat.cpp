#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_DISPATCH_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_DISPATCH_H_

// The dispatch predicates of the Attention forward, in one host-only header so
// that `attention.cu` and `attention_test.cc` answer the same question from the
// same code. `attention.cu` owns the kernels and the branch order; this header
// owns only the shape conditions. See `docs/attention.md` Part 4, Part 6, and
// Part 7, section 5.
//
// Four paths exist:
//   1. the flash family, one forward tile and two backward kernels on
//      `mma.sync`, in a bf16 or fp8 build on a device with tensor cores
//      (Part 7);
//   2. the fp32 fused tile, on a build whose matmuls have no tensor cores to
//      lose to (Part 4);
//   3. the bf16 tensor-core `wmma` tile, for the shapes the flash family leaves
//      to it (Part 6);
//   4. the cuBLAS decomposition, for every other shape.
//
// The fp32 tile reads the query in the model stream, so a head-major query
// keeps the batched path. The flash family reads either layout.

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

// Whether the flash family covers this shape (`docs/attention.md` Part 7,
// section 5). It replaces both earlier tiles for a bf16 build on a device with
// tensor cores: it reads the head-major query the fused `qk_prep` writes, it
// keeps the window in the mask so a full-causal layer takes it too, and it
// covers the two head widths the tree runs. A decode step keeps cuBLAS,
// because `kv_len > seq` needs the cache path and a different tile split.
inline bool FusedAttentionFlashCovers(const AttentionParams& p,
                                      bool tensor_cores) {
  const int width = p.head_dim;
  return kComputeIsBf16 && tensor_cores && (width == 64 || width == 128) &&
         p.window_right == 0 && p.kv_len <= p.seq;
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
