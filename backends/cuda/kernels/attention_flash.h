#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_FLASH_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_FLASH_H_

// What `attention.cu` asks of the flash attention family of
// `docs/attention.md` Part 7: whether this device takes it, and the two
// launches. The family is the device source `attention_flash.cu`, which only a
// bf16 or fp8 build compiles; this header names no vendor type, so a host pass
// reads it.
//
// The other builds link `attention_flash_absent.cc`, which answers the same
// three questions with false and nothing. Both sources define the trio, so
// `attention.cu` holds no guard and cannot tell which one it got
// (docs/simulator.md section 9.2).

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {

// Whether this build and this device take the family for this shape. The shape
// conditions are in `attention_dispatch.h`; this adds the device answer.
bool UseFusedAttentionFlash(const AttentionParams& p);

// Launches the fused forward tile. The caller has already asked
// `UseFusedAttentionFlash`; the stand-in returns without launching anything.
void LaunchFlashAttention(const AttentionParams& params, const ComputeType* q,
                          const ComputeType* k, const ComputeType* v,
                          ComputeType* out, float* stats);

// Launches the two backward kernels. They run in order on the same stream: the
// query-gradient kernel writes the per-row dot product that the key and value
// gradient kernel reads. The caller has already asked
// `UseFusedAttentionFlash`.
void LaunchFlashAttentionBackward(const AttentionParams& params,
                                  const ComputeType* q, const ComputeType* k,
                                  const ComputeType* v, const float* stats,
                                  const ComputeType* dout, ComputeType* dq,
                                  ComputeType* dk, ComputeType* dv);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_ATTENTION_FLASH_H_
