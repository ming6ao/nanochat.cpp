#ifndef NANOCHAT_DEV_KERNELS_DECODE_KERNELS_H_
#define NANOCHAT_DEV_KERNELS_DECODE_KERNELS_H_

// Capture-safe local mirrors of the decode-path seam ops, used to build the
// batch-1 `cuBLAS + CUDA Graphs` baseline in dev/kernels. The seam launches on
// the legacy default stream (backends/cuda/device.cu), which CUDA forbids
// capturing, so the graph baseline needs kernels that take an explicit stream.
//
// These are prototypes, not promoted kernels: they reproduce the backend math
// so the graph result can be checked against the seam result. See
// dev/kernels/README.md and DESIGN.md section 4.3.

#include <cuda_runtime.h>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {

void DecodeEmbeddingFwd(cudaStream_t stream, int tokens, int dim,
                        const int* ids_dev, const ComputeType* table,
                        ComputeType* out);

void DecodeRmsNormFwd(cudaStream_t stream, int rows, int dim, float eps,
                      const ComputeType* x, ComputeType* out, float* rstd);

void DecodeQkPrepFwd(cudaStream_t stream, const QkPrepParams& params,
                     const float* cos, const float* sin, ComputeType* q,
                     ComputeType* k);

void DecodeAttentionFwd(cudaStream_t stream, const AttentionParams& params,
                        const ComputeType* q, const ComputeType* k,
                        const ComputeType* v, ComputeType* out, float* stats);

void DecodePointwiseFwd(cudaStream_t stream, PointwiseOp op, int n,
                        const ComputeType* a, const ComputeType* b, float alpha,
                        float beta, ComputeType* out);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_DECODE_KERNELS_H_
