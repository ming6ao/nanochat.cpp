#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_LINEAR_SHAPE_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_LINEAR_SHAPE_H_

// The Linear GEMM shapes of one model shape
// (docs/tensor-core-gemm-plan.md sections B.1 and B.5).
//
// A Linear layer of `in` inputs and `out` outputs, over `rows` token positions,
// issues three calls of the seam from `src/ops.cc`: `LinearForward`,
// `LinearDgrad`, and `LinearWgrad`. The three functions below fill the same
// `GemmParams` those call sites fill, so the shape table and the call sites
// cannot disagree about the contract of one layer.
//
// The layers of a model shape come from `Config`, which is what the pipeline
// profiles are made of (`python/nanochat_cpp/pipeline.py` `PROFILES`).
//
// Header only and precision free, like `gemm_shape.h`: it fills `GemmParams`,
// and the seam settles the leading dimensions and the strides.

#include <vector>

#include "nanochat/config.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {
namespace linear_shape {

// One Linear layer. The weight is `out` by `in`, which is PyTorch's Linear
// convention and the layout `src/ops.cc` reads.
struct Layer {
  const char* name;
  int in;
  int out;
};

// The Linear layers of one model shape, in the order a step issues them. The
// query projection and the attention output projection have the same shape on
// a model whose query width is its hidden width, and the table keeps one row
// for the two.
inline std::vector<Layer> Layers(const Config& config) {
  return {
      {"q", config.hidden_dim, config.query_dim()},
      {"kv", config.hidden_dim, config.kv_dim()},
      {"proj", config.query_dim(), config.hidden_dim},
      {"fc", config.hidden_dim, config.mlp_dim()},
      {"cproj", config.mlp_dim(), config.hidden_dim},
      {"cls", config.hidden_dim, config.padded_vocab_size},
  };
}

// `y[rows, out] = x[rows, in] @ w[out, in]^T`. The weight is transposed, which
// is the one transpose form a Linear forward takes.
inline GemmParams Forward(int rows, int in, int out) {
  GemmParams params;
  params.m = rows;
  params.n = out;
  params.k = in;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.transpose_b = true;
  return params;
}

// `dx[rows, in] = dy[rows, out] @ w[out, in]`, with no transpose.
inline GemmParams Dgrad(int rows, int in, int out) {
  GemmParams params;
  params.m = rows;
  params.n = in;
  params.k = out;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  return params;
}

// `dw[out, in] += dy[rows, out]^T @ x[rows, in]`, accumulated and stored as
// fp32. The store type is not part of `GemmParams`, so a row of the table
// states it through `ResolveGemm`'s `fp32_c`.
inline GemmParams Wgrad(int rows, int in, int out) {
  GemmParams params;
  params.m = out;
  params.n = in;
  params.k = rows;
  params.alpha = 1.0f;
  params.beta = 1.0f;
  params.transpose_a = true;
  return params;
}

}  // namespace linear_shape
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_LINEAR_SHAPE_H_
