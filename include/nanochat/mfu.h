#ifndef NANOCHAT_MFU_H_
#define NANOCHAT_MFU_H_

#include <cstdint>
#include <string>

#include "nanochat/config.h"

// Model FLOPs utilization (docs/model.md). The accounting follows nanochat:
// every matmul parameter contributes 6 FLOPs/token for a training step
// (2 forward + 4 backward), plus 12 * heads * head_dim * effective_seq per
// attention layer.

namespace nanochat {

// Structural parameter counts for the scaling-law plan (docs/python.md
// section 4.1). Each field mirrors one group of `GPT.num_scaling_params` from
// the reference. `total` counts every allocated parameter and equals
// embeddings + transformer_matrices + lm_head + scalars.
struct ParamBreakdown {
  int64_t total;
  int64_t transformer_matrices;
  int64_t lm_head;
  int64_t embeddings;
  int64_t scalars;
};

// Counts the parameters of `config` by group. This is a structural count of
// `Config`, not a walk of runtime parameter names. It lands beside
// `NumMatmulParams` in src/train.cc and mirrors `GPT.num_scaling_params()`.
// `NumMatmulParams` mirrors `GPT.num_matmul_params()`; the two differ, so use
// the right one.
ParamBreakdown CountParams(const Config& config);

// Forward + backward FLOPs for one training token.
double EstimateFlopsPerToken(const Config& config);

// Forward FLOPs to prefill `num_tokens` (causal, windowed).
double EstimatePrefillFlops(const Config& config, int num_tokens);

// Forward FLOPs to decode one token against `context_len` cached tokens.
double EstimateDecodeFlops(const Config& config, int context_len);

// Peak device FLOPs for a known device name, or 0 when unknown. The table is
// best-effort and mirrors nanochat's `get_peak_flops`.
double PeakFlopsForDevice(const std::string& device_name);

// tokens_per_second * flops_per_token / peak_flops, clamped to [0, 1].
double ComputeMfu(const Config& config, double tokens_per_second,
                  double peak_flops);

}  // namespace nanochat

#endif  // NANOCHAT_MFU_H_
