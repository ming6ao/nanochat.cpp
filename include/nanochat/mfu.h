#ifndef NANOCHAT_MFU_H_
#define NANOCHAT_MFU_H_

#include <string>

#include "nanochat/config.h"

// Model FLOPs utilization (docs/model.md). The accounting follows nanochat:
// every matmul parameter contributes 6 FLOPs/token for a training step
// (2 forward + 4 backward), plus 12 * heads * head_dim * effective_seq per
// attention layer.

namespace nanochat {

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
