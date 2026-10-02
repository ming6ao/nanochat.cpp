#ifndef NANOCHAT_SAMPLER_H_
#define NANOCHAT_SAMPLER_H_

#include <cstdint>

#include "nanochat/rand.h"

// Token sampling for the decode graph (docs/model.md). Sampling runs on the
// host over fp32 logits; the device only produces logits. Vendor-free.

namespace nanochat {

struct SampleParams {
  // Temperature <= 0 selects the argmax (greedy) token.
  float temperature = 1.0f;
  // Keep only the `top_k` highest logits; 0 disables the filter.
  int top_k = 0;
  // Seed for the internal generator when the caller passes no Rand.
  std::uint64_t seed = 42;
};

// Chooses one token id from `logits` (length `vocab`). Applies the top-k filter
// and the temperature, then samples. When `rng` is null the function falls back
// to a deterministic generator seeded from `params.seed`.
int SampleToken(const float* logits, int vocab, const SampleParams& params,
                Rand* rng = nullptr);

}  // namespace nanochat

#endif  // NANOCHAT_SAMPLER_H_
