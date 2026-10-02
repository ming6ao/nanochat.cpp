#ifndef NANOCHAT_DEV_KERNELS_SEQUENCE_REF_H_
#define NANOCHAT_DEV_KERNELS_SEQUENCE_REF_H_

// Host reference for the sequence kernel families (Attention, Classifier,
// Embedding), ported line-for-line from backends/cpu/kernels.cc. The GPU tests
// compare device results against these functions and finite-difference the
// reference forward. Header-only so every dev/kernels test shares one copy.
//
// See docs/testing.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {
namespace seqref {

// --- Attention ------------------------------------------------------------

// Contiguous-grouping GQA head mapping, matching the CPU reference.
inline int KvHead(int h, int num_heads, int num_kv_heads) {
  if (num_kv_heads <= 0 || num_kv_heads >= num_heads) return h;
  const int group = num_heads / num_kv_heads;
  const int kv = h / (group > 0 ? group : 1);
  return std::min(kv, num_kv_heads - 1);
}

inline bool KeyAllowed(const AttentionParams& p, std::int64_t qpos,
                       std::int64_t j) {
  if (p.causal && j > qpos) return false;
  if (p.window_left >= 0 && qpos - j > p.window_left) return false;
  if (p.window_right >= 0 && j - qpos > p.window_right) return false;
  return true;
}

inline std::vector<float> AttentionForward(const AttentionParams& params,
                                           const std::vector<float>& q,
                                           const std::vector<float>& k,
                                           const std::vector<float>& v,
                                           std::vector<float>* stats) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));
  std::vector<float> out(static_cast<std::size_t>(batch) * seq * heads * dim,
                         0.0f);
  stats->assign(
      static_cast<std::size_t>(batch) * heads * seq * 2, 0.0f);
  std::vector<float> acc(dim, 0.0f);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHead(h, heads, kv_heads);
      for (int t = 0; t < seq; ++t) {
        const std::int64_t qpos = kv_len - seq + t;
        const std::int64_t qbase =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        float row_max = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (std::int64_t j = 0; j < kv_len; ++j) {
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(q[qbase + d]) *
                   static_cast<double>(k[kbase + d]);
          }
          const float score = scale * static_cast<float>(dot);
          if (score > row_max) row_max = score;
          any = true;
        }
        const std::int64_t stat_base =
            ((static_cast<std::int64_t>(b) * heads + h) * seq + t) * 2;
        if (!any) {
          for (int d = 0; d < dim; ++d) out[qbase + d] = 0.0f;
          (*stats)[stat_base] = 0.0f;
          (*stats)[stat_base + 1] = 0.0f;
          continue;
        }
        double sum_exp = 0.0;
        for (int d = 0; d < dim; ++d) acc[d] = 0.0f;
        for (std::int64_t j = 0; j < kv_len; ++j) {
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(q[qbase + d]) *
                   static_cast<double>(k[kbase + d]);
          }
          const double weight = std::exp(
              static_cast<double>(scale * static_cast<float>(dot) - row_max));
          sum_exp += weight;
          for (int d = 0; d < dim; ++d) {
            acc[d] += static_cast<float>(weight * v[kbase + d]);
          }
        }
        const float inv = static_cast<float>(1.0 / sum_exp);
        for (int d = 0; d < dim; ++d) out[qbase + d] = acc[d] * inv;
        (*stats)[stat_base] = row_max;
        (*stats)[stat_base + 1] = static_cast<float>(sum_exp);
      }
    }
  }
  return out;
}

inline void AttentionBackward(const AttentionParams& params,
                              const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              const std::vector<float>& stats,
                              const std::vector<float>& dout,
                              std::vector<float>* dq, std::vector<float>* dk,
                              std::vector<float>* dv) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));

  dq->assign(static_cast<std::size_t>(batch) * seq * heads * dim, 0.0f);
  dk->assign(static_cast<std::size_t>(batch) * kv_len * kv_heads * dim, 0.0f);
  dv->assign(static_cast<std::size_t>(batch) * kv_len * kv_heads * dim, 0.0f);

  std::vector<float> probs(kv_len, 0.0f);
  std::vector<float> dp(kv_len, 0.0f);
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHead(h, heads, kv_heads);
      for (int t = 0; t < seq; ++t) {
        const std::int64_t qpos = kv_len - seq + t;
        const std::int64_t qbase =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        const std::int64_t stat_base =
            ((static_cast<std::int64_t>(b) * heads + h) * seq + t) * 2;
        const float row_max = stats[stat_base];
        const float sum_exp = stats[stat_base + 1];
        if (sum_exp <= 0.0f) continue;
        const float inv = 1.0f / sum_exp;
        float weighted_dp = 0.0f;
        for (std::int64_t j = 0; j < kv_len; ++j) {
          probs[j] = 0.0f;
          dp[j] = 0.0f;
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(q[qbase + d]) *
                   static_cast<double>(k[kbase + d]);
          }
          const float p = std::exp(scale * static_cast<float>(dot) - row_max) *
                          inv;
          probs[j] = p;
          double dpd = 0.0;
          for (int d = 0; d < dim; ++d) {
            dpd += static_cast<double>(dout[qbase + d]) *
                   static_cast<double>(v[kbase + d]);
          }
          dp[j] = static_cast<float>(dpd);
          weighted_dp += p * dp[j];
        }
        for (std::int64_t j = 0; j < kv_len; ++j) {
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          const float ds = probs[j] * (dp[j] - weighted_dp);
          const float dp_scale = ds * scale;
          const float p = probs[j];
          for (int d = 0; d < dim; ++d) {
            (*dq)[qbase + d] += dp_scale * k[kbase + d];
            (*dk)[kbase + d] += dp_scale * q[qbase + d];
            (*dv)[kbase + d] += p * dout[qbase + d];
          }
        }
      }
    }
  }
}

// --- Classifier -----------------------------------------------------------

inline std::vector<float> ClassifierForward(const ClassifierParams& params,
                                            const std::vector<float>& logits,
                                            const std::vector<int>& targets) {
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  const float cap = params.softcap;
  std::vector<float> losses(rows, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const int target = targets[r];
    if (target == params.ignore_index) {
      losses[r] = 0.0f;
      continue;
    }
    const std::size_t base = static_cast<std::size_t>(r) * padded;
    float row_max = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(logits[base + j] / cap);
      if (z > row_max) row_max = z;
    }
    double sum_exp = 0.0;
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(logits[base + j] / cap);
      sum_exp += std::exp(static_cast<double>(z - row_max));
    }
    const float z_target = cap * std::tanh(logits[base + target] / cap);
    losses[r] =
        static_cast<float>(std::log(sum_exp)) + row_max - z_target;
  }
  return losses;
}

inline std::vector<float> ClassifierBackward(const ClassifierParams& params,
                                             const std::vector<float>& logits,
                                             const std::vector<int>& targets) {
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  const float cap = params.softcap;
  std::vector<float> dlogits(static_cast<std::size_t>(rows) * padded, 0.0f);
  std::vector<float> probs(vocab, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * padded;
    const int target = targets[r];
    if (target == params.ignore_index) continue;
    float row_max = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(logits[base + j] / cap);
      probs[j] = z;
      if (z > row_max) row_max = z;
    }
    double sum_exp = 0.0;
    for (int j = 0; j < vocab; ++j) {
      sum_exp += std::exp(static_cast<double>(probs[j] - row_max));
    }
    const float inv = static_cast<float>(1.0 / sum_exp);
    for (int j = 0; j < vocab; ++j) {
      const float t = std::tanh(logits[base + j] / cap);
      const float sech2 = 1.0f - t * t;
      const float p = std::exp(probs[j] - row_max) * inv;
      const float onehot = (j == target) ? 1.0f : 0.0f;
      dlogits[base + j] = (p - onehot) * sech2;
    }
  }
  return dlogits;
}

// --- Embedding ------------------------------------------------------------

inline std::vector<float> EmbeddingForward(int tokens, int dim,
                                           const std::vector<int>& ids,
                                           const std::vector<float>& table) {
  std::vector<float> out(static_cast<std::size_t>(tokens) * dim, 0.0f);
  for (int i = 0; i < tokens; ++i) {
    const std::size_t src = static_cast<std::size_t>(ids[i]) * dim;
    const std::size_t dst = static_cast<std::size_t>(i) * dim;
    for (int d = 0; d < dim; ++d) out[dst + d] = table[src + d];
  }
  return out;
}

// Mirrors the CPU contract: zero only the touched rows, then add. `dtable_init`
// is the incoming (persistent) gradient buffer.
inline std::vector<float> EmbeddingBackward(int tokens, int dim,
                                            const std::vector<int>& ids,
                                            const std::vector<float>& dout,
                                            const std::vector<float>& dtable_init) {
  std::vector<float> dtable = dtable_init;
  for (int i = 0; i < tokens; ++i) {
    float* row = dtable.data() + static_cast<std::size_t>(ids[i]) * dim;
    for (int d = 0; d < dim; ++d) row[d] = 0.0f;
  }
  for (int i = 0; i < tokens; ++i) {
    float* row = dtable.data() + static_cast<std::size_t>(ids[i]) * dim;
    const std::size_t src = static_cast<std::size_t>(i) * dim;
    for (int d = 0; d < dim; ++d) row[d] += dout[src + d];
  }
  return dtable;
}

}  // namespace seqref
}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_SEQUENCE_REF_H_
