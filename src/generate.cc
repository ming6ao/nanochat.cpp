// The two inference graphs (docs/model.md): prefill, which runs the trunk over
// a prompt and fills the KV cache, and decode, which attends one token against
// the cache and samples the next id. The topology lives here and in model.cc;
// the reusable pieces come from ops.cc. The cache is host-side storage.
//
// The public `KvCache` type hides the concrete layout, so callers only create a
// cache, hand it to Prefill/Decode, and read its position.

#include "model_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/kernels.h"
#include "nanochat/rand.h"
#include "nanochat/sampler.h"
#include "ops.h"
#include "workspace.h"

namespace nanochat {
namespace {

// Activation slots for one inference pass, sized once for `max_seq` rows and
// reused by every prefill/decode call. The block slots are a single set because
// layers run sequentially.
struct InferenceWorkspace {
  ComputeType* emb_raw = nullptr;
  ComputeType* emb_norm = nullptr;
  ComputeType* x0 = nullptr;
  ComputeType* x = nullptr;
  ComputeType* x_backout = nullptr;
  ComputeType* x_final_pre = nullptr;
  ComputeType* x_final_norm = nullptr;
  float* rstd_emb = nullptr;
  float* rstd_final = nullptr;
  float* smear_sig = nullptr;
  ComputeType* raw_logits = nullptr;
  ops::BlockActivations block;
  Workspace arena;
};

// Host-side key/value cache: one contiguous [max_seq, kv_dim] block per layer
// for keys and values, the smear previous-embedding slot, the current position,
// and the reusable inference workspace.
class KvCacheImpl final : public KvCache {
 public:
  KvCacheImpl(const Config& config, int max_seq)
      : config_(config), max_seq_(max_seq < 1 ? 1 : max_seq) {
    const int layers = config_.num_layers;
    const int hidden = config_.hidden_dim;
    const int kv_dim = config_.kv_dim();
    const std::size_t rows =
        static_cast<std::size_t>(max_seq_) * static_cast<std::size_t>(kv_dim);
    keys_.assign(static_cast<std::size_t>(layers) * rows, ToC(0.0f));
    values_.assign(static_cast<std::size_t>(layers) * rows, ToC(0.0f));
    prev_.assign(static_cast<std::size_t>(hidden), ToC(0.0f));
    BuildWorkspace();
  }

  int pos() const override { return pos_; }
  int capacity() const override { return max_seq_; }

  ComputeType* layer_keys(int layer) {
    return keys_.data() +
           static_cast<std::size_t>(layer) * max_seq_ * config_.kv_dim();
  }
  ComputeType* layer_values(int layer) {
    return values_.data() +
           static_cast<std::size_t>(layer) * max_seq_ * config_.kv_dim();
  }
  ComputeType* prev() { return prev_.data(); }
  bool has_prev() const { return has_prev_; }
  void set_has_prev(bool value) { has_prev_ = value; }
  void set_pos(int value) { pos_ = value; }
  InferenceWorkspace& workspace() { return workspace_; }

 private:
  void BuildWorkspace() {
    const int hidden = config_.hidden_dim;
    const int heads = config_.num_heads;
    const int kv_heads = config_.num_kv_heads;
    const int query_dim = config_.query_dim();
    const int kv_dim = config_.kv_dim();
    const int mlp_dim = config_.mlp_dim();
    const int padded = config_.padded_vocab_size;

    const std::int64_t rows = max_seq_;
    const std::int64_t stats =
        static_cast<std::int64_t>(heads) * max_seq_ * 2;

    struct CSlot {
      ComputeType** dest;
      std::int64_t count;
    };
    struct FSlot {
      float** dest;
      std::int64_t count;
    };
    std::vector<CSlot> compute_slots;
    std::vector<FSlot> float_slots;
    auto c = [&](ComputeType** dest, std::int64_t count) {
      compute_slots.push_back({dest, count});
    };
    auto f = [&](float** dest, std::int64_t count) {
      float_slots.push_back({dest, count});
    };

    InferenceWorkspace& ws = workspace_;
    c(&ws.emb_raw, rows * hidden);
    c(&ws.emb_norm, rows * hidden);
    c(&ws.x0, rows * hidden);
    c(&ws.x, rows * hidden);
    c(&ws.x_backout, rows * hidden);
    c(&ws.x_final_pre, rows * hidden);
    c(&ws.x_final_norm, rows * hidden);
    f(&ws.rstd_emb, rows);
    f(&ws.rstd_final, rows);
    f(&ws.smear_sig, rows);
    c(&ws.raw_logits, rows * padded);

    ops::BlockActivations& a = ws.block;
    c(&a.xr, rows * hidden);
    c(&a.h, rows * hidden);
    f(&a.rstd1, rows);
    c(&a.q_final, rows * query_dim);
    c(&a.k_final, rows * kv_dim);
    c(&a.v_final, rows * kv_dim);
    c(&a.ve_values, rows * kv_dim);
    c(&a.ve_gate, rows * kv_heads);
    f(&a.attn_stats, stats);
    c(&a.attn_out, rows * hidden);
    c(&a.x_mid, rows * hidden);
    c(&a.h2, rows * hidden);
    f(&a.rstd2, rows);
    c(&a.pre_act, rows * mlp_dim);
    c(&a.act, rows * mlp_dim);
    c(&a.scratch_a, rows * hidden);
    c(&a.scratch_b, rows * hidden);
    a.q_pre = nullptr;
    a.k_pre = nullptr;

    std::size_t bytes = 0;
    for (const CSlot& slot : compute_slots) {
      bytes += static_cast<std::size_t>(slot.count) * sizeof(ComputeType);
    }
    for (const FSlot& slot : float_slots) {
      bytes += static_cast<std::size_t>(slot.count) * sizeof(float);
    }
    bytes += 256;  // alignment slack

    ws.arena.Reserve(bytes);
    ws.arena.Reset();
    for (const CSlot& slot : compute_slots) {
      *slot.dest = ws.arena.Alloc<ComputeType>(
          static_cast<std::size_t>(slot.count));
    }
    for (const FSlot& slot : float_slots) {
      *slot.dest =
          ws.arena.Alloc<float>(static_cast<std::size_t>(slot.count));
    }

    // The block writes its output back into the residual stream buffer.
    a.x_out = ws.x;
  }

  Config config_;
  int max_seq_ = 0;
  int pos_ = 0;
  bool has_prev_ = false;
  std::vector<ComputeType> keys_;
  std::vector<ComputeType> values_;
  std::vector<ComputeType> prev_;
  InferenceWorkspace workspace_;
};

// The single inference forward, shared by prefill (T > 1) and decode (T == 1).
// It appends the current key/value rows to `kv`, attends against the whole
// cache, and (optionally) writes the soft-capped last-position logits.
void RunInference(TrainModel* model, const int* tokens, int num_tokens,
                  KvCacheImpl* kv, float* logits_out) {
  const Config& config = model->config();
  const int hidden = config.hidden_dim;
  const int heads = config.num_heads;
  const int kv_heads = config.num_kv_heads;
  const int head_dim = config.head_dim();
  const int padded = config.padded_vocab_size;
  const int vocab = config.vocab_size;
  const int layers = config.num_layers;
  const int half = head_dim / 2;

  const int t0 = kv->pos();
  const int t = num_tokens;
  if (t <= 0) return;
  if (t0 + t > kv->capacity()) {
    std::fprintf(stderr,
                 "nanochat: KV cache overflow: need %d positions, capacity "
                 "%d\n",
                 t0 + t, kv->capacity());
    std::abort();
  }
  const std::int64_t rows = t;

  InferenceWorkspace& ws = kv->workspace();

  // Embedding -> norm.
  kernels::EmbeddingForward(t, hidden, tokens, model->wte(), ws.emb_raw);
  ops::RmsNormForward(rows, hidden, kRmsEps, ws.emb_raw, ws.emb_norm,
                      ws.rstd_emb);

  // Smear. Prefill uses the in-batch previous position; decode uses the cached
  // previous embedding (the "prev embedding slot"). Position 0 of a fresh
  // sequence has no previous token.
  const ComputeType* emb_norm = ws.emb_norm;
  if (t > 1) {
    ops::SmearForward(1, t, hidden, emb_norm, model->smear_gate(),
                      model->smear_lambda(), ws.x0, ws.smear_sig);
    std::memcpy(kv->prev(), emb_norm + (rows - 1) * hidden,
                static_cast<std::size_t>(hidden) * sizeof(ComputeType));
  } else if (kv->has_prev()) {
    const ComputeType* prev = kv->prev();
    float pre = 0.0f;
    for (int j = 0; j < kSmearChannels; ++j) {
      pre += AsF(emb_norm[j]) * AsF(model->smear_gate()[j]);
    }
    const float s = Sigmoid(pre);
    const float gate = model->smear_lambda() * s;
    for (int j = 0; j < hidden; ++j) {
      ws.x0[j] = ToC(AsF(emb_norm[j]) + gate * AsF(prev[j]));
    }
    ws.smear_sig[0] = s;
    std::memcpy(kv->prev(), emb_norm,
                static_cast<std::size_t>(hidden) * sizeof(ComputeType));
  } else {
    std::memcpy(ws.x0, emb_norm,
                static_cast<std::size_t>(rows * hidden) * sizeof(ComputeType));
    std::memcpy(kv->prev(), emb_norm,
                static_cast<std::size_t>(hidden) * sizeof(ComputeType));
  }
  kv->set_has_prev(true);

  // Trunk. The rotary tables are offset to the cache position so a decode token
  // is rotated at its absolute position. Grow the table if inference runs past
  // the last training sequence length.
  model->EnsureRopeCapacity(t0 + t);
  ops::BlockShape shape;
  shape.batch = 1;
  shape.seq = t;
  shape.hidden = hidden;
  shape.num_heads = heads;
  shape.num_kv_heads = kv_heads;
  shape.head_dim = head_dim;
  shape.rms_eps = kRmsEps;
  shape.qk_scale = kQkScale;
  shape.window_right = config.window_right();
  shape.kv_len = t0 + t;
  shape.attn_scale = 0.0f;
  shape.causal = true;

  const float* cos =
      model->cos_table() + static_cast<std::size_t>(t0) * half;
  const float* sin =
      model->sin_table() + static_cast<std::size_t>(t0) * half;

  ops::BlockActivations& a = ws.block;
  const ComputeType* x = ws.x0;
  for (int i = 0; i < layers; ++i) {
    shape.window_left = config.window_left(i);
    a.cache_k = kv->layer_keys(i);
    a.cache_v = kv->layer_values(i);
    a.cache_offset = t0;
    a.attn_k = a.cache_k;
    a.attn_v = a.cache_v;
    ops::BlockForward(shape, model->layer_weights(i), a, tokens, cos, sin, x,
                      ws.x0, model->resid(i), model->x0_lambda(i));
    if (i == layers / 2) {
      std::memcpy(ws.x_backout, a.x_out,
                  static_cast<std::size_t>(rows * hidden) *
                      sizeof(ComputeType));
    }
    x = a.x_out;
  }

  // Backout, final norm, classifier.
  ops::BackoutForward(x, ws.x_backout, model->backout_lambda(), ws.x_final_pre,
                      rows * hidden);
  ops::RmsNormForward(rows, hidden, kRmsEps, ws.x_final_pre, ws.x_final_norm,
                      ws.rstd_final);
  ops::LinearForward(ws.x_final_norm, model->lm_head(), ws.raw_logits, rows,
                     hidden, padded);

  kv->set_pos(t0 + t);

  if (logits_out == nullptr) return;
  // Soft-cap and slice the last position to the real vocabulary.
  const ComputeType* last = ws.raw_logits + (rows - 1) * padded;
  for (int v = 0; v < vocab; ++v) {
    logits_out[v] = kLogitSoftcap * std::tanh(AsF(last[v]) / kLogitSoftcap);
  }
}

}  // namespace

std::unique_ptr<KvCache> CreateKvCache(const Config& config, int max_seq) {
  return std::make_unique<KvCacheImpl>(config, max_seq);
}

void Prefill(Model* model, const int* tokens, int num_tokens, KvCache* kv) {
  if (model == nullptr || kv == nullptr || tokens == nullptr) return;
  RunInference(static_cast<TrainModel*>(model), tokens, num_tokens,
               static_cast<KvCacheImpl*>(kv), nullptr);
}

int Decode(Model* model, int token, KvCache* kv, const SampleParams& params) {
  if (model == nullptr || kv == nullptr) return -1;
  auto* impl = static_cast<TrainModel*>(model);
  auto* cache = static_cast<KvCacheImpl*>(kv);
  std::vector<float> logits(static_cast<std::size_t>(impl->config().vocab_size));
  RunInference(impl, &token, 1, cache, logits.data());
  return SampleToken(logits.data(), impl->config().vocab_size, params, nullptr);
}

void PrefillLogits(Model* model, const int* tokens, int num_tokens, KvCache* kv,
                   float* logits_out) {
  if (model == nullptr || kv == nullptr || tokens == nullptr ||
      logits_out == nullptr) {
    return;
  }
  RunInference(static_cast<TrainModel*>(model), tokens, num_tokens,
               static_cast<KvCacheImpl*>(kv), logits_out);
}

void DecodeLogits(Model* model, int token, KvCache* kv, float* logits_out) {
  if (model == nullptr || kv == nullptr || logits_out == nullptr) return;
  RunInference(static_cast<TrainModel*>(model), &token, 1,
               static_cast<KvCacheImpl*>(kv), logits_out);
}

int SampleToken(const float* logits, int vocab, const SampleParams& params,
                Rand* rng) {
  if (logits == nullptr || vocab <= 0) return -1;

  // Temperature <= 0 selects the argmax (greedy) token.
  if (!(params.temperature > 0.0f)) {
    int best = 0;
    float best_value = logits[0];
    for (int i = 1; i < vocab; ++i) {
      if (logits[i] > best_value) {
        best_value = logits[i];
        best = i;
      }
    }
    return best;
  }

  std::vector<float> scores(logits, logits + vocab);

  // Keep only the `top_k` highest logits.
  const int top_k = params.top_k;
  if (top_k > 0 && top_k < vocab) {
    std::vector<float> sorted(scores);
    std::nth_element(sorted.begin(), sorted.begin() + (top_k - 1), sorted.end(),
                     std::greater<float>());
    const float threshold = sorted[top_k - 1];
    for (int i = 0; i < vocab; ++i) {
      if (scores[i] < threshold) {
        scores[i] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  // Temperature, then a numerically stable softmax.
  const float inv_temp = 1.0f / params.temperature;
  float max_score = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < vocab; ++i) {
    scores[i] *= inv_temp;
    if (scores[i] > max_score) max_score = scores[i];
  }
  double sum = 0.0;
  for (int i = 0; i < vocab; ++i) {
    scores[i] = std::exp(scores[i] - max_score);
    sum += scores[i];
  }

  Rand local(params.seed);
  Rand* generator = rng != nullptr ? rng : &local;
  const float r = generator->NextFloat() * static_cast<float>(sum);
  float cumulative = 0.0f;
  for (int i = 0; i < vocab; ++i) {
    cumulative += scores[i];
    if (r <= cumulative) return i;
  }
  return vocab - 1;
}

}  // namespace nanochat
