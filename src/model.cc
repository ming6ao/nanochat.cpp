// The train graph: a fixed forward pass, a hand-written reverse-order backward
// pass over the saved activations, and one optimizer step. There is no
// autograd engine and no graph IR (DESIGN.md section 6); the topology lives
// here and nowhere else. See docs/model.md and docs/kernels.md.
//
// The forward and backward mirror nanochat's gpt.py exactly:
//   embed -> norm -> smear -> N x (resid blend, QK-norm + RoPE + 1.2 scale,
//   sliding-window GQA with a ResFormer value gate, relu^2 MLP) -> mid-layer
//   backout -> final norm -> lm_head -> softcap -> cross-entropy.
//
// The reusable pieces (Linear, Block, Mlp, ValueResidual, Smear, Backout) live
// in ops.cc; this file only wires them into the training graph and its
// reverse-order backward.

#include "model_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/optim.h"
#include "ops.h"

namespace nanochat {
namespace {

// Deterministic xorshift64* random source for InitWeights.
class Rng {
 public:
  explicit Rng(std::uint64_t seed)
      : state_(seed != 0 ? seed : 0x9e3779b97f4a7c15ull) {}

  std::uint64_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }

  float Uniform01() {
    return static_cast<float>((Next() >> 11) * (1.0 / 9007199254740992.0));
  }

  float Uniform(float lo, float hi) { return lo + (hi - lo) * Uniform01(); }

  float Normal() {
    double u1 = Uniform01();
    if (u1 < 1e-7) u1 = 1e-7;
    const double u2 = Uniform01();
    const double kTwoPi = 6.28318530717958647692;
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                              std::cos(kTwoPi * u2));
  }

 private:
  std::uint64_t state_;
};

void FillNormal(ComputeType* data, std::int64_t count, float stddev, Rng* rng) {
  for (std::int64_t i = 0; i < count; ++i) {
    data[i] = ToC(stddev * rng->Normal());
  }
}

void FillUniform(ComputeType* data, std::int64_t count, float lo, float hi,
                 Rng* rng) {
  for (std::int64_t i = 0; i < count; ++i) {
    data[i] = ToC(rng->Uniform(lo, hi));
  }
}

void FillZero(ComputeType* data, std::int64_t count) {
  kernels::Memset(data, 0,
                  static_cast<std::size_t>(count) * sizeof(ComputeType));
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TrainModel::TrainModel(const Config& config) : Model(config) {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int heads = config_.num_heads;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = config_.mlp_dim();
  const int padded_vocab = config_.padded_vocab_size;

  backout_layer_ = layers / 2;
  lweights_.resize(static_cast<std::size_t>(layers));
  lgrads_.resize(static_cast<std::size_t>(layers));
  lacts_.resize(static_cast<std::size_t>(layers));
  params_.reserve(256);

  // Optimizer grouping order: unembedding, embedding, value embeddings, the
  // per-layer scalars, then the per-block matrices.
  {
    const Param& p = AddParam("lm_head.weight",
                              static_cast<std::int64_t>(padded_vocab) * hidden,
                              padded_vocab, hidden);
    lm_head_ = p.value;
    lm_head_grad_ = p.grad;
  }
  {
    const Param& p = AddParam(
        "transformer.wte.weight",
        static_cast<std::int64_t>(padded_vocab) * hidden, padded_vocab, hidden);
    wte_ = p.value;
    wte_grad_ = p.grad;
  }
  for (int i = 0; i < layers; ++i) {
    if (!config_.has_value_embedding(i)) continue;
    char name[64];
    std::snprintf(name, sizeof(name), "value_embeds.%d.weight", i);
    const Param& p = AddParam(
        name, static_cast<std::int64_t>(padded_vocab) * kv_dim, padded_vocab,
        kv_dim);
    lweights_[i].value_embeds = p.value;
    lgrads_[i].value_embeds_grad = p.grad;
    lweights_[i].has_ve = true;
  }
  {
    const Param& p = AddParam("resid_lambdas", layers, 0, 0);
    resid_ = p.value;
    resid_grad_ = p.grad;
  }
  {
    const Param& p = AddParam("x0_lambdas", layers, 0, 0);
    x0_lambda_ = p.value;
    x0_lambda_grad_ = p.grad;
  }
  {
    const Param& p = AddParam("smear_gate.weight", kSmearChannels, 1,
                              kSmearChannels);
    smear_gate_ = p.value;
    smear_gate_grad_ = p.grad;
  }
  {
    const Param& p = AddParam("smear_lambda", 1, 0, 0);
    smear_lambda_ = p.value;
    smear_lambda_grad_ = p.grad;
  }
  {
    const Param& p = AddParam("backout_lambda", 1, 0, 0);
    backout_lambda_ = p.value;
    backout_lambda_grad_ = p.grad;
  }

  for (int i = 0; i < layers; ++i) {
    ops::BlockWeights& w = lweights_[i];
    ops::BlockGrads& g = lgrads_[i];
    char name[64];
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_q.weight", i);
    {
      const Param& p = AddParam(name,
                                static_cast<std::int64_t>(query_dim) * hidden,
                                query_dim, hidden);
      w.c_q = p.value;
      g.c_q_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_k.weight", i);
    {
      const Param& p = AddParam(name, static_cast<std::int64_t>(kv_dim) * hidden,
                                kv_dim, hidden);
      w.c_k = p.value;
      g.c_k_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_v.weight", i);
    {
      const Param& p = AddParam(name, static_cast<std::int64_t>(kv_dim) * hidden,
                                kv_dim, hidden);
      w.c_v = p.value;
      g.c_v_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_proj.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(hidden) * hidden, hidden, hidden);
      w.c_proj_attn = p.value;
      g.c_proj_attn_grad = p.grad;
    }
    if (config_.has_value_embedding(i)) {
      std::snprintf(name, sizeof(name), "transformer.h.%d.attn.ve_gate.weight",
                    i);
      const Param& p = AddParam(name, kVeGateChannels * kv_heads, kv_heads,
                                kVeGateChannels);
      w.ve_gate = p.value;
      g.ve_gate_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_fc.weight", i);
    {
      const Param& p = AddParam(name,
                                static_cast<std::int64_t>(mlp_dim) * hidden,
                                mlp_dim, hidden);
      w.c_fc = p.value;
      g.c_fc_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_proj.weight", i);
    {
      const Param& p = AddParam(name,
                                static_cast<std::int64_t>(hidden) * mlp_dim,
                                hidden, mlp_dim);
      w.c_proj_mlp = p.value;
      g.c_proj_mlp_grad = p.grad;
    }
  }
}

TrainModel::~TrainModel() {
  for (Param& p : params_) {
    if (p.value != nullptr) kernels::Free(p.value);
    if (p.grad != nullptr) kernels::Free(p.grad);
  }
}

TrainModel::Param& TrainModel::AddParam(const std::string& name,
                                        std::int64_t count, int rows,
                                        int cols) {
  Param p;
  p.name = name;
  p.count = count;
  p.rows = rows;
  p.cols = cols;
  p.value = static_cast<ComputeType*>(
      kernels::Alloc(static_cast<std::size_t>(count) * sizeof(ComputeType)));
  p.grad = static_cast<ComputeType*>(
      kernels::Alloc(static_cast<std::size_t>(count) * sizeof(ComputeType)));
  FillZero(p.value, count);
  FillZero(p.grad, count);
  params_.push_back(std::move(p));
  return params_.back();
}

std::unique_ptr<Model> Model::Create(const Config& config) {
  return std::make_unique<TrainModel>(config);
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void TrainModel::InitWeights(std::uint64_t seed) {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int kv_heads = config_.num_kv_heads;
  const int mlp_dim = config_.mlp_dim();
  const int padded_vocab = config_.padded_vocab_size;
  const int kv_dim = config_.kv_dim();

  Rng rng(seed);

  FillNormal(wte_, static_cast<std::int64_t>(padded_vocab) * hidden, 0.8f, &rng);
  FillNormal(lm_head_, static_cast<std::int64_t>(padded_vocab) * hidden, 0.001f,
             &rng);

  const float s = std::sqrt(3.0f) / std::sqrt(static_cast<float>(hidden));
  const float denom = static_cast<float>(std::max(layers - 1, 1));
  for (int i = 0; i < layers; ++i) {
    resid_[i] = ToC(1.15f - 0.10f * static_cast<float>(i) / denom);
    x0_lambda_[i] = ToC(0.20f - 0.15f * static_cast<float>(i) / denom);
  }
  smear_lambda_[0] = ToC(0.0f);
  backout_lambda_[0] = ToC(0.2f);
  FillUniform(smear_gate_, kSmearChannels, 0.0f, 0.02f, &rng);

  for (int i = 0; i < layers; ++i) {
    ops::BlockWeights& w = lweights_[i];
    FillUniform(const_cast<ComputeType*>(w.c_q),
                static_cast<std::int64_t>(config_.query_dim()) * hidden, -s, s,
                &rng);
    FillUniform(const_cast<ComputeType*>(w.c_k),
                static_cast<std::int64_t>(kv_dim) * hidden, -s, s, &rng);
    FillUniform(const_cast<ComputeType*>(w.c_v),
                static_cast<std::int64_t>(kv_dim) * hidden, -s, s, &rng);
    FillZero(const_cast<ComputeType*>(w.c_proj_attn),
             static_cast<std::int64_t>(hidden) * hidden);
    FillUniform(const_cast<ComputeType*>(w.c_fc),
                static_cast<std::int64_t>(mlp_dim) * hidden, -0.4f * s, 0.4f * s,
                &rng);
    FillZero(const_cast<ComputeType*>(w.c_proj_mlp),
             static_cast<std::int64_t>(hidden) * mlp_dim);
    if (w.has_ve) {
      FillUniform(const_cast<ComputeType*>(w.value_embeds),
                  static_cast<std::int64_t>(padded_vocab) * kv_dim, -s, s,
                  &rng);
      FillUniform(const_cast<ComputeType*>(w.ve_gate),
                  kVeGateChannels * kv_heads, 0.0f, 0.02f, &rng);
    }
  }
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

void TrainModel::BuildRope(int seq) {
  const int head_dim = config_.head_dim();
  const int half = head_dim / 2;
  cos_table_.assign(static_cast<std::size_t>(seq) * half, 0.0f);
  sin_table_.assign(static_cast<std::size_t>(seq) * half, 0.0f);
  for (int t = 0; t < seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float exponent =
          static_cast<float>(2 * d) / static_cast<float>(head_dim);
      const float inv_freq = 1.0f / std::pow(config_.rope_base, exponent);
      const float angle = static_cast<float>(t) * inv_freq;
      cos_table_[static_cast<std::size_t>(t) * half + d] = std::cos(angle);
      sin_table_[static_cast<std::size_t>(t) * half + d] = std::sin(angle);
    }
  }
}

void TrainModel::EnsureRopeCapacity(int seq) {
  const int half = config_.head_dim() / 2;
  const std::size_t needed = static_cast<std::size_t>(seq) * half;
  if (cos_table_.size() >= needed) return;
  BuildRope(seq);
}

void TrainModel::BuildWorkspace(int batch, int seq) {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int heads = config_.num_heads;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = config_.mlp_dim();
  const int padded_vocab = config_.padded_vocab_size;

  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;
  const std::int64_t stats =
      static_cast<std::int64_t>(batch) * heads * seq * 2;

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

  for (int i = 0; i < layers; ++i) {
    ops::BlockActivations& a = lacts_[i];
    c(&a.xr, rows * hidden);
    c(&a.h, rows * hidden);
    f(&a.rstd1, rows);
    c(&a.q_pre, rows * query_dim);
    c(&a.k_pre, rows * kv_dim);
    c(&a.q_final, rows * query_dim);
    c(&a.k_final, rows * kv_dim);
    c(&a.ve_values, rows * kv_dim);
    c(&a.ve_gate, rows * kv_heads);
    c(&a.v_final, rows * kv_dim);
    f(&a.attn_stats, stats);
    c(&a.attn_out, rows * hidden);
    c(&a.x_mid, rows * hidden);
    c(&a.h2, rows * hidden);
    f(&a.rstd2, rows);
    c(&a.pre_act, rows * mlp_dim);
    c(&a.act, rows * mlp_dim);
    c(&a.x_out, rows * hidden);
  }

  c(&emb_raw_, rows * hidden);
  c(&emb_norm_, rows * hidden);
  c(&x0_, rows * hidden);
  c(&x_backout_, rows * hidden);
  c(&x_final_pre_, rows * hidden);
  c(&x_final_norm_, rows * hidden);
  c(&raw_logits_, rows * padded_vocab);
  c(&dlogits_, rows * padded_vocab);
  c(&losses_, rows);
  c(&dx_final_norm_, rows * hidden);
  c(&dx_final_pre_, rows * hidden);
  c(&dbackout_, rows * hidden);
  c(&g_a_, rows * hidden);
  c(&g_b_, rows * hidden);
  c(&de_, rows * hidden);
  c(&demb_, rows * hidden);
  c(&x0_acc_, rows * hidden);
  c(&scratch_a_, rows * hidden);
  c(&scratch_b_, rows * hidden);
  f(&rstd_emb_, rows);
  f(&rstd_final_, rows);
  f(&smear_sig_, rows);

  ops::BlockScratch& sc = block_scratch_;
  c(&sc.dq, rows * query_dim);
  c(&sc.dk, rows * kv_dim);
  c(&sc.dv, rows * kv_dim);
  c(&sc.dh, rows * hidden);
  c(&sc.dxr, rows * hidden);
  c(&sc.dx_mid, rows * hidden);
  c(&sc.dh2, rows * hidden);
  c(&sc.dy_attn, rows * hidden);
  c(&sc.dact, rows * mlp_dim);
  c(&sc.dpre, rows * mlp_dim);
  c(&sc.dve, rows * kv_dim);

  std::size_t bytes = 0;
  for (const CSlot& slot : compute_slots) {
    bytes += static_cast<std::size_t>(slot.count) * sizeof(ComputeType);
  }
  for (const FSlot& slot : float_slots) {
    bytes += static_cast<std::size_t>(slot.count) * sizeof(float);
  }
  bytes += 256;  // alignment slack

  workspace_.Reserve(bytes);
  workspace_.Reset();
  for (const CSlot& slot : compute_slots) {
    *slot.dest = workspace_.Alloc<ComputeType>(
        static_cast<std::size_t>(slot.count));
  }
  for (const FSlot& slot : float_slots) {
    *slot.dest = workspace_.Alloc<float>(static_cast<std::size_t>(slot.count));
  }

  // The two projection scratch slots are shared across layers; every layer's
  // block points at the same arena slots.
  for (int i = 0; i < layers; ++i) {
    lacts_[i].scratch_a = scratch_a_;
    lacts_[i].scratch_b = scratch_b_;
    lacts_[i].attn_k = nullptr;
    lacts_[i].attn_v = nullptr;
  }

  BuildRope(seq);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

float TrainModel::ForwardLoss(const int* tokens, const int* targets, int batch,
                              int seq) {
  if (batch <= 0 || seq <= 0) return 0.0f;
  if (batch != batch_ || seq != seq_ || workspace_.bytes_reserved() == 0) {
    batch_ = batch;
    seq_ = seq;
    BuildWorkspace(batch, seq);
  }
  tokens_.assign(tokens, tokens + static_cast<std::size_t>(batch) * seq);
  targets_.assign(targets, targets + static_cast<std::size_t>(batch) * seq);
  rows_ = static_cast<std::int64_t>(batch) * seq;

  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int padded_vocab = config_.padded_vocab_size;
  const int vocab = config_.vocab_size;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  // Embedding -> norm -> smear.
  kernels::EmbeddingForward(static_cast<int>(rows), hidden, tokens_.data(), wte_,
                            emb_raw_);
  ops::RmsNormForward(rows, hidden, kRmsEps, emb_raw_, emb_norm_, rstd_emb_);
  ops::SmearForward(batch, seq, hidden, emb_norm_, smear_gate_,
                    AsF(smear_lambda_[0]), x0_, smear_sig_);

  ops::BlockShape shape;
  shape.batch = batch;
  shape.seq = seq;
  shape.hidden = hidden;
  shape.num_heads = config_.num_heads;
  shape.num_kv_heads = config_.num_kv_heads;
  shape.head_dim = config_.head_dim();
  shape.rms_eps = kRmsEps;
  shape.qk_scale = kQkScale;
  shape.window_right = config_.window_right();
  shape.kv_len = 0;
  shape.attn_scale = 0.0f;
  shape.causal = true;

  const ComputeType* x = x0_;
  for (int i = 0; i < layers; ++i) {
    shape.window_left = config_.window_left(i);
    ops::BlockForward(shape, lweights_[i], lacts_[i], tokens_.data(),
                      cos_table_.data(), sin_table_.data(), x, x0_,
                      AsF(resid_[i]), AsF(x0_lambda_[i]));
    if (i == backout_layer_) {
      std::memcpy(x_backout_, lacts_[i].x_out,
                  static_cast<std::size_t>(rows * hidden) *
                      sizeof(ComputeType));
    }
    x = lacts_[i].x_out;
  }

  // Mid-layer backout, final norm, classifier, softcap, cross-entropy.
  ops::BackoutForward(x, x_backout_, AsF(backout_lambda_[0]), x_final_pre_,
                      rows * hidden);
  ops::RmsNormForward(rows, hidden, kRmsEps, x_final_pre_, x_final_norm_,
                      rstd_final_);
  ops::LinearForward(x_final_norm_, lm_head_, raw_logits_, rows, hidden,
                     padded_vocab);

  ClassifierParams classifier;
  classifier.rows = static_cast<int>(rows);
  classifier.vocab_size = vocab;
  classifier.padded_vocab_size = padded_vocab;
  classifier.softcap = kLogitSoftcap;
  classifier.ignore_index = -1;
  kernels::ClassifierForward(classifier, raw_logits_, targets_.data(), losses_);

  double total = 0.0;
  for (std::int64_t m = 0; m < rows; ++m) {
    total += static_cast<double>(AsF(losses_[m]));
  }
  return static_cast<float>(total / static_cast<double>(rows));
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void TrainModel::Backward() {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int padded_vocab = config_.padded_vocab_size;
  const int vocab = config_.vocab_size;
  const std::int64_t rows = rows_;
  if (rows == 0) return;

  // Every parameter gradient accumulates over the reverse pass.
  for (Param& p : params_) {
    FillZero(p.grad, p.count);
  }
  FillZero(x0_acc_, rows * hidden);

  // Classifier backward yields the per-row (sum-reduction) gradient; the loss
  // is a batch mean, so scale by 1/rows.
  ClassifierParams classifier;
  classifier.rows = static_cast<int>(rows);
  classifier.vocab_size = vocab;
  classifier.padded_vocab_size = padded_vocab;
  classifier.softcap = kLogitSoftcap;
  classifier.ignore_index = -1;
  kernels::ClassifierBackward(classifier, raw_logits_, targets_.data(),
                              dlogits_);
  kernels::PointwiseForward(PointwiseOp::kScale,
                            static_cast<int>(rows * padded_vocab), dlogits_,
                            nullptr, 1.0f / static_cast<float>(rows), 0.0f,
                            dlogits_);

  ops::LinearWgrad(x_final_norm_, dlogits_, lm_head_grad_, rows, hidden,
                   padded_vocab);
  ops::LinearDgrad(dlogits_, lm_head_, dx_final_norm_, rows, hidden,
                   padded_vocab, false);
  ops::RmsNormBackward(rows, hidden, kRmsEps, x_final_pre_, dx_final_norm_,
                       rstd_final_, dx_final_pre_);

  // Backout: x_final_pre = x_{L-1} - backout * x_backout.
  ops::BackoutBackward(dx_final_pre_, x_backout_, AsF(backout_lambda_[0]),
                       dbackout_, backout_lambda_grad_, rows * hidden);

  // Reverse the blocks. g_a_/g_b_ ping-pong: dcur is the gradient w.r.t. the
  // output of the block about to be differentiated, dnext becomes its input
  // gradient.
  ops::BlockShape shape;
  shape.batch = batch_;
  shape.seq = seq_;
  shape.hidden = hidden;
  shape.num_heads = config_.num_heads;
  shape.num_kv_heads = config_.num_kv_heads;
  shape.head_dim = config_.head_dim();
  shape.rms_eps = kRmsEps;
  shape.qk_scale = kQkScale;
  shape.window_right = config_.window_right();
  shape.kv_len = 0;
  shape.attn_scale = 0.0f;
  shape.causal = true;

  ComputeType* dcur = g_a_;
  ComputeType* dnext = g_b_;
  std::memcpy(dcur, dx_final_pre_,
              static_cast<std::size_t>(rows * hidden) * sizeof(ComputeType));
  for (int i = layers - 1; i >= 0; --i) {
    if (i == backout_layer_) {
      for (std::int64_t j = 0; j < rows * hidden; ++j) {
        dcur[j] = ToC(AsF(dcur[j]) + AsF(dbackout_[j]));
      }
    }
    shape.window_left = config_.window_left(i);
    const ComputeType* x_in = (i == 0) ? x0_ : lacts_[i - 1].x_out;
    ops::BlockBackward(shape, lweights_[i], lacts_[i], lgrads_[i],
                       block_scratch_, tokens_.data(), cos_table_.data(),
                       sin_table_.data(), AsF(resid_[i]),
                       AsF(x0_lambda_[i]), x_in, x0_, dcur, dnext, x0_acc_,
                       resid_grad_ + i, x0_lambda_grad_ + i);
    std::swap(dcur, dnext);
  }

  // `dcur` is the gradient w.r.t. the post-smear embedding x0. The per-layer
  // x0_lambdas terms were accumulated separately in x0_acc_.
  for (std::int64_t j = 0; j < rows * hidden; ++j) {
    dcur[j] = ToC(AsF(dcur[j]) + AsF(x0_acc_[j]));
  }
  ops::SmearBackward(batch_, seq_, hidden, emb_norm_, smear_gate_, smear_sig_,
                     AsF(smear_lambda_[0]), dcur, smear_gate_grad_,
                     smear_lambda_grad_, de_);
  ops::RmsNormBackward(rows, hidden, kRmsEps, emb_raw_, de_, rstd_emb_, demb_);
  kernels::EmbeddingBackward(static_cast<int>(rows), hidden, tokens_.data(),
                             demb_, wte_grad_);
}

// ---------------------------------------------------------------------------
// Train step, parameter views, checkpointing
// ---------------------------------------------------------------------------

float TrainModel::TrainStep(const int* tokens, const int* targets, int batch,
                            int seq, Optimizer* optimizer) {
  if (optimizer != nullptr) optimizer->ZeroGrad();
  const float loss = ForwardLoss(tokens, targets, batch, seq);
  Backward();
  if (optimizer != nullptr) {
    optimizer->Step(++optimizer_step_);
  }
  return loss;
}

std::vector<ParamView> TrainModel::params() const {
  std::vector<ParamView> views;
  views.reserve(params_.size());
  for (const Param& p : params_) {
    ParamView view;
    view.name = p.name.c_str();
    view.value = p.value;
    view.grad = p.grad;
    view.count = p.count;
    view.rows = p.rows;
    view.cols = p.cols;
    views.push_back(view);
  }
  return views;
}

void TrainModel::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) return;
  const char magic[8] = {'N', 'C', 'M', 'D', 'L', '0', '0', '1'};
  out.write(magic, 8);
  const std::uint32_t count = static_cast<std::uint32_t>(params_.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const Param& p : params_) {
    const std::uint16_t name_len = static_cast<std::uint16_t>(p.name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    out.write(p.name.data(), name_len);
    const std::uint64_t elements = static_cast<std::uint64_t>(p.count);
    out.write(reinterpret_cast<const char*>(&elements), sizeof(elements));
    std::vector<float> buffer(static_cast<std::size_t>(p.count));
    for (std::int64_t i = 0; i < p.count; ++i) buffer[i] = AsF(p.value[i]);
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size() * sizeof(float)));
  }
}

void TrainModel::Load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return;
  char magic[8];
  in.read(magic, 8);
  if (!in || std::memcmp(magic, "NCMDL001", 8) != 0) return;
  std::uint32_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  for (std::uint32_t k = 0; k < count; ++k) {
    std::uint16_t name_len = 0;
    in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    std::string name(name_len, '\0');
    in.read(name.data(), name_len);
    std::uint64_t elements = 0;
    in.read(reinterpret_cast<char*>(&elements), sizeof(elements));
    std::vector<float> buffer(static_cast<std::size_t>(elements));
    in.read(reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size() * sizeof(float)));
    if (!in) return;
    for (Param& p : params_) {
      if (p.name != name) continue;
      const std::int64_t limit =
          std::min<std::int64_t>(p.count, static_cast<std::int64_t>(elements));
      for (std::int64_t i = 0; i < limit; ++i) p.value[i] = ToC(buffer[i]);
      break;
    }
  }
}

}  // namespace nanochat
