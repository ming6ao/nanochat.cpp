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
// Numeric detail: PyTorch's `F.rms_norm` defaults its epsilon to
// `torch.finfo(dtype).eps` (1.19e-7 for fp32), and the fixture was produced
// with the default. The model therefore passes that epsilon explicitly instead
// of the kernel default (1e-6) so the parity is exact.

#include "model_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace {

// Matches torch.finfo(torch.float32).eps, the default of F.rms_norm.
constexpr float kRmsEps = 1.1920928955078125e-07f;
constexpr float kLogitSoftcap = 15.0f;
constexpr float kQkScale = 1.2f;
constexpr int kVeGateChannels = 12;
constexpr int kSmearChannels = 24;

template <typename T>
inline float AsF(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return Fp16ToFloat(value);
  }
}

template <typename T = ComputeType>
inline T ToC(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return Fp16FromFloat(value);
  }
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y = x @ w^T, with x [m, k], w [n, k], y [m, n].
void GemmForward(const ComputeType* x, const ComputeType* w, ComputeType* y,
                 std::int64_t m, std::int64_t k, std::int64_t n,
                 bool accumulate = false) {
  GemmParams params;
  params.m = static_cast<int>(m);
  params.n = static_cast<int>(n);
  params.k = static_cast<int>(k);
  params.alpha = 1.0f;
  params.beta = accumulate ? 1.0f : 0.0f;
  // Weights are stored [n, k] (PyTorch's Linear convention); compute
  // x @ w^T by transposing the second operand.
  params.transpose_b = true;
  kernels::Gemm(GemmMode::kForward, params, x, w, y);
}

// dW[n, k] += dy[m, n]^T @ x[m, k]. Parameter gradients are zeroed once per
// backward, so every weight gradient may accumulate.
void GemmWgrad(const ComputeType* x, const ComputeType* dy, ComputeType* dw,
               std::int64_t m, std::int64_t k, std::int64_t n) {
  GemmParams params;
  params.m = static_cast<int>(n);
  params.n = static_cast<int>(k);
  params.k = static_cast<int>(m);
  params.alpha = 1.0f;
  params.beta = 1.0f;
  params.transpose_a = true;
  kernels::Gemm(GemmMode::kWgrad, params, dy, x, dw);
}

// dx[m, k] = dy[m, n] @ w[n, k].
void GemmDgrad(const ComputeType* dy, const ComputeType* w, ComputeType* dx,
               std::int64_t m, std::int64_t k, std::int64_t n,
               bool accumulate = false) {
  GemmParams params;
  params.m = static_cast<int>(m);
  params.n = static_cast<int>(k);
  params.k = static_cast<int>(n);
  params.alpha = 1.0f;
  params.beta = accumulate ? 1.0f : 0.0f;
  kernels::Gemm(GemmMode::kDgrad, params, dy, w, dx);
}

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
  kernels::Memset(data, 0, static_cast<std::size_t>(count) * sizeof(ComputeType));
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
  layers_.resize(static_cast<std::size_t>(layers));
  lparams_.resize(static_cast<std::size_t>(layers));
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
    lparams_[i].value_embeds = p.value;
    lparams_[i].value_embeds_grad = p.grad;
    lparams_[i].has_ve = true;
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
    LayerParams& lp = lparams_[i];
    char name[64];
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_q.weight", i);
    {
      const Param& p = AddParam(name, static_cast<std::int64_t>(query_dim) * hidden,
                                query_dim, hidden);
      lp.c_q = p.value;
      lp.c_q_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_k.weight", i);
    {
      const Param& p = AddParam(name, static_cast<std::int64_t>(kv_dim) * hidden,
                                kv_dim, hidden);
      lp.c_k = p.value;
      lp.c_k_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_v.weight", i);
    {
      const Param& p = AddParam(name, static_cast<std::int64_t>(kv_dim) * hidden,
                                kv_dim, hidden);
      lp.c_v = p.value;
      lp.c_v_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_proj.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(hidden) * hidden, hidden, hidden);
      lp.c_proj_attn = p.value;
      lp.c_proj_attn_grad = p.grad;
    }
    if (config_.has_value_embedding(i)) {
      std::snprintf(name, sizeof(name), "transformer.h.%d.attn.ve_gate.weight",
                    i);
      const Param& p = AddParam(name, kVeGateChannels * kv_heads, kv_heads,
                                kVeGateChannels);
      lp.ve_gate = p.value;
      lp.ve_gate_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_fc.weight", i);
    {
      const Param& p = AddParam(name,
                                static_cast<std::int64_t>(mlp_dim) * hidden,
                                mlp_dim, hidden);
      lp.c_fc = p.value;
      lp.c_fc_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_proj.weight", i);
    {
      const Param& p = AddParam(name,
                                static_cast<std::int64_t>(hidden) * mlp_dim,
                                hidden, mlp_dim);
      lp.c_proj_mlp = p.value;
      lp.c_proj_mlp_grad = p.grad;
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
    LayerParams& lp = lparams_[i];
    FillUniform(lp.c_q, static_cast<std::int64_t>(config_.query_dim()) * hidden,
                -s, s, &rng);
    FillUniform(lp.c_k, static_cast<std::int64_t>(kv_dim) * hidden, -s, s,
                &rng);
    FillUniform(lp.c_v, static_cast<std::int64_t>(kv_dim) * hidden, -s, s,
                &rng);
    FillZero(lp.c_proj_attn, static_cast<std::int64_t>(hidden) * hidden);
    FillUniform(lp.c_fc, static_cast<std::int64_t>(mlp_dim) * hidden,
                -0.4f * s, 0.4f * s, &rng);
    FillZero(lp.c_proj_mlp, static_cast<std::int64_t>(hidden) * mlp_dim);
    if (lp.has_ve) {
      FillUniform(lp.value_embeds,
                  static_cast<std::int64_t>(padded_vocab) * kv_dim, -s, s,
                  &rng);
      FillUniform(lp.ve_gate, kVeGateChannels * kv_heads, 0.0f, 0.02f, &rng);
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
      const float inv_freq =
          1.0f / std::pow(config_.rope_base, exponent);
      const float angle = static_cast<float>(t) * inv_freq;
      cos_table_[static_cast<std::size_t>(t) * half + d] = std::cos(angle);
      sin_table_[static_cast<std::size_t>(t) * half + d] = std::sin(angle);
    }
  }
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
    LayerActs& la = layers_[i];
    c(&la.xr, rows * hidden);
    c(&la.h, rows * hidden);
    f(&la.rstd1, rows);
    c(&la.q_pre, rows * query_dim);
    c(&la.k_pre, rows * kv_dim);
    c(&la.q_final, rows * query_dim);
    c(&la.k_final, rows * kv_dim);
    c(&la.ve_values, rows * kv_dim);
    c(&la.ve_gate, rows * kv_heads);
    c(&la.v_final, rows * kv_dim);
    f(&la.attn_stats, stats);
    c(&la.attn_out, rows * hidden);
    c(&la.x_mid, rows * hidden);
    c(&la.h2, rows * hidden);
    f(&la.rstd2, rows);
    c(&la.pre_act, rows * mlp_dim);
    c(&la.act, rows * mlp_dim);
    c(&la.x_out, rows * hidden);
    c(&la.dq, rows * query_dim);
    c(&la.dk, rows * kv_dim);
    c(&la.dv, rows * kv_dim);
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
  c(&dh_, rows * hidden);
  c(&dxr_, rows * hidden);
  c(&dx_mid_, rows * hidden);
  c(&dh2_, rows * hidden);
  c(&dy_attn_, rows * hidden);
  c(&dact_, rows * mlp_dim);
  c(&dpre_, rows * mlp_dim);
  c(&de_, rows * hidden);
  c(&demb_, rows * hidden);
  c(&dve_, rows * kv_dim);
  c(&x0_acc_, rows * hidden);
  c(&scratch_a_, rows * hidden);
  c(&scratch_b_, rows * hidden);
  f(&rstd_emb_, rows);
  f(&rstd_final_, rows);
  f(&smear_sig_, rows);

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

  BuildRope(seq);
}

// ---------------------------------------------------------------------------
// Scalar gates (smear, ResFormer value gate)
// ---------------------------------------------------------------------------

void TrainModel::SmearForward(int batch, int seq) {
  const int hidden = config_.hidden_dim;
  const float lambda = AsF(smear_lambda_[0]);
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      const std::int64_t dst = (static_cast<std::int64_t>(b) * seq + t) * hidden;
      if (t == 0) {
        std::memcpy(x0_ + dst, emb_norm_ + dst,
                    static_cast<std::size_t>(hidden) * sizeof(ComputeType));
        smear_sig_[b * seq + t] = 0.0f;
        continue;
      }
      const std::int64_t src =
          (static_cast<std::int64_t>(b) * seq + t - 1) * hidden;
      float pre = 0.0f;
      for (int j = 0; j < kSmearChannels; ++j) {
        pre += AsF(emb_norm_[dst + j]) * AsF(smear_gate_[j]);
      }
      const float sig = Sigmoid(pre);
      smear_sig_[b * seq + t] = sig;
      const float gate = lambda * sig;
      for (int j = 0; j < hidden; ++j) {
        x0_[dst + j] = ToC(AsF(emb_norm_[dst + j]) +
                           gate * AsF(emb_norm_[src + j]));
      }
    }
  }
}

void TrainModel::SmearBackward(int batch, int seq, const ComputeType* d_x0) {
  const int hidden = config_.hidden_dim;
  const float lambda = AsF(smear_lambda_[0]);
  const float* w = smear_gate_;

  // Identity path: x0[t] contains e[t] directly.
  std::memcpy(de_, d_x0,
              static_cast<std::size_t>(batch) * seq * hidden *
                  sizeof(ComputeType));
  double dlambda = 0.0;
  for (int b = 0; b < batch; ++b) {
    for (int t = 1; t < seq; ++t) {
      const std::int64_t m = (static_cast<std::int64_t>(b) * seq + t) * hidden;
      const std::int64_t prev =
          (static_cast<std::int64_t>(b) * seq + t - 1) * hidden;
      const float sig = smear_sig_[b * seq + t];
      const float gate = lambda * sig;

      double dgate = 0.0;
      for (int j = 0; j < hidden; ++j) {
        dgate += static_cast<double>(AsF(d_x0[m + j])) *
                 static_cast<double>(AsF(emb_norm_[prev + j]));
      }
      const float dgate_f = static_cast<float>(dgate);
      for (int j = 0; j < hidden; ++j) {
        de_[prev + j] = ToC(AsF(de_[prev + j]) + dgate_f * gate);
      }
      dlambda += static_cast<double>(dgate_f) * sig;
      const float dpre = dgate_f * lambda * sig * (1.0f - sig);
      for (int j = 0; j < kSmearChannels; ++j) {
        smear_gate_grad_[j] =
            ToC(AsF(smear_gate_grad_[j]) + dpre * AsF(emb_norm_[m + j]));
        de_[m + j] =
            ToC(AsF(de_[m + j]) + dpre * AsF(w[j]));
      }
    }
  }
  smear_lambda_grad_[0] =
      ToC(AsF(smear_lambda_grad_[0]) + static_cast<float>(dlambda));
}

void TrainModel::VeGateForward(int layer, std::int64_t rows) {
  LayerActs& la = layers_[layer];
  LayerParams& lp = lparams_[layer];
  const int hidden = config_.hidden_dim;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int kv_dim = kv_heads * head_dim;
  for (std::int64_t m = 0; m < rows; ++m) {
    const ComputeType* hrow = la.h + m * hidden;
    for (int kh = 0; kh < kv_heads; ++kh) {
      const ComputeType* wrow = lp.ve_gate + kh * kVeGateChannels;
      float pre = 0.0f;
      for (int j = 0; j < kVeGateChannels; ++j) {
        pre += AsF(hrow[j]) * AsF(wrow[j]);
      }
      const float gate = 3.0f * Sigmoid(pre);
      la.ve_gate[m * kv_heads + kh] = ToC(gate);
      ComputeType* vrow = la.v_final + m * kv_dim + kh * head_dim;
      const ComputeType* verow = la.ve_values + m * kv_dim + kh * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        vrow[d] = ToC(AsF(vrow[d]) + gate * AsF(verow[d]));
      }
    }
  }
}

void TrainModel::VeGateBackward(int layer, std::int64_t rows) {
  LayerActs& la = layers_[layer];
  LayerParams& lp = lparams_[layer];
  const int hidden = config_.hidden_dim;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int kv_dim = kv_heads * head_dim;
  for (std::int64_t m = 0; m < rows; ++m) {
    const ComputeType* hrow = la.h + m * hidden;
    for (int kh = 0; kh < kv_heads; ++kh) {
      const float gate = AsF(la.ve_gate[m * kv_heads + kh]);
      const ComputeType* dvrow = la.dv + m * kv_dim + kh * head_dim;
      const ComputeType* verow = la.ve_values + m * kv_dim + kh * head_dim;
      ComputeType* dverow = dve_ + m * kv_dim + kh * head_dim;
      double dgate = 0.0;
      for (int d = 0; d < head_dim; ++d) {
        const float dv = AsF(dvrow[d]);
        dgate += static_cast<double>(dv) * static_cast<double>(AsF(verow[d]));
        dverow[d] = ToC(gate * dv);
      }
      // gate = 3*sigmoid(pre) => d(gate)/d(pre) = gate*(1 - gate/3).
      const float dpre =
          static_cast<float>(dgate) * gate * (1.0f - gate / 3.0f);
      for (int j = 0; j < kVeGateChannels; ++j) {
        lp.ve_gate_grad[kh * kVeGateChannels + j] =
            ToC(AsF(lp.ve_gate_grad[kh * kVeGateChannels + j]) +
                dpre * AsF(hrow[j]));
        dh_[m * hidden + j] =
            ToC(AsF(dh_[m * hidden + j]) +
                dpre * AsF(lp.ve_gate[kh * kVeGateChannels + j]));
      }
    }
  }
  kernels::EmbeddingBackward(static_cast<int>(rows), kv_dim, tokens_.data(),
                             dve_, lp.value_embeds_grad);
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
  const int heads = config_.num_heads;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = config_.mlp_dim();
  const int padded_vocab = config_.padded_vocab_size;
  const int vocab = config_.vocab_size;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  RmsNormParams norm;
  norm.rows = static_cast<int>(rows);
  norm.dim = hidden;
  norm.eps = kRmsEps;

  // Embedding -> norm -> smear.
  kernels::EmbeddingForward(static_cast<int>(rows), hidden, tokens_.data(), wte_,
                            emb_raw_);
  kernels::RmsNormForward(norm, emb_raw_, emb_norm_, rstd_emb_);
  SmearForward(batch, seq);

  QkPrepParams qk;
  qk.batch = batch;
  qk.seq = seq;
  qk.num_heads = heads;
  qk.num_kv_heads = kv_heads;
  qk.head_dim = head_dim;
  qk.eps = kRmsEps;
  qk.scale = kQkScale;

  const ComputeType* x = x0_;
  for (int i = 0; i < layers; ++i) {
    LayerActs& la = layers_[i];
    LayerParams& lp = lparams_[i];

    const float alpha = AsF(resid_[i]);
    const float beta = AsF(x0_lambda_[i]);
    kernels::PointwiseForward(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden),
                              x, x0_, alpha, beta, la.xr);
    kernels::RmsNormForward(norm, la.xr, la.h, la.rstd1);

    // Query / key projections. QkPrep is in-place, so the pre-norm rows are
    // copied aside first: they are what makes the normalization statistics
    // recoverable in QkPrepBackward.
    GemmForward(la.h, lp.c_q, la.q_final, rows, hidden, query_dim);
    std::memcpy(la.q_pre, la.q_final,
                static_cast<std::size_t>(rows * query_dim) *
                    sizeof(ComputeType));
    GemmForward(la.h, lp.c_k, la.k_final, rows, hidden, kv_dim);
    std::memcpy(la.k_pre, la.k_final,
                static_cast<std::size_t>(rows * kv_dim) * sizeof(ComputeType));

    // Value projection, then the ResFormer value residual.
    GemmForward(la.h, lp.c_v, la.v_final, rows, hidden, kv_dim);
    if (lp.has_ve) {
      kernels::EmbeddingForward(static_cast<int>(rows), kv_dim, tokens_.data(),
                                lp.value_embeds, la.ve_values);
      VeGateForward(i, rows);
    }

    kernels::QkPrepForward(qk, cos_table_.data(), sin_table_.data(), la.q_final,
                           la.k_final);

    AttentionParams attn;
    attn.batch = batch;
    attn.seq = seq;
    attn.num_heads = heads;
    attn.num_kv_heads = kv_heads;
    attn.head_dim = head_dim;
    attn.causal = true;
    attn.window_left = config_.window_left(i);
    attn.window_right = config_.window_right();
    attn.kv_len = 0;
    attn.scale = 0.0f;
    kernels::AttentionForward(attn, la.q_final, la.k_final, la.v_final,
                              la.attn_out, la.attn_stats);

    // Attention output projection and the residual.
    GemmForward(la.attn_out, lp.c_proj_attn, scratch_a_, rows, hidden, hidden);
    kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                              static_cast<int>(rows * hidden), scratch_a_, la.xr,
                              1.0f, 1.0f, la.x_mid);

    // MLP: norm -> c_fc -> relu^2 -> c_proj.
    kernels::RmsNormForward(norm, la.x_mid, la.h2, la.rstd2);
    GemmForward(la.h2, lp.c_fc, la.pre_act, rows, hidden, mlp_dim);
    kernels::PointwiseForward(PointwiseOp::kReluSquare,
                              static_cast<int>(rows * mlp_dim), la.pre_act,
                              nullptr, 0.0f, 0.0f, la.act);
    GemmForward(la.act, lp.c_proj_mlp, scratch_b_, rows, mlp_dim, hidden);
    kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                              static_cast<int>(rows * hidden), scratch_b_,
                              la.x_mid, 1.0f, 1.0f, la.x_out);

    if (i == backout_layer_) {
      std::memcpy(x_backout_, la.x_out,
                  static_cast<std::size_t>(rows * hidden) *
                      sizeof(ComputeType));
    }
    x = la.x_out;
  }

  // Mid-layer backout, final norm, classifier, softcap, cross-entropy.
  const float backout = AsF(backout_lambda_[0]);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), x, x_backout_,
                            1.0f, -backout, x_final_pre_);
  kernels::RmsNormForward(norm, x_final_pre_, x_final_norm_, rstd_final_);
  GemmForward(x_final_norm_, lm_head_, raw_logits_, rows, hidden, padded_vocab);

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

  RmsNormParams norm;
  norm.rows = static_cast<int>(rows);
  norm.dim = hidden;
  norm.eps = kRmsEps;

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
  kernels::PointwiseForward(PointwiseOp::kScale, static_cast<int>(rows * padded_vocab),
                            dlogits_, nullptr,
                            1.0f / static_cast<float>(rows), 0.0f, dlogits_);

  GemmWgrad(x_final_norm_, dlogits_, lm_head_grad_, rows, hidden, padded_vocab);
  GemmDgrad(dlogits_, lm_head_, dx_final_norm_, rows, hidden, padded_vocab);
  kernels::RmsNormBackward(norm, x_final_pre_, dx_final_norm_, rstd_final_,
                           dx_final_pre_);

  // Backout: x_final_pre = x_{L-1} - backout * x_backout.
  const float backout = AsF(backout_lambda_[0]);
  double dbackout = 0.0;
  for (std::int64_t j = 0; j < rows * hidden; ++j) {
    dbackout += static_cast<double>(AsF(dx_final_pre_[j])) *
                static_cast<double>(AsF(x_backout_[j]));
    dbackout_[j] = ToC(-backout * AsF(dx_final_pre_[j]));
  }
  backout_lambda_grad_[0] =
      ToC(AsF(backout_lambda_grad_[0]) - static_cast<float>(dbackout));

  // Reverse the blocks. g_a_/g_b_ ping-pong: dcur is the gradient w.r.t. the
  // output of the block about to be differentiated, dnext becomes its input
  // gradient.
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
    LayerBackward(i, dcur, dnext, rows);
    std::swap(dcur, dnext);
  }

  // `dcur` is the gradient w.r.t. the post-smear embedding x0. The per-layer
  // x0_lambdas terms were accumulated separately in x0_acc_.
  for (std::int64_t j = 0; j < rows * hidden; ++j) {
    dcur[j] = ToC(AsF(dcur[j]) + AsF(x0_acc_[j]));
  }
  SmearBackward(batch_, seq_, dcur);
  kernels::RmsNormBackward(norm, emb_raw_, de_, rstd_emb_, demb_);
  kernels::EmbeddingBackward(static_cast<int>(rows), hidden, tokens_.data(),
                             demb_, wte_grad_);
}

void TrainModel::LayerBackward(int layer, const ComputeType* dout,
                               ComputeType* dx_in, std::int64_t rows) {
  LayerActs& la = layers_[layer];
  LayerParams& lp = lparams_[layer];
  const int hidden = config_.hidden_dim;
  const int heads = config_.num_heads;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = config_.mlp_dim();

  RmsNormParams norm;
  norm.rows = static_cast<int>(rows);
  norm.dim = hidden;
  norm.eps = kRmsEps;

  // --- MLP: x_out = x_mid + c_proj(relu^2(c_fc(h2))). ---
  GemmWgrad(la.act, dout, lp.c_proj_mlp_grad, rows, mlp_dim, hidden);
  GemmDgrad(dout, lp.c_proj_mlp, dact_, rows, mlp_dim, hidden);
  kernels::PointwiseBackward(PointwiseOp::kReluSquare,
                             static_cast<int>(rows * mlp_dim), la.pre_act,
                             nullptr, dact_, 0.0f, 0.0f, dpre_, nullptr);
  GemmWgrad(la.h2, dpre_, lp.c_fc_grad, rows, hidden, mlp_dim);
  GemmDgrad(dpre_, lp.c_fc, dh2_, rows, hidden, mlp_dim);
  kernels::RmsNormBackward(norm, la.x_mid, dh2_, la.rstd2, dx_mid_);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), dout, dx_mid_,
                            1.0f, 1.0f, dx_mid_);

  // --- Attention: x_mid = xr + c_proj(attention(q, k, v)). ---
  GemmWgrad(la.attn_out, dx_mid_, lp.c_proj_attn_grad, rows, hidden, hidden);
  GemmDgrad(dx_mid_, lp.c_proj_attn, dy_attn_, rows, hidden, hidden);

  AttentionParams attn;
  attn.batch = batch_;
  attn.seq = seq_;
  attn.num_heads = heads;
  attn.num_kv_heads = kv_heads;
  attn.head_dim = head_dim;
  attn.causal = true;
  attn.window_left = config_.window_left(layer);
  attn.window_right = config_.window_right();
  attn.kv_len = 0;
  attn.scale = 0.0f;
  kernels::AttentionBackward(attn, la.q_final, la.k_final, la.v_final,
                             la.attn_stats, dy_attn_, la.dq, la.dk, la.dv);

  // Value projection and the ResFormer gate.
  GemmWgrad(la.h, la.dv, lp.c_v_grad, rows, hidden, kv_dim);
  GemmDgrad(la.dv, lp.c_v, dh_, rows, hidden, kv_dim, false);
  if (lp.has_ve) {
    VeGateBackward(layer, rows);
  }

  // QK-norm + RoPE + scale, then the query/key projections.
  QkPrepParams qk;
  qk.batch = batch_;
  qk.seq = seq_;
  qk.num_heads = heads;
  qk.num_kv_heads = kv_heads;
  qk.head_dim = head_dim;
  qk.eps = kRmsEps;
  qk.scale = kQkScale;
  kernels::QkPrepBackward(qk, cos_table_.data(), sin_table_.data(), la.dq,
                          la.dk, la.q_pre, la.k_pre);
  GemmWgrad(la.h, la.q_pre, lp.c_q_grad, rows, hidden, query_dim);
  GemmDgrad(la.q_pre, lp.c_q, dh_, rows, hidden, query_dim, true);
  GemmWgrad(la.h, la.k_pre, lp.c_k_grad, rows, hidden, kv_dim);
  GemmDgrad(la.k_pre, lp.c_k, dh_, rows, hidden, kv_dim, true);

  // Pre-attention norm, then the residual blend xr = a*x + b*x0. The residual
  // path x_mid = xr + a contributes dx_mid_ directly to dxr.
  kernels::RmsNormBackward(norm, la.xr, dh_, la.rstd1, dxr_);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), dx_mid_, dxr_,
                            1.0f, 1.0f, dxr_);

  const ComputeType* xin = (layer == 0) ? x0_ : layers_[layer - 1].x_out;
  const float alpha = AsF(resid_[layer]);
  const float beta = AsF(x0_lambda_[layer]);
  double dresid = 0.0;
  double dx0lambda = 0.0;
  for (std::int64_t j = 0; j < rows * hidden; ++j) {
    const float d = AsF(dxr_[j]);
    dresid += static_cast<double>(d) * static_cast<double>(AsF(xin[j]));
    dx0lambda += static_cast<double>(d) * static_cast<double>(AsF(x0_[j]));
    dx_in[j] = ToC(alpha * d);
    x0_acc_[j] = ToC(AsF(x0_acc_[j]) + beta * d);
  }
  resid_grad_[layer] =
      ToC(AsF(resid_grad_[layer]) + static_cast<float>(dresid));
  x0_lambda_grad_[layer] =
      ToC(AsF(x0_lambda_grad_[layer]) + static_cast<float>(dx0lambda));
}

// ---------------------------------------------------------------------------
// Train step, parameter views, checkpointing
// ---------------------------------------------------------------------------

float TrainModel::TrainStep(const int* tokens, const int* targets, int batch,
                            int seq, Optimizer* optimizer) {
  (void)optimizer;  // The optimizer wiring lands in the optimizer workstream.
  const float loss = ForwardLoss(tokens, targets, batch, seq);
  Backward();
  // Minimal, deterministic descent so the graph is a real training step.
  const float lr = 0.02f;
  for (Param& p : params_) {
    for (std::int64_t i = 0; i < p.count; ++i) {
      p.value[i] = ToC(AsF(p.value[i]) - lr * AsF(p.grad[i]));
    }
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
