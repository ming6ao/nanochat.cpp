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

#include "src/model_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/optim.h"
#include "src/ops.h"

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
  if (count <= 0) return;
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    host[static_cast<std::size_t>(i)] = ToC(stddev * rng->Normal());
  }
  kernels::Memcpy(data, host.data(),
                  static_cast<std::size_t>(count) * sizeof(ComputeType),
                  CopyDir::kHostToDevice);
}

void FillUniform(ComputeType* data, std::int64_t count, float lo, float hi,
                 Rng* rng) {
  if (count <= 0) return;
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    host[static_cast<std::size_t>(i)] = ToC(rng->Uniform(lo, hi));
  }
  kernels::Memcpy(data, host.data(),
                  static_cast<std::size_t>(count) * sizeof(ComputeType),
                  CopyDir::kHostToDevice);
}

void FillZero(ComputeType* data, std::int64_t count) {
  kernels::Memset(data, 0,
                  static_cast<std::size_t>(count) * sizeof(ComputeType));
}

// Reads a single `ComputeType` that may live in device memory (CUDA backend)
// or host memory (CPU backend). Used for the scalar parameters `resid_`,
// `x0_lambda_`, `smear_lambda_`, and `backout_lambda_`.
float ReadHost(const ComputeType* p) {
  ComputeType raw = ToC(0.0f);
  kernels::Memcpy(&raw, p, sizeof(ComputeType), CopyDir::kDeviceToHost);
  return AsF(raw);
}

// Number of targets that are not the ignore index (-1). The classifier writes
// a zero loss for an ignored row and the backward zeroes its gradient, so
// these rows do not enter the mean (docs/post-training.md section 2.1).
std::int64_t CountValidTargets(const int* targets, std::int64_t rows) {
  std::int64_t valid = 0;
  for (std::int64_t i = 0; i < rows; ++i) {
    if (targets[i] != -1) ++valid;
  }
  return valid;
}

// Stages a host float array for a kernel pointer. The CPU reference backend
// reads host memory; the CUDA backend reads a device copy. The CUDA classifier
// entry point synchronizes before it returns, so the buffer is safe to release
// at the end of the scope that owns it.
class StagedRowScale {
 public:
  StagedRowScale(const float* host, std::int64_t count) {
    if (host != nullptr && count > 0) {
      const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
      ptr_ = static_cast<float*>(kernels::Alloc(bytes));
      kernels::Memcpy(ptr_, host, bytes, CopyDir::kHostToDevice);
    }
  }
  ~StagedRowScale() {
    if (ptr_ != nullptr) kernels::Free(ptr_);
  }
  StagedRowScale(const StagedRowScale&) = delete;
  StagedRowScale& operator=(const StagedRowScale&) = delete;

  const float* ptr() const { return ptr_; }

 private:
  float* ptr_ = nullptr;
};

// Copies a device buffer through the host and converts every element. The
// master/compute sync runs only at initialization and at a checkpoint load, so
// the round trip is never on the training path.
template <typename Src, typename Dst, typename Convert>
void ConvertDeviceBuffer(const Src* src, Dst* dst, std::int64_t count,
                         Convert convert) {
  std::vector<Src> input(static_cast<std::size_t>(count));
  kernels::Memcpy(input.data(), src,
                  static_cast<std::size_t>(count) * sizeof(Src),
                  CopyDir::kDeviceToHost);
  std::vector<Dst> output(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    output[static_cast<std::size_t>(i)] =
        convert(input[static_cast<std::size_t>(i)]);
  }
  kernels::Memcpy(dst, output.data(),
                  static_cast<std::size_t>(count) * sizeof(Dst),
                  CopyDir::kHostToDevice);
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
    const Param& p = AddParam("transformer.wte.weight",
                              static_cast<std::int64_t>(padded_vocab) * hidden,
                              padded_vocab, hidden);
    wte_ = p.value;
    wte_grad_ = p.grad;
  }
  for (int i = 0; i < layers; ++i) {
    if (!config_.has_value_embedding(i)) continue;
    char name[64];
    std::snprintf(name, sizeof(name), "value_embeds.%d.weight", i);
    const Param& p =
        AddParam(name, static_cast<std::int64_t>(padded_vocab) * kv_dim,
                 padded_vocab, kv_dim);
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
    const Param& p =
        AddParam("smear_gate.weight", kSmearChannels, 1, kSmearChannels);
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
      const Param& p =
          AddParam(name, static_cast<std::int64_t>(query_dim) * hidden,
                   query_dim, hidden);
      w.c_q = p.value;
      g.c_q_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_k.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(kv_dim) * hidden, kv_dim, hidden);
      w.c_k = p.value;
      g.c_k_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.attn.c_v.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(kv_dim) * hidden, kv_dim, hidden);
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
      const Param& p =
          AddParam(name, kVeGateChannels * kv_heads, kv_heads, kVeGateChannels);
      w.ve_gate = p.value;
      g.ve_gate_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_fc.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(mlp_dim) * hidden, mlp_dim, hidden);
      w.c_fc = p.value;
      g.c_fc_grad = p.grad;
    }
    std::snprintf(name, sizeof(name), "transformer.h.%d.mlp.c_proj.weight", i);
    {
      const Param& p = AddParam(
          name, static_cast<std::int64_t>(hidden) * mlp_dim, hidden, mlp_dim);
      w.c_proj_mlp = p.value;
      g.c_proj_mlp_grad = p.grad;
    }
  }
}

TrainModel::~TrainModel() {
  for (Param& p : params_) {
    if (p.value != nullptr) kernels::Free(p.value);
    if (p.grad != nullptr) kernels::Free(p.grad);
#if defined(NANOCHAT_PRECISION_FP16)
    if (p.master != nullptr) kernels::Free(p.master);
#endif
  }
  if (cos_table_dev_ != nullptr) kernels::Free(cos_table_dev_);
  if (sin_table_dev_ != nullptr) kernels::Free(sin_table_dev_);
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
#if defined(NANOCHAT_PRECISION_FP16)
  // The fp32 master weight survives the fp16 round trip of every update. The
  // fp32 build reaches `value` through the same pointer.
  p.master = static_cast<float*>(
      kernels::Alloc(static_cast<std::size_t>(count) * sizeof(float)));
  kernels::Memset(p.master, 0, static_cast<std::size_t>(count) * sizeof(float));
#else
  p.master = reinterpret_cast<float*>(p.value);
#endif
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

  FillNormal(wte_, static_cast<std::int64_t>(padded_vocab) * hidden, 0.8f,
             &rng);
  FillNormal(lm_head_, static_cast<std::int64_t>(padded_vocab) * hidden, 0.001f,
             &rng);

  const float s = std::sqrt(3.0f) / std::sqrt(static_cast<float>(hidden));
  const float denom = static_cast<float>(std::max(layers - 1, 1));
  // Scalar parameters: build on the host, then upload to the (possibly
  // device) parameter buffers.
  std::vector<ComputeType> resid_host(static_cast<std::size_t>(layers));
  std::vector<ComputeType> x0_host(static_cast<std::size_t>(layers));
  for (int i = 0; i < layers; ++i) {
    resid_host[static_cast<std::size_t>(i)] =
        ToC(1.15f - 0.10f * static_cast<float>(i) / denom);
    x0_host[static_cast<std::size_t>(i)] =
        ToC(0.20f - 0.15f * static_cast<float>(i) / denom);
  }
  kernels::Memcpy(resid_, resid_host.data(),
                  resid_host.size() * sizeof(ComputeType),
                  CopyDir::kHostToDevice);
  kernels::Memcpy(x0_lambda_, x0_host.data(),
                  x0_host.size() * sizeof(ComputeType), CopyDir::kHostToDevice);
  const ComputeType smear_lambda_host = ToC(0.0f);
  const ComputeType backout_lambda_host = ToC(0.2f);
  kernels::Memcpy(smear_lambda_, &smear_lambda_host, sizeof(ComputeType),
                  CopyDir::kHostToDevice);
  kernels::Memcpy(backout_lambda_, &backout_lambda_host, sizeof(ComputeType),
                  CopyDir::kHostToDevice);
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
                static_cast<std::int64_t>(mlp_dim) * hidden, -0.4f * s,
                0.4f * s, &rng);
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
  SyncMaster();
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

void TrainModel::BuildRope(int seq) {
  const int head_dim = config_.head_dim();
  const int half = head_dim / 2;
  const std::size_t needed = static_cast<std::size_t>(seq) * half;
  cos_table_.assign(needed, 0.0f);
  sin_table_.assign(needed, 0.0f);
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
  // The kernels read the tables from device memory, so keep a device copy.
  if (needed > rope_capacity_) {
    if (cos_table_dev_ != nullptr) kernels::Free(cos_table_dev_);
    if (sin_table_dev_ != nullptr) kernels::Free(sin_table_dev_);
    cos_table_dev_ =
        static_cast<float*>(kernels::Alloc(needed * sizeof(float)));
    sin_table_dev_ =
        static_cast<float*>(kernels::Alloc(needed * sizeof(float)));
    rope_capacity_ = needed;
  }
  kernels::Memcpy(cos_table_dev_, cos_table_.data(), needed * sizeof(float),
                  CopyDir::kHostToDevice);
  kernels::Memcpy(sin_table_dev_, sin_table_.data(), needed * sizeof(float),
                  CopyDir::kHostToDevice);
}

void TrainModel::EnsureRopeCapacity(int seq) {
  const int half = config_.head_dim() / 2;
  const std::size_t needed = static_cast<std::size_t>(seq) * half;
  if (needed <= rope_capacity_) return;
  BuildRope(seq);
}

void TrainModel::BuildTrainWorkspace(int batch, int seq) {
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
  const std::int64_t stats = static_cast<std::int64_t>(batch) * heads * seq * 2;

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
    *slot.dest =
        workspace_.Alloc<ComputeType>(static_cast<std::size_t>(slot.count));
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

void TrainModel::BuildEvalWorkspace(int batch, int seq) {
  const int hidden = config_.hidden_dim;
  const int heads = config_.num_heads;
  const int kv_heads = config_.num_kv_heads;
  const int head_dim = config_.head_dim();
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = config_.mlp_dim();
  const int padded_vocab = config_.padded_vocab_size;

  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;
  const std::int64_t stats = static_cast<std::int64_t>(batch) * heads * seq * 2;

  // Keep the classifier logits chunk small so the eval arena fits a large
  // batch (docs/eval.md section 10.6). The cross-entropy term is per row, so
  // the chunk boundary does not change the result.
  const std::int64_t budget = eval_logits_budget_;
  const std::int64_t per_row =
      static_cast<std::int64_t>(padded_vocab) * sizeof(ComputeType);
  std::int64_t chunk = per_row > 0 ? budget / per_row : rows;
  if (chunk < 1) chunk = 1;
  if (chunk > rows) chunk = rows;
  eval_chunk_rows_ = chunk;

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

  c(&eval_emb_raw_, rows * hidden);
  c(&eval_emb_norm_, rows * hidden);
  c(&eval_x0_, rows * hidden);
  c(&eval_x_, rows * hidden);
  c(&eval_x_backout_, rows * hidden);
  c(&eval_x_final_pre_, rows * hidden);
  c(&eval_x_final_norm_, rows * hidden);
  c(&eval_logits_, chunk * padded_vocab);
  c(&eval_losses_, rows);
  f(&eval_rstd_emb_, rows);
  f(&eval_rstd_final_, rows);
  f(&eval_smear_sig_, rows);

  // One block set, reused by every layer. Forward-only callers leave the
  // pre-QkPrep saves null and read no KV cache.
  ops::BlockActivations& a = eval_block_;
  c(&a.xr, rows * hidden);
  c(&a.h, rows * hidden);
  f(&a.rstd1, rows);
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

  eval_workspace_.Reserve(bytes);
  eval_workspace_.Reset();
  for (const CSlot& slot : compute_slots) {
    *slot.dest = eval_workspace_.Alloc<ComputeType>(
        static_cast<std::size_t>(slot.count));
  }
  for (const FSlot& slot : float_slots) {
    *slot.dest =
        eval_workspace_.Alloc<float>(static_cast<std::size_t>(slot.count));
  }

  // The block writes its output back into the single residual-stream buffer.
  // `BlockForward` consumes its input into `xr` before it writes `x_out`, so
  // the input and the output can share the buffer.
  a.attn_k = nullptr;
  a.attn_v = nullptr;
  a.cache_k = nullptr;
  a.cache_v = nullptr;
  a.cache_offset = 0;
  a.x_out = eval_x_;

  BuildRope(seq);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

float TrainModel::ForwardLoss(const int* tokens, const int* targets, int batch,
                              int seq) {
  if (batch <= 0 || seq <= 0) return 0.0f;
  const bool save_for_backward = grad_enabled_;
  if (save_for_backward) {
    if (batch != batch_ || seq != seq_ || workspace_.bytes_reserved() == 0) {
      batch_ = batch;
      seq_ = seq;
      BuildTrainWorkspace(batch, seq);
    }
    tokens_.assign(tokens, tokens + static_cast<std::size_t>(batch) * seq);
    targets_.assign(targets, targets + static_cast<std::size_t>(batch) * seq);
    rows_ = static_cast<std::int64_t>(batch) * seq;
    have_grad_activations_ = true;
  } else if (batch != eval_batch_ || seq != eval_seq_ ||
             eval_workspace_.bytes_reserved() == 0) {
    eval_batch_ = batch;
    eval_seq_ = seq;
    BuildEvalWorkspace(batch, seq);
  }

  RunForward(tokens, targets, batch, seq, save_for_backward);

  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;
  // The reference reduces with `F.cross_entropy(..., ignore_index=-1)` and the
  // default mean, which divides by the non-ignored targets. Pretraining has no
  // ignored target, so the divisor is `rows` there and the numbers are
  // unchanged (docs/post-training.md section 2.1).
  const std::int64_t valid = CountValidTargets(targets, rows);
  const ComputeType* losses = ActiveLosses();
  losses_host_.resize(static_cast<std::size_t>(rows));
  kernels::Memcpy(losses_host_.data(), losses,
                  static_cast<std::size_t>(rows) * sizeof(ComputeType),
                  CopyDir::kDeviceToHost);
  double total = 0.0;
  for (std::int64_t m = 0; m < rows; ++m) {
    total +=
        static_cast<double>(AsF(losses_host_[static_cast<std::size_t>(m)]));
  }
  if (valid == 0) return 0.0f;
  return static_cast<float>(total / static_cast<double>(valid));
}

void TrainModel::RunForward(const int* tokens, const int* targets, int batch,
                            int seq, bool save_for_backward) {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int padded_vocab = config_.padded_vocab_size;
  const int vocab = config_.vocab_size;
  const std::int64_t rows = static_cast<std::int64_t>(batch) * seq;

  // Select the arena set. The training set saves every layer for Backward; the
  // evaluation set reuses one block.
  ComputeType* emb_raw = save_for_backward ? emb_raw_ : eval_emb_raw_;
  ComputeType* emb_norm = save_for_backward ? emb_norm_ : eval_emb_norm_;
  ComputeType* x0 = save_for_backward ? x0_ : eval_x0_;
  ComputeType* x_backout = save_for_backward ? x_backout_ : eval_x_backout_;
  ComputeType* x_final_pre =
      save_for_backward ? x_final_pre_ : eval_x_final_pre_;
  ComputeType* x_final_norm =
      save_for_backward ? x_final_norm_ : eval_x_final_norm_;
  float* rstd_emb = save_for_backward ? rstd_emb_ : eval_rstd_emb_;
  float* rstd_final = save_for_backward ? rstd_final_ : eval_rstd_final_;
  float* smear_sig = save_for_backward ? smear_sig_ : eval_smear_sig_;

  // Embedding -> norm -> smear.
  kernels::EmbeddingForward(static_cast<int>(rows), hidden, tokens, wte_,
                            emb_raw);
  ops::RmsNormForward(rows, hidden, kRmsEps, emb_raw, emb_norm, rstd_emb);
  ops::SmearForward(batch, seq, hidden, emb_norm, smear_gate_,
                    ReadHost(smear_lambda_), x0, smear_sig);

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

  const ComputeType* x = x0;
  for (int i = 0; i < layers; ++i) {
    shape.window_left = config_.window_left(i);
    ops::BlockActivations& acts =
        save_for_backward ? lacts_[static_cast<std::size_t>(i)] : eval_block_;
    ops::BlockForward(shape, lweights_[static_cast<std::size_t>(i)], acts,
                      tokens, cos_table(), sin_table(), x, x0,
                      ReadHost(resid_ + i), ReadHost(x0_lambda_ + i));
    // Capture the mid-layer backout activation in both modes: the eval path
    // reads the buffer too, so gating the copy on `save_for_backward` made the
    // forward read uninitialized memory and moved the bits-per-byte result.
    if (i == backout_layer_) {
      kernels::Memcpy(
          x_backout, acts.x_out,
          static_cast<std::size_t>(rows * hidden) * sizeof(ComputeType),
          CopyDir::kDeviceToDevice);
    }
    x = acts.x_out;
  }

  // Mid-layer backout, final norm, classifier, softcap, cross-entropy.
  ops::BackoutForward(x, x_backout, ReadHost(backout_lambda_), x_final_pre,
                      rows * hidden);
  ops::RmsNormForward(rows, hidden, kRmsEps, x_final_pre, x_final_norm,
                      rstd_final);

  ClassifierParams classifier;
  classifier.vocab_size = vocab;
  classifier.padded_vocab_size = padded_vocab;
  classifier.softcap = kLogitSoftcap;
  classifier.ignore_index = -1;
  if (save_for_backward) {
    classifier.rows = static_cast<int>(rows);
    ops::LinearForward(x_final_norm, lm_head_, raw_logits_, rows, hidden,
                       padded_vocab);
    kernels::ClassifierForward(classifier, raw_logits_, targets, losses_);
  } else {
    // Tile the classifier over row chunks so the logits buffer stays small.
    for (std::int64_t off = 0; off < rows; off += eval_chunk_rows_) {
      std::int64_t chunk = rows - off;
      if (chunk > eval_chunk_rows_) chunk = eval_chunk_rows_;
      classifier.rows = static_cast<int>(chunk);
      ops::LinearForward(x_final_norm + off * hidden, lm_head_, eval_logits_,
                         chunk, hidden, padded_vocab);
      kernels::ClassifierForward(classifier, eval_logits_, targets + off,
                                 eval_losses_ + off);
    }
  }
}

void TrainModel::RequireGradActivations(const char* caller) const {
  if (!grad_enabled_) {
    std::fprintf(stderr,
                 "nanochat: %s called with grad mode disabled; the forward "
                 "did not save activations. Enable grad mode before the "
                 "forward.\n",
                 caller);
    std::abort();
  }
  if (!have_grad_activations_) {
    std::fprintf(stderr, "nanochat: %s called before a training forward.\n",
                 caller);
    std::abort();
  }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void TrainModel::ZeroGrad() {
  for (Param& p : params_) {
    FillZero(p.grad, p.count);
  }
}

void TrainModel::Backward() {
  RequireGradActivations("Backward()");
  ZeroGrad();
  BackwardAccumulate(1.0f);
}

void TrainModel::BackwardAccumulate(float scale) {
  RequireGradActivations("BackwardAccumulate()");
  const std::int64_t rows = rows_;
  if (rows == 0) return;
  const std::int64_t valid = CountValidTargets(targets_.data(), rows);
  if (valid == 0) return;
  BackwardInternal(nullptr, scale, static_cast<float>(valid));
}

void TrainModel::BackwardWeighted(const float* row_weights, float scale) {
  RequireGradActivations("BackwardWeighted()");
  if (row_weights == nullptr) {
    std::fprintf(stderr,
                 "nanochat: BackwardWeighted() called with a null row-weight "
                 "buffer.\n");
    std::abort();
  }
  const std::int64_t rows = rows_;
  if (rows == 0) return;
  const std::int64_t valid = CountValidTargets(targets_.data(), rows);
  if (valid == 0) return;
  BackwardInternal(row_weights, scale, static_cast<float>(valid));
}

void TrainModel::BackwardInternal(const float* row_weights, float scale,
                                  float divisor) {
  const int layers = config_.num_layers;
  const int hidden = config_.hidden_dim;
  const int padded_vocab = config_.padded_vocab_size;
  const int vocab = config_.vocab_size;
  const std::int64_t rows = rows_;
  if (rows == 0) return;

  // `x0_acc_` is per-backward workspace, not a parameter gradient, so it is
  // reset here. The parameter gradients are *not* reset: repeated calls sum,
  // which is what gradient accumulation needs.
  FillZero(x0_acc_, rows * hidden);

  // Classifier backward yields the per-row (sum-reduction) gradient; the loss
  // is a mean over valid targets, so divide by the valid count. A non-null
  // `row_weights` folds `row_weights[r] * scale / divisor` into the
  // classifier's per-row scale; a null pointer keeps the uniform path with one
  // scaling pass.
  ClassifierParams classifier;
  classifier.rows = static_cast<int>(rows);
  classifier.vocab_size = vocab;
  classifier.padded_vocab_size = padded_vocab;
  classifier.softcap = kLogitSoftcap;
  classifier.ignore_index = -1;

  const float* host_row_scale = nullptr;
  std::vector<float> effective;
  if (row_weights != nullptr) {
    effective.resize(static_cast<std::size_t>(rows));
    for (std::int64_t m = 0; m < rows; ++m) {
      effective[static_cast<std::size_t>(m)] =
          row_weights[m] * scale * kDefaultLossScale / divisor;
    }
    host_row_scale = effective.data();
  }
  StagedRowScale staged_row_scale(host_row_scale, rows);
  classifier.row_scale = staged_row_scale.ptr();

  kernels::ClassifierBackward(classifier, raw_logits_, targets_.data(),
                              dlogits_);
  if (row_weights == nullptr) {
    kernels::PointwiseForward(
        PointwiseOp::kScale, static_cast<int>(rows * padded_vocab), dlogits_,
        nullptr, scale * kDefaultLossScale / divisor, 0.0f, dlogits_);
  }

  ops::LinearWgrad(x_final_norm_, dlogits_, lm_head_grad_, rows, hidden,
                   padded_vocab);
  ops::LinearDgrad(dlogits_, lm_head_, dx_final_norm_, rows, hidden,
                   padded_vocab, false);
  ops::RmsNormBackward(rows, hidden, kRmsEps, x_final_pre_, dx_final_norm_,
                       rstd_final_, dx_final_pre_);

  // Backout: x_final_pre = x_{L-1} - backout * x_backout.
  ops::BackoutBackward(dx_final_pre_, x_backout_, ReadHost(backout_lambda_),
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
  kernels::Memcpy(dcur, dx_final_pre_,
                  static_cast<std::size_t>(rows * hidden) * sizeof(ComputeType),
                  CopyDir::kDeviceToDevice);
  for (int i = layers - 1; i >= 0; --i) {
    if (i == backout_layer_) {
      kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                                static_cast<int>(rows * hidden), dcur,
                                dbackout_, 1.0f, 1.0f, dcur);
    }
    shape.window_left = config_.window_left(i);
    const ComputeType* x_in = (i == 0) ? x0_ : lacts_[i - 1].x_out;
    ops::BlockBackward(shape, lweights_[i], lacts_[i], lgrads_[i],
                       block_scratch_, tokens_.data(), cos_table(), sin_table(),
                       ReadHost(resid_ + i), ReadHost(x0_lambda_ + i), x_in,
                       x0_, dcur, dnext, x0_acc_, resid_grad_ + i,
                       x0_lambda_grad_ + i);
    std::swap(dcur, dnext);
  }

  // `dcur` is the gradient w.r.t. the post-smear embedding x0. The per-layer
  // x0_lambdas terms were accumulated separately in x0_acc_.
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), dcur, x0_acc_,
                            1.0f, 1.0f, dcur);
  ops::SmearBackward(batch_, seq_, hidden, emb_norm_, smear_gate_, smear_sig_,
                     ReadHost(smear_lambda_), dcur, smear_gate_grad_,
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
  if (!grad_enabled_) {
    std::fprintf(stderr,
                 "nanochat: TrainStep() called with grad mode disabled.\n");
    std::abort();
  }
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
    view.master = p.master;
    view.count = p.count;
    view.rows = p.rows;
    view.cols = p.cols;
    views.push_back(view);
  }
  return views;
}

void TrainModel::SyncMaster() {
#if defined(NANOCHAT_PRECISION_FP16)
  for (Param& p : params_) {
    if (p.master == nullptr || p.value == nullptr || p.count <= 0) continue;
    ConvertDeviceBuffer(p.value, p.master, p.count,
                        [](ComputeType value) { return AsF(value); });
  }
#endif
}

void TrainModel::SyncCompute() {
#if defined(NANOCHAT_PRECISION_FP16)
  for (Param& p : params_) {
    if (p.master == nullptr || p.value == nullptr || p.count <= 0) continue;
    ConvertDeviceBuffer(p.master, p.value, p.count,
                        [](float value) { return ToC(value); });
  }
#endif
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
    // Save the fp32 master weight: the fp16 compute copy cannot hold the fine
    // updates that a resume must keep.
    kernels::Memcpy(buffer.data(), p.master,
                    static_cast<std::size_t>(p.count) * sizeof(float),
                    CopyDir::kDeviceToHost);
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
      std::vector<ComputeType> raw(static_cast<std::size_t>(limit));
      for (std::int64_t i = 0; i < limit; ++i) {
        raw[static_cast<std::size_t>(i)] =
            ToC(buffer[static_cast<std::size_t>(i)]);
      }
      // Restore the fp32 master, then derive the compute copy from it.
      kernels::Memcpy(p.master, buffer.data(),
                      static_cast<std::size_t>(limit) * sizeof(float),
                      CopyDir::kHostToDevice);
      kernels::Memcpy(p.value, raw.data(),
                      static_cast<std::size_t>(limit) * sizeof(ComputeType),
                      CopyDir::kHostToDevice);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar accessors (device-aware)
// ---------------------------------------------------------------------------

float TrainModel::resid(int layer) const { return ReadHost(resid_ + layer); }

float TrainModel::x0_lambda(int layer) const {
  return ReadHost(x0_lambda_ + layer);
}

float TrainModel::smear_lambda() const { return ReadHost(smear_lambda_); }

float TrainModel::backout_lambda() const { return ReadHost(backout_lambda_); }

}  // namespace nanochat
