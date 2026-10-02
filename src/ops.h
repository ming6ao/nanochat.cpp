#ifndef NANOCHAT_SRC_OPS_H_
#define NANOCHAT_SRC_OPS_H_

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

// Reusable, backend-agnostic graph pieces shared by the training graph
// (`model.cc`) and the inference graphs (`generate.cc`). Every function here is
// a thin, named composition of the frozen kernel seam (`nanochat/kernels.h`);
// none of them knows whether it is running on the CPU reference backend or the
// CUDA backend.
//
// The ops are deliberately value-oriented: they take raw `ComputeType*`
// pointers and plain scalar shapes so the same code drives the train forward,
// the hand-written backward, prefill, and single-token decode. The topology
// (which op runs in which order) lives in `model.cc` and `generate.cc`; this
// header only holds the pieces. See docs/model.md and docs/kernels.md.

namespace nanochat {

// Numeric constants shared by every graph. `kRmsEps` matches
// `torch.finfo(torch.float32).eps`, the default of PyTorch's `F.rms_norm`,
// which is what the oracle fixture was produced with.
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

namespace ops {

// ---------------------------------------------------------------------------
// Linear: y = x @ w^T (PyTorch's `nn.Linear` weight layout).
// ---------------------------------------------------------------------------

void LinearForward(const ComputeType* x, const ComputeType* w, ComputeType* y,
                   std::int64_t rows, std::int64_t in, std::int64_t out,
                   bool accumulate = false);

void LinearWgrad(const ComputeType* x, const ComputeType* dy, ComputeType* dw,
                 std::int64_t rows, std::int64_t in, std::int64_t out);

void LinearDgrad(const ComputeType* dy, const ComputeType* w, ComputeType* dx,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 bool accumulate = false);

// ---------------------------------------------------------------------------
// RMSNorm (no learnable scale).
// ---------------------------------------------------------------------------

void RmsNormForward(std::int64_t rows, std::int64_t dim, float eps,
                    const ComputeType* x, ComputeType* out, float* rstd);

void RmsNormBackward(std::int64_t rows, std::int64_t dim, float eps,
                     const ComputeType* x, const ComputeType* dy,
                     const float* rstd, ComputeType* dx);

// ---------------------------------------------------------------------------
// QK-norm + RoPE + scale, in place on the query/key projections. `cos`/`sin`
// point at the first rotary row for this call (the decode path offsets them by
// the KV-cache position).
// ---------------------------------------------------------------------------

void QkPrepForward(int batch, int seq, int num_heads, int num_kv_heads,
                   int head_dim, float eps, float scale, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k);

void QkPrepBackward(int batch, int seq, int num_heads, int num_kv_heads,
                    int head_dim, float eps, float scale, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k);

// ---------------------------------------------------------------------------
// Causal / sliding-window group-query attention. `kv_len == 0` means
// self-attention over `seq`; otherwise the key/value rows may live in a
// KV cache of `kv_len` positions (the decode path).
// ---------------------------------------------------------------------------

struct AttentionShape {
  int batch = 0;
  int seq = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  bool causal = true;
  int window_left = -1;
  int window_right = 0;
  int kv_len = 0;
  float scale = 0.0f;
};

void AttentionForward(const AttentionShape& shape, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats);

void AttentionBackward(const AttentionShape& shape, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv);

// ---------------------------------------------------------------------------
// ResFormer value residual: `v += 3*sigmoid(h[:12] @ ve_gate^T) * ve`, per
// key/value head. `gate_out` saves the per-head gate for the backward.
// ---------------------------------------------------------------------------

void ValueResidualForward(std::int64_t rows, int hidden, int num_kv_heads,
                          int head_dim, const ComputeType* h,
                          const ComputeType* ve, const ComputeType* gate_w,
                          ComputeType* v, ComputeType* gate_out);

void ValueResidualBackward(std::int64_t rows, int hidden, int num_kv_heads,
                           int head_dim, const ComputeType* h,
                           const ComputeType* ve, const ComputeType* gate_w,
                           const ComputeType* gate, const ComputeType* dv,
                           ComputeType* gate_w_grad, ComputeType* dh,
                           ComputeType* dve);

// ---------------------------------------------------------------------------
// relu^2 MLP: `out = x_mid + c_proj(relu^2(c_fc(h)))`. `scratch` receives the
// c_proj output before the residual is added.
// ---------------------------------------------------------------------------

void MlpForward(const ComputeType* h, const ComputeType* c_fc,
                const ComputeType* c_proj, std::int64_t rows,
                std::int64_t hidden, std::int64_t mlp_dim,
                ComputeType* pre_act, ComputeType* act, ComputeType* scratch,
                const ComputeType* x_mid, ComputeType* out);

void MlpBackward(const ComputeType* h, const ComputeType* c_fc,
                 const ComputeType* c_proj, const ComputeType* pre_act,
                 const ComputeType* act, const ComputeType* dout,
                 std::int64_t rows, std::int64_t hidden, std::int64_t mlp_dim,
                 ComputeType* c_fc_grad, ComputeType* c_proj_grad,
                 ComputeType* dh, ComputeType* dact, ComputeType* dpre);

// ---------------------------------------------------------------------------
// Smear: mix the previous token's normalized embedding into the current one
// with a scalar-gated bigram term. Position 0 has no previous token.
// ---------------------------------------------------------------------------

void SmearForward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                  const ComputeType* emb_norm, const ComputeType* gate_w,
                  float lambda, ComputeType* x0, float* sig);

void SmearBackward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                   const ComputeType* emb_norm, const ComputeType* gate_w,
                   const float* sig, float lambda, const ComputeType* d_x0,
                   ComputeType* gate_w_grad, ComputeType* lambda_grad,
                   ComputeType* de);

// ---------------------------------------------------------------------------
// Backout: subtract a scaled mid-layer residual before the final norm.
// ---------------------------------------------------------------------------

void BackoutForward(const ComputeType* x, const ComputeType* x_backout,
                    float lambda, ComputeType* out, std::int64_t n);

void BackoutBackward(const ComputeType* d_pre, const ComputeType* x_backout,
                     float lambda, ComputeType* dbackout,
                     ComputeType* lambda_grad, std::int64_t n);

// ---------------------------------------------------------------------------
// A full transformer block (residual blend -> attention -> MLP). `BlockForward`
// is the only place the block topology is written down; both the training
// forward and the inference graphs call it.
// ---------------------------------------------------------------------------

struct BlockShape {
  int batch = 0;
  int seq = 0;
  int hidden = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rms_eps = kRmsEps;
  float qk_scale = kQkScale;
  int window_left = -1;
  int window_right = 0;
  int kv_len = 0;          // 0 => self-attention over `seq`
  float attn_scale = 0.0f; // <= 0 => 1/sqrt(head_dim)
  bool causal = true;
};

// Parameter views for one block.
struct BlockWeights {
  const ComputeType* c_q = nullptr;
  const ComputeType* c_k = nullptr;
  const ComputeType* c_v = nullptr;
  const ComputeType* c_proj_attn = nullptr;
  const ComputeType* ve_gate = nullptr;       // nullable
  const ComputeType* value_embeds = nullptr;  // nullable
  const ComputeType* c_fc = nullptr;
  const ComputeType* c_proj_mlp = nullptr;
  bool has_ve = false;
};

// Activation slots for one block. Forward-only callers may leave `q_pre`,
// `k_pre`, and the `*_grad` views null.
struct BlockActivations {
  ComputeType* xr = nullptr;
  ComputeType* h = nullptr;
  float* rstd1 = nullptr;
  ComputeType* q_pre = nullptr;   // saved pre-QkPrep query (backward only)
  ComputeType* k_pre = nullptr;   // saved pre-QkPrep key (backward only)
  ComputeType* q_final = nullptr;
  ComputeType* k_final = nullptr;
  ComputeType* ve_values = nullptr;
  ComputeType* ve_gate = nullptr;
  ComputeType* v_final = nullptr;
  float* attn_stats = nullptr;
  ComputeType* attn_out = nullptr;
  ComputeType* x_mid = nullptr;
  ComputeType* h2 = nullptr;
  float* rstd2 = nullptr;
  ComputeType* pre_act = nullptr;
  ComputeType* act = nullptr;
  ComputeType* x_out = nullptr;
  ComputeType* scratch_a = nullptr;
  ComputeType* scratch_b = nullptr;
  // When set, attention reads these key/value rows instead of `k_final`/
  // `v_final` (the inference graphs read the KV cache).
  const ComputeType* attn_k = nullptr;
  const ComputeType* attn_v = nullptr;
  // When set, the freshly computed key/value rows are appended to the cache at
  // `cache_offset` before attention runs. This is how prefill/decode fill the
  // cache while reusing the same block forward.
  ComputeType* cache_k = nullptr;
  ComputeType* cache_v = nullptr;
  int cache_offset = 0;
};

// Parameter gradient views for one block.
struct BlockGrads {
  ComputeType* c_q_grad = nullptr;
  ComputeType* c_k_grad = nullptr;
  ComputeType* c_v_grad = nullptr;
  ComputeType* c_proj_attn_grad = nullptr;
  ComputeType* ve_gate_grad = nullptr;       // nullable
  ComputeType* value_embeds_grad = nullptr;  // nullable
  ComputeType* c_fc_grad = nullptr;
  ComputeType* c_proj_mlp_grad = nullptr;
};

// Backward-only scratch for one block.
struct BlockScratch {
  ComputeType* dq = nullptr;
  ComputeType* dk = nullptr;
  ComputeType* dv = nullptr;
  ComputeType* dh = nullptr;
  ComputeType* dxr = nullptr;
  ComputeType* dx_mid = nullptr;
  ComputeType* dh2 = nullptr;
  ComputeType* dy_attn = nullptr;
  ComputeType* dact = nullptr;
  ComputeType* dpre = nullptr;
  ComputeType* dve = nullptr;
};

// Runs one block forward: `x` and `x0` are the current residual stream and the
// smeared embedding. The `resid`/`x0_lambda` scalars blend them.
void BlockForward(const BlockShape& shape, const BlockWeights& weights,
                  const BlockActivations& acts, const int* tokens,
                  const float* cos, const float* sin, const ComputeType* x,
                  const ComputeType* x0, float resid, float x0_lambda);

// Runs one block backward and accumulates into `x0_acc`. `dout` is the
// upstream gradient; `dx_in` receives the gradient w.r.t. the block input.
void BlockBackward(const BlockShape& shape, const BlockWeights& weights,
                   const BlockActivations& acts, const BlockGrads& grads,
                   const BlockScratch& scratch, const int* tokens,
                   const float* cos, const float* sin, float resid,
                   float x0_lambda, const ComputeType* x_in,
                   const ComputeType* x0, const ComputeType* dout,
                   ComputeType* dx_in, ComputeType* x0_acc,
                   ComputeType* resid_grad, ComputeType* x0_lambda_grad);

}  // namespace ops
}  // namespace nanochat

#endif  // NANOCHAT_SRC_OPS_H_
