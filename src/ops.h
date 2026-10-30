#ifndef NANOCHAT_SRC_OPS_H_
#define NANOCHAT_SRC_OPS_H_

#include <cmath>
#include <cstdint>

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

// The fp8 scratch for one linear call: the two byte operands and the two
// dequantization scales (docs/fp8-training.md sections 4.3 and 6.3). The
// workspace owns the memory. A caller that passes a null scratch, or a scratch
// whose pointers are null, uses the dense compute-type path.
struct Fp8Scratch {
  std::uint8_t* a_bytes = nullptr;  // the activation operand
  std::uint8_t* b_bytes = nullptr;  // the weight operand
  float* a_scale = nullptr;
  float* b_scale = nullptr;

  bool ready() const {
    return a_bytes != nullptr && b_bytes != nullptr && a_scale != nullptr &&
           b_scale != nullptr;
  }
};

// The fp8 eligibility rule (docs/performance-model.md section 5.2): both
// dimensions are divisible by 16, and the smaller dimension is at least 128.
inline bool Fp8Eligible(std::int64_t rows, std::int64_t cols) {
  return rows > 0 && cols > 0 && rows % 16 == 0 && cols % 16 == 0 &&
         std::min(rows, cols) >= 128;
}

// The storage <-> float boundary for the compute type. fp32 is the only
// compute type today, so both directions are the identity.
inline float AsF(ComputeType value) { return value; }

inline ComputeType ToC(float value) { return value; }

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

namespace ops {

// ---------------------------------------------------------------------------
// Linear: y = x @ w^T (PyTorch's `nn.Linear` weight layout).
// ---------------------------------------------------------------------------

void LinearForward(const ComputeType* x, const ComputeType* w, ComputeType* y,
                   std::int64_t rows, std::int64_t in, std::int64_t out,
                   bool accumulate = false, const Fp8Scratch* fp8 = nullptr);

void LinearWgrad(const ComputeType* x, const ComputeType* dy, float* dw,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 const Fp8Scratch* fp8 = nullptr);

void LinearDgrad(const ComputeType* dy, const ComputeType* w, ComputeType* dx,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 bool accumulate = false, const Fp8Scratch* fp8 = nullptr);

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
// ResFormer value residual, expressed as a composition (docs/model-and-device-
// bringup.md section 7.4): a GEMM for the gate dot, the pointwise family for the
// sigmoid (`kGateMul` with a unit operand), the gate factor (`kScale`) and the
// per-head broadcast (`kRowScale`), and the caller's accumulating projection for
// the addition.
//
// `ValueGateForward` computes and saves the gate only: `gate_out[row, kvh] =
// m * sigmoid(dot(h[row, :kVeGateChannels], gate_w[kvh, :]))`. `ValueGateApply`
// writes `v = gate * ve` into the value rows, and the caller accumulates the
// value projection on top of it, so the addition costs no second full-size
// buffer. The gate is a saved slot: the backward reads it rather than
// recomputing it (`spec/block_kinds.md` section 2), so a fused override that
// saves a differently-rounded gate fails the same fixture.
// ---------------------------------------------------------------------------

void ValueGateForward(std::int64_t rows, int hidden, int num_kv_heads,
                      const ComputeType* h, const ComputeType* gate_w,
                      ComputeType* gate_out);

void ValueGateApply(std::int64_t rows, int num_kv_heads, int head_dim,
                    const ComputeType* gate, const ComputeType* ve,
                    ComputeType* v);

void ValueGateBackward(std::int64_t rows, int hidden, int num_kv_heads,
                       int head_dim, const ComputeType* h,
                       const ComputeType* ve, const ComputeType* gate_w,
                       const ComputeType* gate, const ComputeType* dv,
                       float* gate_w_grad, ComputeType* dh, ComputeType* dve);

// ---------------------------------------------------------------------------
// relu^2 MLP: `out = x_mid + c_proj(relu^2(c_fc(h)))`. `scratch` receives the
// c_proj output before the residual is added.
// ---------------------------------------------------------------------------

void MlpForward(const ComputeType* h, const ComputeType* c_fc,
                const ComputeType* c_proj, std::int64_t rows,
                std::int64_t hidden, std::int64_t mlp_dim, ComputeType* pre_act,
                ComputeType* act, ComputeType* scratch,
                const ComputeType* x_mid, ComputeType* out,
                const Fp8Scratch* fp8 = nullptr);

void MlpBackward(const ComputeType* h, const ComputeType* c_fc,
                 const ComputeType* c_proj, const ComputeType* pre_act,
                 const ComputeType* act, const ComputeType* dout,
                 std::int64_t rows, std::int64_t hidden, std::int64_t mlp_dim,
                 float* c_fc_grad, float* c_proj_grad, ComputeType* dh,
                 ComputeType* dact, ComputeType* dpre,
                 const Fp8Scratch* fp8 = nullptr);

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
                   float* gate_w_grad, float* lambda_grad, ComputeType* de);

// ---------------------------------------------------------------------------
// Backout: subtract a scaled mid-layer residual before the final norm.
// ---------------------------------------------------------------------------

void BackoutForward(const ComputeType* x, const ComputeType* x_backout,
                    float lambda, ComputeType* out, std::int64_t n);

void BackoutBackward(const ComputeType* d_pre, const ComputeType* x_backout,
                     float lambda, ComputeType* dbackout, float* lambda_grad,
                     std::int64_t n);

// ---------------------------------------------------------------------------
// Classifier: the loss composition. Logit softcap, cross-entropy over the
// unpadded vocabulary, the ignore index, and the optional per-row weight, all
// built from `Pointwise` (kSoftcap, kExp, kScale, kRowScale), `RowReduce`
// (kMax, kSum, kLogSumExp) and `Embedding` (the per-row target gather).
//
// `row_scale` is a *host* array of `rows` entries: it is an input to this call
// and never a device operand, which is why the composition can multiply by it
// per row without a second device buffer.
//
// The composition is the definition; the fused `ClassifierForward` /
// `ClassifierBackward` families remain an override behind their capability bit.
// ---------------------------------------------------------------------------

struct ClassifierShape {
  int rows = 0;
  int vocab_size = 0;
  int padded_vocab_size = 0;
  float softcap = kLogitSoftcap;
  int ignore_index = -1;
  const float* row_scale = nullptr;  // host array of `rows` weights, or null
};

void ClassifierForward(const ClassifierShape& shape, const ComputeType* logits,
                       const int* targets, float* losses);

void ClassifierBackward(const ClassifierShape& shape, const ComputeType* logits,
                        const int* targets, ComputeType* dlogits);

// ---------------------------------------------------------------------------
// The global gradient-norm composition: the sum of squares of every gradient
// span (`SpanReduce` with no second operand and `kSumSq`-equivalent arithmetic
// over the squares), the norm, and the clip scale. There is no fused family
// behind it in vocabulary 1: `spec/capability_bits.md` section 5 records that
// `grad_norm` has no override bit, so this composition IS the implementation.
//
// `grads` and `counts` are device arrays of `num_spans` entries.
// ---------------------------------------------------------------------------

void GradNormForward(const float* const* grads, const int* counts,
                     int num_spans, float clip, float* out_norm,
                     float* out_scale);

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
  int kv_len = 0;           // 0 => self-attention over `seq`
  float attn_scale = 0.0f;  // <= 0 => 1/sqrt(head_dim)
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
  ComputeType* q_pre = nullptr;  // saved pre-QkPrep query (backward only)
  ComputeType* k_pre = nullptr;  // saved pre-QkPrep key (backward only)
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
  // The fp8 scratch for the eligible forward matmuls
  // (docs/fp8-training.md section 6.3). Null in a dense build or on the
  // evaluation path, so the same block forward serves both.
  Fp8Scratch fp8;
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
  float* c_q_grad = nullptr;
  float* c_k_grad = nullptr;
  float* c_v_grad = nullptr;
  float* c_proj_attn_grad = nullptr;
  float* ve_gate_grad = nullptr;       // nullable
  float* value_embeds_grad = nullptr;  // nullable
  float* c_fc_grad = nullptr;
  float* c_proj_mlp_grad = nullptr;
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
  // The fp8 scratch for the eligible backward matmuls
  // (docs/fp8-training.md section 7). Null in a dense build.
  Fp8Scratch fp8;
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
                   ComputeType* dx_in, ComputeType* x0_acc, float* resid_grad,
                   float* x0_lambda_grad);

}  // namespace ops
}  // namespace nanochat

#endif  // NANOCHAT_SRC_OPS_H_
