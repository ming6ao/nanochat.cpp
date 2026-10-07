#ifndef NANOCHAT_KERNELS_H_
#define NANOCHAT_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "nanochat/tensor.h"

// The backend seam (L1 in DESIGN.md section 2). A backend is complete when
// every symbol below resolves and passes the oracle. This header is frozen and
// vendor-free: it never includes a CUDA/HIP header, pointers are always
// `ComputeType*`, and device streams are opaque. See docs/kernels.md.
//
// Layout conventions:
//   * Every tensor is dense row-major unless a parameter says otherwise.
//   * Token ids and targets are `int` (one per row) and live on the host.
//   * Norm statistics (`rstd`), optimizer moments, and master weights are
//     `float` even in an fp16 build (docs/build.md).

namespace nanochat {

enum class CopyDir {
  kHostToDevice = 0,
  kDeviceToHost = 1,
  kDeviceToDevice = 2,
  kHostToHost = 3,
};

// GEMM operand modes. Forward computes C = op(A) * op(B); the backward modes
// reuse the same entry point with the operands the host graph supplies.
enum class GemmMode {
  kForward = 0,  // C = A * B
  kDgrad = 1,    // gradient with respect to A
  kWgrad = 2,    // gradient with respect to B
};

// Pointwise op codes (relu^2, scale-add, gate-mul, scale, softcap). The same
// code drives forward and backward so the pair cannot drift apart
// (DESIGN.md section 5).
enum class PointwiseOp {
  kScale = 0,       // out = alpha * a
  kScaleAdd = 1,    // out = alpha * a + beta * b
  kGateMul = 2,     // out = sigmoid(a) * b
  kReluSquare = 3,  // out = relu(a)^2
  kSoftcap = 4,     // out = cap * tanh(a / cap)
};

// ---------------------------------------------------------------------------
// Parameter blocks
// ---------------------------------------------------------------------------

inline constexpr float kDefaultRmsNormEps = 1e-6f;
inline constexpr float kDefaultLogitSoftcap = 15.0f;
inline constexpr float kDefaultQkScale = 1.2f;

struct GemmParams {
  int m = 0;  // rows of C
  int n = 0;  // columns of C
  int k = 0;  // reduction extent
  int batch_count = 1;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool transpose_a = false;
  bool transpose_b = false;
  // Leading dimensions / per-batch strides. Zero means "infer a dense,
  // row-major layout from the logical shape".
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
};

struct RmsNormParams {
  int rows = 0;  // rows to reduce over
  int dim = 0;   // row width
  float eps = kDefaultRmsNormEps;
};

struct QkPrepParams {
  int batch = 0;
  int seq = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float eps = kDefaultRmsNormEps;
  float scale = kDefaultQkScale;  // applied to both q and k after RoPE
};

struct AttentionParams {
  int batch = 0;
  int seq = 0;  // query rows
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  bool causal = true;
  int window_left = -1;  // -1 = full context
  int window_right = 0;  // 0 for causal decoding
  // Additive fields (API freeze): key/value rows, which differ from `seq` when
  // decoding against a KV cache; and the softmax scale, where <= 0 means
  // `1 / sqrt(head_dim)`.
  int kv_len = 0;  // 0 => seq (self-attention)
  float scale = 0.0f;
};

// Logit softcap + cross-entropy + unpadded-vocab slice (docs/kernels.md and
// docs/post-training.md section 2.2). `logits` is `rows x padded_vocab_size`;
// only the first `vocab_size` columns are read, and the backward zeroes the
// padded tail.
struct ClassifierParams {
  int rows = 0;
  int vocab_size = 0;
  int padded_vocab_size = 0;
  float softcap = kDefaultLogitSoftcap;
  int ignore_index = -1;
  // Optional per-row weights of length `rows` for the backward. A null pointer
  // keeps the current behavior. When the pointer is not null, the backward
  // multiplies row `r` by `row_scale[r]` after the softcap chain rule. The
  // backward still zeroes every ignored row and the padded vocabulary tail.
  const float* row_scale = nullptr;
};

struct AdamWParams {
  float lr = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-10f;
  float weight_decay = 0.0f;
  int step = 1;  // 1-based, for bias correction
};

struct MuonParams {
  int num_params = 1;  // matrices stacked along axis 0
  int rows = 0;        // trailing row extent of each matrix
  int cols = 0;        // trailing column extent of each matrix
  float lr = 0.0f;
  float momentum = 0.95f;
  float beta2 = 0.9f;
  float weight_decay = 0.0f;
  int ns_steps = 5;
  int red_dim = -1;  // variance reduction axis: -1 (columns) or -2 (rows)
  bool nesterov = true;
};

// Softmax statistics saved by attention forward and consumed by its backward.
// Layout is `[batch, num_heads, seq, 2]` with (max, sum_exp) per query row.
inline int AttentionStatsCount(const AttentionParams& params) {
  return params.batch * params.num_heads * params.seq * 2;
}

// ---------------------------------------------------------------------------
// Kernel entry points
// ---------------------------------------------------------------------------

namespace kernels {

// --- Device / memory (backend-owned) --------------------------------------

void* Alloc(std::size_t bytes);
void Free(void* ptr);
void Memcpy(void* dst, const void* src, std::size_t bytes, CopyDir dir);
void Memset(void* ptr, int value, std::size_t bytes);
void Synchronize();
Caps GetCaps();

// --- Library-backed GEMM ---------------------------------------------------

void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c);

// --- Compute (fwd / bwd) ---------------------------------------------------

void RmsNormForward(const RmsNormParams& params, const ComputeType* x,
                    ComputeType* out, float* rstd);

void RmsNormBackward(const RmsNormParams& params, const ComputeType* x,
                     const ComputeType* dy, const float* rstd, ComputeType* dx);

// Fused RMSNorm(q/k) -> RoPE -> scale, in place on the projection outputs.
// `cos` and `sin` are fp32 rotary tables of shape `[seq, head_dim / 2]`.
// The backward reads the saved forward outputs (q, k) to recover the
// normalization statistics and overwrites them with the input gradients, so
// `q` and `k` are both saved activations and gradient outputs.
void QkPrepForward(const QkPrepParams& params, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k);

void QkPrepBackward(const QkPrepParams& params, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k);

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats);

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv);

void PointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                      const ComputeType* b, float alpha, float beta,
                      ComputeType* out);

void PointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy, float alpha,
                       float beta, ComputeType* da, ComputeType* db);

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses);

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits);

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out);

// `dtable` is a dense gradient buffer; the backend zeroes only the touched rows
// (docs/model.md).
void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, ComputeType* dtable);

void AdamWUpdate(int n, const AdamWParams& params, ComputeType* p,
                 const ComputeType* g, float* m, float* v);

// `stacked_*` are `[num_params, rows, cols]`; `buf1` is the momentum buffer of
// the same shape and `buf2` is the factored second moment of shape
// `[num_params, rows, 1]` or `[num_params, 1, cols]`.
void MuonUpdate(const MuonParams& params, const ComputeType* stacked_grads,
                ComputeType* stacked_params, float* buf1, float* buf2);

// Computes the global L2 norm of `grads`, clips it to `clip` in place, and
// writes the pre-clip norm to `out_norm`.
void GlobalNorm(int n, float clip, ComputeType* grads, float* out_norm);

// Reduces `dot(a, b)` over `n` elements into the single-element buffer `out`:
// `out = scale * dot(a, b)`, or `out += scale * dot(a, b)` when `accumulate`.
// This is the device-side replacement for the host round trip that the scalar
// parameter gradients (resid, x0_lambda, backout_lambda) used to need.
void ScalarDot(const ComputeType* a, const ComputeType* b, int n,
               ComputeType* out, float scale, bool accumulate);

// ResFormer value gate. For each row and key/value head,
//   gate = 3 * sigmoid(dot(h[row, :12], gate_w[kvh, :]))
//   v[row, kvh, :] += gate * ve[row, kvh, :]
// `gate_out` is `[rows, num_kv_heads]` and saves the gate for the backward.
// `hidden` is the row width of `h`; only the first 12 channels feed the gate.
void ValueGateForward(int rows, int hidden, int num_kv_heads, int head_dim,
                      const ComputeType* h, const ComputeType* ve,
                      const ComputeType* gate_w, ComputeType* v,
                      ComputeType* gate_out);

// Backward of ValueGateForward. `gate_w_grad` and `dh` accumulate; `dve` is
// written. `dv` is the incoming gradient w.r.t. the gated value.
void ValueGateBackward(int rows, int hidden, int num_kv_heads, int head_dim,
                       const ComputeType* h, const ComputeType* ve,
                       const ComputeType* gate_w, const ComputeType* gate,
                       const ComputeType* dv, ComputeType* gate_w_grad,
                       ComputeType* dh, ComputeType* dve);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_KERNELS_H_
