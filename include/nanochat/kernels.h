#ifndef NANOCHAT_KERNELS_H_
#define NANOCHAT_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "nanochat/tensor.h"

// The backend seam (L1 in DESIGN.md section 2). A backend is complete when
// every symbol below resolves and passes the oracle. This header is closed and
// vendor-free: it never includes a CUDA/HIP header, pointers are always
// `ComputeType*`, and device streams are opaque. See docs/kernels.md.
//
// Seam version: 2. The version lives in this header comment rather than in one
// symbol (DESIGN.md section 10.3), because a backend implements the header as
// one contract: it is bumped once per change set, under the evolution rules in
// DESIGN.md section 7, and a bring-up pins the number it implements.
//   1  the seam as it stood: the GEMM entry, the scaled fp8 GEMM, the compute
//      families (RmsNorm, QkPrep, Attention, Classifier, Embedding, ValueGate),
//      the two stateful training entries (AdamW, Muon), GradNorm, ScalarDot,
//      and the pointwise pair with a scalar argument list.
//   2  adds `RowReduce` and `SpanReduce`, and replaces the pointwise scalar
//      argument list with `PointwiseParams` plus the kExp, kRsqrt, kLog and
//      kRowScale codes. The reductions are new symbols; the pointwise pair
//      keeps its name and its arithmetic, and the runtime updates those call
//      sites in the same change set.
// One bump covers one change set, however many symbols that set adds. A worker
// who later wants a second header edit reports it instead of making it.
//
// Layout conventions:
//   * Every tensor is dense row-major unless a parameter says otherwise.
//   * Token ids and targets are `int` (one per row) and live on the host.
//   * Norm statistics (`rstd`), optimizer moments, and master weights are
//     `float` on every device (docs/build.md).

namespace nanochat {

// The seam version this header declares. A backend pins the number it
// implements, so an out-of-tree backend cannot bind an older contract by
// accident. See the header comment for what each version covers.
inline constexpr int kKernelSeamVersion = 2;

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

// Pointwise op codes: the five arithmetic codes (four plain, one with a row
// broadcast), the softcap, and the three transcendental codes. The same code
// drives forward and backward so the pair cannot drift apart (DESIGN.md
// section 5).
//
// The enum is closed. One build links the runtime and every backend, so a
// backend's switch covers every code with no `default` arm: a missing code is a
// compile error, never a silent run-time fallthrough. The per-code arithmetic
// and the meaning of alpha and beta are in `PointwiseParams` below.
enum class PointwiseOp {
  kScale = 0,       // out = alpha * a
  kScaleAdd = 1,    // out = alpha * a + beta * b
  kGateMul = 2,     // out = sigmoid(a) * b
  kReluSquare = 3,  // out = relu(a)^2
  kSoftcap = 4,     // out = cap * tanh(a / cap), with cap read from alpha
  kExp = 5,         // out = exp(a)
  kRsqrt = 6,       // out = 1 / sqrt(a)
  kLog = 7,         // out = log(a)
  kRowScale = 8,    // out[i] = alpha * a[i] * b[i % row_width]
};

// The combine op of a reduction. Closed for the same reason as `PointwiseOp`:
// a backend's switch names every code and has no `default` arm.
enum class ReduceOp {
  kSumSq = 0,      // sum of squares: the unitless mean-square argument
  kSum = 1,        // plain sum
  kMax = 2,        // segment maximum, the stable log-sum-exp seed
  kLogSumExp = 3,  // log(sum(exp(x))): nats, computed max-stable
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

// One fp8 operand format. The forward pass uses E4M3 and the backward pass uses
// E5M2 (docs/fp8-training.md section 3.1).
enum class Fp8Format {
  kE4M3 = 0,  // forward operands
  kE5M2 = 1,  // backward operands
};

struct QuantizeFp8Params {
  int count = 0;  // elements in `src`
  Fp8Format format = Fp8Format::kE4M3;
  // The amax floor. An all-zero tensor has an amax of zero, so the scale would
  // be infinite without it (docs/fp8-training.md section 4.1).
  float amax_floor = 1e-12f;
};

// A scaled fp8 matrix multiply: `C = alpha * op(A) * op(B) + beta * C`, with
// fp8 operands and an fp32 accumulator. The output type follows the mode: the
// forward and the input gradient produce an activation, so `c` is
// `ComputeType*`; the weight gradient produces a parameter gradient, so `c` is
// `float*` (docs/fp8-training.md section 7).
//
// `scale_a` and `scale_b` are the dequantization values from `QuantizeFp8`, and
// a null pointer means 1.0. The two formats are separate, because the input and
// weight gradients mix E5M2 and E4M3 in one call (docs/fp8-training.md
// section 7).
struct ScaledGemmParams {
  int m = 0;  // rows of C
  int n = 0;  // columns of C
  int k = 0;  // reduction extent
  int batch_count = 1;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool transpose_a = false;
  bool transpose_b = false;
  bool fast_accum = false;  // reduced-precision reduction; forward only
  Fp8Format format_a = Fp8Format::kE4M3;
  Fp8Format format_b = Fp8Format::kE4M3;
  const float* scale_a = nullptr;  // dequantization value for A
  const float* scale_b = nullptr;  // dequantization value for B
};

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
//
// Two reduction entries, plus the existing `ScalarDot`, and not one descriptor
// with three modes. A single descriptor carrying "per-segment", "ragged list"
// and "scalar dot" modes would merge three different algorithms, three
// different correctness risks, and three different units into one parameter
// block:
//
//   * Algorithm. `RowReduce` is a per-segment reduction whose shape is known at
//     launch, so one segment can reduce inside one CTA. `SpanReduce` reduces a
//     list the host never reads, so the segment boundaries are visible only to
//     the kernel and a span may exceed one CTA. `ScalarDot` is one contiguous
//     dot with a fixed launch. No single launch covers all three, so no single
//     descriptor should pretend to.
//   * Correctness risk. `RowReduce`'s result feeds a divisor (RmsNorm, QkPrep)
//     and a log-sum-exp (Classifier), so a wrong index or a wrong combine is a
//     silently wrong loss and the gate has to be near-exact. `SpanReduce` is an
//     accumulation whose order across spans is already unspecified, so its gate
//     is a tolerance. Sharing one code would put the tolerance gate on the path
//     that needs the strict one.
//   * Units. `RowReduce` returns a unitless mean-square argument (RmsNorm,
//     QkPrep), nats (the Classifier's log-sum-exp), or a raw sum. `SpanReduce`
//     returns a sum of products. Different units mean different reference
//     formulas, which is what the oracle compares against.
//
// `RowReduce` is the highest-risk line in the seam: one reduction must serve
// RmsNorm, QkPrep and Classifier, with different combine ops and different
// offsets. Neither of the other two is a row reduction: `SpanReduce` reduces a
// ragged device-resident list, and `ScalarDot` reduces one contiguous buffer.
//
// The field sets below are closed, exactly as section 7.2 states them. A field
// is not added here because one consumer wants one: if the CPU RmsNorm
// comparison (the open question recorded in section 17) finds a missing field,
// that is a reported follow-up, not a silent second edit to this header.

// One reduction per equally shaped segment: `n_out` scalars from `n_out`
// segments of `count` elements each. Element `j` of segment `s` is
// `a[offset + s * count * stride + j * stride]`, so the same call covers a
// contiguous row (`count` = row width, `stride` = 1, `n_out` = rows) and a
// head_dim slice inside a wider row (`offset` at the head, `count` = head_dim,
// `stride` = 1, `n_out` = 1, or `n_out` = rows * heads when the heads are
// densely packed).
//
// This is the highest-risk line: RmsNorm reads it as a sum of squares per row,
// QkPrep as a sum of squares per head slice, and Classifier as a max-stable
// log-sum-exp per row. One indexing mistake or one wrong combine reaches all
// three operators at once, so each consumer keeps its own oracle gate.
struct RowReduceParams {
  int n_out = 0;            // segments, one output scalar each
  int count = 0;            // elements per segment
  std::int64_t stride = 1;  // element stride inside a segment
  std::int64_t offset = 0;  // element offset of the first segment
  ReduceOp op = ReduceOp::kSumSq;
};

// A ragged reduction to one scalar: `num_spans` segments whose lengths differ.
// The segment pointers and the lengths live in *device* memory, so the caller
// builds the list without a host round trip (`GradNorm`'s `grads` and `counts`
// follow the same rule). `other` is a second list of the same ragged shape:
// when it is not null the reduction is the per-segment dot product of the two,
// so one entry point serves a plain sum and a dot. `scale` multiplies the
// result, and `accumulate` adds to `*out` instead of writing it.
struct SpanReduceParams {
  int num_spans = 0;
  const int* counts = nullptr;  // device array: elements per span, in order
  float scale = 1.0f;
  bool accumulate = false;
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
  // Additive field (API freeze): the row stride of the saved softmax
  // statistics when the backward processes a query sub-block. 0 means the
  // statistics are contiguous with `seq` rows per head. The windowed backward
  // passes the full sequence length so the block kernel can index the shared
  // statistics buffer at its query offset (docs/attention.md Part 2).
  int stats_row_stride = 0;
};

// The pointwise block. It replaces the former scalar argument list
// `(op, n, a, b, alpha, beta)` so that a broadcast and the transcendental codes
// could be added without another positional argument.
//
// Per-code arithmetic and the meaning of alpha and beta:
//   kScale       out = alpha * a                            (beta unused)
//   kScaleAdd    out = alpha * a + beta * b
//   kGateMul     out = sigmoid(a) * b                (alpha, beta unused)
//   kReluSquare  out = relu(a)^2                     (alpha, beta unused)
//   kSoftcap     out = alpha * tanh(a / alpha)  (the cap is alpha; beta unused)
//   kExp         out = exp(a)                        (alpha, beta unused)
//   kRsqrt       out = 1 / sqrt(a)                   (alpha, beta unused)
//   kLog         out = log(a)                        (alpha, beta unused)
//   kRowScale    out[i] = alpha * a[i] * b[i % row_width]     (beta unused)
// An unused value is ignored, not asserted: a stale alpha or beta cannot change
// a result the code does not read.
//
// `row_width` is the width of the row the broadcast reads. `b` holds one row of
// `row_width` entries, replicated down the `count / row_width` rows of `a`; 0
// means `count`, which is the whole-buffer case. The backward writes `da` and
// accumulates the per-row-element sum into the `row_width` entries of `db`, so
// the caller treats `db` as an accumulation buffer.
//
// Accuracy. The arithmetic codes (kScale, kScaleAdd, kGateMul, kReluSquare,
// kRowScale) are bit-exact across devices and the pointwise gate compares them
// bit for bit. The transcendental codes (kSoftcap, kExp, kRsqrt, kLog) are not
// bit-exact across devices and cannot be: exp, tanh and rsqrt differ in the
// last bit between a libm, a device intrinsic, and a hand-written polynomial.
// Their accuracy obligation is therefore a declared bound per code, and it
// lives in spec/pointwise_ops.md, which is the authority for the bounds; this
// header is the authority for the code list. A code with no declared bound
// there does not ship.
struct PointwiseParams {
  PointwiseOp op = PointwiseOp::kScale;
  int count = 0;      // elements in `a`, in `out`, and in `dy` on the backward
  int row_width = 0;  // width of the row a broadcast reads; 0 => count
  float alpha = 1.0f;
  float beta = 0.0f;
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
  // The gradient clip scale, folded into the gradient read. One when the clip
  // is disabled or inactive, so an unscaled read is bit-identical.
  float grad_scale = 1.0f;
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
  // The gradient clip scale, folded into the gradient read (see AdamWParams).
  float grad_scale = 1.0f;
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

// Weight-gradient GEMM. The operands use the build compute type; the result
// accumulates into fp32 parameter-gradient storage.
void GemmWgrad(const GemmParams& params, const ComputeType* a,
               const ComputeType* b, float* c);

// --- Scaled fp8 GEMM (docs/fp8-training.md sections 4 and 5) ---------------

// Quantizes `src` to one byte per element with one tensorwise scale, and writes
// the dequantization value (`1 / scale`) to `*dequant`. The rounding is shared
// with the simulator through `nanochat/fp8.h`, so the device and the CPU
// reference agree (docs/correctness-gates.md section 3.2).
void QuantizeFp8(const QuantizeFp8Params& params, const ComputeType* src,
                 std::uint8_t* dst, float* dequant);

void GemmScaledFp8(GemmMode mode, const ScaledGemmParams& params,
                   const std::uint8_t* a, const std::uint8_t* b, void* c);

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

// `b` is read by kScaleAdd and kGateMul and is the broadcast table for
// kRowScale; the codes that do not read it accept null.
void PointwiseForward(const PointwiseParams& params, const ComputeType* a,
                      const ComputeType* b, ComputeType* out);

// Backward of the matching forward. `da` is always written. `db` is read by
// kScaleAdd (scaled by beta) and accumulated by kRowScale; a null `db` is
// allowed for the codes whose backward does not touch it.
void PointwiseBackward(const PointwiseParams& params, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy,
                       ComputeType* da, ComputeType* db);

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       float* losses);

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits);

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out);

// `dtable` is a dense gradient buffer; the backend zeroes only the touched rows
// (docs/model.md).
void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, float* dtable);

// The optimizer updates the fp32 master weight and writes the compute copy
// (`ComputeType`) beside it. The fp32 build passes one buffer for `master` and
// `value`, so the pointer aliases the parameter (docs/model.md).
void AdamWUpdate(int n, const AdamWParams& params, float* master,
                 ComputeType* value, const float* g, float* m, float* v);

// `stacked_grads` and `stacked_values` are `[num_params, rows, cols]`;
// `stacked_master` is the fp32 master of the same shape. `buf1` is the
// momentum buffer of the same shape and `buf2` is the factored second moment
// of shape `[num_params, rows, 1]` or `[num_params, 1, cols]`.
void MuonUpdate(const MuonParams& params, const float* stacked_grads,
                float* stacked_master, ComputeType* stacked_values, float* buf1,
                float* buf2);

// --- Reductions ------------------------------------------------------------

// Writes one scalar per segment to `out`, a device buffer of
// `RowReduceParams::n_out` floats. `out` is written, never accumulated, and the
// three consumers (RmsNorm, QkPrep, Classifier) read it back as `rstd` input,
// per-head scale input, or row statistics.
void RowReduce(const RowReduceParams& params, const ComputeType* a, float* out);

// Reduces a ragged, device-resident span list to the single scalar `*out`: the
// sum of the spans, or the dot product of `spans` and `other` when `other` is
// not null, scaled by `params.scale` and written to or added to `*out`. The
// order across spans is unspecified, so this entry is not the one a bit-exact
// gate rides on; `RowReduce` is.
void SpanReduce(const SpanReduceParams& params, const ComputeType* const* spans,
                const ComputeType* const* other, float* out);

// Accumulates the sum of squares of `num_spans` gradient buffers in double,
// writes the pre-clip L2 norm to `out_norm`, and writes the clip scale to
// `out_scale`. The scale is one when `clip <= 0` or when the norm does not
// exceed `clip`; otherwise it is `clip / norm`. No gradient buffer is modified:
// the caller folds `*out_scale` into the `grad_scale` of every update, which is
// what keeps the weight decay unscaled. Both outputs are single-element device
// buffers, and either may be null.
//
// `grads` and `counts` are device arrays of `num_spans` entries: the kernel
// reads them, so they cannot live on the host (`EmbeddingForward`'s `ids`
// follow the same rule).
void GradNorm(const float* const* grads, const int* counts, int num_spans,
              float clip, float* out_norm, float* out_scale);

// Reduces `dot(a, b)` over `n` elements into the single-element buffer `out`:
// `out = scale * dot(a, b)`, or `out += scale * dot(a, b)` when `accumulate`.
// This is the device-side replacement for the host round trip that the scalar
// parameter gradients (resid, x0_lambda, backout_lambda) used to need.
void ScalarDot(const ComputeType* a, const ComputeType* b, int n, float* out,
               float scale, bool accumulate);

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
                       const ComputeType* dv, float* gate_w_grad,
                       ComputeType* dh, ComputeType* dve);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_KERNELS_H_
