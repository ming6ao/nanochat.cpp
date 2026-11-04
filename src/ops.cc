// Reusable graph pieces shared by the training graph and the inference graphs.
// This file is the *composition tier* (`spec/block_kinds.md` section 4): every
// named piece below is written over the primitives of the kernel seam - `Gemm`,
// `RowReduce`, `SpanReduce`, `Pointwise` (with all twelve codes of
// `spec/pointwise_ops.md`, among them kAffine, kSigmoid and the kRotatePair
// rotation) and `Embedding` - so a device that supplies the primitive tier can
// run the whole graph, and a backend's fused family is an optimization behind
// an `override_*` capability bit rather than a second implementation the tree
// depends on (`docs/model-and-device-bringup.md` section 7.4 condition 2). No
// composition here stages an activation through the host: the rotary leg, the
// norm's epsilon and the value gate's unit operand all have a code, which is
// what `docs/model-and-device-bringup.md` section 7.6 asks for.
// Two consequences are worth knowing while reading.  * A composition declares
// its saved slots and its backward reads only those (`spec/block_kinds.md`
// section 3). The slots are the caller's buffers, so a fused override and this
// composition can share one workspace: `rstd`, `q_pre`/`k_pre`, `pre_act`,
// `ve_values`/`ve_gate` and `attn_stats` are all named on the way in. *
// `Attention` is the documented exception (section 7.2): its iteration space is
// T x T, no composition of the other primitives reproduces its tiling without
// materializing a T x T tensor, so the attention kind is served by the
// primitive directly. It is the only seam family this file calls that is not a
// primitive.  `src/ops.h` is the inventory; the topology is in `model.cc` and
// `generate.cc`. See docs/model.md and docs/kernels.md.

#include "src/ops.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "src/kernel_params_check.h"

namespace nanochat {
namespace ops {

namespace {

// The value of a scalar argument whose host copy is the one to check. A
// coefficient the caller keeps on device (`ScalarArg::dev`, docs/scalar-
// arguments.md) is read by the kernel, so reading it here would be the host
// round trip `spec/api-surface.md` R3 rules out; that case passes unchecked.
// It exists only in a debug build, where the checks below are the only callers.
#ifndef NDEBUG
bool FiniteArg(ScalarArg arg) {
  return arg.dev != nullptr || std::isfinite(arg.value);
}
#endif

// True when the device advertises the fused family of one composition.  The
// composition below every call site is the definition: it is written over the
// primitive tier, so a device that supplies only the primitives runs the whole
// graph. The fused family is the same arithmetic in one kernel, and the backend
// registers it beside the capability bit it advertises
// (`backends/cuda/fused_overrides.cc`). Until this helper existed the bit was
// declared, planned, and never read, so every device ran the portable fallback:
// the classifier cost a launch and a blocking device-to-host copy per token
// position, and the composition tier cycled activations through the host. A
// training step at 4096 token positions therefore took 2.2 s on an H100 at 10
// percent device utilization.  `GetCaps` is read once: a process owns one
// device, which is the rule the rest of this file already follows.
bool Fused(std::uint64_t feature) {
  static const Caps caps = kernels::GetCaps();
  return caps.HasFeature(feature);
}

// The classifier's fused family reads `row_scale` in device code, one entry per
// row (`backends/cuda/kernels/classifier.cu`, `params.row_scale[row]`). A host
// pointer there faults, which is why this call once kept the whole classifier
// on the composition and paid a blocking device-to-host copy for every row of
// every step. The caller now keeps a device copy beside the host array, so the
// fused family serves the weighted call and the per-row round trip is gone.
//
// A caller that supplies only the host array still takes the composition, which
// reads that array in host code. The check keeps the trap closed rather than
// trusting every caller.
bool FusedClassifierTakesScale(const ClassifierShape& shape) {
  return shape.row_scale == nullptr || shape.row_scale_dev != nullptr;
}

// ---------------------------------------------------------------------------
// Scalar host traffic
// ---------------------------------------------------------------------------
// One kind of host traffic survives in this file, and it is the classifier's
// per-row scalar: one float per row, a constant-sized round trip per row and
// not a staging pass over an activation. It is marked where it is used so a
// later change can delete exactly that. The activation staging that used to sit
// beside it is gone: `kRotatePair` rotates on the device and `kAffine` adds a
// scalar without a constant vector, so neither the rotary leg nor the norm's
// epsilon moves a buffer through the host any more.

float StageScalar(const float* device) {
  float value = 0.0f;
  kernels::Memcpy(&value, device, sizeof(float), CopyDir::kDeviceToHost);
  return value;
}

void StoreScalar(float* device, float value) {
  kernels::Memcpy(device, &value, sizeof(float), CopyDir::kHostToDevice);
}

// ---------------------------------------------------------------------------
// The pointwise family, one call per composition step
// ---------------------------------------------------------------------------

// One pointwise call whose coefficients are scalar arguments, so a caller that
// keeps them on the device reads them there (docs/scalar-arguments.md).
void PwScalars(PointwiseOp op, int count, const ComputeType* a,
               const ComputeType* b, ComputeType* out, ScalarArg alpha,
               ScalarArg beta) {
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", count >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", FiniteArg(alpha));
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", FiniteArg(beta));
  if (count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", a != nullptr && out != nullptr);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               b != nullptr || (op != PointwiseOp::kScaleAdd &&
                                                op != PointwiseOp::kGateMul &&
                                                op != PointwiseOp::kRowScale));
  PointwiseParams params;
  params.op = op;
  params.count = count;
  params.alpha = alpha;
  params.beta = beta;
  kernels::PointwiseForward(params, a, b, out);
}

void Pw(PointwiseOp op, int count, const ComputeType* a, const ComputeType* b,
        ComputeType* out, float alpha = 1.0f, float beta = 0.0f,
        int row_width = 0) {
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", count >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", row_width >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               std::isfinite(alpha) && std::isfinite(beta));
  // A broadcast reads one row of `row_width` entries and replicates it down
  // the rows of `a`, so a width that does not divide the count would read a
  // fraction of a row off the end of `b`.
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               row_width == 0 || count % row_width == 0);
  if (count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", a != nullptr && out != nullptr);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               b != nullptr || (op != PointwiseOp::kScaleAdd &&
                                                op != PointwiseOp::kGateMul &&
                                                op != PointwiseOp::kRowScale));
  PointwiseParams params;
  params.op = op;
  params.count = count;
  params.row_width = row_width;
  params.alpha = alpha;
  params.beta = beta;
  kernels::PointwiseForward(params, a, b, out);
}

void PwBack(PointwiseOp op, int count, const ComputeType* a,
            const ComputeType* b, const ComputeType* dy, ComputeType* da,
            ComputeType* db, float alpha = 1.0f, float beta = 0.0f,
            int row_width = 0) {
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", count >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", row_width >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               std::isfinite(alpha) && std::isfinite(beta));
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               row_width == 0 || count % row_width == 0);
  if (count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise",
                               a != nullptr && dy != nullptr && da != nullptr);
  PointwiseParams params;
  params.op = op;
  params.count = count;
  params.row_width = row_width;
  params.alpha = alpha;
  params.beta = beta;
  kernels::PointwiseBackward(params, a, b, dy, da, db);
}

// An elementwise product: `kRowScale` with a row width of one, which is the
// whole-buffer case of the broadcast (every element is its own row). The
// primitive set has no dedicated multiply code because this is its degenerate
// form (`spec/pointwise_ops.md` section 3.1).
void Mul(const ComputeType* a, const ComputeType* b, ComputeType* out,
         int count) {
  Pw(PointwiseOp::kRowScale, count, a, b, out, 1.0f, 0.0f, /*row_width=*/1);
}

// `out = a op b` collapsed into a per-row reduce, the shape `RmsNorm`,
// `Classifier` and the value gate all need. Named here so the `RowReduceParams`
// fields are spelled once per combine op.
void RowReduce(ReduceOp op, int n_out, int count, const ComputeType* a,
               float* out, std::int64_t offset = 0) {
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", n_out >= 0 && count >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", offset >= 0);
  if (n_out <= 0 || count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", a != nullptr && out != nullptr);
  RowReduceParams params;
  params.n_out = n_out;
  params.count = count;
  params.stride = 1;
  params.offset = offset;
  params.op = op;
  kernels::RowReduce(params, a, out);
}

// ---------------------------------------------------------------------------
// Small device buffers a composition owns for the duration of one call
// ---------------------------------------------------------------------------

// A device buffer of `count` floats, owned by one composition call. The
// window/norm compositions need O(rows) of them, not O(rows * dim): the
// primitives keep the wide intermediate in the caller's buffers, which is the
// whole point of the row broadcast.
class FloatScratch {
 public:
  explicit FloatScratch(int count) : count_(count) {
    if (count_ > 0) {
      data_ = static_cast<float*>(
          kernels::Alloc(static_cast<std::size_t>(count_) * sizeof(float)));
    }
  }
  ~FloatScratch() {
    if (data_ != nullptr) kernels::Free(data_);
  }
  FloatScratch(const FloatScratch&) = delete;
  FloatScratch& operator=(const FloatScratch&) = delete;

  float* data() { return data_; }
  const float* data() const { return data_; }
  int count() const { return count_; }

 private:
  int count_ = 0;
  float* data_ = nullptr;
};

// A device buffer of `count` compute-type elements, owned by one composition
// call. The classifier's per-row row buffer is the only caller: it is O(vocab),
// not O(rows * vocab), because the composition walks one row at a time.
class ComputeScratch {
 public:
  explicit ComputeScratch(int count) : count_(count) {
    if (count_ > 0) {
      data_ = static_cast<ComputeType*>(kernels::Alloc(
          static_cast<std::size_t>(count_) * sizeof(ComputeType)));
    }
  }
  ~ComputeScratch() {
    if (data_ != nullptr) kernels::Free(data_);
  }
  ComputeScratch(const ComputeScratch&) = delete;
  ComputeScratch& operator=(const ComputeScratch&) = delete;

  ComputeType* data() { return data_; }
  const ComputeType* data() const { return data_; }
  int count() const { return count_; }

 private:
  int count_ = 0;
  ComputeType* data_ = nullptr;
};

// The span list `SpanReduce` reads. The seam takes the span pointers, the
// optional second operand pointers, and the per-span counts from device memory
// (`include/nanochat/kernels.h`), so a composition that builds them in stack
// storage stages them first. The optimizer keeps the same tables resident for
// its per-step reduce (`src/optim.cc`, `norm_grads_` and `norm_counts_`); this
// is the one-call form. On the CPU backend `Alloc` returns host memory, so the
// staging copies are memcpy calls and the numbers do not change.
class SpanList {
 public:
  SpanList(const ComputeType* const* spans, const ComputeType* const* other,
           const int* counts, int num_spans) {
    if (num_spans <= 0) return;
    spans_ = StagePointers(spans, num_spans);
    if (other != nullptr) other_ = StagePointers(other, num_spans);
    const std::size_t count_bytes =
        static_cast<std::size_t>(num_spans) * sizeof(int);
    counts_ = static_cast<int*>(kernels::Alloc(count_bytes));
    kernels::Memcpy(counts_, counts, count_bytes, CopyDir::kHostToDevice);
  }
  ~SpanList() {
    if (spans_ != nullptr) kernels::Free(spans_);
    if (other_ != nullptr) kernels::Free(other_);
    if (counts_ != nullptr) kernels::Free(counts_);
  }
  SpanList(const SpanList&) = delete;
  SpanList& operator=(const SpanList&) = delete;

  const ComputeType* const* spans() const { return spans_; }
  const ComputeType* const* other() const { return other_; }
  const int* counts() const { return counts_; }

 private:
  static ComputeType** StagePointers(const ComputeType* const* source,
                                     int count) {
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(ComputeType*);
    ComputeType** device = static_cast<ComputeType**>(kernels::Alloc(bytes));
    kernels::Memcpy(device, source, bytes, CopyDir::kHostToDevice);
    return device;
  }

  ComputeType** spans_ = nullptr;
  ComputeType** other_ = nullptr;
  int* counts_ = nullptr;
};

// The per-row scalars of a composition. The reduction entries write `float`
// (norm statistics, log-sum-exp results) and the pointwise family is typed on
// `ComputeType`. In the fp32 build the two are one buffer and every sync below
// is a no-op; in a bf16 build the composition keeps both and converts through
// the host, which is the staging gap recorded in the node report (a cast entry
// point in the seam, or a pointwise code that reads a float operand).
class RowScalars {
 public:
  explicit RowScalars(int count) : count_(count), floats_(count) {
    if constexpr (std::is_same_v<ComputeType, float>) {
      compute_ = reinterpret_cast<ComputeType*>(floats_.data());
    } else {
      if (count_ > 0) {
        compute_ = static_cast<ComputeType*>(kernels::Alloc(
            static_cast<std::size_t>(count_) * sizeof(ComputeType)));
      }
    }
  }
  ~RowScalars() {
    if constexpr (!std::is_same_v<ComputeType, float>) {
      if (compute_ != nullptr) kernels::Free(compute_);
    }
  }
  RowScalars(const RowScalars&) = delete;
  RowScalars& operator=(const RowScalars&) = delete;

  float* floats() { return floats_.data(); }
  const float* floats() const { return floats_.data(); }
  ComputeType* compute() { return compute_; }
  int count() const { return count_; }

  // float -> ComputeType (a no-op when the two are one buffer).
  //
  // A bf16 build needs both narrowings, and no code in the seam states them:
  // the pointwise family is typed on `ComputeType` for both operands, so it
  // cannot convert its own operand, and `RowReduce` writes `float` because the
  // norm statistic is a divisor and the tree keeps it in fp32. The conversion
  // therefore stays here, in the one class that owns both buffers, and the fp32
  // build never compiles it. Swapping `--precision=bf16` is the only way a
  // composition reaches these two calls.
  void ToCompute() {
    if constexpr (!std::is_same_v<ComputeType, float>) {
      std::vector<float> host(static_cast<std::size_t>(count_));
      kernels::Memcpy(host.data(), floats_.data(), host.size() * sizeof(float),
                      CopyDir::kDeviceToHost);
      std::vector<ComputeType> wide(host.size());
      for (std::size_t i = 0; i < host.size(); ++i) {
        wide[i] = ToC(host[i]);
      }
      kernels::Memcpy(compute_, wide.data(), wide.size() * sizeof(ComputeType),
                      CopyDir::kHostToDevice);
    }
  }

  // ComputeType -> float (a no-op when the two are one buffer).
  void ToFloat() {
    if constexpr (!std::is_same_v<ComputeType, float>) {
      std::vector<ComputeType> wide(static_cast<std::size_t>(count_));
      kernels::Memcpy(wide.data(), compute_, wide.size() * sizeof(ComputeType),
                      CopyDir::kDeviceToHost);
      std::vector<float> host(wide.size());
      for (std::size_t i = 0; i < wide.size(); ++i) host[i] = AsF(wide[i]);
      kernels::Memcpy(floats_.data(), host.data(), host.size() * sizeof(float),
                      CopyDir::kHostToDevice);
    }
  }

 private:
  int count_ = 0;
  FloatScratch floats_;
  ComputeType* compute_ = nullptr;
};

// ---------------------------------------------------------------------------
// The RMSNorm leg, shared by the norm composition and by `QkPrep`
// ---------------------------------------------------------------------------
// `out[r, c] = rstd[r] * x[r, c]` with `rstd[r] = rsqrt(mean_c(x^2) + eps)`.
// `rows` segments of `dim` elements each, `offset` the first segment's element
// offset inside the buffer, so one implementation serves the whole residual
// stream (`QkPrep`'s head rows are the same shape with `dim = head_dim`).
void NormForward(int rows, int dim, int offset, float eps, const ComputeType* x,
                 ComputeType* out, RowScalars* rstd) {
  // 1. sum of squares per segment.
  RowReduce(ReduceOp::kSumSq, rows, dim, x, rstd->floats(), offset);
  rstd->ToCompute();
  // 2. mean square.
  Pw(PointwiseOp::kScale, rows, rstd->compute(), nullptr, rstd->compute(),
     1.0f / static_cast<float>(dim));
  // 3. + eps, on the scalar. `kAffine` is `alpha * (a - beta)`, so a negated
  // epsilon is the exact `a + eps` and no constant vector is uploaded.
  Pw(PointwiseOp::kAffine, rows, rstd->compute(), nullptr, rstd->compute(),
     1.0f, -eps);
  // 4. reciprocal square root: one declared bound for the code, one rounding.
  Pw(PointwiseOp::kRsqrt, rows, rstd->compute(), nullptr, rstd->compute());
  rstd->ToFloat();
  // 5. the normalized rows, through the per-row broadcast.
  Pw(PointwiseOp::kRowScale, rows * dim, rstd->compute(), x, out, 1.0f, 0.0f,
     /*row_width=*/dim);
}

// The backward of `NormForward`, reading only its declared slots (`x` and
// `rstd`):  dot[r] = sum_c x[r,c] * dy[r,c] dx[r,c] = rstd[r] * dy[r,c] -
// rstd[r]^3 * dot[r] / dim * x[r,c] = rstd[r] * (dy[r,c] - c2[r] * x[r,c]),  c2
// = rstd^2 * dot / dim  The second form is the one the primitive set expresses
// without a second full-size buffer: `dx` holds the per-row product in turn.
// The first pass uses `dx` as the product buffer the per-segment dot needs (the
// seam has no two-operand row reduce; see the node report), and the last pass
// multiplies by `rstd` in place.
void NormBackward(int rows, int dim, float /*eps*/, const ComputeType* x,
                  const ComputeType* dy, const ComputeType* rstd,
                  ComputeType* dx) {
  const int count = rows * dim;
  // dx = x * dy, then the per-segment sum of that product. The product is the
  // whole-buffer case of the row broadcast: `kScaleAdd` is a linear combination
  // (`alpha * a + beta * b`) and cannot form one.
  Mul(x, dy, dx, count);
  RowScalars dot(rows);
  RowReduce(ReduceOp::kSum, rows, dim, dx, dot.floats());
  dot.ToCompute();
  // c2 = rstd^2 * dot / dim, held in a second row vector.
  RowScalars c2(rows);
  // c2 starts as a copy of `rstd`, then squares and multiplies in place.
  Pw(PointwiseOp::kScale, rows, rstd, nullptr, c2.compute(), 1.0f);
  Mul(c2.compute(), rstd, c2.compute(), rows);
  Mul(c2.compute(), dot.compute(), c2.compute(), rows);
  Pw(PointwiseOp::kScale, rows, c2.compute(), nullptr, c2.compute(),
     1.0f / static_cast<float>(dim));
  // dx = rstd * (dy - c2 * x).
  Pw(PointwiseOp::kRowScale, count, c2.compute(), x, dx, 1.0f, 0.0f,
     /*row_width=*/dim);
  Pw(PointwiseOp::kScaleAdd, count, dx, dy, dx, -1.0f, 1.0f);
  Pw(PointwiseOp::kRowScale, count, rstd, dx, dx, 1.0f, 0.0f,
     /*row_width=*/dim);
}

// The rotary leg of `QkPrep`: the half-split pair rotation about the model's
// two tables, and the qk scale, in one `kRotatePair` pass over the head rows of
// `tensor` in place. `heads` is the number of head rows (query or key) and the
// buffers are `[batch, seq, heads, head_dim]`. `cos` and `sin` are the model's
// device tables (`src/model_impl.h`, `cos_table()`), which are already
// `[seq, head_dim / 2]`, so this composes with nothing staged.
void RotatePairs(int count, int head_dim, int heads, int seq, float scale,
                 const float* cos, const float* sin, ComputeType* tensor) {
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", count >= 0);
  if (count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "pointwise", cos != nullptr && sin != nullptr && tensor != nullptr);
  PointwiseParams params;
  params.op = PointwiseOp::kRotatePair;
  params.count = count;
  params.row_width = head_dim;
  params.alpha = scale;
  params.heads = heads;
  params.seq = seq;
  params.cos_table = cos;
  params.sin_table = sin;
  kernels::PointwiseForward(params, tensor, /*b=*/nullptr, tensor);
}

// The transposed rotation: the gradient through the rotation and the scale,
// written to `out` from `grad`. The rotation is linear, so this reads no saved
// activation. `out` may alias `grad`: one thread owns both elements of a pair.
void RotatePairsBackward(int count, int head_dim, int heads, int seq,
                         float scale, const float* cos, const float* sin,
                         const ComputeType* grad, ComputeType* out) {
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", count >= 0);
  if (count <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("pointwise", cos != nullptr && sin != nullptr &&
                                                grad != nullptr &&
                                                out != nullptr);
  PointwiseParams params;
  params.op = PointwiseOp::kRotatePair;
  params.count = count;
  params.row_width = head_dim;
  params.alpha = scale;
  params.heads = heads;
  params.seq = seq;
  params.cos_table = cos;
  params.sin_table = sin;
  // `a` is the saved slot of a pointwise leg and this code does not read it,
  // and the tables come from the parameter block, so `b` is null.
  kernels::PointwiseBackward(params, grad, /*b=*/nullptr, grad, out, nullptr);
}

// The backward of `QkPrep`, over primitives only. `saved` holds the pre-QkPrep
// rows on entry and the input gradients on return (the seam's `QkPrepBackward`
// contract, and the reason `q_pre`/`k_pre` are saved slots); `grad` is the
// upstream gradient. `rot` and `prod` are scratch of `count` elements each.
//
// The arithmetic follows the four steps the fused family performs: the norm
// statistic recomputed from the saved rows exactly as `NormForward` computes
// it, the transposed rotation, the per-row dot of the saved row with the
// rotated gradient, and `dx = rstd * g - coeff * x` with `coeff = rstd^3 * dot
// / dim`. The multiplications keep that order so the value matches the
// definition.
void QkPrepApplyBackward(int count, int segments, int head_dim, int heads,
                         int seq, float eps, float scale, const float* cos,
                         const float* sin, const ComputeType* grad,
                         ComputeType* saved, ComputeType* rot,
                         ComputeType* prod) {
  // 1. the statistic, from the saved rows.
  RowScalars rstd(segments);
  RowReduce(ReduceOp::kSumSq, segments, head_dim, saved, rstd.floats());
  rstd.ToCompute();
  Pw(PointwiseOp::kScale, segments, rstd.compute(), nullptr, rstd.compute(),
     1.0f / static_cast<float>(head_dim));
  Pw(PointwiseOp::kAffine, segments, rstd.compute(), nullptr, rstd.compute(),
     1.0f, -eps);
  Pw(PointwiseOp::kRsqrt, segments, rstd.compute(), nullptr, rstd.compute());
  rstd.ToFloat();
  // 2. the transposed rotation and the qk scale.
  RotatePairsBackward(count, head_dim, heads, seq, scale, cos, sin, grad, rot);
  // 3. dot[r] = sum_c saved[r,c] * rot[r,c]. The seam has no two-operand
  // per-row reduce, so the product is materialized first; `prod` is reused
  // below once the reduction has read it.
  Mul(saved, rot, prod, count);
  RowScalars dot(segments);
  RowReduce(ReduceOp::kSum, segments, head_dim, prod, dot.floats());
  dot.ToCompute();
  // 4. coeff = rstd^3 * dot / dim, in the same order the fused family uses.
  RowScalars coeff(segments);
  Pw(PointwiseOp::kScale, segments, rstd.compute(), nullptr, coeff.compute(),
     1.0f);
  Mul(coeff.compute(), rstd.compute(), coeff.compute(), segments);
  Mul(coeff.compute(), rstd.compute(), coeff.compute(), segments);
  Mul(coeff.compute(), dot.compute(), coeff.compute(), segments);
  Pw(PointwiseOp::kScale, segments, coeff.compute(), nullptr, coeff.compute(),
     1.0f / static_cast<float>(head_dim));
  // 5. dx = rstd * g - coeff * x, into the caller's saved slot.
  Pw(PointwiseOp::kRowScale, count, rstd.compute(), rot, rot, 1.0f, 0.0f,
     /*row_width=*/head_dim);
  Pw(PointwiseOp::kRowScale, count, coeff.compute(), saved, prod, 1.0f, 0.0f,
     /*row_width=*/head_dim);
  Pw(PointwiseOp::kScaleAdd, count, rot, prod, saved, 1.0f, -1.0f);
}

}  // namespace

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

void LinearForward(const ComputeType* x, const ComputeType* w, ComputeType* y,
                   std::int64_t rows, std::int64_t in, std::int64_t out,
                   bool accumulate, const Fp8Scratch* fp8) {
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm", rows >= 0 && in >= 0 && out >= 0);
  if (rows <= 0 || in <= 0 || out <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm",
                               x != nullptr && w != nullptr && y != nullptr);
  (void)fp8;  // unused in a dense build
  if constexpr (kFp8Build) {
    // The eligible matmuls take fp8 operands with an fp32 accumulator
    // (docs/fp8-training.md sections 6 and 7). Quantize both operands for this
    // call. The reuse of an operand the same step already quantized is a later
    // optimization (docs/fp8-training.md section 7).
    if (fp8 != nullptr && fp8->ready() && Fp8Eligible(out, in)) {
      QuantizeFp8Params quantize;
      quantize.format = Fp8Format::kE4M3;
      quantize.count = static_cast<int>(rows * in);
      kernels::QuantizeFp8(quantize, x, fp8->a_bytes, fp8->a_scale);
      quantize.count = static_cast<int>(out * in);
      kernels::QuantizeFp8(quantize, w, fp8->b_bytes, fp8->b_scale);
      ScaledGemmParams scaled;
      scaled.m = static_cast<int>(rows);
      scaled.n = static_cast<int>(out);
      scaled.k = static_cast<int>(in);
      scaled.alpha = 1.0f;
      scaled.beta = accumulate ? 1.0f : 0.0f;
      // The weight is stored [out, in] (PyTorch's Linear convention).
      scaled.transpose_b = true;
      // The forward uses the reduced-precision reduction
      // (docs/fp8-training.md section 5.2). Fast accumulation is an Ada (8.9)
      // feature: Hopper (9.0) rejects the descriptor attribute, and
      // `backends/cuda/gemm.cu` exits when it is asked for the flag there.
      // Ask for it only where the device accepts it.
      static const bool kFastAccum = [] {
        const Caps caps = kernels::GetCaps();
        return caps.compute_major == 8 && caps.compute_minor == 9;
      }();
      scaled.fast_accum = kFastAccum;
      scaled.format_a = Fp8Format::kE4M3;
      scaled.format_b = Fp8Format::kE4M3;
      scaled.scale_a = fp8->a_scale;
      scaled.scale_b = fp8->b_scale;
      kernels::GemmScaledFp8(GemmMode::kForward, scaled, fp8->a_bytes,
                             fp8->b_bytes, y);
      return;
    }
  }
  GemmParams params;
  params.m = static_cast<int>(rows);
  params.n = static_cast<int>(out);
  params.k = static_cast<int>(in);
  params.alpha = 1.0f;
  params.beta = accumulate ? 1.0f : 0.0f;
  // Weights are stored [out, in] (PyTorch's Linear convention); compute
  // x @ w^T by transposing the second operand.
  params.transpose_b = true;
  kernels::Gemm(GemmMode::kForward, params, x, w, y);
}

void LinearWgrad(const ComputeType* x, const ComputeType* dy, float* dw,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 const Fp8Scratch* fp8) {
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm", rows >= 0 && in >= 0 && out >= 0);
  if (rows <= 0 || in <= 0 || out <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm",
                               x != nullptr && dy != nullptr && dw != nullptr);
  (void)fp8;  // unused in a dense build
  if constexpr (kFp8Build) {
    // A = the output gradient (E5M2), B = the saved activation (E4M3). The
    // output is the fp32 parameter gradient (docs/fp8-training.md section 7).
    if (fp8 != nullptr && fp8->ready() && Fp8Eligible(out, in)) {
      QuantizeFp8Params quantize;
      quantize.format = Fp8Format::kE5M2;
      quantize.count = static_cast<int>(rows * out);
      kernels::QuantizeFp8(quantize, dy, fp8->a_bytes, fp8->a_scale);
      quantize.format = Fp8Format::kE4M3;
      quantize.count = static_cast<int>(rows * in);
      kernels::QuantizeFp8(quantize, x, fp8->b_bytes, fp8->b_scale);
      ScaledGemmParams scaled;
      scaled.m = static_cast<int>(out);
      scaled.n = static_cast<int>(in);
      scaled.k = static_cast<int>(rows);
      scaled.alpha = 1.0f;
      scaled.beta = 1.0f;
      scaled.transpose_a = true;
      scaled.fast_accum = false;
      scaled.format_a = Fp8Format::kE5M2;
      scaled.format_b = Fp8Format::kE4M3;
      scaled.scale_a = fp8->a_scale;
      scaled.scale_b = fp8->b_scale;
      kernels::GemmScaledFp8(GemmMode::kWgrad, scaled, fp8->a_bytes,
                             fp8->b_bytes, dw);
      return;
    }
  }
  // dw[out, in] += dy[rows, out]^T @ x[rows, in].
  GemmParams params;
  params.m = static_cast<int>(out);
  params.n = static_cast<int>(in);
  params.k = static_cast<int>(rows);
  params.alpha = 1.0f;
  params.beta = 1.0f;
  params.transpose_a = true;
  kernels::GemmWgrad(params, dy, x, dw);
}

void LinearDgrad(const ComputeType* dy, const ComputeType* w, ComputeType* dx,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 bool accumulate, const Fp8Scratch* fp8) {
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm", rows >= 0 && in >= 0 && out >= 0);
  if (rows <= 0 || in <= 0 || out <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("gemm",
                               dy != nullptr && w != nullptr && dx != nullptr);
  (void)fp8;  // unused in a dense build
  if constexpr (kFp8Build) {
    // A = the output gradient (E5M2), B = the weight (E4M3). The output is an
    // activation, so it stores the compute type (docs/fp8-training.md
    // section 7).
    if (fp8 != nullptr && fp8->ready() && Fp8Eligible(out, in)) {
      QuantizeFp8Params quantize;
      quantize.format = Fp8Format::kE5M2;
      quantize.count = static_cast<int>(rows * out);
      kernels::QuantizeFp8(quantize, dy, fp8->a_bytes, fp8->a_scale);
      quantize.format = Fp8Format::kE4M3;
      quantize.count = static_cast<int>(out * in);
      kernels::QuantizeFp8(quantize, w, fp8->b_bytes, fp8->b_scale);
      ScaledGemmParams scaled;
      scaled.m = static_cast<int>(rows);
      scaled.n = static_cast<int>(in);
      scaled.k = static_cast<int>(out);
      scaled.alpha = 1.0f;
      scaled.beta = accumulate ? 1.0f : 0.0f;
      scaled.transpose_a = false;
      scaled.transpose_b = false;
      scaled.fast_accum = false;
      scaled.format_a = Fp8Format::kE5M2;
      scaled.format_b = Fp8Format::kE4M3;
      scaled.scale_a = fp8->a_scale;
      scaled.scale_b = fp8->b_scale;
      kernels::GemmScaledFp8(GemmMode::kDgrad, scaled, fp8->a_bytes,
                             fp8->b_bytes, dx);
      return;
    }
  }
  // dx[rows, in] = dy[rows, out] @ w[out, in].
  GemmParams params;
  params.m = static_cast<int>(rows);
  params.n = static_cast<int>(in);
  params.k = static_cast<int>(out);
  params.alpha = 1.0f;
  params.beta = accumulate ? 1.0f : 0.0f;
  kernels::Gemm(GemmMode::kDgrad, params, dy, w, dx);
}

// ---------------------------------------------------------------------------
// RMSNorm, as a composition over RowReduce and Pointwise
// ---------------------------------------------------------------------------

void RmsNormForward(std::int64_t rows, std::int64_t dim, float eps,
                    const ComputeType* x, ComputeType* out, float* rstd) {
  NANOCHAT_CHECK_KERNEL_PARAMS("rms_norm", rows >= 0 && dim >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("rms_norm", std::isfinite(eps));
  if (rows <= 0 || dim <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "rms_norm", x != nullptr && out != nullptr && rstd != nullptr);
  if (Fused(kFeatureOverrideRmsNorm)) {
    RmsNormParams params;
    params.rows = static_cast<int>(rows);
    params.dim = static_cast<int>(dim);
    params.eps = eps;
    kernels::RmsNormForward(params, x, out, rstd);
    return;
  }
  // The saved slot is the caller's `float* rstd`; the composition keeps a
  // ComputeType view of it for the pointwise steps and copies the answer into
  // the caller's slot at the end.
  const int n = static_cast<int>(rows);
  RowScalars rstd_rows(n);
  NormForward(n, static_cast<int>(dim), 0, eps, x, out, &rstd_rows);
  kernels::Memcpy(rstd, rstd_rows.floats(),
                  static_cast<std::size_t>(n) * sizeof(float),
                  CopyDir::kHostToDevice);
}

void RmsNormBackward(std::int64_t rows, std::int64_t dim, float eps,
                     const ComputeType* x, const ComputeType* dy,
                     const float* rstd, ComputeType* dx) {
  NANOCHAT_CHECK_KERNEL_PARAMS("rms_norm", rows >= 0 && dim >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("rms_norm", std::isfinite(eps));
  if (rows <= 0 || dim <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("rms_norm", x != nullptr && dy != nullptr &&
                                               rstd != nullptr &&
                                               dx != nullptr);
  if (Fused(kFeatureOverrideRmsNorm)) {
    RmsNormParams params;
    params.rows = static_cast<int>(rows);
    params.dim = static_cast<int>(dim);
    params.eps = eps;
    kernels::RmsNormBackward(params, x, dy, rstd, dx);
    return;
  }
  const int n = static_cast<int>(rows);
  // The declared slots are `x` and `rstd`; the pointwise backward reads them
  // through a ComputeType view of the saved statistics.
  RowScalars rstd_rows(n);
  kernels::Memcpy(rstd_rows.floats(), rstd,
                  static_cast<std::size_t>(n) * sizeof(float),
                  CopyDir::kDeviceToHost);
  rstd_rows.ToCompute();
  NormBackward(n, static_cast<int>(dim), eps, x, dy, rstd_rows.compute(), dx);
}

// ---------------------------------------------------------------------------
// The training loss, reduced on the device
// ---------------------------------------------------------------------------
// One entry, one meaning: add this micro-batch's mean loss to a device scalar.
// The per-row losses are fp32 in every build, so the fp32 entry serves a bf16
// build too and the rows never reach the host (`docs/scalar-arguments.md`).

void LossSumForward(const float* losses, std::int64_t rows,
                    std::int64_t divisor, bool accumulate, float* out) {
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", rows >= 0 && divisor >= 0);
  if (rows <= 0 || divisor <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions",
                               losses != nullptr && out != nullptr);
  kernels::SumFp32(losses, static_cast<int>(rows), out,
                   1.0 / static_cast<double>(divisor), accumulate);
}

// ---------------------------------------------------------------------------
// QK-norm + RoPE + scale: both legs over primitives. The rotation is
// `kRotatePair` about the model's two tables, so this composition runs entirely
// on the device and moves nothing to the host.
// ---------------------------------------------------------------------------

void QkPrepForward(int batch, int seq, int num_heads, int num_kv_heads,
                   int head_dim, float eps, float scale, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep",
                               batch >= 0 && seq >= 0 && num_heads >= 0 &&
                                   num_kv_heads >= 0 && head_dim >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep",
                               std::isfinite(eps) && std::isfinite(scale));
  // The rotation is a half-split pair rotation, so the head width is even.
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", head_dim % 2 == 0);
  if (batch <= 0 || seq <= 0 || head_dim <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", cos != nullptr && sin != nullptr);
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", num_heads == 0 || q != nullptr);
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", num_kv_heads == 0 || k != nullptr);
  if (Fused(kFeatureOverrideQkPrep)) {
    QkPrepParams params;
    params.batch = batch;
    params.seq = seq;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.eps = eps;
    params.scale = scale;
    kernels::QkPrepForward(params, cos, sin, q, k);
    return;
  }
  const int rows = batch * seq;
  struct Pass {
    int heads;
    ComputeType* tensor;
  };
  const Pass passes[2] = {{num_heads, q}, {num_kv_heads, k}};
  for (const Pass& pass : passes) {
    if (pass.tensor == nullptr || pass.heads <= 0) continue;
    // The norm leg: one reduction over each head row, then the per-row
    // broadcast. `head_rows` segments of `head_dim` elements, which is exactly
    // the shape `RowReduce` takes.
    RowScalars rstd(rows * pass.heads);
    NormForward(rows * pass.heads, head_dim, 0, eps, pass.tensor, pass.tensor,
                &rstd);
    // The rotation leg: the pair rotation and the qk scale in one pass, on the
    // device, about the model's own tables. Nothing is staged.
    RotatePairs(rows * pass.heads * head_dim, head_dim, pass.heads, seq, scale,
                cos, sin, pass.tensor);
  }
}

void QkPrepBackward(int batch, int seq, int num_heads, int num_kv_heads,
                    int head_dim, float eps, float scale, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep",
                               batch >= 0 && seq >= 0 && num_heads >= 0 &&
                                   num_kv_heads >= 0 && head_dim >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep",
                               std::isfinite(eps) && std::isfinite(scale));
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", head_dim % 2 == 0);
  if (batch <= 0 || seq <= 0 || head_dim <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("qk_prep", cos != nullptr && sin != nullptr);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "qk_prep", num_heads == 0 || (q != nullptr && dq != nullptr));
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "qk_prep", num_kv_heads == 0 || (k != nullptr && dk != nullptr));
  if (Fused(kFeatureOverrideQkPrep)) {
    QkPrepParams params;
    params.batch = batch;
    params.seq = seq;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.eps = eps;
    params.scale = scale;
    kernels::QkPrepBackward(params, cos, sin, dq, dk, q, k);
    return;
  }
  struct Pass {
    int heads;
    const ComputeType* grad;
    ComputeType* saved;
  };
  const Pass passes[2] = {{num_heads, dq, q}, {num_kv_heads, dk, k}};
  const int rows = batch * seq;
  // One scratch pair, wide enough for the larger of the query and key passes,
  // and reused by both. The transposed rotation needs a buffer of its own
  // because `grad` is const, and the row dot needs a product buffer because the
  // seam has no two-operand per-row reduce; neither is host traffic.
  const int widest = std::max(num_heads, num_kv_heads);
  const int scratch_count = rows * widest * head_dim;
  ComputeScratch scratch(2 * scratch_count);
  ComputeType* rot = scratch.data();
  ComputeType* prod = scratch.data() + scratch_count;
  for (const Pass& pass : passes) {
    if (pass.saved == nullptr || pass.grad == nullptr || pass.heads <= 0) {
      continue;
    }
    const int count = rows * pass.heads * head_dim;
    QkPrepApplyBackward(count, rows * pass.heads, head_dim, pass.heads, seq,
                        eps, scale, cos, sin, pass.grad, pass.saved, rot, prod);
  }
}

// ---------------------------------------------------------------------------
// Attention: the documented exception (`spec/block_kinds.md` section 2). Its
// iteration space is T x T, no composition of the other primitives reproduces
// its tiling without materializing a T x T tensor, so the primitive is called
// directly. See `spec/decisions.md` D-008.
// ---------------------------------------------------------------------------

namespace {

AttentionParams ToParams(const AttentionShape& shape) {
  AttentionParams params;
  params.batch = shape.batch;
  params.seq = shape.seq;
  params.num_heads = shape.num_heads;
  params.num_kv_heads = shape.num_kv_heads;
  params.head_dim = shape.head_dim;
  params.causal = shape.causal;
  params.window_left = shape.window_left;
  params.window_right = shape.window_right;
  params.kv_len = shape.kv_len;
  params.scale = shape.scale;
  return params;
}

}  // namespace

void AttentionForward(const AttentionShape& shape, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention", shape.batch >= 0 && shape.seq >= 0 && shape.num_heads >= 0 &&
                       shape.num_kv_heads >= 0 && shape.head_dim >= 0 &&
                       shape.kv_len >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention", shape.window_left >= -1 && shape.window_right >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("attention", std::isfinite(shape.scale));
  if (shape.batch <= 0 || shape.seq <= 0 || shape.num_heads <= 0 ||
      shape.head_dim <= 0) {
    return;
  }
  // One key/value head serves `num_heads / num_kv_heads` query heads, so the
  // group count must divide evenly.
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention",
      shape.num_kv_heads > 0 && shape.num_heads % shape.num_kv_heads == 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("attention",
                               q != nullptr && k != nullptr && v != nullptr &&
                                   out != nullptr && stats != nullptr);
  const AttentionParams params = ToParams(shape);
  kernels::AttentionForward(params, q, k, v, out, stats);
}

void AttentionBackward(const AttentionShape& shape, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention", shape.batch >= 0 && shape.seq >= 0 && shape.num_heads >= 0 &&
                       shape.num_kv_heads >= 0 && shape.head_dim >= 0 &&
                       shape.kv_len >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention", shape.window_left >= -1 && shape.window_right >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("attention", std::isfinite(shape.scale));
  if (shape.batch <= 0 || shape.seq <= 0 || shape.num_heads <= 0 ||
      shape.head_dim <= 0) {
    return;
  }
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention",
      shape.num_kv_heads > 0 && shape.num_heads % shape.num_kv_heads == 0);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "attention", q != nullptr && k != nullptr && v != nullptr &&
                       stats != nullptr && dout != nullptr && dq != nullptr &&
                       dk != nullptr && dv != nullptr);
  const AttentionParams params = ToParams(shape);
  kernels::AttentionBackward(params, q, k, v, stats, dout, dq, dk, dv);
}

// ---------------------------------------------------------------------------
// ResFormer value gate, as a composition
// ---------------------------------------------------------------------------
// forward: pre[row, kvh] = h[row, :G] @ gate_w[kvh, :]        (Gemm) gate[row,
// kvh] = 3 * sigmoid(pre)                 (kGateMul + kScale) v[row, kvh, :] +=
// gate[row, kvh] * ve[row, kvh, :] (kRowScale, the caller accumulates)
// backward: dve = gate * dv                                  (kRowScale)
// dgate[row, kvh] = sum_d dv * ve                  (kRowScale product +
// RowReduce kSum) dpre = dgate * gate * (1 - gate/3)               (pointwise
// codes) gate_w_grad[kvh, :] += dpre^T @ h[:, :G]         (GemmWgrad) dh[row,
// :G] += dpre @ gate_w                     (Gemm, accumulate)

void ValueGateForward(std::int64_t rows, int hidden, int num_kv_heads,
                      const ComputeType* h, const ComputeType* gate_w,
                      ComputeType* gate_out) {
  constexpr int kChannels = kVeGateChannels;
  NANOCHAT_CHECK_KERNEL_PARAMS("value_gate",
                               rows >= 0 && hidden >= 0 && num_kv_heads >= 0);
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "value_gate", h != nullptr && gate_w != nullptr && gate_out != nullptr);
  const int n = static_cast<int>(rows);
  // The gate dot: one GEMM over the leading `G` channels of every hidden row.
  // `lda = hidden` because the rows of `h` are `hidden` wide and only their
  // first `G` columns are the operand.
  RowScalars pre(n * num_kv_heads);
  {
    GemmParams params;
    params.m = n;
    params.n = num_kv_heads;
    params.k = kChannels;
    params.lda = hidden;
    params.transpose_b = true;
    params.alpha = 1.0f;
    params.beta = 0.0f;
    kernels::Gemm(GemmMode::kForward, params, h, gate_w, pre.compute());
  }
  // gate = 3 * sigmoid(pre): `kSigmoid` is the bare sigmoid, and the gate
  // factor `m` is the following `kScale` rather than a third meaning for a
  // code's alpha (`spec/pointwise_ops.md` section 3).
  Pw(PointwiseOp::kSigmoid, n * num_kv_heads, pre.compute(), nullptr, gate_out);
  Pw(PointwiseOp::kScale, n * num_kv_heads, gate_out, nullptr, gate_out, 3.0f);
}

void ValueGateApply(std::int64_t rows, int num_kv_heads, int head_dim,
                    const ComputeType* gate, const ComputeType* ve,
                    ComputeType* v) {
  NANOCHAT_CHECK_KERNEL_PARAMS("value_gate",
                               rows >= 0 && num_kv_heads >= 0 && head_dim >= 0);
  if (rows <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "value_gate", gate != nullptr && ve != nullptr && v != nullptr);
  const int count = static_cast<int>(rows) * num_kv_heads * head_dim;
  // The per-head broadcast: `gate` holds one scalar per (row, key/value head)
  // and `ve` is the dense operand, so the row width is the head width.
  Pw(PointwiseOp::kRowScale, count, gate, ve, v, 1.0f, 0.0f,
     /*row_width=*/head_dim);
}

void ValueGateBackward(std::int64_t rows, int hidden, int num_kv_heads,
                       int head_dim, const ComputeType* h,
                       const ComputeType* ve, const ComputeType* gate_w,
                       const ComputeType* gate, const ComputeType* dv,
                       float* gate_w_grad, ComputeType* dh, ComputeType* dve) {
  constexpr int kChannels = kVeGateChannels;
  NANOCHAT_CHECK_KERNEL_PARAMS("value_gate", rows >= 0 && hidden >= 0 &&
                                                 num_kv_heads >= 0 &&
                                                 head_dim >= 0);
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "value_gate", h != nullptr && ve != nullptr && gate_w != nullptr &&
                        gate != nullptr && dv != nullptr &&
                        gate_w_grad != nullptr && dh != nullptr &&
                        dve != nullptr);
  const int n = static_cast<int>(rows);
  const int count = n * num_kv_heads * head_dim;
  // dve = gate * dv, and the same product feeds the gate's own gradient, so it
  // is computed first and reduced per (row, head) before `dve` is final.
  Mul(dv, ve, dve, count);
  RowScalars dgate(n * num_kv_heads);
  RowReduce(ReduceOp::kSum, n * num_kv_heads, head_dim, dve, dgate.floats());
  dgate.ToCompute();
  // dve = gate * dv (the product above was a scratch use of the output slot).
  Pw(PointwiseOp::kRowScale, count, gate, dv, dve, 1.0f, 0.0f,
     /*row_width=*/head_dim);

  // dpre = dgate * gate * (1 - gate/3). The `1 - x` is `kAffine` with a
  // negated alpha, so no unit vector is uploaded.
  RowScalars dpre(n * num_kv_heads);
  Pw(PointwiseOp::kScale, n * num_kv_heads, gate, nullptr, dpre.compute(),
     1.0f / 3.0f);
  Pw(PointwiseOp::kAffine, n * num_kv_heads, dpre.compute(), nullptr,
     dpre.compute(), -1.0f, 1.0f);
  Mul(dpre.compute(), gate, dpre.compute(), n * num_kv_heads);
  Mul(dpre.compute(), dgate.compute(), dpre.compute(), n * num_kv_heads);

  // gate_w_grad[kvh, :G] += dpre^T @ h[:, :G].
  {
    GemmParams params;
    params.m = num_kv_heads;
    params.n = kChannels;
    params.k = n;
    params.transpose_a = true;
    params.ldb = hidden;
    params.alpha = 1.0f;
    params.beta = 1.0f;
    kernels::GemmWgrad(params, dpre.compute(), h, gate_w_grad);
  }
  // dh[:, :G] += dpre @ gate_w.
  {
    GemmParams params;
    params.m = n;
    params.n = kChannels;
    params.k = num_kv_heads;
    params.ldc = hidden;
    params.alpha = 1.0f;
    params.beta = 1.0f;
    kernels::Gemm(GemmMode::kForward, params, dpre.compute(), gate_w, dh);
  }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

void MlpForward(const ComputeType* h, const ComputeType* c_fc,
                const ComputeType* c_proj, std::int64_t rows,
                std::int64_t hidden, std::int64_t mlp_dim, ComputeType* pre_act,
                ComputeType* act, ComputeType* scratch,
                const ComputeType* x_mid, ComputeType* out,
                const Fp8Scratch* fp8) {
  if (rows <= 0 || hidden <= 0 || mlp_dim <= 0) return;
  LinearForward(h, c_fc, pre_act, rows, hidden, mlp_dim, false, fp8);
  Pw(PointwiseOp::kReluSquare, static_cast<int>(rows * mlp_dim), pre_act,
     nullptr, act);
  LinearForward(act, c_proj, scratch, rows, mlp_dim, hidden, false, fp8);
  Pw(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), scratch, x_mid,
     out, 1.0f, 1.0f);
}

void MlpBackward(const ComputeType* h, const ComputeType* c_fc,
                 const ComputeType* c_proj, const ComputeType* pre_act,
                 const ComputeType* act, const ComputeType* dout,
                 std::int64_t rows, std::int64_t hidden, std::int64_t mlp_dim,
                 float* c_fc_grad, float* c_proj_grad, ComputeType* dh,
                 ComputeType* dact, ComputeType* dpre, const Fp8Scratch* fp8) {
  if (rows <= 0 || hidden <= 0 || mlp_dim <= 0) return;
  LinearWgrad(act, dout, c_proj_grad, rows, mlp_dim, hidden, fp8);
  LinearDgrad(dout, c_proj, dact, rows, mlp_dim, hidden, false, fp8);
  // `relu^2`'s backward needs the sign and the value of the pre-activation, so
  // `act` is derivable from the declared `pre_act` slot and is not declared
  // (`spec/block_kinds.md` section 2).
  PwBack(PointwiseOp::kReluSquare, static_cast<int>(rows * mlp_dim), pre_act,
         nullptr, dact, dpre, nullptr);
  LinearWgrad(h, dpre, c_fc_grad, rows, hidden, mlp_dim, fp8);
  LinearDgrad(dpre, c_fc, dh, rows, hidden, mlp_dim, false, fp8);
}

// ---------------------------------------------------------------------------
// Smear
// ---------------------------------------------------------------------------
// The family is the seam's: a device kernel owns a row, a host loop walks rows
// in order, and this file only names the shape. There is no second
// implementation here (`docs/kernels.md`, the smear row).
namespace {

kernels::SmearParams SmearParamsOf(std::int64_t batch, std::int64_t seq,
                                   std::int64_t hidden, ScalarArg lambda) {
  kernels::SmearParams params;
  params.batch = static_cast<int>(batch);
  params.seq = static_cast<int>(seq);
  params.hidden = static_cast<int>(hidden);
  params.channels = kSmearChannels;
  params.lambda = lambda;
  return params;
}

}  // namespace

void SmearForward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                  const ComputeType* emb_norm, const ComputeType* gate_w,
                  ScalarArg lambda, ComputeType* x0, float* sig) {
  if (batch <= 0 || seq <= 0 || hidden <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("smear", hidden >= kSmearChannels);
  kernels::SmearForward(SmearParamsOf(batch, seq, hidden, lambda), emb_norm,
                        gate_w, x0, sig);
}

void SmearBackward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                   const ComputeType* emb_norm, const ComputeType* gate_w,
                   const float* sig, ScalarArg lambda, const ComputeType* d_x0,
                   float* gate_w_grad, float* lambda_grad, ComputeType* de) {
  if (batch <= 0 || seq <= 0 || hidden <= 0) return;
  NANOCHAT_CHECK_KERNEL_PARAMS("smear", hidden >= kSmearChannels);
  kernels::SmearBackward(SmearParamsOf(batch, seq, hidden, lambda), emb_norm,
                         gate_w, sig, d_x0, gate_w_grad, lambda_grad, de);
}

// ---------------------------------------------------------------------------
// Backout
// ---------------------------------------------------------------------------

void BackoutForward(const ComputeType* x, const ComputeType* x_backout,
                    ScalarArg neg_lambda, ComputeType* out, std::int64_t n) {
  if (n <= 0) return;
  PwScalars(PointwiseOp::kScaleAdd, static_cast<int>(n), x, x_backout, out,
            ScalarArg(1.0f), neg_lambda);
}

void BackoutBackward(const ComputeType* d_pre, const ComputeType* x_backout,
                     ScalarArg neg_lambda, ComputeType* dbackout,
                     float* lambda_grad, std::int64_t n) {
  if (n <= 0) return;
  // The elementwise scale runs through the pointwise family; the scalar dot
  // that the lambda gradient needs is a `SpanReduce` over two ragged lists of
  // one span each, which is the composition `GradNorm` and `ScalarDot` share.
  // The reduce's own scale stays a host constant: it is `-1` on every call and
  // is not the coefficient.
  PwScalars(PointwiseOp::kScale, static_cast<int>(n), d_pre, nullptr, dbackout,
            neg_lambda, ScalarArg(0.0f));
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", d_pre != nullptr &&
                                                 x_backout != nullptr &&
                                                 lambda_grad != nullptr);
  const ComputeType* spans[1] = {d_pre};
  const ComputeType* other[1] = {x_backout};
  const int counts[1] = {static_cast<int>(n)};
  const SpanList list(spans, other, counts, 1);
  SpanReduceParams params;
  params.num_spans = 1;
  params.counts = list.counts();
  params.scale = -1.0f;
  params.accumulate = true;
  kernels::SpanReduce(params, list.spans(), list.other(), lambda_grad);
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------
// Per row: the softcapped logit row (`kSoftcap`), its log-sum-exp (`RowReduce`
// kMax + kExp + kSum), and the target's value (the `Embedding` per-row gather
// at a width of one, which is the addressable-element primitive the seam
// already has). The scalars the two need are copied to the host, which is a
// constant-size round trip per row and not a staging pass over the logits.  The
// row-at-a-time walk is the composition's shape, not a limitation of the
// primitives: the logits rows are `padded_vocab_size` wide and the pointwise
// family is one flat count, so a row at a time is what keeps the composition
// inside the caller's stride without a copy of the whole logits buffer. The
// per-row host scalars are the gap the node report records (a pointwise code
// that reads a float operand, or a `RowReduce` that writes the row's target).

void ClassifierForward(const ClassifierShape& shape, const ComputeType* logits,
                       const int* targets, float* losses) {
  const int rows = shape.rows;
  const int vocab = shape.vocab_size;
  const int padded =
      shape.padded_vocab_size > 0 ? shape.padded_vocab_size : vocab;
  NANOCHAT_CHECK_KERNEL_PARAMS("classifier", rows >= 0 && vocab >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("classifier", padded >= vocab);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "classifier", std::isfinite(shape.softcap) && shape.softcap > 0.0f);
  if (rows <= 0 || vocab <= 0 || logits == nullptr || targets == nullptr ||
      losses == nullptr) {
    return;
  }
  if (Fused(kFeatureOverrideClassifier) && FusedClassifierTakesScale(shape)) {
    // One kernel over every row. The composition below costs four launches and
    // two blocking device-to-host copies per row, which is what made a step at
    // 4096 token positions take seconds while the device sat idle.
    ClassifierParams params;
    params.rows = rows;
    params.vocab_size = vocab;
    params.padded_vocab_size = padded;
    params.softcap = shape.softcap;
    params.ignore_index = shape.ignore_index;
    params.row_scale = shape.row_scale_dev;
    kernels::ClassifierForward(params, logits, targets, losses);
    return;
  }
  ComputeScratch row_buffer(vocab);  // the softcapped logit row
  ComputeScratch gathered(1);        // the target's softcapped value
  FloatScratch lse(1);               // the row's log-sum-exp
  for (int r = 0; r < rows; ++r) {
    if (targets[r] == shape.ignore_index) {
      StoreScalar(losses + r, 0.0f);
      continue;
    }
    const ComputeType* row = logits + static_cast<std::int64_t>(r) * padded;
    // 1. the softcapped row, read in the caller's stride.
    Pw(PointwiseOp::kSoftcap, vocab, row, nullptr, row_buffer.data(),
       shape.softcap);
    // 2. the log-sum-exp of that row, max-stable inside the reduction.
    RowReduce(ReduceOp::kLogSumExp, 1, vocab, row_buffer.data(), lse.data());
    // 3. the target's softcapped value, through the per-row gather. An id
    // outside the vocabulary contributes a zero, which the caller cannot
    // produce (the targets come from the tokenizer), so it is a defensive
    // branch rather than a documented mode.
    float target_value = 0.0f;
    if (targets[r] >= 0 && targets[r] < vocab) {
      NANOCHAT_CHECK_KERNEL_PARAMS(
          "embedding", targets + r != nullptr && row_buffer.data() != nullptr &&
                           gathered.data() != nullptr);
      kernels::EmbeddingForward(1, 1, targets + r, row_buffer.data(),
                                gathered.data());
      // Stage the gathered element at the compute type's own width. A copy of
      // `sizeof(ComputeType)` bytes into a `float` leaves the high bytes of
      // the value stale in a bf16 build, so the loss reads `lse` minus a wrong
      // number.
      ComputeType staged{};
      kernels::Memcpy(&staged, gathered.data(), sizeof(ComputeType),
                      CopyDir::kDeviceToHost);
      target_value = AsF(staged);
    }
    // 4. loss = log-sum-exp - z[target], the same order the fused family uses.
    StoreScalar(losses + r, StageScalar(lse.data()) - target_value);
  }
}

void ClassifierBackward(const ClassifierShape& shape, const ComputeType* logits,
                        const int* targets, ComputeType* dlogits) {
  const int rows = shape.rows;
  const int vocab = shape.vocab_size;
  const int padded =
      shape.padded_vocab_size > 0 ? shape.padded_vocab_size : vocab;
  NANOCHAT_CHECK_KERNEL_PARAMS("classifier", rows >= 0 && vocab >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("classifier", padded >= vocab);
  NANOCHAT_CHECK_KERNEL_PARAMS(
      "classifier", std::isfinite(shape.softcap) && shape.softcap > 0.0f);
  if (rows <= 0 || vocab <= 0 || logits == nullptr || targets == nullptr ||
      dlogits == nullptr) {
    return;
  }
  if (Fused(kFeatureOverrideClassifier) && FusedClassifierTakesScale(shape)) {
    // The fused backward zeroes the padded tail and every ignored row, which is
    // the same contract the composition keeps.
    ClassifierParams params;
    params.rows = rows;
    params.vocab_size = vocab;
    params.padded_vocab_size = padded;
    params.softcap = shape.softcap;
    params.ignore_index = shape.ignore_index;
    params.row_scale = shape.row_scale_dev;
    kernels::ClassifierBackward(params, logits, targets, dlogits);
    return;
  }
  FloatScratch row_max(1);
  FloatScratch sum_exp(1);
  ComputeScratch z(vocab);      // the softcapped row
  ComputeScratch e(vocab);      // exp(z - max)
  ComputeScratch sech2(vocab);  // 1 - (z / cap)^2, the softcap's derivative
  ComputeScratch u(vocab);      // (p - onehot) * sech2 * row_scale
  ComputeScratch gathered(1);   // the target's sech2
  // `kAffine` reads `beta` through `ScalarArg`, so the row maximum stays on the
  // device: `alpha * (a - beta)` is `z - max` with no host read of the
  // reduction's result.
  ScalarArg row_max_arg;
  row_max_arg.dev = row_max.data();
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * padded;
    if (targets[r] == shape.ignore_index) {
      kernels::Memset(dlogits + base, 0,
                      static_cast<std::size_t>(padded) * sizeof(ComputeType));
      continue;
    }
    const ComputeType* row = logits + base;
    // z = cap * tanh(x / cap);  sech2 = 1 - (z / cap)^2.
    Pw(PointwiseOp::kSoftcap, vocab, row, nullptr, z.data(), shape.softcap);
    Pw(PointwiseOp::kScale, vocab, z.data(), nullptr, sech2.data(),
       1.0f / shape.softcap);
    Mul(sech2.data(), sech2.data(), sech2.data(), vocab);
    Pw(PointwiseOp::kAffine, vocab, sech2.data(), nullptr, sech2.data(), -1.0f,
       1.0f);
    // max, then e = exp(z - max). The shift is `kAffine` with the maximum as a
    // device scalar argument, which is what the stable log-sum-exp needs.
    RowReduce(ReduceOp::kMax, 1, vocab, z.data(), row_max.data());
    PwScalars(PointwiseOp::kAffine, vocab, z.data(), nullptr, e.data(),
              ScalarArg(1.0f), row_max_arg);
    Pw(PointwiseOp::kExp, vocab, e.data(), nullptr, e.data());
    RowReduce(ReduceOp::kSum, 1, vocab, e.data(), sum_exp.data());
    const float inv = 1.0f / StageScalar(sum_exp.data());
    // p = e * inv, then (p) * sech2 * row_scale.
    Pw(PointwiseOp::kScale, vocab, e.data(), nullptr, u.data(), inv);
    Mul(u.data(), sech2.data(), u.data(), vocab);
    const float scale = shape.row_scale != nullptr ? shape.row_scale[r] : 1.0f;
    if (scale != 1.0f) {
      Pw(PointwiseOp::kScale, vocab, u.data(), nullptr, u.data(), scale);
    }
    // The one-hot correction: element `target` loses `sech2[target] * scale`.
    // The element is addressed by the same per-row gather the forward uses.
    float sech_target = 0.0f;
    if (targets[r] >= 0 && targets[r] < vocab) {
      NANOCHAT_CHECK_KERNEL_PARAMS("embedding", targets + r != nullptr &&
                                                    sech2.data() != nullptr &&
                                                    gathered.data() != nullptr);
      kernels::EmbeddingForward(1, 1, targets + r, sech2.data(),
                                gathered.data());
      ComputeType staged{};
      kernels::Memcpy(&staged, gathered.data(), sizeof(ComputeType),
                      CopyDir::kDeviceToHost);
      sech_target = AsF(staged);
    }
    ComputeType staged_current{};
    kernels::Memcpy(&staged_current, u.data() + targets[r], sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
    const float current = AsF(staged_current) - sech_target * scale;
    const ComputeType corrected = ToC(current);
    kernels::Memcpy(u.data() + targets[r], &corrected, sizeof(ComputeType),
                    CopyDir::kHostToDevice);
    // Write the row and zero the padded tail (the classifier's own contract:
    // the backward zeroes the padded vocabulary tail).
    kernels::Memcpy(dlogits + base, u.data(),
                    static_cast<std::size_t>(vocab) * sizeof(ComputeType),
                    CopyDir::kDeviceToDevice);
    if (padded > vocab) {
      kernels::Memset(
          dlogits + base + vocab, 0,
          static_cast<std::size_t>(padded - vocab) * sizeof(ComputeType));
    }
  }
}

// ---------------------------------------------------------------------------
// Gradient norm
// ---------------------------------------------------------------------------//
// The reduction is `kernels::GradNorm`, the seam's fp32 span entry: it walks
// the span table on the device, accumulates in double, and writes the norm and
// the clip scale to device buffers. `GradientType` is fp32 in every build
// (`include/nanochat/tensor.h`), and that entry's span list is fp32 by
// contract, so one call serves both precisions.
//
// `SpanReduce` cannot serve this reduction: its operand is `ComputeType`, which
// is bf16 in a bf16 build. Reducing through it therefore needed the gradient
// spans on the host, at one blocking copy and one host scalar pass per span;
// `profiles/h100-base-20step/REPORT.md` section 4.1 measures that cost.
//
// The norm and the clip scale stay on the device for the whole step. Every
// update kernel reads the scale there, so no host read stands between the
// reduction and the updates, and the norm comes back once at the end of the
// step.
void GradNormForward(const float* const* grads, const int* counts,
                     int num_spans, float clip, float* out_norm,
                     float* out_scale) {
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", num_spans >= 0);
  NANOCHAT_CHECK_KERNEL_PARAMS("reductions", std::isfinite(clip));
  if (grads == nullptr || counts == nullptr || num_spans <= 0) return;
  kernels::GradNorm(grads, counts, num_spans, clip, out_norm, out_scale);
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

void BlockForward(const BlockShape& shape, const BlockWeights& weights,
                  const BlockActivations& acts, const int* tokens,
                  const float* cos, const float* sin, const ComputeType* x,
                  const ComputeType* x0, ScalarArg resid, ScalarArg x0_lambda) {
  const std::int64_t rows = static_cast<std::int64_t>(shape.batch) * shape.seq;
  const int hidden = shape.hidden;
  const int heads = shape.num_heads;
  const int kv_heads = shape.num_kv_heads;
  const int head_dim = shape.head_dim;
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = 4 * hidden;
  if (rows <= 0 || hidden <= 0 || heads <= 0 || head_dim <= 0) return;

  // Residual blend, pre-attention norm, then q/k/v projections.
  PwScalars(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), x, x0,
            acts.xr, resid, x0_lambda);
  RmsNormForward(rows, hidden, shape.rms_eps, acts.xr, acts.h, acts.rstd1);

  LinearForward(acts.h, weights.c_q, acts.q_final, rows, hidden, query_dim,
                false, &acts.fp8);
  if (acts.q_pre != nullptr && acts.q_pre != acts.q_final) {
    kernels::Memcpy(
        acts.q_pre, acts.q_final,
        static_cast<std::size_t>(rows * query_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }
  LinearForward(acts.h, weights.c_k, acts.k_final, rows, hidden, kv_dim, false,
                &acts.fp8);
  if (acts.k_pre != nullptr && acts.k_pre != acts.k_final) {
    kernels::Memcpy(
        acts.k_pre, acts.k_final,
        static_cast<std::size_t>(rows * kv_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }

  if (weights.has_ve) {
    // The gate composition writes `gate * ve` into the value rows first, and
    // the value projection accumulates on top of it: `v = c_v(h) + gate * ve`
    // with one add, which is the same rounding the fused family produces
    // without a second full-size buffer.
    kernels::EmbeddingForward(static_cast<int>(rows), kv_dim, tokens,
                              weights.value_embeds, acts.ve_values);
    ValueGateForward(rows, hidden, kv_heads, acts.h, weights.ve_gate,
                     acts.ve_gate);
    ValueGateApply(rows, kv_heads, head_dim, acts.ve_gate, acts.ve_values,
                   acts.v_final);
    LinearForward(acts.h, weights.c_v, acts.v_final, rows, hidden, kv_dim, true,
                  &acts.fp8);
  } else {
    LinearForward(acts.h, weights.c_v, acts.v_final, rows, hidden, kv_dim,
                  false, &acts.fp8);
  }

  QkPrepForward(shape.batch, shape.seq, heads, kv_heads, head_dim,
                shape.rms_eps, shape.qk_scale, cos, sin, acts.q_final,
                acts.k_final);

  // Inference: append the current key/value rows to the KV cache before
  // attention reads the cache back.
  if (acts.cache_k != nullptr) {
    kernels::Memcpy(
        acts.cache_k + static_cast<std::int64_t>(acts.cache_offset) * kv_dim,
        acts.k_final,
        static_cast<std::size_t>(rows * kv_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }
  if (acts.cache_v != nullptr) {
    kernels::Memcpy(
        acts.cache_v + static_cast<std::int64_t>(acts.cache_offset) * kv_dim,
        acts.v_final,
        static_cast<std::size_t>(rows * kv_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }

  AttentionShape attn;
  attn.batch = shape.batch;
  attn.seq = shape.seq;
  attn.num_heads = heads;
  attn.num_kv_heads = kv_heads;
  attn.head_dim = head_dim;
  attn.causal = shape.causal;
  attn.window_left = shape.window_left;
  attn.window_right = shape.window_right;
  attn.kv_len = shape.kv_len;
  attn.scale = shape.attn_scale;
  const ComputeType* attn_k =
      acts.attn_k != nullptr ? acts.attn_k : acts.k_final;
  const ComputeType* attn_v =
      acts.attn_v != nullptr ? acts.attn_v : acts.v_final;
  AttentionForward(attn, acts.q_final, attn_k, attn_v, acts.attn_out,
                   acts.attn_stats);

  LinearForward(acts.attn_out, weights.c_proj_attn, acts.scratch_a, rows,
                hidden, hidden, false, &acts.fp8);
  Pw(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), acts.scratch_a,
     acts.xr, acts.x_mid, 1.0f, 1.0f);

  RmsNormForward(rows, hidden, shape.rms_eps, acts.x_mid, acts.h2, acts.rstd2);
  MlpForward(acts.h2, weights.c_fc, weights.c_proj_mlp, rows, hidden, mlp_dim,
             acts.pre_act, acts.act, acts.scratch_b, acts.x_mid, acts.x_out,
             &acts.fp8);
}

void BlockBackward(const BlockShape& shape, const BlockWeights& weights,
                   const BlockActivations& acts, const BlockGrads& grads,
                   const BlockScratch& scratch, const int* tokens,
                   const float* cos, const float* sin, ScalarArg resid,
                   ScalarArg x0_lambda, const ComputeType* x_in,
                   const ComputeType* x0, const ComputeType* dout,
                   ComputeType* dx_in, ComputeType* x0_acc, float* resid_grad,
                   float* x0_lambda_grad) {
  const std::int64_t rows = static_cast<std::int64_t>(shape.batch) * shape.seq;
  const int hidden = shape.hidden;
  const int heads = shape.num_heads;
  const int kv_heads = shape.num_kv_heads;
  const int head_dim = shape.head_dim;
  const int query_dim = heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int mlp_dim = 4 * hidden;
  if (rows <= 0 || hidden <= 0 || heads <= 0 || head_dim <= 0) return;

  // --- MLP: x_out = x_mid + c_proj(relu^2(c_fc(h2))). ---
  MlpBackward(acts.h2, weights.c_fc, weights.c_proj_mlp, acts.pre_act, acts.act,
              dout, rows, hidden, mlp_dim, grads.c_fc_grad,
              grads.c_proj_mlp_grad, scratch.dh2, scratch.dact, scratch.dpre,
              &scratch.fp8);
  RmsNormBackward(rows, hidden, shape.rms_eps, acts.x_mid, scratch.dh2,
                  acts.rstd2, scratch.dx_mid);
  Pw(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), dout,
     scratch.dx_mid, scratch.dx_mid, 1.0f, 1.0f);

  // --- Attention: x_mid = xr + c_proj(attention(q, k, v)). ---
  LinearWgrad(acts.attn_out, scratch.dx_mid, grads.c_proj_attn_grad, rows,
              hidden, hidden, &scratch.fp8);
  LinearDgrad(scratch.dx_mid, weights.c_proj_attn, scratch.dy_attn, rows,
              hidden, hidden, false, &scratch.fp8);

  AttentionShape attn;
  attn.batch = shape.batch;
  attn.seq = shape.seq;
  attn.num_heads = heads;
  attn.num_kv_heads = kv_heads;
  attn.head_dim = head_dim;
  attn.causal = shape.causal;
  attn.window_left = shape.window_left;
  attn.window_right = shape.window_right;
  attn.kv_len = shape.kv_len;
  attn.scale = shape.attn_scale;
  AttentionBackward(attn, acts.q_final, acts.k_final, acts.v_final,
                    acts.attn_stats, scratch.dy_attn, scratch.dq, scratch.dk,
                    scratch.dv);

  // Value projection and the ResFormer gate.
  LinearWgrad(acts.h, scratch.dv, grads.c_v_grad, rows, hidden, kv_dim,
              &scratch.fp8);
  LinearDgrad(scratch.dv, weights.c_v, scratch.dh, rows, hidden, kv_dim, false,
              &scratch.fp8);
  if (weights.has_ve) {
    ValueGateBackward(rows, hidden, kv_heads, head_dim, acts.h, acts.ve_values,
                      weights.ve_gate, acts.ve_gate, scratch.dv,
                      grads.ve_gate_grad, scratch.dh, scratch.dve);
    kernels::EmbeddingBackward(static_cast<int>(rows), kv_dim, tokens,
                               scratch.dve, grads.value_embeds_grad);
  }

  // QK-norm + RoPE + scale, then the query/key projections.
  QkPrepBackward(shape.batch, shape.seq, heads, kv_heads, head_dim,
                 shape.rms_eps, shape.qk_scale, cos, sin, scratch.dq,
                 scratch.dk, acts.q_pre, acts.k_pre);
  LinearWgrad(acts.h, acts.q_pre, grads.c_q_grad, rows, hidden, query_dim,
              &scratch.fp8);
  LinearDgrad(acts.q_pre, weights.c_q, scratch.dh, rows, hidden, query_dim,
              true, &scratch.fp8);
  LinearWgrad(acts.h, acts.k_pre, grads.c_k_grad, rows, hidden, kv_dim,
              &scratch.fp8);
  LinearDgrad(acts.k_pre, weights.c_k, scratch.dh, rows, hidden, kv_dim, true,
              &scratch.fp8);

  // Pre-attention norm, then the residual blend xr = resid*x + x0_lambda*x0.
  RmsNormBackward(rows, hidden, shape.rms_eps, acts.xr, scratch.dh, acts.rstd1,
                  scratch.dxr);
  Pw(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), scratch.dx_mid,
     scratch.dxr, scratch.dxr, 1.0f, 1.0f);

  // dx_in = resid * dxr; x0_acc += x0_lambda * dxr. Both are elementwise on
  // the device. The two scalar gradients are device reductions, so the block
  // backward stays entirely on the GPU.
  PwScalars(PointwiseOp::kScale, static_cast<int>(rows * hidden), scratch.dxr,
            nullptr, dx_in, resid, ScalarArg(0.0f));
  PwScalars(PointwiseOp::kScaleAdd, static_cast<int>(rows * hidden), x0_acc,
            scratch.dxr, x0_acc, ScalarArg(1.0f), x0_lambda);
  {
    const ComputeType* spans[1] = {scratch.dxr};
    const ComputeType* other[1] = {x_in};
    const ComputeType* other_x0[1] = {x0};
    const int counts[1] = {static_cast<int>(rows * hidden)};
    const SpanList list(spans, other, counts, 1);
    const SpanList list_x0(spans, other_x0, counts, 1);
    SpanReduceParams params;
    params.num_spans = 1;
    params.counts = list.counts();
    params.scale = 1.0f;
    params.accumulate = true;
    kernels::SpanReduce(params, list.spans(), list.other(), resid_grad);
    kernels::SpanReduce(params, list_x0.spans(), list_x0.other(),
                        x0_lambda_grad);
  }
}

}  // namespace ops
}  // namespace nanochat
