// backends/cpu/kernels.cc — the reference implementation of every symbol in
// nanochat/kernels.h (docs/backends.md). It is the correctness baseline, the
// oracle-on-CPU path, and the fallback when no accelerator is present.
//
// Everything here is a naive loop, written for clarity rather than speed:
// double-width accumulators for reductions, float working state, and no
// vectorisation. The build selects ComputeType (fp32 or fp16) at compile time;
// the arithmetic below is always performed in float and converted at the
// storage boundary through AsFloat / ToCompute.
//
// Layout conventions (matched to the host graphs and docs/model.md):
//   * Q/K/V activations are [batch, seq, heads, head_dim] row-major, the same
//     native layout nanochat's SDPA fallback uses (B, T, H, D).
//   * Attention statistics are [batch, num_heads, seq, 2] with (max, sum_exp).
//   * QkPrep proves the RMSNorm -> RoPE -> scale fusion. Because RoPE is a
//     rotation, it commutes with RMSNorm exactly, so the fusion is numerically
//     equivalent to nanochat's RoPE -> RMSNorm order (docs/kernels.md).
//
// QkPrep backward contract: the `q`/`k` buffers passed to QkPrepBackward hold
// the *saved pre-norm projection outputs* (the activations the model kept for
// backward), and are overwritten with the gradient with respect to those
// projection outputs. The upstream gradients `dq`/`dk` are taken with respect
// to the final (RoPE'd, scaled) activation. Saving the pre-norm rows is what
// makes the normalization statistics recoverable, which is why the header says
// the backward "reads the saved forward outputs ... to recover the
// normalization statistics".

#include "nanochat/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace nanochat {
namespace kernels {
namespace {

// Storage <-> float boundary. In an fp16 build the optimizer state and the
// reductions still run in float (docs/precision.md). The conversion helpers are
// templated on the storage type so that `if constexpr` discards the unused
// branch instead of requiring both to compile.
template <typename T>
inline float AsFloatT(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return Fp16ToFloat(v);
  }
}

template <typename T>
inline T ToComputeT(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return Fp16FromFloat(v);
  }
}

inline float AsFloat(ComputeType v) { return AsFloatT(v); }

inline ComputeType ToCompute(float v) { return ToComputeT<ComputeType>(v); }

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Zero a typed ComputeType range without tripping -Wclass-memaccess in an
// fp16 build (Fp16 is a non-trivial struct).
inline void ZeroFill(ComputeType* ptr, std::size_t count) {
  std::fill(ptr, ptr + count, ToCompute(0.0f));
}

// Polar Express coefficients (num_iters=5), from nanochat/optim.py / the
// Polar Express paper (arXiv:2505.16932).
constexpr float kPolarCoeffs[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

}  // namespace

// ---------------------------------------------------------------------------
// Device / memory (backend-owned)
// ---------------------------------------------------------------------------

void* Alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  return std::malloc(bytes);
}

void Free(void* ptr) { std::free(ptr); }

void Memcpy(void* dst, const void* src, std::size_t bytes, CopyDir dir) {
  // Every copy direction is a plain memcpy on the CPU reference backend.
  (void)dir;
  if (bytes == 0) return;
  std::memcpy(dst, src, bytes);
}

void Memset(void* ptr, int value, std::size_t bytes) {
  if (bytes == 0) return;
  std::memset(ptr, value, bytes);
}

void Synchronize() {
  // The reference backend is synchronous: every entry point returns after its
  // work is complete.
}

Caps GetCaps() {
  Caps caps;
  caps.device_index = 0;
  caps.compute_major = 0;
  caps.compute_minor = 0;
  caps.total_memory_bytes = 0;
  caps.is_device = false;
  caps.has_cublas = false;
  caps.has_tensor_cores = false;
  // The CPU reference implements the build precision. It can emulate both, so
  // both are reported as supported; the host fails fast on a build whose
  // precision is not in this set.
  caps.supports_fp32 = true;
  caps.supports_fp16 = true;
  caps.device_name = "cpu";
  return caps;
}

// ---------------------------------------------------------------------------
// Library-backed GEMM
// ---------------------------------------------------------------------------
//
// Universal row-major GEMM: C = alpha * op(A) * op(B) + beta * C, repeated
// `batch_count` times with per-batch strides. `transpose_a`/`transpose_b`
// control the operand layout; `lda`/`ldb`/`ldc` are leading dimensions, where 0
// infers a dense row-major operand. `GemmMode` is advisory: the host supplies
// the already-selected operands, so all three modes compute the same product.
void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c) {
  (void)mode;
  const int m = params.m;
  const int n = params.n;
  const int k = params.k;
  if (m <= 0 || n <= 0 || k <= 0) return;

  // Stored extents (before the transpose flag) and inferred leading dims.
  const int rows_a = params.transpose_a ? k : m;
  const int cols_a = params.transpose_a ? m : k;
  const int rows_b = params.transpose_b ? n : k;
  const int cols_b = params.transpose_b ? k : n;
  const int lda = params.lda > 0 ? params.lda : cols_a;
  const int ldb = params.ldb > 0 ? params.ldb : cols_b;
  const int ldc = params.ldc > 0 ? params.ldc : n;

  const std::int64_t stride_a = params.stride_a > 0
                                    ? params.stride_a
                                    : static_cast<std::int64_t>(rows_a) * lda;
  const std::int64_t stride_b = params.stride_b > 0
                                    ? params.stride_b
                                    : static_cast<std::int64_t>(rows_b) * ldb;
  const std::int64_t stride_c = params.stride_c > 0
                                    ? params.stride_c
                                    : static_cast<std::int64_t>(m) * ldc;
  const int batch = params.batch_count > 0 ? params.batch_count : 1;

  for (int bi = 0; bi < batch; ++bi) {
    const ComputeType* ab = a + bi * stride_a;
    const ComputeType* bb = b + bi * stride_b;
    ComputeType* cb = c + bi * stride_c;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int l = 0; l < k; ++l) {
          const float av =
              params.transpose_a
                  ? AsFloat(ab[static_cast<std::int64_t>(l) * lda + i])
                  : AsFloat(ab[static_cast<std::int64_t>(i) * lda + l]);
          const float bv =
              params.transpose_b
                  ? AsFloat(bb[static_cast<std::int64_t>(j) * ldb + l])
                  : AsFloat(bb[static_cast<std::int64_t>(l) * ldb + j]);
          acc += static_cast<double>(av) * static_cast<double>(bv);
        }
        const std::int64_t ci = static_cast<std::int64_t>(i) * ldc + j;
        const float prior = (params.beta != 0.0f) ? AsFloat(cb[ci]) : 0.0f;
        cb[ci] = ToCompute(params.alpha * static_cast<float>(acc) +
                           params.beta * prior);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

void RmsNormForward(const RmsNormParams& params, const ComputeType* x,
                    ComputeType* out, float* rstd) {
  const int rows = params.rows;
  const int dim = params.dim;
  if (rows <= 0 || dim <= 0) return;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * dim;
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = AsFloat(x[base + d]);
      sum_sq += v * v;
    }
    const float r_row =
        1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
    for (int d = 0; d < dim; ++d) {
      out[base + d] = ToCompute(AsFloat(x[base + d]) * r_row);
    }
    rstd[r] = r_row;
  }
}

void RmsNormBackward(const RmsNormParams& params, const ComputeType* x,
                     const ComputeType* dy, const float* rstd,
                     ComputeType* dx) {
  const int rows = params.rows;
  const int dim = params.dim;
  if (rows <= 0 || dim <= 0) return;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * dim;
    const float r_row = rstd[r];
    double dot = 0.0;
    for (int d = 0; d < dim; ++d) {
      dot += static_cast<double>(AsFloat(x[base + d])) *
             static_cast<double>(AsFloat(dy[base + d]));
    }
    const float coeff =
        r_row * r_row * r_row * static_cast<float>(dot) * inv_dim;
    for (int d = 0; d < dim; ++d) {
      const float xv = AsFloat(x[base + d]);
      dx[base + d] = ToCompute(r_row * AsFloat(dy[base + d]) - coeff * xv);
    }
  }
}

// ---------------------------------------------------------------------------
// QkPrep (RMSNorm -> RoPE -> scale)
// ---------------------------------------------------------------------------

namespace {

// Applies RMSNorm, RoPE, then scale to one [batch, seq, heads, head_dim] tensor
// in place. Shared by the q and k passes of QkPrepForward.
void QkPrepApply(const QkPrepParams& params, const float* cos, const float* sin,
                 ComputeType* tensor, int heads) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float scale = params.scale;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::int64_t base =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        double sum_sq = 0.0;
        for (int d = 0; d < dim; ++d) {
          const double v = AsFloat(tensor[base + d]);
          sum_sq += v * v;
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
        for (int d = 0; d < dim; ++d) {
          tensor[base + d] = ToCompute(AsFloat(tensor[base + d]) * r);
        }
        const std::int64_t row_cos = static_cast<std::int64_t>(t) * half;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = AsFloat(tensor[base + d]);
          const float x2 = AsFloat(tensor[base + half + d]);
          tensor[base + d] = ToCompute((x1 * c + x2 * s) * scale);
          tensor[base + half + d] = ToCompute((-x1 * s + x2 * c) * scale);
        }
      }
    }
  }
}

// Gradients of one QkPrep tensor. `tensor` holds the saved pre-norm rows on
// entry and the input gradients on return; `grad` is the upstream gradient
// w.r.t. the final activation.
void QkPrepApplyBackward(const QkPrepParams& params, const float* cos,
                         const float* sin, const ComputeType* grad,
                         ComputeType* tensor, int heads) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float scale = params.scale;
  std::vector<float> g(dim);
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::int64_t base =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        // Recover the RMSNorm statistic from the saved pre-norm row, and pull
        // the upstream gradient back through scale and RoPE.
        const std::int64_t row_cos = static_cast<std::int64_t>(t) * half;
        double sum_sq = 0.0;
        double dot = 0.0;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = AsFloat(tensor[base + d]);
          const float x2 = AsFloat(tensor[base + half + d]);
          const float g1 = AsFloat(grad[base + d]);
          const float g2 = AsFloat(grad[base + half + d]);
          // dL/d(normed) = scale * R^T(dL/d(final)).
          g[d] = scale * (g1 * c - g2 * s);
          g[half + d] = scale * (g1 * s + g2 * c);
          sum_sq += static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
          dot += static_cast<double>(x1) * g[d] +
                 static_cast<double>(x2) * g[half + d];
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
        const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;
        for (int d = 0; d < dim; ++d) {
          tensor[base + d] =
              ToCompute(r * g[d] - coeff * AsFloat(tensor[base + d]));
        }
      }
    }
  }
}

}  // namespace

void QkPrepForward(const QkPrepParams& params, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  QkPrepApply(params, cos, sin, q, params.num_heads);
  QkPrepApply(params, cos, sin, k, params.num_kv_heads);
}

void QkPrepBackward(const QkPrepParams& params, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  QkPrepApplyBackward(params, cos, sin, dq, q, params.num_heads);
  QkPrepApplyBackward(params, cos, sin, dk, k, params.num_kv_heads);
}

// ---------------------------------------------------------------------------
// Attention (causal / sliding-window / group-query)
// ---------------------------------------------------------------------------

namespace {

// Key/value head for query head `h` under GQA. Contiguous grouping, matching
// PyTorch's enable_gqa (each key/value head serves `group` query heads).
inline int KvHead(int h, int num_heads, int num_kv_heads) {
  if (num_kv_heads <= 0 || num_kv_heads >= num_heads) return h;
  const int group = num_heads / num_kv_heads;
  const int kv = h / (group > 0 ? group : 1);
  return std::min(kv, num_kv_heads - 1);
}

// Whether query row at absolute position `qpos` may attend to key `j`.
inline bool KeyAllowed(const AttentionParams& p, std::int64_t qpos,
                       std::int64_t j) {
  if (p.causal && j > qpos) return false;
  if (p.window_left >= 0 && qpos - j > p.window_left) return false;
  if (p.window_right >= 0 && j - qpos > p.window_right) return false;
  return true;
}

}  // namespace

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));
  std::vector<float> acc(dim);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kvh = KvHead(h, heads, kv_heads);
      for (int t = 0; t < seq; ++t) {
        const std::int64_t qpos = kv_len - seq + t;
        const std::int64_t qbase =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        // Pass 1: max score over the visible window.
        float row_max = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (std::int64_t j = 0; j < kv_len; ++j) {
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(AsFloat(q[qbase + d])) *
                   static_cast<double>(AsFloat(k[kbase + d]));
          }
          const float score = scale * static_cast<float>(dot);
          if (score > row_max) row_max = score;
          any = true;
        }
        const std::int64_t stat_base =
            ((static_cast<std::int64_t>(b) * heads + h) * seq + t) * 2;
        if (!any) {
          for (int d = 0; d < dim; ++d) out[qbase + d] = ToCompute(0.0f);
          stats[stat_base] = 0.0f;
          stats[stat_base + 1] = 0.0f;
          continue;
        }
        // Pass 2: sum of exponentials and unnormalised output.
        double sum_exp = 0.0;
        for (int d = 0; d < dim; ++d) acc[d] = 0.0f;
        for (std::int64_t j = 0; j < kv_len; ++j) {
          if (!KeyAllowed(params, qpos, j)) continue;
          const std::int64_t kbase =
              ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
              dim;
          const std::int64_t vbase = kbase;
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(AsFloat(q[qbase + d])) *
                   static_cast<double>(AsFloat(k[kbase + d]));
          }
          const double weight = std::exp(
              static_cast<double>(scale * static_cast<float>(dot) - row_max));
          sum_exp += weight;
          for (int d = 0; d < dim; ++d) {
            acc[d] += static_cast<float>(weight * AsFloat(v[vbase + d]));
          }
        }
        const float inv = static_cast<float>(1.0 / sum_exp);
        for (int d = 0; d < dim; ++d) {
          out[qbase + d] = ToCompute(acc[d] * inv);
        }
        stats[stat_base] = row_max;
        stats[stat_base + 1] = static_cast<float>(sum_exp);
      }
    }
  }
}

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));

  const std::size_t q_count =
      static_cast<std::size_t>(batch) * seq * heads * dim;
  const std::size_t kv_count =
      static_cast<std::size_t>(batch) * kv_len * kv_heads * dim;
  ZeroFill(dq, q_count);
  ZeroFill(dk, kv_count);
  ZeroFill(dv, kv_count);

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
            dot += static_cast<double>(AsFloat(q[qbase + d])) *
                   static_cast<double>(AsFloat(k[kbase + d]));
          }
          const float p =
              std::exp(scale * static_cast<float>(dot) - row_max) * inv;
          probs[j] = p;
          double dpd = 0.0;
          for (int d = 0; d < dim; ++d) {
            dpd += static_cast<double>(AsFloat(dout[qbase + d])) *
                   static_cast<double>(AsFloat(v[kbase + d]));
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
            const float qv = AsFloat(q[qbase + d]);
            const float kv = AsFloat(k[kbase + d]);
            const float dov = AsFloat(dout[qbase + d]);
            dq[qbase + d] = ToCompute(AsFloat(dq[qbase + d]) + dp_scale * kv);
            dk[kbase + d] = ToCompute(AsFloat(dk[kbase + d]) + dp_scale * qv);
            dv[kbase + d] = ToCompute(AsFloat(dv[kbase + d]) + p * dov);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Pointwise
// ---------------------------------------------------------------------------

void PointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                      const ComputeType* b, float alpha, float beta,
                      ComputeType* out) {
  for (int i = 0; i < n; ++i) {
    const float av = AsFloat(a[i]);
    float result = 0.0f;
    switch (op) {
      case PointwiseOp::kScale:
        result = alpha * av;
        break;
      case PointwiseOp::kScaleAdd:
        result = alpha * av + beta * AsFloat(b[i]);
        break;
      case PointwiseOp::kGateMul:
        result = Sigmoid(av) * AsFloat(b[i]);
        break;
      case PointwiseOp::kReluSquare: {
        const float relu = av > 0.0f ? av : 0.0f;
        result = relu * relu;
        break;
      }
      case PointwiseOp::kSoftcap: {
        const float cap = alpha;
        result = cap * std::tanh(av / cap);
        break;
      }
    }
    out[i] = ToCompute(result);
  }
}

void PointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy, float alpha,
                       float beta, ComputeType* da, ComputeType* db) {
  for (int i = 0; i < n; ++i) {
    const float av = AsFloat(a[i]);
    const float dyv = AsFloat(dy[i]);
    switch (op) {
      case PointwiseOp::kScale:
        if (da != nullptr) da[i] = ToCompute(alpha * dyv);
        break;
      case PointwiseOp::kScaleAdd:
        if (da != nullptr) da[i] = ToCompute(alpha * dyv);
        if (db != nullptr) db[i] = ToCompute(beta * dyv);
        break;
      case PointwiseOp::kGateMul: {
        const float sig = Sigmoid(av);
        const float bv = AsFloat(b[i]);
        if (da != nullptr) da[i] = ToCompute(dyv * bv * sig * (1.0f - sig));
        if (db != nullptr) db[i] = ToCompute(dyv * sig);
        break;
      }
      case PointwiseOp::kReluSquare:
        if (da != nullptr) {
          da[i] = ToCompute(av > 0.0f ? dyv * 2.0f * av : 0.0f);
        }
        break;
      case PointwiseOp::kSoftcap: {
        if (da != nullptr) {
          const float cap = alpha;
          const float t = std::tanh(av / cap);
          da[i] = ToCompute(dyv * (1.0f - t * t));
        }
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Classifier (logit softcap + cross-entropy + vocab slice)
// ---------------------------------------------------------------------------

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses) {
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  if (rows <= 0 || vocab <= 0) return;
  const float cap = params.softcap;
  for (int r = 0; r < rows; ++r) {
    const int target = targets[r];
    if (target == params.ignore_index) {
      losses[r] = ToCompute(0.0f);
      continue;
    }
    const std::int64_t base = static_cast<std::int64_t>(r) * padded;
    // Softcap only the real vocabulary; the padded tail is never read.
    float row_max = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(AsFloat(logits[base + j]) / cap);
      if (z > row_max) row_max = z;
    }
    double sum_exp = 0.0;
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(AsFloat(logits[base + j]) / cap);
      sum_exp += std::exp(static_cast<double>(z - row_max));
    }
    const float z_target =
        cap * std::tanh(AsFloat(logits[base + target]) / cap);
    const float loss =
        static_cast<float>(std::log(sum_exp)) + row_max - z_target;
    losses[r] = ToCompute(loss);
  }
}

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits) {
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  if (rows <= 0 || vocab <= 0) return;
  const float cap = params.softcap;
  std::vector<float> probs(vocab, 0.0f);
  std::vector<float> sech2(vocab, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * padded;
    const int target = targets[r];
    if (target == params.ignore_index) {
      for (int j = 0; j < padded; ++j) dlogits[base + j] = ToCompute(0.0f);
      continue;
    }
    float row_max = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < vocab; ++j) {
      const float z = cap * std::tanh(AsFloat(logits[base + j]) / cap);
      probs[j] = z;
      if (z > row_max) row_max = z;
    }
    double sum_exp = 0.0;
    for (int j = 0; j < vocab; ++j) {
      sum_exp += std::exp(static_cast<double>(probs[j] - row_max));
    }
    const float inv = static_cast<float>(1.0 / sum_exp);
    for (int j = 0; j < vocab; ++j) {
      const float t = std::tanh(AsFloat(logits[base + j]) / cap);
      sech2[j] = 1.0f - t * t;
      const float p = std::exp(probs[j] - row_max) * inv;
      const float onehot = (j == target) ? 1.0f : 0.0f;
      // dL/d(raw logit) = (softmax - onehot) * d(softcap)/d(raw).
      dlogits[base + j] = ToCompute((p - onehot) * sech2[j]);
    }
    for (int j = vocab; j < padded; ++j) dlogits[base + j] = ToCompute(0.0f);
  }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out) {
  if (tokens <= 0 || dim <= 0) return;
  for (int i = 0; i < tokens; ++i) {
    const std::int64_t src = static_cast<std::int64_t>(ids[i]) * dim;
    const std::int64_t dst = static_cast<std::int64_t>(i) * dim;
    for (int d = 0; d < dim; ++d) out[dst + d] = table[src + d];
  }
}

void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, ComputeType* dtable) {
  if (tokens <= 0 || dim <= 0) return;
  // Zero only the rows this batch touches (docs/model.md). Doing this in a
  // first pass makes duplicate ids safe: every touched row is zeroed before any
  // contribution is added.
  for (int i = 0; i < tokens; ++i) {
    ComputeType* row = dtable + static_cast<std::int64_t>(ids[i]) * dim;
    ZeroFill(row, static_cast<std::size_t>(dim));
  }
  for (int i = 0; i < tokens; ++i) {
    ComputeType* row = dtable + static_cast<std::int64_t>(ids[i]) * dim;
    const std::int64_t src = static_cast<std::int64_t>(i) * dim;
    for (int d = 0; d < dim; ++d) {
      row[d] = ToCompute(AsFloat(row[d]) + AsFloat(dout[src + d]));
    }
  }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

void AdamWUpdate(int n, const AdamWParams& params, ComputeType* p,
                 const ComputeType* g, float* m, float* v) {
  if (n <= 0) return;
  const float bias1 =
      1.0f - std::pow(params.beta1, static_cast<float>(params.step));
  const float bias2 =
      1.0f - std::pow(params.beta2, static_cast<float>(params.step));
  const float step_size = params.lr / bias1;
  for (int i = 0; i < n; ++i) {
    const float grad = AsFloat(g[i]);
    // Decoupled weight decay, applied to the parameter before the update.
    float pi = AsFloat(p[i]) * (1.0f - params.lr * params.weight_decay);
    const float mi = m[i] + (1.0f - params.beta1) * (grad - m[i]);
    const float vi = v[i] + (1.0f - params.beta2) * (grad * grad - v[i]);
    m[i] = mi;
    v[i] = vi;
    const float denom = std::sqrt(vi / bias2) + params.eps;
    pi -= step_size * (mi / denom);
    p[i] = ToCompute(pi);
  }
}

// ---------------------------------------------------------------------------
// Muon (momentum -> Polar Express -> variance reduction -> cautious update)
// ---------------------------------------------------------------------------

void MuonUpdate(const MuonParams& params, const ComputeType* stacked_grads,
                ComputeType* stacked_params, float* buf1, float* buf2) {
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;
  const std::size_t mat = static_cast<std::size_t>(rows) * cols;
  const int ns_steps = std::min(std::max(params.ns_steps, 0), 5);
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);

  const int k_extent = std::min(rows, cols);
  std::vector<float> x(mat, 0.0f);
  std::vector<float> a_mat(static_cast<std::size_t>(k_extent) * k_extent, 0.0f);
  std::vector<float> a2_mat(static_cast<std::size_t>(k_extent) * k_extent,
                            0.0f);
  std::vector<float> b_mat(static_cast<std::size_t>(k_extent) * k_extent, 0.0f);
  std::vector<float> prod(mat, 0.0f);

  for (int m = 0; m < num_params; ++m) {
    const std::size_t off = static_cast<std::size_t>(m) * mat;
    const ComputeType* grad = stacked_grads + off;
    ComputeType* param = stacked_params + off;
    float* momentum_buf = buf1 + off;
    float* second_buf =
        buf2 + static_cast<std::size_t>(m) * (reduce_cols ? rows : cols);

    // Nesterov momentum: update the first moment, then the accelerated
    // gradient.
    for (std::size_t i = 0; i < mat; ++i) {
      const float gr = AsFloat(grad[i]);
      momentum_buf[i] =
          momentum_buf[i] + (1.0f - params.momentum) * (gr - momentum_buf[i]);
      x[i] = params.nesterov ? (1.0f - params.momentum) * gr +
                                   params.momentum * momentum_buf[i]
                             : momentum_buf[i];
    }

    // MuonEq row equilibration: rescale each row to the mean row norm.
    {
      double frob_sq = 0.0;
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float target = static_cast<float>(std::sqrt(frob_sq)) /
                           std::sqrt(static_cast<float>(rows));
      for (int r = 0; r < rows; ++r) {
        double row_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          row_sq += static_cast<double>(xv) * xv;
        }
        const float row_norm =
            std::max(static_cast<float>(std::sqrt(row_sq)), 1e-6f);
        const float s = target / row_norm;
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *= s;
        }
      }
    }

    // Normalise the Frobenius norm before the polar iterations.
    {
      double frob_sq = 0.0;
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float div = static_cast<float>(std::sqrt(frob_sq)) * 1.01f + 1e-6f;
      for (std::size_t i = 0; i < mat; ++i) x[i] /= div;
    }

    // Polar Express orthogonalisation.
    const bool tall = rows > cols;
    for (int it = 0; it < ns_steps; ++it) {
      const float ca = kPolarCoeffs[it][0];
      const float cb = kPolarCoeffs[it][1];
      const float cc = kPolarCoeffs[it][2];
      if (tall) {
        // A = X^T X  (cols x cols)
        for (int i = 0; i < cols; ++i) {
          for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int r = 0; r < rows; ++r) {
              s += static_cast<double>(
                       x[static_cast<std::size_t>(r) * cols + i]) *
                   x[static_cast<std::size_t>(r) * cols + j];
            }
            a_mat[static_cast<std::size_t>(i) * cols + j] =
                static_cast<float>(s);
          }
        }
        for (int i = 0; i < cols; ++i) {
          for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int l = 0; l < cols; ++l) {
              s += static_cast<double>(
                       a_mat[static_cast<std::size_t>(i) * cols + l]) *
                   a_mat[static_cast<std::size_t>(l) * cols + j];
            }
            a2_mat[static_cast<std::size_t>(i) * cols + j] =
                static_cast<float>(s);
          }
        }
        for (int i = 0; i < cols; ++i) {
          for (int j = 0; j < cols; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * cols + j;
            b_mat[ij] = cb * a_mat[ij] + cc * a2_mat[ij];
          }
        }
        for (int r = 0; r < rows; ++r) {
          for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int i = 0; i < cols; ++i) {
              s += static_cast<double>(
                       x[static_cast<std::size_t>(r) * cols + i]) *
                   b_mat[static_cast<std::size_t>(i) * cols + j];
            }
            prod[static_cast<std::size_t>(r) * cols + j] =
                static_cast<float>(s);
          }
        }
        for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
      } else {
        // A = X X^T  (rows x rows)
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < rows; ++j) {
            double s = 0.0;
            for (int c = 0; c < cols; ++c) {
              s += static_cast<double>(
                       x[static_cast<std::size_t>(i) * cols + c]) *
                   x[static_cast<std::size_t>(j) * cols + c];
            }
            a_mat[static_cast<std::size_t>(i) * rows + j] =
                static_cast<float>(s);
          }
        }
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < rows; ++j) {
            double s = 0.0;
            for (int l = 0; l < rows; ++l) {
              s += static_cast<double>(
                       a_mat[static_cast<std::size_t>(i) * rows + l]) *
                   a_mat[static_cast<std::size_t>(l) * rows + j];
            }
            a2_mat[static_cast<std::size_t>(i) * rows + j] =
                static_cast<float>(s);
          }
        }
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < rows; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * rows + j;
            b_mat[ij] = cb * a_mat[ij] + cc * a2_mat[ij];
          }
        }
        for (int r = 0; r < rows; ++r) {
          for (int c = 0; c < cols; ++c) {
            double s = 0.0;
            for (int i = 0; i < rows; ++i) {
              s += static_cast<double>(
                       b_mat[static_cast<std::size_t>(r) * rows + i]) *
                   x[static_cast<std::size_t>(i) * cols + c];
            }
            prod[static_cast<std::size_t>(r) * cols + c] =
                static_cast<float>(s);
          }
        }
        for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
      }
    }

    // Muon+ renormalisation: snap the Frobenius norm to sqrt(min(rows, cols)).
    {
      double frob_sq = 0.0;
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float target_norm =
          std::sqrt(static_cast<float>(std::min(rows, cols)));
      const float scale =
          target_norm / std::max(static_cast<float>(std::sqrt(frob_sq)), 1e-6f);
      for (std::size_t i = 0; i < mat; ++i) x[i] *= scale;
    }

    // NorMuon variance reduction with a factored second moment.
    if (reduce_cols) {
      const std::size_t red_size = static_cast<std::size_t>(cols);
      std::vector<float> v_mean(rows, 0.0f);
      double sum_vmean = 0.0;
      for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          s += static_cast<double>(xv) * xv;
        }
        v_mean[r] = static_cast<float>(s / red_size);
        sum_vmean += v_mean[r];
      }
      const float v_norm = std::sqrt(static_cast<float>(sum_vmean) *
                                     static_cast<float>(red_size));
      std::vector<float> step_size(rows, 0.0f);
      for (int r = 0; r < rows; ++r) {
        second_buf[r] =
            second_buf[r] + (1.0f - params.beta2) * (v_mean[r] - second_buf[r]);
        step_size[r] = 1.0f / std::sqrt(std::max(second_buf[r], 1e-10f));
      }
      double sum_scaled = 0.0;
      for (int r = 0; r < rows; ++r) {
        sum_scaled +=
            static_cast<double>(v_mean[r] * static_cast<float>(red_size)) *
            step_size[r] * step_size[r];
      }
      const float v_norm_new = std::sqrt(static_cast<float>(sum_scaled));
      const float denom = std::max(v_norm_new, 1e-10f);
      for (int r = 0; r < rows; ++r) {
        const float final_scale = step_size[r] * (v_norm / denom);
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *= final_scale;
        }
      }
    } else {
      const std::size_t red_size = static_cast<std::size_t>(rows);
      std::vector<float> v_mean(cols, 0.0f);
      double sum_vmean = 0.0;
      for (int c = 0; c < cols; ++c) {
        double s = 0.0;
        for (int r = 0; r < rows; ++r) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          s += static_cast<double>(xv) * xv;
        }
        v_mean[c] = static_cast<float>(s / red_size);
        sum_vmean += v_mean[c];
      }
      const float v_norm = std::sqrt(static_cast<float>(sum_vmean) *
                                     static_cast<float>(red_size));
      std::vector<float> step_size(cols, 0.0f);
      for (int c = 0; c < cols; ++c) {
        second_buf[c] =
            second_buf[c] + (1.0f - params.beta2) * (v_mean[c] - second_buf[c]);
        step_size[c] = 1.0f / std::sqrt(std::max(second_buf[c], 1e-10f));
      }
      double sum_scaled = 0.0;
      for (int c = 0; c < cols; ++c) {
        sum_scaled +=
            static_cast<double>(v_mean[c] * static_cast<float>(red_size)) *
            step_size[c] * step_size[c];
      }
      const float v_norm_new = std::sqrt(static_cast<float>(sum_scaled));
      const float denom = std::max(v_norm_new, 1e-10f);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *=
              step_size[c] * (v_norm / denom);
        }
      }
    }

    // Cautious weight decay + parameter update.
    for (std::size_t i = 0; i < mat; ++i) {
      const float pv = AsFloat(param[i]);
      const float gv = x[i];
      const float decay =
          (gv * pv >= 0.0f) ? params.lr * params.weight_decay * pv : 0.0f;
      param[i] = ToCompute(pv - params.lr * gv - decay);
    }
  }
}

// ---------------------------------------------------------------------------
// GlobalNorm (gradient clipping)
// ---------------------------------------------------------------------------

void GlobalNorm(int n, float clip, ComputeType* grads, float* out_norm) {
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = AsFloat(grads[i]);
    sum_sq += v * v;
  }
  const float norm = static_cast<float>(std::sqrt(sum_sq));
  if (out_norm != nullptr) out_norm[0] = norm;
  if (clip > 0.0f && norm > clip) {
    const float scale = clip / norm;
    for (int i = 0; i < n; ++i) {
      grads[i] = ToCompute(AsFloat(grads[i]) * scale);
    }
  }
}

void ScalarDot(const ComputeType* a, const ComputeType* b, int n,
               ComputeType* out, float scale, bool accumulate) {
  if (n <= 0) return;
  double dot = 0.0;
  for (int i = 0; i < n; ++i) {
    dot +=
        static_cast<double>(AsFloat(a[i])) * static_cast<double>(AsFloat(b[i]));
  }
  const float value = scale * static_cast<float>(dot);
  out[0] = ToCompute(accumulate ? AsFloat(out[0]) + value : value);
}

void ValueGateForward(int rows, int hidden, int num_kv_heads, int head_dim,
                      const ComputeType* h, const ComputeType* ve,
                      const ComputeType* gate_w, ComputeType* v,
                      ComputeType* gate_out) {
  constexpr int kChannels = 12;
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int kv_dim = num_kv_heads * head_dim;
  for (int m = 0; m < rows; ++m) {
    const ComputeType* hrow = h + static_cast<std::int64_t>(m) * hidden;
    for (int kh = 0; kh < num_kv_heads; ++kh) {
      const ComputeType* wrow = gate_w + kh * kChannels;
      float pre = 0.0f;
      for (int j = 0; j < kChannels; ++j) {
        pre += AsFloat(hrow[j]) * AsFloat(wrow[j]);
      }
      const float gate = 3.0f / (1.0f + std::exp(-pre));
      gate_out[static_cast<std::int64_t>(m) * num_kv_heads + kh] =
          ToCompute(gate);
      ComputeType* vrow = v + static_cast<std::int64_t>(m) * kv_dim +
                          static_cast<std::int64_t>(kh) * head_dim;
      const ComputeType* verow = ve + static_cast<std::int64_t>(m) * kv_dim +
                                 static_cast<std::int64_t>(kh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        vrow[d] = ToCompute(AsFloat(vrow[d]) + gate * AsFloat(verow[d]));
      }
    }
  }
}

void ValueGateBackward(int rows, int hidden, int num_kv_heads, int head_dim,
                       const ComputeType* h, const ComputeType* ve,
                       const ComputeType* gate_w, const ComputeType* gate,
                       const ComputeType* dv, ComputeType* gate_w_grad,
                       ComputeType* dh, ComputeType* dve) {
  constexpr int kChannels = 12;
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int kv_dim = num_kv_heads * head_dim;
  for (int m = 0; m < rows; ++m) {
    const ComputeType* hrow = h + static_cast<std::int64_t>(m) * hidden;
    ComputeType* dhrow = dh + static_cast<std::int64_t>(m) * hidden;
    for (int kh = 0; kh < num_kv_heads; ++kh) {
      const std::int64_t off = static_cast<std::int64_t>(m) * kv_dim +
                               static_cast<std::int64_t>(kh) * head_dim;
      const float g =
          AsFloat(gate[static_cast<std::int64_t>(m) * num_kv_heads + kh]);
      double dgate = 0.0;
      for (int d = 0; d < head_dim; ++d) {
        const float dvd = AsFloat(dv[off + d]);
        dgate += static_cast<double>(dvd) *
                 static_cast<double>(AsFloat(ve[off + d]));
        dve[off + d] = ToCompute(g * dvd);
      }
      const float dpre = static_cast<float>(dgate) * g * (1.0f - g / 3.0f);
      for (int j = 0; j < kChannels; ++j) {
        gate_w_grad[kh * kChannels + j] = ToCompute(
            AsFloat(gate_w_grad[kh * kChannels + j]) + dpre * AsFloat(hrow[j]));
        dhrow[j] = ToCompute(AsFloat(dhrow[j]) +
                             dpre * AsFloat(gate_w[kh * kChannels + j]));
      }
    }
  }
}

}  // namespace kernels
}  // namespace nanochat
