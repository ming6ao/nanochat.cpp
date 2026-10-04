#ifndef NANOCHAT_DEV_KERNELS_ROW_REF_H_
#define NANOCHAT_DEV_KERNELS_ROW_REF_H_

// Host reference for the row/elementwise kernel families, ported line-for-line
// from backends/cpu/kernels.cc. The GPU tests compare device results against
// these functions, and the finite-difference checks differentiate the
// reference forward. Header-only so every kernel test shares one copy.
//
// See docs/testing.md.

#include <cmath>
#include <cstddef>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {
namespace rowref {

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// --- RmsNorm --------------------------------------------------------------

inline std::vector<float> RmsNormForward(const RmsNormParams& p,
                                         const std::vector<float>& x,
                                         std::vector<float>* rstd) {
  const int rows = p.rows;
  const int dim = p.dim;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  std::vector<float> out(static_cast<std::size_t>(rows) * dim, 0.0f);
  rstd->assign(rows, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * dim;
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = x[base + d];
      sum_sq += v * v;
    }
    const float rr =
        1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + p.eps);
    for (int d = 0; d < dim; ++d) out[base + d] = x[base + d] * rr;
    (*rstd)[r] = rr;
  }
  return out;
}

inline std::vector<float> RmsNormBackward(const RmsNormParams& p,
                                          const std::vector<float>& x,
                                          const std::vector<float>& dy,
                                          const std::vector<float>& rstd) {
  const int rows = p.rows;
  const int dim = p.dim;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  std::vector<float> dx(static_cast<std::size_t>(rows) * dim, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const std::size_t base = static_cast<std::size_t>(r) * dim;
    const float rr = rstd[r];
    double dot = 0.0;
    for (int d = 0; d < dim; ++d) {
      dot += static_cast<double>(x[base + d]) * dy[base + d];
    }
    const float coeff = rr * rr * rr * static_cast<float>(dot) * inv_dim;
    for (int d = 0; d < dim; ++d) {
      dx[base + d] = rr * dy[base + d] - coeff * x[base + d];
    }
  }
  return dx;
}

// --- Fused residual -------------------------------------------------------

// residual_out = x + residual; out = rmsnorm(residual_out).
inline void FusedResidualForward(const RmsNormParams& p,
                                 const std::vector<float>& x,
                                 std::vector<float>* residual,
                                 std::vector<float>* out,
                                 std::vector<float>* rstd) {
  for (std::size_t i = 0; i < x.size(); ++i) (*residual)[i] += x[i];
  std::vector<float> r;
  *out = RmsNormForward(p, *residual, &r);
  *rstd = r;
}

// The gradient w.r.t. x and the incoming residual are identical.
inline std::vector<float> FusedResidualBackward(
    const RmsNormParams& p, const std::vector<float>& residual,
    const std::vector<float>& rstd, const std::vector<float>& dy) {
  return RmsNormBackward(p, residual, dy, rstd);
}

// --- QkPrep ---------------------------------------------------------------

// Applies RMSNorm, RoPE, then scale to one [batch, seq, heads, head_dim]
// tensor, in place. `heads` selects q (num_heads) or k (num_kv_heads).
inline void QkPrepApply(const QkPrepParams& p, const std::vector<float>& cos,
                        const std::vector<float>& sin, int heads,
                        std::vector<float>* tensor) {
  const int batch = p.batch;
  const int seq = p.seq;
  const int dim = p.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::size_t base =
            ((static_cast<std::size_t>(b) * seq + t) * heads + h) * dim;
        double sum_sq = 0.0;
        for (int d = 0; d < dim; ++d) {
          const double v = (*tensor)[base + d];
          sum_sq += v * v;
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + p.eps);
        for (int d = 0; d < dim; ++d) (*tensor)[base + d] *= r;
        const std::size_t row_cos = static_cast<std::size_t>(t) * half;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = (*tensor)[base + d];
          const float x2 = (*tensor)[base + half + d];
          (*tensor)[base + d] = (x1 * c + x2 * s) * p.scale;
          (*tensor)[base + half + d] = (-x1 * s + x2 * c) * p.scale;
        }
      }
    }
  }
}

inline void QkPrepForward(const QkPrepParams& p, const std::vector<float>& cos,
                          const std::vector<float>& sin, std::vector<float>* q,
                          std::vector<float>* k) {
  QkPrepApply(p, cos, sin, p.num_heads, q);
  QkPrepApply(p, cos, sin, p.num_kv_heads, k);
}

// `grad` is the upstream gradient w.r.t. the final activation; `tensor` holds
// the saved pre-norm rows and is overwritten with the input gradient.
inline void QkPrepApplyBackward(const QkPrepParams& p,
                                const std::vector<float>& cos,
                                const std::vector<float>& sin,
                                const std::vector<float>& grad, int heads,
                                std::vector<float>* tensor) {
  const int batch = p.batch;
  const int seq = p.seq;
  const int dim = p.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  std::vector<float> g(dim, 0.0f);
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::size_t base =
            ((static_cast<std::size_t>(b) * seq + t) * heads + h) * dim;
        const std::size_t row_cos = static_cast<std::size_t>(t) * half;
        double sum_sq = 0.0;
        double dot = 0.0;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = (*tensor)[base + d];
          const float x2 = (*tensor)[base + half + d];
          const float g1 = grad[base + d];
          const float g2 = grad[base + half + d];
          g[d] = p.scale * (g1 * c - g2 * s);
          g[half + d] = p.scale * (g1 * s + g2 * c);
          sum_sq += static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
          dot += static_cast<double>(x1) * g[d] +
                 static_cast<double>(x2) * g[half + d];
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + p.eps);
        const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;
        for (int d = 0; d < dim; ++d) {
          (*tensor)[base + d] = r * g[d] - coeff * (*tensor)[base + d];
        }
      }
    }
  }
}

inline void QkPrepBackward(const QkPrepParams& p, const std::vector<float>& cos,
                           const std::vector<float>& sin,
                           const std::vector<float>& dq,
                           const std::vector<float>& dk, std::vector<float>* q,
                           std::vector<float>* k) {
  QkPrepApplyBackward(p, cos, sin, dq, p.num_heads, q);
  QkPrepApplyBackward(p, cos, sin, dk, p.num_kv_heads, k);
}

// --- Pointwise ------------------------------------------------------------

inline std::vector<float> PointwiseForward(PointwiseOp op,
                                           const std::vector<float>& a,
                                           const std::vector<float>& b,
                                           float alpha, float beta) {
  std::vector<float> out(a.size(), 0.0f);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float av = a[i];
    float result = 0.0f;
    switch (op) {
      case PointwiseOp::kScale:
        result = alpha * av;
        break;
      case PointwiseOp::kScaleAdd:
        result = alpha * av + beta * b[i];
        break;
      case PointwiseOp::kGateMul:
        result = Sigmoid(av) * b[i];
        break;
      case PointwiseOp::kReluSquare: {
        const float relu = av > 0.0f ? av : 0.0f;
        result = relu * relu;
        break;
      }
      case PointwiseOp::kSoftcap:
        result = alpha * std::tanh(av / alpha);
        break;
    }
    out[i] = result;
  }
  return out;
}

inline void PointwiseBackward(PointwiseOp op, const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<float>& dy, float alpha,
                              float beta, std::vector<float>* da,
                              std::vector<float>* db) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float av = a[i];
    const float dyv = dy[i];
    switch (op) {
      case PointwiseOp::kScale:
        if (da != nullptr) (*da)[i] = alpha * dyv;
        break;
      case PointwiseOp::kScaleAdd:
        if (da != nullptr) (*da)[i] = alpha * dyv;
        if (db != nullptr) (*db)[i] = beta * dyv;
        break;
      case PointwiseOp::kGateMul: {
        const float sig = Sigmoid(av);
        if (da != nullptr) (*da)[i] = dyv * b[i] * sig * (1.0f - sig);
        if (db != nullptr) (*db)[i] = dyv * sig;
        break;
      }
      case PointwiseOp::kReluSquare:
        if (da != nullptr) {
          (*da)[i] = av > 0.0f ? dyv * 2.0f * av : 0.0f;
        }
        break;
      case PointwiseOp::kSoftcap: {
        if (da != nullptr) {
          const float t = std::tanh(av / alpha);
          (*da)[i] = dyv * (1.0f - t * t);
        }
        break;
      }
    }
  }
}

// --- GlobalNorm -----------------------------------------------------------

inline float GlobalNorm(int n, float clip, std::vector<float>* grads) {
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += static_cast<double>((*grads)[i]) * (*grads)[i];
  }
  const float norm = static_cast<float>(std::sqrt(sum_sq));
  if (clip > 0.0f && norm > clip) {
    const float s = clip / norm;
    for (int i = 0; i < n; ++i) (*grads)[i] *= s;
  }
  return norm;
}

}  // namespace rowref
}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_ROW_REF_H_
