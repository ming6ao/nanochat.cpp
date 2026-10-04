// Reusable graph pieces shared by the training graph and the inference graphs.
// Every function here is a named composition of the frozen kernel seam; the
// topology lives in model.cc and generate.cc. See ops.h and docs/model.md.

#include "src/ops.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace ops {

namespace {

// Host staging helpers for the ops that have no device kernel: a device buffer
// is copied to a host vector, the (unchanged) host math runs, and the result is
// copied back. Correctness first; these are fusion candidates later.
std::vector<ComputeType> StageToHost(const ComputeType* src,
                                     std::int64_t count) {
  std::vector<ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    kernels::Memcpy(host.data(), src,
                    static_cast<std::size_t>(count) * sizeof(ComputeType),
                    CopyDir::kDeviceToHost);
  }
  return host;
}

void StageToDevice(ComputeType* dst, const std::vector<ComputeType>& host) {
  if (host.empty()) return;
  kernels::Memcpy(dst, host.data(), host.size() * sizeof(ComputeType),
                  CopyDir::kHostToDevice);
}

}  // namespace

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

void LinearForward(const ComputeType* x, const ComputeType* w, ComputeType* y,
                   std::int64_t rows, std::int64_t in, std::int64_t out,
                   bool accumulate) {
  if (rows <= 0 || in <= 0 || out <= 0) return;
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

void LinearWgrad(const ComputeType* x, const ComputeType* dy, ComputeType* dw,
                 std::int64_t rows, std::int64_t in, std::int64_t out) {
  if (rows <= 0 || in <= 0 || out <= 0) return;
  // dw[out, in] += dy[rows, out]^T @ x[rows, in].
  GemmParams params;
  params.m = static_cast<int>(out);
  params.n = static_cast<int>(in);
  params.k = static_cast<int>(rows);
  params.alpha = 1.0f;
  params.beta = 1.0f;
  params.transpose_a = true;
  kernels::Gemm(GemmMode::kWgrad, params, dy, x, dw);
}

void LinearDgrad(const ComputeType* dy, const ComputeType* w, ComputeType* dx,
                 std::int64_t rows, std::int64_t in, std::int64_t out,
                 bool accumulate) {
  if (rows <= 0 || in <= 0 || out <= 0) return;
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
// RMSNorm
// ---------------------------------------------------------------------------

void RmsNormForward(std::int64_t rows, std::int64_t dim, float eps,
                    const ComputeType* x, ComputeType* out, float* rstd) {
  if (rows <= 0 || dim <= 0) return;
  RmsNormParams params;
  params.rows = static_cast<int>(rows);
  params.dim = static_cast<int>(dim);
  params.eps = eps;
  kernels::RmsNormForward(params, x, out, rstd);
}

void RmsNormBackward(std::int64_t rows, std::int64_t dim, float eps,
                     const ComputeType* x, const ComputeType* dy,
                     const float* rstd, ComputeType* dx) {
  if (rows <= 0 || dim <= 0) return;
  RmsNormParams params;
  params.rows = static_cast<int>(rows);
  params.dim = static_cast<int>(dim);
  params.eps = eps;
  kernels::RmsNormBackward(params, x, dy, rstd, dx);
}

// ---------------------------------------------------------------------------
// QK-norm + RoPE + scale
// ---------------------------------------------------------------------------

void QkPrepForward(int batch, int seq, int num_heads, int num_kv_heads,
                   int head_dim, float eps, float scale, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  QkPrepParams params;
  params.batch = batch;
  params.seq = seq;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.eps = eps;
  params.scale = scale;
  kernels::QkPrepForward(params, cos, sin, q, k);
}

void QkPrepBackward(int batch, int seq, int num_heads, int num_kv_heads,
                    int head_dim, float eps, float scale, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  QkPrepParams params;
  params.batch = batch;
  params.seq = seq;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.eps = eps;
  params.scale = scale;
  kernels::QkPrepBackward(params, cos, sin, dq, dk, q, k);
}

// ---------------------------------------------------------------------------
// Attention
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
  if (shape.batch <= 0 || shape.seq <= 0 || shape.num_heads <= 0 ||
      shape.head_dim <= 0) {
    return;
  }
  const AttentionParams params = ToParams(shape);
  kernels::AttentionForward(params, q, k, v, out, stats);
}

void AttentionBackward(const AttentionShape& shape, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  if (shape.batch <= 0 || shape.seq <= 0 || shape.num_heads <= 0 ||
      shape.head_dim <= 0) {
    return;
  }
  const AttentionParams params = ToParams(shape);
  kernels::AttentionBackward(params, q, k, v, stats, dout, dq, dk, dv);
}

// ---------------------------------------------------------------------------
// ResFormer value residual
// ---------------------------------------------------------------------------

void ValueResidualForward(std::int64_t rows, int hidden, int num_kv_heads,
                          int head_dim, const ComputeType* h,
                          const ComputeType* ve, const ComputeType* gate_w,
                          ComputeType* v, ComputeType* gate_out) {
  if (rows <= 0 || hidden < kVeGateChannels || num_kv_heads <= 0 ||
      head_dim <= 0) {
    return;
  }
  kernels::ValueGateForward(static_cast<int>(rows), hidden, num_kv_heads,
                            head_dim, h, ve, gate_w, v, gate_out);
}

void ValueResidualBackward(std::int64_t rows, int hidden, int num_kv_heads,
                           int head_dim, const ComputeType* h,
                           const ComputeType* ve, const ComputeType* gate_w,
                           const ComputeType* gate, const ComputeType* dv,
                           ComputeType* gate_w_grad, ComputeType* dh,
                           ComputeType* dve) {
  if (rows <= 0 || hidden < kVeGateChannels || num_kv_heads <= 0 ||
      head_dim <= 0) {
    return;
  }
  kernels::ValueGateBackward(static_cast<int>(rows), hidden, num_kv_heads,
                             head_dim, h, ve, gate_w, gate, dv, gate_w_grad, dh,
                             dve);
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

void MlpForward(const ComputeType* h, const ComputeType* c_fc,
                const ComputeType* c_proj, std::int64_t rows,
                std::int64_t hidden, std::int64_t mlp_dim, ComputeType* pre_act,
                ComputeType* act, ComputeType* scratch,
                const ComputeType* x_mid, ComputeType* out) {
  if (rows <= 0 || hidden <= 0 || mlp_dim <= 0) return;
  LinearForward(h, c_fc, pre_act, rows, hidden, mlp_dim);
  kernels::PointwiseForward(PointwiseOp::kReluSquare,
                            static_cast<int>(rows * mlp_dim), pre_act, nullptr,
                            0.0f, 0.0f, act);
  LinearForward(act, c_proj, scratch, rows, mlp_dim, hidden);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), scratch, x_mid,
                            1.0f, 1.0f, out);
}

void MlpBackward(const ComputeType* h, const ComputeType* c_fc,
                 const ComputeType* c_proj, const ComputeType* pre_act,
                 const ComputeType* act, const ComputeType* dout,
                 std::int64_t rows, std::int64_t hidden, std::int64_t mlp_dim,
                 ComputeType* c_fc_grad, ComputeType* c_proj_grad,
                 ComputeType* dh, ComputeType* dact, ComputeType* dpre) {
  if (rows <= 0 || hidden <= 0 || mlp_dim <= 0) return;
  LinearWgrad(act, dout, c_proj_grad, rows, mlp_dim, hidden);
  LinearDgrad(dout, c_proj, dact, rows, mlp_dim, hidden, false);
  kernels::PointwiseBackward(PointwiseOp::kReluSquare,
                             static_cast<int>(rows * mlp_dim), pre_act, nullptr,
                             dact, 0.0f, 0.0f, dpre, nullptr);
  LinearWgrad(h, dpre, c_fc_grad, rows, hidden, mlp_dim);
  LinearDgrad(dpre, c_fc, dh, rows, hidden, mlp_dim, false);
}

// ---------------------------------------------------------------------------
// Smear
// ---------------------------------------------------------------------------

void SmearForward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                  const ComputeType* emb_norm, const ComputeType* gate_w,
                  float lambda, ComputeType* x0, float* sig) {
  if (batch <= 0 || seq <= 0 || hidden <= 0) return;
  const std::int64_t rows = batch * seq;
  const std::int64_t count = rows * hidden;
  std::vector<ComputeType> ee = StageToHost(emb_norm, count);
  std::vector<ComputeType> wh = StageToHost(gate_w, kSmearChannels);
  std::vector<ComputeType> xh(static_cast<std::size_t>(count));
  std::vector<float> sh(static_cast<std::size_t>(rows), 0.0f);
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t t = 0; t < seq; ++t) {
      const std::int64_t dst = (b * seq + t) * hidden;
      if (t == 0) {
        std::memcpy(xh.data() + dst, ee.data() + dst,
                    static_cast<std::size_t>(hidden) * sizeof(ComputeType));
        sh[static_cast<std::size_t>(b * seq + t)] = 0.0f;
        continue;
      }
      const std::int64_t src = (b * seq + t - 1) * hidden;
      float pre = 0.0f;
      for (int j = 0; j < kSmearChannels; ++j) {
        pre += AsF(ee[static_cast<std::size_t>(dst + j)]) *
               AsF(wh[static_cast<std::size_t>(j)]);
      }
      const float s = Sigmoid(pre);
      sh[static_cast<std::size_t>(b * seq + t)] = s;
      const float gate = lambda * s;
      for (std::int64_t j = 0; j < hidden; ++j) {
        xh[static_cast<std::size_t>(dst + j)] =
            ToC(AsF(ee[static_cast<std::size_t>(dst + j)]) +
                gate * AsF(ee[static_cast<std::size_t>(src + j)]));
      }
    }
  }
  StageToDevice(x0, xh);
  kernels::Memcpy(sig, sh.data(),
                  static_cast<std::size_t>(rows) * sizeof(float),
                  CopyDir::kHostToDevice);
}

void SmearBackward(std::int64_t batch, std::int64_t seq, std::int64_t hidden,
                   const ComputeType* emb_norm, const ComputeType* gate_w,
                   const float* sig, float lambda, const ComputeType* d_x0,
                   ComputeType* gate_w_grad, ComputeType* lambda_grad,
                   ComputeType* de) {
  if (batch <= 0 || seq <= 0 || hidden <= 0) return;
  const std::int64_t rows = batch * seq;
  const std::int64_t count = rows * hidden;
  std::vector<ComputeType> ee = StageToHost(emb_norm, count);
  std::vector<ComputeType> wh = StageToHost(gate_w, kSmearChannels);
  std::vector<float> sh(static_cast<std::size_t>(rows));
  kernels::Memcpy(sh.data(), sig,
                  static_cast<std::size_t>(rows) * sizeof(float),
                  CopyDir::kDeviceToHost);
  std::vector<ComputeType> dxh = StageToHost(d_x0, count);
  // Identity path: x0[t] contains emb_norm[t] directly.
  std::vector<ComputeType> deh = dxh;
  std::vector<ComputeType> wgh = StageToHost(gate_w_grad, kSmearChannels);
  std::vector<ComputeType> lgh = StageToHost(lambda_grad, 1);
  double dlambda = 0.0;
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t t = 1; t < seq; ++t) {
      const std::int64_t m = (b * seq + t) * hidden;
      const std::int64_t prev = (b * seq + t - 1) * hidden;
      const float s = sh[static_cast<std::size_t>(b * seq + t)];
      const float gate = lambda * s;

      double dgate = 0.0;
      for (std::int64_t j = 0; j < hidden; ++j) {
        dgate +=
            static_cast<double>(AsF(dxh[static_cast<std::size_t>(m + j)])) *
            static_cast<double>(AsF(ee[static_cast<std::size_t>(prev + j)]));
      }
      const float dgate_f = static_cast<float>(dgate);
      // x0[t] = e[t] + gate[t] * e[t-1], so the gradient w.r.t. the previous
      // position is elementwise `dx0[t, j] * gate[t]`, not the scalar
      // `dgate * gate` (which is the gate's own gradient, used below for
      // lambda and the gate weights).
      for (std::int64_t j = 0; j < hidden; ++j) {
        deh[static_cast<std::size_t>(prev + j)] =
            ToC(AsF(deh[static_cast<std::size_t>(prev + j)]) +
                AsF(dxh[static_cast<std::size_t>(m + j)]) * gate);
      }
      dlambda += static_cast<double>(dgate_f) * s;
      const float dpre = dgate_f * lambda * s * (1.0f - s);
      for (int j = 0; j < kSmearChannels; ++j) {
        wgh[static_cast<std::size_t>(j)] =
            ToC(AsF(wgh[static_cast<std::size_t>(j)]) +
                dpre * AsF(ee[static_cast<std::size_t>(m + j)]));
        deh[static_cast<std::size_t>(m + j)] =
            ToC(AsF(deh[static_cast<std::size_t>(m + j)]) +
                dpre * AsF(wh[static_cast<std::size_t>(j)]));
      }
    }
  }
  lgh[0] = ToC(AsF(lgh[0]) + static_cast<float>(dlambda));
  StageToDevice(de, deh);
  StageToDevice(gate_w_grad, wgh);
  StageToDevice(lambda_grad, lgh);
}

// ---------------------------------------------------------------------------
// Backout
// ---------------------------------------------------------------------------

void BackoutForward(const ComputeType* x, const ComputeType* x_backout,
                    float lambda, ComputeType* out, std::int64_t n) {
  if (n <= 0) return;
  kernels::PointwiseForward(PointwiseOp::kScaleAdd, static_cast<int>(n), x,
                            x_backout, 1.0f, -lambda, out);
}

void BackoutBackward(const ComputeType* d_pre, const ComputeType* x_backout,
                     float lambda, ComputeType* dbackout,
                     ComputeType* lambda_grad, std::int64_t n) {
  if (n <= 0) return;
  // The elementwise scale and the scalar dot product both run on the device;
  // no host round trip.
  kernels::PointwiseForward(PointwiseOp::kScale, static_cast<int>(n), d_pre,
                            nullptr, -lambda, 0.0f, dbackout);
  kernels::ScalarDot(d_pre, x_backout, static_cast<int>(n), lambda_grad, -1.0f,
                     true);
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

void BlockForward(const BlockShape& shape, const BlockWeights& weights,
                  const BlockActivations& acts, const int* tokens,
                  const float* cos, const float* sin, const ComputeType* x,
                  const ComputeType* x0, float resid, float x0_lambda) {
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
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), x, x0, resid,
                            x0_lambda, acts.xr);
  RmsNormForward(rows, hidden, shape.rms_eps, acts.xr, acts.h, acts.rstd1);

  LinearForward(acts.h, weights.c_q, acts.q_final, rows, hidden, query_dim);
  if (acts.q_pre != nullptr && acts.q_pre != acts.q_final) {
    kernels::Memcpy(
        acts.q_pre, acts.q_final,
        static_cast<std::size_t>(rows * query_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }
  LinearForward(acts.h, weights.c_k, acts.k_final, rows, hidden, kv_dim);
  if (acts.k_pre != nullptr && acts.k_pre != acts.k_final) {
    kernels::Memcpy(
        acts.k_pre, acts.k_final,
        static_cast<std::size_t>(rows * kv_dim) * sizeof(ComputeType),
        CopyDir::kDeviceToDevice);
  }

  LinearForward(acts.h, weights.c_v, acts.v_final, rows, hidden, kv_dim);
  if (weights.has_ve) {
    kernels::EmbeddingForward(static_cast<int>(rows), kv_dim, tokens,
                              weights.value_embeds, acts.ve_values);
    ValueResidualForward(rows, hidden, kv_heads, head_dim, acts.h,
                         acts.ve_values, weights.ve_gate, acts.v_final,
                         acts.ve_gate);
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
                hidden, hidden);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), acts.scratch_a,
                            acts.xr, 1.0f, 1.0f, acts.x_mid);

  RmsNormForward(rows, hidden, shape.rms_eps, acts.x_mid, acts.h2, acts.rstd2);
  MlpForward(acts.h2, weights.c_fc, weights.c_proj_mlp, rows, hidden, mlp_dim,
             acts.pre_act, acts.act, acts.scratch_b, acts.x_mid, acts.x_out);
}

void BlockBackward(const BlockShape& shape, const BlockWeights& weights,
                   const BlockActivations& acts, const BlockGrads& grads,
                   const BlockScratch& scratch, const int* tokens,
                   const float* cos, const float* sin, float resid,
                   float x0_lambda, const ComputeType* x_in,
                   const ComputeType* x0, const ComputeType* dout,
                   ComputeType* dx_in, ComputeType* x0_acc,
                   ComputeType* resid_grad, ComputeType* x0_lambda_grad) {
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
              grads.c_proj_mlp_grad, scratch.dh2, scratch.dact, scratch.dpre);
  RmsNormBackward(rows, hidden, shape.rms_eps, acts.x_mid, scratch.dh2,
                  acts.rstd2, scratch.dx_mid);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), dout,
                            scratch.dx_mid, 1.0f, 1.0f, scratch.dx_mid);

  // --- Attention: x_mid = xr + c_proj(attention(q, k, v)). ---
  LinearWgrad(acts.attn_out, scratch.dx_mid, grads.c_proj_attn_grad, rows,
              hidden, hidden);
  LinearDgrad(scratch.dx_mid, weights.c_proj_attn, scratch.dy_attn, rows,
              hidden, hidden, false);

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
  LinearWgrad(acts.h, scratch.dv, grads.c_v_grad, rows, hidden, kv_dim);
  LinearDgrad(scratch.dv, weights.c_v, scratch.dh, rows, hidden, kv_dim, false);
  if (weights.has_ve) {
    ValueResidualBackward(rows, hidden, kv_heads, head_dim, acts.h,
                          acts.ve_values, weights.ve_gate, acts.ve_gate,
                          scratch.dv, grads.ve_gate_grad, scratch.dh,
                          scratch.dve);
    kernels::EmbeddingBackward(static_cast<int>(rows), kv_dim, tokens,
                               scratch.dve, grads.value_embeds_grad);
  }

  // QK-norm + RoPE + scale, then the query/key projections.
  QkPrepBackward(shape.batch, shape.seq, heads, kv_heads, head_dim,
                 shape.rms_eps, shape.qk_scale, cos, sin, scratch.dq,
                 scratch.dk, acts.q_pre, acts.k_pre);
  LinearWgrad(acts.h, acts.q_pre, grads.c_q_grad, rows, hidden, query_dim);
  LinearDgrad(acts.q_pre, weights.c_q, scratch.dh, rows, hidden, query_dim,
              true);
  LinearWgrad(acts.h, acts.k_pre, grads.c_k_grad, rows, hidden, kv_dim);
  LinearDgrad(acts.k_pre, weights.c_k, scratch.dh, rows, hidden, kv_dim, true);

  // Pre-attention norm, then the residual blend xr = resid*x + x0_lambda*x0.
  RmsNormBackward(rows, hidden, shape.rms_eps, acts.xr, scratch.dh, acts.rstd1,
                  scratch.dxr);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), scratch.dx_mid,
                            scratch.dxr, 1.0f, 1.0f, scratch.dxr);

  // dx_in = resid * dxr; x0_acc += x0_lambda * dxr. Both are elementwise on
  // the device. The two scalar gradients are device reductions, so the block
  // backward stays entirely on the GPU.
  kernels::PointwiseForward(PointwiseOp::kScale,
                            static_cast<int>(rows * hidden), scratch.dxr,
                            nullptr, resid, 0.0f, dx_in);
  kernels::PointwiseForward(PointwiseOp::kScaleAdd,
                            static_cast<int>(rows * hidden), x0_acc,
                            scratch.dxr, 1.0f, x0_lambda, x0_acc);
  kernels::ScalarDot(scratch.dxr, x_in, static_cast<int>(rows * hidden),
                     resid_grad, 1.0f, true);
  kernels::ScalarDot(scratch.dxr, x0, static_cast<int>(rows * hidden),
                     x0_lambda_grad, 1.0f, true);
}

}  // namespace ops
}  // namespace nanochat
