#ifndef NANOCHAT_DEV_KERNELS_OPTIM_REF_H_
#define NANOCHAT_DEV_KERNELS_OPTIM_REF_H_

// Host reference for the optimizer kernel families, ported line-for-line from
// backends/cpu/kernels.cc. AdamW and Muon are deterministic update rules with
// no backward pass, so the GPU tests compare the device result against these
// functions rather than differentiating a forward. Header-only so every
// optimizer test shares one copy.
//
// The Muon reference is the authoritative description of the five stages:
// Nesterov momentum, MuonEq row equilibration, Polar Express
// orthogonalisation, Muon+ renormalisation, and NorMuon variance reduction,
// followed by the cautious decoupled weight decay.
//
// See docs/testing.md and docs/model.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace dev {
namespace optimref {

// Polar Express coefficients (num_iters=5), matching backends/cpu/kernels.cc.
constexpr float kPolarCoeffs[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

// Applies one AdamW step in place. `p`, `m`, and `v` are updated; `g` is
// read-only.
inline void AdamWUpdate(int n, const AdamWParams& params, std::vector<float>* p,
                        const std::vector<float>& g, std::vector<float>* m,
                        std::vector<float>* v) {
  if (n <= 0) return;
  const float bias1 =
      1.0f - std::pow(params.beta1, static_cast<float>(params.step));
  const float bias2 =
      1.0f - std::pow(params.beta2, static_cast<float>(params.step));
  const float step_size = params.lr / bias1;
  for (int i = 0; i < n; ++i) {
    const float grad = g[i];
    float pi = (*p)[i] * (1.0f - params.lr * params.weight_decay);
    const float mi = (*m)[i] + (1.0f - params.beta1) * (grad - (*m)[i]);
    const float vi = (*v)[i] + (1.0f - params.beta2) * (grad * grad - (*v)[i]);
    (*m)[i] = mi;
    (*v)[i] = vi;
    const float denom = std::sqrt(vi / bias2) + params.eps;
    pi -= step_size * (mi / denom);
    (*p)[i] = pi;
  }
}

// Applies one Muon step in place over `num_params` matrices stacked along axis
// 0. `stacked_params`, `buf1`, and `buf2` are updated; `stacked_grads` is
// read-only.
inline void MuonUpdate(const MuonParams& params,
                       const std::vector<float>& stacked_grads,
                       std::vector<float>* stacked_params,
                       std::vector<float>* buf1, std::vector<float>* buf2) {
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
    const float* grad = stacked_grads.data() + off;
    float* param = stacked_params->data() + off;
    float* momentum_buf = buf1->data() + off;
    float* second_buf = buf2->data() + static_cast<std::size_t>(m) *
                                           (reduce_cols ? rows : cols);

    // Nesterov momentum: update the first moment, then the accelerated
    // gradient.
    for (std::size_t i = 0; i < mat; ++i) {
      const float gr = grad[i];
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
      const float pv = param[i];
      const float gv = x[i];
      const float decay =
          (gv * pv >= 0.0f) ? params.lr * params.weight_decay * pv : 0.0f;
      param[i] = pv - params.lr * gv - decay;
    }
  }
}

// ANVIL's six quintic spectral maps, matching backends/cpu/kernels.cc and
// backends/cuda/kernels/anvil.cu.
constexpr float kAnvilMaps[6][3] = {
    {3.923798038567f, -6.095026865488f, 3.905234618423f},
    {3.278126713798f, -3.328923386476f, 0.989127286973f},
    {3.505298394150f, -5.137358782410f, 1.968325560615f},
    {2.815058591845f, -3.685181239622f, 1.417196497642f},
    {2.245503932403f, -2.443826979899f, 0.963091710461f},
    {2.256537145403f, -2.166840097229f, 0.929501253245f},
};

// Applies one ANVIL step in place over `num_params` matrices stacked along axis
// 0. `stacked_params`, `velocity`, and `lane_energy` are updated;
// `stacked_grads` is read-only. `velocity` holds `2 * num_params * rows * cols`
// floats (fast rail then slow rail); `lane_energy` holds
// `num_params * (red_dim == -1 ? rows : cols)` floats. This is the
// authoritative description of the four stages: twin-rail Nesterov momentum,
// the Frobenius-normalized whitening cascade, the per-lane energy equalizer,
// and the sign-aligned update.
inline void AnvilUpdate(const AnvilParams& params,
                        const std::vector<float>& stacked_grads,
                        std::vector<float>* stacked_params,
                        std::vector<float>* velocity,
                        std::vector<float>* lane_energy) {
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;
  const std::size_t mat = static_cast<std::size_t>(rows) * cols;
  const std::size_t rail_stride = static_cast<std::size_t>(num_params) * mat;
  const int num_maps = std::min(std::max(params.num_maps, 0), 6);
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);
  const bool tall = rows > cols;
  const int min_dim = std::min(rows, cols);
  const std::size_t min_sq = static_cast<std::size_t>(min_dim) * min_dim;
  const int lane_count = reduce_cols ? rows : cols;
  const int lane_len = reduce_cols ? cols : rows;
  const float inv_lane_len = 1.0f / static_cast<float>(lane_len);

  std::vector<float> x(mat, 0.0f);
  std::vector<float> a_mat(min_sq, 0.0f);
  std::vector<float> a2_mat(min_sq, 0.0f);
  std::vector<float> b_mat(min_sq, 0.0f);
  std::vector<float> prod(mat, 0.0f);
  std::vector<float> lane_power(static_cast<std::size_t>(lane_count), 0.0f);
  std::vector<float> lane_scale(static_cast<std::size_t>(lane_count), 0.0f);

  for (int m = 0; m < num_params; ++m) {
    const std::size_t off = static_cast<std::size_t>(m) * mat;
    const float* grad = stacked_grads.data() + off;
    float* param = stacked_params->data() + off;
    float* fast = velocity->data() + off;
    float* slow = velocity->data() + rail_stride + off;
    float* lane =
        lane_energy->data() + static_cast<std::size_t>(m) * lane_count;

    // Twin-rail momentum plus the Nesterov lookahead.
    for (std::size_t i = 0; i < mat; ++i) {
      const float gr = grad[i];
      const float f = fast[i] + (1.0f - params.fast_beta) * (gr - fast[i]);
      const float s = slow[i] + (1.0f - params.slow_beta) * (gr - slow[i]);
      fast[i] = f;
      slow[i] = s;
      const float blend =
          params.fast_weight * f + (1.0f - params.fast_weight) * s;
      x[i] = params.nesterov
                 ? (1.0f - params.momentum) * gr + params.momentum * blend
                 : blend;
    }

    // First Gram, then the Frobenius normalization.
    const auto gram = [&]() {
      if (tall) {
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
      } else {
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
      }
    };
    gram();
    double trace = 0.0;
    for (int i = 0; i < min_dim; ++i) {
      trace +=
          static_cast<double>(a_mat[static_cast<std::size_t>(i) * min_dim + i]);
    }
    const float d = static_cast<float>(std::sqrt(trace)) * 1.05f + 1e-6f;
    const float inv_d = 1.0f / d;
    const float inv_d2 = inv_d * inv_d;
    for (std::size_t i = 0; i < mat; ++i) x[i] *= inv_d;
    for (std::size_t i = 0; i < min_sq; ++i) a_mat[i] *= inv_d2;

    // The whitening cascade.
    for (int k = 0; k < num_maps; ++k) {
      if (k > 0) gram();
      for (int i = 0; i < min_dim; ++i) {
        for (int j = 0; j < min_dim; ++j) {
          double s = 0.0;
          for (int l = 0; l < min_dim; ++l) {
            s += static_cast<double>(
                     a_mat[static_cast<std::size_t>(i) * min_dim + l]) *
                 a_mat[static_cast<std::size_t>(l) * min_dim + j];
          }
          a2_mat[static_cast<std::size_t>(i) * min_dim + j] =
              static_cast<float>(s);
        }
      }
      const float ca = kAnvilMaps[k][0];
      const float cb = kAnvilMaps[k][1];
      const float cc = kAnvilMaps[k][2];
      for (std::size_t i = 0; i < min_sq; ++i) {
        b_mat[i] = cb * a_mat[i] + cc * a2_mat[i];
      }
      if (tall) {
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
      } else {
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
      }
      for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
    }

    // Per-lane energy equalizer.
    for (int l = 0; l < lane_count; ++l) {
      double s = 0.0;
      if (reduce_cols) {
        for (int c = 0; c < cols; ++c) {
          const float v = x[static_cast<std::size_t>(l) * cols + c];
          s += static_cast<double>(v) * v;
        }
      } else {
        for (int r = 0; r < rows; ++r) {
          const float v = x[static_cast<std::size_t>(r) * cols + l];
          s += static_cast<double>(v) * v;
        }
      }
      lane_power[static_cast<std::size_t>(l)] =
          static_cast<float>(s) * inv_lane_len;
    }
    double sum_power = 0.0;
    for (int l = 0; l < lane_count; ++l) {
      sum_power += lane_power[static_cast<std::size_t>(l)];
    }
    const float pre_norm = static_cast<float>(
        std::sqrt(sum_power * static_cast<double>(lane_len)));
    double sum_post = 0.0;
    for (int l = 0; l < lane_count; ++l) {
      const std::size_t idx = static_cast<std::size_t>(l);
      const float power = lane_power[idx];
      const float energy =
          lane[idx] + (1.0f - params.beta2) * (power - lane[idx]);
      lane[idx] = energy;
      const float gain = 1.0f / std::sqrt(std::max(energy, 1e-10f));
      lane_scale[idx] = gain;
      sum_post += static_cast<double>(power * static_cast<float>(lane_len)) *
                  gain * gain;
    }
    const float inv_post =
        pre_norm / std::max(static_cast<float>(std::sqrt(sum_post)), 1e-10f);
    for (int l = 0; l < lane_count; ++l) {
      lane_scale[static_cast<std::size_t>(l)] *= inv_post;
    }
    if (reduce_cols) {
      for (int l = 0; l < lane_count; ++l) {
        const float s = lane_scale[static_cast<std::size_t>(l)];
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(l) * cols + c] *= s;
        }
      }
    } else {
      for (int l = 0; l < lane_count; ++l) {
        const float s = lane_scale[static_cast<std::size_t>(l)];
        for (int r = 0; r < rows; ++r) {
          x[static_cast<std::size_t>(r) * cols + l] *= s;
        }
      }
    }

    // Sign-aligned weight decay gated on the slow rail, then the update.
    for (std::size_t i = 0; i < mat; ++i) {
      const float pv = param[i];
      const float decay =
          (slow[i] * pv >= 0.0f) ? params.lr * params.weight_decay * pv : 0.0f;
      param[i] = pv - decay - params.lr * x[i];
    }
  }
}

}  // namespace optimref
}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_OPTIM_REF_H_
