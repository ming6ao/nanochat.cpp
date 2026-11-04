#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_TESTING_SMEAR_REF_H_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_TESTING_SMEAR_REF_H_

// The host reference of the smear family: the definition the CUDA kernel is
// compared against. It walks rows in order and keeps every intermediate in the
// type the model's arithmetic names, so a finite difference of it is the
// gradient the device must reproduce.
//
// A row is `[channels, hidden)` of a `[rows, hidden]` activation, and a
// sequence is `seq` consecutive rows:
//
//   gate[row] = lambda * sigmoid(dot(emb_norm[row, :channels], gate_w))
//   x0[row]   = emb_norm[row] + gate[row] * emb_norm[row - 1]      (t > 0)
//   x0[row]   = emb_norm[row]                                      (t == 0)
//
// This header is device-independent on purpose: the test compiles it for the
// host, so the reference is the specification and not a second kernel.

#include <cmath>
#include <cstdint>
#include <vector>

#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {
namespace testing {

// Host floats, one per element. The test converts to and from ComputeType at
// the seam, so the reference arithmetic is exact.
inline void SmearForwardRef(const SmearParams& p,
                            const std::vector<float>& emb_norm,
                            const std::vector<float>& gate_w, float* x0,
                            float* sig) {
  const std::int64_t hidden = p.hidden;
  const std::int64_t rows = p.rows();
  const float lambda = p.lambda.host();
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t base = row * hidden;
    const bool first = row % p.seq == 0;
    float pre = 0.0f;
    if (!first) {
      for (int j = 0; j < p.channels; ++j) {
        pre += emb_norm[static_cast<std::size_t>(base + j)] *
               gate_w[static_cast<std::size_t>(j)];
      }
    }
    const float s = first ? 0.0f : 1.0f / (1.0f + std::exp(-pre));
    if (sig != nullptr) sig[row] = s;
    const float gate = lambda * s;
    for (std::int64_t j = 0; j < hidden; ++j) {
      const float previous =
          first ? 0.0f : emb_norm[static_cast<std::size_t>(base - hidden + j)];
      x0[base + j] =
          emb_norm[static_cast<std::size_t>(base + j)] + gate * previous;
    }
  }
}

inline void SmearBackwardRef(const SmearParams& p,
                             const std::vector<float>& emb_norm,
                             const std::vector<float>& gate_w, const float* sig,
                             const std::vector<float>& d_x0, float* gate_w_grad,
                             float* lambda_grad, float* de) {
  const std::int64_t hidden = p.hidden;
  const std::int64_t rows = p.rows();
  const float lambda = p.lambda.host();
  double dlambda = 0.0;
  for (std::int64_t row = 0; row < rows; ++row) {
    const std::int64_t base = row * hidden;
    const bool first = row % p.seq == 0;
    const bool tail = row % p.seq == p.seq - 1;
    for (std::int64_t j = 0; j < hidden; ++j) {
      de[base + j] = d_x0[static_cast<std::size_t>(base + j)];
    }
    const float s = first ? 0.0f : sig[row];
    if (!first) {
      double dgate = 0.0;
      for (std::int64_t j = 0; j < hidden; ++j) {
        dgate += static_cast<double>(d_x0[static_cast<std::size_t>(base + j)]) *
                 static_cast<double>(
                     emb_norm[static_cast<std::size_t>(base - hidden + j)]);
      }
      const float dgate_f = static_cast<float>(dgate);
      dlambda += static_cast<double>(dgate_f) * s;
      const float dpre = dgate_f * lambda * s * (1.0f - s);
      for (int j = 0; j < p.channels; ++j) {
        gate_w_grad[j] += dpre * emb_norm[static_cast<std::size_t>(base + j)];
        de[base + j] += dpre * gate_w[static_cast<std::size_t>(j)];
      }
    }
    if (!tail) {
      const float gate_next = lambda * sig[row + 1];
      for (std::int64_t j = 0; j < hidden; ++j) {
        de[base + j] +=
            gate_next * d_x0[static_cast<std::size_t>(base + hidden + j)];
      }
    }
  }
  *lambda_grad += static_cast<float>(dlambda);
}

}  // namespace testing
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_TESTING_SMEAR_REF_H_
