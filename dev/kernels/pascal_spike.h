#ifndef NANOCHAT_DEV_KERNELS_PASCAL_SPIKE_H_
#define NANOCHAT_DEV_KERNELS_PASCAL_SPIKE_H_

// P0 CUDA/Pascal toolchain spike. This header is vendor-free on purpose: the
// translation unit next to it owns every CUDA call, and host test code stays
// free of CUDA headers. It carries no model logic; the real kernel families
// replace it in P1.

namespace nanochat {
namespace dev {

// out[i] = alpha * a[i] + beta * b[i], computed on the device. Allocates,
// copies, launches, synchronizes, and copies back. Returns 0 on success and a
// CUDA error code otherwise.
int SpikeScaleAddHost(const float* a, const float* b, float alpha, float beta,
                      float* out, int n);

// Writes the compute capability of device 0 into *major and *minor. Returns 0
// on success.
int SpikeComputeCapability(int* major, int* minor);

}  // namespace dev
}  // namespace nanochat

#endif  // NANOCHAT_DEV_KERNELS_PASCAL_SPIKE_H_
