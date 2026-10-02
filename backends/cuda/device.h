#ifndef NANOCHAT_BACKENDS_CUDA_DEVICE_H_
#define NANOCHAT_BACKENDS_CUDA_DEVICE_H_

// Backend-internal host helpers shared by every CUDA translation unit. This
// header is deliberately *not* part of the frozen seam: it may include CUDA
// headers, because only code under backends/cuda includes it. Code above the
// seam (src/**, kernels.h) never sees it. See DESIGN.md section 2.1.

#include <cuda_runtime.h>

#include <cstddef>

namespace nanochat {
namespace cuda_backend {

// The stream every backend kernel and every host copy runs on. The CUDA backend
// uses the legacy default stream (a null handle) so kernel launches, the
// synchronous cudaMemcpy/cudaMemset calls below, and kernels::Synchronize()
// stay ordered without any extra bookkeeping.
cudaStream_t Stream();

// Fail-fast wrapper around a CUDA runtime call. The seam has no error channel,
// so an unrecoverable driver error aborts rather than silently producing
// garbage. `what` names the failing call in the message.
void CheckCuda(cudaError_t status, const char* what);

// Consumes and reports the error state left by an asynchronous launch. Call
// after a kernel launch to catch invalid configuration and bad arguments.
void CheckLastError(const char* what);

#if defined(__CUDACC__)
// Launches `kernel` on the backend stream with the given grid, block, and
// dynamic shared-memory size, then checks the launch. Available only to device
// translation units; host-only gcc translation units never see the template.
template <typename Kernel, typename... Args>
void Launch(Kernel kernel, dim3 grid, dim3 block, std::size_t shared,
            Args... args) {
  kernel<<<grid, block, shared, Stream()>>>(args...);
  CheckLastError("kernel launch");
}
#endif  // defined(__CUDACC__)

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_DEVICE_H_
