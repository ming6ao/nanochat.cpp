#ifndef NANOCHAT_BACKENDS_CUDA_DEVICE_H_
#define NANOCHAT_BACKENDS_CUDA_DEVICE_H_

// Backend-internal host helpers shared by every CUDA translation unit. This
// header is deliberately *not* part of the frozen seam: it may include CUDA
// headers, because only code under backends/cuda includes it. Code above the
// seam (src/**, kernels.h) never sees it. See DESIGN.md section 2.1.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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

// ---------------------------------------------------------------------------
// The capability declaration `GetCaps()` reports
// ---------------------------------------------------------------------------
//
// `Caps::feature_mask` is filled from exactly two sources and no third one
// (docs/model-and-device-bringup.md section 9 rule 1, `spec/capability_bits.md`
// section 1): the caps.json-derived profile entry for the device the runtime
// reports, and the precision facts the silicon itself reports. Both helpers
// below are exposed so a test can compare `GetCaps()` against the two inputs it
// claims to read instead of against a copy of the answer.

// The bits `backends/cuda/caps.json` declares for `device_name`, read through
// the derived profile entry (`nanochat/device_profile.h`) with the same
// normalized join the profile lookup uses. Returns 0 when no entry matches, or
// when the entry carries no mask because the derivation from `caps.json` has
// not regenerated the table yet. Zero means "this backend has declared nothing
// for this device", never "this backend can do nothing": the precision bits
// still come from the silicon below.
std::uint64_t DeclaredFeatureMask(const char* device_name);

// The precision bits the silicon at compute capability `major.minor` reports.
// fp32 is unconditional, because fp32 *is* this backend's compute type
// (`nanochat/tensor.h`'s `ComputeType`): a card that could not do fp32 could
// not run one kernel here. bf16 arithmetic needs Ampere's bf16 units (sm_80),
// and the fp8 E4M3/E5M2 tensor core arrives with Ada (sm_89; Hopper reports
// 9.0, docs/fp8-training.md section 5.3).
//
// This answer is definitive for those four bits: `GetCaps()` drops whatever the
// declaration says about precision and uses this instead. The runtime is the
// authority on what the silicon does and `caps.json` is the authority on what
// the backend implements, so a profile written for the fastest member of a
// family cannot make a Pascal card claim bf16, and a card that does have the
// units is not withheld by a declaration that has not caught up.
std::uint64_t PrecisionFeatureBits(int major, int minor);

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
