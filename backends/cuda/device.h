#ifndef NANOCHAT_BACKENDS_CUDA_DEVICE_H_
#define NANOCHAT_BACKENDS_CUDA_DEVICE_H_

// Backend-internal host helpers shared by every CUDA translation unit. This
// header is deliberately *not* part of the frozen seam: it may include CUDA
// headers, because only code under backends/cuda includes it. Code above the
// seam (src/**, kernels.h) never sees it. See docs/build.md, dependency
// invariants.

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

// Prints the same one line `CheckCuda` prints for a failure, without exiting.
// For a failure the caller may recover from by taking another path, which is
// what a refused cooperative launch is: the launch site reports it once and the
// caller decides whether it has a second driver to run.
void ReportCudaError(cudaError_t status, const char* what);

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
// 9.0, docs/precision.md Part 1, section 5.3).
//
// This answer is definitive for those four bits: `GetCaps()` drops whatever the
// declaration says about precision and uses this instead. The runtime is the
// authority on what the silicon does and `caps.json` is the authority on what
// the backend implements, so a profile written for the fastest member of a
// family cannot make a Pascal card claim bf16, and a card that does have the
// units is not withheld by a declaration that has not caught up.
std::uint64_t PrecisionFeatureBits(int major, int minor);

// ---------------------------------------------------------------------------
// Cooperative launches and the persistent grid
// ---------------------------------------------------------------------------
//
// The fused Muon update has two drivers over one device function: a cooperative
// driver that launches a single grid and separates the sixteen phases with grid
// barriers, and a phase driver that launches one grid per phase
// (docs/kernels.md section 4). Three questions decide which one runs,
// and all three are put to the driver rather than to a capability bit: whether
// the device accepts a cooperative launch at all, how many blocks of the fused
// kernel's resources stay resident at once (the size of the persistent grid),
// and whether one launch of that grid was accepted. A device that answers the
// first with "no" -- the simulator answers 0 for `cudaDevAttrCooperativeLaunch`
// -- takes the phase loop, so neither `GetCaps()` nor `caps.json` gains a bit
// for any of this (docs/kernels.md section 8).

// The typed outcome of one cooperative launch. A refused launch is a value and
// not an exit: the reason a cooperative launch is rejected is one the caller
// acts on, by running the other driver. `cudaLaunchCooperativeKernel` reports
// its failures as a status, where `<<<>>>` reports them through the last-error
// slot, so the status is what this surface is built on.
enum class CooperativeLaunchOutcome {
  // The driver accepted the launch.
  kLaunched,
  // `cudaErrorCooperativeLaunchTooLarge`: the grid asks for more blocks than
  // the device can keep resident at once. This is the expected refusal rather
  // than a malfunction, and it is the caller's one log line that reports it, so
  // the launch site stays silent and the fallback is logged once.
  kTooLarge,
  // Any other launch failure. The launch site has already reported it; a caller
  // that has a second path may take it, and one that has not should stop.
  kFailed,
};

// `outcome` as the word a log line uses: "launched",
// "cooperative-launch-too-large", or "failed".
const char* CooperativeLaunchOutcomeName(CooperativeLaunchOutcome outcome);

// The typed classification of one CUDA status: `cudaSuccess` is `kLaunched`,
// `cudaErrorCooperativeLaunchTooLarge` is `kTooLarge`, and every other failure
// is `kFailed`. Pure, so the fallback can be tested without provoking the
// failure, and two call sites cannot classify the same status differently.
CooperativeLaunchOutcome ClassifyCooperativeLaunch(cudaError_t status);

// Whether the driver's `cudaDevAttrCooperativeLaunch` answer means "yes". One
// is the only value a driver sends; the test is `!= 0` so a driver that reports
// a truthy value other than one is not read as a refusal. The same shape as
// `PrecisionFeatureBits` above: a device fact in, a yes-or-no out, and no
// capability-vocabulary bit anywhere in between.
bool CooperativeLaunchSupported(int cooperative_launch_attribute);

// Asks the current device the question above. False means the fused build takes
// the phase loop: the simulator answers 0 for that attribute, so the
// cooperative driver is never attempted there (docs/kernels.md
// section 8).
bool SupportsCooperativeLaunch();

// The multiprocessor count of the current device, the second factor of the
// persistent grid. A query and not a constant: the fused driver sizes the grid
// from the card it is on.
int MultiprocessorCount();

// How many blocks one multiprocessor holds at once for `kernel` at
// `block_threads` threads with `dynamic_shared_bytes` of dynamic shared
// memory (cudaOccupancyMaxActiveBlocksPerMultiprocessor). `kernel` is the
// address of the kernel that will be launched, because the answer depends on
// the registers and static shared memory that kernel spends: a query about
// another kernel is a different question. 0 means the device cannot run that
// configuration at all, and is reported as it is, because the caller sizes a
// grid from the answer.
int ActiveBlocksPerMultiprocessor(const void* kernel, int block_threads,
                                  std::size_t dynamic_shared_bytes);

// The persistent grid the cooperative driver launches: the active blocks per
// multiprocessor for (`kernel`, `block_threads`, `dynamic_shared_bytes`) times
// `MultiprocessorCount()`. One helper so the grid the driver sizes, the grid it
// launches, and the number its fallback note quotes cannot disagree. It is 84
// on the host card: 3 blocks per SM on 28 SMs.
int ResidentBlocks(const void* kernel, int block_threads,
                   std::size_t dynamic_shared_bytes);

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

namespace detail {

// `cudaLaunchCooperativeKernel` takes its arguments as `void*[]`, one entry per
// parameter pointing at that parameter's value, where a `<<<>>>` launch takes
// them inline. The struct below builds that array from the same argument list,
// so a caller of `LaunchCooperative` passes exactly what it would pass to
// `Launch`.
template <typename... Args>
struct LaunchArguments {
  explicit LaunchArguments(Args&... args)
      : values{const_cast<void*>(static_cast<const void*>(&args))...} {}

  // A kernel with no parameters still needs a non-empty array, because a
  // zero-length array is not valid C++.
  void* values[sizeof...(Args) == 0 ? 1u : sizeof...(Args)];
};

}  // namespace detail

// Launches `kernel` as one cooperative grid on the backend stream with the
// given grid, block, and dynamic shared-memory size, and returns the typed
// outcome instead of aborting on a refusal:
//
//   const CooperativeLaunchOutcome outcome =
//       LaunchCooperative(MuonFusedKernel, dim3(grid), dim3(256), shared, ...);
//   if (outcome == CooperativeLaunchOutcome::kTooLarge) {
//     // one log line, then the phase driver
//   }
//
// The last-error check matches `Launch`'s: a launch failure is reported at the
// launch site and consumed there, so the next `CheckLastError` in the stream
// does not report the same failure a second time. `cudaLaunchCooperativeKernel`
// sets both the return value and the last-error slot, so the slot is drained
// whatever the outcome.
template <typename Kernel, typename... Args>
CooperativeLaunchOutcome LaunchCooperative(Kernel kernel, dim3 grid, dim3 block,
                                           std::size_t shared, Args... args) {
  detail::LaunchArguments<Args...> packed(args...);
  const cudaError_t status =
      cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), grid,
                                  block, packed.values, shared, Stream());
  if (status != cudaSuccess) {
    cudaGetLastError();
    const CooperativeLaunchOutcome outcome = ClassifyCooperativeLaunch(status);
    // A too-large grid is the answer that sends the caller to the phase driver,
    // so the caller's fallback line reports it. Every other failure is
    // unexpected and is reported here, at the launch site.
    if (outcome == CooperativeLaunchOutcome::kFailed) {
      ReportCudaError(status, "cooperative kernel launch");
    }
    return outcome;
  }
  CheckLastError("cooperative kernel launch");
  return CooperativeLaunchOutcome::kLaunched;
}
#endif  // defined(__CUDACC__)

}  // namespace cuda_backend
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_DEVICE_H_
