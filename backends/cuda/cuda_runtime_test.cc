// T1 GPU correctness for the CUDA device runtime and the cuBLAS GEMM. Host
// code only: it calls the sealed nanochat/kernels.h API, so it exercises
// exactly the seam the rest of the project links against and stays free of
// vendor headers. Tiny shapes only, per docs/testing.md.
//
// It also covers the launch surface of `backends/cuda/device.h` that the fused
// Muon driver is built on (docs/kernels.md sections 4 and 8): the
// typed outcome of a cooperative launch and the resident-block query the
// persistent grid is sized from. That surface is backend-internal rather than
// part of the sealed seam, so it is read through `backends/cuda/device.h` --
// the header is CUDA headers and no vendor kernel constructs, and the one
// device translation unit the test links is the test-only `launch_probe`.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backends/cuda/device.h"
#include "backends/cuda/launch_probe.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::Caps;
using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::kComputeDType;
using nanochat::kernels::Alloc;
using nanochat::kernels::Free;
using nanochat::kernels::Gemm;
using nanochat::kernels::GetCaps;
using nanochat::kernels::Memcpy;
using nanochat::kernels::Memset;
using nanochat::kernels::Synchronize;

float ToFloat(float value) { return value; }

ComputeType FromFloat(float value) { return value; }

constexpr float kTolerance = 1e-3f;

double g_worst_error = 0.0;

// `NANOCHAT_SIM_PROFILE` is set when the simulator interposes the runtime
// (docs/simulator.md section 5.1), and the mock answers the queries below with
// fabricated values on purpose: 0 for cooperative-launch support and 1 block
// per multiprocessor. The quantities that hold on the host card are therefore
// pinned for a real device only, and the launch itself is attempted only where
// the device says it can run one.
bool Simulated() {
  const char* profile = std::getenv("NANOCHAT_SIM_PROFILE");
  return profile != nullptr && *profile != '\0';
}

int g_failures = 0;

void Expect(bool condition, const char* what) {
  if (condition) return;
  std::printf("FAIL: %s\n", what);
  ++g_failures;
}

// Stores A and B in the layout the transpose flags imply, computes the host
// reference for C = alpha * op(A) * op(B) + beta * C, runs the device GEMM, and
// compares. All three modes share this universal formula.
bool RunCase(GemmMode mode, bool ta, bool tb, int m, int n, int k, int batch,
             float alpha, float beta) {
  const int rows_a = ta ? k : m;
  const int lda = ta ? m : k;
  const int rows_b = tb ? n : k;
  const int ldb = tb ? k : n;
  const int ldc = n;

  const std::int64_t stride_a = static_cast<std::int64_t>(rows_a) * lda;
  const std::int64_t stride_b = static_cast<std::int64_t>(rows_b) * ldb;
  const std::int64_t stride_c = static_cast<std::int64_t>(m) * ldc;

  const std::size_t total_a = static_cast<std::size_t>(batch) * stride_a;
  const std::size_t total_b = static_cast<std::size_t>(batch) * stride_b;
  const std::size_t total_c = static_cast<std::size_t>(batch) * stride_c;

  std::vector<ComputeType> host_a(total_a);
  std::vector<ComputeType> host_b(total_b);
  std::vector<ComputeType> host_c(total_c);
  std::vector<float> ref_a(total_a);
  std::vector<float> ref_b(total_b);
  std::vector<float> ref_c(total_c);

  for (std::size_t i = 0; i < total_a; ++i) {
    host_a[i] =
        FromFloat(0.25f * static_cast<float>(static_cast<int>(i % 9) - 4));
    ref_a[i] = ToFloat(host_a[i]);
  }
  for (std::size_t i = 0; i < total_b; ++i) {
    host_b[i] =
        FromFloat(0.2f * static_cast<float>(static_cast<int>(i % 7) - 3));
    ref_b[i] = ToFloat(host_b[i]);
  }
  for (std::size_t i = 0; i < total_c; ++i) {
    host_c[i] = FromFloat(0.1f * static_cast<float>(static_cast<int>(i % 5)));
    ref_c[i] = ToFloat(host_c[i]);
  }

  for (int bi = 0; bi < batch; ++bi) {
    const float* a = ref_a.data() + bi * stride_a;
    const float* b = ref_b.data() + bi * stride_b;
    float* c = ref_c.data() + bi * stride_c;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int l = 0; l < k; ++l) {
          const float av = ta ? a[l * lda + i] : a[i * lda + l];
          const float bv = tb ? b[j * ldb + l] : b[l * ldb + j];
          acc += av * bv;
        }
        c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
      }
    }
  }

  void* dev_a = Alloc(total_a * sizeof(ComputeType));
  void* dev_b = Alloc(total_b * sizeof(ComputeType));
  void* dev_c = Alloc(total_c * sizeof(ComputeType));
  Memcpy(dev_a, host_a.data(), total_a * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(dev_b, host_b.data(), total_b * sizeof(ComputeType),
         CopyDir::kHostToDevice);
  Memcpy(dev_c, host_c.data(), total_c * sizeof(ComputeType),
         CopyDir::kHostToDevice);

  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.batch_count = batch;
  params.alpha = alpha;
  params.beta = beta;
  params.transpose_a = ta;
  params.transpose_b = tb;
  Gemm(mode, params, static_cast<const ComputeType*>(dev_a),
       static_cast<const ComputeType*>(dev_b),
       static_cast<ComputeType*>(dev_c));
  Synchronize();
  Memcpy(host_c.data(), dev_c, total_c * sizeof(ComputeType),
         CopyDir::kDeviceToHost);

  Free(dev_a);
  Free(dev_b);
  Free(dev_c);

  for (std::size_t i = 0; i < total_c; ++i) {
    const float got = ToFloat(host_c[i]);
    const double diff = std::fabs(got - ref_c[i]);
    if (diff > g_worst_error) g_worst_error = diff;
    if (diff > kTolerance) {
      std::fprintf(stderr,
                   "gemm mismatch (mode=%d ta=%d tb=%d m=%d n=%d k=%d "
                   "batch=%d at %zu): got %f expected %f\n",
                   static_cast<int>(mode), ta ? 1 : 0, tb ? 1 : 0, m, n, k,
                   batch, i, got, ref_c[i]);
      return false;
    }
  }
  return true;
}

// The cooperative-launch surface and the resident-block query
// (docs/kernels.md sections 4 and 8). The first block is pure host
// arithmetic: the classification of a status, and the reading of the
// cooperative-launch attribute the simulator answers 0 for. The second block
// asks the device, and pins the numbers the design sizes the persistent grid
// from: 3 blocks per SM on the host card's 28 SMs, 84 blocks of persistent
// grid.
void CheckLaunchSurface(const Caps& caps) {
  using nanochat::cuda_backend::ActiveBlocksPerMultiprocessor;
  using nanochat::cuda_backend::ClassifyCooperativeLaunch;
  using nanochat::cuda_backend::CooperativeLaunchOutcome;
  using nanochat::cuda_backend::CooperativeLaunchOutcomeName;
  using nanochat::cuda_backend::CooperativeLaunchSupported;
  using nanochat::cuda_backend::MultiprocessorCount;
  using nanochat::cuda_backend::ResidentBlocks;
  using nanochat::cuda_backend::SupportsCooperativeLaunch;
  using nanochat::cuda_backend::testing::kProbeBlockThreads;
  using nanochat::cuda_backend::testing::kProbeDynamicSharedBytes;
  using nanochat::cuda_backend::testing::ProbeCooperativeLaunch;
  using nanochat::cuda_backend::testing::ProbeKernel;

  // --- the typed outcome, without provoking a failure ----------------------
  const CooperativeLaunchOutcome classified_ok =
      ClassifyCooperativeLaunch(cudaSuccess);
  Expect(classified_ok == CooperativeLaunchOutcome::kLaunched,
         "cudaSuccess classifies as kLaunched");
  const CooperativeLaunchOutcome classified_too_large =
      ClassifyCooperativeLaunch(cudaErrorCooperativeLaunchTooLarge);
  Expect(classified_too_large == CooperativeLaunchOutcome::kTooLarge,
         "cudaErrorCooperativeLaunchTooLarge classifies as kTooLarge");
  Expect(ClassifyCooperativeLaunch(cudaErrorInvalidConfiguration) ==
             CooperativeLaunchOutcome::kFailed,
         "any other launch error classifies as kFailed");
  // The three names are distinct, so a log line cannot say the wrong thing.
  Expect(std::string(CooperativeLaunchOutcomeName(classified_ok)) !=
             std::string(CooperativeLaunchOutcomeName(classified_too_large)),
         "kLaunched and kTooLarge have distinct names");
  Expect(std::string(CooperativeLaunchOutcomeName(classified_too_large)) !=
             std::string(CooperativeLaunchOutcomeName(
                 CooperativeLaunchOutcome::kFailed)),
         "kTooLarge and kFailed have distinct names");

  // The simulator answers 0 for `cudaDevAttrCooperativeLaunch` and the fused
  // build then takes the phase loop (section 8), so a 0 must read as "no" and
  // only a non-zero as "yes".
  Expect(!CooperativeLaunchSupported(0), "attribute 0 is a refusal");
  Expect(CooperativeLaunchSupported(1), "attribute 1 is support");
  Expect(CooperativeLaunchSupported(2),
         "a truthy attribute other than 1 is still support");

  // --- what the device answers --------------------------------------------
  const bool simulated = Simulated();
  const bool supports_cooperative = SupportsCooperativeLaunch();
  std::printf("cuda_runtime: cooperative launch %s, %d SMs\n",
              supports_cooperative ? "supported" : "not supported",
              MultiprocessorCount());
  if (simulated) {
    std::printf(
        "cuda_runtime: NOTE simulated device; the mock answers 1 active block "
        "per SM, so the host-card quantities are not pinned here\n");
  }

  const int multiprocessors = MultiprocessorCount();
  Expect(multiprocessors > 0, "the device reports at least one SM");
  const int blocks_per_sm = ActiveBlocksPerMultiprocessor(
      ProbeKernel(), kProbeBlockThreads, kProbeDynamicSharedBytes);
  Expect(blocks_per_sm >= 1,
         "the fused kernel's resources leave at least one block per SM");
  const int resident = ResidentBlocks(ProbeKernel(), kProbeBlockThreads,
                                      kProbeDynamicSharedBytes);
  Expect(resident == blocks_per_sm * multiprocessors,
         "the resident-block count is the per-SM count times the SM count");
  std::printf(
      "cuda_runtime: %d blocks per SM at %d threads and %zu bytes of dynamic "
      "shared memory, %d resident blocks\n",
      blocks_per_sm, kProbeBlockThreads, kProbeDynamicSharedBytes, resident);

  // The host card (gtx1080ti): 3 blocks per SM on 28 SMs, which is the 84-block
  // persistent grid docs/kernels.md section 4 sizes the cooperative
  // driver from. Pinned for the real sm_61 device only.
  if (!simulated && caps.compute_major == 6 && caps.compute_minor == 1 &&
      multiprocessors == 28) {
    Expect(blocks_per_sm == 3, "3 blocks per SM on the host card");
    Expect(resident == 84, "84 resident blocks on the host card");
  }

  // --- one launch of that grid, and one the device must refuse -------------
  if (!supports_cooperative) {
    std::printf(
        "cuda_runtime: NOTE no cooperative launch on this device; the fused "
        "driver takes the phase loop\n");
    return;
  }
  int blocks_ran = -1;
  const CooperativeLaunchOutcome launched =
      ProbeCooperativeLaunch(resident, &blocks_ran);
  Expect(launched == CooperativeLaunchOutcome::kLaunched,
         "a cooperative launch of the whole persistent grid is accepted");
  Expect(blocks_ran == resident,
         "every resident block of that launch ran to completion");

  // The forced refusal: a grid four times the device's resident capacity. The
  // typed outcome must say so, one launch later, and nothing may run.
  blocks_ran = -1;
  const CooperativeLaunchOutcome refused =
      ProbeCooperativeLaunch(resident * 4 + 1, &blocks_ran);
  Expect(refused == CooperativeLaunchOutcome::kTooLarge,
         "a grid past the resident capacity returns kTooLarge, not a fatal "
         "error");
  Expect(blocks_ran == 0, "a refused cooperative launch runs no block");
  std::printf(
      "cuda_runtime: cooperative launch outcome %s for a grid of %d, "
      "%s for a grid of %d\n",
      CooperativeLaunchOutcomeName(launched), resident,
      CooperativeLaunchOutcomeName(refused), resident * 4 + 1);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  const nanochat::Caps caps = GetCaps();
  if (!caps.is_device || !caps.has_cublas) {
    std::fprintf(stderr, "cuda_runtime: no device or cuBLAS available\n");
    return 1;
  }
  if (!caps.Supports(kComputeDType)) {
    std::fprintf(stderr, "cuda_runtime: build precision not supported\n");
    return 1;
  }
  std::printf("cuda_runtime: device %s, compute capability %d.%d\n",
              caps.device_name, caps.compute_major, caps.compute_minor);

  // Device round trip: alloc, copy in, memset, copy back, free.
  {
    constexpr std::size_t kCount = 64;
    const std::size_t bytes = kCount * sizeof(ComputeType);
    void* buffer = Alloc(bytes);
    std::vector<ComputeType> host(kCount, FromFloat(1.0f));
    Memcpy(buffer, host.data(), bytes, CopyDir::kHostToDevice);
    Memset(buffer, 0, bytes);
    Memcpy(host.data(), buffer, bytes, CopyDir::kDeviceToHost);
    Free(buffer);
    Synchronize();
    for (std::size_t i = 0; i < kCount; ++i) {
      if (ToFloat(host[i]) != 0.0f) {
        std::fprintf(stderr, "cuda_runtime: memset round trip failed\n");
        return 1;
      }
    }
  }

  struct Case {
    GemmMode mode;
    bool ta;
    bool tb;
    int m;
    int n;
    int k;
    int batch;
    float alpha;
    float beta;
  };
  const Case cases[] = {
      {GemmMode::kForward, false, false, 4, 3, 5, 1, 1.0f, 0.0f},
      {GemmMode::kForward, false, true, 3, 4, 5, 1, 0.5f, 0.25f},
      {GemmMode::kForward, true, false, 3, 4, 5, 1, 1.0f, 0.0f},
      {GemmMode::kForward, true, true, 4, 3, 5, 1, 1.0f, 0.5f},
      {GemmMode::kForward, false, false, 2, 3, 4, 3, 1.0f, 0.0f},
      {GemmMode::kDgrad, true, false, 3, 2, 4, 1, 1.0f, 0.0f},
      {GemmMode::kWgrad, false, true, 2, 4, 3, 1, 1.0f, 0.0f},
  };

  for (const Case& c : cases) {
    if (!RunCase(c.mode, c.ta, c.tb, c.m, c.n, c.k, c.batch, c.alpha, c.beta)) {
      return 1;
    }
  }

  // The launch surface the fused Muon driver is built on. It reports its own
  // failures and counts them, so the GEMM cases above keep their exit code.
  CheckLaunchSurface(caps);
  if (g_failures != 0) {
    std::fprintf(stderr, "cuda_runtime: %d launch-surface check(s) failed\n",
                 g_failures);
    return 1;
  }

  std::printf("cuda_runtime: ok, %zu gemm cases verified",
              sizeof(cases) / sizeof(cases[0]));
  std::printf(" (max abs error %.3g, tolerance %.3g)\n", g_worst_error,
              kTolerance);
  return 0;
}
