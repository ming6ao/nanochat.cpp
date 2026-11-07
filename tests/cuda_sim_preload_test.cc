// tests/cuda_sim_preload_test.cc -- the real interposition path
// (docs/simulator.md sections 5.1, 5.2, and 12.2).
//
// Unlike `//tests:cuda_sim_test` (an in-process mock unit test), this test
// links the *real* CUDA backend -- `backends/cuda/device.cu` and
// `backends/cuda/gemm.cu` -- and is launched with
// `LD_PRELOAD=libcuda_sim_interposer.so`. The dynamic linker therefore binds
// the calls below to the mock. The test then reads the mock's JSON log from
// `NANOCHAT_SIM_LOG` and asserts the binding canary and the named semantic
// invariants.
//
// This is what makes the API interposer real rather than a library that is
// merely linked in. It is `manual`: `tools/nanochat simulate --suite api`
// supplies `LD_PRELOAD`, `NANOCHAT_SIM_LOG`, and `NANOCHAT_SIM_PROFILE`. The
// GEMM invariant in particular derives from the real `gemm.cu` decision (which
// compute type and algorithm the device code chooses), not from a record the
// test constructed itself.
//
// The fused Muon build's launch surface is driven the same way
// (docs/kernels.md section 8). `device.cu`'s
// `SupportsCooperativeLaunch`, `MultiprocessorCount`,
// `ActiveBlocksPerMultiprocessor`, and `ResidentBlocks` are called here, so the
// answers in the log are the ones the real backend read from the preloaded
// library, and one cooperative launch goes through `LaunchCooperative` with
// `//backends/cuda:launch_probe`'s kernel -- the one kernel symbol a host-only
// test can launch. The mock runs no block, so the probe reports that no block
// wrote; what the checks ride on is the record: the grid, the block size, and
// the shared-memory size the driver gave, and the
// `cudaErrorCooperativeLaunchTooLarge` refusal a grid above the mock's
// residency earns.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backends/cuda/device.h"
#include "backends/cuda/launch_probe.h"
#include "nanochat/device_profile.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

std::vector<std::string> Lines(const std::string& log) {
  std::vector<std::string> lines;
  std::string current;
  for (char c : log) {
    if (c == '\n') {
      if (!current.empty()) lines.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) lines.push_back(current);
  return lines;
}

std::string ReadFile(const char* path) {
  std::string out;
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) return out;
  char buffer[4096];
  std::size_t n = 0;
  while ((n = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
    out.append(buffer, n);
  }
  std::fclose(file);
  return out;
}

bool Contains(const std::string& log, const char* needle) {
  return log.find(needle) != std::string::npos;
}

// True when the log holds one `call` record for `name` that carries every one
// of `tokens`. A record is one line, so the tokens cannot be satisfied by two
// different calls.
bool HasRecord(const std::string& log, const char* name,
               const std::vector<std::string>& tokens) {
  const std::string call = std::string("\"call\":\"") + name + "\"";
  for (const std::string& line : Lines(log)) {
    if (line.find(call) == std::string::npos) continue;
    bool all = true;
    for (const std::string& token : tokens) {
      if (line.find(token) == std::string::npos) {
        all = false;
        break;
      }
    }
    if (all) return true;
  }
  return false;
}

// The canary is a *per-run* binding proof: the first record must name the
// profile this run simulates. A canary for any other profile is the failure the
// negative check below exercises (docs/simulator.md section 12.2, invariant 7).
bool CanaryNamesProfile(const std::string& log, const std::string& profile) {
  const std::vector<std::string> lines = Lines(log);
  if (lines.empty()) return false;
  if (lines.front().find("\"kind\":\"canary\"") == std::string::npos) {
    return false;
  }
  return lines.front().find("\"profile\":\"" + profile + "\"") !=
         std::string::npos;
}

// Drives the real backend through the mock. `kernels::GetCaps` reads the
// fabricated `cudaDeviceProp` (the ABI the P0 fix repairs); `Gemm` with
// `batch_count == 1` takes gemm.cu's `cublasSgemm` path and `batch_count > 1`
// its `cublasGemmStridedBatchedEx` path.
int DriveTheRealBackend(const std::string& profile_name) {
  const nanochat::DeviceProfile* expected =
      nanochat::FindDeviceProfile(profile_name);
  ExpectTrue(expected != nullptr,
             "the simulated profile resolves in the device table");

  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  ExpectTrue(caps.is_device, "GetCaps reports a device through the mock");
  if (expected != nullptr) {
    ExpectTrue(std::string(caps.device_name) == expected->name,
               "the fabricated prop names the simulated profile");
    ExpectTrue(caps.compute_major == expected->compute_major &&
                   caps.compute_minor == expected->compute_minor,
               "the fabricated prop reports the profile compute capability");
    ExpectTrue(caps.has_tensor_cores == expected->has_tensor_cores,
               "the fabricated prop reports the profile tensor-core flag");
  }

  constexpr int kM = 4;
  constexpr int kN = 4;
  constexpr int kK = 4;
  float* a = static_cast<float*>(nanochat::kernels::Alloc(
      static_cast<std::size_t>(kM) * kK * sizeof(float)));
  float* b = static_cast<float*>(nanochat::kernels::Alloc(
      static_cast<std::size_t>(kK) * kN * sizeof(float)));
  float* c = static_cast<float*>(nanochat::kernels::Alloc(
      static_cast<std::size_t>(kM) * kN * sizeof(float)));
  ExpectTrue(a != nullptr && b != nullptr && c != nullptr,
             "Alloc returns device memory");
  if (a == nullptr || b == nullptr || c == nullptr) return 1;

  std::vector<float> host_a(static_cast<std::size_t>(kM) * kK, 1.0f);
  std::vector<float> host_b(static_cast<std::size_t>(kK) * kN, 0.5f);
  nanochat::kernels::Memcpy(a, host_a.data(), host_a.size() * sizeof(float),
                            nanochat::CopyDir::kHostToDevice);
  nanochat::kernels::Memcpy(b, host_b.data(), host_b.size() * sizeof(float),
                            nanochat::CopyDir::kHostToDevice);
  nanochat::kernels::Memset(c, 0, sizeof(float) * kM * kN);
  nanochat::kernels::Synchronize();

  nanochat::GemmParams single;
  single.m = kM;
  single.n = kN;
  single.k = kK;
  single.batch_count = 1;
  nanochat::kernels::Gemm(nanochat::GemmMode::kForward, single, a, b, c);

  nanochat::GemmParams batched = single;
  batched.batch_count = 2;
  batched.stride_a = kM * kK;
  batched.stride_b = kK * kN;
  batched.stride_c = kM * kN;
  nanochat::kernels::Gemm(nanochat::GemmMode::kForward, batched, a, b, c);

  nanochat::kernels::Synchronize();

  // A device-to-host read proves the copy path returned the right bytes.
  std::vector<float> host_c(static_cast<std::size_t>(kM) * kN, 0.0f);
  nanochat::kernels::Memcpy(host_c.data(), c, host_c.size() * sizeof(float),
                            nanochat::CopyDir::kDeviceToHost);
  ExpectTrue(host_c.size() == static_cast<std::size_t>(kM) * kN,
             "the device-to-host copy returned the output");

  // The scaled fp8 path (docs/simulator.md Part 7, section 5). The mock
  // records and validates the cublasLt call shape; it does not compute. Only an
  // fp8 profile accepts the call, so the block is guarded by the capability.
  if (caps.supports_fp8) {
    constexpr int kFp8 = 16;
    const std::size_t fp8_count = static_cast<std::size_t>(kFp8) * kFp8;
    std::uint8_t* qx =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(fp8_count));
    std::uint8_t* qw =
        static_cast<std::uint8_t*>(nanochat::kernels::Alloc(fp8_count));
    float* scale_x =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));
    float* scale_w =
        static_cast<float*>(nanochat::kernels::Alloc(sizeof(float)));
    // The forward output is the compute type (docs/precision.md Part 1,
    // section 7).
    nanochat::ComputeType* y = static_cast<nanochat::ComputeType*>(
        nanochat::kernels::Alloc(fp8_count * sizeof(nanochat::ComputeType)));
    const bool allocated = qx != nullptr && qw != nullptr &&
                           scale_x != nullptr && scale_w != nullptr &&
                           y != nullptr;
    ExpectTrue(allocated, "the fp8 buffers allocate");
    if (allocated) {
      // A finite, positive scale: the mock checks the range, not the value
      // (docs/simulator.md Part 7, section 5.2).
      const float one = 1.0f;
      nanochat::kernels::Memcpy(scale_x, &one, sizeof(float),
                                nanochat::CopyDir::kHostToDevice);
      nanochat::kernels::Memcpy(scale_w, &one, sizeof(float),
                                nanochat::CopyDir::kHostToDevice);
      nanochat::ScaledGemmParams scaled;
      scaled.m = kFp8;
      scaled.n = kFp8;
      scaled.k = kFp8;
      scaled.transpose_b = true;
      scaled.format_a = nanochat::Fp8Format::kE4M3;
      scaled.format_b = nanochat::Fp8Format::kE4M3;
      scaled.scale_a = scale_x;
      scaled.scale_b = scale_w;
      nanochat::kernels::GemmScaledFp8(nanochat::GemmMode::kForward, scaled, qx,
                                       qw, y);
      nanochat::kernels::Synchronize();
    }
    nanochat::kernels::Free(qx);
    nanochat::kernels::Free(qw);
    nanochat::kernels::Free(scale_x);
    nanochat::kernels::Free(scale_w);
    nanochat::kernels::Free(y);
  }

  nanochat::kernels::Free(a);
  nanochat::kernels::Free(b);
  nanochat::kernels::Free(c);
  return 0;
}

// The fused Muon driver's launch surface, through the real backend
// (`backends/cuda/device.cu`) and therefore through whatever `LD_PRELOAD` put
// in front of the CUDA runtime (docs/kernels.md section 8).
//
// The three queries are ordinary host functions, so a host-only test can ask
// them; the launch needs a kernel symbol, which `//backends/cuda:launch_probe`
// supplies -- a kernel compiled with the fused kernel's launch resources, so
// the occupancy answer is about a real kernel and not about an address the test
// invented.
void DriveTheFusedLaunchSurface() {
  using nanochat::cuda_backend::ActiveBlocksPerMultiprocessor;
  using nanochat::cuda_backend::CooperativeLaunchOutcome;
  using nanochat::cuda_backend::MultiprocessorCount;
  using nanochat::cuda_backend::ResidentBlocks;
  using nanochat::cuda_backend::SupportsCooperativeLaunch;
  using nanochat::cuda_backend::testing::kProbeBlockThreads;
  using nanochat::cuda_backend::testing::kProbeDynamicSharedBytes;
  using nanochat::cuda_backend::testing::ProbeCooperativeLaunch;
  using nanochat::cuda_backend::testing::ProbeKernel;

  // The first question the driver puts to the device. Zero means "no", and a
  // driver that reads it takes the phase loop: this is the simulated run's
  // answer, read through the real backend's own helper.
  ExpectTrue(!SupportsCooperativeLaunch(),
             "the mock answers 0 for cudaDevAttrCooperativeLaunch, so the "
             "phase driver is the one a simulated run takes");
  ExpectTrue(MultiprocessorCount() == 1, "the mock reports one multiprocessor");

  // The persistent grid the cooperative driver would size: one block per
  // multiprocessor over one multiprocessor.
  const void* kernel = ProbeKernel();
  ExpectTrue(ActiveBlocksPerMultiprocessor(kernel, kProbeBlockThreads,
                                           kProbeDynamicSharedBytes) == 1,
             "the occupancy query answers one block per SM through the mock");
  ExpectTrue(
      ResidentBlocks(kernel, kProbeBlockThreads, kProbeDynamicSharedBytes) == 1,
      "a persistent grid sized from the mock is one block");

  // An accepted cooperative launch. The mock records the launch and runs no
  // block, so `blocks_ran` stays zero; the record is what the log checks below
  // read.
  int blocks_ran = -1;
  ExpectTrue(ProbeCooperativeLaunch(1, &blocks_ran) ==
                 CooperativeLaunchOutcome::kLaunched,
             "a one-block cooperative launch is accepted through the mock");
  ExpectTrue(blocks_ran == 0,
             "the mock records the cooperative launch and runs no block");

  // The refusal the fused driver answers with its phase fallback: a grid above
  // what the device keeps resident at once.
  blocks_ran = -1;
  ExpectTrue(ProbeCooperativeLaunch(2, &blocks_ran) ==
                 CooperativeLaunchOutcome::kTooLarge,
             "a grid above the resident capacity reports "
             "cudaErrorCooperativeLaunchTooLarge");
  ExpectTrue(blocks_ran == 0, "a refused cooperative launch runs no block");
}

void CheckLog(const std::string& log, const std::string& profile,
              bool expect_tensor_cores) {
  // Invariant 7: the binding canary.
  ExpectTrue(CanaryNamesProfile(log, profile),
             "the canary names NANOCHAT_SIM_PROFILE");
  const std::string other = profile == "h100" ? "t4" : "h100";
  ExpectTrue(!CanaryNamesProfile(log, other),
             "a canary for a different profile is not accepted");

  // The mock actually ran and interposed the real backend's calls.
  ExpectTrue(Contains(log, "\"call\":\"cudaMalloc\""),
             "the mock recorded cudaMalloc");
  ExpectTrue(Contains(log, "\"call\":\"cudaGetDeviceProperties_v2\""),
             "the mock recorded cudaGetDeviceProperties_v2");
  ExpectTrue(Contains(log, "\"call\":\"cudaMemcpy\""),
             "the mock recorded cudaMemcpy");
  ExpectTrue(Contains(log, "\"call\":\"cudaMemset\""),
             "the mock recorded cudaMemset");
  ExpectTrue(Contains(log, "\"call\":\"cudaDeviceSynchronize\""),
             "the mock recorded cudaDeviceSynchronize");

  // Both GEMM entry points were taken by the real gemm.cu. The single call is
  // `cublasSgemm_v2`; the batched call builds a cublasLt plan on its first use
  // and falls back to `cublasGemmStridedBatchedEx` only when the heuristic
  // returns no algorithm (`backends/cuda/gemm.cu`, `GemmBatched`). The mock
  // answers the heuristic with one algorithm, so the batched call here is a
  // `cublasLtMatmul` with fp32 operands.
  ExpectTrue(Contains(log, "\"call\":\"cublasSgemm_v2\""),
             "gemm.cu's single-call path used cublasSgemm_v2");
  bool batched = false;
  for (const std::string& line : Lines(log)) {
    if (line.find("\"call\":\"cublasGemmStridedBatchedEx\"") !=
        std::string::npos) {
      batched = true;
      continue;
    }
    if (line.find("\"call\":\"cublasLtMatmul\"") == std::string::npos) {
      continue;
    }
    // The fp8 call is the one the block below checks; it carries fp8 operands.
    if (line.find("\"fp8\":false") == std::string::npos) continue;
    if (line.find("\"verdict\":\"ok") == std::string::npos) continue;
    batched = true;
  }
  ExpectTrue(batched, "gemm.cu's batched path reached a batched GEMM");

  // Invariant 8: every *accepted* batched GEMM accumulates in fp32. The
  // compute type is the one the real gemm.cu chose, in whichever entry point it
  // took: the operation descriptor's compute type on the cublasLt path, or the
  // compute type the call carries on the strided-batched path. On a
  // non-tensor-core profile there is no such invariant; instead none may be
  // reported as tensor-core.
  bool checked = false;
  bool saw_tensor_core = false;
  for (const std::string& line : Lines(log)) {
    const bool lt =
        line.find("\"call\":\"cublasLtMatmul\"") != std::string::npos;
    const bool strided =
        line.find("cublasGemmStridedBatchedEx") != std::string::npos;
    if (!lt && !strided) continue;
    if (line.find("\"tensor_cores\":true") != std::string::npos) {
      saw_tensor_core = true;
    }
    // The fp8 call is the one the block below checks, not a batched GEMM.
    if (lt && line.find("\"fp8\":false") == std::string::npos) continue;
    // An accepted verdict may carry the legacy-enum note appended, so match the
    // `ok` prefix rather than the exact token.
    if (line.find("\"verdict\":\"ok") == std::string::npos) continue;
    checked = true;
    ExpectTrue(line.find("\"compute_type\":68") != std::string::npos,
               "the batched GEMM accumulates with CUBLAS_COMPUTE_32F");
  }
  if (expect_tensor_cores) {
    ExpectTrue(checked, "at least one accepted batched GEMM was recorded");
  } else {
    ExpectTrue(!saw_tensor_core,
               "a non-tensor-core profile records no tensor-core GEMM");
  }

  // The scaled fp8 call shape (docs/simulator.md Part 7, section 5.2). The
  // fp8 call is the cublasLt matmul whose operands are fp8; a batched fp32 GEMM
  // also goes through cublasLt, so the operand types select it.
  const nanochat::DeviceProfile* entry = nanochat::FindDeviceProfile(profile);
  if (entry != nullptr && entry->supports_fp8) {
    ExpectTrue(Contains(log, "\"call\":\"cublasLtMatmul\""),
               "the fp8 path issued cublasLtMatmul");
    bool accepted = false;
    for (const std::string& line : Lines(log)) {
      if (line.find("\"call\":\"cublasLtMatmul\"") == std::string::npos) {
        continue;
      }
      if (line.find("\"fp8\":true") == std::string::npos) continue;
      ExpectTrue(line.find("\"verdict\":\"ok") != std::string::npos,
                 "the fp8 cublasLt call was accepted on an fp8 profile");
      ExpectTrue(line.find("\"c_type\":0") != std::string::npos,
                 "the fp8 call writes an fp32 accumulator");
      accepted = true;
    }
    ExpectTrue(accepted, "at least one cublasLtMatmul carried an fp8 operand");
  }

  // The fused Muon build's three runtime names (docs/kernels.md
  // section 8). A fused build asks all three even when the launch it finally
  // makes is the phase driver's, so every one of them is a record of what the
  // real backend asked and what the mock answered.
  ExpectTrue(HasRecord(log, "cudaDeviceGetAttribute",
                       {"\"attribute\":95", "\"value\":0"}),
             "the mock answered 0 for cudaDevAttrCooperativeLaunch");
  ExpectTrue(HasRecord(log, "cudaDeviceGetAttribute",
                       {"\"attribute\":16", "\"value\":1"}),
             "the mock answered one multiprocessor");
  ExpectTrue(HasRecord(log, "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                       {"\"blocks\":1", "\"verdict\":\"ok\""}),
             "the mock answered one block per multiprocessor, and recorded it");
  const std::string probe_threads =
      std::to_string(nanochat::cuda_backend::testing::kProbeBlockThreads);
  ExpectTrue(
      HasRecord(
          log, "cudaLaunchCooperativeKernel",
          {"\"grid_x\":1", "\"block_threads\":" + probe_threads,
           "\"shared_bytes\":" +
               std::to_string(
                   nanochat::cuda_backend::testing::kProbeDynamicSharedBytes),
           "\"cooperative\":true", "\"verdict\":\"ok\""}),
      "the accepted cooperative launch is recorded with its grid, "
      "block, and shared size");
  ExpectTrue(HasRecord(log, "cudaLaunchCooperativeKernel",
                       {"\"grid_blocks\":2", "\"resident_blocks\":1"}),
             "the refused cooperative launch is recorded with both block "
             "counts");

  // A log without addresses stays comparable across runs.
  ExpectTrue(log.find("0x") == std::string::npos,
             "no pointer value appears in the log");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  const char* log_path = std::getenv("NANOCHAT_SIM_LOG");
  if (log_path == nullptr || *log_path == '\0') {
    std::printf(
        "cuda sim preload: NANOCHAT_SIM_LOG is unset; run this through "
        "`tools/nanochat simulate --suite api`\n");
    return 0;
  }
  // The mock truncates its log on the first record (the canary), so the log
  // path itself must not be removed: the real CUDA runtime makes its first
  // call from an ELF constructor before `main`.

  const char* profile = std::getenv("NANOCHAT_SIM_PROFILE");
  const std::string wanted =
      (profile != nullptr && *profile != '\0') ? profile : "";

  DriveTheRealBackend(wanted);
  DriveTheFusedLaunchSurface();

  const std::string log = ReadFile(log_path);
  if (log.empty()) {
    Fail("the mock wrote no log; is LD_PRELOAD set to the interposer?");
  } else {
    const nanochat::DeviceProfile* expected =
        nanochat::FindDeviceProfile(wanted);
    CheckLog(log, wanted, expected != nullptr && expected->has_tensor_cores);
  }

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("cuda sim preload: all checks passed (profile=%s)\n",
              wanted.c_str());
  return 0;
}
