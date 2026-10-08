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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

  nanochat::kernels::Free(a);
  nanochat::kernels::Free(b);
  nanochat::kernels::Free(c);
  return 0;
}

void CheckLog(const std::string& log, const std::string& profile,
              bool expect_tensor_cores) {
  // Invariant 7: the binding canary.
  ExpectTrue(CanaryNamesProfile(log, profile),
             "the canary names NANOCHAT_SIM_PROFILE");
  const std::string other = profile == "h100" ? "h200" : "h100";
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

  // Both GEMM entry points were taken by the real gemm.cu.
  ExpectTrue(Contains(log, "\"call\":\"cublasSgemm_v2\""),
             "gemm.cu's single-call path used cublasSgemm_v2");
  ExpectTrue(Contains(log, "\"call\":\"cublasGemmStridedBatchedEx\""),
             "gemm.cu's batched path used cublasGemmStridedBatchedEx");

  // Invariant 8: every *accepted* tensor-core batched GEMM accumulates in
  // fp32. The compute type is the one the real gemm.cu chose. On a
  // non-tensor-core profile there is no such invariant; instead none may be
  // reported as tensor-core.
  bool checked = false;
  bool saw_tensor_core = false;
  for (const std::string& line : Lines(log)) {
    if (line.find("cublasGemmStridedBatchedEx") == std::string::npos) continue;
    if (line.find("\"tensor_cores\":true") == std::string::npos) continue;
    saw_tensor_core = true;
    // An accepted verdict may carry the legacy-enum note appended, so match the
    // `ok` prefix rather than the exact token.
    if (line.find("\"verdict\":\"ok") == std::string::npos) continue;
    checked = true;
    ExpectTrue(line.find("\"compute_type\":68") != std::string::npos,
               "the tensor-core GEMM accumulates with CUBLAS_COMPUTE_32F");
  }
  if (expect_tensor_cores) {
    ExpectTrue(checked, "at least one accepted tensor-core GEMM was recorded");
  } else {
    ExpectTrue(!saw_tensor_core,
               "a non-tensor-core profile records no tensor-core GEMM");
  }

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
