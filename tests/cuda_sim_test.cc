// tests/cuda_sim_test.cc -- the API interposer's semantic invariants
// (docs/simulator.md sections 5.3 and 12.2), as an in-process *mock unit test*.
//
// The test links `//tools/cuda_sim:cuda_sim_interposer_impl` and drives the
// mock's own entry points, so every check is hermetic: no GPU, no driver, no
// toolkit, and no committed log to compare against. It asserts the named
// invariants, never byte equality:
//
//   7. The binding canary is the first record and it names the simulated
//      profile; a canary for any other profile is not accepted.
//   8. The log holds at least one `cudaMalloc`, one kernel launch, and one
//      cuBLAS call; every *accepted* `cublasGemmStridedBatchedEx` under a
//      tensor-core profile has `compute_type == CUBLAS_COMPUTE_32F`; no launch
//      above 48 KB lacks a preceding `cudaFuncSetAttribute`; and no pointer
//      value appears anywhere in the log.
//   9. A synthetic 64 KB launch with no attribute is rejected, and the same
//      launch after the attribute is accepted.
//
// Plus the two load-bearing handle contracts: `cublasCreate_v2` writes a
// non-null handle, and `__cudaRegisterFatBinary` returns a non-null token.
//
// This is not a binding test: nothing here links the CUDA backend or exercises
// `LD_PRELOAD`. That path is `//tests:cuda_sim_preload_test`, run by
// `tools/nanochat simulate --suite api`, and it is what derives the tensor-core
// compute-type invariant from the real `gemm.cu` decision.

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/sandbox.h"
#include "tools/cuda_sim/cuda_sim_abi.h"

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

// A stable address the interposer can key its attribute table on.
void DummyKernel() {}

// The canary is a per-run binding proof: the first record must name the profile
// the run simulates. A canary for any other profile is rejected.
bool CanaryNamesProfile(const std::string& log, const std::string& profile) {
  const std::vector<std::string> lines = Lines(log);
  if (lines.empty()) return false;
  if (lines.front().find("\"kind\":\"canary\"") == std::string::npos) {
    return false;
  }
  return lines.front().find("\"profile\":\"" + profile + "\"") !=
         std::string::npos;
}

void TestCanary() {
  const std::vector<std::string> lines = Lines(nanochat_sim_log());
  ExpectTrue(!lines.empty(), "the log holds a record");
  if (lines.empty()) return;
  ExpectTrue(lines.front().find("\"kind\":\"canary\"") != std::string::npos,
             "the first record is the binding canary");
  ExpectTrue(CanaryNamesProfile(nanochat_sim_log(), "h100"),
             "the canary names NANOCHAT_SIM_PROFILE");
  // The negative case: a canary for a different profile must not be accepted.
  ExpectTrue(!CanaryNamesProfile(nanochat_sim_log(), "h200"),
             "a canary for another profile is not accepted");
  ExpectTrue(std::string(nanochat_sim_profile()) == "h100",
             "the interposer reports the profile");
}

void TestFamilyCoverage() {
  const std::string log = nanochat_sim_log();
  ExpectTrue(log.find("\"call\":\"cudaMalloc\"") != std::string::npos,
             "the log holds a cudaMalloc");
  ExpectTrue(log.find("\"call\":\"cudaLaunchKernel\"") != std::string::npos,
             "the log holds a kernel launch");
  ExpectTrue(
      log.find("\"call\":\"cublasGemmStridedBatchedEx\"") != std::string::npos,
      "the log holds a cuBLAS call");
}

void TestNoPointerValues() {
  const std::string log = nanochat_sim_log();
  // A pointer value would print as a `0x`-prefixed hexadecimal run. The log
  // must stay comparable across runs, so no address may appear.
  ExpectTrue(log.find("0x") == std::string::npos,
             "no pointer value appears in the log");
}

void TestTensorCoreComputeType() {
  const std::vector<std::string> lines = Lines(nanochat_sim_log());
  bool checked = false;
  for (const std::string& line : lines) {
    if (line.find("cublasGemmStridedBatchedEx") == std::string::npos) continue;
    if (line.find("\"tensor_cores\":true") == std::string::npos) continue;
    // Only accepted calls carry the invariant; a rejected call is a negative
    // test exercising the rule. An accepted verdict may carry the legacy-enum
    // note appended, so match the `ok` prefix rather than the exact token.
    if (line.find("\"verdict\":\"ok") == std::string::npos) continue;
    checked = true;
    ExpectTrue(line.find("\"compute_type\":68") != std::string::npos,
               "a tensor-core GEMM accumulates with CUBLAS_COMPUTE_32F");
  }
  ExpectTrue(checked, "at least one tensor-core GEMM was recorded");
}

// Invariant 8: no launch above the 48 KB default is accepted unless a
// cudaFuncSetAttribute for that function came first.
void TestSharedMemoryRule() {
  const std::vector<std::string> lines = Lines(nanochat_sim_log());
  bool attribute_seen = false;
  int rejected = 0;
  int accepted = 0;
  for (const std::string& line : lines) {
    if (line.find("\"call\":\"cudaFuncSetAttribute\"") != std::string::npos) {
      attribute_seen = true;
    }
    if (line.find("\"call\":\"cudaLaunchKernel\"") == std::string::npos) {
      continue;
    }
    if (line.find("\"shared_bytes\":65536") == std::string::npos) continue;
    if (line.find("\"verdict\":\"ok\"") != std::string::npos) {
      ++accepted;
      ExpectTrue(attribute_seen,
                 "an above-48 KB launch follows cudaFuncSetAttribute");
    } else {
      ++rejected;
    }
  }
  ExpectTrue(rejected >= 1,
             "the 64 KB launch without the attribute is rejected");
  ExpectTrue(accepted >= 1, "the 64 KB launch after the attribute is accepted");
}

void TestHandleContracts() {
  ExpectTrue(std::string(cublasGetStatusString(kCublasStatusSuccess)) ==
                 "CUBLAS_STATUS_SUCCESS",
             "cublasGetStatusString resolves");

  // The fat-binary token is non-null, which is what the registration hooks
  // pass back.
  int marker = 0;
  void** token = __cudaRegisterFatBinary(&marker);
  ExpectTrue(token != nullptr, "__cudaRegisterFatBinary returns a token");
  ExpectTrue(*token != nullptr, "the token is non-null");
  __cudaRegisterFunction(token, reinterpret_cast<const void*>(&DummyKernel),
                         nullptr, "dummy_kernel", 0, nullptr, nullptr, nullptr,
                         nullptr, nullptr);
  __cudaRegisterFatBinaryEnd(token);
}

// The synthetic case of section 12.2 (invariant 9): a launch above the 48 KB
// default must fail until the matching attribute is set.
void TestSharedMemoryOrdering() {
  const void* func = reinterpret_cast<const void*>(&DummyKernel);
  const dim3 grid(1);
  const dim3 block(64);
  const cudaError_t before =
      cudaLaunchKernel(func, grid, block, nullptr, 65536, nullptr);
  ExpectTrue(before != kCudaSuccess,
             "a 64 KB launch without the attribute is rejected");
  ExpectTrue(
      cudaFuncSetAttribute(func, kCudaFuncAttributeMaxDynamicSharedMemorySize,
                           65536) == kCudaSuccess,
      "the attribute is accepted within the profile limit");
  const cudaError_t after =
      cudaLaunchKernel(func, grid, block, nullptr, 65536, nullptr);
  ExpectTrue(after == kCudaSuccess,
             "the same launch is accepted once the attribute is set");

  // The profile's own limit still bounds the attribute.
  ExpectTrue(
      cudaFuncSetAttribute(func, kCudaFuncAttributeMaxDynamicSharedMemorySize,
                           1 << 30) != kCudaSuccess,
      "an attribute above the profile limit is rejected");
}

void TestBlockLimit() {
  const void* func = reinterpret_cast<const void*>(&DummyKernel);
  const dim3 grid(1);
  // h100 allows 1024 threads per block; 2048 must be refused.
  ExpectTrue(cudaLaunchKernel(func, grid, dim3(2048), nullptr, 0, nullptr) !=
                 kCudaSuccess,
             "a block above the profile limit is rejected");
  ExpectTrue(cudaLaunchKernel(func, grid, dim3(0), nullptr, 0, nullptr) !=
                 kCudaSuccess,
             "a zero-sized block is rejected");
}

void TestAllocationBounds() {
  // Freeing an untracked pointer is an error, not a crash.
  int on_the_stack = 0;
  ExpectTrue(cudaFree(&on_the_stack) != kCudaSuccess,
             "freeing an untracked pointer is rejected");

  // A memset outside a live allocation is rejected, not silently applied. The
  // stack range is not tracked.
  int host_buffer[8] = {};
  ExpectTrue(cudaMemset(host_buffer, 0, sizeof(host_buffer)) != kCudaSuccess,
             "a memset outside a live allocation is rejected");
}

void TestHandleOrder() {
  // A cuBLAS call before any create is rejected (the mock only accepts a live
  // handle), and a destroyed handle is not reused: `gemm.cu` creates the handle
  // lazily, so the null-handle case is its first call.
  float alpha = 1.0f;
  float beta = 0.0f;
  float a = 0.0f;
  float b = 0.0f;
  float c = 0.0f;
  const int before = nanochat_sim_record_count();
  ExpectTrue(
      cublasGemmStridedBatchedEx(
          nullptr, kCublasOpN, kCublasOpN, 1, 1, 1, &alpha, &a, kCudaR32F, 4, 0,
          &b, kCudaR32F, 4, 0, &beta, &c, kCudaR32F, 4, 0, 1, kCublasCompute32F,
          kCublasGemmDefaultTensorOp) != kCublasStatusSuccess,
      "a cuBLAS call before cublasCreate_v2 is rejected");
  ExpectTrue(nanochat_sim_record_count() > before, "the rejection is recorded");

  cublasHandle_t handle = nullptr;
  ExpectTrue(cublasCreate_v2(&handle) == kCublasStatusSuccess,
             "cublasCreate_v2 succeeds");
  ExpectTrue(handle != nullptr, "cublasCreate_v2 writes a non-null handle");
  ExpectTrue(cublasDestroy(handle) == kCublasStatusSuccess,
             "cublasDestroy succeeds");
  ExpectTrue(
      cublasGemmStridedBatchedEx(
          handle, kCublasOpN, kCublasOpN, 1, 1, 1, &alpha, &a, kCudaR32F, 4, 0,
          &b, kCudaR32F, 4, 0, &beta, &c, kCudaR32F, 4, 0, 1, kCublasCompute32F,
          kCublasGemmDefaultTensorOp) != kCublasStatusSuccess,
      "a destroyed handle is not reused");
}

// The tensor-core compute-type rule (docs/simulator.md section 5.3): a GEMM on
// a tensor-core profile must accumulate in fp32, so `CUBLAS_COMPUTE_16F` -- the
// Pascal non-tensor-core path -- is rejected.
void TestTensorCoreComputeTypeRule() {
  cublasHandle_t handle = nullptr;
  ExpectTrue(cublasCreate_v2(&handle) == kCublasStatusSuccess,
             "cublasCreate_v2 succeeds");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  float a = 0.0f;
  float b = 0.0f;
  float c = 0.0f;
  const int before = nanochat_sim_record_count();
  ExpectTrue(cublasGemmStridedBatchedEx(
                 handle, kCublasOpN, kCublasOpN, 8, 8, 8, &alpha, &a, kCudaR32F,
                 8, 64, &b, kCudaR32F, 8, 64, &beta, &c, kCudaR32F, 8, 64, 2,
                 kCublasCompute16F, kCublasGemmDefault) != kCublasStatusSuccess,
             "CUBLAS_COMPUTE_16F is rejected on a tensor-core profile");
  ExpectTrue(nanochat_sim_record_count() > before, "the rejection is recorded");
  ExpectTrue(cublasDestroy(handle) == kCublasStatusSuccess,
             "cublasDestroy succeeds");
}

// A representative sequence: an allocation, a launch, and a GEMM under the
// simulated profile, all of which the invariant checks above then read.
void DriveTheBackend() {
  void* device_buffer = nullptr;
  ExpectTrue(cudaMalloc(&device_buffer, 4096) == kCudaSuccess,
             "cudaMalloc succeeds");
  ExpectTrue(device_buffer != nullptr, "cudaMalloc returns memory");
  const std::vector<float> host(16, 1.5f);
  ExpectTrue(cudaMemcpy(device_buffer, host.data(), 16 * sizeof(float),
                        kCudaMemcpyHostToDevice) == kCudaSuccess,
             "the host-to-device copy succeeds");
  ExpectTrue(cudaMemset(device_buffer, 0, 16 * sizeof(float)) == kCudaSuccess,
             "cudaMemset succeeds");

  cudaDevicePropPrefix prop;
  ExpectTrue(cudaGetDeviceProperties_v2(&prop, 0) == kCudaSuccess,
             "cudaGetDeviceProperties_v2 succeeds");
  ExpectTrue(std::string(prop.name) == "h100",
             "the fabricated property names the profile");
  ExpectTrue(prop.major == 9 && prop.minor == 0,
             "the fabricated property reports compute capability 9.0");
  int device = -1;
  ExpectTrue(cudaGetDevice(&device) == kCudaSuccess && device == 0,
             "cudaGetDevice reports device 0");

  const void* func = reinterpret_cast<const void*>(&DummyKernel);
  ExpectTrue(cudaLaunchKernel(func, dim3(4), dim3(64), nullptr, 0, nullptr) ==
                 kCudaSuccess,
             "an in-limits launch succeeds");

  cublasHandle_t handle = nullptr;
  ExpectTrue(cublasCreate_v2(&handle) == kCublasStatusSuccess,
             "cublasCreate_v2 succeeds");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  ExpectTrue(cublasGemmStridedBatchedEx(
                 handle, kCublasOpN, kCublasOpN, 8, 8, 8, &alpha, device_buffer,
                 kCudaR32F, 8, 64, device_buffer, kCudaR32F, 8, 64, &beta,
                 device_buffer, kCudaR32F, 8, 64, 2, kCublasCompute32F,
                 kCublasGemmDefaultTensorOp) == kCublasStatusSuccess,
             "the tensor-core GEMM is accepted");
  ExpectTrue(cublasDestroy(handle) == kCublasStatusSuccess,
             "cublasDestroy succeeds");

  ExpectTrue(cudaFree(device_buffer) == kCudaSuccess, "cudaFree succeeds");
  ExpectTrue(cudaDeviceSynchronize() == kCudaSuccess,
             "cudaDeviceSynchronize succeeds");
  ExpectTrue(cudaGetLastError() == kCudaSuccess, "cudaGetLastError is clear");
}

}  // namespace

// A run that simulates a different profile must produce a canary naming that
// profile. This exercises the negative canary case: the log-scan checks accept
// only the profile named by `NANOCHAT_SIM_PROFILE`. It resets the log, so it
// runs last.
void TestCanaryNegative() {
  setenv("NANOCHAT_SIM_PROFILE", "h200", 1);
  nanochat_sim_reset();
  void* pointer = nullptr;
  ExpectTrue(cudaMalloc(&pointer, 64) == kCudaSuccess,
             "cudaMalloc under the h200 profile");
  ExpectTrue(CanaryNamesProfile(nanochat_sim_log(), "h200"),
             "the canary names the profile in effect for this run");
  ExpectTrue(!CanaryNamesProfile(nanochat_sim_log(), "h100"),
             "a canary for h100 is not accepted under h200");
  cudaFree(pointer);
  setenv("NANOCHAT_SIM_PROFILE", "h100", 1);
  nanochat_sim_reset();
}

int main() {
  nanochat::RequireSandboxOrDie("test");
  setenv("NANOCHAT_SIM_PROFILE", "h100", 1);
  nanochat_sim_reset();

  TestHandleContracts();
  DriveTheBackend();
  TestSharedMemoryOrdering();
  TestBlockLimit();
  TestAllocationBounds();
  TestHandleOrder();
  TestTensorCoreComputeTypeRule();

  TestCanary();
  TestFamilyCoverage();
  TestNoPointerValues();
  TestTensorCoreComputeType();
  TestSharedMemoryRule();
  TestCanaryNegative();

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("cuda sim interposer: all checks passed (%d records)\n",
              nanochat_sim_record_count());
  return 0;
}
