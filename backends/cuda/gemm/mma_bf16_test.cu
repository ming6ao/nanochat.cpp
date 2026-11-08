// The gate of the bf16 micro-tile header (backends/cuda/gemm/mma_bf16.cuh),
// phase 1 of docs/kernels.md Part 3.
//
// Two rows, one reference:
//
//   * `mma_bf16:host` runs the header's host arm, which is the arm the
//   emulation
//     engine compiles. It needs no card, so it is the fast gate and it runs on
//     the smallest machine.
//   * `mma_bf16:device` runs the header's device arm, which is `ldmatrix` and
//     `mma.sync.aligned.m16n8k16`, in one warp on the card.
//
// Both are compared against the same scalar triple loop over the same input, so
// the two arms answer to one definition and the lane map is checked and not
// assumed. Every one of the four operand orientations is covered, because the
// orientations are what the staged layouts of `gemm_tile.cuh` produce.
//
// Run it through the entry point:
//
//   tools/nanochat run t1-gpu -- ./bazel-bin/backends/cuda/gemm/mma_bf16_test
//
// See docs/kernels.md Part 3 and dev/kernels/wgmma_probe_report.md.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "backends/cuda/gemm/mma_bf16.cuh"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::Bf16;
using nanochat::kernels::gemm_bf16::FragmentA;
using nanochat::kernels::gemm_bf16::FragmentB;
using nanochat::kernels::gemm_bf16::FragmentC;
using nanochat::kernels::gemm_bf16::kMmaK;
using nanochat::kernels::gemm_bf16::kMmaM;
using nanochat::kernels::gemm_bf16::kMmaN;

//: The staged lead of both operands. A pad of eight elements is what the
//: shipped tile uses, so the gate runs on a lead that is not a power of two.
constexpr int kLead = kMmaK + 8;

//: The rows of each staged operand. A is m by k, and B is 16 by 8 staged either
//: as 8 rows of 16 or as 16 rows of 8, so one region serves both orientations.
constexpr int kStagedRows = kMmaK;

// The values. A pattern, so a misplaced element changes the answer.
float AValue(int row, int k) {
  return 0.01f * static_cast<float>((row * 7 + k * 3) % 13) - 0.05f;
}
float BValue(int k, int column) {
  return 0.02f * static_cast<float>((column * 5 + k * 11) % 17) - 0.1f;
}

// The one definition both arms answer to: C = A * B over the micro-tile.
//
// The logical matrices do not depend on the orientation. The orientation is a
// property of the staged memory alone, so this function takes no flag and both
// arms are compared against one result.
void Reference(float* c) {
  for (int row = 0; row < kMmaM; ++row) {
    for (int column = 0; column < kMmaN; ++column) {
      double sum = 0.0;
      for (int k = 0; k < kMmaK; ++k) {
        sum += static_cast<double>(AValue(row, k)) *
               static_cast<double>(BValue(k, column));
      }
      c[row * kMmaN + column] = static_cast<float>(sum);
    }
  }
}

// Stages A in the orientation the flag names, with the pad the tile uses.
void StageA(std::vector<Bf16>* storage, bool transpose_a) {
  const int rows = transpose_a ? kMmaK : kMmaM;
  const int columns = transpose_a ? kMmaM : kMmaK;
  storage->assign(static_cast<std::size_t>(kStagedRows) * kLead, Bf16(0.0f));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      const float value = transpose_a ? AValue(c, r) : AValue(r, c);
      (*storage)[static_cast<std::size_t>(r) * kLead + c] = Bf16(value);
    }
  }
}

// Stages B in the orientation the flag names.
void StageB(std::vector<Bf16>* storage, bool transpose_b) {
  const int rows = transpose_b ? kMmaN : kMmaK;
  const int columns = transpose_b ? kMmaK : kMmaN;
  storage->assign(static_cast<std::size_t>(kStagedRows) * kLead, Bf16(0.0f));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      // A plain B stages k by n, so a staged row is a k index and a staged
      // column is an n index. A transposed B stages n by k, so they swap.
      const float value = transpose_b ? BValue(c, r) : BValue(r, c);
      (*storage)[static_cast<std::size_t>(r) * kLead + c] = Bf16(value);
    }
  }
}

// --- The host arm -----------------------------------------------------------

float CheckHost(bool transpose_a, bool transpose_b) {
  std::vector<Bf16> a;
  std::vector<Bf16> b;
  StageA(&a, transpose_a);
  StageB(&b, transpose_b);
  // The header names the vendor bf16 type, and the project type is the same two
  // bytes, so the view is a cast. The shipped kernel makes the same one.
  const __nv_bfloat16* staged_a =
      reinterpret_cast<const __nv_bfloat16*>(a.data());
  const __nv_bfloat16* staged_b =
      reinterpret_cast<const __nv_bfloat16*>(b.data());
  FragmentA fa;
  FragmentB fb;
  FragmentC fc;
  nanochat::kernels::gemm_bf16::Fill(fc, 0.0f);
  if (transpose_a) {
    nanochat::kernels::gemm_bf16::LoadA<true>(fa, staged_a, kLead);
  } else {
    nanochat::kernels::gemm_bf16::LoadA<false>(fa, staged_a, kLead);
  }
  if (transpose_b) {
    nanochat::kernels::gemm_bf16::LoadB<true>(fb, staged_b, kLead);
  } else {
    nanochat::kernels::gemm_bf16::LoadB<false>(fb, staged_b, kLead);
  }
  nanochat::kernels::gemm_bf16::Mma(fc, fa, fb);
  std::vector<float> got(static_cast<std::size_t>(kMmaM) * kMmaN, 0.0f);
  nanochat::kernels::gemm_bf16::StoreFp32(got.data(), kMmaN, fc);

  std::vector<float> want(static_cast<std::size_t>(kMmaM) * kMmaN, 0.0f);
  Reference(want.data());
  float worst = 0.0f;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::fmax(worst, std::fabs(got[i] - want[i]));
  }
  return worst;
}

// --- The device arm ---------------------------------------------------------

// One warp, one micro-tile, both operands staged in shared memory.
__global__ void MmaBf16Kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                              float* c, int lead, bool transpose_a,
                              bool transpose_b) {
  __shared__ __align__(16) __nv_bfloat16 stage_a[kStagedRows * kLead];
  __shared__ __align__(16) __nv_bfloat16 stage_b[kStagedRows * kLead];
  for (int i = static_cast<int>(threadIdx.x); i < kStagedRows * kLead;
       i += static_cast<int>(blockDim.x)) {
    stage_a[i] = a[i];
    stage_b[i] = b[i];
  }
  __syncthreads();

  FragmentA fa;
  FragmentB fb;
  FragmentC fc;
  nanochat::kernels::gemm_bf16::Fill(fc, 0.0f);
  if (transpose_a) {
    nanochat::kernels::gemm_bf16::LoadA<true>(fa, stage_a, kLead);
  } else {
    nanochat::kernels::gemm_bf16::LoadA<false>(fa, stage_a, kLead);
  }
  if (transpose_b) {
    nanochat::kernels::gemm_bf16::LoadB<true>(fb, stage_b, kLead);
  } else {
    nanochat::kernels::gemm_bf16::LoadB<false>(fb, stage_b, kLead);
  }
  nanochat::kernels::gemm_bf16::Mma(fc, fa, fb);
  __syncthreads();
  nanochat::kernels::gemm_bf16::StoreFp32(c, lead, fc);
}

float CheckDevice(bool transpose_a, bool transpose_b) {
  std::vector<Bf16> host_a;
  std::vector<Bf16> host_b;
  StageA(&host_a, transpose_a);
  StageB(&host_b, transpose_b);
  const std::size_t staged = static_cast<std::size_t>(kStagedRows) * kLead;
  __nv_bfloat16* device_a = nullptr;
  __nv_bfloat16* device_b = nullptr;
  float* device_c = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_a), staged * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_b), staged * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_c),
             static_cast<std::size_t>(kMmaM) * kMmaN * sizeof(float));
  cudaMemcpy(device_a, host_a.data(), staged * sizeof(Bf16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), staged * sizeof(Bf16),
             cudaMemcpyHostToDevice);
  cudaMemset(device_c, 0,
             static_cast<std::size_t>(kMmaM) * kMmaN * sizeof(float));

  MmaBf16Kernel<<<1, 32>>>(device_a, device_b, device_c, kMmaN, transpose_a,
                           transpose_b);
  const cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::printf("mma_bf16: device launch failed: %s\n",
                cudaGetErrorString(status));
    return -1.0f;
  }

  std::vector<float> got(static_cast<std::size_t>(kMmaM) * kMmaN, 0.0f);
  cudaMemcpy(got.data(), device_c, got.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  std::vector<float> want(got.size(), 0.0f);
  Reference(want.data());
  float worst = 0.0f;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::fmax(worst, std::fabs(got[i] - want[i]));
  }
  return worst;
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf(
      "mma_bf16 micro-tile gate: ldmatrix and mma.sync against one "
      "reference\n");

  //: bf16 carries eight mantissa bits, so one product is exact only to about
  //: three decimal digits. The reference accumulates in double and the arms in
  //: fp32, so the bound is loose on purpose and still catches a moved element.
  constexpr float kTolerance = 2e-4f;

  const bool orientations[4][2] = {
      {false, false}, {false, true}, {true, false}, {true, true}};

  int failures = 0;
  for (int i = 0; i < 4; ++i) {
    const float worst = CheckHost(orientations[i][0], orientations[i][1]);
    const bool pass = worst <= kTolerance;
    std::printf(
        "mma_bf16:host transpose_a=%d transpose_b=%d max_abs_error=%.3e %s\n",
        orientations[i][0] ? 1 : 0, orientations[i][1] ? 1 : 0,
        static_cast<double>(worst), pass ? "PASS" : "FAIL");
    if (!pass) ++failures;
  }

  // The device row needs a card at `sm_80` or newer. This host has neither, so
  // a failure to reach a device is reported and not counted as a kernel defect.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  cudaDeviceProp properties{};
  if (device_count > 0) cudaGetDeviceProperties(&properties, 0);
  if (device_count == 0 || properties.major < 8) {
    std::printf("mma_bf16:device skipped: no card at sm_80 or newer\n");
  } else {
    std::printf("mma_bf16:device card=%s compute=%d.%d\n", properties.name,
                properties.major, properties.minor);
    for (int i = 0; i < 4; ++i) {
      const float worst = CheckDevice(orientations[i][0], orientations[i][1]);
      const bool pass = worst >= 0.0f && worst <= kTolerance;
      std::printf(
          "mma_bf16:device transpose_a=%d transpose_b=%d max_abs_error=%.3e "
          "%s\n",
          orientations[i][0] ? 1 : 0, orientations[i][1] ? 1 : 0,
          static_cast<double>(worst), pass ? "PASS" : "FAIL");
      if (!pass) ++failures;
    }
  }

  std::printf("mma_bf16:result failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}
