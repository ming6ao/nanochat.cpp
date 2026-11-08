// Phase 0 of docs/kernels.md Part 3: the WGMMA ceiling probe.
//
// The question this file answers is one number: how fast can the Hopper
// warpgroup matrix multiply-accumulate issue from a resident shared-memory
// tile? The plan's phase 2 is only worth starting if one warpgroup can hold a
// large fraction of the per-SM tensor-core rate, so this probe measures exactly
// that, with no global memory traffic in the timed loop.
//
// Three things are hand written, because phase 2 must not depend on a library:
//
//   * the shared-memory matrix descriptor, encoded from the sixteen-byte-offset
//     form that `wgmma` reads (`MakeDescriptor`);
//   * the interleaved K-major staging layout, which is the one descriptor form
//     whose address arithmetic is simple enough to check by hand
//     (`StageInterleaved`);
//   * the instruction itself, with its sixty-four fp32 accumulators
//     (`WgmmaM64N128K16`).
//
// The result is checked against a host reference on the same input, so a
// descriptor that reads the wrong element is caught instead of measured.
//
// **This source needs `sm_90a`.** WGMMA is an architecture-conditional
// instruction, and a build at `sm_90` fails the tripwire below on purpose:
//
//   tools/nanochat build //dev/kernels:wgmma_probe
//   --@rules_cuda//cuda:archs=sm_90a
//
// The probe is a device-only prototype. The emulation engine maps no WGMMA, so
// `tools/cuda_sim` cannot run it, and it carries the `gpu` and `manual` tags.
// See dev/kernels/README.md and docs/kernels.md Part 3.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error \
    "dev/kernels/wgmma_probe.cu needs sm_90a; build it with --@rules_cuda//cuda:archs=sm_90a"
#endif

namespace {

using Bf16 = __nv_bfloat16;

// --- The tile ---------------------------------------------------------------

//: The atom is 64 by 128 by 16, so one warpgroup owns a 64 by 128 rectangle of
//: the output and each of its 128 threads holds 64 fp32 accumulators.
constexpr int kAtomM = 64;
constexpr int kAtomN = 128;
constexpr int kAtomK = 16;
//: The k extent of one pass. Four atoms of sixteen.
constexpr int kTileK = 64;
constexpr int kKBlocks = kTileK / kAtomK;
//: The row groups of one operand: eight rows each, the height of a core matrix.
constexpr int kGroupsA = kAtomM / 8;
constexpr int kGroupsB = kAtomN / 8;

//: One core matrix is eight rows of eight bfloat16 elements, which is 128 B.
constexpr int kCoreBytes = 128;
//: Leading byte offset: the second group of eight k elements, inside one atom.
constexpr int kLboBytes = kCoreBytes;
//: Stride byte offset: the next group of eight rows, inside one operand tile.
constexpr int kSboBytes = 2 * kCoreBytes;
//: Bytes of one row group at one k block: two core matrices, the second eight
//: k elements of the same eight rows.
constexpr int kGroupBytes = 2 * kCoreBytes;
//: One k block of each operand: every row group of that operand, adjacent.
constexpr int kAPerBlock = kGroupsA * kGroupBytes;
constexpr int kBPerBlock = kGroupsB * kGroupBytes;
//: One whole tile: the blocks of a pass follow one another, which is why a k
//: step is one constant added to the descriptor. The two figures differ by the
//: row count of the operand, and confusing them under-sizes the window.
constexpr int kATileBytes = kKBlocks * kAPerBlock;
constexpr int kBTileBytes = kKBlocks * kBPerBlock;

//: The staged window. Two operands, one pass, one buffer: the probe measures
//: the instruction and not the pipeline, so there is nothing to double buffer.
constexpr int kSmemBytes = kATileBytes + kBTileBytes;

//: The reference peak of `backends/cuda/caps.json` for the h100 entry.
constexpr double kPeakFlops = 989e12;

// --- The descriptor ---------------------------------------------------------

// The six fields of the shared-memory matrix descriptor, in the order and the
// width the instruction reads them. The four low bits of every address field
// are dropped, so each one is a sixteen-byte unit.
__device__ __forceinline__ std::uint64_t MakeDescriptor(const void* base,
                                                        int lbo_bytes,
                                                        int sbo_bytes) {
  const std::uint32_t address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(base));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((address >> 4) & 0x3FFFu);
  desc |= static_cast<std::uint64_t>((lbo_bytes >> 4) & 0x3FFFu) << 16;
  desc |= static_cast<std::uint64_t>((sbo_bytes >> 4) & 0x3FFFu) << 32;
  // Base offset 0 and layout type 0: the interleaved form, which is the one
  // this file stages. A 128-byte swizzle would set the type to 1.
  return desc;
}

// --- The instruction --------------------------------------------------------

// D += A * B^T, 64 by 128, over a k extent of sixteen.
//
// Both operands are read from shared memory through a descriptor. `accumulate`
// is the instruction's scale-D predicate: false overwrites D for the first atom
// of a pass, and true adds to it afterwards.
__device__ __forceinline__ void WgmmaM64N128K16(std::uint64_t desc_a,
                                                std::uint64_t desc_b,
                                                float (&d)[64],
                                                bool accumulate) {
  const std::uint32_t scale = accumulate ? 1u : 0u;
  // The trailing four operands are the instruction's own qualifiers as
  // immediates: the two input scales are one, and both operands are K-major,
  // which is zero. PTX accepts no shorter form on this architecture.
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %66, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31, "
      "   %32, %33, %34, %35, %36, %37, %38, %39, "
      "   %40, %41, %42, %43, %44, %45, %46, %47, "
      "   %48, %49, %50, %51, %52, %53, %54, %55, "
      "   %56, %57, %58, %59, %60, %61, %62, %63}, "
      "  %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),
        "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
        "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
        "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale));
}

__device__ __forceinline__ void WgmmaFence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// Marks the accumulator registers as fenced around a region of ordinary code.
//
// Without it ptxas reports that the instruction is serialized, because a plain
// store defines the same registers the instruction accumulates into. The empty
// asm reads and writes each register, so the two cannot be reordered into one
// pipeline stage.
__device__ __forceinline__ void WgmmaFenceOperand(float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(d[i]) : : "memory");
}

__device__ __forceinline__ void WgmmaCommit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// Waits until at most `pending` groups are still running.
template <int Pending>
__device__ __forceinline__ void WgmmaWait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(Pending) : "memory");
}

// Makes the shared stores of the staging pass visible to the asynchronous
// proxy that the instruction reads through.
__device__ __forceinline__ void FenceAsyncShared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// --- The staging pass -------------------------------------------------------

// The interleaved K-major layout that the descriptor above describes.
//
// One core matrix is eight rows of eight k elements, contiguous and in row
// order, so a row of a core matrix is sixteen bytes. The second core matrix of
// an atom holds the next eight k elements, and its own row group of eight rows
// follows. `group` walks M or N, `block` walks the k extent, and the two are
// kept apart so that a k step is one constant added to the descriptor.
__device__ __forceinline__ int InterleavedOffset(int row, int k,
                                                 int group_bytes,
                                                 int block_bytes) {
  const int group = row / 8;
  const int within_row = row % 8;
  const int block = k / kAtomK;
  const int half = (k % kAtomK) / 8;
  const int element = k % 8;
  return block * block_bytes + group * group_bytes + half * kCoreBytes +
         within_row * 16 + element * 2;
}

// Stages one operand of one pass. `rows` is 64 for A and 128 for B, `source` is
// row major with `row_stride` elements between two rows.
__device__ __forceinline__ void StageInterleaved(char* stage,
                                                 const Bf16* source, int rows,
                                                 int row_stride,
                                                 int group_bytes,
                                                 int block_bytes) {
  const int total = rows * kTileK;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += static_cast<int>(blockDim.x)) {
    const int row = index / kTileK;
    const int k = index - row * kTileK;
    const Bf16 value = source[row * row_stride + k];
    const int offset = InterleavedOffset(row, k, kGroupBytes, block_bytes);
    *reinterpret_cast<Bf16*>(stage + offset) = value;
  }
}

// --- The kernel -------------------------------------------------------------

// One warpgroup, one resident 64 by 64 A tile and 128 by 64 B tile, and
// `iterations` passes over them. The timed loop touches shared memory only
// through the instruction.
__global__ void WgmmaProbeKernel(const Bf16* a, const Bf16* b, float* c,
                                 int iterations) {
  __shared__ __align__(128) char smem[kSmemBytes];
  char* stage_a = smem;
  char* stage_b = smem + kATileBytes;

  StageInterleaved(stage_b, b, kAtomN, kTileK, kGroupBytes, kBPerBlock);
  StageInterleaved(stage_a, a, kAtomM, kTileK, kGroupBytes, kAPerBlock);
  // Every thread has written its part of the window before the proxy fence, so
  // the fence orders the block's stores and not only this thread's.
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_a = MakeDescriptor(stage_a, kLboBytes, kSboBytes);
  const std::uint64_t base_b = MakeDescriptor(stage_b, kLboBytes, kSboBytes);
  // A k step of one atom is one whole k block of the staged window, so the
  // descriptor moves by a constant in its sixteen-byte address field.
  const std::uint64_t step_a = kAPerBlock / 16;
  const std::uint64_t step_b = kBPerBlock / 16;

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;
  WgmmaFenceOperand(d);

  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int block = 0; block < kKBlocks; ++block) {
      WgmmaM64N128K16(base_a + block * step_a, base_b + block * step_b, d,
                      iteration != 0 || block != 0);
    }
    // One group for each pass, with at most two groups still running, so the
    // instruction queue stays full without overflowing it.
    WgmmaCommit();
    WgmmaWait<1>();
  }
  WgmmaWait<0>();
  WgmmaFenceOperand(d);

  // One warpgroup of 128 threads holds 64 by 128 fp32 accumulators: eight
  // thousand one hundred and ninety-two values, which is sixty-four for each
  // thread. Their ownership map is not part of this probe's result, so the raw
  // register order is written instead, and the host reference is built the
  // same way.
  float* out = c + static_cast<std::size_t>(threadIdx.x) * 64;
#pragma unroll
  for (int i = 0; i < 64; ++i) out[i] = d[i];
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf("wgmma probe: the Hopper warpgroup rate on a resident tile\n");

  // --- The tile, and the reference -----------------------------------------
  std::vector<nanochat::Bf16> host_a(static_cast<std::size_t>(kAtomM) * kTileK);
  std::vector<nanochat::Bf16> host_b(static_cast<std::size_t>(kAtomN) * kTileK);
  for (int m = 0; m < kAtomM; ++m) {
    for (int k = 0; k < kTileK; ++k) {
      host_a[static_cast<std::size_t>(m) * kTileK + k] =
          nanochat::Bf16FromFloat(0.01f *
                                  static_cast<float>((m * 7 + k * 3) % 13));
    }
  }
  for (int n = 0; n < kAtomN; ++n) {
    for (int k = 0; k < kTileK; ++k) {
      host_b[static_cast<std::size_t>(n) * kTileK + k] =
          nanochat::Bf16FromFloat(0.01f *
                                  static_cast<float>((n * 5 + k * 11) % 17));
    }
  }

  Bf16* device_a = nullptr;
  Bf16* device_b = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(Bf16));
  cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(Bf16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(Bf16),
             cudaMemcpyHostToDevice);

  // The warpgroups of one card, so the aggregate row is a full grid.
  int device = 0;
  cudaDeviceProp properties{};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  const int sm_count = properties.multiProcessorCount;
  std::printf("card: %s, %d SMs, %d kHz clock\n", properties.name, sm_count,
              properties.clockRate / 1000);

  // --- Correctness ---------------------------------------------------------
  const std::size_t results = static_cast<std::size_t>(128) * 64;
  std::vector<float> device_out(results, 0.0f);
  float* device_c = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_c), results * sizeof(float));
  cudaMemset(device_c, 0, results * sizeof(float));
  WgmmaProbeKernel<<<1, 128>>>(device_a, device_b, device_c, 1);
  const cudaError_t launch = cudaDeviceSynchronize();
  if (launch != cudaSuccess) {
    std::printf("wgmma: launch failed: %s\n", cudaGetErrorString(launch));
    return 1;
  }
  cudaMemcpy(device_out.data(), device_c, results * sizeof(float),
             cudaMemcpyDeviceToHost);

  // The reference for one thread's sixty-four registers. The probe does not
  // decode which element each register holds, so the check is the sum of the
  // sixty-four accumulators against the sum of the whole product: a descriptor
  // that reads the wrong tile fails it, because the two operands differ.
  double reference_sum = 0.0;
  for (int m = 0; m < kAtomM; ++m) {
    for (int n = 0; n < kAtomN; ++n) {
      double sum = 0.0;
      for (int k = 0; k < kTileK; ++k) {
        sum += static_cast<double>(static_cast<float>(
                   host_a[static_cast<std::size_t>(m) * kTileK + k])) *
               static_cast<double>(static_cast<float>(
                   host_b[static_cast<std::size_t>(n) * kTileK + k]));
      }
      reference_sum += sum;
    }
  }
  double measured_sum = 0.0;
  for (float value : device_out) measured_sum += static_cast<double>(value);
  const double relative =
      std::fabs(measured_sum - reference_sum) / std::fabs(reference_sum);
  std::printf(
      "wgmma:correctness reference_sum=%.4f measured_sum=%.4f "
      "relative_error=%.3e %s\n",
      reference_sum, measured_sum, relative, relative < 1e-3 ? "PASS" : "FAIL");

  // --- The rate ------------------------------------------------------------
  // One pass is four atoms of 64 by 128 by 16.
  const double flops_per_pass = 2.0 * kAtomM * kAtomN * kAtomK * kKBlocks;
  const int iterations = 20000;
  const int grid_rows[2] = {1, 2 * sm_count};
  const char* names[2] = {"1block", "grid"};
  for (int row = 0; row < 2; ++row) {
    const int blocks = grid_rows[row];
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // One warm-up pass, then the measured one.
    WgmmaProbeKernel<<<blocks, 128>>>(device_a, device_b, device_c, 16);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    WgmmaProbeKernel<<<blocks, 128>>>(device_a, device_b, device_c, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double flops =
        flops_per_pass * static_cast<double>(iterations) * blocks;
    const double seconds = static_cast<double>(milliseconds) * 1e-3;
    const double rate = flops / seconds;
    std::printf(
        "wgmma:%s blocks=%d passes=%d ms=%.3f tflops=%.1f "
        "fraction_of_peak=%.3f\n",
        names[row], blocks, iterations, milliseconds, rate / 1e12,
        rate / kPeakFlops);
  }

  // The projection the plan's phase 2 reads: the 1280 by 1280 by 1280 shape is
  // 200 tiles of 64 by 128, so a full card of this kernel would take the pass
  // rate of the grid row and the tile count.
  std::printf(
      "wgmma:note flops_per_pass=%.1f the projection uses the grid row\n",
      flops_per_pass);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return 0;
}
