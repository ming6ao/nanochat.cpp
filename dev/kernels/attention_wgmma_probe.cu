// The two facts the Hopper attention tile needs before it can be written: the
// accumulator map of the warpgroup instruction, and the rate it reaches when
// its operands are staged in the form it actually reads.
//
// **The form.** The instruction reads each operand as a run of core matrices,
// and a core matrix is eight rows of eight elements with the rows sixteen bytes
// apart. A row-major tile is not readable: the instruction then reads one
// element out of every eight along each axis. An earlier version of this probe
// staged row-major tiles and did not notice for three runs, because its check
// compared the **sum** of a product, and that quantity is invariant under the
// row and column permutation a wrong form induces. A probe of one element at a
// time shows the difference at once, and that probe is kept here.
//
// **The map.** The accumulator of `wgmma.mma_async.m64n128k16` is 64 by 128
// fp32 values over 128 threads, so each thread holds 64. This file measures
// which (row, column) each of those 64 registers holds, in two ways, and prints
// both: a one-hot query row against a one-hot key row says where one element
// lands, and two separable patterns decode the whole table so that a reader can
// check the map against the counts.
//
// **The rate.** With the staged form settled, the last row times the score
// product from a resident tile, which is the ceiling of the instruction on this
// card.
//
// The target states `archs = ["sm_90a"]` for itself, so no command line carries
// a flag, and the tripwire below refuses a build that names another
// architecture. The emulation engine maps no warpgroup instruction, so the
// probe is device only: it carries the `gpu` and `manual` tags and its gate is
// the H100 leg.
//
//   modal run tools/modal/app.py::card_job --name attention_wgmma

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error \
    "dev/kernels/attention_wgmma_probe.cu needs sm_90a; its target states archs = [\"sm_90a\"]"
#endif

namespace {

using Bf16 = __nv_bfloat16;

// --- The tile ---------------------------------------------------------------

//: Threads of one warpgroup, which is the width the instruction spans.
constexpr int kWarpgroupThreads = 128;

//: Query rows, key rows, and the head width: the `m`, `n`, and `k` extents of
//: one score product.
constexpr int kAtomM = 64;
constexpr int kAtomN = 128;
constexpr int kDim = 128;
//: The `k` extent of one atom, the atoms of the whole tile, and the registers
//: one thread holds.
constexpr int kAtomK = 16;
constexpr int kAtoms = kDim / kAtomK;
constexpr int kRegisters = kAtomM * kAtomN / kWarpgroupThreads;

//: The interleaved core-matrix form. One core matrix is eight rows of eight
//: elements, sixteen bytes to a row, so the two halves of the `k` axis of one
//: row group are one leading offset apart and the next row group follows at one
//: stride offset.
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 2 * kLboBytes;

constexpr int kQueryBytes = kAtomM * kDim * sizeof(Bf16);
constexpr int kKeyBytes = kAtomN * kDim * sizeof(Bf16);
constexpr int kSmemBytes = kQueryBytes + kKeyBytes;

//: One `k` step of a descriptor, in sixteen-byte units, for a tile of `rows`.
//: The two operands differ because their tiles have different heights.
__host__ __device__ constexpr int AtomStep(int rows) {
  return (rows / 8) * kSboBytes / 16;
}
static_assert(kAtoms * AtomStep(kAtomM) * 16 == kQueryBytes,
              "the query tile must fill its interleaved form");
static_assert(kAtoms * AtomStep(kAtomN) * 16 == kKeyBytes,
              "the key tile must fill its interleaved form");

//: The reference peak of `backends/cuda/caps.json` for the h100 entry.
constexpr double kPeakFlops = 989e12;

// --- The staging ------------------------------------------------------------

// The interleaved offset of one element of a row-major source tile.
__device__ __forceinline__ int InterleavedOffset(int row, int k, int rows) {
  const int block = k / kAtomK;
  const int half = (k % kAtomK) / 8;
  const int element = k % 8;
  const int group = row / 8;
  const int within_row = row % 8;
  return block * AtomStep(rows) * 16 + group * kSboBytes + half * kLboBytes +
         within_row * 16 + element * 2;
}

// Stages one operand of the whole tile. The source is row major.
__device__ __forceinline__ void StageInterleaved(char* stage,
                                                 const Bf16* source, int rows) {
  const int total = rows * kDim;
  for (int index = static_cast<int>(threadIdx.x); index < total;
       index += kWarpgroupThreads) {
    const int row = index / kDim;
    const int k = index - row * kDim;
    *reinterpret_cast<Bf16*>(stage + InterleavedOffset(row, k, rows)) =
        source[index];
  }
}

// The six fields of the shared-memory matrix descriptor: the address in sixteen
// byte units, the leading offset, the stride offset, base offset zero, and
// layout type zero for the interleaved form.
__device__ __forceinline__ std::uint64_t MakeDescriptor(const void* base) {
  const std::uint32_t address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(base));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((address >> 4) & 0x3FFFu);
  desc |= static_cast<std::uint64_t>((kLboBytes >> 4) & 0x3FFFu) << 16;
  desc |= static_cast<std::uint64_t>((kSboBytes >> 4) & 0x3FFFu) << 32;
  return desc;
}

// --- The instruction --------------------------------------------------------

// D += A * B^T over one 64 by 128 by 16 atom. Both operands come from shared
// memory, and the scale-D predicate is false for the first atom of a pass.
__device__ __forceinline__ void WgmmaM64N128K16(std::uint64_t desc_a,
                                                std::uint64_t desc_b,
                                                float (&d)[kRegisters],
                                                bool accumulate) {
  const std::uint32_t scale = accumulate ? 1u : 0u;
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

__device__ __forceinline__ void WgmmaFenceOperand(float (&d)[kRegisters]) {
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) {
    asm volatile("" : "+f"(d[i]) : : "memory");
  }
}

__device__ __forceinline__ void WgmmaCommit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void WgmmaWait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(Pending) : "memory");
}

__device__ __forceinline__ void FenceAsyncShared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// --- The kernel -------------------------------------------------------------

// One warpgroup, one resident query tile and one resident key tile in the
// interleaved form, and `passes` score products over them. The registers leave
// raw, and every reader below decodes them.
__global__ void ProbeKernel(const Bf16* __restrict__ q,
                            const Bf16* __restrict__ k, float* __restrict__ out,
                            int passes) {
  extern __shared__ __align__(1024) char smem[];
  char* query = smem;
  char* key = smem + kQueryBytes;
  StageInterleaved(query, q, kAtomM);
  StageInterleaved(key, k, kAtomN);
  __syncthreads();
  FenceAsyncShared();
  WgmmaFence();

  const std::uint64_t base_query = MakeDescriptor(query);
  const std::uint64_t base_key = MakeDescriptor(key);
  float d[kRegisters];
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) d[i] = 0.0f;
  WgmmaFenceOperand(d);

  for (int pass = 0; pass < passes; ++pass) {
#pragma unroll
    for (int atom = 0; atom < kAtoms; ++atom) {
      WgmmaM64N128K16(base_query + atom * AtomStep(kAtomM),
                      base_key + atom * AtomStep(kAtomN), d,
                      pass != 0 || atom != 0);
    }
    WgmmaCommit();
    WgmmaWait<1>();
  }
  WgmmaWait<0>();
  WgmmaFenceOperand(d);

  float* destination = out + static_cast<std::size_t>(threadIdx.x) * kRegisters;
#pragma unroll
  for (int i = 0; i < kRegisters; ++i) destination[i] = d[i];
}

// The result of one pass, for the decoders below.
std::vector<float> RunOnce(const Bf16* device_q, const Bf16* device_k,
                           float* device_out) {
  cudaMemset(
      device_out, 0,
      static_cast<std::size_t>(kWarpgroupThreads) * kRegisters * sizeof(float));
  ProbeKernel<<<1, kWarpgroupThreads, kSmemBytes>>>(device_q, device_k,
                                                    device_out, 1);
  cudaDeviceSynchronize();
  std::vector<float> host(
      static_cast<std::size_t>(kWarpgroupThreads) * kRegisters, 0.0f);
  cudaMemcpy(host.data(), device_out, host.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  return host;
}

void Upload(const std::vector<nanochat::Bf16>& source, Bf16* device) {
  cudaMemcpy(device, source.data(), source.size() * sizeof(nanochat::Bf16),
             cudaMemcpyHostToDevice);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf(
      "attention wgmma probe: the accumulator map and the ceiling, %d by %d "
      "by %d, interleaved form\n",
      kAtomM, kAtomN, kDim);

  int device = 0;
  cudaDeviceProp properties{};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  const int sm_count = properties.multiProcessorCount;
  std::printf("card: %s, %d SMs, %d kHz clock\n", properties.name, sm_count,
              properties.clockRate / 1000);

  cudaFuncSetAttribute(reinterpret_cast<const void*>(ProbeKernel),
                       cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

  std::vector<nanochat::Bf16> host_q(static_cast<std::size_t>(kAtomM) * kDim);
  std::vector<nanochat::Bf16> host_k(static_cast<std::size_t>(kAtomN) * kDim);
  Bf16* device_q = nullptr;
  Bf16* device_k = nullptr;
  float* device_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&device_q),
             host_q.size() * sizeof(nanochat::Bf16));
  cudaMalloc(reinterpret_cast<void**>(&device_k),
             host_k.size() * sizeof(nanochat::Bf16));
  cudaMalloc(
      reinterpret_cast<void**>(&device_out),
      static_cast<std::size_t>(kWarpgroupThreads) * kRegisters * sizeof(float));

  // --- One element at a time ------------------------------------------------
  // A one-hot query row against a one-hot key row leaves one non-zero value in
  // the product, and the slot that holds it is read off without any assumption
  // about the order of the registers.
  struct Pixel {
    int m;
    int n;
  };
  const Pixel pixels[] = {{0, 0},  {0, 1},  {0, 2},   {0, 7},    {0, 8},
                          {1, 0},  {2, 0},  {4, 0},   {8, 0},    {16, 0},
                          {32, 0}, {63, 0}, {0, 127}, {63, 127}, {17, 33}};
  int found = 0;
  for (const Pixel& pixel : pixels) {
    std::fill(host_q.begin(), host_q.end(), nanochat::Bf16FromFloat(0.0f));
    std::fill(host_k.begin(), host_k.end(), nanochat::Bf16FromFloat(0.0f));
    host_q[static_cast<std::size_t>(pixel.m) * kDim] =
        nanochat::Bf16FromFloat(1.0f);
    host_k[static_cast<std::size_t>(pixel.n) * kDim] =
        nanochat::Bf16FromFloat(1.0f);
    Upload(host_q, device_q);
    Upload(host_k, device_k);
    const std::vector<float> out = RunOnce(device_q, device_k, device_out);
    int lane = -1;
    int index = -1;
    for (int l = 0; l < kWarpgroupThreads; ++l) {
      for (int i = 0; i < kRegisters; ++i) {
        if (std::fabs(out[static_cast<std::size_t>(l) * kRegisters + i]) >
            0.5f) {
          lane = l;
          index = i;
        }
      }
    }
    if (lane >= 0) ++found;
    std::printf("attention-wgmma:pixel m=%d n=%d lane=%d index=%d\n", pixel.m,
                pixel.n, lane, index);
  }
  std::printf("attention-wgmma:pixel_summary probed=%zu found=%d\n",
              sizeof(pixels) / sizeof(pixels[0]), found);

  // --- The whole table ------------------------------------------------------
  // Two products decode the map: the first depends on the row only and the
  // second on the column only, and both patterns are exact in bf16, so a
  // decoded value is an index and not an estimate.
  std::vector<nanochat::Bf16> row_q(host_q.size());
  std::vector<nanochat::Bf16> row_k(host_k.size());
  std::vector<nanochat::Bf16> col_q(host_q.size());
  std::vector<nanochat::Bf16> col_k(host_k.size());
  for (int m = 0; m < kAtomM; ++m) {
    for (int k = 0; k < kDim; ++k) {
      row_q[static_cast<std::size_t>(m) * kDim + k] =
          nanochat::Bf16FromFloat(1.0f + static_cast<float>(m) / 64.0f);
      col_q[static_cast<std::size_t>(m) * kDim + k] =
          nanochat::Bf16FromFloat(1.0f);
    }
  }
  for (int n = 0; n < kAtomN; ++n) {
    for (int k = 0; k < kDim; ++k) {
      row_k[static_cast<std::size_t>(n) * kDim + k] =
          nanochat::Bf16FromFloat(1.0f);
      col_k[static_cast<std::size_t>(n) * kDim + k] =
          nanochat::Bf16FromFloat(1.0f + static_cast<float>(n) / 128.0f);
    }
  }
  Upload(row_q, device_q);
  Upload(row_k, device_k);
  const std::vector<float> rows = RunOnce(device_q, device_k, device_out);
  Upload(col_q, device_q);
  Upload(col_k, device_k);
  const std::vector<float> cols = RunOnce(device_q, device_k, device_out);

  auto row_of = [](float value) {
    return static_cast<int>(std::lround((value / 128.0 - 1.0) * 64.0));
  };
  auto col_of = [](float value) {
    return static_cast<int>(std::lround((value / 128.0 - 1.0) * 128.0));
  };
  std::vector<int> seen(static_cast<std::size_t>(kAtomM) * kAtomN, 0);
  int outside = 0;
  for (int lane = 0; lane < kWarpgroupThreads; ++lane) {
    for (int index = 0; index < kRegisters; ++index) {
      const std::size_t slot = static_cast<std::size_t>(lane) * kRegisters +
                               static_cast<std::size_t>(index);
      const int row = row_of(rows[slot]);
      const int col = col_of(cols[slot]);
      if (row < 0 || row >= kAtomM || col < 0 || col >= kAtomN) {
        ++outside;
        continue;
      }
      seen[static_cast<std::size_t>(row) * kAtomN + col] += 1;
    }
  }
  int missing = 0;
  int repeated = 0;
  for (int count : seen) {
    if (count == 0) ++missing;
    if (count > 1) ++repeated;
  }
  std::printf(
      "attention-wgmma:map_summary outside=%d missing=%d repeated=%d of %d\n",
      outside, missing, repeated, kAtomM * kAtomN);
  for (int lane : {0, 1, 4, 32}) {
    std::printf("attention-wgmma:map lane=%d", lane);
    for (int index = 0; index < kRegisters; ++index) {
      const std::size_t slot = static_cast<std::size_t>(lane) * kRegisters +
                               static_cast<std::size_t>(index);
      std::printf(" %d:%d", row_of(rows[slot]), col_of(cols[slot]));
    }
    std::printf("\n");
  }

  // --- The rate -------------------------------------------------------------
  // The score product of the whole tile, from a resident tile.
  const double flops_per_pass = 2.0 * kAtomM * kAtomN * kDim;
  const int passes = 20000;
  std::fill(host_q.begin(), host_q.end(), nanochat::Bf16FromFloat(0.5f));
  std::fill(host_k.begin(), host_k.end(), nanochat::Bf16FromFloat(0.5f));
  Upload(host_q, device_q);
  Upload(host_k, device_k);
  for (int blocks : {1, sm_count}) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    ProbeKernel<<<blocks, kWarpgroupThreads, kSmemBytes>>>(device_q, device_k,
                                                           device_out, 8);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    ProbeKernel<<<blocks, kWarpgroupThreads, kSmemBytes>>>(device_q, device_k,
                                                           device_out, passes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    const double flops = flops_per_pass * static_cast<double>(passes) * blocks;
    const double rate = flops / (static_cast<double>(milliseconds) * 1e-3);
    std::printf(
        "attention-wgmma:resident blocks=%d passes=%d ms=%.3f tflops=%.1f "
        "fraction_of_peak=%.3f\n",
        blocks, passes, milliseconds, rate / 1e12, rate / kPeakFlops);
  }

  std::printf(
      "attention-wgmma:note smem_bytes=%d lbo=%d sbo=%d "
      "query_atom_step=%d key_atom_step=%d\n",
      kSmemBytes, kLboBytes, kSboBytes, AtomStep(kAtomM), AtomStep(kAtomN));

  cudaFree(device_q);
  cudaFree(device_k);
  cudaFree(device_out);
  return 0;
}
