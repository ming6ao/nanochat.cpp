// The fp32 shared-memory GEMM of docs/kernels.md Part 3, section 6 phase 3.
//
// Phase 3 is the chain leg on Turing. The T4 declares no bf16
// (`backends/cuda/caps.json`), so the post-training chain runs there in fp32,
// and `sm_75` has neither a warpgroup instruction nor a Tensor Memory
// Accelerator. This kernel therefore uses no tensor core at all: it stages both
// operands in shared memory, holds one eight by eight accumulator tile in the
// registers of each thread, and multiplies with plain fused multiply-add.
//
// Three parts follow, in the order the kernel runs them:
//
//   * the k loop, which walks `Plan::kStages` staged slices and copies slice
//     `n + 1` while it computes slice `n`;
//   * the tile loop, which walks the block grid of one call, batch by batch;
//   * the epilogue, which applies alpha, beta, and the bounds and writes C in
//     sixteen-byte chunks where the destination allows one.
//
// **Both operands stage k-major**, `As[k][m]` and `Bs[k][n]`, so one thread
// reads its whole register tile with four sixteen-byte loads: two for A and two
// for B. That layout is what makes the tile conflict free without a pad that
// costs a block its second slot on the multiprocessor, and it costs one
// transposing copy for the two operand forms whose memory runs along the other
// axis.
//
// This source exists only in an fp32 build, so it holds no guard: the build
// selects it (`backends/cuda/gemm/BUILD.bazel`).

#include "backends/cuda/gemm/gemm_fp32_launch.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/gemm_fp32_plan.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

//: One sixteen-byte access. The structure is a plain aligned array, so the
//: device compiler emits one 128-bit load or store, and the host pass of the
//: emulation engine, which holds no `float4`, can still name the width.
struct alignas(16) Vector4 {
  float e[4];
};

__device__ __forceinline__ Vector4 LoadVector4(const float* base, int index) {
  return *reinterpret_cast<const Vector4*>(base + index);
}

__device__ __forceinline__ void StoreVector4(float* base, int index,
                                             const Vector4& value) {
  *reinterpret_cast<Vector4*>(base + index) = value;
}

//: The staged window of one block: `Plan::kStages` slices of A and of B, and
//: nothing else. The epilogue reads and writes global memory, so it needs no
//: scratch and the window never has to be split.
template <typename PlanT>
struct alignas(16) StagedWindow {
  float a[PlanT::kStages * PlanT::kStageAElements];
  float b[PlanT::kStages * PlanT::kStageBElements];
};

// Stages one k slice of A into `As[k][m]`.
//
// A plain operand is `m` by `k` with `k` contiguous, so the thread index walks
// `k` fastest and the global read stays coalesced along the row. A transposed
// operand is `k` by `m` with `m` contiguous, so the index walks `m` fastest and
// the read stays coalesced along the column that is contiguous. Either way the
// staged write is k-major, and a slice past the operand edge stages zero so the
// product never gains a term.
template <typename PlanT, bool TransA>
__device__ __forceinline__ void StageA(const float* a, int lda, int m0, int k0,
                                       int m_extent, int k_extent,
                                       float* stage) {
  constexpr int kElements = PlanT::kTileK * PlanT::kTileM;
  for (int index = static_cast<int>(threadIdx.x); index < kElements;
       index += PlanT::kThreads) {
    int m = 0;
    int k = 0;
    if constexpr (TransA) {
      k = index / PlanT::kTileM;
      m = index - k * PlanT::kTileM;
    } else {
      m = index / PlanT::kTileK;
      k = index - m * PlanT::kTileK;
    }
    const int row = m0 + m;
    const int column = k0 + k;
    float value = 0.0f;
    if (row < m_extent && column < k_extent) {
      const long long offset = TransA
                                   ? static_cast<long long>(column) * lda + row
                                   : static_cast<long long>(row) * lda + column;
      value = a[offset];
    }
    stage[k * PlanT::kLeadM + m] = value;
  }
}

// Stages one k slice of B into `Bs[k][n]`, by the same rule: a plain operand is
// `k` by `n` with `n` contiguous, and a transposed one is `n` by `k` with `k`
// contiguous.
template <typename PlanT, bool TransB>
__device__ __forceinline__ void StageB(const float* b, int ldb, int n0, int k0,
                                       int n_extent, int k_extent,
                                       float* stage) {
  constexpr int kElements = PlanT::kTileK * PlanT::kTileN;
  for (int index = static_cast<int>(threadIdx.x); index < kElements;
       index += PlanT::kThreads) {
    int n = 0;
    int k = 0;
    if constexpr (TransB) {
      n = index / PlanT::kTileK;
      k = index - n * PlanT::kTileK;
    } else {
      k = index / PlanT::kTileN;
      n = index - k * PlanT::kTileN;
    }
    const int row = k0 + k;
    const int column = n0 + n;
    float value = 0.0f;
    if (row < k_extent && column < n_extent) {
      const long long offset = TransB
                                   ? static_cast<long long>(column) * ldb + row
                                   : static_cast<long long>(row) * ldb + column;
      value = b[offset];
    }
    stage[k * PlanT::kLeadN + n] = value;
  }
}

// One staged slice against one register tile. Each thread reads its A values
// and its B values with one sixteen-byte load for each four of them, then takes
// sixty-four products. The loads broadcast inside a warp, because the sixteen
// threads of one row of the block read the same eight A values, so no bank
// repeats.
template <typename PlanT>
__device__ __forceinline__ void ComputeSlice(
    const float* as, const float* bs, int m_local, int n_local,
    float (&acc)[PlanT::kThreadTileM][PlanT::kThreadTileN]) {
  constexpr int kChunksM = PlanT::kThreadTileM / 4;
  constexpr int kChunksN = PlanT::kThreadTileN / 4;
  static_assert(PlanT::kThreadTileM % 4 == 0 && PlanT::kThreadTileN % 4 == 0,
                "a thread tile is whole sixteen-byte chunks");
  for (int kk = 0; kk < PlanT::kTileK; ++kk) {
    float a_reg[PlanT::kThreadTileM];
    float b_reg[PlanT::kThreadTileN];
    for (int chunk = 0; chunk < kChunksM; ++chunk) {
      const Vector4 values =
          LoadVector4(as, kk * PlanT::kLeadM + m_local + 4 * chunk);
      for (int e = 0; e < 4; ++e) a_reg[4 * chunk + e] = values.e[e];
    }
    for (int chunk = 0; chunk < kChunksN; ++chunk) {
      const Vector4 values =
          LoadVector4(bs, kk * PlanT::kLeadN + n_local + 4 * chunk);
      for (int e = 0; e < 4; ++e) b_reg[4 * chunk + e] = values.e[e];
    }
    for (int i = 0; i < PlanT::kThreadTileM; ++i) {
      for (int j = 0; j < PlanT::kThreadTileN; ++j) {
        acc[i][j] += a_reg[i] * b_reg[j];
      }
    }
  }
}

// The epilogue of one thread's register tile. `alpha` multiplies the
// accumulator, `beta` reads C where the call asks for it, and a row or a column
// past the operand edge is skipped rather than masked. A chunk of four columns
// is one sixteen-byte store when the destination holds four whole columns and
// its address shares the boundary; otherwise the four elements go one at a
// time.
template <typename PlanT>
__device__ __forceinline__ void StoreTile(
    const float (&acc)[PlanT::kThreadTileM][PlanT::kThreadTileN], float* c_base,
    const ResolvedGemm& args, int m0, int n0, int m_local, int n_local) {
  for (int i = 0; i < PlanT::kThreadTileM; ++i) {
    const int row = m0 + m_local + i;
    if (row >= args.m) continue;
    float* c_row =
        c_base + static_cast<long long>(row) * args.ldc + n0 + n_local;
    for (int j = 0; j < PlanT::kThreadTileN; j += 4) {
      const int column = n0 + n_local + j;
      if (column >= args.n) break;
      const bool whole =
          column + 3 < args.n &&
          reinterpret_cast<std::uintptr_t>(c_row + j) % sizeof(Vector4) == 0;
      if (whole) {
        Vector4 value;
        for (int e = 0; e < 4; ++e) {
          float element = args.alpha * acc[i][j + e];
          if (args.beta != 0.0f) element += args.beta * c_row[j + e];
          value.e[e] = element;
        }
        StoreVector4(c_row, j, value);
      } else {
        for (int e = 0; e < 4; ++e) {
          if (column + e >= args.n) break;
          float element = args.alpha * acc[i][j + e];
          if (args.beta != 0.0f) element += args.beta * c_row[j + e];
          c_row[j + e] = element;
        }
      }
    }
  }
}

// One tile of one call, with the operand form of the call fixed at compile
// time. The grid is one block for each tile of each batch, and the loop keeps a
// caller that sizes the grid below that count correct.
template <typename PlanT, bool TransA, bool TransB>
__global__ void __launch_bounds__(PlanT::kThreads, gemm_fp32::kMinBlocksPerSm)
    GemmFp32Kernel(const float* a, const float* b, float* c,
                   ResolvedGemm args) {
  __shared__ StagedWindow<PlanT> window;

  const int thread = static_cast<int>(threadIdx.x);
  const int m_local = (thread % PlanT::kThreadsM) * PlanT::kThreadTileM;
  const int n_local = (thread / PlanT::kThreadsM) * PlanT::kThreadTileN;

  const int grid_n = (args.n + PlanT::kTileN - 1) / PlanT::kTileN;
  const int grid_m = (args.m + PlanT::kTileM - 1) / PlanT::kTileM;
  const int tiles = grid_m * grid_n;
  const int total = tiles * args.batch_count;
  const int slices = (args.k + PlanT::kTileK - 1) / PlanT::kTileK;

  for (int index = static_cast<int>(blockIdx.x); index < total;
       index += static_cast<int>(gridDim.x)) {
    const int batch = index / tiles;
    const int tile = index - batch * tiles;
    const int m_tile = tile / grid_n;
    const int n_tile = tile - m_tile * grid_n;
    const int m0 = m_tile * PlanT::kTileM;
    const int n0 = n_tile * PlanT::kTileN;
    const float* a_base = a + static_cast<long long>(batch) * args.stride_a;
    const float* b_base = b + static_cast<long long>(batch) * args.stride_b;

    float acc[PlanT::kThreadTileM][PlanT::kThreadTileN];
    for (int i = 0; i < PlanT::kThreadTileM; ++i) {
      for (int j = 0; j < PlanT::kThreadTileN; ++j) acc[i][j] = 0.0f;
    }

    // The pipeline. Slice 0 is staged first, and each iteration then stages
    // slice `n + 1` into the other buffer while it computes slice `n`. The
    // barrier at the end of an iteration is what makes the next slice whole for
    // every warp. A barrier between the staging and the compute is not needed:
    // the two touch different buffers.
    StageA<PlanT, TransA>(a_base, args.lda, m0, 0, args.m, args.k, window.a);
    StageB<PlanT, TransB>(b_base, args.ldb, n0, 0, args.n, args.k, window.b);
    __syncthreads();

    for (int slice = 0; slice < slices; ++slice) {
      const int next = slice + 1;
      if (next < slices) {
        const int buffer = next % PlanT::kStages;
        StageA<PlanT, TransA>(a_base, args.lda, m0, next * PlanT::kTileK,
                              args.m, args.k,
                              window.a + buffer * PlanT::kStageAElements);
        StageB<PlanT, TransB>(b_base, args.ldb, n0, next * PlanT::kTileK,
                              args.n, args.k,
                              window.b + buffer * PlanT::kStageBElements);
      }
      const int buffer = slice % PlanT::kStages;
      ComputeSlice<PlanT>(window.a + buffer * PlanT::kStageAElements,
                          window.b + buffer * PlanT::kStageBElements, m_local,
                          n_local, acc);
      __syncthreads();
    }

    StoreTile<PlanT>(acc, c + static_cast<long long>(batch) * args.stride_c,
                     args, m0, n0, m_local, n_local);
    // The next tile of this block stages into the same window, so every warp
    // must be done with the slice it computed before the copy starts.
    __syncthreads();
  }
}

// Resolves the call, then launches one grid of that schedule. The staged window
// is static, so the launch needs no shared-memory attribute and no opt-in
// limit: a schedule that fits the 48 KiB static window is the whole
// requirement.
template <typename PlanT, bool TransA, bool TransB>
bool LaunchPlan(const GemmParams& params, const float* a, const float* b,
                float* c) {
  const ResolvedGemm gemm = ResolveGemm(params, /*fp32_c=*/true);
  if (!IsTileCallable(gemm)) return false;
  const long long tiles =
      static_cast<long long>((gemm.m + PlanT::kTileM - 1) / PlanT::kTileM) *
      static_cast<long long>((gemm.n + PlanT::kTileN - 1) / PlanT::kTileN) *
      gemm.batch_count;
  const long long blocks = tiles > 0 ? tiles : 1;
  cuda_backend::Launch(GemmFp32Kernel<PlanT, TransA, TransB>,
                       dim3(static_cast<unsigned int>(blocks)),
                       dim3(static_cast<unsigned int>(PlanT::kThreads)),
                       /*shared=*/0, a, b, c, gemm);
  return true;
}

// The four operand forms of one schedule.
template <typename PlanT>
bool LaunchForm(const GemmParams& params, const float* a, const float* b,
                float* c) {
  if (params.transpose_a) {
    return params.transpose_b ? LaunchPlan<PlanT, true, true>(params, a, b, c)
                              : LaunchPlan<PlanT, true, false>(params, a, b, c);
  }
  return params.transpose_b ? LaunchPlan<PlanT, false, true>(params, a, b, c)
                            : LaunchPlan<PlanT, false, false>(params, a, b, c);
}

// The sweep: one branch for each schedule of `gemm_fp32_plan.h`. The
// assertion is what keeps the two lists from drifting apart.
bool LaunchSchedule(int schedule, const GemmParams& params, const float* a,
                    const float* b, float* c) {
  static_assert(gemm_fp32::kScheduleCount == 2,
                "one branch for each schedule of the sweep");
  if (schedule == 1) return LaunchForm<gemm_fp32::PlanHalf>(params, a, b, c);
  return LaunchForm<gemm_fp32::PlanWide>(params, a, b, c);
}

}  // namespace

bool LaunchGemmFp32(const GemmParams& params, const float* a, const float* b,
                    float* c, int schedule) {
  const int index = (schedule >= 0 && schedule < gemm_fp32::kScheduleCount)
                        ? schedule
                        : gemm_fp32::kDefaultSchedule;
  return LaunchSchedule(index, params, a, b, c);
}

}  // namespace kernels
}  // namespace nanochat
