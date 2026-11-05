// The fused Muon update: one phase-parameterised device function, its 16
// phases, the three epilogue folds of the Polar Express iteration, and the two
// drivers over one device body (docs/muon-fused-kernel.md sections 4 to 6).
//
// `MuonUpdate` (muon.cu) issues 39 launches for one shape group of one step at
// the shipped `ns_steps` of 5: fifteen batched cuBLAS GEMMs and twenty-four
// hand-written kernels. This file folds the whole stage into 28 phase
// executions over one device function, so one launch per shape group carries
// the step and the epilogue folds drop two buffers and two passes over the
// stack.
//
// Two drivers run the same body, and neither one owns any arithmetic:
//
//   * the phase driver launches one grid per phase execution, sized for that
//     phase. It is the simulator's path (the emulator runs blocks one at a
//     time, so a grid barrier there would deadlock), the fallback for a card
//     without a cooperative launch, and the bisection tool a phase gate uses;
//   * the cooperative driver launches one grid for the whole group and puts a
//     grid barrier between the executions. The grid is the resident-block query
//     of `backends/cuda/device.h`, and a refused launch
//     (`cudaErrorCooperativeLaunchTooLarge`) falls back to the phase driver
//     with one log line.
//
// One grid serves every phase of the cooperative driver, so a launch fixes the
// block size, the grid size, and the dynamic shared size for the whole kernel.
// Each phase therefore takes a flat work-item index and every resident block
// strides over the items of the phase it is in: an elementwise phase strides by
// gridDim * blockDim elements (or 16-byte vectors), and a phase whose item is a
// block of work (a row, a matrix, a GEMM tile) strides by gridDim items. The
// phase with the most items sets nothing here, because the persistent grid is
// the query's answer; the phase driver sizes each of its grids separately,
// which is one reason to keep it.
//
// The workspace is `[x | xn | a | b]` from one cached allocation:
//
//   * `x` and `xn` are the iterate and its ping-pong partner. Phase 10 writes
//     the combined iterate into the partner with the `ca * x + prod` epilogue,
//     so the `prod` buffer of the multi-launch path doubles as the second
//     iterate and no third full-size buffer is needed;
//   * `a` holds `A = X^T X` (tall) or `X X^T` (wide), and `b` holds the
//     combined `b = cb * a + cc * a * a` written from the phase 9 accumulator,
//     which removes the separate `a2` buffer and the pass over `min_dim^2` that
//     filled it.
//
// That is `2 * total + 2 * min_total` against the multi-launch path's
// `2 * total + 3 * min_total`, and `EnsureMuonWorkspace` in muon.cu sizes it.
// Every whole-matrix statistic keeps its block count and its chunk order
// (`MuonPartialBlocks`), and both partial passes stay in double, so those
// stages stay bit exact against the multi-launch path; the Polar Express GEMM's
// reduction order is the one arithmetic difference, and the tests compare it at
// the family's 1e-5 tolerance against optim_ref.h.
//
// See backends/cuda/kernels/muon_fused.h, docs/kernels.md, and docs/model.md.

#include "backends/cuda/kernels/muon_fused.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#endif

namespace nanochat {
namespace kernels {

namespace {

using cuda_backend::CooperativeLaunchOutcome;
using cuda_kernels::AsFloatDev;
using cuda_kernels::ComputeVector;
using cuda_kernels::FloatVector;
using cuda_kernels::kFloatPerVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorReady;

// Polar Express coefficients (num_iters=5), from nanochat/optim.py and the
// Polar Express paper (arXiv:2505.16932). Kept in sync with
// backends/cpu/kernels.cc and backends/cuda/kernels/muon.cu. The fused device
// function reads them from the plan rather than from this table, because a
// device function cannot index a host `constexpr` array; the plan carries a
// copy per launch.
constexpr float kPolarCoeffs[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

// --- The linear schedule of phase executions -------------------------------
//
// Phase ids, as section 4 numbers them.
constexpr int kPhaseMomentum = 1;
constexpr int kPhaseFrobPartial = 2;
constexpr int kPhaseFrobFinish = 3;
constexpr int kPhaseEquilibrate = 4;
constexpr int kPhaseDivisorPartial = 5;
constexpr int kPhaseDivisorFinish = 6;
constexpr int kPhaseDivisorScale = 7;
constexpr int kPhaseSquare = 8;
constexpr int kPhaseSquareSquare = 9;
constexpr int kPhaseApplySquare = 10;
constexpr int kPhaseRenormPartial = 11;
constexpr int kPhaseRenormFinish = 12;
constexpr int kPhaseRenormScale = 13;
constexpr int kPhaseMeanSquare = 14;
constexpr int kPhaseMoments = 15;
constexpr int kPhaseNorMuonScale = 16;

constexpr int kPhaseCount = 16;
// Seven executions before the Polar Express loop, three per iteration, and six
// after it: 28 at the shipped `ns_steps` of 5, which is 27 grid barriers
// against the multi-launch path's 39 launches.
constexpr int kExecsBefore = 7;
constexpr int kExecsAfter = 6;

// The phase and the Polar Express iteration of one execution of the schedule.
// The GEMM phases carry the iteration, because the ping-pong pair they read and
// write alternate with it; every other phase ignores it.
//
// Both sides call this: the kernel decodes the execution it is running, and the
// phase driver decodes the one it is about to launch so it can size that grid.
__host__ __device__ __forceinline__ void FusedDecodeExec(int exec, int ns_steps,
                                                         int* phase,
                                                         int* iter) {
  if (exec < kExecsBefore) {
    *phase = exec + 1;
    *iter = 0;
    return;
  }
  const int body = exec - kExecsBefore;
  if (body < 3 * ns_steps) {
    *phase = kPhaseSquare + body % 3;
    *iter = body / 3;
    return;
  }
  *phase = kPhaseRenormPartial + (body - 3 * ns_steps);
  *iter = 0;
}

// --- The batched GEMM of one Polar Express iteration -----------------------
//
// The schedule the phase 1 measurement fixed (section 4): 256 threads under
// `__launch_bounds__(256, 3)`, a 128 by 64 block tile, an 8 by 4 micro-tile per
// thread (32 fp32 accumulators in 16 by 16 thread groups), two double-buffered
// 8-deep k slices in 12 KiB of shared memory, and one flat
// `(matrix, tile_i, tile_j)` index per block. `dev/kernels/muon_fused_gemm.cuh`
// is the measured prototype this is ported from; the loaders, the tile, and the
// unwinding are the prototype's, so the rate that was measured is the rate this
// path runs.
constexpr int kGemmThreads = 256;
constexpr int kGemmBlockM = 128;
constexpr int kGemmBlockN = 64;
constexpr int kGemmSliceK = 8;
constexpr int kGemmMicroM = 8;
constexpr int kGemmMicroN = 4;
constexpr int kGemmBlocksPerSm = 3;
constexpr int kThreadCols = kGemmBlockN / kGemmMicroN;
constexpr int kThreadRows = kGemmThreads / kThreadCols;
constexpr int kQuadsPerSlice = kGemmSliceK / 4;
constexpr int kBThreads = kGemmSliceK * (kGemmBlockN / 4);
// The two double-buffered slices of both operands.
constexpr int kGemmSmemFloats = 2 * (kGemmBlockM + kGemmBlockN) * kGemmSliceK;
// One scalar per thread for the block-wide reductions the statistic phases run.
// Two arrays, because a `double` partial pass and a `float` statistic must each
// reduce in the tree `BlockReduceSum` uses, and neither may alias the other.
// Every launch of this kernel is `kGemmThreads` wide, so one entry per thread
// is the whole scratch.
constexpr int kReduceScalars = kGemmThreads;

static_assert(kThreadRows * kGemmMicroM == kGemmBlockM,
              "one block tile must be covered by the thread rows");
static_assert(kThreadRows * kThreadCols == kGemmThreads,
              "one block must be covered by the thread grid");

// The largest stacked group the fused path is selected for, in `ComputeType`
// elements: `MuonFusedSelected` is the shape rule of section 4, and this is its
// one number. It is measured and not chosen. `muon_benchmark`'s `muon_fused:`
// rows against its `muon:` rows at both profiles (`h100` and `big`) say the
// fused step is slower for every group on the host card: from 13 % on the
// 256x512 key/value group to 76 % on 1280x5120, where the hand GEMM loses most
// to cuBLAS. The empty set is therefore the measured answer, and zero states it
// as a number: every group takes the multi-launch path, where `kernels::Gemm`
// is cuBLAS, and promoting a group is one measurement and this constant.
//
// The rows are in backends/cuda/kernels/README.md, under "The fused path on
// this card".
constexpr long long kMuonFusedMaxElements = 0;

// One 16-byte group of operand values, the width the measured schedule loads.
// A plain struct rather than `float4`, because the emulator compiles this file
// with the host compiler and its CUDA stub has no vector types
// (tools/cuda_sim/emu/cuda_emu.h); the alignment is what makes the load one
// 128-bit access on the device.
struct alignas(16) FusedQuad {
  float e[4];
};

__device__ __forceinline__ FusedQuad LoadQuad(const float* p) {
  return *reinterpret_cast<const FusedQuad*>(p);
}

__device__ __forceinline__ FusedQuad MakeQuad(float x, float y, float z,
                                              float w) {
  FusedQuad q;
  q.e[0] = x;
  q.e[1] = y;
  q.e[2] = z;
  q.e[3] = w;
  return q;
}

// The five operand forms of the family (section 4). The first four are
// expressible with the transpose flag pair `GemmParams` already carries.
enum class FusedForm {
  kXtX,  // A = X^T X    tall branch: transpose_a, operands (x, x)
  kXXt,  // A = X X^T    wide branch: transpose_b, operands (x, x)
  kAA,   // A2 = A A     both operands are the square matrix
  kXB,  // prod = X B   tall branch: plain by plain, square operand on the right
  kBX,  // prod = B X   wide branch: plain by plain, square operand on the left
};

// Which side of a plain product the min-by-min square operand sits on.
// `transpose_a` and `transpose_b` cannot express the wide-branch product: both
// `X B` (tall) and `B X` (wide) are plain operand times plain operand, and only
// the side the square matrix sits on separates them. The selector names that
// side, so the pair (flags, side) selects all five forms.
enum class FusedSide {
  kRight,  // square operand is `b`: C = X * square    (`X B`)
  kLeft,   // square operand is `a`: C = square * X    (`B X`)
};

// What the store of one tile does after the reduction. The three epilogue folds
// of section 4 land here, and they are the whole point of the fused path: the
// product never travels through memory between the GEMM and the elementwise
// stage that consumed it in the multi-launch path.
enum class FusedEpilogue {
  kWrite,        // C = op(A) * op(B)                     (phase 8)
  kFoldSquare,   // C = cb * a + cc * acc                 (phase 9)
  kFoldIterate,  // C = ca * x + acc                      (phase 10)
};

// One batched GEMM in the seam's convention: `C[p] = op(A[p]) * op(B[p])` for
// `p` in `[0, batch)`, row-major, plus the epilogue the phase folds in.
struct FusedGemmOp {
  int m = 0;
  int n = 0;
  int k = 0;
  const float* a = nullptr;
  const float* b = nullptr;
  float* c = nullptr;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  bool transpose_a = false;
  bool transpose_b = false;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
  FusedEpilogue epilogue = FusedEpilogue::kWrite;
  // The buffer the fold reads, at the same `(row, column)` as the accumulator:
  // `a` for the `b = cb*a + cc*acc` fold, the current iterate for the
  // `ca*x + acc` one.
  const float* fold = nullptr;
  int fold_ld = 0;
  float ca = 0.0f;
  float cb = 0.0f;
  float cc = 0.0f;
};

// --- The plan the two drivers launch --------------------------------------
//
// Everything one launch of the fused device function needs. Trivially copyable,
// because it is a kernel parameter: the pointers, the geometry `MuonUpdate`
// decides, the scalars, and the slice of the schedule this launch covers.
struct FusedPlan {
  int num_params = 0;
  int rows = 0;
  int cols = 0;
  std::int64_t mat = 0;        // rows * cols, one matrix
  std::int64_t total = 0;      // num_params * mat
  int min_dim = 0;             // min(rows, cols): the square extent
  std::int64_t min_total = 0;  // num_params * min_dim * min_dim
  int red_index = 0;           // NorMuon statistics per matrix
  int red_size = 0;            // the extent one statistic averages
  bool reduce_cols = false;    // one statistic per row
  bool tall = false;           // rows > cols: A = X^T X, prod = X B
  bool vector = false;         // the elementwise phases move 16-byte vectors
  bool nesterov = false;
  int partial_blocks = 0;  // blocks per matrix in a partial pass
  int ns_steps = 0;
  float momentum = 0.0f;
  float beta2 = 0.0f;
  float lr = 0.0f;
  float weight_decay = 0.0f;
  float renorm_target = 0.0f;  // sqrt(min_dim)
  ScalarArg grad_scale;
  float coeffs[5][3] = {};
  // The workspace slices: the iterate, its ping-pong partner, and the two
  // min-by-min buffers.
  ComputeType* x = nullptr;
  ComputeType* xn = nullptr;
  ComputeType* a = nullptr;
  ComputeType* b = nullptr;
  // The caller's state.
  const float* grads = nullptr;
  float* master = nullptr;
  float* buf1 = nullptr;
  float* buf2 = nullptr;
  ComputeType* values = nullptr;
  // The statistics scratch `[stat | vmean | scale]` and the double partials.
  float* stat = nullptr;
  float* vmean = nullptr;
  float* scale = nullptr;
  double* partials = nullptr;
  // The slice of the schedule this launch covers: one execution for a phase
  // launch, all of them for a cooperative launch.
  int exec_begin = 0;
  int exec_count = 0;
};

// --- The loaders -----------------------------------------------------------
//
// Every loader comes in a pair. `Gather` reads one thread's four elements of
// one k slice into registers and `Scatter` writes them into the shared tile, so
// a slice's two global reads are both in flight before either store waits on
// one: one memory round trip per slice instead of two.
//
// The four loaders differ only in which extent of the stored matrix is
// contiguous: `transpose_a` says the stored `A` is `k` by `m` (so `m` is
// contiguous) while a plain `A` is `m` by `k` (so `k` is), and the same holds
// for `B`.

// op(A), transposed: stored k by m, so m is contiguous. Thread `tid` owns four
// columns of one k row: `kk = tid / (kGemmBlockM / 4)` of the slice and columns
// `m0 + quad * 4 .. + 3`.
__device__ __forceinline__ FusedQuad GatherATransposed(const float* base,
                                                       int ld, int m0, int k0,
                                                       int m_ext, int k_ext,
                                                       int tid) {
  const int quad = tid % (kGemmBlockM / 4);
  const int kk = tid / (kGemmBlockM / 4);
  const int gm = m0 + quad * 4;
  const int gk = k0 + kk;
  if ((gm + 4 <= m_ext) && (gk < k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gk) * ld + gm);
  }
  const float* row = base + static_cast<std::int64_t>(gk) * ld;
  const bool in_row = gk < k_ext;
  return MakeQuad(in_row && gm + 0 < m_ext ? row[gm + 0] : 0.0f,
                  in_row && gm + 1 < m_ext ? row[gm + 1] : 0.0f,
                  in_row && gm + 2 < m_ext ? row[gm + 2] : 0.0f,
                  in_row && gm + 3 < m_ext ? row[gm + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterATransposed(const FusedQuad& v, int tid,
                                                   float (*dst)[kGemmBlockM]) {
  *reinterpret_cast<FusedQuad*>(
      &dst[tid / (kGemmBlockM / 4)][(tid % (kGemmBlockM / 4)) * 4]) = v;
}

// op(A), plain: stored m by k, so k is contiguous. Thread `tid` owns four k of
// one row: row `m0 + tid / kQuadsPerSlice` and k `k0 + (tid %
// kQuadsPerSlice) * 4 .. + 3`. The four values land in four k slices of the
// shared tile, which is the transpose the inner product wants.
__device__ __forceinline__ FusedQuad GatherAPlain(const float* base, int ld,
                                                  int m0, int k0, int m_ext,
                                                  int k_ext, int tid) {
  const int mm = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  const int gm = m0 + mm;
  const int gk = k0 + kq;
  if ((gm < m_ext) && (gk + 4 <= k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gm) * ld + gk);
  }
  const float* row = base + static_cast<std::int64_t>(gm) * ld;
  const bool in_row = gm < m_ext;
  return MakeQuad(in_row && gk + 0 < k_ext ? row[gk + 0] : 0.0f,
                  in_row && gk + 1 < k_ext ? row[gk + 1] : 0.0f,
                  in_row && gk + 2 < k_ext ? row[gk + 2] : 0.0f,
                  in_row && gk + 3 < k_ext ? row[gk + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterAPlain(const FusedQuad& v, int tid,
                                              float (*dst)[kGemmBlockM]) {
  const int mm = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  dst[kq + 0][mm] = v.e[0];
  dst[kq + 1][mm] = v.e[1];
  dst[kq + 2][mm] = v.e[2];
  dst[kq + 3][mm] = v.e[3];
}

// op(B), plain: stored k by n, so n is contiguous. Only `kBThreads` threads own
// B tile elements: thread `tid < 128` owns four columns of one k row,
// `kk = tid / (kGemmBlockN / 4)` and columns `n0 + (tid % (kGemmBlockN / 4)) *
// 4 .. + 3`.
__device__ __forceinline__ FusedQuad GatherBPlain(const float* base, int ld,
                                                  int n0, int k0, int n_ext,
                                                  int k_ext, int tid) {
  const int kk = tid / (kGemmBlockN / 4);
  const int quad = (tid % (kGemmBlockN / 4)) * 4;
  const int gn = n0 + quad;
  const int gk = k0 + kk;
  if ((gn + 4 <= n_ext) && (gk < k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gk) * ld + gn);
  }
  const float* row = base + static_cast<std::int64_t>(gk) * ld;
  const bool in_row = gk < k_ext;
  return MakeQuad(in_row && gn + 0 < n_ext ? row[gn + 0] : 0.0f,
                  in_row && gn + 1 < n_ext ? row[gn + 1] : 0.0f,
                  in_row && gn + 2 < n_ext ? row[gn + 2] : 0.0f,
                  in_row && gn + 3 < n_ext ? row[gn + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterBPlain(const FusedQuad& v, int tid,
                                              float (*dst)[kGemmBlockN]) {
  *reinterpret_cast<FusedQuad*>(
      &dst[tid / (kGemmBlockN / 4)][(tid % (kGemmBlockN / 4)) * 4]) = v;
}

// op(B), transposed: stored n by k, so k is contiguous. Thread `tid < 128` owns
// four k of one column: column `n0 + tid / kQuadsPerSlice` and k slices
// `k0 + (tid % kQuadsPerSlice) * 4 .. + 3`.
__device__ __forceinline__ FusedQuad GatherBTransposed(const float* base,
                                                       int ld, int n0, int k0,
                                                       int n_ext, int k_ext,
                                                       int tid) {
  const int nn = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  const int gn = n0 + nn;
  const int gk = k0 + kq;
  if ((gn < n_ext) && (gk + 4 <= k_ext) && (ld % 4 == 0)) {
    return LoadQuad(base + static_cast<std::int64_t>(gn) * ld + gk);
  }
  const float* row = base + static_cast<std::int64_t>(gn) * ld;
  const bool in_row = gn < n_ext;
  return MakeQuad(in_row && gk + 0 < k_ext ? row[gk + 0] : 0.0f,
                  in_row && gk + 1 < k_ext ? row[gk + 1] : 0.0f,
                  in_row && gk + 2 < k_ext ? row[gk + 2] : 0.0f,
                  in_row && gk + 3 < k_ext ? row[gk + 3] : 0.0f);
}

__device__ __forceinline__ void ScatterBTransposed(const FusedQuad& v, int tid,
                                                   float (*dst)[kGemmBlockN]) {
  const int nn = tid / kQuadsPerSlice;
  const int kq = (tid % kQuadsPerSlice) * 4;
  dst[kq + 0][nn] = v.e[0];
  dst[kq + 1][nn] = v.e[1];
  dst[kq + 2][nn] = v.e[2];
  dst[kq + 3][nn] = v.e[3];
}

// One k slice of both operands of one block: both global reads are issued
// before either store consumes one.
__device__ __forceinline__ void LoadSlice(const float* ap, const float* bp,
                                          int lda, int ldb, bool transpose_a,
                                          bool transpose_b, int m0, int n0,
                                          int m, int n, int k, int k0, int tid,
                                          float (*sa)[kGemmBlockM],
                                          float (*sb)[kGemmBlockN]) {
  const FusedQuad a_regs = transpose_a
                               ? GatherATransposed(ap, lda, m0, k0, m, k, tid)
                               : GatherAPlain(ap, lda, m0, k0, m, k, tid);
  FusedQuad b_regs = MakeQuad(0.0f, 0.0f, 0.0f, 0.0f);
  if (tid < kBThreads) {
    b_regs = transpose_b ? GatherBTransposed(bp, ldb, n0, k0, n, k, tid)
                         : GatherBPlain(bp, ldb, n0, k0, n, k, tid);
  }
  if (transpose_a) {
    ScatterATransposed(a_regs, tid, sa);
  } else {
    ScatterAPlain(a_regs, tid, sa);
  }
  if (tid < kBThreads) {
    if (transpose_b) {
      ScatterBTransposed(b_regs, tid, sb);
    } else {
      ScatterBPlain(b_regs, tid, sb);
    }
  }
}

// One `(matrix, tile_i, tile_j)` of one GEMM: the prototype's tile, with the
// phase's epilogue applied at the store.
__device__ __forceinline__ void FusedGemmTile(const FusedGemmOp& op, int matrix,
                                              int tile_i, int tile_j,
                                              float* s_gemm) {
  const int m0 = tile_i * kGemmBlockM;
  const int n0 = tile_j * kGemmBlockN;
  const float* ap = op.a + static_cast<std::int64_t>(matrix) * op.stride_a;
  const float* bp = op.b + static_cast<std::int64_t>(matrix) * op.stride_b;
  float* cp = op.c + static_cast<std::int64_t>(matrix) * op.stride_c;
  const float* foldp =
      op.fold != nullptr
          ? op.fold + static_cast<std::int64_t>(matrix) * op.stride_c
          : nullptr;

  const int tid = static_cast<int>(threadIdx.x);
  const int row0 = (tid / kThreadCols) * kGemmMicroM;
  const int col0 = (tid % kThreadCols) * kGemmMicroN;

  float acc[kGemmMicroM][kGemmMicroN];
#pragma unroll
  for (int i = 0; i < kGemmMicroM; ++i) {
#pragma unroll
    for (int j = 0; j < kGemmMicroN; ++j) acc[i][j] = 0.0f;
  }

  int slice = 0;
  float (*sa)[kGemmBlockM] = reinterpret_cast<float (*)[kGemmBlockM]>(
      s_gemm + static_cast<std::ptrdiff_t>(slice) * kGemmSliceK * kGemmBlockM);
  float (*sb)[kGemmBlockN] = reinterpret_cast<float (*)[kGemmBlockN]>(
      s_gemm + 2 * kGemmSliceK * kGemmBlockM +
      static_cast<std::ptrdiff_t>(slice) * kGemmSliceK * kGemmBlockN);
  LoadSlice(ap, bp, op.lda, op.ldb, op.transpose_a, op.transpose_b, m0, n0,
            op.m, op.n, op.k, 0, tid, sa, sb);
  __syncthreads();

  for (int k0 = 0; k0 < op.k; k0 += kGemmSliceK) {
    if (k0 + kGemmSliceK < op.k) {
      float (*na)[kGemmBlockM] = reinterpret_cast<float (*)[kGemmBlockM]>(
          s_gemm +
          static_cast<std::ptrdiff_t>(slice ^ 1) * kGemmSliceK * kGemmBlockM);
      float (*nb)[kGemmBlockN] = reinterpret_cast<float (*)[kGemmBlockN]>(
          s_gemm + 2 * kGemmSliceK * kGemmBlockM +
          static_cast<std::ptrdiff_t>(slice ^ 1) * kGemmSliceK * kGemmBlockN);
      LoadSlice(ap, bp, op.lda, op.ldb, op.transpose_a, op.transpose_b, m0, n0,
                op.m, op.n, op.k, k0 + kGemmSliceK, tid, na, nb);
    }
    // Hoisted row and column bases: the per-k address is then a constant offset
    // from one register, so the slice's loads cost no extra address arithmetic.
    const float* a_row = &sa[0][row0];
    const float* b_col = &sb[0][col0];
// Two k at a time: 64 independent FMAs is enough to cover the FMA latency, and
// unrolling the whole slice pushes the accumulators past the register budget
// `__launch_bounds__(256, 3)` allows and spills.
#pragma unroll 2
    for (int kk = 0; kk < kGemmSliceK; ++kk) {
      // The 32 accumulators of the 8 by 4 micro-tile: the eight rows of this
      // thread against the four columns of this thread, one statement each, so
      // every one of them stays a register.
      const FusedQuad a0 = LoadQuad(a_row + kk * kGemmBlockM);
      const FusedQuad a1 = LoadQuad(a_row + kk * kGemmBlockM + 4);
      const FusedQuad bv = LoadQuad(b_col + kk * kGemmBlockN);
      acc[0][0] = fmaf(a0.e[0], bv.e[0], acc[0][0]);
      acc[0][1] = fmaf(a0.e[0], bv.e[1], acc[0][1]);
      acc[0][2] = fmaf(a0.e[0], bv.e[2], acc[0][2]);
      acc[0][3] = fmaf(a0.e[0], bv.e[3], acc[0][3]);
      acc[1][0] = fmaf(a0.e[1], bv.e[0], acc[1][0]);
      acc[1][1] = fmaf(a0.e[1], bv.e[1], acc[1][1]);
      acc[1][2] = fmaf(a0.e[1], bv.e[2], acc[1][2]);
      acc[1][3] = fmaf(a0.e[1], bv.e[3], acc[1][3]);
      acc[2][0] = fmaf(a0.e[2], bv.e[0], acc[2][0]);
      acc[2][1] = fmaf(a0.e[2], bv.e[1], acc[2][1]);
      acc[2][2] = fmaf(a0.e[2], bv.e[2], acc[2][2]);
      acc[2][3] = fmaf(a0.e[2], bv.e[3], acc[2][3]);
      acc[3][0] = fmaf(a0.e[3], bv.e[0], acc[3][0]);
      acc[3][1] = fmaf(a0.e[3], bv.e[1], acc[3][1]);
      acc[3][2] = fmaf(a0.e[3], bv.e[2], acc[3][2]);
      acc[3][3] = fmaf(a0.e[3], bv.e[3], acc[3][3]);
      acc[4][0] = fmaf(a1.e[0], bv.e[0], acc[4][0]);
      acc[4][1] = fmaf(a1.e[0], bv.e[1], acc[4][1]);
      acc[4][2] = fmaf(a1.e[0], bv.e[2], acc[4][2]);
      acc[4][3] = fmaf(a1.e[0], bv.e[3], acc[4][3]);
      acc[5][0] = fmaf(a1.e[1], bv.e[0], acc[5][0]);
      acc[5][1] = fmaf(a1.e[1], bv.e[1], acc[5][1]);
      acc[5][2] = fmaf(a1.e[1], bv.e[2], acc[5][2]);
      acc[5][3] = fmaf(a1.e[1], bv.e[3], acc[5][3]);
      acc[6][0] = fmaf(a1.e[2], bv.e[0], acc[6][0]);
      acc[6][1] = fmaf(a1.e[2], bv.e[1], acc[6][1]);
      acc[6][2] = fmaf(a1.e[2], bv.e[2], acc[6][2]);
      acc[6][3] = fmaf(a1.e[2], bv.e[3], acc[6][3]);
      acc[7][0] = fmaf(a1.e[3], bv.e[0], acc[7][0]);
      acc[7][1] = fmaf(a1.e[3], bv.e[1], acc[7][1]);
      acc[7][2] = fmaf(a1.e[3], bv.e[2], acc[7][2]);
      acc[7][3] = fmaf(a1.e[3], bv.e[3], acc[7][3]);
    }
    __syncthreads();
    slice ^= 1;
    sa = reinterpret_cast<float (*)[kGemmBlockM]>(
        s_gemm +
        static_cast<std::ptrdiff_t>(slice) * kGemmSliceK * kGemmBlockM);
    sb = reinterpret_cast<float (*)[kGemmBlockN]>(
        s_gemm + 2 * kGemmSliceK * kGemmBlockM +
        static_cast<std::ptrdiff_t>(slice) * kGemmSliceK * kGemmBlockN);
  }

  // The store, with the phase's epilogue folded in.
#pragma unroll
  for (int i = 0; i < kGemmMicroM; ++i) {
    const int gm = m0 + row0 + i;
    if (gm >= op.m) break;
    float* row = cp + static_cast<std::int64_t>(gm) * op.ldc + n0 + col0;
    const float* fold_row =
        foldp != nullptr
            ? foldp + static_cast<std::int64_t>(gm) * op.fold_ld + n0 + col0
            : nullptr;
#pragma unroll
    for (int j = 0; j < kGemmMicroN; ++j) {
      if (n0 + col0 + j >= op.n) continue;
      float value = acc[i][j];
      if (op.epilogue == FusedEpilogue::kFoldSquare) {
        value = op.cb * fold_row[j] + op.cc * value;
      } else if (op.epilogue == FusedEpilogue::kFoldIterate) {
        value = op.ca * fold_row[j] + value;
      }
      row[j] = value;
    }
  }
}

// The iterate the Polar Express loop leaves its answer in: phase 10 writes the
// partner buffer, so the parity of `ns_steps` picks the one the post-loop
// phases read.
__device__ __forceinline__ ComputeType* FusedCurrentIterate(
    const FusedPlan& p) {
  return (p.ns_steps % 2 == 0) ? p.x : p.xn;
}
// The iterate and its partner at the start of Polar Express iteration `iter`.
__device__ __forceinline__ void FusedIteratePair(const FusedPlan& p, int iter,
                                                 ComputeType** cur,
                                                 ComputeType** nxt) {
  *cur = (iter % 2 == 0) ? p.x : p.xn;
  *nxt = (iter % 2 == 0) ? p.xn : p.x;
}

// The op of one form of one Polar Express iteration, resolved from the group's
// extents and the plan's buffers. This is the port of `MuonPolarGemm` in
// dev/kernels/muon_fused_gemm.cu, on the device: the tall branch reduces over
// `rows` and the wide branch over `cols`, so the min-by-min extent is
// `min(rows, cols)` in both, and `side` is what names the square operand of a
// plain product.
__device__ __forceinline__ FusedGemmOp FusedPolarOp(const FusedPlan& p,
                                                    FusedForm form,
                                                    FusedSide side, int iter) {
  ComputeType* cur = nullptr;
  ComputeType* nxt = nullptr;
  FusedIteratePair(p, iter, &cur, &nxt);
  const float* x = reinterpret_cast<const float*>(cur);
  // `A` is the buffer of the square product and of its `a * a`, and the
  // combined `b` is the min-by-min operand of the application product: phase 10
  // multiplies by what phase 9 wrote, not by what phase 8 wrote.
  const float* square = reinterpret_cast<const float*>(p.a);
  const float* combined = reinterpret_cast<const float*>(p.b);
  const std::int64_t min_sq = static_cast<std::int64_t>(p.min_dim) * p.min_dim;

  FusedGemmOp op;
  op.ca = p.coeffs[iter][0];
  op.cb = p.coeffs[iter][1];
  op.cc = p.coeffs[iter][2];
  switch (form) {
    case FusedForm::kXtX:
      // A = X^T X, the tall branch: reductions over `rows`, so the product is
      // `cols` by `cols` and the stored `x` is read transposed on the left.
      op.m = p.cols;
      op.n = p.cols;
      op.k = p.rows;
      op.a = x;
      op.lda = p.cols;
      op.transpose_a = true;
      op.b = x;
      op.ldb = p.cols;
      op.c = reinterpret_cast<float*>(p.a);
      op.ldc = p.min_dim;
      op.stride_a = p.mat;
      op.stride_b = p.mat;
      op.stride_c = min_sq;
      break;
    case FusedForm::kXXt:
      // A = X X^T, the wide branch: reductions over `cols`, so the product is
      // `rows` by `rows` and the stored `x` is read transposed on the right.
      op.m = p.rows;
      op.n = p.rows;
      op.k = p.cols;
      op.a = x;
      op.lda = p.cols;
      op.b = x;
      op.ldb = p.cols;
      op.transpose_b = true;
      op.c = reinterpret_cast<float*>(p.a);
      op.ldc = p.min_dim;
      op.stride_a = p.mat;
      op.stride_b = p.mat;
      op.stride_c = min_sq;
      break;
    case FusedForm::kAA:
      // A2 = A A over the square min-by-min buffer, with the `cb * a + cc *
      // acc` epilogue: `b` is written from the accumulator, so nothing ever
      // holds `a * a` on its own.
      op.m = p.min_dim;
      op.n = p.min_dim;
      op.k = p.min_dim;
      op.a = square;
      op.lda = p.min_dim;
      op.b = square;
      op.ldb = p.min_dim;
      op.c = reinterpret_cast<float*>(p.b);
      op.ldc = p.min_dim;
      op.stride_a = min_sq;
      op.stride_b = min_sq;
      op.stride_c = min_sq;
      op.epilogue = FusedEpilogue::kFoldSquare;
      op.fold = square;
      op.fold_ld = p.min_dim;
      break;
    case FusedForm::kXB:
    case FusedForm::kBX: {
      // prod = X B (tall) or prod = B X (wide), where `B` is the combined
      // matrix phase 9 wrote. Both are plain operand times plain operand, so
      // both transpose flags are clear in both and only the side the square
      // operand sits on separates them.
      const bool square_left =
          form == FusedForm::kBX || side == FusedSide::kLeft;
      op.m = p.rows;
      op.n = p.cols;
      op.k = square_left ? p.rows : p.cols;
      op.a = square_left ? combined : x;
      op.lda = square_left ? p.min_dim : p.cols;
      op.b = square_left ? x : combined;
      op.ldb = square_left ? p.cols : p.min_dim;
      op.c = reinterpret_cast<float*>(nxt);
      op.ldc = p.cols;
      op.stride_a = square_left ? min_sq : p.mat;
      op.stride_b = square_left ? p.mat : min_sq;
      op.stride_c = p.mat;
      op.epilogue = FusedEpilogue::kFoldIterate;
      op.fold = x;
      op.fold_ld = p.cols;
      break;
    }
  }
  return op;
}

// One GEMM phase: block `b` of the grid takes tile item `b`, `b + gridDim`, and
// so on, so one persistent grid covers a batch of any size.
__device__ __forceinline__ void FusedGemmPhase(const FusedPlan& p,
                                               const FusedGemmOp& op,
                                               float* s_gemm) {
  const int tiles_m = (op.m + kGemmBlockM - 1) / kGemmBlockM;
  const int tiles_n = (op.n + kGemmBlockN - 1) / kGemmBlockN;
  const int per_matrix = tiles_m * tiles_n;
  const int items = p.num_params * per_matrix;
  for (int item = static_cast<int>(blockIdx.x); item < items;
       item += static_cast<int>(gridDim.x)) {
    const int matrix = item / per_matrix;
    const int within = item - matrix * per_matrix;
    const int tile_j = within % tiles_n;
    const int tile_i = within / tiles_n;
    FusedGemmTile(op, matrix, tile_i, tile_j, s_gemm);
  }
}

// --- Block-wide reductions -------------------------------------------------
//
// `BlockReduceSum` in device_utils.cuh declares its own static shared array per
// instantiation, and this kernel already spends its shared budget on the GEMM
// tile, so the two reductions below take their arrays from the kernel's
// scratch. Both run the same halving tree in the same order, which is what
// keeps the stages that use them bit exact against the multi-launch path: the
// double one reproduces the whole-matrix partials, and the float one reproduces
// the row equilibration, the mean squares, and the factored second moment.
__device__ __forceinline__ double BlockSumDouble(double value,
                                                 double* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  __syncthreads();
  scratch[tid] = value;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) >> 1; offset > 0;
       offset >>= 1) {
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ float BlockSumFloat(float value, float* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  __syncthreads();
  scratch[tid] = value;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) >> 1; offset > 0;
       offset >>= 1) {
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    __syncthreads();
  }
  return scratch[0];
}

// --- The phases ------------------------------------------------------------
//
// One device body for all 16 phases, over the flat work-item index. `iter` is
// the Polar Express iteration the GEMM phases belong to.
__device__ void FusedPhase(const FusedPlan& p, int phase, int iter,
                           float* s_gemm, double* s_double, float* s_float) {
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const long long element_stride =
      static_cast<long long>(gridDim.x) * blockDim.x;
  // The iterate the post-loop phases work on. Phases 1 to 7 own `x` (the
  // momentum output) and never look at the parity of the loop.
  const ComputeType* cur = FusedCurrentIterate(p);

  switch (phase) {
    case kPhaseMomentum: {
      // buf1[i] += (1 - momentum) * (grad[i] - buf1[i]); the iterate carries
      // either the Nesterov-accelerated gradient or the plain first moment.
      const float scale = p.grad_scale.host();
      if (p.vector) {
        const long long vectors = p.total / kVectorElems;
        ComputeVector* dst = reinterpret_cast<ComputeVector*>(p.x);
        const FloatVector* grads =
            reinterpret_cast<const FloatVector*>(p.grads);
        FloatVector* buf1 = reinterpret_cast<FloatVector*>(p.buf1);
        for (long long v =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             v < vectors; v += element_stride) {
          ComputeVector value = dst[v];
          const FloatVector* gbase = grads + v * kFloatPerVector;
          FloatVector* bbase = buf1 + v * kFloatPerVector;
          FloatVector next[kFloatPerVector];
#pragma unroll
          for (int j = 0; j < kFloatPerVector; ++j) {
            const FloatVector g = gbase[j];
            const FloatVector b = bbase[j];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              const float gr = g.e[s] * scale;
              const float bv = b.e[s];
              const float nb = bv + (1.0f - p.momentum) * (gr - bv);
              next[j].e[s] = nb;
              value.e[4 * j + s] = ToComputeDev(
                  p.nesterov ? (1.0f - p.momentum) * gr + p.momentum * nb : nb);
            }
          }
#pragma unroll
          for (int j = 0; j < kFloatPerVector; ++j) bbase[j] = next[j];
          dst[v] = value;
        }
      } else {
        for (long long i =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             i < p.total; i += element_stride) {
          const float gr = p.grads[i] * scale;
          const float b = p.buf1[i] + (1.0f - p.momentum) * (gr - p.buf1[i]);
          p.buf1[i] = b;
          p.x[i] = ToComputeDev(
              p.nesterov ? (1.0f - p.momentum) * gr + p.momentum * b : b);
        }
      }
      break;
    }
    case kPhaseFrobPartial:
    case kPhaseDivisorPartial:
    case kPhaseRenormPartial: {
      // Partials of one matrix, one per block item, so a matrix is reduced by
      // `partial_blocks` blocks at once. The accumulation is double and the
      // chunk bounds are the multi-launch path's, so a partial is the same
      // number to the last bit.
      const ComputeType* src = phase == kPhaseRenormPartial ? cur : p.x;
      const int items = p.num_params * p.partial_blocks;
      const long long chunk = (p.mat + p.partial_blocks - 1) / p.partial_blocks;
      for (int item = static_cast<int>(blockIdx.x); item < items;
           item += static_cast<int>(gridDim.x)) {
        const int matrix = item / p.partial_blocks;
        const int block = item - matrix * p.partial_blocks;
        const ComputeType* xm = src + static_cast<long long>(matrix) * p.mat;
        const long long begin = static_cast<long long>(block) * chunk;
        const long long end = begin + chunk < p.mat ? begin + chunk : p.mat;
        double local = 0.0;
        for (long long i = begin + tid; i < end; i += threads) {
          const double v = AsFloatDev(xm[i]);
          local += v * v;
        }
        const double sum = BlockSumDouble(local, s_double);
        if (tid == 0) {
          p.partials[static_cast<long long>(matrix) * p.partial_blocks +
                     block] = sum;
        }
      }
      break;
    }
    case kPhaseFrobFinish:
    case kPhaseDivisorFinish:
    case kPhaseRenormFinish: {
      // The statistic of every matrix of the group from its partials: the
      // Frobenius norm, the MuonEq divisor, or the Muon+ scale. One thread per
      // matrix, because the work is `num_params` short sums.
      for (int matrix = static_cast<int>(blockIdx.x) * threads + tid;
           matrix < p.num_params; matrix += static_cast<int>(element_stride)) {
        double sum = 0.0;
        for (int block = 0; block < p.partial_blocks; ++block) {
          sum += p.partials[static_cast<long long>(matrix) * p.partial_blocks +
                            block];
        }
        const float norm = sqrtf(static_cast<float>(sum));
        float value = norm;
        if (phase == kPhaseDivisorFinish) {
          value = norm * 1.01f + 1e-6f;
        } else if (phase == kPhaseRenormFinish) {
          value = p.renorm_target / fmaxf(norm, 1e-6f);
        }
        p.stat[matrix] = value;
      }
      break;
    }
    case kPhaseEquilibrate: {
      // MuonEq: each row to the mean row norm, one block per (matrix, row).
      const int items = p.num_params * p.rows;
      for (int item = static_cast<int>(blockIdx.x); item < items;
           item += static_cast<int>(gridDim.x)) {
        const int matrix = item / p.rows;
        const int row_index = item - matrix * p.rows;
        ComputeType* row = p.x + static_cast<long long>(matrix) * p.mat +
                           static_cast<long long>(row_index) * p.cols;
        float local = 0.0f;
        for (int c = tid; c < p.cols; c += threads) {
          const float v = AsFloatDev(row[c]);
          local += v * v;
        }
        const float row_norm =
            fmaxf(sqrtf(BlockSumFloat(local, s_float)), 1e-6f);
        const float target = p.stat[matrix] / sqrtf(static_cast<float>(p.rows));
        const float s = target / row_norm;
        for (int c = tid; c < p.cols; c += threads) {
          row[c] = ToComputeDev(AsFloatDev(row[c]) * s);
        }
      }
      break;
    }
    case kPhaseDivisorScale:
    case kPhaseRenormScale: {
      // One per-matrix factor over the whole stack: the equilibration divisor
      // divides the momentum output before the loop, and the renormalisation
      // scale multiplies what the loop left.
      const bool divide = phase == kPhaseDivisorScale;
      ComputeType* dst = divide ? p.x : FusedCurrentIterate(p);
      if (p.vector) {
        ComputeVector* vectors = reinterpret_cast<ComputeVector*>(dst);
        const long long vector_count = p.total / kVectorElems;
        for (long long v =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             v < vector_count; v += element_stride) {
          // `mat` is a whole number of vectors, so one vector never spans two
          // matrices and one factor serves all of its elements.
          const float f = p.stat[(v * kVectorElems) / p.mat];
          ComputeVector value = vectors[v];
#pragma unroll
          for (int k = 0; k < kVectorElems; ++k) {
            const float e = AsFloatDev(value.e[k]);
            value.e[k] = ToComputeDev(divide ? e / f : e * f);
          }
          vectors[v] = value;
        }
      } else {
        for (long long i =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             i < p.total; i += element_stride) {
          const float f = p.stat[i / p.mat];
          const float v = AsFloatDev(dst[i]);
          dst[i] = ToComputeDev(divide ? v / f : v * f);
        }
      }
      break;
    }
    case kPhaseSquare: {
      // A = X^T X (tall) or X X^T (wide).
      const FusedGemmOp op =
          FusedPolarOp(p, p.tall ? FusedForm::kXtX : FusedForm::kXXt,
                       FusedSide::kRight, iter);
      FusedGemmPhase(p, op, s_gemm);
      break;
    }
    case kPhaseSquareSquare: {
      // A2 = A A with the `b = cb * a + cc * acc` epilogue: the `a2` buffer and
      // the pass over `min_dim^2` that filled it are gone.
      const FusedGemmOp op =
          FusedPolarOp(p, FusedForm::kAA, FusedSide::kRight, iter);
      FusedGemmPhase(p, op, s_gemm);
      break;
    }
    case kPhaseApplySquare: {
      // prod = X B (tall) or B X (wide), with the `xnext = ca * x + prod`
      // epilogue: the combined iterate is written straight into the partner
      // buffer, so the `prod` buffer is the ping-pong partner and no third
      // full-size buffer exists.
      const FusedGemmOp op =
          FusedPolarOp(p, p.tall ? FusedForm::kXB : FusedForm::kBX,
                       p.tall ? FusedSide::kRight : FusedSide::kLeft, iter);
      FusedGemmPhase(p, op, s_gemm);
      break;
    }
    case kPhaseMeanSquare: {
      // NorMuon's first statistic: one mean square per row, or one per column.
      // A row is reduced by one block per (matrix, row); a column by one thread
      // per column, which keeps the reads of a warp on consecutive columns of
      // the same row.
      if (p.reduce_cols) {
        const int items = p.num_params * p.rows;
        for (int item = static_cast<int>(blockIdx.x); item < items;
             item += static_cast<int>(gridDim.x)) {
          const int matrix = item / p.rows;
          const int row_index = item - matrix * p.rows;
          const ComputeType* row = cur +
                                   static_cast<long long>(matrix) * p.mat +
                                   static_cast<long long>(row_index) * p.cols;
          float local = 0.0f;
          for (int c = tid; c < p.cols; c += threads) {
            const float v = AsFloatDev(row[c]);
            local += v * v;
          }
          const float sum = BlockSumFloat(local, s_float);
          if (tid == 0) {
            p.vmean[static_cast<long long>(matrix) * p.rows + row_index] =
                sum / static_cast<float>(p.cols);
          }
        }
      } else {
        const int column_blocks = (p.cols + threads - 1) / threads;
        const int items = p.num_params * column_blocks;
        for (int item = static_cast<int>(blockIdx.x); item < items;
             item += static_cast<int>(gridDim.x)) {
          const int matrix = item / column_blocks;
          const int column_block = item - matrix * column_blocks;
          const ComputeType* xm = cur + static_cast<long long>(matrix) * p.mat;
          const int c = column_block * threads + tid;
          if (c >= p.cols) continue;
          float local = 0.0f;
          for (int r = 0; r < p.rows; ++r) {
            const float v =
                AsFloatDev(xm[static_cast<long long>(r) * p.cols + c]);
            local += v * v;
          }
          p.vmean[static_cast<long long>(matrix) * p.cols + c] =
              local / static_cast<float>(p.rows);
        }
      }
      break;
    }
    case kPhaseMoments: {
      // The factored second moment and the per-index step of one matrix, one
      // block per matrix over `red_index` entries, in the composition's order:
      // `v_norm` from the pre-update mean squares, then `sum_scaled` from the
      // same values.
      for (int matrix = static_cast<int>(blockIdx.x); matrix < p.num_params;
           matrix += static_cast<int>(gridDim.x)) {
        float* vmean = p.vmean + static_cast<long long>(matrix) * p.red_index;
        float* second = p.buf2 + static_cast<long long>(matrix) * p.red_index;
        const float size = static_cast<float>(p.red_size);

        float local_sum = 0.0f;
        for (int i = tid; i < p.red_index; i += threads) local_sum += vmean[i];
        const float sum_vmean = BlockSumFloat(local_sum, s_float);
        const float v_norm = sqrtf(sum_vmean * size);

        float local_scaled = 0.0f;
        for (int i = tid; i < p.red_index; i += threads) {
          const float mean = vmean[i];
          const float sm = second[i] + (1.0f - p.beta2) * (mean - second[i]);
          second[i] = sm;
          const float step = 1.0f / sqrtf(fmaxf(sm, 1e-10f));
          vmean[i] = step;
          local_scaled += (mean * size) * step * step;
        }
        const float sum_scaled = BlockSumFloat(local_scaled, s_float);
        const float denom = fmaxf(sqrtf(sum_scaled), 1e-10f);
        if (tid == 0) p.scale[matrix] = v_norm / denom;
      }
      break;
    }
    case kPhaseNorMuonScale: {
      // The NorMuon scale along the reduction axis, with the cautious update as
      // its epilogue. The fold is bit exact because neither stage accumulates:
      // the scaled value is one rounded product, and the update reads it.
      if (p.vector) {
        const long long vector_count = p.total / kVectorElems;
        const ComputeVector* src = reinterpret_cast<const ComputeVector*>(cur);
        ComputeVector* dst = reinterpret_cast<ComputeVector*>(p.values);
        FloatVector* master = reinterpret_cast<FloatVector*>(p.master);
        for (long long v =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             v < vector_count; v += element_stride) {
          const ComputeVector value = src[v];
          const long long first = v * kVectorElems;
          const int matrix = static_cast<int>(first / p.mat);
          const long long within =
              first - static_cast<long long>(matrix) * p.mat;
          const float global = p.scale[matrix];
          FloatVector* mbase = master + v * kFloatPerVector;
          FloatVector next[kFloatPerVector];
          ComputeVector result;
#pragma unroll
          for (int j = 0; j < kFloatPerVector; ++j) {
            const FloatVector m = mbase[j];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              const long long w = within + 4 * j + s;
              const long long index = p.reduce_cols ? w / p.cols : w % p.cols;
              const float factor =
                  p.vmean[static_cast<long long>(matrix) * p.red_index +
                          index] *
                  global;
              const float gv = AsFloatDev(value.e[4 * j + s]) * factor;
              const float pv = m.e[s];
              // Cautious decay: only shrink a parameter that agrees in sign
              // with the update direction.
              const float decay =
                  (gv * pv >= 0.0f) ? p.lr * p.weight_decay * pv : 0.0f;
              const float pi = pv - p.lr * gv - decay;
              next[j].e[s] = pi;
              result.e[4 * j + s] = ToComputeDev(pi);
            }
          }
#pragma unroll
          for (int j = 0; j < kFloatPerVector; ++j) mbase[j] = next[j];
          dst[v] = result;
        }
      } else {
        for (long long i =
                 static_cast<long long>(blockIdx.x) * blockDim.x + tid;
             i < p.total; i += element_stride) {
          const long long matrix = i / p.mat;
          const long long within = i - matrix * p.mat;
          const long long index =
              p.reduce_cols ? within / p.cols : within % p.cols;
          const float factor =
              p.vmean[matrix * p.red_index + index] * p.scale[matrix];
          const float gv = AsFloatDev(cur[i]) * factor;
          const float pv = p.master[i];
          const float decay =
              (gv * pv >= 0.0f) ? p.lr * p.weight_decay * pv : 0.0f;
          const float pi = pv - p.lr * gv - decay;
          p.master[i] = pi;
          p.values[i] = ToComputeDev(pi);
        }
      }
      break;
    }
    default:
      break;
  }
}

// The grid barrier between two phase executions of the cooperative driver.
//
// `grid.sync()` is the design's barrier (section 4) and needs the launch to be
// cooperative, which is what the driver checks before it launches. At
// `ns_steps` 5 there are 27 of them per group, against the 39 launches the
// multi-launch path issues.
//
// The emulator compiles this file with the host compiler, which has no
// cooperative groups, and it runs blocks one at a time, so a grid barrier there
// could not be expressed and could not be reached: `tools/cuda_sim` answers 0
// for `cudaDevAttrCooperativeLaunch`, so an emulated build takes the phase
// driver. The barrier degrades to a block barrier so one device body still
// compiles in both builds.
__device__ __forceinline__ void FusedGridBarrier() {
#if defined(__CUDACC__)
  cooperative_groups::this_grid().sync();
#else
  __syncthreads();
#endif
}

// The fused device function: one phase execution, or a whole group's schedule
// when a cooperative launch covers it.
__global__ void __launch_bounds__(kGemmThreads, kGemmBlocksPerSm)
    MuonFusedKernel(FusedPlan plan) {
  // The GEMM tile of the two double-buffered k slices, and the two reduction
  // scratch arrays. Static shared memory, so a launch fixes nothing dynamic and
  // the occupancy query is asked for block size alone.
  __shared__ float s_gemm[kGemmSmemFloats];
  __shared__ double s_double[kReduceScalars];
  __shared__ float s_float[kReduceScalars];
  for (int e = 0; e < plan.exec_count; ++e) {
    // A phase execution reads what the previous one wrote across the whole
    // grid, and a launch boundary is the phase driver's barrier; inside one
    // launch this is it.
    if (e > 0) FusedGridBarrier();
    int phase = 0;
    int iter = 0;
    FusedDecodeExec(plan.exec_begin + e, plan.ns_steps, &phase, &iter);
    FusedPhase(plan, phase, iter, s_gemm, s_double, s_float);
  }
}

// --- The host side --------------------------------------------------------

// One cached allocation is `[x | xn | a | b]`; `MakePlan` also states the same
// layout for muon.cu's multi-launch path, which uses the four slices as
// `[x | prod | a | b]`. A vector-wise elementwise phase needs `total` to be a
// whole number of 16-byte groups and the per-matrix factor to be constant
// across one group, so `mat` must be one too.

// The blocks one phase wants when it is launched on its own. A block-granular
// phase takes one block per work item, capped so a shape with millions of rows
// still launches (a block strides over the rest); an elementwise phase takes
// one block per `kGemmThreads` items, like the family's vector grids.
constexpr long long kMaxPhaseBlocks = 1 << 20;

int Blocks(long long items, int threads) {
  if (items <= 0) return 1;
  const long long blocks = (items + threads - 1) / threads;
  return static_cast<int>(blocks < kMaxPhaseBlocks ? blocks : kMaxPhaseBlocks);
}

int BlockItems(long long items) {
  if (items <= 0) return 1;
  return static_cast<int>(items < kMaxPhaseBlocks ? items : kMaxPhaseBlocks);
}

int PhaseGrid(const FusedPlan& p, int phase) {
  const long long elements = p.vector ? p.total / kVectorElems : p.total;
  switch (phase) {
    case kPhaseMomentum:
    case kPhaseDivisorScale:
    case kPhaseRenormScale:
    case kPhaseNorMuonScale:
      return Blocks(elements, kGemmThreads);
    case kPhaseFrobPartial:
    case kPhaseDivisorPartial:
    case kPhaseRenormPartial:
      return BlockItems(static_cast<long long>(p.num_params) *
                        p.partial_blocks);
    case kPhaseFrobFinish:
    case kPhaseDivisorFinish:
    case kPhaseRenormFinish:
      return Blocks(p.num_params, kGemmThreads);
    case kPhaseEquilibrate:
      return BlockItems(static_cast<long long>(p.num_params) * p.rows);
    case kPhaseMeanSquare:
      if (p.reduce_cols) {
        return BlockItems(static_cast<long long>(p.num_params) * p.rows);
      }
      return BlockItems(static_cast<long long>(p.num_params) *
                        ((p.cols + kGemmThreads - 1) / kGemmThreads));
    case kPhaseMoments:
      return BlockItems(p.num_params);
    case kPhaseSquare:
    case kPhaseSquareSquare:
    case kPhaseApplySquare: {
      // One block per (matrix, tile). The square product and `a * a` are both
      // min by min; the application product is the full rows by cols.
      const bool apply = phase == kPhaseApplySquare;
      const int m = apply ? p.rows : p.min_dim;
      const int n = apply ? p.cols : p.min_dim;
      const long long tiles =
          static_cast<long long>((m + kGemmBlockM - 1) / kGemmBlockM) *
          ((n + kGemmBlockN - 1) / kGemmBlockN);
      return BlockItems(static_cast<long long>(p.num_params) * tiles);
    }
    default:
      return 1;
  }
}

// The plan for one group, from the caller's buffers. `workspace` is sliced as
// `[x | xn | a | b]` and `scratch` as `[stat | vmean | scale]`.
FusedPlan MakePlan(const MuonParams& params, const float* stacked_grads,
                   float* stacked_master, ComputeType* stacked_values,
                   float* buf1, float* buf2, ComputeType* workspace,
                   float* scratch, double* partials) {
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  const std::int64_t mat = static_cast<std::int64_t>(rows) * cols;
  const std::int64_t total = static_cast<std::int64_t>(num_params) * mat;
  const int min_dim = rows < cols ? rows : cols;
  const std::int64_t min_total =
      static_cast<std::int64_t>(num_params) * min_dim * min_dim;
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);

  FusedPlan plan;
  plan.num_params = num_params;
  plan.rows = rows;
  plan.cols = cols;
  plan.mat = mat;
  plan.total = total;
  plan.min_dim = min_dim;
  plan.min_total = min_total;
  plan.red_index = reduce_cols ? rows : cols;
  plan.red_size = reduce_cols ? cols : rows;
  plan.reduce_cols = reduce_cols;
  plan.tall = rows > cols;
  plan.nesterov = params.nesterov;
  plan.partial_blocks = MuonPartialBlocks(mat);
  plan.ns_steps = std::min(std::max(params.ns_steps, 0), 5);
  plan.momentum = params.momentum;
  plan.beta2 = params.beta2;
  plan.lr = params.lr;
  plan.weight_decay = params.weight_decay;
  plan.renorm_target = sqrtf(static_cast<float>(min_dim));
  plan.grad_scale = params.grad_scale;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) plan.coeffs[i][j] = kPolarCoeffs[i][j];
  }

  plan.x = workspace;
  plan.xn = plan.x + total;
  plan.a = plan.xn + total;
  plan.b = plan.a + min_total;
  plan.grads = stacked_grads;
  plan.master = stacked_master;
  plan.buf1 = buf1;
  plan.buf2 = buf2;
  plan.values = stacked_values;
  plan.stat = scratch;
  plan.vmean =
      plan.stat + MuonScratchOffset(static_cast<std::size_t>(num_params));
  plan.scale =
      plan.vmean +
      MuonScratchOffset(static_cast<std::size_t>(num_params) * plan.red_index);
  plan.partials = partials;

  // One flag for every elementwise phase: a 16-byte vector per thread, which is
  // the width the rest of the family moves. `mat` must be a whole number of
  // them because the per-matrix factor phases divide by it.
  plan.vector =
      VectorReady(total, {plan.x, plan.xn, plan.grads, plan.buf1, plan.master,
                          plan.values, plan.stat, plan.vmean, plan.scale}) &&
      mat % kVectorElems == 0;
  return plan;
}

// The one line a refused cooperative launch is allowed. A refusal is the
// expected answer on a card whose grid the fused kernel cannot fill, so the
// launch site stays silent, the fallback reports it once, and a training loop
// does not print it every step.
void LogCooperativeFallback(CooperativeLaunchOutcome outcome) {
  static bool logged = false;
  if (logged) return;
  logged = true;
  std::fprintf(stderr,
               "muon fused: cooperative launch %s; the phase driver runs the "
               "update\n",
               cuda_backend::CooperativeLaunchOutcomeName(outcome));
}

// One phase execution per launch, each grid sized for its phase. This is the
// simulator's path, the fallback for a card without a cooperative launch, and
// the phase-bisection tool: a phase gate reads one launch's cost at a time.
void RunPhaseDriver(FusedPlan plan) {
  const int executions = MuonFusedExecutions(plan.ns_steps);
  for (int exec = 0; exec < executions; ++exec) {
    int phase = 0;
    int iter = 0;
    FusedDecodeExec(exec, plan.ns_steps, &phase, &iter);
    (void)iter;
    plan.exec_begin = exec;
    plan.exec_count = 1;
    // The grid is sized for the phase this launch runs, decoded here so the
    // driver and the kernel read one schedule.
    cuda_backend::Launch(
        MuonFusedKernel,
        dim3(static_cast<unsigned int>(PhaseGrid(plan, phase))),
        dim3(static_cast<unsigned int>(kGemmThreads)), 0, plan);
  }
}

// One launch for the whole group, with a grid barrier between the phase
// executions, and the phase driver as the fallback. The grid is the
// resident-block query, so the launch asks for exactly what the device can keep
// resident at once -- which is what makes the barriers sound.
MuonFusedRun RunCooperativeDriver(const FusedPlan& plan,
                                  const MuonFusedOptions& options) {
#if defined(__CUDACC__)
  const std::size_t shared = 0;
  const int resident = cuda_backend::ResidentBlocks(
      reinterpret_cast<const void*>(MuonFusedKernel), kGemmThreads, shared);
  if (resident <= 0) {
    // No block of this kernel fits: the phase driver still runs, one launch at
    // a time, so the update happens.
    RunPhaseDriver(plan);
    MuonFusedRun run;
    run.driver = MuonFusedDriver::kPhase;
    return run;
  }
  const long long grid =
      static_cast<long long>(resident) * options.grid_multiplier;
  FusedPlan cooperative = plan;
  cooperative.exec_begin = 0;
  cooperative.exec_count = MuonFusedExecutions(plan.ns_steps);
  const CooperativeLaunchOutcome outcome = cuda_backend::LaunchCooperative(
      MuonFusedKernel, dim3(static_cast<unsigned int>(grid)),
      dim3(static_cast<unsigned int>(kGemmThreads)), shared, cooperative);
  if (outcome == CooperativeLaunchOutcome::kLaunched) {
    MuonFusedRun run;
    run.driver = MuonFusedDriver::kCooperative;
    return run;
  }
  // One log line, then the driver the device will take. The device body, the
  // workspace, and the arithmetic are identical, so the answer is the phase
  // driver's.
  LogCooperativeFallback(outcome);
  RunPhaseDriver(plan);
  MuonFusedRun run;
  run.driver = MuonFusedDriver::kPhase;
  run.fell_back = true;
  return run;
#else
  // The emulated build has no cooperative launch: the mock answers 0 for
  // `cudaDevAttrCooperativeLaunch`, so `MuonFusedUpdate` never asks for this
  // driver there. The source keeps one body nevertheless.
  (void)options;
  RunPhaseDriver(plan);
  MuonFusedRun run;
  run.driver = MuonFusedDriver::kPhase;
  return run;
#endif
}

}  // namespace

int MuonPartialBlocks(long long mat) {
  long long blocks = (mat + 8191) / 8192;
  if (blocks < 1) blocks = 1;
  if (blocks > 32) blocks = 32;
  return static_cast<int>(blocks);
}

std::size_t MuonFusedWorkspaceElements(int num_params, int rows, int cols) {
  if (num_params <= 0 || rows <= 0 || cols <= 0) return 0;
  const std::size_t mat = static_cast<std::size_t>(rows) * cols;
  const std::size_t total = static_cast<std::size_t>(num_params) * mat;
  const std::size_t min_dim =
      static_cast<std::size_t>(rows < cols ? rows : cols);
  const std::size_t min_total =
      static_cast<std::size_t>(num_params) * min_dim * min_dim;
  return 2 * total + 2 * min_total;
}

std::size_t MuonFusedScratchFloats(int num_params, int rows, int cols) {
  if (num_params <= 0 || rows <= 0 || cols <= 0) return 0;
  // The mean squares of the larger extent settle the size, so one scratch
  // allocation serves both `red_dim` values of a group (the smaller one simply
  // leaves the tail unused) and neither path has to agree on the rule.
  const std::size_t red_index =
      static_cast<std::size_t>(rows > cols ? rows : cols);
  const std::size_t stat = static_cast<std::size_t>(num_params);
  return MuonScratchOffset(stat) +
         MuonScratchOffset(static_cast<std::size_t>(num_params) * red_index) +
         MuonScratchOffset(stat);
}

std::size_t MuonFusedPartialDoubles(int num_params, int rows, int cols) {
  if (num_params <= 0 || rows <= 0 || cols <= 0) return 0;
  const long long mat = static_cast<long long>(rows) * cols;
  return static_cast<std::size_t>(num_params) * MuonPartialBlocks(mat);
}

const char* MuonFusedDriverName(MuonFusedDriver driver) {
  switch (driver) {
    case MuonFusedDriver::kAuto:
      return "auto";
    case MuonFusedDriver::kPhase:
      return "phase";
    case MuonFusedDriver::kCooperative:
    default:
      return "cooperative";
  }
}

int MuonFusedExecutions(int ns_steps) {
  const int steps = std::min(std::max(ns_steps, 0), 5);
  return kExecsBefore + 3 * steps + kExecsAfter;
}

int MuonFusedPhaseCount() { return kPhaseCount; }

MuonFusedRun MuonFusedUpdate(const MuonParams& params,
                             const float* stacked_grads, float* stacked_master,
                             ComputeType* stacked_values, float* buf1,
                             float* buf2, ComputeType* workspace,
                             float* scratch, double* partials,
                             const MuonFusedOptions& options) {
  MuonFusedRun run;
  if (params.rows <= 0 || params.cols <= 0) return run;
  const FusedPlan plan =
      MakePlan(params, stacked_grads, stacked_master, stacked_values, buf1,
               buf2, workspace, scratch, partials);

  // The one launch the two drivers differ in. A device that answers 0 for
  // `cudaDevAttrCooperativeLaunch` -- the simulator does -- takes the phase
  // loop as its chosen path, so this is a selection and not a fallback and logs
  // nothing.
  const bool want_cooperative = options.driver != MuonFusedDriver::kPhase;
  if (!want_cooperative || !cuda_backend::SupportsCooperativeLaunch()) {
    RunPhaseDriver(plan);
    run.driver = MuonFusedDriver::kPhase;
    return run;
  }
  return RunCooperativeDriver(plan, options);
}

bool MuonFusedSelected(const MuonParams& params) {
  // The fused schedule is the fp32 one: its GEMM loads float4 quads out of
  // `ComputeType` storage, which is only the operand it means at the fp32 build
  // precision. A bf16 build keeps the multi-launch cuBLAS path for every group,
  // the way the promoted attention forward does.
  if (kComputeIsBf16) return false;
  if (params.rows <= 0 || params.cols <= 0) return false;
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  // A group takes the fused path when its Polar Express arithmetic is small
  // enough that the removed launch gaps and the folded elementwise passes pay
  // for the hand GEMM's rate gap (section 3's inequality), and the threshold is
  // the measured one (`kMuonFusedMaxElements`). On the host card no group of
  // either profile clears it, so the fused path is reachable -- the test drives
  // both of its drivers, and the benchmark measures both paths group by group
  // -- but it is the shipped default for no shape.
  const long long total =
      static_cast<long long>(num_params) * params.rows * params.cols;
  return total <= kMuonFusedMaxElements;
}

}  // namespace kernels
}  // namespace nanochat
