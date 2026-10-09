// CUDA Muon family: Nesterov momentum, MuonEq row equilibration, Polar Express
// orthogonalisation through batched cuBLAS GEMMs, Muon+ renormalisation,
// NorMuon variance reduction with a factored second moment, and the cautious
// decoupled weight decay. The math mirrors backends/cpu/kernels.cc exactly.
//
// The stacked parameters are [num_params, rows, cols]. The Polar Express
// iterations are the expensive part and run as strided-batched GEMMs over the
// whole stack through kernels::Gemm; the reduction stages (equilibration,
// renorm, NorMuon) run one block per matrix, and the pure elementwise stages
// run grid-stride over the stack.
//
// Layout notes for the tall (rows > cols) and wide (rows <= cols) cases:
//   * tall:  A = X^T X  (cols x cols), prod = X B
//   * wide:  A = X X^T  (rows x rows), prod = B X
// A, A^2, and B therefore occupy [num_params, min(rows, cols), min(rows,
// cols)].
//
// See docs/kernels.md and docs/model.md.

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::ToComputeDev;

// Polar Express coefficients (num_iters=5), from nanochat/optim.py / the Polar
// Express paper (arXiv:2505.16932). Kept in sync with backends/cpu/kernels.cc.
constexpr float kPolarCoeffs[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

// --- Momentum -------------------------------------------------------------

// buf1[i] += (1 - momentum) * (grad[i] - buf1[i]); x carries either the
// Nesterov-accelerated gradient or the plain first moment. The momentum
// buffer and the working matrix are fp32 even in the fp16 build, so the
// orthogonalization below never rounds through half (the loss otherwise rises
// again after a few hundred steps).
__global__ void MuonMomentumKernel(long long total, float momentum,
                                   bool nesterov,
                                   const ComputeType* __restrict__ grads,
                                   float* __restrict__ buf1,
                                   float* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float gr = AsFloatDev(grads[i]);
    const float b = buf1[i] + (1.0f - momentum) * (gr - buf1[i]);
    buf1[i] = b;
    x[i] = nesterov ? (1.0f - momentum) * gr + momentum * b : b;
  }
}

// --- MuonEq equilibration + Frobenius normalisation ------------------------

// Sum of squares of one matrix, reduced across the calling block.
__device__ __forceinline__ float MatrixSumSq(const float* xm, long long mat) {
  float local = 0.0f;
  for (long long i = threadIdx.x; i < mat; i += blockDim.x) {
    const float v = xm[i];
    local += v * v;
  }
  return BlockReduceSum(local);
}

// Frobenius norm of each matrix (one block per matrix), written to `frob`.
__global__ void MuonFrobKernel(int rows, int cols, const float* __restrict__ x,
                               float* __restrict__ frob) {
  const long long mat = static_cast<long long>(rows) * cols;
  const float* xm = x + static_cast<long long>(blockIdx.x) * mat;
  const float sum = MatrixSumSq(xm, mat);
  if (threadIdx.x == 0) frob[blockIdx.x] = sqrtf(sum);
}

// Rescale each row to the mean row norm (one block per (matrix, row)).
__global__ void MuonEquilibrateRowKernel(int rows, int cols,
                                         float* __restrict__ x,
                                         const float* __restrict__ frob) {
  const int p = blockIdx.y;
  const int r = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  float* row =
      x + static_cast<long long>(p) * mat + static_cast<long long>(r) * cols;
  float local = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = row[c];
    local += v * v;
  }
  const float row_norm = fmaxf(sqrtf(BlockReduceSum(local)), 1e-6f);
  const float target = frob[p] / sqrtf(static_cast<float>(rows));
  const float s = target / row_norm;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row[c] *= s;
  }
}

// Divide each matrix by 1.01 * ||X||_F + 1e-6 (one block per matrix).
__global__ void MuonEquilibrateDivKernel(int rows, int cols,
                                         float* __restrict__ x) {
  const long long mat = static_cast<long long>(rows) * cols;
  float* xm = x + static_cast<long long>(blockIdx.x) * mat;
  const float div = sqrtf(MatrixSumSq(xm, mat)) * 1.01f + 1e-6f;
  for (long long i = threadIdx.x; i < mat; i += blockDim.x) {
    xm[i] /= div;
  }
}

// --- Polar Express elementwise steps ---------------------------------------

// b = cb * a + cc * a2, elementwise over the stacked min x min buffers.
__global__ void MuonBKernel(long long count, float cb, float cc,
                            const float* __restrict__ a,
                            const float* __restrict__ a2,
                            float* __restrict__ b) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    b[i] = cb * a[i] + cc * a2[i];
  }
}

// x = ca * x + prod, elementwise over the stacked matrices.
__global__ void MuonAxpbyKernel(long long count, float ca,
                                float* __restrict__ x,
                                const float* __restrict__ prod) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] = ca * x[i] + prod[i];
  }
}

// --- Muon+ renormalisation -------------------------------------------------

// Snap the Frobenius norm of each matrix to sqrt(min(rows, cols)).
__global__ void MuonRenormKernel(int rows, int cols, float* __restrict__ x) {
  const long long mat = static_cast<long long>(rows) * cols;
  float* xm = x + static_cast<long long>(blockIdx.x) * mat;
  const int tid = threadIdx.x;

  float local = 0.0f;
  for (long long i = tid; i < mat; i += blockDim.x) {
    const float v = xm[i];
    local += v * v;
  }
  const float frob = BlockReduceSum(local);
  const float target = sqrtf(static_cast<float>(rows < cols ? rows : cols));
  const float scale = target / fmaxf(sqrtf(frob), 1e-6f);
  for (long long i = tid; i < mat; i += blockDim.x) {
    xm[i] *= scale;
  }
}

// --- NorMuon variance reduction --------------------------------------------

// One block per matrix. Computes the per-index mean square, updates the
// factored second moment, and rescales the matrix by the normalized step size.
// `reduce_cols` selects the reduction axis: rows (one entry per row) or
// columns (one entry per column), matching the CPU reference's `red_dim`.
__global__ void MuonNorMuonKernel(int rows, int cols, bool reduce_cols,
                                  float beta2, float* __restrict__ x,
                                  float* __restrict__ buf2,
                                  float* __restrict__ scratch) {
  const int p = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  float* xm = x + static_cast<long long>(p) * mat;
  const int red_index = reduce_cols ? rows : cols;
  const int red_size = reduce_cols ? cols : rows;
  float* second = buf2 + static_cast<long long>(p) * red_index;
  float* vmean = scratch + static_cast<long long>(p) * red_index;
  const int tid = threadIdx.x;

  // Pass 1: mean square along the reduction axis.
  for (int i = 0; i < red_index; ++i) {
    float local = 0.0f;
    if (reduce_cols) {
      const float* row = xm + static_cast<long long>(i) * cols;
      for (int c = tid; c < cols; c += blockDim.x) {
        const float v = row[c];
        local += v * v;
      }
    } else {
      for (int r = tid; r < rows; r += blockDim.x) {
        const float v = xm[static_cast<long long>(r) * cols + i];
        local += v * v;
      }
    }
    const float s = BlockReduceSum(local);
    if (tid == 0) vmean[i] = s / static_cast<float>(red_size);
  }
  __syncthreads();

  float lsum = 0.0f;
  for (int i = tid; i < red_index; i += blockDim.x) lsum += vmean[i];
  const float sum_vmean = BlockReduceSum(lsum);
  const float v_norm = sqrtf(sum_vmean * static_cast<float>(red_size));

  // Pass 2: update the second moment and derive the per-index step size.
  float lscaled = 0.0f;
  for (int i = tid; i < red_index; i += blockDim.x) {
    const float vm = vmean[i];
    const float sm = second[i] + (1.0f - beta2) * (vm - second[i]);
    second[i] = sm;
    const float step = 1.0f / sqrtf(fmaxf(sm, 1e-10f));
    vmean[i] = step;
    lscaled += (vm * static_cast<float>(red_size)) * step * step;
  }
  const float sum_scaled = BlockReduceSum(lscaled);
  const float denom = fmaxf(sqrtf(sum_scaled), 1e-10f);
  const float global_scale = v_norm / denom;

  // Pass 3: apply step_size * (v_norm / denom) along the reduction axis.
  if (reduce_cols) {
    for (int r = 0; r < rows; ++r) {
      const float s = vmean[r] * global_scale;
      float* row = xm + static_cast<long long>(r) * cols;
      for (int c = tid; c < cols; c += blockDim.x) {
        row[c] *= s;
      }
    }
  } else {
    for (int c = 0; c < cols; ++c) {
      const float s = vmean[c] * global_scale;
      for (int r = tid; r < rows; r += blockDim.x) {
        const long long idx = static_cast<long long>(r) * cols + c;
        xm[idx] *= s;
      }
    }
  }
}

// --- Cautious weight decay + parameter update ------------------------------

__global__ void MuonApplyKernel(long long total, float lr, float weight_decay,
                                float* __restrict__ master,
                                ComputeType* __restrict__ value,
                                const float* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float pv = master[i];
    const float gv = x[i];
    // Cautious decay: only shrink a parameter that agrees in sign with the
    // update direction.
    const float decay = (gv * pv >= 0.0f) ? lr * weight_decay * pv : 0.0f;
    const float pi = pv - lr * gv - decay;
    master[i] = pi;
    value[i] = ToComputeDev(pi);
  }
}

int GridSize(long long count, int threads) {
  const long long blocks = (count + threads - 1) / threads;
  constexpr long long kMaxBlocks = 512;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

// Cached working buffers for MuonUpdate. cudaMalloc/cudaFree synchronize the
// device and dominate the update when they run once per group per step, so the
// buffers are allocated once at the largest size any group needs and reused.
// The workspace is fp32 even in the fp16 build: the orthogonalization squares
// the working matrix, and rounding it through half loses the update.
float* g_muon_workspace = nullptr;
std::size_t g_muon_workspace_capacity = 0;
float* g_muon_scratch = nullptr;
std::size_t g_muon_scratch_capacity = 0;

void EnsureMuonWorkspace(std::size_t elements, std::size_t scratch_floats) {
  if (elements > g_muon_workspace_capacity) {
    if (g_muon_workspace != nullptr) Free(g_muon_workspace);
    g_muon_workspace = static_cast<float*>(Alloc(elements * sizeof(float)));
    g_muon_workspace_capacity = elements;
  }
  if (scratch_floats > g_muon_scratch_capacity) {
    if (g_muon_scratch != nullptr) Free(g_muon_scratch);
    g_muon_scratch = static_cast<float*>(Alloc(scratch_floats * sizeof(float)));
    g_muon_scratch_capacity = scratch_floats;
  }
}

}  // namespace

void MuonUpdate(const MuonParams& params, const ComputeType* stacked_grads,
                float* stacked_master, ComputeType* stacked_values, float* buf1,
                float* buf2) {
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;

  const long long mat = static_cast<long long>(rows) * cols;
  const long long total = static_cast<long long>(num_params) * mat;
  const int ns_steps = std::min(std::max(params.ns_steps, 0), 5);
  // `red_dim == -1` reasons over columns (one statistic per row); `-2` over
  // rows (one statistic per column); anything else follows the CPU default of
  // reducing the larger extent.
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);
  const int min_dim = rows < cols ? rows : cols;
  const long long min_sq = static_cast<long long>(min_dim) * min_dim;
  const long long min_total = min_sq * num_params;
  const bool tall = rows > cols;

  // Working matrix (momentum output, orthonormalised in place) and the
  // per-matrix GEMM operands. The workspace stays fp32 so the orthogonalization
  // does not lose the update; the elementwise kernels read the fp16 gradient
  // and write the fp32 result. The layout is [x | prod | a | a2 | b], from one
  // cached allocation.
  const int red_index = reduce_cols ? rows : cols;
  EnsureMuonWorkspace(static_cast<std::size_t>(2 * total + 3 * min_total),
                      static_cast<std::size_t>(num_params) * red_index);
  float* x = g_muon_workspace;
  float* prod = x + total;
  float* a = prod + total;
  float* a2 = a + min_total;
  float* b = a2 + min_total;
  float* scratch = g_muon_scratch;

  const int kThreads = 256;
  const int elem_grid = GridSize(total, kThreads);
  const int min_grid = GridSize(min_total, kThreads);

  cuda_backend::Launch(MuonMomentumKernel, dim3(elem_grid), dim3(kThreads), 0,
                       total, params.momentum, params.nesterov, stacked_grads,
                       buf1, x);
  cuda_backend::Launch(MuonFrobKernel, dim3(num_params), dim3(kThreads), 0,
                       rows, cols, x, scratch);
  cuda_backend::Launch(MuonEquilibrateRowKernel, dim3(rows, num_params),
                       dim3(kThreads), 0, rows, cols, x, scratch);
  cuda_backend::Launch(MuonEquilibrateDivKernel, dim3(num_params),
                       dim3(kThreads), 0, rows, cols, x);

  for (int it = 0; it < ns_steps; ++it) {
    const float ca = kPolarCoeffs[it][0];
    const float cb = kPolarCoeffs[it][1];
    const float cc = kPolarCoeffs[it][2];
    if (tall) {
      // A = X^T X  (cols x cols).
      GemmParams ga;
      ga.m = cols;
      ga.n = cols;
      ga.k = rows;
      ga.batch_count = num_params;
      ga.transpose_a = true;
      ga.stride_a = mat;
      ga.stride_b = mat;
      ga.stride_c = min_sq;
      GemmF32(GemmMode::kForward, ga, x, x, a);

      // A2 = A A.
      GemmParams ga2;
      ga2.m = cols;
      ga2.n = cols;
      ga2.k = cols;
      ga2.batch_count = num_params;
      ga2.stride_a = min_sq;
      ga2.stride_b = min_sq;
      ga2.stride_c = min_sq;
      GemmF32(GemmMode::kForward, ga2, a, a, a2);

      cuda_backend::Launch(MuonBKernel, dim3(min_grid), dim3(kThreads), 0,
                           min_total, cb, cc, a, a2, b);

      // prod = X B  (rows x cols).
      GemmParams gp;
      gp.m = rows;
      gp.n = cols;
      gp.k = cols;
      gp.batch_count = num_params;
      gp.stride_a = mat;
      gp.stride_b = min_sq;
      gp.stride_c = mat;
      GemmF32(GemmMode::kForward, gp, x, b, prod);
    } else {
      // A = X X^T  (rows x rows).
      GemmParams ga;
      ga.m = rows;
      ga.n = rows;
      ga.k = cols;
      ga.batch_count = num_params;
      ga.transpose_b = true;
      ga.stride_a = mat;
      ga.stride_b = mat;
      ga.stride_c = min_sq;
      GemmF32(GemmMode::kForward, ga, x, x, a);

      // A2 = A A.
      GemmParams ga2;
      ga2.m = rows;
      ga2.n = rows;
      ga2.k = rows;
      ga2.batch_count = num_params;
      ga2.stride_a = min_sq;
      ga2.stride_b = min_sq;
      ga2.stride_c = min_sq;
      GemmF32(GemmMode::kForward, ga2, a, a, a2);

      cuda_backend::Launch(MuonBKernel, dim3(min_grid), dim3(kThreads), 0,
                           min_total, cb, cc, a, a2, b);

      // prod = B X  (rows x cols).
      GemmParams gp;
      gp.m = rows;
      gp.n = cols;
      gp.k = rows;
      gp.batch_count = num_params;
      gp.stride_a = min_sq;
      gp.stride_b = mat;
      gp.stride_c = mat;
      GemmF32(GemmMode::kForward, gp, b, x, prod);
    }

    cuda_backend::Launch(MuonAxpbyKernel, dim3(elem_grid), dim3(kThreads), 0,
                         total, ca, x, prod);
  }

  cuda_backend::Launch(MuonRenormKernel, dim3(num_params), dim3(kThreads), 0,
                       rows, cols, x);
  cuda_backend::Launch(MuonNorMuonKernel, dim3(num_params), dim3(kThreads), 0,
                       rows, cols, reduce_cols, params.beta2, x, buf2, scratch);
  cuda_backend::Launch(MuonApplyKernel, dim3(elem_grid), dim3(kThreads), 0,
                       total, params.lr, params.weight_decay, stacked_master,
                       stacked_values, x);
}

}  // namespace kernels
}  // namespace nanochat
