// CUDA ANVIL family: twin-rail Nesterov momentum, the Frobenius-normalized
// whitening cascade (six quintic spectral maps through batched cuBLAS GEMMs),
// the per-lane energy equalizer, and the sign-aligned cautious update. The math
// mirrors backends/cpu/kernels.cc and backends/cuda/kernels/testing/optim_ref.h
// exactly. See docs/optimizer-anvil-design.md.
//
// The stacked parameters are [num_params, rows, cols]. The cascade is the
// expensive part and runs as strided-batched GEMMs over the whole stack through
// kernels::Gemm; the reduction stages (Frobenius normalization, lane
// equalizer) run one block per matrix, and the pure elementwise stages run
// grid-stride over the stack.
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

// ANVIL's six quintic spectral maps, from modded-nanogpt's
// `track_1_short/optim/anvil.py`. Kept in sync with backends/cpu/kernels.cc and
// backends/cuda/kernels/testing/optim_ref.h.
constexpr float kAnvilMaps[6][3] = {
    {3.923798038567f, -6.095026865488f, 3.905234618423f},
    {3.278126713798f, -3.328923386476f, 0.989127286973f},
    {3.505298394150f, -5.137358782410f, 1.968325560615f},
    {2.815058591845f, -3.685181239622f, 1.417196497642f},
    {2.245503932403f, -2.443826979899f, 0.963091710461f},
    {2.256537145403f, -2.166840097229f, 0.929501253245f},
};

// --- Twin-rail momentum + Nesterov lookahead -------------------------------

// velocity[i] and velocity[rail_stride + i] are the fast and slow rails; x
// carries the blended, Nesterov-accelerated gradient.
__global__ void AnvilMomentumKernel(long long total, long long rail_stride,
                                    float fast_beta, float slow_beta,
                                    float fast_weight, float momentum,
                                    bool nesterov,
                                    const ComputeType* __restrict__ grads,
                                    float* __restrict__ velocity,
                                    ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float gr = AsFloatDev(grads[i]);
    float* fast = velocity + i;
    float* slow = velocity + rail_stride + i;
    const float f = *fast + (1.0f - fast_beta) * (gr - *fast);
    const float s = *slow + (1.0f - slow_beta) * (gr - *slow);
    *fast = f;
    *slow = s;
    const float blend = fast_weight * f + (1.0f - fast_weight) * s;
    x[i] = ToComputeDev(nesterov ? (1.0f - momentum) * gr + momentum * blend
                                 : blend);
  }
}

// --- Frobenius normalization -----------------------------------------------

// The first Gram is taken on the unnormalized X; its trace is ||X||_F^2, which
// gives the normalization for free. One block per matrix: reduce the diagonal
// of A, then rescale X by 1/d and A by 1/d^2.
__global__ void AnvilNormKernel(int rows, int cols, int min_dim,
                                ComputeType* __restrict__ x,
                                ComputeType* __restrict__ a) {
  const int p = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  const long long min_sq = static_cast<long long>(min_dim) * min_dim;
  ComputeType* xm = x + static_cast<long long>(p) * mat;
  ComputeType* am = a + static_cast<long long>(p) * min_sq;
  const int tid = threadIdx.x;

  float local = 0.0f;
  for (int i = tid; i < min_dim; i += blockDim.x) {
    local += AsFloatDev(am[static_cast<long long>(i) * min_dim + i]);
  }
  const float trace = BlockReduceSum(local);
  const float d = sqrtf(fmaxf(trace, 0.0f)) * 1.05f + 1e-6f;
  const float inv_d = 1.0f / d;
  const float inv_d2 = inv_d * inv_d;
  for (long long i = tid; i < mat; i += blockDim.x) {
    xm[i] = ToComputeDev(AsFloatDev(xm[i]) * inv_d);
  }
  for (long long i = tid; i < min_sq; i += blockDim.x) {
    am[i] = ToComputeDev(AsFloatDev(am[i]) * inv_d2);
  }
}

// --- Cascade elementwise steps ---------------------------------------------

// b = cb * a + cc * a2, elementwise over the stacked min x min buffers.
__global__ void AnvilBKernel(long long count, float cb, float cc,
                             const ComputeType* __restrict__ a,
                             const ComputeType* __restrict__ a2,
                             ComputeType* __restrict__ b) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    b[i] = ToComputeDev(cb * AsFloatDev(a[i]) + cc * AsFloatDev(a2[i]));
  }
}

// x = ca * x + prod, elementwise over the stacked matrices.
__global__ void AnvilAxpbyKernel(long long count, float ca,
                                 ComputeType* __restrict__ x,
                                 const ComputeType* __restrict__ prod) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] = ToComputeDev(ca * AsFloatDev(x[i]) + AsFloatDev(prod[i]));
  }
}

// --- Per-lane energy equalizer ---------------------------------------------

// One block per matrix. Computes each lane's mean square, updates the lane
// energy EMA, and rescales the matrix by the per-lane gain normalized back to
// the pre-equalization Frobenius norm. `reduce_cols` selects the lane axis:
// columns (one lane per row) or rows (one lane per column), matching the CPU
// reference's `red_dim`.
__global__ void AnvilEqualizerKernel(int rows, int cols, bool reduce_cols,
                                     float beta2, ComputeType* __restrict__ x,
                                     float* __restrict__ lane_energy,
                                     float* __restrict__ scratch) {
  const int p = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  ComputeType* xm = x + static_cast<long long>(p) * mat;
  const int lane_count = reduce_cols ? rows : cols;
  const int lane_len = reduce_cols ? cols : rows;
  const float inv_lane_len = 1.0f / static_cast<float>(lane_len);
  float* lane = lane_energy + static_cast<long long>(p) * lane_count;
  float* power = scratch + static_cast<long long>(p) * lane_count;
  const int tid = threadIdx.x;

  // Pass 1: per-lane mean square.
  for (int l = 0; l < lane_count; ++l) {
    float local = 0.0f;
    if (reduce_cols) {
      const ComputeType* row = xm + static_cast<long long>(l) * cols;
      for (int c = tid; c < cols; c += blockDim.x) {
        const float v = AsFloatDev(row[c]);
        local += v * v;
      }
    } else {
      for (int r = tid; r < rows; r += blockDim.x) {
        const float v = AsFloatDev(xm[static_cast<long long>(r) * cols + l]);
        local += v * v;
      }
    }
    const float s = BlockReduceSum(local);
    if (tid == 0) power[l] = s * inv_lane_len;
  }
  __syncthreads();

  float lsum = 0.0f;
  for (int l = tid; l < lane_count; l += blockDim.x) lsum += power[l];
  const float sum_power = BlockReduceSum(lsum);
  const float pre_norm = sqrtf(sum_power * static_cast<float>(lane_len));

  // Pass 2: update the lane energy and stash each lane's gain in `power`.
  float lpost = 0.0f;
  for (int l = tid; l < lane_count; l += blockDim.x) {
    const float pw = power[l];
    const float energy = lane[l] + (1.0f - beta2) * (pw - lane[l]);
    lane[l] = energy;
    const float gain = rsqrtf(fmaxf(energy, 1e-10f));
    power[l] = gain;
    lpost += (pw * static_cast<float>(lane_len)) * gain * gain;
  }
  const float sum_post = BlockReduceSum(lpost);
  const float inv_post = pre_norm / fmaxf(sqrtf(sum_post), 1e-10f);
  __syncthreads();

  // Pass 3: apply gain * (pre_norm / post_norm) along the lane axis.
  if (reduce_cols) {
    for (int l = 0; l < lane_count; ++l) {
      const float s = power[l] * inv_post;
      ComputeType* row = xm + static_cast<long long>(l) * cols;
      for (int c = tid; c < cols; c += blockDim.x) {
        row[c] = ToComputeDev(AsFloatDev(row[c]) * s);
      }
    }
  } else {
    for (int l = 0; l < lane_count; ++l) {
      const float s = power[l] * inv_post;
      for (int r = tid; r < rows; r += blockDim.x) {
        const long long idx = static_cast<long long>(r) * cols + l;
        xm[idx] = ToComputeDev(AsFloatDev(xm[idx]) * s);
      }
    }
  }
}

// --- Sign-aligned cautious update ------------------------------------------

__global__ void AnvilApplyKernel(long long total, long long rail_stride,
                                 float lr, float weight_decay,
                                 float* __restrict__ master,
                                 ComputeType* __restrict__ value,
                                 const float* __restrict__ velocity,
                                 const ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float pv = master[i];
    const float gv = AsFloatDev(x[i]);
    // Cautious decay: only shrink a parameter whose sign agrees with the slow
    // rail, the denoised gradient estimate.
    const float decay = (velocity[rail_stride + i] * pv >= 0.0f)
                            ? lr * weight_decay * pv
                            : 0.0f;
    const float pi = pv - decay - lr * gv;
    master[i] = pi;
    value[i] = ToComputeDev(pi);
  }
}

int GridSize(long long count, int threads) {
  const long long blocks = (count + threads - 1) / threads;
  constexpr long long kMaxBlocks = 512;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

// Cached working buffers for AnvilUpdate. cudaMalloc/cudaFree synchronize the
// device and dominate the update when they run once per group per step, so the
// buffers are allocated once at the largest size any group needs and reused.
ComputeType* g_anvil_workspace = nullptr;
std::size_t g_anvil_workspace_capacity = 0;
float* g_anvil_scratch = nullptr;
std::size_t g_anvil_scratch_capacity = 0;

void EnsureAnvilWorkspace(std::size_t elements, std::size_t scratch_floats) {
  if (elements > g_anvil_workspace_capacity) {
    if (g_anvil_workspace != nullptr) Free(g_anvil_workspace);
    g_anvil_workspace =
        static_cast<ComputeType*>(Alloc(elements * sizeof(ComputeType)));
    g_anvil_workspace_capacity = elements;
  }
  if (scratch_floats > g_anvil_scratch_capacity) {
    if (g_anvil_scratch != nullptr) Free(g_anvil_scratch);
    g_anvil_scratch =
        static_cast<float*>(Alloc(scratch_floats * sizeof(float)));
    g_anvil_scratch_capacity = scratch_floats;
  }
}

}  // namespace

void AnvilUpdate(const AnvilParams& params, const ComputeType* stacked_grads,
                 float* stacked_master, ComputeType* stacked_values,
                 float* velocity, float* lane_energy) {
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;

  const long long mat = static_cast<long long>(rows) * cols;
  const long long total = static_cast<long long>(num_params) * mat;
  const long long rail_stride = total;
  const int num_maps = std::min(std::max(params.num_maps, 0), 6);
  // `red_dim == -1` reasons over columns (one lane per row); `-2` over rows
  // (one lane per column); anything else follows the larger extent, matching
  // the CPU reference.
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);
  const int min_dim = rows < cols ? rows : cols;
  const long long min_sq = static_cast<long long>(min_dim) * min_dim;
  const long long min_total = min_sq * num_params;
  const bool tall = rows > cols;
  const int lane_count = reduce_cols ? rows : cols;

  // Working matrix (momentum output, whitened in place) and the per-matrix
  // GEMM operands. All stay ComputeType so they can feed cuBLAS. The layout is
  // [x | prod | a | a2 | b], from one cached allocation.
  EnsureAnvilWorkspace(static_cast<std::size_t>(2 * total + 3 * min_total),
                       static_cast<std::size_t>(num_params) * lane_count);
  ComputeType* x = g_anvil_workspace;
  ComputeType* prod = x + total;
  ComputeType* a = prod + total;
  ComputeType* a2 = a + min_total;
  ComputeType* b = a2 + min_total;
  float* scratch = g_anvil_scratch;

  const int kThreads = 256;
  const int elem_grid = GridSize(total, kThreads);
  const int min_grid = GridSize(min_total, kThreads);

  cuda_backend::Launch(AnvilMomentumKernel, dim3(elem_grid), dim3(kThreads), 0,
                       total, rail_stride, params.fast_beta, params.slow_beta,
                       params.fast_weight, params.momentum, params.nesterov,
                       stacked_grads, velocity, x);

  // First Gram, then the Frobenius normalization of X and A.
  if (tall) {
    GemmParams ga;
    ga.m = cols;
    ga.n = cols;
    ga.k = rows;
    ga.batch_count = num_params;
    ga.transpose_a = true;
    ga.stride_a = mat;
    ga.stride_b = mat;
    ga.stride_c = min_sq;
    Gemm(GemmMode::kForward, ga, x, x, a);
  } else {
    GemmParams ga;
    ga.m = rows;
    ga.n = rows;
    ga.k = cols;
    ga.batch_count = num_params;
    ga.transpose_b = true;
    ga.stride_a = mat;
    ga.stride_b = mat;
    ga.stride_c = min_sq;
    Gemm(GemmMode::kForward, ga, x, x, a);
  }
  {
    // One block per matrix: reduce the trace of A and rescale both buffers in
    // place.
    cuda_backend::Launch(AnvilNormKernel, dim3(num_params), dim3(kThreads), 0,
                         rows, cols, min_dim, x, a);
  }

  for (int k = 0; k < num_maps; ++k) {
    const float ca = kAnvilMaps[k][0];
    const float cb = kAnvilMaps[k][1];
    const float cc = kAnvilMaps[k][2];
    if (k > 0) {
      if (tall) {
        GemmParams ga;
        ga.m = cols;
        ga.n = cols;
        ga.k = rows;
        ga.batch_count = num_params;
        ga.transpose_a = true;
        ga.stride_a = mat;
        ga.stride_b = mat;
        ga.stride_c = min_sq;
        Gemm(GemmMode::kForward, ga, x, x, a);
      } else {
        GemmParams ga;
        ga.m = rows;
        ga.n = rows;
        ga.k = cols;
        ga.batch_count = num_params;
        ga.transpose_b = true;
        ga.stride_a = mat;
        ga.stride_b = mat;
        ga.stride_c = min_sq;
        Gemm(GemmMode::kForward, ga, x, x, a);
      }
    }

    // A2 = A A.
    GemmParams ga2;
    ga2.m = min_dim;
    ga2.n = min_dim;
    ga2.k = min_dim;
    ga2.batch_count = num_params;
    ga2.stride_a = min_sq;
    ga2.stride_b = min_sq;
    ga2.stride_c = min_sq;
    Gemm(GemmMode::kForward, ga2, a, a, a2);

    cuda_backend::Launch(AnvilBKernel, dim3(min_grid), dim3(kThreads), 0,
                         min_total, cb, cc, a, a2, b);

    if (tall) {
      // prod = X B  (rows x cols).
      GemmParams gp;
      gp.m = rows;
      gp.n = cols;
      gp.k = cols;
      gp.batch_count = num_params;
      gp.stride_a = mat;
      gp.stride_b = min_sq;
      gp.stride_c = mat;
      Gemm(GemmMode::kForward, gp, x, b, prod);
    } else {
      // prod = B X  (rows x cols).
      GemmParams gp;
      gp.m = rows;
      gp.n = cols;
      gp.k = rows;
      gp.batch_count = num_params;
      gp.stride_a = min_sq;
      gp.stride_b = mat;
      gp.stride_c = mat;
      Gemm(GemmMode::kForward, gp, b, x, prod);
    }

    cuda_backend::Launch(AnvilAxpbyKernel, dim3(elem_grid), dim3(kThreads), 0,
                         total, ca, x, prod);
  }

  cuda_backend::Launch(AnvilEqualizerKernel, dim3(num_params), dim3(kThreads),
                       0, rows, cols, reduce_cols, params.beta2, x, lane_energy,
                       scratch);
  cuda_backend::Launch(AnvilApplyKernel, dim3(elem_grid), dim3(kThreads), 0,
                       total, rail_stride, params.lr, params.weight_decay,
                       stacked_master, stacked_values, velocity, x);
}

}  // namespace kernels
}  // namespace nanochat
