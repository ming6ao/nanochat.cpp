// CUDA Muon family: Nesterov momentum, MuonEq row equilibration, Polar Express
// orthogonalisation through batched cuBLAS GEMMs, Muon+ renormalisation,
// NorMuon variance reduction with a factored second moment, and the cautious
// decoupled weight decay. The math mirrors backends/cpu/kernels.cc exactly.
//
// Two paths run this stage, and a group takes one of them by shape
// (`MuonFusedSelected`):
//
//   * the multi-launch path below, one launch per operation, with the three
//     Polar Express products going through `kernels::Gemm` so cuBLAS carries
//     them. It is the path for a group whose arithmetic dominates;
//   * the fused path in muon_fused.cu: one phase-parameterised device function,
//     one launch per phase (or one cooperative launch for the whole group), and
//     three epilogue folds that drop two buffers and two passes over the stack.
//     It is the path for a group small enough that the removed launch gaps and
//     the folded passes pay for its hand-written GEMM.
//
// Both paths read and write one workspace and one statistics scratch, sliced
// the same way, so a group can move between them without a caller noticing.
//
// The stacked parameters are [num_params, rows, cols]. The Polar Express
// iterations run as strided-batched GEMMs over the whole stack through
// kernels::Gemm, one cuBLAS call per GEMM and per group. Every whole-matrix
// statistic (the Frobenius norm, the equilibration divisor, the renormalisation
// scale) is reduced by many blocks per matrix, and every per-row or per-column
// statistic (the row equilibration, the NorMuon mean squares) by one block or
// one thread per row or column, so no stage serialises a matrix on one block.
// The pure elementwise stages run grid-stride over the stack.
//
// Layout notes for the tall (rows > cols) and wide (rows <= cols) cases:
//   * tall:  A = X^T X  (cols x cols), prod = X B
//   * wide:  A = X X^T  (rows x rows), prod = B X
// A and B therefore occupy [num_params, min(rows, cols), min(rows, cols)].
// There is no `A^2` buffer: `b = cb * a + cc * a * a` writes the product over
// itself in place, which is one of the two min-by-min slices the fused path's
// folds remove from both paths (backends/cuda/kernels/muon_fused.h).
//
// See docs/kernels.md and docs/model.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "backends/cuda/device.h"
#include "backends/cuda/kernels/device_utils.cuh"
#include "backends/cuda/kernels/muon_fused.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

using cuda_kernels::AsFloatDev;
using cuda_kernels::BlockReduceSum;
using cuda_kernels::ComputeVector;
using cuda_kernels::FloatVector;
using cuda_kernels::kFloatPerVector;
using cuda_kernels::kVectorElems;
using cuda_kernels::ToComputeDev;
using cuda_kernels::VectorGrid;
using cuda_kernels::VectorReady;

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
// Nesterov-accelerated gradient or the plain first moment. `grad_scale` carries
// the global clip scale (one when the clip is inactive).
__global__ void MuonMomentumKernel(long long total, float momentum,
                                   bool nesterov, ScalarArg grad_scale,
                                   const float* __restrict__ grads,
                                   float* __restrict__ buf1,
                                   ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const float scale = grad_scale.host();
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float gr = grads[i] * scale;
    const float b = buf1[i] + (1.0f - momentum) * (gr - buf1[i]);
    buf1[i] = b;
    x[i] = ToComputeDev(nesterov ? (1.0f - momentum) * gr + momentum * b : b);
  }
}

// --- Blocked matrix statistics ---------------------------------------------
//
// Every whole-matrix statistic is reduced by many blocks per matrix, not by
// one. A group is a handful of matrices, so one block per matrix leaves almost
// the whole card idle: the eight matrices of the 2048x512 group were reduced by
// eight blocks, and each of them walked a megaelement alone. The stage is
// therefore one blocked partial pass, one short finish, and one full-grid
// scale, which is the same arithmetic at the bandwidth the card has.

// Partials of one matrix, one per block of the `blockIdx.x` dimension of the
// launch, so a matrix is reduced by `gridDim.x` blocks at once. The partials
// are double: a four-mega-element sum in float would lose the low bits the
// reference keeps.
__global__ void MuonMatrixPartialsKernel(int rows, int cols,
                                         const ComputeType* __restrict__ x,
                                         int blocks_per_matrix,
                                         double* __restrict__ partials) {
  const long long mat = static_cast<long long>(rows) * cols;
  const int p = blockIdx.y;
  const ComputeType* xm = x + static_cast<long long>(p) * mat;
  const long long chunk = (mat + blocks_per_matrix - 1) / blocks_per_matrix;
  const long long begin = static_cast<long long>(blockIdx.x) * chunk;
  const long long end = begin + chunk < mat ? begin + chunk : mat;
  double local = 0.0;
  for (long long i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const double v = AsFloatDev(xm[i]);
    local += v * v;
  }
  const double sum = BlockReduceSum(local);
  if (threadIdx.x == 0) {
    partials[static_cast<long long>(p) * blocks_per_matrix + blockIdx.x] = sum;
  }
}

// Which whole-matrix statistic the finish kernel writes.
enum class MatrixStat {
  kFrobenius,  // sqrt(sum of squares): the norm the rows normalise to
  kEquilibrationDivisor,  // 1.01 * ||X||_F + 1e-6: the MuonEq divisor
  kRenormScale,           // sqrt(min(rows, cols)) / ||X||_F: the Muon+ scale
};

// The statistic of every matrix of the group from its partials. One block
// covers the whole group, because the work is `num_params` short sums.
__global__ void MuonMatrixStatKernel(int num_params, int blocks_per_matrix,
                                     MatrixStat mode, float target,
                                     const double* __restrict__ partials,
                                     float* __restrict__ out) {
  for (int p = threadIdx.x; p < num_params; p += blockDim.x) {
    double sum = 0.0;
    for (int b = 0; b < blocks_per_matrix; ++b) {
      sum += partials[static_cast<long long>(p) * blocks_per_matrix + b];
    }
    const float norm = sqrtf(static_cast<float>(sum));
    float value = norm;
    if (mode == MatrixStat::kEquilibrationDivisor) {
      value = norm * 1.01f + 1e-6f;
    } else if (mode == MatrixStat::kRenormScale) {
      value = target / fmaxf(norm, 1e-6f);
    }
    out[p] = value;
  }
}

// Applies one per-matrix factor to every element of the stack, multiplying or
// dividing, which is the shape of both the equilibration divisor and the
// renormalisation scale.
__global__ void MuonScaleMatricesKernel(long long total, long long mat,
                                        bool divide,
                                        const float* __restrict__ factor,
                                        ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float f = factor[i / mat];
    const float v = AsFloatDev(x[i]);
    x[i] = ToComputeDev(divide ? v / f : v * f);
  }
}

// Rescale each row to the mean row norm: the MuonEq equilibration. One block
// per (matrix, row), so the rows of a group are normalised in parallel.
__global__ void MuonEquilibrateRowKernel(int rows, int cols,
                                         ComputeType* __restrict__ x,
                                         const float* __restrict__ frob) {
  const int p = blockIdx.y;
  const int r = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  ComputeType* row =
      x + static_cast<long long>(p) * mat + static_cast<long long>(r) * cols;
  float local = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = AsFloatDev(row[c]);
    local += v * v;
  }
  const float row_norm = fmaxf(sqrtf(BlockReduceSum(local)), 1e-6f);
  const float target = frob[p] / sqrtf(static_cast<float>(rows));
  const float s = target / row_norm;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row[c] = ToComputeDev(AsFloatDev(row[c]) * s);
  }
}

// --- Polar Express elementwise steps ---------------------------------------

// b = cb * a + cc * a2, elementwise over the stacked min x min buffers. `b` may
// be `a2` itself: the combination is per index, so the destination can be the
// second operand, which is what lets the caller skip the separate `a2` buffer.
__global__ void MuonBKernel(long long count, float cb, float cc,
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
__global__ void MuonAxpbyKernel(long long count, float ca,
                                ComputeType* __restrict__ x,
                                const ComputeType* __restrict__ prod) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] = ToComputeDev(ca * AsFloatDev(x[i]) + AsFloatDev(prod[i]));
  }
}

// --- NorMuon variance reduction --------------------------------------------
//
// The first statistic is one mean square per row or per column, and both are
// reduced by many blocks: a row by one block per (matrix, row), a column by one
// thread per column, which keeps the reads of a warp on consecutive columns of
// the same row. The second moment and the per-index step then run one block per
// matrix over `red_index` entries, and the scale runs over the whole stack.

// One mean square per row of every matrix.
__global__ void MuonRowMeanSquareKernel(int rows, int cols,
                                        const ComputeType* __restrict__ x,
                                        float* __restrict__ vmean) {
  const int p = blockIdx.y;
  const int r = blockIdx.x;
  const long long mat = static_cast<long long>(rows) * cols;
  const ComputeType* row =
      x + static_cast<long long>(p) * mat + static_cast<long long>(r) * cols;
  float local = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = AsFloatDev(row[c]);
    local += v * v;
  }
  const float sum = BlockReduceSum(local);
  if (threadIdx.x == 0) {
    vmean[static_cast<long long>(p) * rows + r] =
        sum / static_cast<float>(cols);
  }
}

// One mean square per column of every matrix. Thread `t` owns column
// `blockIdx.x * blockDim.x + t` and walks the rows, so the warp's loads are
// consecutive columns of one row.
__global__ void MuonColMeanSquareKernel(int rows, int cols,
                                        const ComputeType* __restrict__ x,
                                        float* __restrict__ vmean) {
  const int p = blockIdx.y;
  const long long mat = static_cast<long long>(rows) * cols;
  const ComputeType* xm = x + static_cast<long long>(p) * mat;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float local = 0.0f;
  for (int r = 0; r < rows; ++r) {
    const float v = AsFloatDev(xm[static_cast<long long>(r) * cols + c]);
    local += v * v;
  }
  vmean[static_cast<long long>(p) * cols + c] =
      local / static_cast<float>(rows);
}

// The factored second moment and the per-index step size of one matrix, with
// the matrix's global scale. One block per matrix over `red_index` entries.
// The arithmetic is the composition's, in the same order: `v_norm` from the
// pre-update mean squares, then `sum_scaled` from the same values.
__global__ void MuonNorMuonMomentsKernel(int red_index, int red_size,
                                         float beta2, float* __restrict__ vmean,
                                         float* __restrict__ second,
                                         float* __restrict__ scale) {
  const int p = blockIdx.x;
  float* vm = vmean + static_cast<long long>(p) * red_index;
  float* sec = second + static_cast<long long>(p) * red_index;
  const int tid = threadIdx.x;
  const float size = static_cast<float>(red_size);

  float lsum = 0.0f;
  for (int i = tid; i < red_index; i += blockDim.x) lsum += vm[i];
  const float sum_vmean = BlockReduceSum(lsum);
  const float v_norm = sqrtf(sum_vmean * size);

  float lscaled = 0.0f;
  for (int i = tid; i < red_index; i += blockDim.x) {
    const float mean = vm[i];
    const float sm = sec[i] + (1.0f - beta2) * (mean - sec[i]);
    sec[i] = sm;
    const float step = 1.0f / sqrtf(fmaxf(sm, 1e-10f));
    vm[i] = step;
    lscaled += (mean * size) * step * step;
  }
  const float sum_scaled = BlockReduceSum(lscaled);
  const float denom = fmaxf(sqrtf(sum_scaled), 1e-10f);
  if (tid == 0) scale[p] = v_norm / denom;
}

// Applies `step_size * global_scale` along the reduction axis.
__global__ void MuonNorMuonScaleKernel(long long total, long long mat, int cols,
                                       int red_index, bool reduce_cols,
                                       const float* __restrict__ vmean,
                                       const float* __restrict__ scale,
                                       ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const long long p = i / mat;
    const long long within = i - p * mat;
    const long long index = reduce_cols ? within / cols : within % cols;
    const float s = vmean[p * red_index + index] * scale[p];
    x[i] = ToComputeDev(AsFloatDev(x[i]) * s);
  }
}

// --- Cautious weight decay + parameter update ------------------------------

__global__ void MuonApplyKernel(long long total, float lr, float weight_decay,
                                float* __restrict__ master,
                                ComputeType* __restrict__ value,
                                const ComputeType* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const float pv = master[i];
    const float gv = AsFloatDev(x[i]);
    // Cautious decay: only shrink a parameter that agrees in sign with the
    // update direction.
    const float decay = (gv * pv >= 0.0f) ? lr * weight_decay * pv : 0.0f;
    const float pi = pv - lr * gv - decay;
    master[i] = pi;
    value[i] = ToComputeDev(pi);
  }
}

// --- Vector access ----------------------------------------------------------
//
// Every element-wise kernel of this family gives one parameter to one thread
// and caps its grid at 512 blocks, so a warp loads 64 bytes and at most 131,072
// threads cover a 2.6 GB state. Each thread now takes one aligned 16-byte
// vector, and the vector grids carry a cap that fits a modern card. The
// arithmetic per element does not change, so the results are bit identical.

// Nesterov-accelerated gradient or the plain first moment. The gradient and the
// momentum buffer are fp32 and the parameter copy is `ComputeType`, so the fp32
// streams advance `kFloatPerVector` groups at a time and the parameter stream
// advances one vector.
__global__ void MuonMomentumVectorKernel(long long vectors, float momentum,
                                         bool nesterov, ScalarArg grad_scale,
                                         const FloatVector* __restrict__ grads,
                                         FloatVector* __restrict__ buf1,
                                         ComputeVector* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const float scale = grad_scale.host();
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    ComputeVector value = x[v];
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
        const float nb = bv + (1.0f - momentum) * (gr - bv);
        next[j].e[s] = nb;
        value.e[4 * j + s] = ToComputeDev(
            nesterov ? (1.0f - momentum) * gr + momentum * nb : nb);
      }
    }
#pragma unroll
    for (int j = 0; j < kFloatPerVector; ++j) bbase[j] = next[j];
    x[v] = value;
  }
}

__global__ void MuonScaleMatricesVectorKernel(long long vectors, long long mat,
                                              bool divide,
                                              const float* __restrict__ factor,
                                              ComputeVector* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    // The host requires `mat` to be a whole number of vectors, so one vector
    // never spans two matrices and one factor serves all of its elements.
    const float f = factor[(v * kVectorElems) / mat];
    ComputeVector value = x[v];
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      const float e = AsFloatDev(value.e[k]);
      value.e[k] = ToComputeDev(divide ? e / f : e * f);
    }
    x[v] = value;
  }
}

__global__ void MuonNorMuonScaleVectorKernel(long long vectors, long long mat,
                                             int cols, int red_index,
                                             bool reduce_cols,
                                             const float* __restrict__ vmean,
                                             const float* __restrict__ scale,
                                             ComputeVector* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    ComputeVector value = x[v];
    const long long first = v * kVectorElems;
    const long long p = first / mat;
    const long long within = first - p * mat;
    const float global = scale[p];
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      const long long w = within + k;
      const long long index = reduce_cols ? w / cols : w % cols;
      const float s = vmean[p * red_index + index] * global;
      value.e[k] = ToComputeDev(AsFloatDev(value.e[k]) * s);
    }
    x[v] = value;
  }
}

__global__ void MuonAxpbyVectorKernel(long long vectors, float ca,
                                      ComputeVector* __restrict__ x,
                                      const ComputeVector* __restrict__ prod) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const ComputeVector p = prod[v];
    ComputeVector value = x[v];
#pragma unroll
    for (int k = 0; k < kVectorElems; ++k) {
      value.e[k] =
          ToComputeDev(ca * AsFloatDev(value.e[k]) + AsFloatDev(p.e[k]));
    }
    x[v] = value;
  }
}

__global__ void MuonApplyVectorKernel(long long vectors, float lr,
                                      float weight_decay,
                                      FloatVector* __restrict__ master,
                                      ComputeVector* __restrict__ value,
                                      const ComputeVector* __restrict__ x) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long v =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vectors; v += stride) {
    const ComputeVector update = x[v];
    FloatVector* mbase = master + v * kFloatPerVector;
    FloatVector next[kFloatPerVector];
    ComputeVector result;
#pragma unroll
    for (int j = 0; j < kFloatPerVector; ++j) {
      const FloatVector m = mbase[j];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float pv = m.e[s];
        const float gv = AsFloatDev(update.e[4 * j + s]);
        // Cautious decay: only shrink a parameter that agrees in sign with the
        // update direction.
        const float decay = (gv * pv >= 0.0f) ? lr * weight_decay * pv : 0.0f;
        const float pi = pv - lr * gv - decay;
        next[j].e[s] = pi;
        result.e[4 * j + s] = ToComputeDev(pi);
      }
    }
#pragma unroll
    for (int j = 0; j < kFloatPerVector; ++j) mbase[j] = next[j];
    value[v] = result;
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
// They are per device thread, so two device threads do not share one buffer
// (docs/distributed.md Part 3, section 8.1).
thread_local ComputeType* g_muon_workspace = nullptr;
thread_local std::size_t g_muon_workspace_capacity = 0;
thread_local float* g_muon_scratch = nullptr;
thread_local std::size_t g_muon_scratch_capacity = 0;
thread_local double* g_muon_partials = nullptr;
thread_local std::size_t g_muon_partials_capacity = 0;

// Grows the three cached buffers to the sizes the caller asks for, which is
// `[x | prod | a | b]` in `ComputeType`, `[stat | vmean | scale]` in floats,
// and a double partial per (matrix, block) of a whole-matrix pass. The partial
// count comes from `MuonPartialBlocks` (backends/cuda/kernels/muon_fused.h),
// which both paths of this stage use, so the block count, the chunk order, and
// therefore the statistic cannot drift between them.
void EnsureMuonWorkspace(std::size_t elements, std::size_t scratch_floats,
                         std::size_t partial_doubles) {
  if (elements > g_muon_workspace_capacity) {
    if (g_muon_workspace != nullptr) Free(g_muon_workspace);
    g_muon_workspace =
        static_cast<ComputeType*>(Alloc(elements * sizeof(ComputeType)));
    g_muon_workspace_capacity = elements;
  }
  if (scratch_floats > g_muon_scratch_capacity) {
    if (g_muon_scratch != nullptr) Free(g_muon_scratch);
    g_muon_scratch = static_cast<float*>(Alloc(scratch_floats * sizeof(float)));
    g_muon_scratch_capacity = scratch_floats;
  }
  if (partial_doubles > g_muon_partials_capacity) {
    if (g_muon_partials != nullptr) Free(g_muon_partials);
    g_muon_partials =
        static_cast<double*>(Alloc(partial_doubles * sizeof(double)));
    g_muon_partials_capacity = partial_doubles;
  }
}

}  // namespace

void MuonUpdate(const MuonParams& params, const float* stacked_grads,
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

  // Working matrix (momentum output, orthonormalised in place), its ping-pong
  // partner, and the two per-matrix GEMM operands. All stay ComputeType so they
  // can feed cuBLAS. The layout is [x | prod | a | b], from one cached
  // allocation, and it is the layout the fused path slices as
  // [x | xn | a | b] (`MuonFusedWorkspaceElements`): the `prod` buffer is the
  // fused path's second iterate, and `b` takes the product of `A * A` in place
  // here and the folded `cb * a + cc * acc` there, so neither path needs the
  // third min-by-min buffer the family used to carry.
  const int red_index = reduce_cols ? rows : cols;
  const int red_size = reduce_cols ? cols : rows;
  const int partial_blocks = MuonPartialBlocks(mat);
  // The scratch holds the per-matrix statistic (`num_params`), then the NorMuon
  // mean squares (`num_params * red_index`), then the NorMuon global scales
  // (`num_params`). Both paths slice it the same way; the allocation carries
  // the larger reduction extent so one buffer serves both `red_dim` values.
  const std::size_t stat_len = static_cast<std::size_t>(num_params);
  const std::size_t vmean_len =
      static_cast<std::size_t>(num_params) * red_index;
  EnsureMuonWorkspace(static_cast<std::size_t>(2 * total + 2 * min_total),
                      MuonFusedScratchFloats(num_params, rows, cols),
                      static_cast<std::size_t>(partial_blocks) * num_params);
  ComputeType* x = g_muon_workspace;
  ComputeType* prod = x + total;
  ComputeType* a = prod + total;
  ComputeType* b = a + min_total;
  float* stat = g_muon_scratch;
  float* vmean = stat + MuonScratchOffset(stat_len);
  float* scale = vmean + MuonScratchOffset(vmean_len);
  double* partials_dev = g_muon_partials;

  // The fused path, when the group is one the folds pay for
  // (`MuonFusedSelected`, section 4's shape rule). It runs the whole stage out
  // of the same workspace, scratch, and double partials, so a group that takes
  // it and one that does not are interchangeable at this seam.
  if (MuonFusedSelected(params)) {
    MuonFusedUpdate(params, stacked_grads, stacked_master, stacked_values, buf1,
                    buf2, g_muon_workspace, g_muon_scratch, g_muon_partials);
    return;
  }

  const int kThreads = 256;
  const int elem_grid = GridSize(total, kThreads);
  const int min_grid = GridSize(min_total, kThreads);
  const dim3 partial_grid(partial_blocks, num_params);
  const int col_blocks = (cols + kThreads - 1) / kThreads;

  // The whole-matrix statistic of every matrix of the group, from one blocked
  // partial pass and one short finish.
  const auto matrix_stat = [&](MatrixStat mode, float target) {
    cuda_backend::Launch(MuonMatrixPartialsKernel, partial_grid, dim3(kThreads),
                         0, rows, cols, x, partial_blocks, partials_dev);
    cuda_backend::Launch(MuonMatrixStatKernel, dim3(1), dim3(kThreads), 0,
                         num_params, partial_blocks, mode, target, partials_dev,
                         stat);
  };

  // The clip scale is either the host value or a device scalar the reduction
  // wrote, so the optimizer step needs no host read of the clip.
  if (VectorReady(total, {x, stacked_grads, buf1})) {
    const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
    cuda_backend::Launch(MuonMomentumVectorKernel, dim3(vector_grid),
                         dim3(kThreads), 0, total / kVectorElems,
                         params.momentum, params.nesterov, params.grad_scale,
                         reinterpret_cast<const FloatVector*>(stacked_grads),
                         reinterpret_cast<FloatVector*>(buf1),
                         reinterpret_cast<ComputeVector*>(x));
  } else {
    cuda_backend::Launch(MuonMomentumKernel, dim3(elem_grid), dim3(kThreads), 0,
                         total, params.momentum, params.nesterov,
                         params.grad_scale, stacked_grads, buf1, x);
  }
  // MuonEq: each row to the mean row norm, then the matrix to unit Frobenius
  // norm, which is what makes the Polar Express coefficients apply.
  matrix_stat(MatrixStat::kFrobenius, 0.0f);
  cuda_backend::Launch(MuonEquilibrateRowKernel, dim3(rows, num_params),
                       dim3(kThreads), 0, rows, cols, x, stat);
  matrix_stat(MatrixStat::kEquilibrationDivisor, 0.0f);
  if (mat % kVectorElems == 0 && VectorReady(total, {x, stat})) {
    const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
    cuda_backend::Launch(MuonScaleMatricesVectorKernel, dim3(vector_grid),
                         dim3(kThreads), 0, total / kVectorElems, mat, true,
                         stat, reinterpret_cast<ComputeVector*>(x));
  } else {
    cuda_backend::Launch(MuonScaleMatricesKernel, dim3(elem_grid),
                         dim3(kThreads), 0, total, mat, true, stat, x);
  }

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
      Gemm(GemmMode::kForward, ga, x, x, a);

      // A2 = A A, then b = cb * a + cc * A2 with the product landing in `b`
      // itself: the combination is per index, so the destination is also its
      // second operand and the separate `a2` buffer is gone.
      GemmParams ga2;
      ga2.m = cols;
      ga2.n = cols;
      ga2.k = cols;
      ga2.batch_count = num_params;
      ga2.stride_a = min_sq;
      ga2.stride_b = min_sq;
      ga2.stride_c = min_sq;
      Gemm(GemmMode::kForward, ga2, a, a, b);

      cuda_backend::Launch(MuonBKernel, dim3(min_grid), dim3(kThreads), 0,
                           min_total, cb, cc, a, b, b);

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
      Gemm(GemmMode::kForward, ga, x, x, a);

      // A2 = A A, then b = cb * a + cc * A2 with the product landing in `b`
      // itself (see the tall branch).
      GemmParams ga2;
      ga2.m = rows;
      ga2.n = rows;
      ga2.k = rows;
      ga2.batch_count = num_params;
      ga2.stride_a = min_sq;
      ga2.stride_b = min_sq;
      ga2.stride_c = min_sq;
      Gemm(GemmMode::kForward, ga2, a, a, b);

      cuda_backend::Launch(MuonBKernel, dim3(min_grid), dim3(kThreads), 0,
                           min_total, cb, cc, a, b, b);

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

    if (VectorReady(total, {x, prod})) {
      const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
      cuda_backend::Launch(MuonAxpbyVectorKernel, dim3(vector_grid),
                           dim3(kThreads), 0, total / kVectorElems, ca,
                           reinterpret_cast<ComputeVector*>(x),
                           reinterpret_cast<const ComputeVector*>(prod));
    } else {
      cuda_backend::Launch(MuonAxpbyKernel, dim3(elem_grid), dim3(kThreads), 0,
                           total, ca, x, prod);
    }
  }

  // Muon+ renormalisation: the Frobenius norm snaps back to sqrt(min) of the
  // matrix's extents.
  matrix_stat(MatrixStat::kRenormScale, sqrtf(static_cast<float>(min_dim)));
  if (mat % kVectorElems == 0 && VectorReady(total, {x, stat})) {
    const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
    cuda_backend::Launch(MuonScaleMatricesVectorKernel, dim3(vector_grid),
                         dim3(kThreads), 0, total / kVectorElems, mat, false,
                         stat, reinterpret_cast<ComputeVector*>(x));
  } else {
    cuda_backend::Launch(MuonScaleMatricesKernel, dim3(elem_grid),
                         dim3(kThreads), 0, total, mat, false, stat, x);
  }

  // NorMuon: one mean square per row (or per column), the factored second
  // moment, then the normalized step.
  if (reduce_cols) {
    cuda_backend::Launch(MuonRowMeanSquareKernel, dim3(rows, num_params),
                         dim3(kThreads), 0, rows, cols, x, vmean);
  } else {
    cuda_backend::Launch(MuonColMeanSquareKernel, dim3(col_blocks, num_params),
                         dim3(kThreads), 0, rows, cols, x, vmean);
  }
  cuda_backend::Launch(MuonNorMuonMomentsKernel, dim3(num_params),
                       dim3(kThreads), 0, red_index, red_size, params.beta2,
                       vmean, buf2, scale);
  if (mat % kVectorElems == 0 && VectorReady(total, {x, vmean, scale})) {
    const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
    cuda_backend::Launch(MuonNorMuonScaleVectorKernel, dim3(vector_grid),
                         dim3(kThreads), 0, total / kVectorElems, mat, cols,
                         red_index, reduce_cols, vmean, scale,
                         reinterpret_cast<ComputeVector*>(x));
  } else {
    cuda_backend::Launch(MuonNorMuonScaleKernel, dim3(elem_grid),
                         dim3(kThreads), 0, total, mat, cols, red_index,
                         reduce_cols, vmean, scale, x);
  }
  if (VectorReady(total, {x, stacked_master, stacked_values})) {
    const int vector_grid = VectorGrid(total / kVectorElems, kThreads);
    cuda_backend::Launch(MuonApplyVectorKernel, dim3(vector_grid),
                         dim3(kThreads), 0, total / kVectorElems, params.lr,
                         params.weight_decay,
                         reinterpret_cast<FloatVector*>(stacked_master),
                         reinterpret_cast<ComputeVector*>(stacked_values),
                         reinterpret_cast<const ComputeVector*>(x));
  } else {
    cuda_backend::Launch(MuonApplyKernel, dim3(elem_grid), dim3(kThreads), 0,
                         total, params.lr, params.weight_decay, stacked_master,
                         stacked_values, x);
  }
}

}  // namespace kernels
}  // namespace nanochat
