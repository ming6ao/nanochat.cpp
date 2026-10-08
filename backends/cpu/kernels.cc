// backends/cpu/kernels.cc -- the reference implementation of every symbol in
// nanochat/kernels.h (docs/build.md). It is the correctness baseline, the
// oracle-on-CPU path, and the fallback when no accelerator is present.
//
// Most entry points here are naive loops, written for clarity rather than
// speed: double-width accumulators for reductions, float working state, and no
// vectorisation. `Gemm` is the exception: it blocks the output rows and
// columns into register tiles and packs the `b` operand, because the language
// model spends most of its time there. The build selects ComputeType (fp32 or
// fp16) at compile time;
// the arithmetic below is always performed in float and converted at the
// storage boundary through AsFloat / ToCompute.
//
// Layout conventions (matched to the host graphs and docs/model.md):
//   * Q/K/V activations are [batch, seq, heads, head_dim] row-major, the same
//     native layout nanochat's SDPA fallback uses (B, T, H, D).
//   * Attention statistics are [batch, num_heads, seq, 2] with (max, sum_exp).
//   * QkPrep proves the RMSNorm -> RoPE -> scale fusion. Because RoPE is a
//     rotation, it commutes with RMSNorm exactly, so the fusion is numerically
//     equivalent to nanochat's RoPE -> RMSNorm order (docs/kernels.md).
//
// QkPrep backward contract: the `q`/`k` buffers passed to QkPrepBackward hold
// the *saved pre-norm projection outputs* (the activations the model kept for
// backward), and are overwritten with the gradient with respect to those
// projection outputs. The upstream gradients `dq`/`dk` are taken with respect
// to the final (RoPE'd, scaled) activation. Saving the pre-norm rows is what
// makes the normalization statistics recoverable, which is why the header says
// the backward "reads the saved forward outputs ... to recover the
// normalization statistics".

#include "nanochat/kernels.h"
#include "nanochat/device_profile.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace nanochat {
namespace kernels {
namespace {

// Storage <-> float boundary. In an fp16 build the optimizer state and the
// reductions still run in float (docs/build.md). The conversion helpers are
// templated on the storage type so that `if constexpr` discards the unused
// branch instead of requiring both to compile.
template <typename T>
inline float AsFloatT(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return Fp16ToFloat(v);
  }
}

template <typename T>
inline T ToComputeT(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return Fp16FromFloat(v);
  }
}

inline float AsFloat(ComputeType v) { return AsFloatT(v); }

inline ComputeType ToCompute(float v) { return ToComputeT<ComputeType>(v); }

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Zero a typed ComputeType range without tripping -Wclass-memaccess in an
// fp16 build (Fp16 is a non-trivial struct).
inline void ZeroFill(ComputeType* ptr, std::size_t count) {
  std::fill(ptr, ptr + count, ToCompute(0.0f));
}

// Honour the NANOCHAT_NUM_THREADS contract. The sandbox sets it together with
// OMP_NUM_THREADS, but a library thread pool reads NANOCHAT_NUM_THREADS. Apply
// it once, before the first parallel region.
inline void ConfigureThreads() {
#if defined(_OPENMP)
  static const bool configured = [] {
    const char* env = std::getenv("NANOCHAT_NUM_THREADS");
    if (env != nullptr) {
      const int n = std::atoi(env);
      if (n > 0) omp_set_num_threads(n);
    }
    return true;
  }();
  (void)configured;
#endif
}

// Polar Express coefficients (num_iters=5), from nanochat/optim.py / the
// Polar Express paper (arXiv:2505.16932).
constexpr float kPolarCoeffs[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

// ANVIL's six quintic spectral maps, from modded-nanogpt's
// `track_1_short/optim/anvil.py`. Kept in sync with the CUDA family
// (backends/cuda/kernels/anvil.cu) and the host reference
// (backends/cuda/kernels/testing/optim_ref.h).
constexpr float kAnvilMaps[6][3] = {
    {3.923798038567f, -6.095026865488f, 3.905234618423f},
    {3.278126713798f, -3.328923386476f, 0.989127286973f},
    {3.505298394150f, -5.137358782410f, 1.968325560615f},
    {2.815058591845f, -3.685181239622f, 1.417196497642f},
    {2.245503932403f, -2.443826979899f, 0.963091710461f},
    {2.256537145403f, -2.166840097229f, 0.929501253245f},
};

}  // namespace

// ---------------------------------------------------------------------------
// Device / memory (backend-owned)
// ---------------------------------------------------------------------------

void* Alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  return std::malloc(bytes);
}

void Free(void* ptr) { std::free(ptr); }

void Memcpy(void* dst, const void* src, std::size_t bytes, CopyDir dir) {
  // Every copy direction is a plain memcpy on the CPU reference backend.
  (void)dir;
  if (bytes == 0) return;
  std::memcpy(dst, src, bytes);
}

void Memset(void* ptr, int value, std::size_t bytes) {
  if (bytes == 0) return;
  std::memset(ptr, value, bytes);
}

void Synchronize() {
  // The reference backend is synchronous: every entry point returns after its
  // work is complete.
}

Caps GetCaps() {
#if defined(NANOCHAT_SIMULATOR)
  // Reference engine (docs/simulator.md section 4.1): `--config=sim` compiles
  // this branch in, and `NANOCHAT_SIM_PROFILE` names the target device. The
  // host graph then sees a Hopper-class (or Turing/Ampere) device while every
  // value is still computed by the loops below. The profile changes the
  // reported caps, never the arithmetic; `//tests:sim_numerics_test` asserts
  // that equality. An unset or unknown profile falls through to the host caps,
  // so the default CPU behaviour is unchanged.
  if (const char* sim = std::getenv("NANOCHAT_SIM_PROFILE");
      sim != nullptr && *sim != '\0') {
    if (const DeviceProfile* profile = FindDeviceProfile(sim);
        profile != nullptr) {
      return CapsFromProfile(*profile);
    }
  }
#endif
  Caps caps;
  caps.device_index = 0;
  caps.compute_major = 0;
  caps.compute_minor = 0;
  caps.total_memory_bytes = 0;
  caps.is_device = false;
  caps.has_cublas = false;
  caps.has_tensor_cores = false;
  // The CPU reference implements the build precision. It can emulate both, so
  // both are reported as supported; the host fails fast on a build whose
  // precision is not in this set.
  caps.supports_fp32 = true;
  caps.supports_fp16 = true;
  caps.device_name = "cpu";
  return caps;
}
// ---------------------------------------------------------------------------
// Library-backed GEMM
// ---------------------------------------------------------------------------
//
// Universal row-major GEMM: C = alpha * op(A) * op(B) + beta * C, repeated
// `batch_count` times with per-batch strides. `transpose_a`/`transpose_b`
// control the operand layout; `lda`/`ldb`/`ldc` are leading dimensions, where 0
// infers a dense row-major operand. `GemmMode` is advisory: the host supplies
// the already-selected operands, so all three modes compute the same product.
//
// The body packs the `b` panel for one column block into a contiguous buffer.
// It then walks the output rows in `kGemmMr`-row blocks. Each block packs its
// `a` rows and runs a `kGemmMr` by `kGemmNr` register kernel over the whole
// reduction. The reduction stays in `double`, exactly as the shipped loop.
namespace {

// The Gemm register tile: `kGemmMr` output rows by `kGemmNr` output columns
// held in `double` accumulators. The packed `b` panel is `kGemmNc` columns
// wide. The whole reduction stays in one `double` sum, so the rounding matches
// the shipped scalar code.
constexpr int kGemmMr = 4;
constexpr int kGemmNr = 16;
constexpr int kGemmNc = 128;

// One register tile: acc[MR][NR] += A[MR x k] * B[k x NR]. `ap` is the packed
// A row block (`ap[i * ap_stride + l]`); `bp` is the packed B panel
// (`bp[l * bp_stride + j]`). The unroll pragmas keep every accumulator in a
// vector register.
inline void GemmMicroBody(int k, const float* __restrict ap, int ap_stride,
                          const float* __restrict bp, int bp_stride,
                          double acc[kGemmMr][kGemmNr]) {
  for (int l = 0; l < k; ++l) {
    double av[kGemmMr];
#pragma GCC unroll 4
    for (int i = 0; i < kGemmMr; ++i) {
      av[i] = static_cast<double>(ap[i * ap_stride + l]);
    }
    const float* b = bp + static_cast<std::int64_t>(l) * bp_stride;
    double bv[kGemmNr];
#pragma GCC unroll 8
    for (int j = 0; j < kGemmNr; ++j) bv[j] = static_cast<double>(b[j]);
#pragma GCC unroll 4
    for (int i = 0; i < kGemmMr; ++i) {
#pragma GCC unroll 8
      for (int j = 0; j < kGemmNr; ++j) acc[i][j] += av[i] * bv[j];
    }
  }
}

void GemmMicroBase(int k, const float* __restrict ap, int ap_stride,
                   const float* __restrict bp, int bp_stride,
                   double acc[kGemmMr][kGemmNr]) {
  GemmMicroBody(k, ap, ap_stride, bp, bp_stride, acc);
}

#if defined(__x86_64__) || defined(__i386__)
// The same kernel compiled for AVX2 with fused multiply-add. Gemm dispatches to
// it only when the running CPU reports both features, so the baseline binary
// stays portable.
__attribute__((target("avx2,fma"))) void GemmMicroAvx2(
    int k, const float* __restrict ap, int ap_stride,
    const float* __restrict bp, int bp_stride, double acc[kGemmMr][kGemmNr]) {
  GemmMicroBody(k, ap, ap_stride, bp, bp_stride, acc);
}
#endif

using GemmMicroFn = void (*)(int, const float*, int, const float*, int,
                             double (*)[kGemmNr]);

// Resolve the widest micro-kernel the running CPU supports. The answer is
// cached because the feature test reads the CPU identification registers.
GemmMicroFn ResolveGemmMicro() {
#if defined(__x86_64__) || defined(__i386__)
  static const bool avx2_fma =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (avx2_fma) return &GemmMicroAvx2;
#endif
  return &GemmMicroBase;
}

// Pack the `mr` rows of A that start at row `ic` into a contiguous row-major
// block `apack[i * k + l]`. `transpose_a` selects the stored operand layout.
template <typename T>
void PackGemmA(const T* a, int lda, int ic, int mr, int k, bool transpose_a,
               float* apack) {
  if (transpose_a) {
    // a is stored [k, m]: element a[l, ic + i]. Walk l in the outer loop so the
    // rows of each reduction step come from one contiguous line.
    for (int l = 0; l < k; ++l) {
      const T* src = a + static_cast<std::int64_t>(l) * lda + ic;
      for (int i = 0; i < mr; ++i) {
        apack[static_cast<std::int64_t>(i) * k + l] = AsFloatT(src[i]);
      }
    }
  } else {
    // a is stored [m, k]: element a[ic + i, l].
    for (int i = 0; i < mr; ++i) {
      const T* src = a + static_cast<std::int64_t>(ic + i) * lda;
      float* dst = apack + static_cast<std::int64_t>(i) * k;
      for (int l = 0; l < k; ++l) dst[l] = AsFloatT(src[l]);
    }
  }
}

// Pack `groups` full register-width column groups of B into the shared panel.
// Group `g` holds the columns `[g * kGemmNr, (g + 1) * kGemmNr)` with
// `bpanel[g * group_stride + l * kGemmNr + j]`. The register-width inner axis
// is contiguous, so the micro-kernel walks the panel forward. `transpose_b`
// selects the stored operand layout.
template <typename T>
void PackGemmB(const T* b, int ldb, int jc, int groups, int k, int group_stride,
               bool transpose_b, float* bpanel) {
  if (transpose_b) {
    // b is stored [n, k]: element b[jc + g * kGemmNr + j, l].
    for (int g = 0; g < groups; ++g) {
      float* out = bpanel + static_cast<std::int64_t>(g) * group_stride;
      for (int j = 0; j < kGemmNr; ++j) {
        const T* src =
            b + static_cast<std::int64_t>(jc + g * kGemmNr + j) * ldb;
        for (int l = 0; l < k; ++l) {
          out[static_cast<std::int64_t>(l) * kGemmNr + j] = AsFloatT(src[l]);
        }
      }
    }
  } else {
    // b is stored [k, n]: element b[l, jc + g * kGemmNr + j].
    for (int l = 0; l < k; ++l) {
      const T* src = b + static_cast<std::int64_t>(l) * ldb + jc;
      for (int g = 0; g < groups; ++g) {
        float* out = bpanel + static_cast<std::int64_t>(g) * group_stride +
                     static_cast<std::int64_t>(l) * kGemmNr;
        for (int j = 0; j < kGemmNr; ++j) {
          out[j] = AsFloatT(src[g * kGemmNr + j]);
        }
      }
    }
  }
}

// Reduce one output element with the stored layouts. The partial tiles at the
// right and bottom edges use it, where the register tile does not fit.
template <typename T>
float GemmDotProduct(const T* a, const T* b, int lda, int ldb, int i, int j,
                     int k, bool transpose_a, bool transpose_b) {
  double sum = 0.0;
  for (int l = 0; l < k; ++l) {
    const float av =
        AsFloatT(transpose_a ? a[static_cast<std::int64_t>(l) * lda + i]
                             : a[static_cast<std::int64_t>(i) * lda + l]);
    const float bv =
        AsFloatT(transpose_b ? b[static_cast<std::int64_t>(j) * ldb + l]
                             : b[static_cast<std::int64_t>(l) * ldb + j]);
    sum += static_cast<double>(av) * static_cast<double>(bv);
  }
  return static_cast<float>(sum);
}

// Write one output element. A zero `beta` never reads the previous value, which
// matches the shipped contract.
inline ComputeType GemmStore(ComputeType* dst, float partial, float alpha,
                             float beta) {
  if (beta != 0.0f) {
    return ToCompute(alpha * partial + beta * AsFloat(*dst));
  }
  return ToCompute(alpha * partial);
}

}  // namespace

void Gemm(GemmMode mode, const GemmParams& params, const ComputeType* a,
          const ComputeType* b, ComputeType* c) {
  (void)mode;
  const int m = params.m;
  const int n = params.n;
  const int k = params.k;
  if (m <= 0 || n <= 0 || k <= 0) return;

  // Stored extents (before the transpose flag) and inferred leading dims.
  const int rows_a = params.transpose_a ? k : m;
  const int cols_a = params.transpose_a ? m : k;
  const int rows_b = params.transpose_b ? n : k;
  const int cols_b = params.transpose_b ? k : n;
  const int lda = params.lda > 0 ? params.lda : cols_a;
  const int ldb = params.ldb > 0 ? params.ldb : cols_b;
  const int ldc = params.ldc > 0 ? params.ldc : n;

  const std::int64_t stride_a = params.stride_a > 0
                                    ? params.stride_a
                                    : static_cast<std::int64_t>(rows_a) * lda;
  const std::int64_t stride_b = params.stride_b > 0
                                    ? params.stride_b
                                    : static_cast<std::int64_t>(rows_b) * ldb;
  const std::int64_t stride_c = params.stride_c > 0
                                    ? params.stride_c
                                    : static_cast<std::int64_t>(m) * ldc;
  const int batch = params.batch_count > 0 ? params.batch_count : 1;

  const float alpha = params.alpha;
  const float beta = params.beta;
  const bool transpose_a = params.transpose_a;
  const bool transpose_b = params.transpose_b;
  const int panel_groups = (std::min(kGemmNc, n) + kGemmNr - 1) / kGemmNr;
  const int group_stride = k * kGemmNr;
  const GemmMicroFn micro = ResolveGemmMicro();

  for (int bi = 0; bi < batch; ++bi) {
    const ComputeType* ab = a + bi * stride_a;
    const ComputeType* bb = b + bi * stride_b;
    ComputeType* cb = c + bi * stride_c;

    // The packed `b` panel is shared by every worker and rewritten for each
    // column block. The packed `a` block is private to a worker.
    std::vector<float> bpanel(static_cast<std::size_t>(group_stride) *
                              panel_groups);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
      std::vector<float> apack(static_cast<std::size_t>(kGemmMr) * k);
      for (int jc = 0; jc < n; jc += kGemmNc) {
        const int nc = std::min(kGemmNc, n - jc);
#if defined(_OPENMP)
#pragma omp single
#endif
        {
          PackGemmB(bb, ldb, jc, nc / kGemmNr, k, group_stride, transpose_b,
                    bpanel.data());
        }
        // The `single` construct above has an implicit barrier, so every worker
        // sees the finished panel.
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int ic = 0; ic < m; ic += kGemmMr) {
          const int mr = std::min(kGemmMr, m - ic);
          if (mr == kGemmMr) {
            PackGemmA(ab, lda, ic, mr, k, transpose_a, apack.data());
            const int groups = nc / kGemmNr;
            for (int g = 0; g < groups; ++g) {
              double acc[kGemmMr][kGemmNr] = {};
              const float* bp =
                  bpanel.data() + static_cast<std::int64_t>(g) * group_stride;
              micro(k, apack.data(), k, bp, kGemmNr, acc);
              for (int i = 0; i < kGemmMr; ++i) {
                ComputeType* crow = cb +
                                    static_cast<std::int64_t>(ic + i) * ldc +
                                    jc + g * kGemmNr;
                for (int j = 0; j < kGemmNr; ++j) {
                  crow[j] = GemmStore(crow + j, static_cast<float>(acc[i][j]),
                                      alpha, beta);
                }
              }
            }
            // Partial column tile at the right edge.
            for (int j = groups * kGemmNr; j < nc; ++j) {
              for (int i = 0; i < kGemmMr; ++i) {
                ComputeType* crow =
                    cb + static_cast<std::int64_t>(ic + i) * ldc + jc + j;
                const float partial =
                    GemmDotProduct(ab, bb, lda, ldb, ic + i, jc + j, k,
                                   transpose_a, transpose_b);
                crow[0] = GemmStore(crow, partial, alpha, beta);
              }
            }
          } else {
            // Partial row tile at the bottom edge.
            for (int i = 0; i < mr; ++i) {
              ComputeType* crow =
                  cb + static_cast<std::int64_t>(ic + i) * ldc + jc;
              for (int j = 0; j < nc; ++j) {
                const float partial =
                    GemmDotProduct(ab, bb, lda, ldb, ic + i, jc + j, k,
                                   transpose_a, transpose_b);
                crow[j] = GemmStore(crow + j, partial, alpha, beta);
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

void RmsNormForward(const RmsNormParams& params, const ComputeType* x,
                    ComputeType* out, float* rstd) {
  const int rows = params.rows;
  const int dim = params.dim;
  if (rows <= 0 || dim <= 0) return;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * dim;
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = AsFloat(x[base + d]);
      sum_sq += v * v;
    }
    const float r_row =
        1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
    for (int d = 0; d < dim; ++d) {
      out[base + d] = ToCompute(AsFloat(x[base + d]) * r_row);
    }
    rstd[r] = r_row;
  }
}

void RmsNormBackward(const RmsNormParams& params, const ComputeType* x,
                     const ComputeType* dy, const float* rstd,
                     ComputeType* dx) {
  const int rows = params.rows;
  const int dim = params.dim;
  if (rows <= 0 || dim <= 0) return;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (int r = 0; r < rows; ++r) {
    const std::int64_t base = static_cast<std::int64_t>(r) * dim;
    const float r_row = rstd[r];
    double dot = 0.0;
    for (int d = 0; d < dim; ++d) {
      dot += static_cast<double>(AsFloat(x[base + d])) *
             static_cast<double>(AsFloat(dy[base + d]));
    }
    const float coeff =
        r_row * r_row * r_row * static_cast<float>(dot) * inv_dim;
    for (int d = 0; d < dim; ++d) {
      const float xv = AsFloat(x[base + d]);
      dx[base + d] = ToCompute(r_row * AsFloat(dy[base + d]) - coeff * xv);
    }
  }
}

// ---------------------------------------------------------------------------
// QkPrep (RMSNorm -> RoPE -> scale)
// ---------------------------------------------------------------------------

namespace {

// Applies RMSNorm, RoPE, then scale to one [batch, seq, heads, head_dim] tensor
// in place. Shared by the q and k passes of QkPrepForward.
void QkPrepApply(const QkPrepParams& params, const float* cos, const float* sin,
                 ComputeType* tensor, int heads) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float scale = params.scale;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::int64_t base =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        double sum_sq = 0.0;
        for (int d = 0; d < dim; ++d) {
          const double v = AsFloat(tensor[base + d]);
          sum_sq += v * v;
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
        for (int d = 0; d < dim; ++d) {
          tensor[base + d] = ToCompute(AsFloat(tensor[base + d]) * r);
        }
        const std::int64_t row_cos = static_cast<std::int64_t>(t) * half;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = AsFloat(tensor[base + d]);
          const float x2 = AsFloat(tensor[base + half + d]);
          tensor[base + d] = ToCompute((x1 * c + x2 * s) * scale);
          tensor[base + half + d] = ToCompute((-x1 * s + x2 * c) * scale);
        }
      }
    }
  }
}

// Gradients of one QkPrep tensor. `tensor` holds the saved pre-norm rows on
// entry and the input gradients on return; `grad` is the upstream gradient
// w.r.t. the final activation.
void QkPrepApplyBackward(const QkPrepParams& params, const float* cos,
                         const float* sin, const ComputeType* grad,
                         ComputeType* tensor, int heads) {
  const int batch = params.batch;
  const int seq = params.seq;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const int half = dim / 2;
  const float inv_dim = 1.0f / static_cast<float>(dim);
  const float scale = params.scale;
  std::vector<float> g(dim);
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < heads; ++h) {
        const std::int64_t base =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        // Recover the RMSNorm statistic from the saved pre-norm row, and pull
        // the upstream gradient back through scale and RoPE.
        const std::int64_t row_cos = static_cast<std::int64_t>(t) * half;
        double sum_sq = 0.0;
        double dot = 0.0;
        for (int d = 0; d < half; ++d) {
          const float c = cos[row_cos + d];
          const float s = sin[row_cos + d];
          const float x1 = AsFloat(tensor[base + d]);
          const float x2 = AsFloat(tensor[base + half + d]);
          const float g1 = AsFloat(grad[base + d]);
          const float g2 = AsFloat(grad[base + half + d]);
          // dL/d(normed) = scale * R^T(dL/d(final)).
          g[d] = scale * (g1 * c - g2 * s);
          g[half + d] = scale * (g1 * s + g2 * c);
          sum_sq += static_cast<double>(x1) * x1 + static_cast<double>(x2) * x2;
          dot += static_cast<double>(x1) * g[d] +
                 static_cast<double>(x2) * g[half + d];
        }
        const float r =
            1.0f / std::sqrt(static_cast<float>(sum_sq) * inv_dim + params.eps);
        const float coeff = r * r * r * static_cast<float>(dot) * inv_dim;
        for (int d = 0; d < dim; ++d) {
          tensor[base + d] =
              ToCompute(r * g[d] - coeff * AsFloat(tensor[base + d]));
        }
      }
    }
  }
}

}  // namespace

void QkPrepForward(const QkPrepParams& params, const float* cos,
                   const float* sin, ComputeType* q, ComputeType* k) {
  QkPrepApply(params, cos, sin, q, params.num_heads);
  QkPrepApply(params, cos, sin, k, params.num_kv_heads);
}

void QkPrepBackward(const QkPrepParams& params, const float* cos,
                    const float* sin, const ComputeType* dq,
                    const ComputeType* dk, ComputeType* q, ComputeType* k) {
  QkPrepApplyBackward(params, cos, sin, dq, q, params.num_heads);
  QkPrepApplyBackward(params, cos, sin, dk, k, params.num_kv_heads);
}

// ---------------------------------------------------------------------------
// Attention (causal / sliding-window / group-query)
// ---------------------------------------------------------------------------

namespace {

// Key/value head for query head `h` under GQA. Contiguous grouping, matching
// PyTorch's enable_gqa (each key/value head serves `group` query heads).
inline int KvHead(int h, int num_heads, int num_kv_heads) {
  if (num_kv_heads <= 0 || num_kv_heads >= num_heads) return h;
  const int group = num_heads / num_kv_heads;
  const int kv = h / (group > 0 ? group : 1);
  return std::min(kv, num_kv_heads - 1);
}

// Whether query row at absolute position `qpos` may attend to key `j`.
inline bool KeyAllowed(const AttentionParams& p, std::int64_t qpos,
                       std::int64_t j) {
  if (p.causal && j > qpos) return false;
  if (p.window_left >= 0 && qpos - j > p.window_left) return false;
  if (p.window_right >= 0 && j - qpos > p.window_right) return false;
  return true;
}

}  // namespace

void AttentionForward(const AttentionParams& params, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats) {
  ConfigureThreads();
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));
  const std::int64_t total_rows =
      static_cast<std::int64_t>(batch) * heads * seq;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<float> acc(dim);
    // Cache the scaled score of every allowed key. The softmax pass then reads
    // the cached value instead of recomputing the query-key dot product. Each
    // score is the same float the shipped code compares and exponentiates.
    std::vector<float> scores(kv_len);
#if defined(_OPENMP)
#pragma omp for schedule(guided)
#endif
    for (std::int64_t row = 0; row < total_rows; ++row) {
      const int t = static_cast<int>(row % seq);
      const int h = static_cast<int>((row / seq) % heads);
      const int b = static_cast<int>(row / (seq * heads));
      const int kvh = KvHead(h, heads, kv_heads);
      const std::int64_t qpos = kv_len - seq + t;
      const std::int64_t qbase =
          ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
      // Pass 1: max score over the visible window.
      float row_max = -std::numeric_limits<float>::infinity();
      bool any = false;
      for (std::int64_t j = 0; j < kv_len; ++j) {
        if (!KeyAllowed(params, qpos, j)) continue;
        const std::int64_t kbase =
            ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
            dim;
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
          dot += static_cast<double>(AsFloat(q[qbase + d])) *
                 static_cast<double>(AsFloat(k[kbase + d]));
        }
        const float score = scale * static_cast<float>(dot);
        scores[j] = score;
        if (score > row_max) row_max = score;
        any = true;
      }
      const std::int64_t stat_base =
          ((static_cast<std::int64_t>(b) * heads + h) * seq + t) * 2;
      if (!any) {
        for (int d = 0; d < dim; ++d) out[qbase + d] = ToCompute(0.0f);
        stats[stat_base] = 0.0f;
        stats[stat_base + 1] = 0.0f;
        continue;
      }
      // Pass 2: sum of exponentials and unnormalised output.
      double sum_exp = 0.0;
      for (int d = 0; d < dim; ++d) acc[d] = 0.0f;
      for (std::int64_t j = 0; j < kv_len; ++j) {
        if (!KeyAllowed(params, qpos, j)) continue;
        const std::int64_t kbase =
            ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
            dim;
        const std::int64_t vbase = kbase;
        const double weight =
            std::exp(static_cast<double>(scores[j] - row_max));
        sum_exp += weight;
        for (int d = 0; d < dim; ++d) {
          acc[d] += static_cast<float>(weight * AsFloat(v[vbase + d]));
        }
      }
      const float inv = static_cast<float>(1.0 / sum_exp);
      for (int d = 0; d < dim; ++d) {
        out[qbase + d] = ToCompute(acc[d] * inv);
      }
      stats[stat_base] = row_max;
      stats[stat_base + 1] = static_cast<float>(sum_exp);
    }
  }
}

void AttentionBackward(const AttentionParams& params, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv) {
  ConfigureThreads();
  const int batch = params.batch;
  const int seq = params.seq;
  const int heads = params.num_heads;
  const int kv_heads = params.num_kv_heads > 0 ? params.num_kv_heads : heads;
  const int dim = params.head_dim;
  if (batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return;
  const std::int64_t kv_len =
      params.kv_len > 0 ? params.kv_len : static_cast<std::int64_t>(seq);
  if (kv_len <= 0) return;
  const float scale = params.scale > 0.0f
                          ? params.scale
                          : 1.0f / std::sqrt(static_cast<float>(dim));

  const std::size_t q_count =
      static_cast<std::size_t>(batch) * seq * heads * dim;
  const std::size_t kv_count =
      static_cast<std::size_t>(batch) * kv_len * kv_heads * dim;
  ZeroFill(dq, q_count);
  ZeroFill(dk, kv_count);
  ZeroFill(dv, kv_count);

  const std::int64_t total_rows =
      static_cast<std::int64_t>(batch) * heads * seq;
  // `probs[row, j]` is the softmax probability and `ds[row, j]` is the
  // softmax-gradient term. Phase 1 fills both while it accumulates dq. Phase 2
  // reduces them into dk and dv. Both phases keep the shipped summation order.
  const std::size_t red_size = static_cast<std::size_t>(total_rows) * kv_len;
  std::vector<float> probs(red_size, 0.0f);
  std::vector<float> ds(red_size, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<float> dp(kv_len);
#if defined(_OPENMP)
#pragma omp for schedule(guided)
#endif
    for (std::int64_t row = 0; row < total_rows; ++row) {
      const int t = static_cast<int>(row % seq);
      const int h = static_cast<int>((row / seq) % heads);
      const int b = static_cast<int>(row / (seq * heads));
      const int kvh = KvHead(h, heads, kv_heads);
      const std::int64_t qpos = kv_len - seq + t;
      const std::int64_t qbase =
          ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
      const std::int64_t stat_base =
          ((static_cast<std::int64_t>(b) * heads + h) * seq + t) * 2;
      const float row_max = stats[stat_base];
      const float sum_exp = stats[stat_base + 1];
      if (sum_exp <= 0.0f) continue;
      const float inv = 1.0f / sum_exp;
      const std::size_t idx0 = static_cast<std::size_t>(row) * kv_len;
      float weighted_dp = 0.0f;
      for (std::int64_t j = 0; j < kv_len; ++j) {
        dp[j] = 0.0f;
        if (!KeyAllowed(params, qpos, j)) continue;
        const std::int64_t kbase =
            ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
            dim;
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
          dot += static_cast<double>(AsFloat(q[qbase + d])) *
                 static_cast<double>(AsFloat(k[kbase + d]));
        }
        const float p =
            std::exp(scale * static_cast<float>(dot) - row_max) * inv;
        probs[idx0 + static_cast<std::size_t>(j)] = p;
        double dpd = 0.0;
        for (int d = 0; d < dim; ++d) {
          dpd += static_cast<double>(AsFloat(dout[qbase + d])) *
                 static_cast<double>(AsFloat(v[kbase + d]));
        }
        dp[j] = static_cast<float>(dpd);
        weighted_dp += p * dp[j];
      }
      for (std::int64_t j = 0; j < kv_len; ++j) {
        if (!KeyAllowed(params, qpos, j)) continue;
        const std::int64_t kbase =
            ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) *
            dim;
        const float p = probs[idx0 + static_cast<std::size_t>(j)];
        const float dsj = p * (dp[j] - weighted_dp);
        ds[idx0 + static_cast<std::size_t>(j)] = dsj;
        const float dp_scale = dsj * scale;
        for (int d = 0; d < dim; ++d) {
          const float kv = AsFloat(k[kbase + d]);
          dq[qbase + d] = ToCompute(AsFloat(dq[qbase + d]) + dp_scale * kv);
        }
      }
    }
  }

  // Phase 2: accumulate dk and dv. A key/value row is owned by one worker, so
  // no atomics are needed. The inner order (query head, then query row) matches
  // the shipped accumulation order.
  const int group = (kv_heads > 0 && kv_heads < heads) ? heads / kv_heads : 1;
  const std::int64_t total_keys =
      static_cast<std::int64_t>(batch) * kv_heads * kv_len;
#if defined(_OPENMP)
#pragma omp parallel for schedule(guided)
#endif
  for (std::int64_t key = 0; key < total_keys; ++key) {
    const int j = static_cast<int>(key % kv_len);
    const int kvh = static_cast<int>((key / kv_len) % kv_heads);
    const int b = static_cast<int>(key / (kv_len * kv_heads));
    int h_start;
    int h_end;
    if (kv_heads > 0 && kv_heads < heads) {
      h_start = kvh * group;
      h_end = std::min(heads, h_start + group);
    } else {
      h_start = kvh;
      h_end = std::min(heads, kvh + 1);
    }
    if (h_start >= h_end) continue;
    const std::int64_t kbase =
        ((static_cast<std::int64_t>(b) * kv_len + j) * kv_heads + kvh) * dim;
    for (int h = h_start; h < h_end; ++h) {
      for (int t = 0; t < seq; ++t) {
        const std::int64_t qpos = kv_len - seq + t;
        if (!KeyAllowed(params, qpos, j)) continue;
        const std::int64_t qbase =
            ((static_cast<std::int64_t>(b) * seq + t) * heads + h) * dim;
        const std::int64_t row =
            (static_cast<std::int64_t>(b) * heads + h) * seq + t;
        const std::size_t idx = static_cast<std::size_t>(row) * kv_len +
                                static_cast<std::size_t>(j);
        const float p = probs[idx];
        const float dp_scale = ds[idx] * scale;
        for (int d = 0; d < dim; ++d) {
          const float qv = AsFloat(q[qbase + d]);
          const float dov = AsFloat(dout[qbase + d]);
          dk[kbase + d] = ToCompute(AsFloat(dk[kbase + d]) + dp_scale * qv);
          dv[kbase + d] = ToCompute(AsFloat(dv[kbase + d]) + p * dov);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Pointwise
// ---------------------------------------------------------------------------

void PointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                      const ComputeType* b, float alpha, float beta,
                      ComputeType* out) {
  for (int i = 0; i < n; ++i) {
    const float av = AsFloat(a[i]);
    float result = 0.0f;
    switch (op) {
      case PointwiseOp::kScale:
        result = alpha * av;
        break;
      case PointwiseOp::kScaleAdd:
        result = alpha * av + beta * AsFloat(b[i]);
        break;
      case PointwiseOp::kGateMul:
        result = Sigmoid(av) * AsFloat(b[i]);
        break;
      case PointwiseOp::kReluSquare: {
        const float relu = av > 0.0f ? av : 0.0f;
        result = relu * relu;
        break;
      }
      case PointwiseOp::kSoftcap: {
        const float cap = alpha;
        result = cap * std::tanh(av / cap);
        break;
      }
    }
    out[i] = ToCompute(result);
  }
}

void PointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy, float alpha,
                       float beta, ComputeType* da, ComputeType* db) {
  for (int i = 0; i < n; ++i) {
    const float av = AsFloat(a[i]);
    const float dyv = AsFloat(dy[i]);
    switch (op) {
      case PointwiseOp::kScale:
        if (da != nullptr) da[i] = ToCompute(alpha * dyv);
        break;
      case PointwiseOp::kScaleAdd:
        if (da != nullptr) da[i] = ToCompute(alpha * dyv);
        if (db != nullptr) db[i] = ToCompute(beta * dyv);
        break;
      case PointwiseOp::kGateMul: {
        const float sig = Sigmoid(av);
        const float bv = AsFloat(b[i]);
        if (da != nullptr) da[i] = ToCompute(dyv * bv * sig * (1.0f - sig));
        if (db != nullptr) db[i] = ToCompute(dyv * sig);
        break;
      }
      case PointwiseOp::kReluSquare:
        if (da != nullptr) {
          da[i] = ToCompute(av > 0.0f ? dyv * 2.0f * av : 0.0f);
        }
        break;
      case PointwiseOp::kSoftcap: {
        if (da != nullptr) {
          const float cap = alpha;
          const float t = std::tanh(av / cap);
          da[i] = ToCompute(dyv * (1.0f - t * t));
        }
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Classifier (logit softcap + cross-entropy + vocab slice)
// ---------------------------------------------------------------------------

void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses) {
  ConfigureThreads();
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  if (rows <= 0 || vocab <= 0) return;
  const float cap = params.softcap;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    // The softcap is computed once per element, then the max and the sum walk
    // the cached values. The arithmetic is the shipped arithmetic.
    std::vector<float> z(vocab);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
      const int target = targets[r];
      if (target == params.ignore_index) {
        losses[r] = ToCompute(0.0f);
        continue;
      }
      const std::int64_t base = static_cast<std::int64_t>(r) * padded;
      // Softcap only the real vocabulary; the padded tail is never read.
      float row_max = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < vocab; ++j) {
        const float zj = cap * std::tanh(AsFloat(logits[base + j]) / cap);
        z[j] = zj;
        if (zj > row_max) row_max = zj;
      }
      double sum_exp = 0.0;
      for (int j = 0; j < vocab; ++j) {
        sum_exp += std::exp(static_cast<double>(z[j] - row_max));
      }
      const float loss =
          static_cast<float>(std::log(sum_exp)) + row_max - z[target];
      losses[r] = ToCompute(loss);
    }
  }
}

void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits) {
  ConfigureThreads();
  const int rows = params.rows;
  const int vocab = params.vocab_size;
  const int padded =
      params.padded_vocab_size > 0 ? params.padded_vocab_size : vocab;
  if (rows <= 0 || vocab <= 0) return;
  const float cap = params.softcap;
  // Optional per-row weights (docs/post-training.md section 2.2). A null
  // pointer keeps the unweighted arithmetic exactly as it was.
  const float* row_scale = params.row_scale;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    // `probs` holds the softcapped logit and `sech2` holds 1 - tanh^2, so each
    // element needs one tanh. The two exponentials of the shipped code stay
    // separate: the sum uses the double overload and the output uses the
    // float overload, exactly as the shipped code does.
    std::vector<float> probs(vocab);
    std::vector<float> sech2(vocab);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
      const std::int64_t base = static_cast<std::int64_t>(r) * padded;
      const int target = targets[r];
      if (target == params.ignore_index) {
        for (int j = 0; j < padded; ++j) dlogits[base + j] = ToCompute(0.0f);
        continue;
      }
      float row_max = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < vocab; ++j) {
        const float t = std::tanh(AsFloat(logits[base + j]) / cap);
        sech2[j] = 1.0f - t * t;
        const float z = cap * t;
        probs[j] = z;
        if (z > row_max) row_max = z;
      }
      double sum_exp = 0.0;
      for (int j = 0; j < vocab; ++j) {
        sum_exp += std::exp(static_cast<double>(probs[j] - row_max));
      }
      const float inv = static_cast<float>(1.0 / sum_exp);
      if (row_scale != nullptr) {
        const float scale = row_scale[r];
        for (int j = 0; j < vocab; ++j) {
          const float p = std::exp(probs[j] - row_max) * inv;
          const float onehot = (j == target) ? 1.0f : 0.0f;
          // dL/d(raw logit) = (softmax - onehot) * d(softcap)/d(raw), then
          // the per-row weight multiplies the whole row.
          dlogits[base + j] = ToCompute((p - onehot) * sech2[j] * scale);
        }
      } else {
        for (int j = 0; j < vocab; ++j) {
          const float p = std::exp(probs[j] - row_max) * inv;
          const float onehot = (j == target) ? 1.0f : 0.0f;
          // dL/d(raw logit) = (softmax - onehot) * d(softcap)/d(raw).
          dlogits[base + j] = ToCompute((p - onehot) * sech2[j]);
        }
      }
      // The padded vocabulary tail is always zero.
      for (int j = vocab; j < padded; ++j) dlogits[base + j] = ToCompute(0.0f);
    }
  }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out) {
  if (tokens <= 0 || dim <= 0) return;
  for (int i = 0; i < tokens; ++i) {
    const std::int64_t src = static_cast<std::int64_t>(ids[i]) * dim;
    const std::int64_t dst = static_cast<std::int64_t>(i) * dim;
    for (int d = 0; d < dim; ++d) out[dst + d] = table[src + d];
  }
}

void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, ComputeType* dtable) {
  if (tokens <= 0 || dim <= 0) return;
  // Add every token row to the dense gradient buffer. The caller clears the
  // buffer once with ZeroGrad, so repeated backward calls sum. A duplicate id
  // adds every occurrence. The scatter-add needs no zeroing pass
  // (docs/model.md).
  for (int i = 0; i < tokens; ++i) {
    ComputeType* row = dtable + static_cast<std::int64_t>(ids[i]) * dim;
    const std::int64_t src = static_cast<std::int64_t>(i) * dim;
    for (int d = 0; d < dim; ++d) {
      row[d] = ToCompute(AsFloat(row[d]) + AsFloat(dout[src + d]));
    }
  }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

void AdamWUpdate(int n, const AdamWParams& params, ComputeType* p,
                 const ComputeType* g, float* m, float* v) {
  if (n <= 0) return;
  const float bias1 =
      1.0f - std::pow(params.beta1, static_cast<float>(params.step));
  const float bias2 =
      1.0f - std::pow(params.beta2, static_cast<float>(params.step));
  const float step_size = params.lr / bias1;
  for (int i = 0; i < n; ++i) {
    const float grad = AsFloat(g[i]);
    // Decoupled weight decay, applied to the parameter before the update.
    float pi = AsFloat(p[i]) * (1.0f - params.lr * params.weight_decay);
    const float mi = m[i] + (1.0f - params.beta1) * (grad - m[i]);
    const float vi = v[i] + (1.0f - params.beta2) * (grad * grad - v[i]);
    m[i] = mi;
    v[i] = vi;
    const float denom = std::sqrt(vi / bias2) + params.eps;
    pi -= step_size * (mi / denom);
    p[i] = ToCompute(pi);
  }
}

// ---------------------------------------------------------------------------
// Muon (momentum -> Polar Express -> variance reduction -> cautious update)
// ---------------------------------------------------------------------------

namespace {

// Float matrix multiply used by MuonUpdate. The Muon working state is always
// float, so the tuned `Gemm` (whose operands are ComputeType) is only callable
// when ComputeType is float. The fp32 build forwards to `Gemm` and keeps its
// double accumulator; the fp16 build keeps a portable double-accumulator
// fallback.
void MuonMatmul(int m, int n, int k, const float* a, const float* b, float* c,
                bool transpose_a, bool transpose_b) {
#if defined(NANOCHAT_PRECISION_FP16)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int l = 0; l < k; ++l) {
        const float av = transpose_a ? a[l * m + i] : a[i * k + l];
        const float bv = transpose_b ? b[j * k + l] : b[l * n + j];
        sum += static_cast<double>(av) * static_cast<double>(bv);
      }
      c[i * n + j] = static_cast<float>(sum);
    }
  }
#else
  GemmParams params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.transpose_a = transpose_a;
  params.transpose_b = transpose_b;
  Gemm(GemmMode::kForward, params, a, b, c);
#endif
}

}  // namespace

void MuonUpdate(const MuonParams& params, const ComputeType* stacked_grads,
                ComputeType* stacked_params, float* buf1, float* buf2) {
  ConfigureThreads();
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;
  const std::size_t mat = static_cast<std::size_t>(rows) * cols;
  const int ns_steps = std::min(std::max(params.ns_steps, 0), 5);
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);
  const bool tall = rows > cols;

  const int k_extent = std::min(rows, cols);
  std::vector<float> x(mat, 0.0f);
  std::vector<float> a_mat(static_cast<std::size_t>(k_extent) * k_extent, 0.0f);
  std::vector<float> a2_mat(static_cast<std::size_t>(k_extent) * k_extent,
                            0.0f);
  std::vector<float> b_mat(static_cast<std::size_t>(k_extent) * k_extent, 0.0f);
  std::vector<float> prod(mat, 0.0f);

  for (int m = 0; m < num_params; ++m) {
    const std::size_t off = static_cast<std::size_t>(m) * mat;
    const ComputeType* grad = stacked_grads + off;
    ComputeType* param = stacked_params + off;
    float* momentum_buf = buf1 + off;
    float* second_buf =
        buf2 + static_cast<std::size_t>(m) * (reduce_cols ? rows : cols);

    // Nesterov momentum: update the first moment, then the accelerated
    // gradient.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < mat; ++i) {
      const float gr = AsFloat(grad[i]);
      momentum_buf[i] =
          momentum_buf[i] + (1.0f - params.momentum) * (gr - momentum_buf[i]);
      x[i] = params.nesterov ? (1.0f - params.momentum) * gr +
                                   params.momentum * momentum_buf[i]
                             : momentum_buf[i];
    }

    // MuonEq row equilibration: rescale each row to the mean row norm.
    {
      double frob_sq = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : frob_sq)
#endif
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float target = static_cast<float>(std::sqrt(frob_sq)) /
                           std::sqrt(static_cast<float>(rows));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int r = 0; r < rows; ++r) {
        double row_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          row_sq += static_cast<double>(xv) * xv;
        }
        const float row_norm =
            std::max(static_cast<float>(std::sqrt(row_sq)), 1e-6f);
        const float s = target / row_norm;
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *= s;
        }
      }
    }

    // Normalise the Frobenius norm before the polar iterations.
    {
      double frob_sq = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : frob_sq)
#endif
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float div = static_cast<float>(std::sqrt(frob_sq)) * 1.01f + 1e-6f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::size_t i = 0; i < mat; ++i) x[i] /= div;
    }

    // Polar Express orthogonalisation.
    for (int it = 0; it < ns_steps; ++it) {
      const float ca = kPolarCoeffs[it][0];
      const float cb = kPolarCoeffs[it][1];
      const float cc = kPolarCoeffs[it][2];
      if (tall) {
        // A = X^T X  (cols x cols)
        MuonMatmul(cols, cols, rows, x.data(), x.data(), a_mat.data(), true,
                   false);
        MuonMatmul(cols, cols, cols, a_mat.data(), a_mat.data(), a2_mat.data(),
                   false, false);
        for (int i = 0; i < cols; ++i) {
          for (int j = 0; j < cols; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * cols + j;
            b_mat[ij] = cb * a_mat[ij] + cc * a2_mat[ij];
          }
        }
        MuonMatmul(rows, cols, cols, x.data(), b_mat.data(), prod.data(), false,
                   false);
        for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
      } else {
        // A = X X^T  (rows x rows)
        MuonMatmul(rows, rows, cols, x.data(), x.data(), a_mat.data(), false,
                   true);
        MuonMatmul(rows, rows, rows, a_mat.data(), a_mat.data(), a2_mat.data(),
                   false, false);
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < rows; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * rows + j;
            b_mat[ij] = cb * a_mat[ij] + cc * a2_mat[ij];
          }
        }
        MuonMatmul(rows, cols, rows, b_mat.data(), x.data(), prod.data(), false,
                   false);
        for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
      }
    }

    // Muon+ renormalisation: snap the Frobenius norm to sqrt(min(rows, cols)).
    {
      double frob_sq = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : frob_sq)
#endif
      for (std::size_t i = 0; i < mat; ++i) {
        frob_sq += static_cast<double>(x[i]) * x[i];
      }
      const float target_norm =
          std::sqrt(static_cast<float>(std::min(rows, cols)));
      const float scale =
          target_norm / std::max(static_cast<float>(std::sqrt(frob_sq)), 1e-6f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::size_t i = 0; i < mat; ++i) x[i] *= scale;
    }

    // NorMuon variance reduction with a factored second moment.
    if (reduce_cols) {
      const std::size_t red_size = static_cast<std::size_t>(cols);
      std::vector<float> v_mean(rows, 0.0f);
      double sum_vmean = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : sum_vmean)
#endif
      for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          s += static_cast<double>(xv) * xv;
        }
        v_mean[r] = static_cast<float>(s / red_size);
        sum_vmean += v_mean[r];
      }
      const float v_norm = std::sqrt(static_cast<float>(sum_vmean) *
                                     static_cast<float>(red_size));
      std::vector<float> step_size(rows, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int r = 0; r < rows; ++r) {
        second_buf[r] =
            second_buf[r] + (1.0f - params.beta2) * (v_mean[r] - second_buf[r]);
        step_size[r] = 1.0f / std::sqrt(std::max(second_buf[r], 1e-10f));
      }
      double sum_scaled = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : sum_scaled)
#endif
      for (int r = 0; r < rows; ++r) {
        sum_scaled +=
            static_cast<double>(v_mean[r] * static_cast<float>(red_size)) *
            step_size[r] * step_size[r];
      }
      const float v_norm_new = std::sqrt(static_cast<float>(sum_scaled));
      const float denom = std::max(v_norm_new, 1e-10f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int r = 0; r < rows; ++r) {
        const float final_scale = step_size[r] * (v_norm / denom);
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *= final_scale;
        }
      }
    } else {
      const std::size_t red_size = static_cast<std::size_t>(rows);
      std::vector<float> v_mean(cols, 0.0f);
      double sum_vmean = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : sum_vmean)
#endif
      for (int c = 0; c < cols; ++c) {
        double s = 0.0;
        for (int r = 0; r < rows; ++r) {
          const float xv = x[static_cast<std::size_t>(r) * cols + c];
          s += static_cast<double>(xv) * xv;
        }
        v_mean[c] = static_cast<float>(s / red_size);
        sum_vmean += v_mean[c];
      }
      const float v_norm = std::sqrt(static_cast<float>(sum_vmean) *
                                     static_cast<float>(red_size));
      std::vector<float> step_size(cols, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int c = 0; c < cols; ++c) {
        second_buf[c] =
            second_buf[c] + (1.0f - params.beta2) * (v_mean[c] - second_buf[c]);
        step_size[c] = 1.0f / std::sqrt(std::max(second_buf[c], 1e-10f));
      }
      double sum_scaled = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : sum_scaled)
#endif
      for (int c = 0; c < cols; ++c) {
        sum_scaled +=
            static_cast<double>(v_mean[c] * static_cast<float>(red_size)) *
            step_size[c] * step_size[c];
      }
      const float v_norm_new = std::sqrt(static_cast<float>(sum_scaled));
      const float denom = std::max(v_norm_new, 1e-10f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(r) * cols + c] *=
              step_size[c] * (v_norm / denom);
        }
      }
    }

    // Cautious weight decay + parameter update.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < mat; ++i) {
      const float pv = AsFloat(param[i]);
      const float gv = x[i];
      const float decay =
          (gv * pv >= 0.0f) ? params.lr * params.weight_decay * pv : 0.0f;
      param[i] = ToCompute(pv - params.lr * gv - decay);
    }
  }
}

// ---------------------------------------------------------------------------
// ANVIL (twin-rail momentum -> whitening cascade -> lane equalizer -> update)
// ---------------------------------------------------------------------------
//
// One ANVIL update for a stacked group, mirroring modded-nanogpt's
// `track_1_short/optim/anvil.py` and docs/optimizer-anvil-design.md. The
// working state is float (like Muon's), so the fp32 build can drive `Gemm` and
// the fp16 build keeps a double-accumulator fallback.

void AnvilUpdate(const AnvilParams& params, const ComputeType* stacked_grads,
                 ComputeType* stacked_params, float* velocity,
                 float* lane_energy) {
  ConfigureThreads();
  const int num_params = params.num_params > 0 ? params.num_params : 1;
  const int rows = params.rows;
  const int cols = params.cols;
  if (rows <= 0 || cols <= 0) return;
  const std::size_t mat = static_cast<std::size_t>(rows) * cols;
  const std::size_t rail_stride = static_cast<std::size_t>(num_params) * mat;
  const int num_maps = std::min(std::max(params.num_maps, 0), 6);
  // `red_dim == -1` reduces over columns (one lane per row); `-2` over rows
  // (one lane per column); anything else follows the larger extent, matching
  // MuonUpdate.
  const bool reduce_cols =
      params.red_dim == -1 || (params.red_dim != -2 && rows >= cols);
  const bool tall = rows > cols;
  const int min_dim = std::min(rows, cols);
  const std::size_t min_sq = static_cast<std::size_t>(min_dim) * min_dim;
  const int lane_count = reduce_cols ? rows : cols;
  const int lane_len = reduce_cols ? cols : rows;
  const float inv_lane_len = 1.0f / static_cast<float>(lane_len);

  std::vector<float> x(mat, 0.0f);
  std::vector<float> a_mat(min_sq, 0.0f);
  std::vector<float> a2_mat(min_sq, 0.0f);
  std::vector<float> b_mat(min_sq, 0.0f);
  std::vector<float> prod(mat, 0.0f);
  std::vector<float> lane_power(static_cast<std::size_t>(lane_count), 0.0f);
  std::vector<float> lane_scale(static_cast<std::size_t>(lane_count), 0.0f);

  for (int m = 0; m < num_params; ++m) {
    const std::size_t off = static_cast<std::size_t>(m) * mat;
    const ComputeType* grad = stacked_grads + off;
    ComputeType* param = stacked_params + off;
    float* fast = velocity + off;
    float* slow = velocity + rail_stride + off;
    float* lane = lane_energy + static_cast<std::size_t>(m) * lane_count;

    // Twin-rail momentum plus the Nesterov lookahead.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < mat; ++i) {
      const float gr = AsFloat(grad[i]);
      const float f = fast[i] + (1.0f - params.fast_beta) * (gr - fast[i]);
      const float s = slow[i] + (1.0f - params.slow_beta) * (gr - slow[i]);
      fast[i] = f;
      slow[i] = s;
      const float blend =
          params.fast_weight * f + (1.0f - params.fast_weight) * s;
      x[i] = params.nesterov
                 ? (1.0f - params.momentum) * gr + params.momentum * blend
                 : blend;
    }

    // The first Gram is taken on the unnormalized X; its trace is ||X||_F^2,
    // which gives the Frobenius normalization for free.
    if (tall) {
      MuonMatmul(cols, cols, rows, x.data(), x.data(), a_mat.data(), true,
                 false);
    } else {
      MuonMatmul(rows, rows, cols, x.data(), x.data(), a_mat.data(), false,
                 true);
    }
    double trace = 0.0;
    for (int i = 0; i < min_dim; ++i) {
      trace +=
          static_cast<double>(a_mat[static_cast<std::size_t>(i) * min_dim + i]);
    }
    const float d = std::sqrt(static_cast<float>(trace)) * 1.05f + 1e-6f;
    const float inv_d = 1.0f / d;
    const float inv_d2 = inv_d * inv_d;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < mat; ++i) x[i] *= inv_d;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < min_sq; ++i) a_mat[i] *= inv_d2;

    // The whitening cascade: `B = b*A + c*(A@A)`, then `X = a*X + X@B` (or
    // `X = a*X + B@X` when wide).
    for (int k = 0; k < num_maps; ++k) {
      if (k > 0) {
        if (tall) {
          MuonMatmul(cols, cols, rows, x.data(), x.data(), a_mat.data(), true,
                     false);
        } else {
          MuonMatmul(rows, rows, cols, x.data(), x.data(), a_mat.data(), false,
                     true);
        }
      }
      MuonMatmul(min_dim, min_dim, min_dim, a_mat.data(), a_mat.data(),
                 a2_mat.data(), false, false);
      const float ca = kAnvilMaps[k][0];
      const float cb = kAnvilMaps[k][1];
      const float cc = kAnvilMaps[k][2];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::size_t i = 0; i < min_sq; ++i) {
        b_mat[i] = cb * a_mat[i] + cc * a2_mat[i];
      }
      if (tall) {
        MuonMatmul(rows, cols, cols, x.data(), b_mat.data(), prod.data(), false,
                   false);
      } else {
        MuonMatmul(rows, cols, rows, b_mat.data(), x.data(), prod.data(), false,
                   false);
      }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (std::size_t i = 0; i < mat; ++i) x[i] = ca * x[i] + prod[i];
    }

    // Per-lane energy equalizer (NorMuon's low-rank variance estimate).
    for (int l = 0; l < lane_count; ++l) {
      double s = 0.0;
      if (reduce_cols) {
        for (int c = 0; c < cols; ++c) {
          const float v = x[static_cast<std::size_t>(l) * cols + c];
          s += static_cast<double>(v) * v;
        }
      } else {
        for (int r = 0; r < rows; ++r) {
          const float v = x[static_cast<std::size_t>(r) * cols + l];
          s += static_cast<double>(v) * v;
        }
      }
      lane_power[static_cast<std::size_t>(l)] =
          static_cast<float>(s) * inv_lane_len;
    }
    double sum_power = 0.0;
    for (int l = 0; l < lane_count; ++l) {
      sum_power += lane_power[static_cast<std::size_t>(l)];
    }
    const float pre_norm =
        std::sqrt(static_cast<float>(sum_power) * static_cast<float>(lane_len));
    double sum_post = 0.0;
    for (int l = 0; l < lane_count; ++l) {
      const std::size_t idx = static_cast<std::size_t>(l);
      const float power = lane_power[idx];
      const float energy =
          lane[idx] + (1.0f - params.beta2) * (power - lane[idx]);
      lane[idx] = energy;
      const float gain = 1.0f / std::sqrt(std::max(energy, 1e-10f));
      lane_scale[idx] = gain;
      sum_post += static_cast<double>(power * static_cast<float>(lane_len)) *
                  gain * gain;
    }
    const float inv_post =
        pre_norm / std::max(std::sqrt(static_cast<float>(sum_post)), 1e-10f);
    for (int l = 0; l < lane_count; ++l) {
      lane_scale[static_cast<std::size_t>(l)] *= inv_post;
    }
    if (reduce_cols) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int l = 0; l < lane_count; ++l) {
        const float s = lane_scale[static_cast<std::size_t>(l)];
        for (int c = 0; c < cols; ++c) {
          x[static_cast<std::size_t>(l) * cols + c] *= s;
        }
      }
    } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int l = 0; l < lane_count; ++l) {
        const float s = lane_scale[static_cast<std::size_t>(l)];
        for (int r = 0; r < rows; ++r) {
          x[static_cast<std::size_t>(r) * cols + l] *= s;
        }
      }
    }

    // Cautious (sign-aligned) weight decay gated on the slow rail, then the
    // update. `weight_decay` already carries the outer `lr`, so the product
    // `lr * weight_decay` is the reference's `lr^2` decay coefficient.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < mat; ++i) {
      const float pv = AsFloat(param[i]);
      const float decay =
          (slow[i] * pv >= 0.0f) ? params.lr * params.weight_decay * pv : 0.0f;
      param[i] = ToCompute(pv - decay - params.lr * x[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// GlobalNorm (gradient clipping)
// ---------------------------------------------------------------------------

void GlobalNorm(int n, float clip, ComputeType* grads, float* out_norm) {
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = AsFloat(grads[i]);
    sum_sq += v * v;
  }
  const float norm = static_cast<float>(std::sqrt(sum_sq));
  if (out_norm != nullptr) out_norm[0] = norm;
  if (clip > 0.0f && norm > clip) {
    const float scale = clip / norm;
    for (int i = 0; i < n; ++i) {
      grads[i] = ToCompute(AsFloat(grads[i]) * scale);
    }
  }
}

void ScalarDot(const ComputeType* a, const ComputeType* b, int n,
               ComputeType* out, float scale, bool accumulate) {
  if (n <= 0) return;
  double dot = 0.0;
  for (int i = 0; i < n; ++i) {
    dot +=
        static_cast<double>(AsFloat(a[i])) * static_cast<double>(AsFloat(b[i]));
  }
  const float value = scale * static_cast<float>(dot);
  out[0] = ToCompute(accumulate ? AsFloat(out[0]) + value : value);
}

void ValueGateForward(int rows, int hidden, int num_kv_heads, int head_dim,
                      const ComputeType* h, const ComputeType* ve,
                      const ComputeType* gate_w, ComputeType* v,
                      ComputeType* gate_out) {
  constexpr int kChannels = 12;
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int kv_dim = num_kv_heads * head_dim;
  for (int m = 0; m < rows; ++m) {
    const ComputeType* hrow = h + static_cast<std::int64_t>(m) * hidden;
    for (int kh = 0; kh < num_kv_heads; ++kh) {
      const ComputeType* wrow = gate_w + kh * kChannels;
      float pre = 0.0f;
      for (int j = 0; j < kChannels; ++j) {
        pre += AsFloat(hrow[j]) * AsFloat(wrow[j]);
      }
      const float gate = 3.0f / (1.0f + std::exp(-pre));
      gate_out[static_cast<std::int64_t>(m) * num_kv_heads + kh] =
          ToCompute(gate);
      ComputeType* vrow = v + static_cast<std::int64_t>(m) * kv_dim +
                          static_cast<std::int64_t>(kh) * head_dim;
      const ComputeType* verow = ve + static_cast<std::int64_t>(m) * kv_dim +
                                 static_cast<std::int64_t>(kh) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        vrow[d] = ToCompute(AsFloat(vrow[d]) + gate * AsFloat(verow[d]));
      }
    }
  }
}

void ValueGateBackward(int rows, int hidden, int num_kv_heads, int head_dim,
                       const ComputeType* h, const ComputeType* ve,
                       const ComputeType* gate_w, const ComputeType* gate,
                       const ComputeType* dv, ComputeType* gate_w_grad,
                       ComputeType* dh, ComputeType* dve) {
  constexpr int kChannels = 12;
  if (rows <= 0 || hidden < kChannels || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int kv_dim = num_kv_heads * head_dim;
  for (int m = 0; m < rows; ++m) {
    const ComputeType* hrow = h + static_cast<std::int64_t>(m) * hidden;
    ComputeType* dhrow = dh + static_cast<std::int64_t>(m) * hidden;
    for (int kh = 0; kh < num_kv_heads; ++kh) {
      const std::int64_t off = static_cast<std::int64_t>(m) * kv_dim +
                               static_cast<std::int64_t>(kh) * head_dim;
      const float g =
          AsFloat(gate[static_cast<std::int64_t>(m) * num_kv_heads + kh]);
      double dgate = 0.0;
      for (int d = 0; d < head_dim; ++d) {
        const float dvd = AsFloat(dv[off + d]);
        dgate += static_cast<double>(dvd) *
                 static_cast<double>(AsFloat(ve[off + d]));
        dve[off + d] = ToCompute(g * dvd);
      }
      const float dpre = static_cast<float>(dgate) * g * (1.0f - g / 3.0f);
      for (int j = 0; j < kChannels; ++j) {
        gate_w_grad[kh * kChannels + j] = ToCompute(
            AsFloat(gate_w_grad[kh * kChannels + j]) + dpre * AsFloat(hrow[j]));
        dhrow[j] = ToCompute(AsFloat(dhrow[j]) +
                             dpre * AsFloat(gate_w[kh * kChannels + j]));
      }
    }
  }
}

}  // namespace kernels
}  // namespace nanochat
