// The fp8 arm of the Hopper mainloop (docs/kernels.md section 10.4).
//
// The seam's fp8 entry is `GemmScaledFp8`, which cublasLt serves today. This
// file puts the same call on the warp-specialized mainloop the bf16 candidates
// use, with E4M3 operands and an fp32 accumulator.
//
// **The dequantization reads the device.** `GemmScaledFp8` states one
// dequantization value for each operand as a device pointer, and the tree's
// rule is that such a scalar never reaches the host (docs/kernels.md Part 2
// section 1). One single-thread launch therefore forms `alpha * scale_a *
// scale_b` into a device float, and the epilogue reads alpha from that address
// through `Sm90LinearCombination`'s `alpha_ptr`. No host read and no
// synchronizing copy stands between the quantization and the multiply.
//
// **One tile.** `kFp8Tile` is one row and not a registry entry, because an fp8
// axis over the four bf16 rows would be sixty-four instantiations. One measured
// tile answers the campaign's first question: does an fp8 mainloop beat
// cublasLt at all?
//
// See docs/kernels.md, docs/precision.md Part 1, and
// backends/cuda/gemm/README.md.

#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/cutlass_kernel.cuh"
#include "backends/cuda/gemm/cutlass_launch.h"
#include "backends/cuda/gemm/gemm_shape.h"

namespace nanochat {
namespace kernels {

namespace {

using cutlass_detail::Facts;
using cutlass_detail::Launch;

// The two operand elements the seam states. E4M3 is the forward's pair and the
// second operand of both backward calls; E5M2 is the output gradient.
using Fp8E4M3 = cutlass::float_e4m3_t;
using Fp8E5M2 = cutlass::float_e5m2_t;

static_assert(sizeof(Fp8E4M3) == 1 && sizeof(Fp8E5M2) == 1,
              "an fp8 operand is one byte, which is what ResolveGemm's dense "
              "leading dimensions and the alignment rule both assume");

// The tiles, as the template arguments the kernel body takes. A cluster of one
// model in the third mode is the batch, so the third extent is one.
using ForwardTile =
    cute::Shape<cute::Int<kFp8Tile.tile_m>, cute::Int<kFp8Tile.tile_n>,
                cute::Int<kFp8Tile.tile_k>>;
using BackwardTile = cute::Shape<cute::Int<kFp8BackwardTile.tile_m>,
                                 cute::Int<kFp8BackwardTile.tile_n>,
                                 cute::Int<kFp8BackwardTile.tile_k>>;
using ClusterShape = cute::Shape<cute::Int<kFp8Tile.cluster_m>,
                                 cute::Int<kFp8Tile.cluster_n>, cute::_1>;

// The one floating product the epilogue needs, formed on the device so no scale
// crosses to the host.
__global__ void ScaleProductKernel(const float* __restrict__ scale_a,
                                   const float* __restrict__ scale_b,
                                   float alpha, float* __restrict__ out) {
  out[0] = alpha * scale_a[0] * scale_b[0];
}

// Four bytes of device memory, allocated once for the thread, so the product
// has an address that outlives the launch. It is not part of the GEMM
// workspace: that buffer is CUTLASS's and the epilogue reads this slot while
// the kernel runs, so the two must not alias.
float* AlphaSlot() {
  static thread_local float* slot = nullptr;
  if (slot == nullptr) slot = static_cast<float*>(Alloc(sizeof(float)));
  return slot;
}

// True when this arm can read the call: one of the three forms the seam issues,
// no reduced-precision reduction, and two scales.
bool CallSupported(const ScaledGemmParams& params) {
  return Fp8ScaledFormSupported(params.format_a, params.format_b,
                                params.transpose_a, params.transpose_b) &&
         !params.fast_accum && params.scale_a != nullptr &&
         params.scale_b != nullptr;
}

// The seam's dense call, resolved by the same rules the bf16 path uses. The
// fp8 seam states no leading dimensions, so every one of them is the dense
// default, and alpha is one because the device slot carries it.
ResolvedGemm Resolve(const ScaledGemmParams& params, bool fp32_c) {
  GemmParams dense;
  dense.m = params.m;
  dense.n = params.n;
  dense.k = params.k;
  dense.batch_count = 1;
  dense.alpha = 1.0f;
  dense.beta = params.beta;
  dense.transpose_a = params.transpose_a;
  dense.transpose_b = params.transpose_b;
  return ResolveGemm(dense, fp32_c);
}

// One call, at the form's own tile and operand pair. Returns false when this
// arm does not take it, so the caller keeps cublasLt.
//
// The store is part of the form, and the seam states it: the forward and the
// dgrad store the compute type, and the wgrad stores the fp32 parameter
// gradient. A call with the other store is refused here rather than
// instantiated, which is what keeps the arm at four kernels.
template <class ElementD>
bool LaunchScaled(const ScaledGemmParams& params, const std::uint8_t* a,
                  const std::uint8_t* b, ElementD* c) {
  if (!Facts().hopper) return false;
  if (!CallSupported(params)) return false;
  const bool fp32_c = sizeof(ElementD) == sizeof(float);
  const ResolvedGemm gemm = Resolve(params, fp32_c);
  const CutlassRefusal refusal = JudgeScaledFp8(gemm, a, b, c);
  if (refusal != CutlassRefusal::kNone) return false;

  float* alpha = AlphaSlot();
  if (alpha == nullptr) return false;
  cuda_backend::Launch(ScaleProductKernel, dim3(1), dim3(1), 0, params.scale_a,
                       params.scale_b, params.alpha, alpha);

  if (params.format_a == Fp8Format::kE4M3) {
    // The forward: K-major in both operands, at the shallower tile.
    return Launch<ElementD, /*TransA=*/false, /*TransB=*/true, ForwardTile,
                  ClusterShape, kFp8Tile.stages, Fp8E4M3, Fp8E4M3>(gemm, a, b,
                                                                   c, alpha);
  }
  if (params.transpose_a) {
    // The wgrad: M-major in A, and the fp32 parameter gradient as the store.
    if constexpr (sizeof(ElementD) == sizeof(float)) {
      return Launch<ElementD, /*TransA=*/true, /*TransB=*/false, BackwardTile,
                    ClusterShape, kFp8BackwardTile.stages, Fp8E5M2, Fp8E4M3>(
          gemm, a, b, c, alpha);
    }
    return false;
  }
  // The dgrad: N-major in B, and the compute type as the store.
  if constexpr (sizeof(ElementD) == sizeof(ComputeType)) {
    return Launch<ElementD, /*TransA=*/false, /*TransB=*/false, BackwardTile,
                  ClusterShape, kFp8BackwardTile.stages, Fp8E5M2, Fp8E4M3>(
        gemm, a, b, c, alpha);
  }
  return false;
}

}  // namespace

bool LaunchScaledFp8Cutlass(const ScaledGemmParams& params,
                            const std::uint8_t* a, const std::uint8_t* b,
                            ComputeType* c) {
  return LaunchScaled<ComputeType>(params, a, b, c);
}

bool LaunchScaledFp8CutlassToFp32(const ScaledGemmParams& params,
                                  const std::uint8_t* a, const std::uint8_t* b,
                                  float* c) {
  return LaunchScaled<float>(params, a, b, c);
}

}  // namespace kernels
}  // namespace nanochat
