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

// The operand element. E4M3 is the format `src/fp8_policy.h` makes eligible and
// the one `precision.md` pins the tolerance of.
using CutlassFp8 = cutlass::float_e4m3_t;

static_assert(sizeof(CutlassFp8) == 1,
              "an fp8 operand is one byte, which is what ResolveGemm's dense "
              "leading dimensions and the alignment rule both assume");

// The tile, as the template arguments the kernel body takes. A cluster of one
// model in the third mode is the batch, so the third extent is one.
using TileShape =
    cute::Shape<cute::Int<kFp8Tile.tile_m>, cute::Int<kFp8Tile.tile_n>,
                cute::Int<kFp8Tile.tile_k>>;
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

// True when this arm can read the call. Three tests, and the third is a
// property of the instruction: the SM90 fp8 GMMA exists only in the K-major
// pair, so A must be plain and B transposed. The other three operand forms of
// `GemmParams` have no fp8 instruction on this architecture, and CUTLASS states
// that as a static assert ("No eligible GMMA operator for request
// configuration"), so the host refuses them and cublasLt keeps those calls.
bool CallSupported(const ScaledGemmParams& params) {
  return params.format_a == Fp8Format::kE4M3 &&
         params.format_b == Fp8Format::kE4M3 && !params.fast_accum &&
         params.scale_a != nullptr && params.scale_b != nullptr &&
         !params.transpose_a && params.transpose_b;
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

// One call, both store forms. Returns false when this arm does not take it, so
// the caller keeps cublasLt.
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
  const auto* operand_a = reinterpret_cast<const CutlassFp8*>(a);
  const auto* operand_b = reinterpret_cast<const CutlassFp8*>(b);
  // The K-major pair, which is the only operand form an fp8 GMMA has.
  return Launch<ElementD, /*TransA=*/false, /*TransB=*/true, TileShape,
                ClusterShape, kFp8Tile.stages, CutlassFp8>(gemm, operand_a,
                                                           operand_b, c, alpha);
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
