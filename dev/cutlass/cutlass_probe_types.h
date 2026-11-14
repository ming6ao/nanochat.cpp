#ifndef NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_TYPES_H_
#define NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_TYPES_H_

// The kernel types of the CUTLASS bring-up probe (docs/kernels.md Part 3
// section 5). This header names CUDA and vendor types, so only a device target
// reads it.
//
// **Naming a type is cheap; emitting its kernel body is not.** Measured here at
// CUDA 12.0: the thirteen row types cost 17 s and 1.3 GiB, and one kernel body
// costs 42 s and 3.0 GiB at `-O0`, `-O1`, and `-O2` alike. The bodies therefore
// live in `cutlass_probe_entry.cu`, one per compiler action, and this header
// holds only the types and the two row tables.

#include <cstddef>
#include <tuple>

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_MMA_SM90A_ENABLED)
#error \
    "dev/cutlass/cutlass_probe_types.h needs sm_90a; every probe target states archs = [\"sm_90a\"]"
#endif

namespace nanochat {
namespace cutlass_probe {

using Bf16 = cutlass::bfloat16_t;

//: The Hopper tile every seam case uses. The builder picks the stage count from
//: the window, so the tile and the cluster are the whole statement.
using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

//: CUTLASS names a stride over [M, K, L] for A and [N, K, L] for B, so its
//: `RowMajor` tag means mode-1 contiguous for A and mode-0 contiguous for B.
//: The seam states each operand form by its contiguous axis, and these two
//: aliases read in those terms.
template <bool TransA>
struct LayoutOfA {
  using type = cute::conditional_t<TransA, cutlass::layout::ColumnMajor,
                                   cutlass::layout::RowMajor>;
};

template <bool TransB>
struct LayoutOfB {
  using type = cute::conditional_t<TransB, cutlass::layout::ColumnMajor,
                                   cutlass::layout::RowMajor>;
};

//: The store alignment, in elements: 16 bytes of the element. The epilogue
//: refuses less ("C/D Should meet TMA alignment requirement").
template <class Element>
constexpr int AlignmentOf() {
  return static_cast<int>(128 / (8 * sizeof(Element)));
}

//: `C = alpha * op(A) * op(B) + beta * C` in one fused visitor. C and D are the
//: same buffer, because the seam accumulates in place.
template <class ElementD>
struct EpilogueOf {
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  using Operation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementD, ElementScalar,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using Collective = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
      ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, ElementD, cutlass::layout::RowMajor,
      AlignmentOf<ElementD>(), ElementD, cutlass::layout::RowMajor,
      AlignmentOf<ElementD>(), cutlass::epilogue::TmaWarpSpecializedCooperative,
      Operation>::CollectiveOp;
};

//: One kernel of the seam: an operand orientation pair and a store type.
//: `StageCountAutoCarveout` leaves the epilogue its share of the window. The
//: three `k` values are the row's whole report, and each is compile-time.
template <class ElementD, bool TransA, bool TransB, class Tile = TileShape>
struct KernelOf {
  using LayoutA = typename LayoutOfA<TransA>::type;
  using LayoutB = typename LayoutOfB<TransB>::type;
  using Epilogue = typename EpilogueOf<ElementD>::Collective;

  //: A plain A is K contiguous and a transposed B is; the other two are not. A
  //: tag that disagreed would compile and multiply the wrong operands.
  static_assert(cutlass::gemm::detail::is_k_major_A<LayoutA>() == !TransA,
                "A tag disagrees with the seam: a plain A is K contiguous");
  static_assert(
      cutlass::gemm::detail::is_k_major_B<LayoutB>() == TransB,
      "B tag disagrees with the seam: a transposed B is K contiguous");

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Bf16, LayoutA, 8,
      Bf16, LayoutB, 8, float, Tile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           Mainloop, Epilogue,
                                           cutlass::gemm::PersistentScheduler>;

  static constexpr std::size_t kSharedBytes =
      sizeof(typename Kernel::SharedStorage);
  static constexpr int kThreads = static_cast<int>(Kernel::MaxThreadsPerBlock);
  static constexpr int kStages = Mainloop::DispatchPolicy::Stages;
};

// --- The two row tables -----------------------------------------------------

//: The five tile candidates, each sized by the builder. The store is fp32, so
//: the epilogue carve-out is the larger of the two.
//:
//: The schedule refuses a tile narrower than 128 in M, because two consumer
//: warpgroups split the M extent
//: (`sm90_gemm_tma_warpspecialized_cooperative.hpp:374`). The probe's first
//: build rejected a 64 by 256 candidate for that reason.
struct Tiles {
  using T0 = KernelOf<float, false, false,
                      cute::Shape<cute::_128, cute::_256, cute::_64>>;
  using T1 = KernelOf<float, false, false,
                      cute::Shape<cute::_128, cute::_256, cute::_32>>;
  using T2 = KernelOf<float, false, false,
                      cute::Shape<cute::_128, cute::_128, cute::_64>>;
  using T3 = KernelOf<float, false, false,
                      cute::Shape<cute::_128, cute::_128, cute::_128>>;
  using T4 = KernelOf<float, false, false,
                      cute::Shape<cute::_256, cute::_128, cute::_64>>;
};

//: The eight seam cases at the primary tile. A forward or an input-gradient
//: call stores `ComputeType`; a weight gradient stores fp32.
struct Cases {
  using F0 = KernelOf<Bf16, false, false>;
  using F1 = KernelOf<Bf16, true, false>;
  using F2 = KernelOf<Bf16, false, true>;
  using F3 = KernelOf<Bf16, true, true>;
  using W0 = KernelOf<float, false, false>;
  using W1 = KernelOf<float, true, false>;
  using W2 = KernelOf<float, false, true>;
  using W3 = KernelOf<float, true, true>;
};

//: The kernel body of each build entry, in build order. Rows 0 and 5 name one
//: type (the 128x256x64 tile at an fp32 store is case W0), so there are twelve
//: entries for thirteen rows and `kEntryOfRow` maps a row to its entry.
using Entries = std::tuple<Tiles::T0, Tiles::T1, Tiles::T2, Tiles::T3,
                           Tiles::T4, Cases::F0, Cases::F1, Cases::F2,
                           Cases::F3, Cases::W1, Cases::W2, Cases::W3>;

//: One entry's kernel type, by the index the build compiles.
template <int kEntry>
using EntryType = std::tuple_element_t<kEntry, Entries>;

}  // namespace cutlass_probe
}  // namespace nanochat

#endif  // NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_TYPES_H_
