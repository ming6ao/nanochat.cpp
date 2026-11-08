// The device side of the CUTLASS bring-up probe (docs/kernels.md Part 3
// section 5). The header states the three questions; this file answers them
// with the compiler.
//
// **It runs no kernel and needs no card.** Every row is compile-time data:
// `sizeof(Kernel::SharedStorage)` for the window, `DispatchPolicy::Stages` for
// the pipeline depth, and `MaxThreadsPerBlock` for the block. Taking the
// address of `cutlass::device_kernel<Kernel>` forces the kernel body itself to
// be instantiated and compiled, so a mainloop that does not build fails here
// instead of at the adapter.
//
// **The layouts are asserted, not assumed.** CUTLASS names a stride over the
// modes [M, K, L] for A and [N, K, L] for B, and its `RowMajor` tag means
// mode-1 contiguous for A but mode-0 contiguous for B. The seam's four operand
// forms are stated in terms of which axis is contiguous, so the asserts below
// tie each tag to the axis the seam documents. A tag that disagrees would still
// compile, and would silently compute a transposed product.
//
// See docs/kernels.md Part 3 and backends/cuda/gemm/README.md.

#include <cstddef>

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "dev/cutlass/cutlass_probe.h"

#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_MMA_SM90A_ENABLED)
#error \
    "dev/cutlass/cutlass_probe.cu needs sm_90a; the target states archs = [\"sm_90a\"]"
#endif

namespace nanochat {
namespace cutlass_probe {
namespace {

using Bf16 = cutlass::bfloat16_t;

//: The window one Hopper block may ask for: 227 KiB of the 228 KiB the SM has.
constexpr std::size_t kWindowBytes = 227 * 1024;

//: The Hopper tile the probe reports for every seam case. The builder picks the
//: stage count from the window, so the tile and the cluster are the whole
//: statement.
using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

//: The four operand forms of `GemmParams`, as the two tags each one needs.
//: CUTLASS names A over [M, K, L] and B over [N, K, L], so the "RowMajor" tag
//: is mode-1 contiguous for A and mode-0 contiguous for B. The two convenience
//: aliases below read in the seam's own terms instead.
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

//: The store type's alignment, in elements: 16 bytes of the element. A bf16
//: store therefore holds 8, an fp32 store 4, and the epilogue refuses less
//: ("C/D Should meet TMA alignment requirement").
template <class Element>
constexpr int AlignmentOf() {
  return static_cast<int>(128 / (8 * sizeof(Element)));
}

//: The epilogue discharges `C = alpha * op(A) * op(B) + beta * C` in one fused
//: visitor. `ElementC` and `ElementD` are the same buffer, because the seam
//: accumulates in place.
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

//: One kernel of the seam contract: an operand orientation pair and a store
//: type. `StageCountAutoCarveout` leaves the epilogue its share of the window
//: and gives the mainloop the rest.
template <class ElementD, bool TransA, bool TransB, class Tile = TileShape>
struct KernelOf {
  using LayoutA = typename LayoutOfA<TransA>::type;
  using LayoutB = typename LayoutOfB<TransB>::type;
  using Epilogue = typename EpilogueOf<ElementD>::Collective;

  //: The seam's statement of the four operand forms. `k` is the contiguous axis
  //: of a plain A and of a transposed B, and it is the strided axis of the
  //: other two. A tag that disagrees with the caller's transposes would compile
  //: and multiply the wrong operands.
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

  //: The address of the kernel entry. Instantiating it is the point of the
  //: probe: a mainloop that does not compile fails here.
  static const void* Entry() {
    return reinterpret_cast<const void*>(&cutlass::device_kernel<Kernel>);
  }
};

// --- The rows ---------------------------------------------------------------

//: The five tile candidates of the plan, each sized by the builder. The store
//: is fp32, so the epilogue carve-out is the larger of the two.
//:
//: **A tile narrower than 128 in M does not exist on this schedule.** The
//: cooperative kernel asserts it
//: ("Cooperative kernel requires Tile Size to be greater than or equal to 128
//: along the M-dimension",
//: `sm90_gemm_tma_warpspecialized_cooperative.hpp:374`), because two consumer
//: warpgroups split the M extent between them. The plan's candidate list held a
//: 64 by 256 tile, and the first build of this probe rejected it. That is the
//: probe working: the same tile would have failed in the adapter, after the
//: harness was written.
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

const TileReport kTiles[] = {
    {"128x256x64", 128, 256, 64, 2, Tiles::T0::kStages, Tiles::T0::kSharedBytes,
     Tiles::T0::kThreads, Tiles::T0::kSharedBytes <= kWindowBytes},
    {"128x256x32", 128, 256, 32, 2, Tiles::T1::kStages, Tiles::T1::kSharedBytes,
     Tiles::T1::kThreads, Tiles::T1::kSharedBytes <= kWindowBytes},
    {"128x128x64", 128, 128, 64, 2, Tiles::T2::kStages, Tiles::T2::kSharedBytes,
     Tiles::T2::kThreads, Tiles::T2::kSharedBytes <= kWindowBytes},
    {"128x128x128", 128, 128, 128, 2, Tiles::T3::kStages,
     Tiles::T3::kSharedBytes, Tiles::T3::kThreads,
     Tiles::T3::kSharedBytes <= kWindowBytes},
    {"256x128x64", 256, 128, 64, 2, Tiles::T4::kStages, Tiles::T4::kSharedBytes,
     Tiles::T4::kThreads, Tiles::T4::kSharedBytes <= kWindowBytes},
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

const CaseReport kCases[] = {
    {"bf16 store, plain A, plain B", false, false, false,
     Cases::F0::kSharedBytes, Cases::F0::kThreads},
    {"bf16 store, transposed A, plain B", true, false, false,
     Cases::F1::kSharedBytes, Cases::F1::kThreads},
    {"bf16 store, plain A, transposed B", false, true, false,
     Cases::F2::kSharedBytes, Cases::F2::kThreads},
    {"bf16 store, transposed A, transposed B", true, true, false,
     Cases::F3::kSharedBytes, Cases::F3::kThreads},
    {"fp32 store, plain A, plain B", false, false, true,
     Cases::W0::kSharedBytes, Cases::W0::kThreads},
    {"fp32 store, transposed A, plain B", true, false, true,
     Cases::W1::kSharedBytes, Cases::W1::kThreads},
    {"fp32 store, plain A, transposed B", false, true, true,
     Cases::W2::kSharedBytes, Cases::W2::kThreads},
    {"fp32 store, transposed A, transposed B", true, true, true,
     Cases::W3::kSharedBytes, Cases::W3::kThreads},
};

//: The kernel entries, one per row. Naming the address is what compiles the
//: mainloop body: a `__global__` function template is emitted only when
//: something odr-uses it, and a probe that names only the type would compile
//: the class and skip the kernel.
const void* const kTileEntries[] = {
    Tiles::T0::Entry(), Tiles::T1::Entry(), Tiles::T2::Entry(),
    Tiles::T3::Entry(), Tiles::T4::Entry(),
};

const void* const kCaseEntries[] = {
    Cases::F0::Entry(), Cases::F1::Entry(), Cases::F2::Entry(),
    Cases::F3::Entry(), Cases::W0::Entry(), Cases::W1::Entry(),
    Cases::W2::Entry(), Cases::W3::Entry(),
};

}  // namespace

int TileCount() { return static_cast<int>(sizeof(kTiles) / sizeof(kTiles[0])); }

const TileReport& Tile(int index) { return kTiles[index]; }

const void* TileEntry(int index) { return kTileEntries[index]; }

int LiveCases() { return static_cast<int>(sizeof(kCases) / sizeof(kCases[0])); }

const CaseReport& Case(int index) { return kCases[index]; }

const void* CaseEntry(int index) { return kCaseEntries[index]; }

}  // namespace cutlass_probe
}  // namespace nanochat
