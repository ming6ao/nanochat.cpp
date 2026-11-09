#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_KERNEL_CUH_
#define NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_KERNEL_CUH_

// One candidate's kernel body: the mapping from the seam's `GemmParams` to
// CUTLASS's `Arguments`, parameterised by the tile (docs/kernels.md section
// 11.4).
//
// **The mainloop is CUTLASS's.** The copy atoms, the pipeline, the barrier
// scheme, the epilogue, and the tile scheduler all come from the vendored
// header tree. What this file adds is the stride mapping of the seam's four
// operand forms and the eight instantiations one candidate holds.
//
// **Eight instantiations are the seam's shape, not a design choice.** The four
// operand orientations of `GemmParams` are four distinct CUTLASS kernels,
// because a layout tag is a compile-time parameter. The two store types double
// that: a forward or an input gradient stores `ComputeType` and a weight
// gradient stores fp32, which is a different epilogue element.
//
// **Three settings are not the builder's defaults.** `PersistentScheduler`
// keeps a block resident over a run of tiles, which is the half of the
// scheduler that `backends/cuda/gemm/README.md` section 15 measured as the
// larger one. `LinearCombination` is the fused `alpha * acc + beta * C`. The
// stage count is explicit, because it is a candidate field: the registry states
// the count `dev/cutlass/README.md` section 3 measured as fitting the window
// for that tile.
//
// **The operand types differ by name and not by bits.** The seam's `Bf16` is a
// two-byte struct that holds the top half of a float, and CUTLASS's
// `bfloat16_t` is the same two bytes. The static asserts below state that, so a
// change to either type fails to compile instead of reading the wrong number.
//
// **The device facts, the workspace, and the refusal line are shared.** They
// sit here rather than in one candidate's translation unit, because every
// candidate needs them and a second copy would ask the driver a second time per
// call.
//
// This header includes vendor headers, so only a bf16 or fp8 build compiles it.
// The emulation engine maps no Tensor Memory Accelerator and no warpgroup
// instruction (docs/simulator.md section 9.2), so nothing here has an
// off-device leg.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "backends/cuda/device.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/kernels.h"
#include "nanochat/tensor.h"

namespace nanochat {
namespace kernels {
namespace cutlass_detail {

using CutlassBf16 = cutlass::bfloat16_t;

// The two types are the same two bytes. A bf16 is the top half of a float in
// both, and the kernel reads the operands as CUTLASS's type, so this pair of
// asserts is what makes the reinterpret_cast below a statement of fact.
static_assert(sizeof(CutlassBf16) == sizeof(Bf16),
              "the seam's Bf16 and cutlass::bfloat16_t must be the same width");
static_assert(
    alignof(CutlassBf16) == alignof(Bf16),
    "the seam's Bf16 and cutlass::bfloat16_t must share an alignment");

// The vendor type of a seam element. CUTLASS builds a Tensor Memory Accelerator
// descriptor from the element type, so the epilogue's element must be a type it
// knows: its own bf16 and `float`. A seam type handed to it fails to compile
// ("Unknown TMA Format!"), which is why the two asserts above exist and why
// this mapping is a reinterpretation and not a conversion. The bytes do not
// move.
template <class Element>
struct VendorElementOf {
  using type = Element;
};

template <>
struct VendorElementOf<Bf16> {
  using type = CutlassBf16;
};

// The store type's alignment in elements, which is sixteen bytes of it.
template <class Element>
constexpr int AlignmentOf() {
  return static_cast<int>(128 / (8 * sizeof(Element)));
}

// --- The device -------------------------------------------------------------

// What the card answers once, cached for the thread, because the seam is called
// per layer per step and a device query per call would cost more than the
// descriptor it protects.
struct DeviceFacts {
  // True on a compute capability 9.0 part, which is the only one an `sm_90a`
  // cubin runs on.
  bool hopper = false;
  int device_id = 0;
  int sm_count = 0;
};

inline DeviceFacts QueryDeviceFacts() {
  DeviceFacts facts;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return facts;
  facts.device_id = device;
  int major = 0;
  int minor = 0;
  int sm_count = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess) {
    return facts;
  }
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  facts.sm_count = sm_count;
  // An `sm_90a` cubin runs on 9.0 and on nothing else, so the test is exact and
  // not a range. A 10.x part would need its own mainloop.
  facts.hopper = major == 9 && minor == 0;
  return facts;
}

inline const DeviceFacts& Facts() {
  thread_local const DeviceFacts facts = QueryDeviceFacts();
  return facts;
}

// --- The workspace ----------------------------------------------------------
//
// The persistent scheduler asks for a workspace whose size depends on the
// shape, so the buffer grows to the largest call the thread has seen and stays.
// That is the same lifetime rule the cuBLAS plan cache follows in `gemm.cu`,
// and for the same reason: one steady-state allocation instead of one per call.

inline void* WorkspaceFor(std::size_t bytes) {
  static thread_local void* buffer = nullptr;
  static thread_local std::size_t capacity = 0;
  if (bytes == 0) return nullptr;
  if (bytes > capacity) {
    if (buffer != nullptr) Free(buffer);
    buffer = Alloc(bytes);
    capacity = buffer != nullptr ? bytes : 0;
  }
  return buffer;
}

// `status` as a line on stderr, and false so the caller takes the next path.
inline bool Refuse(cutlass::Status status) {
  std::fprintf(stderr,
               "note: the Hopper GEMM refused a call (%s); the hand tile or "
               "cuBLAS serves it\n",
               cutlassGetStatusString(status));
  return false;
}

// --- The candidate's kernel --------------------------------------------------

// The fused `C = alpha * op(A) * op(B) + beta * C`, in the cooperative
// warp-specialized epilogue. C and D are the same buffer, because the seam
// accumulates in place.
template <class ElementD, class TileShape, class ClusterShape>
struct EpilogueOf {
  using Operation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, float, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using Collective = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
      ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, float,
      float, ElementD, cutlass::layout::RowMajor, AlignmentOf<ElementD>(),
      ElementD, cutlass::layout::RowMajor, AlignmentOf<ElementD>(),
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      Operation>::CollectiveOp;
};

// One kernel of the seam: an operand orientation pair and a store type, at one
// candidate's tile.
//
// `ElementAB` is the operand element, and it is a parameter because the fp8 arm
// runs the same body with one-byte operands (docs/kernels.md section 10.4). The
// default is the bf16 pair every row of the registry uses, so no existing
// instantiation changed when the parameter arrived.
//
// The two layout tags are the seam's four operand forms. CUTLASS names a stride
// over the modes [M, K, L] for A and [N, K, L] for B, so `RowMajor` is mode 1
// contiguous for A and mode 0 contiguous for B. The seam states each form by
// its contiguous axis, which is the axis the asserts below name.
template <class ElementD, bool TransA, bool TransB, class TileShape,
          class ClusterShape, int Stages, class ElementAB = CutlassBf16>
struct KernelOf {
  using LayoutA = cute::conditional_t<TransA, cutlass::layout::ColumnMajor,
                                      cutlass::layout::RowMajor>;
  using LayoutB = cute::conditional_t<TransB, cutlass::layout::ColumnMajor,
                                      cutlass::layout::RowMajor>;

  // A plain operand is K contiguous and a transposed one is not; a transposed B
  // is K contiguous and a plain one is not.
  static_assert(cutlass::gemm::detail::is_k_major_A<LayoutA>() == !TransA,
                "A: a plain operand is K contiguous, a transposed one is not");
  static_assert(cutlass::gemm::detail::is_k_major_B<LayoutB>() == TransB,
                "B: a transposed operand is K contiguous, a plain one is not");
  static_assert(Stages > 0,
                "a candidate states the pipeline depth it measured");

  using Epilogue =
      typename EpilogueOf<ElementD, TileShape, ClusterShape>::Collective;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA,
      AlignmentOf<ElementAB>(), ElementAB, LayoutB, AlignmentOf<ElementAB>(),
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Stages>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           Mainloop, Epilogue,
                                           cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;

  // The stride of A over [M, K, L]. A plain A is `k` contiguous, so its M
  // stride is the leading dimension and its K stride is one. A transposed A is
  // the other way round.
  static StrideA StrideAOf(int lda, std::int64_t batch) {
    if constexpr (TransA) {
      return StrideA{cute::Int<1>{}, static_cast<std::int64_t>(lda), batch};
    } else {
      return StrideA{static_cast<std::int64_t>(lda), cute::Int<1>{}, batch};
    }
  }

  // The stride of B over [N, K, L], by the same rule and the other tag.
  static StrideB StrideBOf(int ldb, std::int64_t batch) {
    if constexpr (TransB) {
      return StrideB{static_cast<std::int64_t>(ldb), cute::Int<1>{}, batch};
    } else {
      return StrideB{cute::Int<1>{}, static_cast<std::int64_t>(ldb), batch};
    }
  }

  // The stride of C and D over [M, N, L]. The seam always stores C row major,
  // so N is contiguous whatever the operand tags are.
  static StrideD StrideDOf(int ldc, std::int64_t batch) {
    return StrideD{static_cast<std::int64_t>(ldc), cute::Int<1>{}, batch};
  }
};

// --- The launch -------------------------------------------------------------

// `alpha_dev` is the optional device scalar the epilogue multiplies the
// accumulator by. The fp8 arm passes the product of the two operand scales
// there, so a dequantized call never reads a scale on the host (docs/kernels.md
// section 10.4, and Part 2 section 1 for the rule).
//
// The operands arrive as `void*` and are read as `ElementAB`, which is a vendor
// type and never the seam's. A seam type here would ask CUTLASS for an
// instruction over a struct it has no trait for, and its answer is the static
// assert "No eligible GMMA operator for request configuration".
template <class ElementD, bool TransA, bool TransB, class TileShape,
          class ClusterShape, int Stages, class ElementAB = CutlassBf16>
bool Launch(const ResolvedGemm& gemm, const void* a, const void* b, ElementD* c,
            const float* alpha_dev = nullptr) {
  using VendorD = typename VendorElementOf<ElementD>::type;
  using Kernel = KernelOf<VendorD, TransA, TransB, TileShape, ClusterShape,
                          Stages, ElementAB>;
  using Gemm = typename Kernel::Gemm;

  const auto* operand_a = static_cast<const ElementAB*>(a);
  const auto* operand_b = static_cast<const ElementAB*>(b);
  auto* result = reinterpret_cast<VendorD*>(c);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      cute::Shape<int, int, int, int>{gemm.m, gemm.n, gemm.k, gemm.batch_count},
      {operand_a, Kernel::StrideAOf(gemm.lda, gemm.stride_a), operand_b,
       Kernel::StrideBOf(gemm.ldb, gemm.stride_b)},
      {{},
       result,
       typename Kernel::StrideC{gemm.ldc, cute::Int<1>{}, gemm.stride_c},
       result,
       Kernel::StrideDOf(gemm.ldc, gemm.stride_c)},
      cutlass::KernelHardwareInfo{}};
  args.epilogue.thread.alpha = gemm.alpha;
  args.epilogue.thread.beta = gemm.beta;
  if (alpha_dev != nullptr) {
    // The epilogue loads alpha from device memory instead of the value above,
    // and `alpha` is then the unused seed (Sm90LinearCombination's
    // `alpha_ptr`).
    args.epilogue.thread.alpha_ptr = alpha_dev;
  }
  args.hw_info.device_id = Facts().device_id;
  args.hw_info.sm_count = Facts().sm_count;

  // CUTLASS checks the alignment and extent rules a second time, and it is the
  // authority on what this kernel accepts. `JudgeCutlass` decides whether to
  // ask; this decides whether the answer is yes.
  const cutlass::Status valid = Gemm::can_implement(args);
  if (valid != cutlass::Status::kSuccess) return Refuse(valid);

  void* workspace = WorkspaceFor(Gemm::get_workspace_size(args));
  const cutlass::Status status =
      Gemm().run(args, workspace, cuda_backend::Stream());
  if (status != cutlass::Status::kSuccess) return Refuse(status);
  cuda_backend::CheckLastError("cutlass gemm");
  return true;
}

// One candidate's four operand orientations, at one store type. The four are
// separate kernels, so this is a switch over the seam's two transpose flags.
template <class TileShape, class ClusterShape, int Stages, class ElementD,
          class ElementAB = CutlassBf16>
bool LaunchOriented(const ResolvedGemm& gemm, const void* a, const void* b,
                    ElementD* c, const float* alpha_dev = nullptr) {
  if (!gemm.transpose_a) {
    return gemm.transpose_b
               ? Launch<ElementD, false, true, TileShape, ClusterShape, Stages,
                        ElementAB>(gemm, a, b, c, alpha_dev)
               : Launch<ElementD, false, false, TileShape, ClusterShape, Stages,
                        ElementAB>(gemm, a, b, c, alpha_dev);
  }
  return gemm.transpose_b
             ? Launch<ElementD, true, true, TileShape, ClusterShape, Stages,
                      ElementAB>(gemm, a, b, c, alpha_dev)
             : Launch<ElementD, true, false, TileShape, ClusterShape, Stages,
                      ElementAB>(gemm, a, b, c, alpha_dev);
}

// The active blocks for each multiprocessor that the driver answers for one
// candidate's kernel, or zero when the query failed. The epilogue element and
// the operand form do not change the residency of the cooperative schedule, so
// one instantiation of the candidate answers for all eight.
template <class TileShape, class ClusterShape, int Stages>
int OccupancyOf() {
  using Kernel =
      KernelOf<CutlassBf16, false, false, TileShape, ClusterShape, Stages>;
  // The thread count and the shared-memory size are members of the kernel body,
  // which is the type the driver launches. The device adapter above it states
  // neither.
  using Body = typename Kernel::Kernel;
  int blocks = 0;
  const cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, cutlass::device_kernel<Body>, Body::MaxThreadsPerBlock,
      Body::SharedStorageSize);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return blocks;
}

}  // namespace cutlass_detail
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_CUTLASS_KERNEL_CUH_
