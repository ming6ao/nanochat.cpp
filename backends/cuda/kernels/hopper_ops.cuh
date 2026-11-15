#ifndef NANOCHAT_BACKENDS_CUDA_KERNELS_HOPPER_OPS_CUH_
#define NANOCHAT_BACKENDS_CUDA_KERNELS_HOPPER_OPS_CUH_

// The four Hopper mechanisms a warp-specialized tile needs, behind one
// interface (docs/kernels.md Part 3 section 4 and docs/attention.md Part 8).
//
// The four are the warpgroup matrix multiply-accumulate, the Tensor Memory
// Accelerator, the `mbarrier` handoff, and the dynamic shared window. This
// header is the one place a kernel names them, so the kernel body states what
// it wants and not how the instruction spells it.
//
// **Three arms, one interface.** A `.cu` source reaches this header from two
// builds, and the arm decides what a call becomes.
//
//   1. The device arm, `__CUDA_ARCH__ >= 900`: the instructions in inline
//      assembly, which only a device pass can parse.
//   2. The emulation arm: the same calls computed on the host by
//      `tools/cuda_sim/emu/hopper_ops_host.h`. The emulator's prelude sets
//      `NANOCHAT_HOPPER_EMU_ARM`, and that macro is what selects it.
//   3. The empty arm: a host pass of nvcc in a device build. The functions
//      exist, so a kernel body compiles, and nothing runs them because no host
//      caller launches a kernel through them.
//
// The precedent is `mma_bf16.cuh`, which wraps the `ldmatrix` and `mma.sync`
// pair for the hand GEMM the same way.
//
// **What the emulation checks that a rate cannot.** The mapped instruction
// reads its operand through the descriptor, so the emulation decodes the same
// fields the device encodes: the sixteen-byte address unit, the fourteen-bit
// fields, and the layout type. Every read it makes is bounds-checked against
// the staged region its descriptor named. The probe of
// `dev/kernels/wgmma_bench.cu` found three defects on a card, and every one was
// addressing; two of them fail on the CPU through this header.
//
// **What the emulation does not check.** It completes every asynchronous step
// at the point of issue, so a missing fence, a missing wait, and a missing
// transaction count are invisible here. The parking rule against that is a card
// run, and `docs/simulator.md` section 9.4 states it.

#include <cstddef>
#include <cstdint>

#include <cuda.h>

namespace nanochat {
namespace cuda_kernels {
namespace hopper {

//: The threads one warpgroup spans, which is the width the instruction spans.
constexpr int kWarpgroupThreads = 128;

//: The operand layout, which is the descriptor's own layout type.
enum class OperandLayout {
  //: Core matrices of eight rows of eight elements, sixteen bytes a row.
  kInterleaved = 0,
  //: A row-major panel of sixty-four columns, permuted by the 128-byte swizzle.
  kSwizzled128 = 1,
};

// --- The two lane maps, which both arms state ------------------------------
//
// The instruction is warpgroup wide, so an accumulator slot names a (row,
// column) by the lane that holds it and the register index. `docs/attention.md`
// Part 8 section 5 publishes both maps and the probe measured them one element
// at a time; the emulation reads them from here and not from a copy.

//: The row of one accumulator slot of `m64nNk16`.
constexpr int SlotRow(int lane, int index) {
  return 16 * (lane / 32) + (lane % 32) / 4 + 8 * ((index / 2) % 2);
}

//: The column of one accumulator slot of `m64nNk16`.
constexpr int SlotColumn(int lane, int index) {
  return 8 * (index / 4) + 2 * (lane % 4) + (index % 2);
}

//: The row of one element of the register operand: four `b16x2` registers a
//: thread, eight elements, sixty-four query rows by sixteen keys.
constexpr int RegisterRow(int lane, int reg) {
  return 16 * (lane / 32) + (lane % 32) / 4 + 8 * (reg % 2);
}

//: The key of one element of the register operand, `half` being the element of
//: the packed pair.
constexpr int RegisterKey(int lane, int reg, int half) {
  return 8 * (reg / 2) + 2 * (lane % 4) + half;
}

//: The descriptor's field encoding: the address in sixteen-byte units, the
//: leading offset, the stride offset, the base offset, and the layout type. One
//: statement for every arm, so a device build and the emulation cannot drift.
__host__ __device__ constexpr std::uint64_t DescriptorBits(
    std::uint32_t address, int leading_bytes, int stride_bytes,
    OperandLayout layout, int base_offset) {
  return static_cast<std::uint64_t>((address >> 4) & 0x3FFFu) |
         (static_cast<std::uint64_t>((leading_bytes >> 4) & 0x3FFFu) << 16) |
         (static_cast<std::uint64_t>((stride_bytes >> 4) & 0x3FFFu) << 32) |
         (static_cast<std::uint64_t>(base_offset & 0x7) << 49) |
         (static_cast<std::uint64_t>(layout) << 62);
}

#if defined(NANOCHAT_HOPPER_EMU_ARM)

#include "tools/cuda_sim/emu/hopper_ops_host.h"

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// ---------------------------------------------------------------------------
// The device arm
// ---------------------------------------------------------------------------

//: The shared-memory matrix descriptor: sixty-four bits of fields the
//: instruction reads. The address is in sixteen-byte units, so a staged tile
//: past a quarter of a mebibyte needs its own descriptor.
struct MatrixDescriptor {
  std::uint64_t bits;
};

//: The descriptors the two products need, stated once. Layout zero is the
//: interleaved form; layout one is the 128-byte swizzle.
__device__ __forceinline__ MatrixDescriptor
MakeMatrixDescriptor(const void* start, int leading_bytes, int stride_bytes,
                     OperandLayout layout, int base_offset = 0) {
  return MatrixDescriptor{DescriptorBits(
      static_cast<std::uint32_t>(__cvta_generic_to_shared(start)),
      leading_bytes, stride_bytes, layout, base_offset)};
}

//: A staged region, which the emulation bounds every operand read against. The
//: device instruction has no such check, so this is a no-op there.
__device__ __forceinline__ void DeclareSharedOperand(const void* /*base*/,
                                                     std::size_t /*bytes*/) {}

//: The score product: `d(row, key) += sum over the head width of Q(row, .) *
//: K(key, .)`, one sixteen-deep atom of it. One template holds both key widths,
//: because the accumulator count is the key width: sixty-four slots a thread at
//: a hundred and twenty-eight keys and thirty-two at sixty-four. The four flags
//: after the operands are the ones `dev/kernels/wgmma_bench.cu` measured and
//: every staged row it records was checked against.
template <int KeyWidth>
__device__ __forceinline__ void ScoreAtom(MatrixDescriptor query,
                                          MatrixDescriptor key, float* d,
                                          bool accumulate) {
  static_assert(KeyWidth == 128 || KeyWidth == 64,
                "the instruction holds a hundred and twenty-eight keys or "
                "sixty-four");
  const std::uint32_t scale = accumulate ? 1u : 0u;
  if constexpr (KeyWidth == 128) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %66, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
        "   %16, %17, %18, %19, %20, %21, %22, %23, "
        "   %24, %25, %26, %27, %28, %29, %30, %31, "
        "   %32, %33, %34, %35, %36, %37, %38, %39, "
        "   %40, %41, %42, %43, %44, %45, %46, %47, "
        "   %48, %49, %50, %51, %52, %53, %54, %55, "
        "   %56, %57, %58, %59, %60, %61, %62, %63}, "
        "  %64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
          "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
          "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]),
          "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]),
          "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]),
          "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(query.bits), "l"(key.bits), "r"(scale));
  } else {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %34, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
        "   %16, %17, %18, %19, %20, %21, %22, %23, "
        "   %24, %25, %26, %27, %28, %29, %30, %31}, "
        "  %32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
          "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(query.bits), "l"(key.bits), "r"(scale));
  }
}

//: The value product with the probability tile in registers: `d(row, dim) +=
//: sum over the keys of P(row, .) * V(., dim)`, one sixteen-key atom of it.
__device__ __forceinline__ void ValueAtom(const std::uint32_t (&a)[4],
                                          MatrixDescriptor value, float* d) {
  asm volatile(
      "{\n"
      "  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "  {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      "   %8,  %9,  %10, %11, %12, %13, %14, %15, "
      "   %16, %17, %18, %19, %20, %21, %22, %23, "
      "   %24, %25, %26, %27, %28, %29, %30, %31, "
      "   %32, %33, %34, %35, %36, %37, %38, %39, "
      "   %40, %41, %42, %43, %44, %45, %46, %47, "
      "   %48, %49, %50, %51, %52, %53, %54, %55, "
      "   %56, %57, %58, %59, %60, %61, %62, %63}, "
      "  {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),
        "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
        "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
        "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]),
        "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
        "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
        "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(value.bits));
}

//: The operand fence: it bounds the region the asynchronous instruction reads
//: from the ordinary stores that define the same registers. Without it `ptxas`
//: serializes the instruction against those stores; the probe records the
//: advisory and the fence that answers it.
__device__ __forceinline__ void Fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

template <int Count>
__device__ __forceinline__ void FenceAccumulator(float* d) {
#pragma unroll
  for (int i = 0; i < Count; ++i) asm volatile("" : "+f"(d[i]) : : "memory");
}

template <int Count>
__device__ __forceinline__ void FencePacked(std::uint32_t* a) {
#pragma unroll
  for (int i = 0; i < Count; ++i) asm volatile("" : "+r"(a[i]) : : "memory");
}

__device__ __forceinline__ void Commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void Wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(Pending) : "memory");
}

// --- The handoff -----------------------------------------------------------

//: The shared address of one object, in the form the assembly takes.
__device__ __forceinline__ std::uint32_t SharedWord(const void* address) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
}

__device__ __forceinline__ void BarrierInit(std::uint64_t* barrier,
                                            int arrivals) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(SharedWord(barrier)),
      "r"(arrivals));
}

__device__ __forceinline__ void BarrierExpectTx(std::uint64_t* barrier,
                                                int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   SharedWord(barrier)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void BarrierArrive(std::uint64_t* barrier) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(SharedWord(barrier))
      : "memory");
}

//: States that this thread's asynchronous copies complete the barrier, so the
//: producer issues a stage and moves on instead of waiting for its own copy.
__device__ __forceinline__ void BarrierArriveOnCopies(std::uint64_t* barrier) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(
                   SharedWord(barrier))
               : "memory");
}

__device__ __forceinline__ void BarrierWait(std::uint64_t* barrier, int phase) {
  std::uint32_t complete = 0;
  do {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(complete)
        : "r"(SharedWord(barrier)), "r"(phase)
        : "memory");
  } while (complete == 0);
}

//: `bar.sync id, count`: a barrier over `count` threads of the block, so a role
//: waits for its own warpgroups and not for the whole block.
template <int Count>
__device__ __forceinline__ void NamedBarrier(int id) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "n"(Count) : "memory");
}

// --- The tiled copy --------------------------------------------------------

//: One tiled copy from global memory into shared memory, and the barrier it
//: lands on. The destination is the box in the map's own axis order, so the
//: layout of the staged tile is the map's and not this function's.
template <int Rank>
__device__ __forceinline__ void TmaLoad(void* destination,
                                        const CUtensorMap& map,
                                        const int* coordinates,
                                        std::uint64_t* barrier) {
  static_assert(Rank >= 2 && Rank <= 5, "a tiled copy is rank two to five");
  const std::uint32_t target = SharedWord(destination);
  const std::uint32_t arrive = SharedWord(barrier);
  const std::uint64_t address = reinterpret_cast<std::uint64_t>(&map);
  if constexpr (Rank == 2) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(target),
        "l"(address), "r"(coordinates[0]), "r"(coordinates[1]), "r"(arrive)
        : "memory");
  } else if constexpr (Rank == 3) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(target),
        "l"(address), "r"(coordinates[0]), "r"(coordinates[1]),
        "r"(coordinates[2]), "r"(arrive)
        : "memory");
  } else if constexpr (Rank == 4) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(
            target),
        "l"(address), "r"(coordinates[0]), "r"(coordinates[1]),
        "r"(coordinates[2]), "r"(coordinates[3]), "r"(arrive)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n" ::"r"(
            target),
        "l"(address), "r"(coordinates[0]), "r"(coordinates[1]),
        "r"(coordinates[2]), "r"(coordinates[3]), "r"(coordinates[4]),
        "r"(arrive)
        : "memory");
  }
}

// --- Fences and the register budget ---------------------------------------

//: Orders the ordinary stores that staged a tile ahead of the asynchronous
//: proxy that reads it.
__device__ __forceinline__ void FenceAsyncShared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <int Count>
__device__ __forceinline__ void RegistersGive() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Count) : "memory");
}

template <int Count>
__device__ __forceinline__ void RegistersTake() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Count) : "memory");
}

#else

// ---------------------------------------------------------------------------
// The empty arm: a host pass of nvcc in a device build
// ---------------------------------------------------------------------------
//
// The functions exist so a kernel body compiles, and no host code path calls
// them: a kernel runs on a device, where the first arm is what was compiled.
// The emulation arm is what gives these calls a host meaning.

struct MatrixDescriptor {
  std::uint64_t bits;
};

inline MatrixDescriptor MakeMatrixDescriptor(const void* /*start*/,
                                             int /*leading_bytes*/,
                                             int /*stride_bytes*/,
                                             OperandLayout /*layout*/,
                                             int /*base_offset*/ = 0) {
  return MatrixDescriptor{0};
}

inline void DeclareSharedOperand(const void* /*base*/, std::size_t /*bytes*/) {}
template <int KeyWidth>
inline void ScoreAtom(MatrixDescriptor /*query*/, MatrixDescriptor /*key*/,
                      float* /*d*/, bool /*accumulate*/) {}
inline void ValueAtom(const std::uint32_t (& /*a*/)[4],
                      MatrixDescriptor /*value*/, float* /*d*/) {}
inline void Fence() {}
template <int Count>
inline void FenceAccumulator(float* /*d*/) {}
template <int Count>
inline void FencePacked(std::uint32_t* /*a*/) {}
inline void Commit() {}
template <int Pending>
inline void Wait() {}
inline std::uint32_t SharedWord(const void* /*address*/) { return 0; }
inline void BarrierInit(std::uint64_t* /*barrier*/, int /*arrivals*/) {}
inline void BarrierExpectTx(std::uint64_t* /*barrier*/, int /*bytes*/) {}
inline void BarrierArrive(std::uint64_t* /*barrier*/) {}
inline void BarrierArriveOnCopies(std::uint64_t* /*barrier*/) {}
inline void BarrierWait(std::uint64_t* /*barrier*/, int /*phase*/) {}
template <int Count>
inline void NamedBarrier(int /*id*/) {}
template <int Rank>
inline void TmaLoad(void* /*destination*/, const CUtensorMap& /*map*/,
                    const int* /*coordinates*/, std::uint64_t* /*barrier*/) {}
inline void FenceAsyncShared() {}
template <int Count>
inline void RegistersGive() {}
template <int Count>
inline void RegistersTake() {}

#endif  // the three arms

}  // namespace hopper
}  // namespace cuda_kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_KERNELS_HOPPER_OPS_CUH_
