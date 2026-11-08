#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_MMA_BF16_CUH_
#define NANOCHAT_BACKENDS_CUDA_GEMM_MMA_BF16_CUH_

// The bf16 matrix multiply-accumulate micro-tile of the hand GEMM
// (docs/kernels.md Part 3, phase 1).
//
// `wmma::load_matrix_sync` compiles to plain loads plus a register permute on
// `sm_90`, and `wmma::mma_sync` is a 16 by 16 by 16 instruction. The Hopper
// instruction set offers the cheaper pair: `ldmatrix` reads a fragment straight
// out of shared memory into the lanes the multiply-accumulate wants, and
// `mma.sync.aligned.m16n8k16` takes those fragments directly. This header is
// that pair, and it is mechanism 1 of section 4 of that part.
//
// **Two arms, one lane map.** The device arm is the instruction pair in inline
// assembly, with the four fragments the instruction defines. The host arm holds
// the whole micro-tile in every thread and computes it from shared memory,
// which is what the emulation engine can run; every thread of a warp therefore
// writes the same values, and the result is idempotent. The host arm is exact
// and slower by the warp width, and the kernel that includes this header has
// one body for both.
//
// **A refusal.** The device arm needs `sm_80` for `ldmatrix` and for the bf16
// multiply-accumulate, which the tree's bf16 build already requires. On a card
// below that, the host arm runs and is correct.
//
// **The lane map**, with `lane` the thread's index in its warp, `group` the
// value `lane / 4`, and `thread` the value `lane % 4`:
//
//   A, four 32-bit registers, one 16 by 16 tile:
//     register 0  (row group,     k 2*thread and 2*thread+1)
//     register 1  (row group + 8, k 2*thread and 2*thread+1)
//     register 2  (row group,     k 2*thread+8 and 2*thread+9)
//     register 3  (row group + 8, k 2*thread+8 and 2*thread+9)
//
//   B, two 32-bit registers, one 16 by 8 tile:
//     register 0  (k 2*thread and 2*thread+1, column group)
//     register 1  (k 2*thread+8 and 2*thread+9, column group)
//
//   C, four fp32 values, one 16 by 8 tile:
//     register 0  (row group,     column 2*thread)
//     register 1  (row group,     column 2*thread+1)
//     register 2  (row group + 8, column 2*thread)
//     register 3  (row group + 8, column 2*thread+1)
//
// The header is backend-internal and hidden from the frozen seam.

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nanochat {
namespace kernels {
namespace gemm_bf16 {

//: The micro-tile. The instruction is fixed at this shape.
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// Which arm this pass compiles. The device arm is the instruction pair; the
// host arm is the plain computation the emulator runs. This must be a
// preprocessor symbol and not a constant, because `#if` cannot read a constant:
// with a constant every pass would compile the host arm and a kernel would call
// a host function.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define NANOCHAT_MMA_BF16_DEVICE_ARM 1
#else
#define NANOCHAT_MMA_BF16_DEVICE_ARM 0
#endif

// --- The fragments ----------------------------------------------------------

#if NANOCHAT_MMA_BF16_DEVICE_ARM

//: The A operand: four 32-bit registers, each two bf16.
struct FragmentA {
  std::uint32_t reg[4];
};

//: The B operand: two 32-bit registers, each two bf16.
struct FragmentB {
  std::uint32_t reg[2];
};

//: The accumulator: four fp32 values.
struct FragmentC {
  float reg[4];
};

#else

//: The host arm holds the whole micro-tile, so the whole warp writes the same
//: result and the emulator needs no cross-lane traffic.
struct FragmentA {
  __nv_bfloat16 x[kMmaM * kMmaK];
};

struct FragmentB {
  __nv_bfloat16 x[kMmaK * kMmaN];
};

struct FragmentC {
  float x[kMmaM * kMmaN];
};

#endif

// --- The lane map, for the device arm and for the store ---------------------

namespace detail {

// The row and the column of the first element of one A register.
__host__ __device__ inline void AElement(int lane, int reg, int* row,
                                         int* column) {
  const int group = lane / 4;
  const int thread = lane % 4;
  *row = group + (reg % 2) * 8;
  *column = thread * 2 + (reg / 2) * 8;
}

// The k index and the column of the first element of one B register.
__host__ __device__ inline void BElement(int lane, int reg, int* k,
                                         int* column) {
  const int group = lane / 4;
  const int thread = lane % 4;
  *k = thread * 2 + reg * 8;
  *column = group;
}

// The row and the column of one accumulator register.
__host__ __device__ inline void CElement(int lane, int reg, int* row,
                                         int* column) {
  const int group = lane / 4;
  const int thread = lane % 4;
  *row = group + (reg / 2) * 8;
  *column = thread * 2 + (reg % 2);
}

}  // namespace detail

// --- The device arm ---------------------------------------------------------

#if NANOCHAT_MMA_BF16_DEVICE_ARM

// Loads one 16 by 16 A tile. `base` points at the tile's first element and
// `lead` is the element stride between two staged rows.
//
// A plain operand stages m by k with k contiguous, so the staged rows are the m
// axis and the plain form reads them. A transposed operand stages k by m, so
// the staged rows are the k axis and the transpose form turns them back. The
// four 8 by 8 tiles of the load reach the registers in the order the lane map
// wants, and the two orientations differ in which axis the addresses walk.
template <bool TransA>
__host__ __device__ __forceinline__ void LoadA(FragmentA& frag,
                                               const __nv_bfloat16* base,
                                               int lead) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int tile = lane / 8;
  const int within = lane % 8;
  int row = 0;
  int column = 0;
  if (TransA) {
    row = (tile / 2) * 8 + within;  // the staged rows are k
    column = (tile % 2) * 8;        // the staged columns are m
  } else {
    row = (tile % 2) * 8 + within;  // the staged rows are m
    column = (tile / 2) * 8;        // the staged columns are k
  }
  const std::uint32_t address = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(base + row * lead + column));
  if (TransA) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]), "=r"(frag.reg[2]),
          "=r"(frag.reg[3])
        : "r"(address));
  } else {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]), "=r"(frag.reg[2]),
          "=r"(frag.reg[3])
        : "r"(address));
  }
}

// Loads one 16 by 8 B tile through the same two orientations.
//
// A plain operand stages k by n with n contiguous, so the staged rows are k and
// the transpose form turns them. A transposed operand stages n by k with k
// contiguous, so the staged rows are n and the plain form reads them.
template <bool TransB>
__host__ __device__ __forceinline__ void LoadB(FragmentB& frag,
                                               const __nv_bfloat16* base,
                                               int lead) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int tile = lane / 8;
  const int within = lane % 8;
  // The two 8 by 8 matrices of a B tile are its two k halves, so they lie along
  // the k axis. A transposed operand stages n by k, so the second matrix is
  // eight columns to the right of the first, on the same eight rows. A plain
  // operand stages k by n, so the second matrix is eight rows below the first.
  const std::uint32_t address = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(TransB ? base + within * lead + tile * 8
                                      : base + (tile * 8 + within) * lead));
  if (TransB) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(frag.reg[0]), "=r"(frag.reg[1])
                 : "r"(address));
  } else {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(address));
  }
}

// D = A * B + D over the 16 by 8 by 16 micro-tile.
__host__ __device__ __forceinline__ void Mma(FragmentC& d, const FragmentA& a,
                                             const FragmentB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d.reg[0]), "+f"(d.reg[1]), "+f"(d.reg[2]), "+f"(d.reg[3])
      : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]),
        "r"(b.reg[0]), "r"(b.reg[1]));
}

// --- The host arm -----------------------------------------------------------

#else

// The host arm reads the tile through the same lane indices and computes the
// whole micro-tile in every thread, which is exact and needs no cross-lane
// traffic.
template <bool TransA>
__host__ __device__ inline void LoadA(FragmentA& frag,
                                      const __nv_bfloat16* base, int lead) {
  for (int row = 0; row < kMmaM; ++row) {
    for (int k = 0; k < kMmaK; ++k) {
      frag.x[row * kMmaK + k] =
          TransA ? base[k * lead + row] : base[row * lead + k];
    }
  }
}

template <bool TransB>
__host__ __device__ inline void LoadB(FragmentB& frag,
                                      const __nv_bfloat16* base, int lead) {
  for (int k = 0; k < kMmaK; ++k) {
    for (int column = 0; column < kMmaN; ++column) {
      frag.x[k * kMmaN + column] =
          TransB ? base[column * lead + k] : base[k * lead + column];
    }
  }
}

__host__ __device__ inline void Mma(FragmentC& d, const FragmentA& a,
                                    const FragmentB& b) {
  for (int row = 0; row < kMmaM; ++row) {
    for (int column = 0; column < kMmaN; ++column) {
      float sum = d.x[row * kMmaN + column];
      for (int k = 0; k < kMmaK; ++k) {
        sum += static_cast<float>(a.x[row * kMmaK + k]) *
               static_cast<float>(b.x[k * kMmaN + column]);
      }
      d.x[row * kMmaN + column] = sum;
    }
  }
}

#endif  // NANOCHAT_MMA_BF16_DEVICE_ARM

// --- The store, on both arms ------------------------------------------------

//: The accumulator registers one lane owns: the four of the instruction.
constexpr int kLaneRegisters = 4;

// The row and the column, inside the micro-tile, of accumulator register `reg`
// of `lane`. The epilogue of the kernel reads it, so the element a lane stores
// is the element the instruction gave that lane, and it is not assumed.
__host__ __device__ __forceinline__ void ElementOf(int lane, int reg, int* row,
                                                   int* column) {
  detail::CElement(lane, reg, row, column);
}

// The value of that element. The device arm reads the register the lane owns.
// The host arm holds the whole tile in every thread, so it reads the element
// the lane map names: the same element for the same (lane, reg), and every lane
// still writes four elements of its own. A store that wrote the whole tile from
// every lane would read a value a neighbour had already replaced whenever
// `beta` is not zero.
__host__ __device__ __forceinline__ float ElementValue(const FragmentC& c,
                                                       int lane, int reg) {
  (void)lane;
#if NANOCHAT_MMA_BF16_DEVICE_ARM
  return c.reg[reg];
#else
  int row = 0;
  int column = 0;
  detail::CElement(lane, reg, &row, &column);
  return c.x[row * kMmaN + column];
#endif
}

// Writes one micro-tile of the accumulator to an fp32 destination, row major
// with `lead` floats between two rows.
__host__ __device__ __forceinline__ void StoreFp32(float* dst, int lead,
                                                   const FragmentC& c) {
#if NANOCHAT_MMA_BF16_DEVICE_ARM
  const int lane = static_cast<int>(threadIdx.x) % 32;
  for (int reg = 0; reg < kLaneRegisters; ++reg) {
    int row = 0;
    int column = 0;
    detail::CElement(lane, reg, &row, &column);
    dst[row * lead + column] = c.reg[reg];
  }
#else
  // The host arm holds the whole tile, so one thread writes all of it and the
  // warp's other threads write the same values. The result is idempotent.
  for (int row = 0; row < kMmaM; ++row) {
    for (int column = 0; column < kMmaN; ++column) {
      dst[row * lead + column] = c.x[row * kMmaN + column];
    }
  }
#endif
}

// The same for a bf16 destination.
__host__ __device__ __forceinline__ void StoreBf16(__nv_bfloat16* dst, int lead,
                                                   const FragmentC& c) {
#if NANOCHAT_MMA_BF16_DEVICE_ARM
  const int lane = static_cast<int>(threadIdx.x) % 32;
  for (int reg = 0; reg < kLaneRegisters; ++reg) {
    int row = 0;
    int column = 0;
    detail::CElement(lane, reg, &row, &column);
    dst[row * lead + column] = __float2bfloat16(c.reg[reg]);
  }
#else
  for (int row = 0; row < kMmaM; ++row) {
    for (int column = 0; column < kMmaN; ++column) {
      dst[row * lead + column] = __float2bfloat16(c.x[row * kMmaN + column]);
    }
  }
#endif
}

//: Fills the accumulator with one value.
__host__ __device__ __forceinline__ void Fill(FragmentC& c, float value) {
#if NANOCHAT_MMA_BF16_DEVICE_ARM
  c.reg[0] = value;
  c.reg[1] = value;
  c.reg[2] = value;
  c.reg[3] = value;
#else
  for (int i = 0; i < kMmaM * kMmaN; ++i) c.x[i] = value;
#endif
}

#undef NANOCHAT_MMA_BF16_DEVICE_ARM

}  // namespace gemm_bf16
}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_MMA_BF16_CUH_
