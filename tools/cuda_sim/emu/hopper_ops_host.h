#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_HOPPER_OPS_HOST_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_HOPPER_OPS_HOST_H_

// The emulation arm of the four Hopper mechanisms
// (docs/simulator.md section 9.2).
//
// This header is *included inside* `namespace nanochat::cuda_kernels::hopper`,
// after `backends/cuda/kernels/hopper_ops.cuh` has stated the lane maps, the
// operand layout, and the interface. It opens no namespace of its own, and it
// states no symbol that header does not already name.
//
// **What it computes.** Each mechanism becomes the plain arithmetic the
// instruction performs, driven by the same descriptor bits the device encodes.
//
//   * The score product reads both operands out of the staged tile through the
//     descriptor, one core matrix address at a time, and each thread computes
//     the sixty-four accumulator slots the lane map says it owns. The cost is
//     the instruction's own arithmetic and not the warp width times it.
//   * The value product reads the probability tile from the calling thread's
//   own
//     four registers, through the register map, and the value tile through a
//     descriptor.
//   * A tiled copy is the box the map states, in the map's axis order, with the
//     128-byte swizzle applied to the innermost atom.
//   * A barrier is a cooperative wait: the fiber blocks and the scheduler
//     re-tests the object, so a handoff between roles makes progress.
//
// **What it checks.** Both products resolve every operand byte through the
// registered staging region of the block and refuse a read outside it, and the
// descriptor encoder refuses a leading or stride offset that the fourteen-bit
// field cannot hold or that is not a multiple of sixteen bytes. Two of the
// three defects the probe found on a card fail here instead.
//
// **What it does not check.** Every asynchronous step completes at the point of
// issue. A missing fence, a missing wait, and a transaction count that does not
// match the copy are invisible. A card run still decides those.

// --- The descriptor ---------------------------------------------------------

//: The shared-memory matrix descriptor. The fields are the device's: the
//: address in sixteen-byte units, the leading offset, the stride offset, the
//: base offset, and the layout type.
struct MatrixDescriptor {
  std::uint64_t bits;
};

namespace detail {

//: The fields of a descriptor, decoded from the same bits the device encodes.
struct DecodedDescriptor {
  std::uint32_t address;  // the shared address of the operand's first byte
  int leading;            // the offset between the two halves of the k axis
  int stride;             // the offset between two row groups
  int base;               // the base offset, in bytes
  OperandLayout layout;
};

inline DecodedDescriptor DecodeDescriptor(const MatrixDescriptor& descriptor) {
  DecodedDescriptor decoded;
  decoded.address = static_cast<std::uint32_t>(descriptor.bits & 0x3FFFu) << 4;
  decoded.leading = static_cast<int>((descriptor.bits >> 16) & 0x3FFFu) << 4;
  decoded.stride = static_cast<int>((descriptor.bits >> 32) & 0x3FFFu) << 4;
  decoded.base = static_cast<int>((descriptor.bits >> 49) & 0x7u) << 4;
  decoded.layout = static_cast<OperandLayout>((descriptor.bits >> 62) & 0x3u);
  return decoded;
}

//: The 128-byte swizzle of one shared address. The device permutes the sixteen-
//: byte chunk index of a hundred and twenty-eight byte row with the row's own
//: index inside a kilobyte frame, and it does so on the address it computed. So
//: the model is the address arithmetic and then this, and not a column formula:
//: `dev/kernels/wgmma_bench.cu` measured the form with a panel step and an atom
//: step, and only the address model reproduces both.
inline std::uint32_t Swizzle128(std::uint32_t address) {
  return address ^ (((address >> 7) & 7u) << 4);
}

//: The byte offset of one element of a staged tile, relative to the
//: descriptor's address, before any swizzle. `row` is the first axis the
//: instruction walks and `axis` the second, which is the head width for the
//: score product and the key index for the value product.
inline int LayoutOffset(OperandLayout layout, int row, int axis, int leading,
                        int stride) {
  if (layout == OperandLayout::kInterleaved) {
    // A core matrix is eight rows of eight elements, sixteen bytes a row. The
    // two halves of the sixteen-deep `k` axis are one leading offset apart and
    // the next eight rows follow at one stride offset.
    return (row / 8) * stride + (axis / 8) * leading + (row % 8) * 16 +
           (axis % 8) * 2;
  }
  // The 128-byte swizzle: a row is a hundred and twenty-eight bytes, so a group
  // of eight rows is one kilobyte and the next group follows at one stride
  // offset. The caller advanced the descriptor to this atom's columns.
  return (row / 8) * stride + (row % 8) * 128 + axis * 2;
}

//: A bf16 word as a float. The conversion is the top half of the float, which
//: is what a bf16 is.
inline float Bf16ToFloat(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

//: One element of a staged tile, through a descriptor, bounds-checked against
//: the region the descriptor named. A read outside that region is the defect
//: the probe found on a card, and it stops here with the figures a reader
//: needs.
inline float OperandElement(const MatrixDescriptor& descriptor, int row,
                            int axis) {
  const DecodedDescriptor decoded = DecodeDescriptor(descriptor);
  const std::uint32_t relative = static_cast<std::uint32_t>(
      LayoutOffset(decoded.layout, row, axis, decoded.leading, decoded.stride));
  const std::uint32_t unswizzled =
      decoded.address + static_cast<std::uint32_t>(decoded.base) + relative;
  const std::uint32_t token = decoded.layout == OperandLayout::kSwizzled128
                                  ? Swizzle128(unswizzled)
                                  : unswizzled;
  const ::nanochat::emu::SharedRegion* region =
      ::nanochat::emu::SharedRegionOf(token);
  if (region == nullptr || token + 2 > region->token + region->bytes) {
    std::fprintf(
        stderr,
        "emu: a Hopper operand read past its staged region: descriptor "
        "address %u, layout %d, row %d, axis %d, read at byte %u, region of "
        "%zu bytes at %u\n",
        decoded.address, static_cast<int>(decoded.layout), row, axis, token,
        region != nullptr ? region->bytes : 0u,
        region != nullptr ? region->token : 0u);
    std::abort();
  }
  std::uint16_t bits = 0;
  std::memcpy(&bits,
              static_cast<const char*>(region->base) + (token - region->token),
              sizeof(bits));
  return Bf16ToFloat(bits);
}

//: The probability tile of one atom as the four lanes of a quad hold it: two
//: row groups of eight rows by sixteen keys. The instruction's first operand is
//: distributed, so an accumulation over sixteen keys reads four threads'
//: registers. On a device the tensor core does that. Here the quad exchanges
//: its four packed registers, once per atom and not once per element.
struct QuadOperand {
  float element[2][16];  // [row group][key]
};

inline QuadOperand GatherQuad(const std::uint32_t (&a)[4], int lane) {
  QuadOperand quad{};
  // The quad is four adjacent lanes of a warp: the shuffle's lane space is the
  // warp and not the warpgroup, so the base is taken modulo the warp.
  const int base = (lane % 32) & ~3;
  for (int source = 0; source < 4; ++source) {
    const int owner = base + source;
    for (int reg = 0; reg < 4; ++reg) {
      // The emulator maps a shuffle for `int`, `float`, and `double`, so a
      // packed pair travels as the bits of an `int`. The second register of a
      // pair holds the other row group, which is why `reg % 2` is the group.
      const int bits = static_cast<int>(a[reg]);
      const std::uint32_t packed =
          static_cast<std::uint32_t>(__shfl_sync(0xFFFFFFFFu, bits, owner));
      for (int half = 0; half < 2; ++half) {
        const std::uint16_t word = static_cast<std::uint16_t>(
            half == 0 ? (packed & 0xFFFFu) : (packed >> 16));
        quad.element[reg % 2][RegisterKey(owner, reg, half)] =
            Bf16ToFloat(word);
      }
    }
  }
  return quad;
}

}  // namespace detail

inline MatrixDescriptor MakeMatrixDescriptor(const void* start,
                                             int leading_bytes,
                                             int stride_bytes,
                                             OperandLayout layout,
                                             int base_offset = 0) {
  return MatrixDescriptor{DescriptorBits(
      static_cast<std::uint32_t>(__cvta_generic_to_shared(start)),
      leading_bytes, stride_bytes, layout, base_offset)};
}

inline void DeclareSharedOperand(const void* base, std::size_t bytes) {
  ::nanochat::emu::RegisterSharedRegion(base, bytes);
}

// --- The products -----------------------------------------------------------

template <int KeyWidth>
inline void ScoreAtom(MatrixDescriptor query, MatrixDescriptor key, float* d,
                      bool accumulate) {
  static_assert(KeyWidth == 128 || KeyWidth == 64,
                "the instruction holds a hundred and twenty-eight keys or "
                "sixty-four");
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;
  const int slots = KeyWidth / 2;
#pragma unroll 1
  for (int index = 0; index < slots; ++index) {
    const int row = SlotRow(lane, index);
    const int column = SlotColumn(lane, index);
    float sum = 0.0f;
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      sum += detail::OperandElement(query, row, step) *
             detail::OperandElement(key, column, step);
    }
    d[index] = accumulate ? d[index] + sum : sum;
  }
}

inline void ValueAtom(const std::uint32_t (&a)[4], MatrixDescriptor value,
                      float* d) {
  const int lane = static_cast<int>(threadIdx.x) % kWarpgroupThreads;
  const detail::QuadOperand quad = detail::GatherQuad(a, lane);
#pragma unroll 1
  for (int index = 0; index < 64; ++index) {
    const float* probability = quad.element[(index / 2) % 2];
    const int column = SlotColumn(lane, index);
    float sum = 0.0f;
#pragma unroll 1
    for (int step = 0; step < 16; ++step) {
      sum += probability[step] * detail::OperandElement(value, column, step);
    }
    d[index] += sum;
  }
}

// --- The instruction's boundaries -------------------------------------------
//
// The emulation is synchronous, so a fence, a commit, and a wait have nothing
// to order and nothing to release. They stay as calls, because a kernel body is
// one text for both arms.

inline void Fence() {}
template <int Count>
inline void FenceAccumulator(float* /*d*/) {}
template <int Count>
inline void FencePacked(std::uint32_t* /*a*/) {}
inline void Commit() {}
template <int Pending>
inline void Wait() {}
inline void FenceAsyncShared() {}
template <int Count>
inline void RegistersGive() {}
template <int Count>
inline void RegistersTake() {}

// --- The handoff ------------------------------------------------------------

inline std::uint32_t SharedWord(const void* address) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
}

inline void BarrierInit(std::uint64_t* barrier, int arrivals) {
  ::nanochat::emu::MbarrierInit(barrier, arrivals);
}

// The arm states both the arrival and the bytes the copy will land, which is
// what `mbarrier.arrive.expect_tx` does in one instruction.
inline void BarrierExpectTx(std::uint64_t* barrier, int bytes) {
  ::nanochat::emu::MbarrierExpectBytes(barrier, bytes);
  ::nanochat::emu::MbarrierArriveAt(barrier);
}

inline void BarrierArrive(std::uint64_t* barrier) {
  ::nanochat::emu::MbarrierArriveAt(barrier);
}

// The copies this thread issued are complete at the point they were issued, so
// the arrival the device defers is the arrival the emulation makes now.
inline void BarrierArriveOnCopies(std::uint64_t* barrier) {
  ::nanochat::emu::MbarrierArriveAt(barrier);
}

inline void BarrierWait(std::uint64_t* barrier, int phase) {
  ::nanochat::emu::MbarrierWaitParity(barrier, phase);
}

template <int Count>
inline void NamedBarrier(int id) {
  ::nanochat::emu::NamedBarrierSync(id, Count);
}

// --- The tiled copy ---------------------------------------------------------

namespace detail {

//: The byte offset of one element of a box inside the shared destination. The
//: box lands in the map's axis order with the innermost axis fastest, so the
//: offset is the mixed-radix index of the element times its size. A swizzled
//: box is the rank-two panel the engine likes; a box of a higher rank is the
//: plain form.
inline std::size_t BoxOffset(const CudaSimTensorMapFields& fields,
                             const int* index, std::size_t element_bytes) {
  if (fields.rank == 2 && fields.swizzle == CU_TENSOR_MAP_SWIZZLE_128B) {
    const int row = index[1];
    const int column = index[0];
    const std::uint32_t unswizzled =
        static_cast<std::uint32_t>((row / 8) * 1024 + (row % 8) * 128 +
                                   column * static_cast<int>(element_bytes));
    return Swizzle128(unswizzled);
  }
  std::size_t linear = 0;
  std::size_t extent = 1;
  for (unsigned int axis = 0; axis < fields.rank; ++axis) {
    linear += extent * static_cast<std::size_t>(index[axis]);
    extent *= fields.box[axis];
  }
  return linear * element_bytes;
}

}  // namespace detail

template <int Rank>
inline void TmaLoad(void* destination, const CUtensorMap& map,
                    const int* coordinates, std::uint64_t* barrier) {
  static_assert(Rank >= 2 && Rank <= 5, "a tiled copy is rank two to five");
  const CudaSimTensorMapFields& fields = ::nanochat::emu::TensorMapFields(map);
  if (static_cast<int>(fields.rank) != Rank) {
    std::fprintf(stderr, "emu: a tiled copy of rank %d read a map of rank %u\n",
                 Rank, fields.rank);
    std::abort();
  }
  if (Rank > 2 && fields.swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
    std::fprintf(stderr,
                 "emu: a swizzled tiled copy of rank %d is not mapped; the "
                 "engine's swizzled box is rank two\n",
                 Rank);
    std::abort();
  }
  const std::size_t element_bytes =
      ::nanochat::emu::TensorMapElementBytes(fields.data_type);
  std::size_t elements = 1;
  for (int axis = 0; axis < Rank; ++axis) elements *= fields.box[axis];
  const std::size_t bytes = elements * element_bytes;
  ::nanochat::emu::RegisterSharedRegion(destination, bytes);

  int index[kCudaSimTensorMapRank] = {};
  for (std::size_t i = 0; i < elements; ++i) {
    std::size_t rest = i;
    for (int axis = 0; axis < Rank; ++axis) {
      index[axis] = static_cast<int>(rest % fields.box[axis]);
      rest /= fields.box[axis];
    }
    // The source address, or a zero fill when the coordinate is past the
    // extent, which is what the engine's out-of-bounds rule states.
    bool inside = true;
    std::size_t source = 0;
    for (int axis = Rank - 1; axis >= 0; --axis) {
      const long long coordinate =
          static_cast<long long>(coordinates[axis]) + index[axis];
      if (coordinate < 0 ||
          coordinate >= static_cast<long long>(fields.dimension[axis])) {
        inside = false;
        break;
      }
      if (axis == 0) {
        source += static_cast<std::size_t>(coordinate) * element_bytes;
      } else {
        source += static_cast<std::size_t>(coordinate) * fields.stride[axis];
      }
    }
    char* target = static_cast<char*>(destination) +
                   static_cast<std::ptrdiff_t>(
                       detail::BoxOffset(fields, index, element_bytes));
    if (inside) {
      std::memcpy(target, static_cast<const char*>(fields.base) + source,
                  element_bytes);
    } else {
      std::memset(target, 0, element_bytes);
    }
  }
  if (barrier != nullptr) {
    ::nanochat::emu::MbarrierCopiedBytes(barrier, static_cast<int>(bytes));
  }
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_HOPPER_OPS_HOST_H_
