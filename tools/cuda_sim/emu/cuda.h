#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_H_

// The driver API stub: the Tensor Memory Accelerator's map and its encoder
// (docs/simulator.md section 9.2).
//
// A device source that issues a tiled copy includes `<cuda.h>`, because the map
// is a driver object: the host builds it with `cuTensorMapEncodeTiled` and the
// kernel takes it by value. No toolkit include path is reachable from this
// package, so this header supplies the type, the four enumerations a map
// states, and the encoder.
//
// **The map is opaque, and the emulator reads it through a registry.** A real
// `CUtensorMap` is a hundred and twenty-eight opaque bytes, and the driver
// decodes it. This header keeps the same envelope, so the type has the same
// size and the same copy behaviour, and it stores a serial number in the first
// eight bytes. The fields of the map are in a process registry keyed by that
// number. The envelope can therefore be copied, passed by value, and captured
// by a fiber without losing its meaning, which a pointer-keyed registry would.
//
// **The encoder validates.** The real encoder refuses a box of rank zero, a
// rank above five, an empty extent, a misaligned base, a stride that is not a
// multiple of sixteen bytes, and an interleaved element stride of one. This one
// refuses the same list, so a map that a card rejects is rejected here.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

using CUresult = int;
using cuuint16_t = unsigned short;
using cuuint32_t = unsigned int;
using cuuint64_t = unsigned long long;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;

enum CUtensorMapDataType {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
  CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
  CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
  CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
  CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
};

enum CUtensorMapInterleave {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
  CU_TENSOR_MAP_INTERLEAVE_16B = 1,
  CU_TENSOR_MAP_INTERLEAVE_32B = 2,
};

enum CUtensorMapSwizzle {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B = 3,
  CU_TENSOR_MAP_SWIZZLE_64B = 2,
  CU_TENSOR_MAP_SWIZZLE_128B = 1,
};

enum CUtensorMapL2promotion {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2,
  CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3,
};

enum CUtensorMapFloatOOBfill {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = 1,
};

// The map's envelope. The size is the driver's, and the alignment is the one a
// kernel parameter of this type is passed with.
struct alignas(64) CUtensorMap {
  unsigned char opaque[128];
};

constexpr int kCudaSimTensorMapRank = 5;

//: The fields of one map, as this stub keeps them. The real driver keeps the
//: same six facts in its own encoding.
struct CudaSimTensorMapFields {
  CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  unsigned int rank = 0;
  const void* base = nullptr;
  cuuint64_t dimension[kCudaSimTensorMapRank] = {};
  cuuint64_t stride[kCudaSimTensorMapRank] = {};
  cuuint32_t box[kCudaSimTensorMapRank] = {};
  cuuint32_t element_stride[kCudaSimTensorMapRank] = {};
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
};

namespace nanochat {
namespace emu {

//: The registry behind the envelope. One vector for the process, because a map
//: outlives the block that reads it and one block runs at a time.
inline std::vector<CudaSimTensorMapFields>& TensorMapRegistry() {
  static std::vector<CudaSimTensorMapFields>* registry =
      new std::vector<CudaSimTensorMapFields>();
  return *registry;
}

inline std::mutex& TensorMapMutex() {
  static std::mutex* mutex = new std::mutex();
  return *mutex;
}

//: The fields of a map. A serial number this stub does not know is a map the
//: host never encoded, and a copy that reads one aborts with that statement
//: rather than reading an arbitrary box.
inline const CudaSimTensorMapFields& TensorMapFields(const CUtensorMap& map) {
  std::lock_guard<std::mutex> lock(TensorMapMutex());
  std::size_t serial = 0;
  std::memcpy(&serial, map.opaque, sizeof(serial));
  std::vector<CudaSimTensorMapFields>& registry = TensorMapRegistry();
  if (serial == 0 || serial > registry.size()) {
    std::fprintf(stderr,
                 "emu: a tiled copy read a Tensor Memory Accelerator map the "
                 "host never encoded (serial %zu of %zu)\n",
                 serial, registry.size());
    std::abort();
  }
  return registry[serial - 1];
}

//: The bytes one element of a map's data type occupies.
inline std::size_t TensorMapElementBytes(CUtensorMapDataType type) {
  switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
    default:
      return 8;
  }
}

}  // namespace emu
}  // namespace nanochat

// Encodes one tiled map. The rank is the number of axes, `dimensions` and
// `box_dimensions` state each axis from the innermost out, and `global_strides`
// states the byte distance between two elements of every axis but the innermost
// one, as the driver asks.
inline CUresult cuTensorMapEncodeTiled(
    CUtensorMap* map, CUtensorMapDataType data_type, cuuint32_t rank,
    void* global_address, const cuuint64_t* global_dimensions,
    const cuuint64_t* global_strides, const cuuint32_t* box_dimensions,
    const cuuint32_t* element_strides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2_promotion,
    CUtensorMapFloatOOBfill oob_fill) {
  namespace emu = ::nanochat::emu;
  if (map == nullptr || global_address == nullptr || rank == 0 ||
      rank > kCudaSimTensorMapRank || global_dimensions == nullptr ||
      global_strides == nullptr || box_dimensions == nullptr ||
      element_strides == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::size_t element_bytes = emu::TensorMapElementBytes(data_type);
  // The driver's pointer rule: the base is aligned to sixteen bytes.
  if ((reinterpret_cast<std::uintptr_t>(global_address) & 15u) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // The driver's stride rule, stated for every axis past the innermost one.
  for (cuuint32_t axis = 0; axis + 1 < rank; ++axis) {
    if (global_strides[axis] == 0) return CUDA_ERROR_INVALID_VALUE;
    if ((global_strides[axis] % 16u) != 0) return CUDA_ERROR_INVALID_VALUE;
  }
  CudaSimTensorMapFields fields;
  fields.data_type = data_type;
  fields.rank = rank;
  fields.base = global_address;
  for (cuuint32_t axis = 0; axis < rank; ++axis) {
    if (global_dimensions[axis] == 0) return CUDA_ERROR_INVALID_VALUE;
    if (box_dimensions[axis] == 0) return CUDA_ERROR_INVALID_VALUE;
    if (box_dimensions[axis] > global_dimensions[axis]) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (element_strides[axis] != 1 && element_strides[axis] != 2 &&
        element_strides[axis] != 4 && element_strides[axis] != 8) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    // The innermost axis of a tiled copy is the one the swizzle permutes, so a
    // map that strides it is the interleaved form and not this one.
    if (axis == 0 && element_strides[0] != 1) return CUDA_ERROR_INVALID_VALUE;
    fields.dimension[axis] = global_dimensions[axis];
    fields.stride[axis] = axis == 0 ? 0 : global_strides[axis - 1];
    fields.box[axis] = box_dimensions[axis];
    fields.element_stride[axis] = element_strides[axis];
  }
  // The swizzle must fit the innermost box: a hundred and twenty-eight byte
  // swizzle is an atom of eight rows of a hundred and twenty-eight bytes.
  const std::size_t inner_bytes =
      static_cast<std::size_t>(fields.box[0]) * element_bytes;
  const std::size_t atom_bytes = swizzle == CU_TENSOR_MAP_SWIZZLE_128B  ? 128
                                 : swizzle == CU_TENSOR_MAP_SWIZZLE_64B ? 64
                                 : swizzle == CU_TENSOR_MAP_SWIZZLE_32B ? 32
                                                                        : 0;
  if (atom_bytes != 0 && inner_bytes != atom_bytes)
    return CUDA_ERROR_INVALID_VALUE;
  fields.interleave = interleave;
  fields.swizzle = swizzle;
  fields.l2 = l2_promotion;
  fields.oob = oob_fill;

  std::lock_guard<std::mutex> lock(emu::TensorMapMutex());
  emu::TensorMapRegistry().push_back(fields);
  const std::size_t serial = emu::TensorMapRegistry().size();
  std::memset(map->opaque, 0, sizeof(map->opaque));
  std::memcpy(map->opaque, &serial, sizeof(serial));
  return CUDA_SUCCESS;
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_CUDA_H_
