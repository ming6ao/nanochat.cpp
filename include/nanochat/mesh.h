#ifndef NANOCHAT_MESH_H_
#define NANOCHAT_MESH_H_

#include <cstddef>
#include <string>
#include <vector>

#include "nanochat/tensor.h"

// The device mesh for the single-client multi-device runtime
// (docs/client-runtime-design.md). One client process controls every local
// accelerator and sees them as one system.
//
// This header is a leaf: it includes only `nanochat/tensor.h` and the standard
// library, and it carries host metadata only. The active backend implements
// `EnumerateDevices`, so a backend may include this header for that one symbol
// without reaching above the kernel seam. See DESIGN.md section 2.1.

namespace nanochat {

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

// A vendor-free description of one local accelerator. `ordinal` is the index in
// the `EnumerateDevices` list; `index` is the backend device index that a
// thread binds with `cudaSetDevice` (or the CPU device ordinal, which is always
// 0). `name` points at a string literal or a static buffer, so it stays valid
// for the life of the process.
struct DeviceInfo {
  int ordinal = 0;
  int index = 0;
  const char* name = "cpu";
  std::size_t total_memory_bytes = 0;
  int compute_major = 0;
  int compute_minor = 0;
  bool is_device = false;
  bool has_tensor_cores = false;
};

// Enumerates the local accelerators. The active backend implements this
// function, in the same way it implements `kernels::GetCaps`. The CPU
// reference backend returns one entry. A caller must not free the result; the
// entries are values.
std::vector<DeviceInfo> EnumerateDevices();

// Binds the calling thread to the backend device at `index`. The runtime calls
// this once per device thread before it builds a replica, so the thread-affine
// rule of docs/client-runtime-design.md section 8 holds. The CPU reference
// backend makes it a no-op; the CUDA backend calls `cudaSetDevice`.
void SelectDevice(int index);

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

// The one mesh axis the runtime supports. The fixed graph is data parallel, so
// the mesh is rank 1 and the axis is named "data"
// (docs/client-runtime-design.md section 5.2).
inline constexpr const char* kMeshDataAxis = "data";

// A named device grid. The shape has one entry per axis and the axis names
// have the same length. The runtime accepts a rank-1 mesh today; `Validate`
// rejects any other shape. `MeshSpec` is a value type.
struct MeshSpec {
  std::vector<int> shape;
  std::vector<std::string> axis_names;
  // The backend device index at each mesh position, in row-major order.
  std::vector<int> device_indices;

  // Builds a rank-1 data mesh over the devices `0, 1, ... , size - 1`. A size
  // below 1 becomes 1.
  static MeshSpec Data(int size);

  // Builds a rank-1 data mesh over the given device indices.
  static MeshSpec Data(std::vector<int> device_indices);

  // The product of the axis sizes.
  int size() const;

  // The number of axes.
  int axes() const { return static_cast<int>(shape.size()); }

  // Validates the mesh against the supported set. On failure, writes a reason
  // to `*error` when `error` is not null and returns false.
  bool Validate(std::string* error = nullptr) const;
};

}  // namespace nanochat

#endif  // NANOCHAT_MESH_H_
