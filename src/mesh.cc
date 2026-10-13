// src/mesh.cc -- the host-side mesh for the single-client multi-device runtime
// (docs/client-runtime-design.md). This file holds no vendor code and no
// compute: it validates the mesh shape and the axis name, so the runtime and
// the Python client share one rule. `EnumerateDevices` is backend-provided.

#include "nanochat/mesh.h"

#include <string>
#include <utility>

namespace nanochat {
namespace {

constexpr const char* kDataAxis = kMeshDataAxis;

}  // namespace

MeshSpec MeshSpec::Data(int size) {
  const int count = size < 1 ? 1 : size;
  std::vector<int> device_indices(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    device_indices[static_cast<std::size_t>(i)] = i;
  }
  return Data(std::move(device_indices));
}

MeshSpec MeshSpec::Data(std::vector<int> device_indices) {
  MeshSpec mesh;
  if (device_indices.empty()) device_indices.push_back(0);
  mesh.shape = {static_cast<int>(device_indices.size())};
  mesh.axis_names = {kDataAxis};
  mesh.device_indices = std::move(device_indices);
  return mesh;
}

int MeshSpec::size() const {
  int total = 1;
  for (int dim : shape) total *= dim;
  return total;
}

bool MeshSpec::Validate(std::string* error) const {
  const auto fail = [error](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (shape.empty()) {
    return fail("the mesh has no axes");
  }
  if (shape.size() != axis_names.size()) {
    return fail("the mesh shape and the axis names differ in length");
  }
  if (shape.size() != 1) {
    return fail("the runtime supports a rank-1 mesh only; this mesh has " +
                std::to_string(shape.size()) + " axes");
  }
  if (shape[0] < 1) {
    return fail("the mesh axis size must be positive");
  }
  if (axis_names[0] != kDataAxis) {
    return fail("the one mesh axis must be named \"" + std::string(kDataAxis) +
                "\"; this mesh names it \"" + axis_names[0] + "\"");
  }
  if (!device_indices.empty() &&
      device_indices.size() != static_cast<std::size_t>(size())) {
    return fail("the device index list does not match the mesh size");
  }
  return true;
}

}  // namespace nanochat
