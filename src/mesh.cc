// src/mesh.cc -- the host-side mesh for the single-client multi-device runtime
// (docs/client-runtime-design.md). This file holds no vendor code and no
// compute: it validates the mesh shape and the axis name, so the runtime and
// the Python client share one rule. `EnumerateDevices` is backend-provided.

#include "nanochat/mesh.h"

#include <string>

namespace nanochat {
namespace {

constexpr const char* kDataAxis = kMeshDataAxis;

}  // namespace

MeshSpec MeshSpec::Data(int size) {
  MeshSpec mesh;
  mesh.shape = {size < 1 ? 1 : size};
  mesh.axis_names = {kDataAxis};
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
  return true;
}

}  // namespace nanochat
