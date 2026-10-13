// src/mesh_test.cc -- the device mesh for the single-client multi-device
// runtime (docs/client-runtime-design.md sections 5.1 and 5.2).
//
// Covers the data-mesh construction, the rank-1 rule, the axis name, and the
// backend device enumeration. CPU only; no GPU and no broker.

#include <cstdio>
#include <string>
#include <vector>

#include "nanochat/mesh.h"
#include "nanochat/sandbox.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void TestDataMesh() {
  const nanochat::MeshSpec mesh = nanochat::MeshSpec::Data(4);
  std::string error;
  if (!mesh.Validate(&error)) {
    Fail("a data mesh failed validation: " + error);
  }
  if (mesh.size() != 4) Fail("a data mesh of 4 has the wrong size");
  if (mesh.axes() != 1) Fail("a data mesh must have one axis");
  if (mesh.axis_names.size() != 1 ||
      mesh.axis_names[0] != nanochat::kMeshDataAxis) {
    Fail("a data mesh has the wrong axis name");
  }
}

void TestDataMeshClampsSize() {
  const nanochat::MeshSpec mesh = nanochat::MeshSpec::Data(0);
  std::string error;
  if (!mesh.Validate(&error)) Fail("a data mesh of 0 failed validation");
  if (mesh.size() != 1) Fail("a data mesh of 0 must clamp to 1");
}

void TestDeviceIndices() {
  const nanochat::MeshSpec mesh = nanochat::MeshSpec::Data({3, 1});
  if (mesh.size() != 2) Fail("a device-index mesh has the wrong size");
  if (mesh.device_indices != std::vector<int>({3, 1})) {
    Fail("the device indices are wrong");
  }
  std::string error;
  if (!mesh.Validate(&error)) Fail("a device-index mesh failed validation");
  // The reference backend binds every thread to the one host device.
  nanochat::SelectDevice(0);
}

void TestRejectsRankTwo() {
  nanochat::MeshSpec mesh;
  mesh.shape = {2, 2};
  mesh.axis_names = {"data", "model"};
  std::string error;
  if (mesh.Validate(&error)) Fail("a rank-2 mesh passed validation");
  if (error.empty()) Fail("the rank-2 rejection carries no reason");
}

void TestRejectsUnknownAxis() {
  nanochat::MeshSpec mesh;
  mesh.shape = {2};
  mesh.axis_names = {"model"};
  std::string error;
  if (mesh.Validate(&error)) Fail("an unknown axis name passed validation");
  if (error.empty()) Fail("the axis-name rejection carries no reason");
}

void TestRejectsEmptyMesh() {
  nanochat::MeshSpec mesh;
  std::string error;
  if (mesh.Validate(&error)) Fail("an empty mesh passed validation");
}

void TestEnumerateDevices() {
  const std::vector<nanochat::DeviceInfo> devices =
      nanochat::EnumerateDevices();
  if (devices.empty()) Fail("EnumerateDevices returned no device");
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].name == nullptr || devices[i].name[0] == '\0') {
      Fail("a device has an empty name");
    }
    if (devices[i].ordinal != static_cast<int>(i)) {
      Fail("a device ordinal does not match its list position");
    }
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  TestDataMesh();
  TestDataMeshClampsSize();
  TestDeviceIndices();
  TestRejectsRankTwo();
  TestRejectsUnknownAxis();
  TestRejectsEmptyMesh();
  TestEnumerateDevices();
  if (g_failures != 0) {
    std::fprintf(stderr, "mesh_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("mesh_test: ok\n");
  return 0;
}
