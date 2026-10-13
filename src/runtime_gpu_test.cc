// src/runtime_gpu_test.cc -- two device threads that share one card
// (docs/client-runtime-design.md section 17 phase 3).
//
// The gate: the mesh runtime builds two model replicas on two threads that
// share one physical card. The threads bind with `SelectDevice`, and the
// backend state is thread-local, so the two replicas must stay equal after a
// step. A broken device binding or shared backend scratch shows a parameter
// mismatch. The collective stages the gradients through the kernel seam, so the
// device pointers never reach the host.
//
// The test needs one accelerator only: the mesh uses the same device index at
// both positions. The `gpu` tag keeps it out of the default CPU test loop;
// `//src:mesh_runtime_test` is the CPU proof.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/kernels.h"
#include "nanochat/mesh.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/mesh_runtime.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::MeshRuntime;
using nanochat::MeshRuntimeConfig;
using nanochat::MeshSpec;
using nanochat::Model;
using nanochat::ParamView;

constexpr std::uint64_t kSeed = 20241001;
int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

std::vector<int> MakeIds(int rows, int vocab, int a, int b) {
  std::vector<int> ids(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    ids[static_cast<std::size_t>(i)] = (i * a + b) % vocab;
  }
  return ids;
}

// Copies one parameter to the host through the kernel seam, so the value is
// valid on a device backend.
std::vector<ComputeType> HostValues(const ParamView& view) {
  std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
  if (view.count > 0 && view.value != nullptr) {
    nanochat::kernels::Memcpy(
        host.data(), view.value,
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        CopyDir::kDeviceToHost);
  }
  return host;
}

void CompareReplicas(const Model& a, const Model& b) {
  const std::vector<ParamView> pa = a.params();
  const std::vector<ParamView> pb = b.params();
  if (pa.size() != pb.size()) {
    Fail("the replicas have different parameter counts");
    return;
  }
  // The replicas must be byte-identical, so a byte compare is exact.
  for (std::size_t i = 0; i < pa.size(); ++i) {
    const std::vector<ComputeType> left = HostValues(pa[i]);
    const std::vector<ComputeType> right = HostValues(pb[i]);
    if (left.size() != right.size()) {
      Fail("a parameter has a different element count");
      return;
    }
    if (!left.empty() && std::memcmp(left.data(), right.data(),
                                     left.size() * sizeof(ComputeType)) != 0) {
      Fail(std::string("replica parameter mismatch: ") + pa[i].name);
      return;
    }
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  const std::vector<nanochat::DeviceInfo> devices =
      nanochat::EnumerateDevices();
  if (devices.empty() || !devices[0].is_device) {
    std::printf("runtime_gpu_test: skipped (no accelerator)\n");
    return 0;
  }
  const int index = devices[0].index;

  const Config model = TinyConfig();
  MeshRuntimeConfig config;
  config.model = model;
  config.batch = 4;
  config.grad_accum = 2;
  config.seq = 8;
  config.seed = kSeed;
  config.scheduler.num_iterations = 2;

  const int rows = config.batch * config.seq;
  const std::vector<int> tokens = MakeIds(rows, model.vocab_size, 7, 3);
  const std::vector<int> targets = MakeIds(rows, model.vocab_size, 11, 5);

  // One replica on one device: the device thread and the model, without the
  // collective.
  {
    const MeshSpec single_mesh = MeshSpec::Data(std::vector<int>{index});
    MeshRuntime single(single_mesh, config);
    single.TrainStep(tokens.data(), targets.data());
  }

  // Two replicas on the same card: the in-process collective across two device
  // contexts.
  const MeshSpec mesh = MeshSpec::Data(std::vector<int>{index, index});
  MeshRuntime pair(mesh, config);
  if (pair.world_size() != 2) Fail("world_size is not 2");
  for (int step = 0; step < 2; ++step) {
    pair.TrainStep(tokens.data(), targets.data());
    const nanochat::MeshMetrics& metrics = pair.metrics();
    if (static_cast<int>(metrics.loss.size()) != pair.world_size()) {
      Fail("metrics.loss must hold one entry per replica");
    }
    if (!(metrics.step_seconds > 0.0f)) {
      Fail("step_seconds must be positive");
    }
  }
  CompareReplicas(*pair.model(0), *pair.model(1));

  if (g_failures != 0) {
    std::fprintf(stderr, "runtime_gpu_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("runtime_gpu_test: ok (%s)\n", devices[0].name);
  return 0;
}
