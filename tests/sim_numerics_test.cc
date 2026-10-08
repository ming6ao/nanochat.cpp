// tests/sim_numerics_test.cc -- the reference engine's numerical invariant
// (docs/simulator.md section 4.4).
//
// The simulator's reference engine changes the *capabilities* a run reports,
// never the arithmetic. This test drives the real model through a forward and a
// backward pass, records the logits and every parameter gradient, then repeats
// the whole thing with `NANOCHAT_SIM_PROFILE` naming each target device, and
// requires every recorded value to be bit-identical.
//
// The equality is structural: no CPU numeric path reads `GetCaps()`, so with
// the current graph a profile *cannot* perturb a value. The test is therefore a
// regression guard -- it pins that invariant so a future numeric path that did
// read the caps would show up as a diff -- not a proof that no path could. The
// live assertions (under `--config=sim`) confirm the override is actually
// active and that the caps describe the requested device.
//
// The test builds in both fp32 and fp16 (`--config=fp16`), so the fp16 half
// pins that the half-storage path is untouched by the profile too. Under
// `--config=sim` it additionally asserts that the override is actually active,
// so a run that silently fell through to the host caps cannot pass.
//
// No GPU, no fixture, no torch: tier S0, a plain T0 CPU test.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/device_profile.h"
#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"
#include "src/model_impl.h"

namespace {

using nanochat::Caps;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::DeviceProfile;
using nanochat::Model;
using nanochat::ParamView;
using nanochat::TrainModel;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

// Copies a (possibly device-resident) buffer to host floats.
std::vector<float> ReadCompute(const nanochat::ComputeType* src,
                               std::int64_t count) {
  std::vector<nanochat::ComputeType> host(static_cast<std::size_t>(count));
  if (count > 0) {
    nanochat::kernels::Memcpy(
        host.data(), src,
        static_cast<std::size_t>(count) * sizeof(nanochat::ComputeType),
        CopyDir::kDeviceToHost);
  }
  std::vector<float> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] =
        nanochat::AsF(host[static_cast<std::size_t>(i)]);
  }
  return out;
}

// A tiny but structurally complete shape: two blocks, grouped-query attention,
// a sliding window, and a value embedding on the last layer.
Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.rope_base = 100000.0f;
  config.window_pattern = "SL";
  return config;
}

constexpr std::uint64_t kSeed = 20240607;
constexpr int kBatch = 2;
constexpr int kSeq = 8;

// Everything one forward/backward pass produces that the profile must not
// change.
struct Snapshot {
  std::vector<float> logits;
  std::vector<std::vector<float>> grads;
  std::vector<std::string> grad_names;
  float loss = 0.0f;
};

Snapshot RunOnePass() {
  const Config config = TinyConfig();
  const int rows = kBatch * kSeq;
  const int vocab = config.vocab_size;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(kSeed);
  auto* impl = static_cast<TrainModel*>(model.get());

  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] = (i * 7 + 3) % vocab;
    targets[static_cast<std::size_t>(i)] = (i * 11 + 5) % vocab;
  }

  Snapshot snapshot;
  snapshot.loss =
      model->ForwardLoss(tokens.data(), targets.data(), kBatch, kSeq);
  snapshot.logits =
      ReadCompute(impl->raw_logits(),
                  static_cast<std::int64_t>(rows) * config.padded_vocab_size);
  model->Backward();

  for (const ParamView& view : model->params()) {
    snapshot.grad_names.emplace_back(view.name);
    snapshot.grads.push_back(ReadCompute(view.grad, view.count));
  }
  return snapshot;
}

// Compares two snapshots bit for bit. The profile must not perturb a single
// value, so the comparison is exact rather than tolerance-based.
void ExpectIdentical(const Snapshot& a, const Snapshot& b,
                     const std::string& what) {
  ExpectTrue(a.logits == b.logits, what + ": logits are bit-identical");
  ExpectTrue(a.loss == b.loss, what + ": loss is bit-identical");
  ExpectTrue(a.grad_names == b.grad_names,
             what + ": parameter order is unchanged");
  ExpectTrue(a.grads.size() == b.grads.size(),
             what + ": parameter count is unchanged");
  const std::size_t n = std::min(a.grads.size(), b.grads.size());
  for (std::size_t i = 0; i < n; ++i) {
    ExpectTrue(a.grads[i] == b.grads[i],
               Format("%s: grad/%s is bit-identical", what.c_str(),
                      a.grad_names[i].c_str()));
  }
}

void SetProfile(const char* name) {
  if (name == nullptr) {
    ::unsetenv("NANOCHAT_SIM_PROFILE");
    return;
  }
  ::setenv("NANOCHAT_SIM_PROFILE", name, 1);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");

  // The baseline: no profile, the host CPU reference caps.
  SetProfile(nullptr);
  // Only the non-simulator branch below reads `host_caps`; under
  // `--config=sim` the caps are asserted against the profile instead.
  [[maybe_unused]] const Caps host_caps = nanochat::kernels::GetCaps();
  const Snapshot baseline = RunOnePass();

  for (const DeviceProfile* profile : nanochat::AllDeviceProfiles()) {
    SetProfile(profile->name);
    const Caps caps = nanochat::kernels::GetCaps();
#if defined(NANOCHAT_SIMULATOR)
    // Compiled with `--config=sim`: the override must be live, and the caps
    // must describe the requested device rather than the host.
    ExpectTrue(caps.device_name == profile->name,
               std::string("sim: GetCaps reports ") + profile->name);
    ExpectTrue(caps.is_device,
               std::string("sim: ") + profile->name + " reports a device");
    ExpectTrue(caps.compute_major == profile->compute_major &&
                   caps.compute_minor == profile->compute_minor,
               std::string("sim: ") + profile->name +
                   " reports the profile compute capability");
    ExpectTrue(caps.total_memory_bytes == profile->total_memory_bytes,
               std::string("sim: ") + profile->name +
                   " reports the profile memory budget");
#else
    // Built without the override the environment variable is inert, which is
    // itself the default-CPU guarantee.
    ExpectTrue(caps.device_name == host_caps.device_name,
               "without --config=sim the profile variable is inert");
#endif
    const Snapshot simulated = RunOnePass();
    ExpectIdentical(baseline, simulated,
                    std::string("profile ") + profile->name);
  }

  SetProfile(nullptr);
  const Snapshot restored = RunOnePass();
  ExpectIdentical(baseline, restored, "profile cleared");

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("sim numerics: all checks passed (precision=%s, simulator=%s)\n",
#if defined(NANOCHAT_PRECISION_FP16)
              "fp16",
#else
              "fp32",
#endif
#if defined(NANOCHAT_SIMULATOR)
              "on"
#else
              "off"
#endif
  );
  return 0;
}
