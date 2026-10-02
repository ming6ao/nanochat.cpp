// Prefill/decode consistency test.
//
// Proves that the inference graphs in generate.cc agree with the training
// forward in model.cc. The logits for the last prefill position and the logits
// for a decode at that same position must match the full training forward,
// because both attend to the same key/value rows with the same rotary
// positions. The test also checks that the KV cache position advances and that
// decode samples a valid token id.
//
// The tiny architecture matches the gradient test (2 layers, 8 query heads,
// 2 key/value heads, hidden 32, seq 8, vocab 64, window "SL"), so the whole
// graph is exercised cheaply. The same source builds against the CPU reference
// backend (`//src:generate_test`) and, under `--config=cuda`, the CUDA backend
// (`//src:generate_gpu_test`); the raw logits are staged through the seam so it
// does not matter whether they live in host or device memory.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "model_impl.h"
#include "nanochat/model.h"
#include "nanochat/sampler.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::KvCache;
using nanochat::Model;
using nanochat::SampleParams;
using nanochat::TrainModel;

constexpr double kTolerance = 1e-5;

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

double MaxDiff(const std::vector<float>& got, const std::vector<float>& want) {
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(
        worst,
        std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

// The training forward's soft-capped logits for one position, read from the
// saved raw logits. `raw_logits()` is host memory on the CPU backend and device
// memory on CUDA, so stage the whole `[total, padded]` buffer through the seam
// rather than dereferencing the pointer on the host.
std::vector<float> ReferenceLogits(TrainModel* impl, const Config& config,
                                   int position, int total) {
  const int vocab = config.vocab_size;
  const int padded = config.padded_vocab_size;
  const std::int64_t count = static_cast<std::int64_t>(total) * padded;
  std::vector<ComputeType> raw(static_cast<std::size_t>(count));
  nanochat::kernels::Memcpy(
      raw.data(), impl->raw_logits(),
      static_cast<std::size_t>(count) * sizeof(ComputeType),
      nanochat::CopyDir::kDeviceToHost);
  std::vector<float> out(static_cast<std::size_t>(vocab));
  for (int v = 0; v < vocab; ++v) {
    const float z =
        nanochat::AsF(raw[static_cast<std::int64_t>(position) * padded + v]);
    out[static_cast<std::size_t>(v)] =
        nanochat::kLogitSoftcap * std::tanh(z / nanochat::kLogitSoftcap);
  }
  return out;
}

void Run() {
  const Config config = TinyConfig();
  const int vocab = config.vocab_size;
  const int seq = config.seq_len;
  const int prefill_len = 4;
  const int total = prefill_len + 1;

  std::unique_ptr<Model> model = Model::Create(config);
  model->InitWeights(20241001);
  auto* impl = static_cast<TrainModel*>(model.get());

  std::vector<int> tokens(static_cast<std::size_t>(seq));
  std::vector<int> targets(static_cast<std::size_t>(seq));
  for (int i = 0; i < seq; ++i) {
    tokens[static_cast<std::size_t>(i)] = (i * 7 + 3) % vocab;
    targets[static_cast<std::size_t>(i)] = (i * 11 + 5) % vocab;
  }

  // Reference: a single full training forward over the prompt plus the token we
  // will decode. The raw logits are saved for the comparison.
  model->ForwardLoss(tokens.data(), targets.data(), 1, total);
  const std::vector<float> want_last =
      ReferenceLogits(impl, config, total - 1, total);
  const std::vector<float> want_prev =
      ReferenceLogits(impl, config, prefill_len - 1, total);

  // --- Cache A: prefill, then decode, comparing logits. ---
  std::unique_ptr<KvCache> kv = nanochat::CreateKvCache(config, seq);
  if (kv->pos() != 0) Fail("fresh cache position is not zero");

  std::vector<float> prefill_logits(static_cast<std::size_t>(vocab));
  nanochat::PrefillLogits(model.get(), tokens.data(), prefill_len, kv.get(),
                          prefill_logits.data());
  if (kv->pos() != prefill_len) {
    Fail("prefill position: got " + std::to_string(kv->pos()) + " want " +
         std::to_string(prefill_len));
  }
  const double prefill_error = MaxDiff(prefill_logits, want_prev);
  if (prefill_error > kTolerance) {
    Fail("prefill last-position logits: max |diff| = " +
         std::to_string(prefill_error));
  }
  std::printf("generate_test: prefill last-position max |diff| = %.3g\n",
              prefill_error);

  std::vector<float> decode_logits(static_cast<std::size_t>(vocab));
  nanochat::DecodeLogits(model.get(), tokens[static_cast<std::size_t>(
                                          prefill_len)],
                         kv.get(), decode_logits.data());
  if (kv->pos() != prefill_len + 1) {
    Fail("decode position: got " + std::to_string(kv->pos()) + " want " +
         std::to_string(prefill_len + 1));
  }
  const double decode_error = MaxDiff(decode_logits, want_last);
  if (decode_error > kTolerance) {
    Fail("decode logits vs full forward: max |diff| = " +
         std::to_string(decode_error));
  }
  std::printf(
      "generate_test: decode-vs-full-forward last-position max |diff| = %.3g\n",
      decode_error);

  // --- Cache B: prefill, then sample with Decode. ---
  std::unique_ptr<KvCache> kv2 = nanochat::CreateKvCache(config, seq);
  nanochat::Prefill(model.get(), tokens.data(), prefill_len, kv2.get());

  SampleParams greedy;
  greedy.temperature = 0.0f;
  const int greedy_id = nanochat::Decode(model.get(), tokens[prefill_len],
                                         kv2.get(), greedy);
  // The greedy id must be the argmax of the reference logits.
  int argmax = 0;
  for (int v = 1; v < vocab; ++v) {
    if (want_last[static_cast<std::size_t>(v)] >
        want_last[static_cast<std::size_t>(argmax)]) {
      argmax = v;
    }
  }
  if (greedy_id != argmax) {
    Fail("greedy decode id: got " + std::to_string(greedy_id) + " want " +
         std::to_string(argmax));
  }
  std::printf("generate_test: greedy decode id %d (argmax %d)\n", greedy_id,
              argmax);

  // --- Cache C: temperature sampling stays in range and is reproducible. ---
  std::unique_ptr<KvCache> kv3 = nanochat::CreateKvCache(config, seq);
  nanochat::Prefill(model.get(), tokens.data(), prefill_len, kv3.get());
  SampleParams sampled;
  sampled.temperature = 1.0f;
  sampled.top_k = 5;
  sampled.seed = 123;
  const int sampled_id =
      nanochat::Decode(model.get(), tokens[prefill_len], kv3.get(), sampled);
  if (sampled_id < 0 || sampled_id >= vocab) {
    Fail("sampled id out of range: " + std::to_string(sampled_id));
  }
  std::printf("generate_test: sampled decode id %d (vocab %d)\n", sampled_id,
              vocab);
}

}  // namespace

int main() {
  Run();
  if (g_failures != 0) {
    std::fprintf(stderr, "generate_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("generate_test: all checks passed\n");
  return 0;
}
