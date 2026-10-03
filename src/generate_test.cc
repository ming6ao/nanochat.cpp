// Prefill/decode consistency test.
//
// Proves that the inference graphs in generate.cc agree with the training
// forward in model.cc. The logits for the last prefill position and the logits
// for a decode at that same position must match the full training forward,
// because both attend to the same key/value rows with the same rotary
// positions. The test also checks that the KV cache position advances, that
// decode samples a valid token id, and that batched generation
// (`GenerateBatch`) reproduces a naive single-row loop (prefill one cache, then
// decode it token by token) for greedy and sampled decoding, including the
// per-row terminal id and the returned 1/0 mask.
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
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) -
                                      static_cast<double>(want[i])));
  }
  return worst;
}

// A single row produced without the batched engine: prefill one cache and
// decode it one token at a time. This is the reference the batched loop must
// reproduce exactly for greedy decoding and, because `Decode` re-seeds itself
// from `params.seed` on every call, for sampled decoding as well.
struct NaiveRow {
  std::vector<int> tokens;
  std::vector<std::uint8_t> mask;
};

NaiveRow NaiveGenerate(Model* model, const Config& config, const int* prompt,
                       int prompt_len, const nanochat::GenerateParams& params,
                       int stop_id) {
  NaiveRow row;
  std::vector<int> effective;
  if (params.bos_id >= 0) effective.push_back(params.bos_id);
  for (int i = 0; i < prompt_len; ++i) effective.push_back(prompt[i]);
  if (effective.empty()) return row;

  const int max_new = params.max_tokens > 0 ? params.max_tokens : 0;
  const int capacity = static_cast<int>(effective.size()) + max_new + 1;
  std::unique_ptr<KvCache> kv = nanochat::CreateKvCache(config, capacity);
  const int prefill_len = static_cast<int>(effective.size()) - 1;
  if (prefill_len > 0) {
    nanochat::Prefill(model, effective.data(), prefill_len, kv.get());
  }
  row.tokens = effective;
  row.mask.assign(effective.size(), 0);

  SampleParams sample;
  sample.temperature = params.temperature;
  sample.top_k = params.top_k;
  sample.seed = params.seed;

  int next = effective.back();
  for (int step = 0; step < max_new; ++step) {
    const int token = nanochat::Decode(model, next, kv.get(), sample);
    if (token < 0) break;
    if (stop_id >= 0 && token == stop_id) break;
    row.tokens.push_back(token);
    row.mask.push_back(1);
    next = token;
  }
  return row;
}

// Runs one batched generation and checks every row against the naive loop,
// including the 1/0 mask (0 for prompt ids, 1 for sampled ids) and the
// invariant that a terminal id is never emitted.
void CheckBatchMatchesNaive(Model* model, const Config& config,
                            const int* prompt, int prompt_len,
                            const nanochat::GenerateParams& params,
                            const std::vector<int>& stops,
                            const std::string& label) {
  std::vector<nanochat::GeneratedSequence> rows;
  nanochat::GenerateBatch(model, prompt, prompt_len, params, &rows);
  if (rows.size() != stops.size()) {
    Fail(label + ": row count " + std::to_string(rows.size()) + " want " +
         std::to_string(stops.size()));
    return;
  }
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const NaiveRow want =
        NaiveGenerate(model, config, prompt, prompt_len, params, stops[r]);
    if (rows[r].tokens != want.tokens) {
      Fail(label + ": row " + std::to_string(r) +
           " tokens differ from the naive loop");
    }
    if (rows[r].mask != want.mask) {
      Fail(label + ": row " + std::to_string(r) +
           " mask differs from the naive loop");
    }
    if (rows[r].mask.size() != rows[r].tokens.size()) {
      Fail(label + ": row " + std::to_string(r) +
           " mask and tokens are not aligned");
    }
    for (std::size_t i = 0; i < rows[r].mask.size(); ++i) {
      if (rows[r].mask[i] != 0 && rows[r].mask[i] != 1) {
        Fail(label + ": row " + std::to_string(r) + " mask entry " +
             std::to_string(i) + " is not 0 or 1");
        break;
      }
    }
    const int stop = stops[r];
    if (stop >= 0) {
      for (int id : rows[r].tokens) {
        if (id == stop) {
          Fail(label + ": row " + std::to_string(r) +
               " emitted its terminal id");
          break;
        }
      }
    }
  }
  std::printf("generate_test: %s matched the naive single-row loop\n",
              label.c_str());
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
  nanochat::DecodeLogits(model.get(),
                         tokens[static_cast<std::size_t>(prefill_len)],
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
  const int greedy_id =
      nanochat::Decode(model.get(), tokens[prefill_len], kv2.get(), greedy);
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

  // --- Batched generation matches a naive single-row loop. ---
  const int batched_prompt_len = 4;
  std::vector<int> batched_prompt(tokens.begin(),
                                  tokens.begin() + batched_prompt_len);

  // Probe the deterministic greedy continuation once so the terminal ids below
  // are guaranteed to be sampled; this exercises the early-stop path instead of
  // only the maximum-token path.
  nanochat::GenerateParams greedy_probe;
  greedy_probe.num_samples = 1;
  greedy_probe.max_tokens = 6;
  greedy_probe.temperature = 0.0f;
  greedy_probe.seed = 4242;
  greedy_probe.bos_id = 1;
  const NaiveRow probe_row =
      NaiveGenerate(model.get(), config, batched_prompt.data(),
                    batched_prompt_len, greedy_probe, -1);
  const std::size_t probe_prefix =
      static_cast<std::size_t>(batched_prompt_len) +
      (greedy_probe.bos_id >= 0 ? 1 : 0);
  if (probe_row.tokens.size() < probe_prefix + 4) {
    Fail("greedy probe did not produce enough tokens");
  } else {
    // Greedy with a prepended beginning-of-sequence id and a different terminal
    // id per row, so the rows stop at different lengths.
    const std::vector<int> greedy_stops = {
        probe_row.tokens[probe_prefix + 1],
        probe_row.tokens[probe_prefix + 2],
        probe_row.tokens[probe_prefix + 3]};
    nanochat::GenerateParams greedy_batch;
    greedy_batch.num_samples = 3;
    greedy_batch.max_tokens = 6;
    greedy_batch.temperature = 0.0f;
    greedy_batch.top_k = 0;
    greedy_batch.seed = 4242;
    greedy_batch.bos_id = 1;
    greedy_batch.stop_ids = greedy_stops.data();
    CheckBatchMatchesNaive(model.get(), config, batched_prompt.data(),
                           batched_prompt_len, greedy_batch, greedy_stops,
                           "greedy batch");

    // Greedy with one shared terminal id, exercising the `stop_id` path.
    nanochat::GenerateParams global_stop;
    global_stop.num_samples = 2;
    global_stop.max_tokens = 6;
    global_stop.temperature = 0.0f;
    global_stop.top_k = 0;
    global_stop.seed = 4242;
    global_stop.bos_id = 1;
    global_stop.stop_id = probe_row.tokens[probe_prefix + 2];
    CheckBatchMatchesNaive(model.get(), config, batched_prompt.data(),
                           batched_prompt_len, global_stop,
                           std::vector<int>(2, global_stop.stop_id),
                           "global stop batch");
  }

  // Sampled decoding under a fixed seed; each row must still match the naive
  // loop, and two identical calls must produce identical rows.
  nanochat::GenerateParams sampled_batch;
  sampled_batch.num_samples = 2;
  sampled_batch.max_tokens = 4;
  sampled_batch.temperature = 1.0f;
  sampled_batch.top_k = 4;
  sampled_batch.seed = 99;
  sampled_batch.stop_id = 7;
  const std::vector<int> sampled_stops = {7, 7};
  sampled_batch.stop_ids = sampled_stops.data();
  CheckBatchMatchesNaive(model.get(), config, batched_prompt.data(),
                         batched_prompt_len, sampled_batch, sampled_stops,
                         "sampled batch");

  std::vector<nanochat::GeneratedSequence> sampled_a;
  std::vector<nanochat::GeneratedSequence> sampled_b;
  nanochat::GenerateBatch(model.get(), batched_prompt.data(),
                          batched_prompt_len, sampled_batch, &sampled_a);
  nanochat::GenerateBatch(model.get(), batched_prompt.data(),
                          batched_prompt_len, sampled_batch, &sampled_b);
  if (sampled_a.size() != sampled_b.size()) {
    Fail("sampled batch is not reproducible: row count changed");
  } else {
    for (std::size_t r = 0; r < sampled_a.size(); ++r) {
      if (sampled_a[r].tokens != sampled_b[r].tokens ||
          sampled_a[r].mask != sampled_b[r].mask) {
        Fail("sampled batch is not reproducible at row " + std::to_string(r));
      }
    }
  }
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
