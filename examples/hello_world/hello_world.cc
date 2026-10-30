// The smallest end-to-end inference example: a tiny model, a fixed seed, and
// no model file. It proves the public API works outside src/:
// RequireSandboxOrDie, Config, Model::Create, InitWeights, GenerateParams, and
// GenerateBatch (docs/checkpoint-metadata-and-examples-plan.md section 6,
// decisions E1-E5, stage B.1).
//
// Run it with:
//   tools/nanochat run t0-cpu -- bazel-bin/examples/hello_world/hello_world
//
// To try a larger config, change this one line in TinyConfig() (keep head_dim
// even and num_kv_heads a divisor of num_heads):
//   config.hidden_dim = 32;
//
// The same source is also `//examples/hello_world:hello_world_test`. For the
// fixed seed it pins the row count, the shape, and the mask, but not the token
// ids, which a legitimate kernel change may move (stage B.4).

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"

namespace {

// A tiny architecture that passes Config::Validate: hidden_dim is at least 24,
// head_dim is even (32 / 4), num_kv_heads divides num_heads (4 / 2), and
// padded_vocab_size equals vocab_size.
nanochat::Config TinyConfig() {
  nanochat::Config config;
  config.num_layers = 2;
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 16;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

constexpr std::uint64_t kSeed = 20240517;  // weights and sampling, fixed
constexpr int kPrompt[] = {1, 5, 9, 13};   // ids: no tokenizer is needed
constexpr int kPromptLen = 4;
constexpr int kRows = 3;
constexpr int kMaxTokens = 4;
constexpr int kRowLen = kPromptLen + kMaxTokens;

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("example");

  std::string error;
  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(TinyConfig(), &error);
  if (model == nullptr) {
    std::fprintf(stderr, "hello_world: bad config: %s\n", error.c_str());
    return 1;
  }
  model->InitWeights(kSeed);

  nanochat::GenerateParams params;
  params.num_samples = kRows;
  params.max_tokens = kMaxTokens;
  params.temperature = 0.8f;
  params.top_k = 8;
  params.seed = kSeed;

  std::vector<nanochat::GeneratedSequence> rows;
  nanochat::GenerateBatch(model.get(), kPrompt, kPromptLen, params, &rows);

  int failures = static_cast<int>(rows.size()) != kRows;
  for (const nanochat::GeneratedSequence& row : rows) {
    std::printf("row:");
    for (int id : row.tokens) std::printf(" %d", id);
    std::printf("\n");
    // The stage B.4 guard: pin the shape and the mask, never the ids.
    const int tokens = static_cast<int>(row.tokens.size());
    if (tokens != kRowLen || static_cast<int>(row.mask.size()) != kRowLen) {
      ++failures;
      continue;
    }
    for (int i = 0; i < kRowLen; ++i) {
      failures += row.mask[i] != (i < kPromptLen ? 0 : 1);
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "hello_world: %d check(s) failed\n", failures);
    return 1;
  }
  return 0;
}
