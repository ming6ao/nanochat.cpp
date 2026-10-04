// `eval_bench` -- a small end-to-end benchmark for the forward-only evaluation
// path (docs/grad-mode.md).
//
// It builds a model, runs `ForwardLoss` under `NoGradGuard` on random tokens,
// and reports the per-forward time, the token rate, and the bytes the eval
// arena reserves. It is the reproduction tool for the C++ column of the
// memory/throughput comparison in docs/performance.md.
//
// The benchmark times the model forward only. It does not read a data shard.
// It does not run the training forward. So the benchmark stays small. It also
// fits the card at shapes where the training workspace does not.
//
// Run through the entry point:
//
//   tools/nanochat build --config=cuda //src:eval_bench
//   tools/nanochat bench -- bazel-bin/src/eval_bench --batch 32 --seq 1024
//
// The same binary built without `--config=cuda` runs the CPU reference
// backend, so the file stays backend-agnostic.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/model_impl.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: eval_bench [options]\n"
               "  --batch N              batch size (default 32)\n"
               "  --iters N              timed forwards (default 10)\n"
               "  --warmup N             warm-up forwards (default 3)\n"
               "  --seed N               token/weight seed (default 42)\n"
               "  --json                 write a nanochat.bench.v1 JSON row\n"
               "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
               "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

// Deterministic xorshift64* so a run reproduces on any backend.
class Rng {
 public:
  explicit Rng(std::uint64_t seed)
      : state_(seed != 0 ? seed : 0x9e3779b97f4a7c15ull) {}
  std::uint64_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }
  int Token(int vocab) {
    return static_cast<int>(Next() % static_cast<std::uint64_t>(vocab));
  }

 private:
  std::uint64_t state_;
};

// Frozen Phase 0 end-to-end baseline for the production SSSL shape. The
// benchmark records the measured rate next to this constant, so one JSON report
// shows the gate result. The numbers live in docs/attention-baseline-e2e.json.
constexpr double kBaselineTokensPerSecond = 55000.0;
constexpr double kBaselineForwardMs = 74.472727;

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  nanochat::Config model_config;
  int batch = 32;
  int iters = 10;
  int warmup = 3;
  std::uint64_t seed = 42;
  bool json = false;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (flag == "--json") {
      json = true;
      continue;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "eval_bench: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "eval_bench: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--batch") {
      batch = nanochat::cli::ParseInt(value, &parsed) ? parsed : 0;
    } else if (flag == "--iters") {
      iters = nanochat::cli::ParseInt(value, &parsed) ? parsed : 0;
    } else if (flag == "--warmup") {
      warmup = nanochat::cli::ParseInt(value, &parsed) ? parsed : 0;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else {
      std::fprintf(stderr, "eval_bench: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  const int seq = model_config.seq_len;
  if (batch <= 0 || seq <= 0 || iters <= 0 || warmup < 0) {
    std::fprintf(stderr, "eval_bench: bad shape or iteration count\n");
    return 2;
  }

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  model->InitWeights(seed);

  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  std::vector<int> tokens(static_cast<std::size_t>(rows));
  std::vector<int> targets(static_cast<std::size_t>(rows));
  Rng rng(seed);
  for (std::int64_t i = 0; i < rows; ++i) {
    tokens[static_cast<std::size_t>(i)] = rng.Token(model_config.vocab_size);
    targets[static_cast<std::size_t>(i)] = rng.Token(model_config.vocab_size);
  }

  float sink = 0.0f;
  {
    // The guard selects the forward-only arena and does not save activations.
    nanochat::NoGradGuard guard(model.get());
    for (int i = 0; i < warmup; ++i) {
      sink += model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    }
    nanochat::kernels::Synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      sink += model->ForwardLoss(tokens.data(), targets.data(), batch, seq);
    }
    nanochat::kernels::Synchronize();
    const auto finish = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(finish - start).count();
    const double ms_per_forward = elapsed / static_cast<double>(iters) * 1e3;
    const double tokens_per_second =
        static_cast<double>(rows) * static_cast<double>(iters) / elapsed;
    const auto* impl = static_cast<nanochat::TrainModel*>(model.get());
    const std::size_t arena_bytes = impl->workspace_bytes();

    char shape[160];
    std::snprintf(
        shape, sizeof(shape), "B=%d T=%d L=%d H=%d KV=%d C=%d V=%d W=%s", batch,
        seq, model_config.num_layers, model_config.num_heads,
        model_config.num_kv_heads, model_config.hidden_dim,
        model_config.padded_vocab_size, model_config.window_pattern.c_str());

    if (json) {
      std::printf(
          "{\"schema\":\"nanochat.bench.v1\",\"rows\":[{\"name\":"
          "\"eval_forward\",\"shape\":\"%s\",\"ms\":%.6f,\"gflops\":0.0}],"
          "\"arena_bytes\":%zu,\"tokens_per_second\":%.1f,"
          "\"forward_ms\":%.6f,\"baseline_tokens_per_second\":%.1f,"
          "\"baseline_forward_ms\":%.6f,\"speedup_vs_baseline\":%.4f,"
          "\"batch\":%d,\"seq\":%d}\n",
          shape, ms_per_forward, arena_bytes, tokens_per_second, ms_per_forward,
          kBaselineTokensPerSecond, kBaselineForwardMs,
          tokens_per_second / kBaselineTokensPerSecond, batch, seq);
    } else {
      std::printf("eval_bench: %s\n", shape);
      std::printf("  arena       : %zu bytes (%.1f MiB)\n", arena_bytes,
                  static_cast<double>(arena_bytes) / (1024.0 * 1024.0));
      std::printf("  ms/forward  : %.3f\n", ms_per_forward);
      std::printf("  tokens/s    : %.0f\n", tokens_per_second);
      std::printf("  baseline    : %.0f tokens/s (%.3f ms/forward)\n",
                  kBaselineTokensPerSecond, kBaselineForwardMs);
      std::printf("  speedup     : %.4fx vs baseline\n",
                  tokens_per_second / kBaselineTokensPerSecond);
      std::printf(
          "  mean loss   : %.6f\n",
          static_cast<double>(sink / static_cast<float>(warmup + iters)));
    }
  }

  return 0;
}
