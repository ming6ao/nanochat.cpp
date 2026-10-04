// `eval_main` -- forward-only bits-per-byte evaluation. Reads pre-tokenized
// shards, builds the model, optionally loads a checkpoint, and prints the bpb
// for the train and/or validation split.
//
// The reference's `--split-tokens` describes how many tokens of a split to
// evaluate; here it is converted to a batch count with
// `steps = split_tokens / (batch * seq_len)`, mirroring `scripts/base_eval.py`.
//
//   tools/nanochat eval -- <eval_main> --val-shard data/val.bin --model
//   model.ckpt
//   tools/nanochat eval -- <eval_main> --train-shard data/train.bin
//   --val-shard data/val.bin --split-tokens 8192

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/train.h"

namespace {

void Usage() {
  std::fprintf(
      stderr,
      "usage: eval_main [options]\n"
      "  --train-shard PATH     training shard (repeatable, csv)\n"
      "  --val-shard PATH       validation shard (repeatable, csv)\n"
      "  --steps N              number of batches to evaluate\n"
      "  --split-tokens N       tokens per split; overrides --steps with\n"
      "                         N / (batch * seq_len) batches\n"
      "  --batch N              batch size\n"
      "  --seq N                sequence length (model flag below)\n"
      "  --model PATH           checkpoint to load\n"
      "  --seed N               weight init seed when no checkpoint\n"
      "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
      "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

// Evaluates one split (a non-empty shard list) and prints its bpb. Returns the
// result so a caller could compare splits.
float EvalSplit(nanochat::Model* model, const std::vector<std::string>& shards,
                const char* split, int batch, int seq, int steps,
                std::uint64_t seed) {
  nanochat::DataLoader loader(shards, batch, seq, seed, false);
  const float bpb = nanochat::EvalBpb(model, &loader, steps);
  std::printf("eval_main: %s bpb %.6f\n", split, static_cast<double>(bpb));
  return bpb;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("eval");

  nanochat::Config model_config;
  std::vector<std::string> train_shards;
  std::vector<std::string> val_shards;
  std::string model_path;
  int batch = 8;
  int steps = 8;
  int split_tokens = -1;
  std::uint64_t seed = 42;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "eval_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "eval_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--train-shard") {
      nanochat::cli::AppendCsv(value, &train_shards);
    } else if (flag == "--val-shard") {
      nanochat::cli::AppendCsv(value, &val_shards);
    } else if (flag == "--steps") {
      steps = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--split-tokens") {
      split_tokens =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--batch") {
      batch = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--model") {
      model_path = value;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else {
      std::fprintf(stderr, "eval_main: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (train_shards.empty() && val_shards.empty()) {
    std::fprintf(stderr,
                 "eval_main: at least one --train-shard or --val-shard is "
                 "required\n");
    return 2;
  }
  if (batch <= 0 || model_config.seq_len <= 0) {
    std::fprintf(stderr, "eval_main: --batch and --seq must be > 0\n");
    return 2;
  }
  if (split_tokens > 0) {
    const int per_step = batch * model_config.seq_len;
    steps = per_step > 0 ? split_tokens / per_step : 0;
    if (steps <= 0) steps = 1;
  }
  if (steps <= 0) {
    std::fprintf(stderr, "eval_main: --steps and --split-tokens must be > 0\n");
    return 2;
  }

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  if (!model_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "eval_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  } else {
    model->InitWeights(seed);
  }

  if (!train_shards.empty()) {
    EvalSplit(model.get(), train_shards, "train", batch, model_config.seq_len,
              steps, seed);
  }
  if (!val_shards.empty()) {
    EvalSplit(model.get(), val_shards, "val", batch, model_config.seq_len,
              steps, seed);
  }
  return 0;
}
