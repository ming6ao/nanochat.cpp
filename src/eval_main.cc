// `eval_main` — forward-only bits-per-byte evaluation. Reads pre-tokenized
// shards, builds the model, optionally loads a checkpoint, and prints the bpb.
//
//   tools/nanochat eval -- <eval_main> --val-shard data/val.bin --model model.ckpt

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cli.h"
#include "nanochat/dataloader.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "train.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: eval_main [options]\n"
               "  --val-shard PATH       validation shard (repeatable, csv)\n"
               "  --steps N              number of batches to evaluate\n"
               "  --batch N              batch size\n"
               "  --seq N                sequence length (model flag below)\n"
               "  --model PATH           checkpoint to load\n"
               "  --seed N               weight init seed when no checkpoint\n"
               "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
               "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("eval");

  nanochat::Config model_config;
  std::vector<std::string> shards;
  std::string model_path;
  int batch = 8;
  int steps = 8;
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
    if (flag == "--val-shard") {
      nanochat::cli::AppendCsv(value, &shards);
    } else if (flag == "--steps") {
      steps = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
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

  if (shards.empty()) {
    std::fprintf(stderr, "eval_main: at least one --val-shard is required\n");
    return 2;
  }
  if (batch <= 0 || steps <= 0) {
    std::fprintf(stderr, "eval_main: --batch and --steps must be > 0\n");
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

  nanochat::DataLoader loader(std::move(shards), batch, model_config.seq_len,
                              seed, false);
  const float bpb = nanochat::EvalBpb(model.get(), &loader, steps);
  std::printf("eval_main: val bpb %.6f\n", static_cast<double>(bpb));
  return 0;
}
