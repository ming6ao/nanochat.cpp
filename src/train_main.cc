// `train_main` — the training CLI. A hand-written parser keeps the tree free of
// a third-party argument library. The actual loop lives in `src/train.cc`.
//
//   tools/nanochat train -- <train_main> --train-shard data/train.bin ...
//
// Every run must go through `tools/nanochat`; see docs/sandbox.md. The entry
// point refuses to start outside the sandbox (`RequireSandboxOrDie`).

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "cli.h"
#include "nanochat/sandbox.h"
#include "train.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: train_main [options]\n"
               "  --train-shard PATH     training shard (repeatable, csv)\n"
               "  --val-shard PATH       validation shard (repeatable, csv)\n"
               "  --batch N              batch size\n"
               "  --num-iterations N     optimizer steps\n"
               "  --log-every N          log cadence\n"
               "  --eval-every N         evaluation cadence (0 disables)\n"
               "  --save-every N         checkpoint cadence (0 disables)\n"
               "  --eval-steps N         validation batches per evaluation\n"
               "  --seed N               weight init and data seed\n"
               "  --checkpoint PATH      write the final checkpoint here\n"
               "  --resume PATH          resume weights from here\n"
               "  --log PATH             also write the run log here\n"
               "  --device NAME          device name for MFU\n"
               "  --no-shuffle           read shards in file order\n"
               "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
               "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("train");

  nanochat::TrainConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (flag == "--no-shuffle") {
      config.shuffle = false;
      continue;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &config.model)) {
        std::fprintf(stderr, "train_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "train_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--train-shard") {
      nanochat::cli::AppendCsv(value, &config.train_shards);
    } else if (flag == "--val-shard") {
      nanochat::cli::AppendCsv(value, &config.val_shards);
    } else if (flag == "--batch") {
      config.batch =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--num-iterations") {
      config.num_iterations =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--log-every") {
      config.log_every =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--eval-every") {
      config.eval_every =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--save-every") {
      config.save_every =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--eval-steps") {
      config.eval_steps =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--seed") {
      config.seed =
          nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else if (flag == "--checkpoint") {
      config.checkpoint_path = value;
    } else if (flag == "--resume") {
      config.resume_path = value;
    } else if (flag == "--log") {
      config.log_path = value;
    } else if (flag == "--device") {
      config.device_name = value;
    } else {
      std::fprintf(stderr, "train_main: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (config.train_shards.empty()) {
    std::fprintf(stderr,
                 "train_main: at least one --train-shard is required\n");
    return 2;
  }
  if (config.num_iterations <= 0 || config.batch <= 0) {
    std::fprintf(stderr,
                 "train_main: --num-iterations and --batch must be > 0\n");
    return 2;
  }

  nanochat::TrainLoop loop(std::move(config));
  const float loss = loop.Run();
  std::printf("train_main: final loss %.6f at step %d\n",
              static_cast<double>(loss), loop.last_step());
  return 0;
}
