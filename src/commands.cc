// The subcommand table and the one dispatch path (docs/build.md).

#include "src/command.h"

#include <cstdio>
#include <string_view>

#include "nanochat/sandbox.h"

namespace nanochat {
namespace {

// The table, in help order. A new subcommand adds one row here and one
// `Run<Name>` function in `src/<name>_command.cc`. The empty name ends it.
const Command kCommands[] = {
    {"train", "train_main", "one training run over packed parquet rows",
     &RunTrain},
    {"eval", "eval_main", "forward-only bits-per-byte evaluation", &RunEval},
    {"generate", "generate_main", "sample tokens from a checkpoint",
     &RunGenerate},
    {"score", "score_main", "stateless scorer over one evaluation fixture",
     &RunScore},
    {"sft", "sft_main", "supervised fine-tuning on a packed SFT shard",
     &RunSft},
    {"rl_worker", "rl_worker",
     "the reinforcement-learning worker the Python bridge drives",
     &RunRlWorker},
    {"rl_step", "rl_step", "one file-driven reinforcement-learning step",
     &RunRlStep},
    {"roofline", "roofline", "the performance model and the roofline report",
     &RunRoofline},
    {"eval_bench", "eval_bench", "the forward-only evaluation benchmark",
     &RunEvalBench},
    {},
};

}  // namespace

const Command* Commands() { return kCommands; }

int RunCommandByName(std::string_view name, int argc, char** argv) {
  for (const Command* command = kCommands; !command->name.empty(); ++command) {
    if (command->name != name) continue;
    // One guard for every subcommand. `SandboxProfile` reads the environment,
    // so the task name only labels the failure message.
    RequireSandboxOrDie(command->log_prefix);
    return command->run(argc, argv);
  }
  std::fprintf(stderr, "nanochat_cli: unknown command \"%.*s\"\n",
               static_cast<int>(name.size()), name.data());
  return 2;
}

int PrintCommandList() {
  std::printf("usage: nanochat_cli <command> [options]\n\n");
  std::printf("commands:\n");
  for (const Command* command = kCommands; !command->name.empty(); ++command) {
    std::printf("  %-12.*s %.*s\n", static_cast<int>(command->name.size()),
                command->name.data(), static_cast<int>(command->summary.size()),
                command->summary.data());
  }
  std::printf(
      "\nRun `nanochat_cli <command> --help` for the options of one "
      "command.\n");
  return 0;
}

}  // namespace nanochat
