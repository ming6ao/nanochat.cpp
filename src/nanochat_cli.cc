// `nanochat_cli` -- the one stage command binary (docs/build.md).
//
//   tools/nanochat train -- <nanochat_cli> train --train-parquet
//   'data/*.parquet'
//       --tokenizer tokenizer.nctoken ...
//
// Every run must go through `tools/nanochat`; see docs/sandbox.md. The
// subcommand guard refuses to start outside the sandbox. Each command lives in
// `src/<name>_command.cc`, and `src/<name>_main.cc` is a thin alias that calls
// the same handler, so an old label or path keeps working.

#include <string_view>

#include "src/command.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    nanochat::PrintCommandList();
    return 2;
  }
  const std::string_view requested = argv[1];
  if (requested == "help" || requested == "-h" || requested == "--help") {
    return nanochat::PrintCommandList();
  }
  // Drop the subcommand word, so the handler sees a normal `main` argument
  // list: `argv[0]` is the program name and the flags start at `argv[1]`.
  return nanochat::RunCommandByName(requested, argc - 1, argv + 1);
}
