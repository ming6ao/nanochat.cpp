// `train_main` -- the old program name, kept as an alias of
// `nanochat_cli train`. The handler lives in `src/train_command.cc`.
//
// This file exists so an existing label, path, or script keeps working while
// the callers move to the one binary (docs/build.md). Delete it once every
// caller names `nanochat_cli train`.

#include "src/command.h"

int main(int argc, char** argv) {
  return nanochat::RunCommandByName("train", argc, argv);
}
