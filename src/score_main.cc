// `score_main` -- the old program name, kept as an alias of
// `nanochat_cli score`. The handler lives in
// `src/score_command.cc`.
//
// This file exists so an existing label, path, or script keeps working while
// the callers move to the one binary (docs/build.md). Delete it once every
// caller names `nanochat_cli score`.

#include "src/command.h"

int main(int argc, char** argv) {
  return nanochat::RunCommandByName("score", argc, argv);
}
