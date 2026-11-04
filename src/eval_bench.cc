// `eval_bench` -- the old program name, kept as an alias of
// `nanochat_cli eval_bench`. The benchmark lives in
// `src/eval_bench_command.cc`.
//
// This file exists so an existing label, path, or script keeps working while
// the callers move to the one binary (docs/build.md). Delete it once every
// caller names `nanochat_cli eval_bench`.

#include "src/command.h"

int main(int argc, char** argv) {
  return nanochat::RunCommandByName("eval_bench", argc, argv);
}
