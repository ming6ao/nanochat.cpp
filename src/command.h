#ifndef NANOCHAT_SRC_COMMAND_H_
#define NANOCHAT_SRC_COMMAND_H_

#include <string_view>

// The subcommand seam of `nanochat_cli`.
//
// One binary runs every stage command, so the flag parser, the sandbox check,
// and the help text exist once instead of in one `main` per command. A command
// lives in `src/<name>_command.cc` and exports one `Run<Name>` function with
// the signature of a `main`: it ignores `argv[0]` and reads flags from
// `argv[1]`.
//
// The old program names stay as thin aliases, `src/<name>_main.cc`, so every
// existing label, path, and log line keeps working while the callers move over
// (docs/build.md). See `src/nanochat_cli.cc` for the entry point.
//
// `argv` stays a `char**` because `main` fixes it, and every handler signature
// follows `main` so an alias can pass its argument list through unchanged. All
// other text here is `std::string_view`.

namespace nanochat {

// One subcommand. The three strings point at string literals in the table, so
// they outlive every use and own nothing.
struct Command {
  // The word after `nanochat_cli`, and the task name the sandbox guard reports.
  std::string_view name;
  // The prefix of every diagnostic, which is the old program name so a reader
  // and a log pattern keep matching (`tools/dump_fixtures.py`). It stays a
  // `const char*` because `RequireSandboxOrDie` takes a C string, and the
  // sandbox header is frozen (AGENTS.md section 1).
  const char* log_prefix;
  // One line for `nanochat_cli help`.
  std::string_view summary;
  // The handler. `argv[0]` is the program name and is not read as a flag.
  int (*run)(int argc, char** argv);
};

// The table, terminated by an entry whose `name` is empty.
const Command* Commands();

// Runs the subcommand `name`. Applies the sandbox guard, then calls the
// handler with `argv[0]` still in place. Returns 2 for an unknown name.
int RunCommandByName(std::string_view name, int argc, char** argv);

// Prints the subcommand list. Returns 0.
int PrintCommandList();

// The command handlers, one per `src/<name>_command.cc`.
int RunTrain(int argc, char** argv);
int RunEval(int argc, char** argv);
int RunGenerate(int argc, char** argv);
int RunScore(int argc, char** argv);
int RunSft(int argc, char** argv);
int RunRlWorker(int argc, char** argv);
int RunRlStep(int argc, char** argv);
int RunRoofline(int argc, char** argv);
int RunEvalBench(int argc, char** argv);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_COMMAND_H_
