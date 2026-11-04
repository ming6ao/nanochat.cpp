// The flag parser of `nanochat_cli` (src/cli.h).
//
// One binary runs every stage command, so this parser is on the path of every
// command line the launcher builds. The test pins the rules the nine old flag
// loops implemented separately: both spellings of a flag, the aliases, the
// shared model flags, the bare switches, and the exit status of every failure.

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectTrue(bool value, const char* what) {
  if (!value) Fail(std::string("expected true: ") + what);
}

void ExpectInt(int actual, int expected, const char* what) {
  if (actual != expected) {
    Fail(std::string(what) + ": expected " + std::to_string(expected) +
         ", got " + std::to_string(actual));
  }
}

void ExpectStatus(int actual, int expected, const char* what) {
  if (actual != expected) {
    Fail(std::string(what) + ": expected status " + std::to_string(expected) +
         ", got " + std::to_string(actual));
  }
}

// Builds the argument vector the parser reads. `argv[0]` is the program name.
std::vector<std::string> g_storage;

std::vector<char*> Args(std::initializer_list<const char*> tokens) {
  g_storage.clear();
  g_storage.emplace_back("nanochat_cli");
  for (const char* token : tokens) g_storage.emplace_back(token);
  std::vector<char*> argv;
  for (std::string& token : g_storage) argv.push_back(token.data());
  return argv;
}

int Parse(nanochat::cli::FlagParser* flags,
          std::initializer_list<const char*> tokens) {
  std::vector<char*> argv = Args(tokens);
  return flags->Parse(static_cast<int>(argv.size()), argv.data());
}

// The space-separated spelling takes the next argument.
void TestSpaceSpelling() {
  int batch = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddInt("--batch", &batch);
  ExpectStatus(Parse(&flags, {"--batch", "7"}), -1, "space spelling");
  ExpectInt(batch, 7, "space spelling value");
}

// The planner uses the `--flag=value` spelling, so the parser takes both.
void TestEqualsSpelling() {
  int seq = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddInt("--seq", &seq);
  ExpectStatus(Parse(&flags, {"--seq=32"}), -1, "equals spelling");
  ExpectInt(seq, 32, "equals spelling value");
}

// A negative number is a value, not the next flag.
void TestNegativeValue() {
  int rank = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddInt("--rank", &rank);
  ExpectStatus(Parse(&flags, {"--rank", "-1"}), -1, "negative value");
  ExpectInt(rank, -1, "negative value");
}

void TestStringAndU64() {
  std::string path;
  std::uint64_t seed = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddString("--log", &path);
  flags.AddU64("--seed", &seed);
  ExpectStatus(Parse(&flags, {"--log", "run.log", "--seed", "42"}), -1,
               "string and u64");
  ExpectTrue(path == "run.log", "string value");
  ExpectTrue(seed == 42, "u64 value");
}

// `--input` is a second spelling of `--in`, which two commands accept.
void TestAlias() {
  std::string input;
  nanochat::cli::FlagParser flags("test");
  flags.AddString("--in", &input);
  flags.AddAlias("--input", "--in");
  ExpectStatus(Parse(&flags, {"--input", "fixture.bin"}), -1, "alias");
  ExpectTrue(input == "fixture.bin", "alias value");
  ExpectTrue(flags.Saw("--in"), "alias records the canonical name");
}

// The ten model flags come from `ApplyModelFlag`, so they bind the same way in
// every command that declares them.
void TestModelFlags() {
  nanochat::Config config;
  nanochat::cli::FlagParser flags("test");
  flags.AddModelFlags(&config);
  ExpectStatus(
      Parse(&flags, {"--layers", "4", "--hidden", "64", "--window-pattern",
                     "SSL", "--value-embedding", "1"}),
      -1, "model flags");
  ExpectInt(config.num_layers, 4, "model flag --layers");
  ExpectInt(config.hidden_dim, 64, "model flag --hidden");
  ExpectTrue(config.window_pattern == "SSL", "model flag --window-pattern");
  ExpectTrue(config.value_embedding, "model flag --value-embedding");
  ExpectTrue(flags.Saw("--layers"), "model flag is recorded as seen");
}

// A bare switch takes no value, and `--reference-copies` clears a flag.
void TestSwitches() {
  bool json = false;
  bool transposed = true;
  nanochat::cli::FlagParser flags("test");
  flags.AddSwitch("--json", &json);
  flags.AddSwitchOff("--reference-copies", &transposed);
  ExpectStatus(Parse(&flags, {"--json", "--reference-copies"}), -1, "switches");
  ExpectTrue(json, "--json sets the flag");
  ExpectTrue(!transposed, "--reference-copies clears the flag");
}

// A switch rejects an attached value instead of ignoring it.
void TestSwitchRejectsValue() {
  bool json = false;
  nanochat::cli::FlagParser flags("test");
  flags.AddSwitch("--json", &json);
  ExpectStatus(Parse(&flags, {"--json=1"}), 2, "switch with a value");
}

// A custom handler runs at its position, so the command line order wins.
void TestOrderedHandler() {
  int value = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddInt("--value", &value);
  flags.AddValue("--reset", [&value](std::string_view text) {
    value = nanochat::cli::ParseInt(text, &value) ? value : -1;
    return true;
  });
  // The reset comes last, so it wins.
  ExpectStatus(Parse(&flags, {"--value", "5", "--reset", "100"}), -1,
               "ordered");
  ExpectInt(value, 100, "later flag wins");
  // The reset comes first, so the later bind wins.
  ExpectStatus(Parse(&flags, {"--reset", "100", "--value", "5"}), -1,
               "ordered");
  ExpectInt(value, 5, "later bind wins");
}

// A handler that rejects its value stops the parse.
void TestHandlerRejects() {
  nanochat::cli::FlagParser flags("test");
  flags.AddValue("--hw-check", [](std::string_view value) {
    return std::string(value) == "auto" || std::string(value) == "off";
  });
  ExpectStatus(Parse(&flags, {"--hw-check", "off"}), -1, "accepted handler");
  ExpectStatus(Parse(&flags, {"--hw-check", "maybe"}), 2, "rejected handler");
}

// Every failure returns the process status the caller passes to `return`.
void TestFailures() {
  int batch = 0;
  nanochat::cli::FlagParser flags("test");
  flags.AddInt("--batch", &batch);
  ExpectStatus(Parse(&flags, {}), -1, "no arguments parses");
  ExpectStatus(Parse(&flags, {"--batch", "notanumber"}), 2, "bad value");
  ExpectStatus(Parse(&flags, {"--batch"}), 2, "missing value");
  ExpectStatus(Parse(&flags, {"--unknown", "1"}), 2, "unknown flag");
  // A bad value must not write a partial result into the target.
  batch = 9;
  Parse(&flags, {"--batch", "notanumber"});
  ExpectInt(batch, 9, "a failed parse leaves the target unchanged");
}

// `-h` and `--help` stop the parse and return 0.
void TestHelp() {
  nanochat::cli::FlagParser flags("test");
  ExpectStatus(Parse(&flags, {"-h"}), 0, "-h");
  ExpectStatus(Parse(&flags, {"--help"}), 0, "--help");
  ExpectStatus(Parse(&flags, {"--help", "--bad"}), 0, "help stops the parse");
}

}  // namespace

int main() {
  TestSpaceSpelling();
  TestEqualsSpelling();
  TestNegativeValue();
  TestStringAndU64();
  TestAlias();
  TestModelFlags();
  TestSwitches();
  TestSwitchRejectsValue();
  TestOrderedHandler();
  TestHandlerRejects();
  TestFailures();
  TestHelp();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("cli_test: the flag parser passes every group\n");
  return 0;
}
