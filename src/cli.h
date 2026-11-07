#ifndef NANOCHAT_SRC_CLI_H_
#define NANOCHAT_SRC_CLI_H_

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/config.h"

// Tiny, dependency-free help for the CLI entry points. It owns the shared
// `Config` flags, the comma-separated lists, and the one flag parser every
// subcommand of `nanochat_cli` registers its options with. A subcommand binds
// its flags in one place, beside the feature they configure, so the parsing
// rules cannot drift between commands (docs/build.md).
//
// Text is `std::string_view` throughout, because every name and value here is
// read-only for the length of one call. A parameter that owns nothing does not
// own a `std::string`, and a reference parameter would build a temporary at
// every string-literal call site. The three exceptions are a `char**` argument
// list, which `main` fixes, and `EnvInt` and `EnvString`, which pass their name
// to `std::getenv` and so need a NUL-terminated C string.

namespace nanochat {
namespace cli {

// Parses a whole token, so trailing text is an error. `std::from_chars` needs
// no NUL terminator, reports overflow instead of clamping, and ignores the
// locale, which `std::strtof` does not.
inline bool ParseInt(std::string_view text, int* out) {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

inline bool ParseFloat(std::string_view text, float* out) {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

inline bool ParseU64(std::string_view text, std::uint64_t* out) {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *out);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

inline std::vector<std::string> SplitCsv(std::string_view text) {
  std::vector<std::string> parts;
  std::string current;
  for (char c : text) {
    if (c == ',') {
      if (!current.empty()) parts.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

inline void AppendCsv(std::string_view text, std::vector<std::string>* out) {
  for (std::string& part : SplitCsv(text)) out->push_back(std::move(part));
}

// Reads an integer environment variable, returning `fallback` when it is
// unset or unparseable. The launcher sets the `NANOCHAT_*` variables in the
// style of `torchrun` (docs/distributed.md section 10, section 4).
//
// `name` stays a `const char*` because `std::getenv` takes a C string.
inline int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  int parsed = 0;
  if (value == nullptr || !ParseInt(value, &parsed)) return fallback;
  return parsed;
}

// Reads a string environment variable, returning `fallback` when it is unset.
// `name` stays a `const char*` for the same reason as in `EnvInt`.
inline std::string EnvString(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  return std::string(value);
}

// True when `flag` configures the shared model `Config`. Every field a stage
// command line can set lives here, so a run recorded from a command line can be
// reproduced from it.
inline bool IsModelFlag(std::string_view flag) {
  return flag == "--layers" || flag == "--heads" || flag == "--kv-heads" ||
         flag == "--hidden" || flag == "--seq" || flag == "--vocab" ||
         flag == "--padded-vocab" || flag == "--window-pattern" ||
         flag == "--rope-base" || flag == "--value-embedding";
}

// Applies one model flag. Returns false when the flag or its value is invalid.
inline bool ApplyModelFlag(std::string_view flag, std::string_view value,
                           Config* config) {
  if (config == nullptr) return false;
  if (flag == "--layers") return ParseInt(value, &config->num_layers);
  if (flag == "--heads") return ParseInt(value, &config->num_heads);
  if (flag == "--kv-heads") return ParseInt(value, &config->num_kv_heads);
  if (flag == "--hidden") return ParseInt(value, &config->hidden_dim);
  if (flag == "--seq") return ParseInt(value, &config->seq_len);
  if (flag == "--vocab") return ParseInt(value, &config->vocab_size);
  if (flag == "--padded-vocab") {
    return ParseInt(value, &config->padded_vocab_size);
  }
  if (flag == "--window-pattern") {
    config->window_pattern.assign(value);
    return true;
  }
  if (flag == "--rope-base") return ParseFloat(value, &config->rope_base);
  if (flag == "--value-embedding") {
    // A boolean field takes exactly `0` or `1`; any other word is an error, so
    // a stage cannot half-apply the field (docs/post-training.md B2).
    if (value == "1") {
      config->value_embedding = true;
      return true;
    }
    if (value == "0") {
      config->value_embedding = false;
      return true;
    }
    return false;
  }
  return false;
}

// Parses a subcommand's options into the fields the caller binds.
//
// One `nanochat_cli` binary runs many subcommands, so the parse rules live here
// once instead of in one loop per command. A command declares its flags with
// `Add*`, then calls `Parse`. Both `--flag value` and `--flag=value` are
// accepted, because the planner uses the second form.
//
// A flag that cannot be parsed is an error. The alternatives the old loops used
// were silent: a bad `--scalar-lr` became 0.0 and a bad `--batch` became 0, so
// a typo changed the run instead of stopping it. `Parse` stops the run.
class FlagParser {
 public:
  // `program` is the prefix of every diagnostic, for example "train_main".
  // It keeps the text a reader and a log regex already match.
  explicit FlagParser(std::string_view program);
  FlagParser(const FlagParser&) = delete;
  FlagParser& operator=(const FlagParser&) = delete;

  // The text `-h` prints, and that an error repeats. Kept in the command, so it
  // must outlive this parser; every caller passes a string literal.
  void SetUsage(std::string_view text);

  void AddString(std::string_view name, std::string* target);
  void AddInt(std::string_view name, int* target);
  void AddFloat(std::string_view name, float* target);
  void AddU64(std::string_view name, std::uint64_t* target);
  // A bare switch that sets the target to `true`.
  void AddSwitch(std::string_view name, bool* target);
  // A bare switch that sets the target to `false`.
  void AddSwitchOff(std::string_view name, bool* target);
  // A second spelling of a flag that is already registered. Register the
  // canonical flag first: the alias copies the kind, the target, and the
  // handler that the canonical flag bound.
  void AddAlias(std::string_view name, std::string_view canonical);
  // The ten model flags `ApplyModelFlag` handles, bound to `config`.
  void AddModelFlags(Config* config);
  // A flag that needs its own logic, such as a preset or an enumerated word.
  // It runs at the flag's position in the argument list, so later flags still
  // override an earlier one and the order on the command line is what wins.
  // Returns false when the value is invalid.
  void AddValue(std::string_view name,
                std::function<bool(std::string_view value)> handler);

  // Applies `argv[1..argc)`.
  //
  // Returns -1 when every flag was applied, so the caller continues. Otherwise
  // returns the process exit status: 0 after `-h`, and 2 after a bad flag.
  int Parse(int argc, char** argv);

  // True when `name` was applied, so a command can react to one flag after the
  // parse. `name` is the canonical spelling.
  bool Saw(std::string_view name) const;

 private:
  enum class Kind {
    kString,
    kInt,
    kFloat,
    kU64,
    kSwitch,
    kSwitchOff,
    kModel,
    kValue,
  };

  // One bound flag. The parser holds no ownership of any target. The two names
  // are owned strings, because `entries_` reallocates and a view into a short
  // string would dangle when the small-string buffer moves.
  struct Entry {
    std::string name;       // the spelling on the command line
    std::string canonical;  // the spelling `Saw` and `ApplyModelFlag` use
    Kind kind = Kind::kString;
    void* target = nullptr;
    std::function<bool(std::string_view value)> handler;  // for `kValue`
  };

  const Entry* Find(std::string_view name) const;
  void Bind(std::string_view name, Kind kind, void* target);
  bool Apply(const Entry& entry, std::string_view value);
  bool NeedsValue(Kind kind) const;

  std::string program_;
  // Points at the command's own literal, which outlives this parser.
  std::string_view usage_;
  std::vector<Entry> entries_;
  // Owned, for the same reallocation reason as `Entry::name`.
  std::vector<std::string> seen_;
};

}  // namespace cli
}  // namespace nanochat

#endif  // NANOCHAT_SRC_CLI_H_
