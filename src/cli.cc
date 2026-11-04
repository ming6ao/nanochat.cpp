// The one flag parser of `nanochat_cli` (docs/build.md).
//
// A subcommand declares its options with `Add*` and calls `Parse` once. The
// rules that used to be copied into every entry point -- the two spellings of a
// flag, the help request, the missing value, the unknown flag -- live here.

#include "src/cli.h"

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

namespace nanochat {
namespace cli {
namespace {

constexpr int kContinue = -1;
constexpr int kHelp = 0;
constexpr int kBadFlag = 2;

// Splits `--name=value` into its two parts. `value_after_equals` is false when
// the token carries no `=`.
std::string_view NameOf(std::string_view token, bool* value_after_equals) {
  const std::size_t equals = token.find('=');
  if (equals == std::string_view::npos) {
    *value_after_equals = false;
    return token;
  }
  *value_after_equals = true;
  return token.substr(0, equals);
}

}  // namespace

FlagParser::FlagParser(std::string_view program) : program_(program) {}

void FlagParser::SetUsage(std::string_view text) { usage_ = text; }

void FlagParser::Bind(std::string_view name, Kind kind, void* target) {
  Entry entry;
  entry.name = name;
  entry.canonical = name;
  entry.kind = kind;
  entry.target = target;
  entries_.push_back(std::move(entry));
}

void FlagParser::AddString(std::string_view name, std::string* target) {
  Bind(name, Kind::kString, target);
}

void FlagParser::AddInt(std::string_view name, int* target) {
  Bind(name, Kind::kInt, target);
}

void FlagParser::AddFloat(std::string_view name, float* target) {
  Bind(name, Kind::kFloat, target);
}

void FlagParser::AddU64(std::string_view name, std::uint64_t* target) {
  Bind(name, Kind::kU64, target);
}

void FlagParser::AddSwitch(std::string_view name, bool* target) {
  Bind(name, Kind::kSwitch, target);
}

void FlagParser::AddSwitchOff(std::string_view name, bool* target) {
  Bind(name, Kind::kSwitchOff, target);
}

void FlagParser::AddAlias(std::string_view name, std::string_view canonical) {
  for (const Entry& entry : entries_) {
    if (entry.name != canonical) continue;
    Entry alias = entry;  // the kind, the target, and the handler
    alias.name = name;
    alias.canonical = canonical;
    entries_.push_back(std::move(alias));
    return;
  }
}

void FlagParser::AddModelFlags(Config* config) {
  static constexpr std::string_view kNames[] = {
      "--layers",    "--heads",           "--kv-heads",     "--hidden",
      "--seq",       "--vocab",           "--padded-vocab", "--window-pattern",
      "--rope-base", "--value-embedding",
  };
  for (std::string_view name : kNames) Bind(name, Kind::kModel, config);
}

void FlagParser::AddValue(std::string_view name,
                          std::function<bool(std::string_view value)> handler) {
  Entry entry;
  entry.name = name;
  entry.canonical = name;
  entry.kind = Kind::kValue;
  entry.handler = std::move(handler);
  entries_.push_back(std::move(entry));
}

const FlagParser::Entry* FlagParser::Find(std::string_view name) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

bool FlagParser::NeedsValue(Kind kind) const {
  return kind != Kind::kSwitch && kind != Kind::kSwitchOff;
}

bool FlagParser::Apply(const Entry& entry, std::string_view value) {
  switch (entry.kind) {
    case Kind::kSwitch:
      *static_cast<bool*>(entry.target) = true;
      return true;
    case Kind::kSwitchOff:
      *static_cast<bool*>(entry.target) = false;
      return true;
    case Kind::kString:
      static_cast<std::string*>(entry.target)->assign(value);
      return true;
    case Kind::kInt:
      return ParseInt(value, static_cast<int*>(entry.target));
    case Kind::kFloat:
      return ParseFloat(value, static_cast<float*>(entry.target));
    case Kind::kU64:
      return ParseU64(value, static_cast<std::uint64_t*>(entry.target));
    case Kind::kModel:
      return ApplyModelFlag(entry.canonical, value,
                            static_cast<Config*>(entry.target));
    case Kind::kValue:
      return entry.handler != nullptr && entry.handler(value);
  }
  return false;
}

bool FlagParser::Saw(std::string_view name) const {
  for (const std::string& seen : seen_) {
    if (seen == name) return true;
  }
  return false;
}

int FlagParser::Parse(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view token = argv[i] == nullptr ? "" : argv[i];
    if (token == "-h" || token == "--help") {
      if (!usage_.empty()) std::fwrite(usage_.data(), 1, usage_.size(), stderr);
      return kHelp;
    }

    bool has_inline_value = false;
    const std::string_view name = NameOf(token, &has_inline_value);
    const Entry* entry = Find(name);
    if (entry == nullptr) {
      std::fprintf(stderr, "%.*s: unknown flag %.*s\n",
                   static_cast<int>(program_.size()), program_.data(),
                   static_cast<int>(name.size()), name.data());
      if (!usage_.empty()) std::fwrite(usage_.data(), 1, usage_.size(), stderr);
      return kBadFlag;
    }

    // `--flag=value` carries its text on the token; a bare `--flag` takes the
    // next argument. A switch takes neither.
    std::string_view value;
    if (NeedsValue(entry->kind)) {
      if (has_inline_value) {
        value = token.substr(token.find('=') + 1);
      } else {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "%.*s: %.*s needs a value\n",
                       static_cast<int>(program_.size()), program_.data(),
                       static_cast<int>(name.size()), name.data());
          return kBadFlag;
        }
        value = argv[++i];
      }
    } else if (has_inline_value) {
      std::fprintf(stderr, "%.*s: %.*s takes no value\n",
                   static_cast<int>(program_.size()), program_.data(),
                   static_cast<int>(name.size()), name.data());
      return kBadFlag;
    }

    if (!Apply(*entry, value)) {
      std::fprintf(stderr, "%.*s: bad value for %.*s\n",
                   static_cast<int>(program_.size()), program_.data(),
                   static_cast<int>(name.size()), name.data());
      return kBadFlag;
    }
    seen_.push_back(entry->canonical);
  }
  return kContinue;
}

}  // namespace cli
}  // namespace nanochat
