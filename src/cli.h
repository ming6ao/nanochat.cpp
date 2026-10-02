#ifndef NANOCHAT_SRC_CLI_H_
#define NANOCHAT_SRC_CLI_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "nanochat/config.h"

// Tiny, dependency-free help for the CLI entry points. It only knows how to
// parse the shared `Config` flags and comma-separated lists; each `*_main.cc`
// owns its own flag loop so the flags stay close to the feature they configure.

namespace nanochat {
namespace cli {

inline bool ParseInt(const char* text, int* out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *out = static_cast<int>(value);
  return true;
}

inline bool ParseFloat(const char* text, float* out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0') return false;
  *out = value;
  return true;
}

inline bool ParseU64(const char* text, std::uint64_t* out) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *out = static_cast<std::uint64_t>(value);
  return true;
}

inline std::vector<std::string> SplitCsv(const std::string& text) {
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

inline void AppendCsv(const std::string& text, std::vector<std::string>* out) {
  for (std::string& part : SplitCsv(text)) out->push_back(std::move(part));
}

// True when `flag` configures the shared model `Config`.
inline bool IsModelFlag(const std::string& flag) {
  return flag == "--layers" || flag == "--heads" || flag == "--kv-heads" ||
         flag == "--hidden" || flag == "--seq" || flag == "--vocab" ||
         flag == "--padded-vocab" || flag == "--window-pattern" ||
         flag == "--rope-base";
}

// Applies one model flag. Returns false when the flag or its value is invalid.
inline bool ApplyModelFlag(const std::string& flag, const char* value,
                           Config* config) {
  if (config == nullptr || value == nullptr) return false;
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
    config->window_pattern = value;
    return true;
  }
  if (flag == "--rope-base") return ParseFloat(value, &config->rope_base);
  return false;
}

}  // namespace cli
}  // namespace nanochat

#endif  // NANOCHAT_SRC_CLI_H_
