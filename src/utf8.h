#ifndef NANOCHAT_SRC_UTF8_H_
#define NANOCHAT_SRC_UTF8_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// UTF-8 helpers for the native tokenizer. The splitter decodes one code point
// at a time, and the decoder copies a byte string with the lossy `replace`
// mode (docs/tokenizer.md sections 6 and 8.1).

namespace nanochat {

enum class Utf8Status {
  kComplete,    // A valid code point; `length` bytes.
  kIncomplete,  // A valid prefix at the end of the input; `length` bytes.
  kInvalid,     // An ill-formed sequence; `length` leading bytes.
};

struct Utf8Sequence {
  Utf8Status status = Utf8Status::kInvalid;
  std::size_t length = 0;
  std::uint32_t value = 0;
};

// Reads the sequence at `text[start]`. A `kInvalid` result consumes the
// longest valid prefix, so the caller continues at the ill-formed byte.
Utf8Sequence ReadUtf8(std::string_view text, std::size_t start);

// Appends the text with the lossy `replace` mode. Every ill-formed sequence
// becomes one U+FFFD replacement character.
void AppendUtf8Replaced(std::string_view text, std::string* out);

// Returns the text with the lossy `replace` mode.
std::string Utf8Replaced(std::string_view text);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_UTF8_H_
