#ifndef NANOCHAT_SRC_SPLIT_PATTERN_H_
#define NANOCHAT_SRC_SPLIT_PATTERN_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// The fixed nanochat split pattern (docs/tokenizer.md section 2.5). The
// scanner is hand written (decision D1). It supports the one nanochat pattern
// only, so it has no third-party dependency.

namespace nanochat {

// The fixed nanochat pattern as a UTF-8 string. The loader compares the
// pattern in the artifact with this string.
std::string NanochatSplitPattern();

// Splits `text` with the fixed pattern. Every returned view points into
// `text`. The scanner tries the seven alternatives in order and takes the
// first match.
std::vector<std::string_view> SplitPattern(std::string_view text);

// The three Unicode predicates of the pattern: letter (`\p{L}`), number
// (`\p{N}`), and whitespace (`\s`). The tables come from Unicode 16.0.0
// (decision D2). A lookup uses a binary search.
bool IsUnicodeLetter(std::uint32_t code_point);
bool IsUnicodeNumber(std::uint32_t code_point);
bool IsUnicodeWhitespace(std::uint32_t code_point);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_SPLIT_PATTERN_H_
