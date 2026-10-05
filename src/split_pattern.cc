#include "src/split_pattern.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/unicode_tables.inc"
#include "src/utf8.h"

namespace nanochat {
namespace {

// Binary search over one ascending, inclusive range table.
bool InRanges(const unicode::Range* ranges, std::size_t count,
              std::uint32_t code_point) {
  std::size_t low = 0;
  std::size_t high = count;
  while (low < high) {
    const std::size_t mid = low + (high - low) / 2;
    if (code_point < ranges[mid].first) {
      high = mid;
    } else if (code_point > ranges[mid].last) {
      low = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

// Reads one code point. An ill-formed sequence reads as one byte, so the
// scanner always makes progress.
std::uint32_t CodepointAt(std::string_view text, std::size_t position,
                          std::size_t* length) {
  const Utf8Sequence sequence = ReadUtf8(text, position);
  if (sequence.status == Utf8Status::kComplete) {
    *length = sequence.length;
    return sequence.value;
  }
  *length = 1;
  return static_cast<std::uint32_t>(static_cast<unsigned char>(text[position]));
}

char AsciiLower(char value) {
  if (value >= 'A' && value <= 'Z') {
    return static_cast<char>(value - 'A' + 'a');
  }
  return value;
}

// Alternative 1: an apostrophe plus a contraction. The group is
// `(?i:[sdmt]|ll|ve|re)`.
std::size_t MatchContraction(std::string_view text, std::size_t start) {
  if (text[start] != '\'') return start;
  if (start + 1 >= text.size()) return start;
  const char first = AsciiLower(text[start + 1]);
  if (first == 's' || first == 'd' || first == 'm' || first == 't') {
    return start + 2;
  }
  if (start + 2 >= text.size()) return start;
  const char second = AsciiLower(text[start + 2]);
  if ((first == 'l' && second == 'l') || (first == 'v' && second == 'e') ||
      (first == 'r' && second == 'e')) {
    return start + 3;
  }
  return start;
}

// Alternative 2: `[^\r\n\p{L}\p{N}]?+\p{L}+`. The optional symbol uses a
// possessive quantifier, so it never gives back its match.
std::size_t MatchOptionalSymbolLetters(std::string_view text,
                                       std::size_t start) {
  std::size_t position = start;
  if (position < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, position, &length);
    if (code_point != '\r' && code_point != '\n' &&
        !IsUnicodeLetter(code_point) && !IsUnicodeNumber(code_point)) {
      position += length;
    }
  }
  std::size_t letters = 0;
  while (position < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, position, &length);
    if (!IsUnicodeLetter(code_point)) break;
    position += length;
    ++letters;
  }
  return letters > 0 ? position : start;
}

// Alternative 3: `\p{N}{1,2}`. One or two numbers, greedy.
std::size_t MatchOneOrTwoNumbers(std::string_view text, std::size_t start) {
  std::size_t position = start;
  int count = 0;
  while (count < 2 && position < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, position, &length);
    if (!IsUnicodeNumber(code_point)) break;
    position += length;
    ++count;
  }
  return count > 0 ? position : start;
}

// Alternative 4: ` ?[^\s\p{L}\p{N}]++[\r\n]*`. The symbol run uses a
// possessive quantifier.
std::size_t MatchSpaceSymbolsLineEnds(std::string_view text,
                                      std::size_t start) {
  std::size_t position = start;
  if (position < text.size() && text[position] == ' ') {
    ++position;
  }
  std::size_t symbols = 0;
  while (position < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, position, &length);
    if (IsUnicodeWhitespace(code_point) || IsUnicodeLetter(code_point) ||
        IsUnicodeNumber(code_point)) {
      break;
    }
    position += length;
    ++symbols;
  }
  if (symbols == 0) return start;
  while (position < text.size() &&
         (text[position] == '\r' || text[position] == '\n')) {
    ++position;
  }
  return position;
}

// Alternative 5: `\s*[\r\n]`. The greedy `\s*` gives back code points until a
// line end matches. A line end is one byte, so a byte scan finds it.
std::size_t MatchWhitespaceLineEnd(std::string_view text, std::size_t start) {
  std::size_t run_end = start;
  while (run_end < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, run_end, &length);
    if (!IsUnicodeWhitespace(code_point)) break;
    run_end += length;
  }
  for (std::size_t position = run_end; position > start; --position) {
    const char value = text[position - 1];
    if (value == '\r' || value == '\n') return position;
  }
  return start;
}

// Alternative 6: `\s+(?!\S)`. The greedy `\s+` gives back one code point so
// that the lookahead holds. At the end of the text the lookahead holds with no
// give-back.
std::size_t MatchTrailingWhitespace(std::string_view text, std::size_t start) {
  std::size_t run_end = start;
  std::size_t last_start = start;
  std::size_t count = 0;
  while (run_end < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, run_end, &length);
    if (!IsUnicodeWhitespace(code_point)) break;
    last_start = run_end;
    run_end += length;
    ++count;
  }
  if (count == 0) return start;
  if (run_end == text.size()) return run_end;
  if (count >= 2) return last_start;
  return start;
}

// Alternative 7: `\s+`. A whitespace run, greedy.
std::size_t MatchWhitespaceRun(std::string_view text, std::size_t start) {
  std::size_t position = start;
  while (position < text.size()) {
    std::size_t length = 0;
    const std::uint32_t code_point = CodepointAt(text, position, &length);
    if (!IsUnicodeWhitespace(code_point)) break;
    position += length;
  }
  return position;
}

// Tries the seven alternatives in order and takes the first match. The method
// returns `start` when no alternative matches.
std::size_t MatchAt(std::string_view text, std::size_t start) {
  std::size_t end = MatchContraction(text, start);
  if (end != start) return end;
  end = MatchOptionalSymbolLetters(text, start);
  if (end != start) return end;
  end = MatchOneOrTwoNumbers(text, start);
  if (end != start) return end;
  end = MatchSpaceSymbolsLineEnds(text, start);
  if (end != start) return end;
  end = MatchWhitespaceLineEnd(text, start);
  if (end != start) return end;
  end = MatchTrailingWhitespace(text, start);
  if (end != start) return end;
  return MatchWhitespaceRun(text, start);
}

}  // namespace

std::string NanochatSplitPattern() {
  return "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|"
         "\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|"
         "\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
}

std::vector<std::string_view> SplitPattern(std::string_view text) {
  std::vector<std::string_view> pieces;
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t end = MatchAt(text, start);
    if (end <= start) {
      // The pattern matches every code point, so this path is defensive only.
      std::size_t length = 1;
      CodepointAt(text, start, &length);
      end = start + length;
    }
    pieces.emplace_back(text.substr(start, end - start));
    start = end;
  }
  return pieces;
}

bool IsUnicodeLetter(std::uint32_t code_point) {
  return InRanges(unicode::kLetterRanges, unicode::kLetterRangeCount,
                  code_point);
}

bool IsUnicodeNumber(std::uint32_t code_point) {
  return InRanges(unicode::kNumberRanges, unicode::kNumberRangeCount,
                  code_point);
}

bool IsUnicodeWhitespace(std::uint32_t code_point) {
  return InRanges(unicode::kWhitespaceRanges, unicode::kWhitespaceRangeCount,
                  code_point);
}

}  // namespace nanochat
