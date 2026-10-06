// src/tokenizer/split_pattern_test.cc -- the fixed nanochat splitter.
//
// Checks the curated split cases from docs/tokenizer.md section 6, the
// possessive and lookahead edge cases, the Unicode table boundaries, and a
// deterministic random corpus.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/unicode_tables.inc"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Join(const std::vector<std::string_view>& pieces) {
  std::string joined;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    if (i != 0) joined += ", ";
    joined += std::string(pieces[i]);
  }
  return joined;
}

void ExpectPieces(const std::string& input,
                  const std::vector<std::string>& expected) {
  const std::vector<std::string_view> pieces = nanochat::SplitPattern(input);
  if (pieces.size() != expected.size()) {
    Fail("split of '" + input + "' gave [" + Join(pieces) + "], expected " +
         std::to_string(expected.size()) + " pieces");
    return;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (pieces[i] != expected[i]) {
      Fail("split of '" + input + "' piece " + std::to_string(i) + " is '" +
           std::string(pieces[i]) + "', expected '" + expected[i] + "'");
      return;
    }
  }
}

void TestPatternText() {
  const std::string expected =
      "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|"
      "\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|"
      "\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
  if (nanochat::NanochatSplitPattern() != expected) {
    Fail("NanochatSplitPattern does not match the frozen pattern");
  }
}

void TestCuratedSplits() {
  // The possessive edge case from docs/tokenizer.md section 6.
  ExpectPieces(" !abc", {" !", "abc"});
  ExpectPieces("", {});
  // Alternative 1: apostrophe plus contraction.
  ExpectPieces("don't", {"don", "'t"});
  ExpectPieces("I'll", {"I", "'ll"});
  ExpectPieces("we've", {"we", "'ve"});
  ExpectPieces("they're", {"they", "'re"});
  ExpectPieces("he's", {"he", "'s"});
  ExpectPieces("'l", {"'l"});
  // Alternative 2: optional symbol, then letters.
  ExpectPieces("@user", {"@user"});
  ExpectPieces("cafe\xCC\x81", {"cafe", "\xCC\x81"});
  ExpectPieces(
      "\xE2\x80\x83"
      "a",
      {"\xE2\x80\x83"
       "a"});
  ExpectPieces(
      "a\xE2\x80\x83"
      "b",
      {"a",
       "\xE2\x80\x83"
       "b"});
  // Alternative 3: one or two numbers.
  ExpectPieces("1234", {"12", "34"});
  ExpectPieces("3.14", {"3", ".", "14"});
  ExpectPieces("123 456", {"12", "3", " ", "45", "6"});
  // Alternative 4: optional space, symbols, and line ends.
  ExpectPieces("Hello, World!", {"Hello", ",", " World", "!"});
  ExpectPieces("!!!\n", {"!!!\n"});
  // Alternative 5: whitespace, then a line end.
  ExpectPieces("\n", {"\n"});
  ExpectPieces("\r\n", {"\r\n"});
  ExpectPieces("  \n", {"  \n"});
  ExpectPieces("a\nb", {"a", "\n", "b"});
  ExpectPieces("one\r\ntwo", {"one", "\r\n", "two"});
  // Alternative 6: trailing whitespace, with the negative lookahead.
  ExpectPieces("word ", {"word", " "});
  ExpectPieces("a  ", {"a", "  "});
  ExpectPieces("a   b", {"a", "  ", " b"});
  // Alternative 7: a whitespace run.
  ExpectPieces("   ", {"   "});
  ExpectPieces("  \t  ", {"  \t  "});
  // Multi-byte letters and whitespace.
  ExpectPieces("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
               {"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"});
  ExpectPieces("\xC2\xA0", {"\xC2\xA0"});
}

void TestUnicodeTables() {
  const auto check = [](const nanochat::unicode::Range* ranges,
                        std::size_t count, bool (*predicate)(std::uint32_t),
                        const char* name) {
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t first = ranges[i].first;
      const std::uint32_t last = ranges[i].last;
      if (first > last) {
        Fail(std::string(name) + " range is empty or inverted");
        return;
      }
      if (!predicate(first) || !predicate(last)) {
        Fail(std::string(name) + " range endpoints are not classified");
        return;
      }
      if (first > 0 && predicate(first - 1)) {
        Fail(std::string(name) + " classifies the value below a range");
        return;
      }
      if (last < 0x10FFFFu && predicate(last + 1)) {
        Fail(std::string(name) + " classifies the value above a range");
        return;
      }
    }
  };
  check(nanochat::unicode::kLetterRanges, nanochat::unicode::kLetterRangeCount,
        nanochat::IsUnicodeLetter, "letter");
  check(nanochat::unicode::kNumberRanges, nanochat::unicode::kNumberRangeCount,
        nanochat::IsUnicodeNumber, "number");
  check(nanochat::unicode::kWhitespaceRanges,
        nanochat::unicode::kWhitespaceRangeCount, nanochat::IsUnicodeWhitespace,
        "whitespace");

  if (!nanochat::IsUnicodeLetter('A') || !nanochat::IsUnicodeLetter(0x65E5) ||
      nanochat::IsUnicodeLetter('1') || nanochat::IsUnicodeLetter(' ')) {
    Fail("letter predicate disagrees with known code points");
  }
  if (!nanochat::IsUnicodeNumber('7') || !nanochat::IsUnicodeNumber(0x0660) ||
      nanochat::IsUnicodeNumber('a')) {
    Fail("number predicate disagrees with known code points");
  }
  if (!nanochat::IsUnicodeWhitespace(' ') ||
      !nanochat::IsUnicodeWhitespace('\t') ||
      !nanochat::IsUnicodeWhitespace('\n') ||
      !nanochat::IsUnicodeWhitespace(0x00A0) ||
      nanochat::IsUnicodeWhitespace('a')) {
    Fail("whitespace predicate disagrees with known code points");
  }
}

void TestRandomCorpus() {
  const std::vector<std::string> atoms = {
      "a",        "Z",        "0",        "7",           "'",
      "!",        "?",        ".",        ",",           " ",
      "\t",       "\n",       "\r",       "\xC3\xA9",    "\xE6\x97\xA5",
      "\xCE\xB1", "\xD0\x96", "\xC2\xA0", "\xE2\x80\x83"};
  std::mt19937 generator(20240517u);
  std::uniform_int_distribution<int> length_distribution(0, 12);
  std::uniform_int_distribution<std::size_t> atom_distribution(
      0, atoms.size() - 1);
  for (int trial = 0; trial < 20000; ++trial) {
    const int length = length_distribution(generator);
    std::string text;
    for (int index = 0; index < length; ++index) {
      text += atoms[atom_distribution(generator)];
    }
    const std::vector<std::string_view> pieces = nanochat::SplitPattern(text);
    std::string rebuilt;
    for (std::string_view piece : pieces) {
      if (piece.empty()) {
        Fail("random split produced an empty piece");
        return;
      }
      rebuilt += piece;
      const std::vector<std::string_view> again = nanochat::SplitPattern(piece);
      if (again.size() != 1 || again[0] != piece) {
        Fail("random split is not idempotent for '" + text + "'");
        return;
      }
    }
    if (rebuilt != text) {
      Fail("random split does not rebuild '" + text + "'");
      return;
    }
  }
}

}  // namespace

int main() {
  TestPatternText();
  TestCuratedSplits();
  TestUnicodeTables();
  TestRandomCorpus();
  if (g_failures != 0) {
    std::fprintf(stderr, "split_pattern_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("split_pattern_test: ok\n");
  return 0;
}
