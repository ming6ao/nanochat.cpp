// src/tokenizer/utf8_test.cc -- the UTF-8 code-point helpers.
//
// `Utf8CodePoints` and `Utf8Prefix` mirror Python's `len(str)` and `str[:n]`.
// The trainer uses them for `--doc-cap` and `--max-chars`, so the native crop
// matches the reference `tok_train.py` on non-ASCII text.

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

#include "src/tokenizer/utf8.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectCount(std::string_view text, std::size_t expected) {
  const std::size_t actual = nanochat::Utf8CodePoints(text);
  if (actual != expected) {
    Fail("Utf8CodePoints(" + std::string(text) + ") = " +
         std::to_string(actual) + ", want " + std::to_string(expected));
  }
}

void ExpectPrefix(std::string_view text, std::size_t cap,
                  std::string_view expected) {
  const std::string_view actual = nanochat::Utf8Prefix(text, cap);
  if (actual != expected) {
    Fail("Utf8Prefix(" + std::string(text) + ", " + std::to_string(cap) +
         ") = " + std::string(actual) + ", want " + std::string(expected));
  }
}

}  // namespace

int main() {
  // One byte per code point.
  ExpectCount("", 0);
  ExpectCount("abc", 3);
  ExpectCount("\x7F", 1);

  // Multi-byte code points: two, three, and four bytes.
  ExpectCount("a\xC3\xA9", 2);         // a, e-acute
  ExpectCount("\xE4\xB8\xAD", 1);      // CJK
  ExpectCount("\xF0\x9F\x8C\x8D", 1);  // globe emoji
  ExpectCount("a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x8C\x8D", 4);

  // Prefixes keep whole code points.
  const std::string_view mixed = "a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x8C\x8D";
  ExpectPrefix(mixed, 0, "");
  ExpectPrefix(mixed, 1, "a");
  ExpectPrefix(mixed, 2, "a\xC3\xA9");
  ExpectPrefix(mixed, 3, "a\xC3\xA9\xE4\xB8\xAD");
  ExpectPrefix(mixed, 4, mixed);
  ExpectPrefix(mixed, 99, mixed);

  // An incomplete sequence at the end counts as one code point and stays whole.
  ExpectCount("\xE4\xB8", 1);
  ExpectPrefix("\xE4\xB8", 1, "\xE4\xB8");

  if (g_failures != 0) {
    std::fprintf(stderr, "utf8_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("utf8_test: ok\n");
  return 0;
}
