#include "src/tokenizer/utf8.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace nanochat {
namespace {

constexpr char kReplacement[] = "\xEF\xBF\xBD";

// Returns the expected byte count for a valid lead byte.
std::size_t ExpectedLength(std::uint8_t lead) {
  if (lead <= 0xDF) return 2;
  if (lead <= 0xEF) return 3;
  return 4;
}

// Returns the valid second-byte range for a three- or four-byte lead.
void SecondByteRange(std::uint8_t lead, std::uint8_t* low, std::uint8_t* high) {
  *low = 0x80;
  *high = 0xBF;
  if (lead == 0xE0) {
    *low = 0xA0;
  } else if (lead == 0xED) {
    *high = 0x9F;
  } else if (lead == 0xF0) {
    *low = 0x90;
  } else if (lead == 0xF4) {
    *high = 0x8F;
  }
}

std::uint8_t ByteAt(std::string_view text, std::size_t position) {
  return static_cast<std::uint8_t>(static_cast<unsigned char>(text[position]));
}

std::uint32_t DecodeValue(std::string_view text, std::size_t start,
                          std::size_t length) {
  std::uint32_t value =
      static_cast<std::uint32_t>(ByteAt(text, start) & (0xFFu >> (length + 1)));
  for (std::size_t index = 1; index < length; ++index) {
    value = (value << 6) | (ByteAt(text, start + index) & 0x3Fu);
  }
  return value;
}

}  // namespace

Utf8Sequence ReadUtf8(std::string_view text, std::size_t start) {
  Utf8Sequence result;
  if (start >= text.size()) {
    result.status = Utf8Status::kIncomplete;
    result.length = 0;
    return result;
  }
  const std::uint8_t first = ByteAt(text, start);
  if (first < 0x80) {
    result.status = Utf8Status::kComplete;
    result.length = 1;
    result.value = first;
    return result;
  }
  if (first < 0xC2 || first > 0xF4) {
    result.length = 1;  // Invalid lead byte. Consume one byte.
    return result;
  }
  const std::size_t expected = ExpectedLength(first);
  const std::size_t available = text.size() - start;
  std::uint8_t low = 0x80;
  std::uint8_t high = 0xBF;
  if (expected >= 3) {
    SecondByteRange(first, &low, &high);
  }
  if (available < 2) {
    result.status = Utf8Status::kIncomplete;
    result.length = available;
    return result;
  }
  const std::uint8_t second = ByteAt(text, start + 1);
  if (second < low || second > high) {
    result.length = 1;  // Invalid second byte. Consume the lead byte only.
    return result;
  }
  for (std::size_t index = 2; index < expected; ++index) {
    if (available <= index) {
      result.status = Utf8Status::kIncomplete;
      result.length = available;
      return result;
    }
    const std::uint8_t byte = ByteAt(text, start + index);
    if (byte < 0x80 || byte > 0xBF) {
      result.status = Utf8Status::kInvalid;
      result.length = index;
      return result;
    }
  }
  result.status = Utf8Status::kComplete;
  result.length = expected;
  result.value = DecodeValue(text, start, expected);
  return result;
}

void AppendUtf8Replaced(std::string_view text, std::string* out) {
  std::size_t position = 0;
  while (position < text.size()) {
    const Utf8Sequence sequence = ReadUtf8(text, position);
    if (sequence.status == Utf8Status::kComplete) {
      out->append(text.data() + position, sequence.length);
    } else {
      out->append(kReplacement, sizeof(kReplacement) - 1);
    }
    position += sequence.length;
  }
}

std::string Utf8Replaced(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  AppendUtf8Replaced(text, &out);
  return out;
}

}  // namespace nanochat
