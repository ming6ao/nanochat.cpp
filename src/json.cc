// The small strict JSON reader declared in src/json.h. See that header for the
// strictness rules; this file is the recursion, the position tracking, and the
// duplicate-key check.
//
// The parser is a hand-written recursive descent over the byte string. It is
// deliberately not incremental and not streaming: both callers read one small
// file into memory once.

#include "src/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace nanochat {
namespace json {
namespace {

std::string Offset(int offset, const std::string& what) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), " at byte %d", offset);
  return what + buffer;
}

class Reader {
 public:
  Reader(const std::string& text, std::string* error)
      : text_(text), error_(error) {}

  bool Read(Value* out) {
    SkipWhitespace();
    if (!ParseValue(out, 0)) return false;
    SkipWhitespace();
    if (pos_ != text_.size()) {
      Fail("unexpected trailing data");
      return false;
    }
    return true;
  }

 private:
  bool Fail(const std::string& what) {
    if (error_ != nullptr) *error_ = Offset(static_cast<int>(pos_), what);
    return false;
  }

  void SkipWhitespace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool Literal(const char* word) {
    const std::string expected(word);
    if (text_.compare(pos_, expected.size(), expected) != 0) {
      return Fail("expected " + expected);
    }
    pos_ += expected.size();
    return true;
  }

  bool ParseString(std::string* out) {
    if (pos_ >= text_.size() || text_[pos_] != '"') return Fail("expected a string");
    ++pos_;
    out->clear();
    while (true) {
      if (pos_ >= text_.size()) return Fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return true;
      if (static_cast<unsigned char>(c) < 0x20) {
        return Fail("unescaped control character in a string");
      }
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) return Fail("unterminated escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          // The two files this reader parses are ASCII, so a \\u escape is
          // accepted only when it decodes to ASCII. A non-ASCII code point is a
          // named error rather than a silently mangled string: nothing in a
          // spec or a caps file needs one, and a truncated UTF-8 encoder here
          // would be an invisible corruption.
          if (pos_ + 4 > text_.size()) return Fail("truncated \\u escape");
          int code = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_ + i];
            int digit = -1;
            if (h >= '0' && h <= '9') digit = h - '0';
            if (h >= 'a' && h <= 'f') digit = h - 'a' + 10;
            if (h >= 'A' && h <= 'F') digit = h - 'A' + 10;
            if (digit < 0) return Fail("malformed \\u escape");
            code = code * 16 + digit;
          }
          pos_ += 4;
          if (code == 0 || code > 0x7f) {
            return Fail("a \\u escape outside ASCII is not supported");
          }
          out->push_back(static_cast<char>(code));
          break;
        }
        default:
          return Fail("unknown escape");
      }
    }
  }

  bool ParseNumber(Value* out) {
    const std::size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
    bool digits = false;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
      ++pos_;
      digits = true;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
        digits = true;
      }
    }
    if (!digits) return Fail("expected a number");
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
        ++pos_;
      }
      bool exp_digits = false;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
        exp_digits = true;
      }
      if (!exp_digits) return Fail("expected a number after the exponent");
    }
    const std::string token = text_.substr(start, pos_ - start);
    const double value = std::strtod(token.c_str(), nullptr);
    if (!std::isfinite(value)) return Fail("number is not finite");
    out->kind_ = Kind::kNumber;
    out->number_ = value;
    return true;
  }

  bool ParseValue(Value* out, int depth) {
    if (depth > kMaxDepth) return Fail("nesting is too deep");
    SkipWhitespace();
    if (pos_ >= text_.size()) return Fail("unexpected end of input");
    const char c = text_[pos_];
    if (c == '{') return ParseObject(out, depth);
    if (c == '[') return ParseArray(out, depth);
    if (c == '"') {
      out->kind_ = Kind::kString;
      return ParseString(&out->string_);
    }
    if (c == 't') {
      if (!Literal("true")) return false;
      out->kind_ = Kind::kBool;
      out->bool_ = true;
      return true;
    }
    if (c == 'f') {
      if (!Literal("false")) return false;
      out->kind_ = Kind::kBool;
      out->bool_ = false;
      return true;
    }
    if (c == 'n') {
      if (!Literal("null")) return false;
      out->kind_ = Kind::kNull;
      return true;
    }
    return ParseNumber(out);
  }

  bool ParseArray(Value* out, int depth) {
    ++pos_;  // '['
    out->kind_ = Kind::kArray;
    SkipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      Value element;
      if (!ParseValue(&element, depth + 1)) return false;
      out->array_.push_back(std::move(element));
      SkipWhitespace();
      if (pos_ >= text_.size()) return Fail("unterminated array");
      if (text_[pos_] == ',') {
        ++pos_;
        SkipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == ']') {
          return Fail("trailing comma in an array");
        }
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return Fail("expected ',' or ']'");
    }
  }

  bool ParseObject(Value* out, int depth) {
    ++pos_;  // '{'
    out->kind_ = Kind::kObject;
    SkipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      SkipWhitespace();
      std::string key;
      if (!ParseString(&key)) return false;
      SkipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != ':') {
        return Fail("expected ':'");
      }
      ++pos_;
      Value element;
      if (!ParseValue(&element, depth + 1)) return false;
      // A duplicated key is an error: a closed field set has no "last one
      // wins", and a silent overwrite hides the one the author meant.
      if (out->object_.find(key) != out->object_.end()) {
        return Fail("duplicate key \"" + key + "\"");
      }
      out->object_.emplace(std::move(key), std::move(element));
      SkipWhitespace();
      if (pos_ >= text_.size()) return Fail("unterminated object");
      if (text_[pos_] == ',') {
        ++pos_;
        SkipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == '}') {
          return Fail("trailing comma in an object");
        }
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return Fail("expected ',' or '}'");
    }
  }

  const std::string& text_;
  std::string* error_;
  std::size_t pos_ = 0;
};

}  // namespace

const Value* Value::Find(const std::string& key) const {
  if (kind_ != Kind::kObject) return nullptr;
  const auto it = object_.find(key);
  return it == object_.end() ? nullptr : &it->second;
}

bool Value::Integer(int* out) const {
  if (kind_ != Kind::kNumber || out == nullptr) return false;
  if (!std::isfinite(number_)) return false;
  if (number_ != std::floor(number_)) return false;
  if (number_ < static_cast<double>(std::numeric_limits<int>::min()) ||
      number_ > static_cast<double>(std::numeric_limits<int>::max())) {
    return false;
  }
  *out = static_cast<int>(number_);
  return true;
}

const char* Value::KindName() const {
  switch (kind_) {
    case Kind::kNull: return "null";
    case Kind::kBool: return "bool";
    case Kind::kNumber: return "number";
    case Kind::kString: return "string";
    case Kind::kArray: return "array";
    case Kind::kObject: return "object";
  }
  return "unknown";
}

bool Parse(const std::string& text, Value* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) {
    if (error != nullptr) *error = "no output value";
    return false;
  }
  Value parsed;
  Reader reader(text, error);
  if (!reader.Read(&parsed)) return false;
  *out = std::move(parsed);
  return true;
}

bool ParseFile(const std::string& path, Value* out, std::string* error) {
  std::ifstream in(path);
  if (!in.good()) {
    if (error != nullptr) *error = "cannot read " + path;
    return false;
  }
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  std::string why;
  if (!Parse(text, out, &why)) {
    if (error != nullptr) *error = path + ": " + why;
    return false;
  }
  return true;
}

}  // namespace json
}  // namespace nanochat
