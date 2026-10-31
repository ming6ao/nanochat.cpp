#ifndef NANOCHAT_SRC_JSON_H_
#define NANOCHAT_SRC_JSON_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>

// A small, strict JSON reader: the standard library only, no network, no
// schema. It exists so the two files this layer reads as data - a model's
// `spec.json` and a backend's `caps.json` - are read by one parser instead of
// two hand-rolled scanners, and so a malformed file is rejected with a position
// instead of being half-understood.
//
// Strictness, because both callers are closed field sets
// (`spec/versions.md` section 3.1):
//
//   * a duplicated key is an error, not a last-one-wins;
//   * a trailing comma, a comment, an unquoted key, and a bare value are errors;
//   * the nesting depth is bounded (`kMaxDepth`);
//   * every value is one of the six JSON types, and a caller that asks for the
//     wrong type gets null back rather than a coercion.
//
// Numbers are kept as `double`. An integer field is read with `Integer()`,
// which rejects a fractional value instead of truncating it.
//
// This header is private to `src/`: it is not part of the public contract and
// nothing above this layer may depend on it.

namespace nanochat {
namespace json {

enum class Kind { kNull = 0, kBool = 1, kNumber = 2, kString = 3,
                  kArray = 4, kObject = 5 };

// The largest accepted nesting depth. A spec and a caps file are shallow; the
// bound keeps a hostile file from exhausting the stack.
inline constexpr int kMaxDepth = 32;

class Value {
 public:
  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::kNull; }
  bool is_bool() const { return kind_ == Kind::kBool; }
  bool is_number() const { return kind_ == Kind::kNumber; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_array() const { return kind_ == Kind::kArray; }
  bool is_object() const { return kind_ == Kind::kObject; }

  bool boolean() const { return bool_; }
  double number() const { return number_; }
  const std::string& string() const { return string_; }
  const std::vector<Value>& array() const { return array_; }
  const std::map<std::string, Value>& object() const { return object_; }

  // The value of `key`, or null when this is not an object or the key is
  // absent. An absent key and an explicit JSON `null` are the same answer here,
  // so a caller that needs "present but null" checks `Has()` first.
  const Value* Find(const std::string& key) const;
  bool Has(const std::string& key) const { return Find(key) != nullptr; }

  // The number `value` when this is an integral number that fits in `int`.
  // Returns false otherwise, so a caller never silently truncates 1.5 to 1.
  bool Integer(int* out) const;

  // A human-readable name of the kind, for error messages.
  const char* KindName() const;

  // The payload. It is public because the parser in `json.cc` is the only
  // writer and it is a separate translation unit; every reader goes through the
  // accessors above, which are the ones that check the kind.
  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Value> array_;
  std::map<std::string, Value> object_;
};

// Parses `text` into `*out`. Returns false and writes a one-line reason, with
// the byte offset, to `*error` when that is not null. `*out` is only written on
// success.
bool Parse(const std::string& text, Value* out, std::string* error = nullptr);

// Reads `path` and parses it. A missing or unreadable file is an error, and the
// message names the path.
bool ParseFile(const std::string& path, Value* out,
               std::string* error = nullptr);

}  // namespace json
}  // namespace nanochat

#endif  // NANOCHAT_SRC_JSON_H_
