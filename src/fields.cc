// src/fields.cc -- the forward- and backward-compatible field visitor
// (docs/checkpoint-metadata-and-examples-plan.md stage A.1,
// docs/checkpoint-format.md section 2.1).
//
// This is the port of `gemma.cpp` `io/fields.cc`: the word stream, the reader,
// and the writer behind `nanochat/fields.h`. The header carries the contract
// and the format, this file carries the code, so a reviewer can diff the two
// files side by side.
//
// The one rule that is not in the upstream file: the block never returns a
// status. A bad value sets the visitor's sticky flag, a truncated block counts
// the fields it could not read, and the `IFields::ReadResult` counters reach
// the caller (design decisions D5, D6, and D7).

#include "nanochat/fields.h"

#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace nanochat {
namespace {

// Four characters ride in one word, so a string of `n` characters takes
// `DivCeil(n, kCharsPerWord)` words.
constexpr std::size_t kCharsPerWord = sizeof(std::uint32_t);

// The longest string the block carries, in characters: the word cap of
// `nanochat/fields.h` expressed in the unit the length word holds.
constexpr std::size_t kMaxStringChars = kMaxStringWords * kCharsPerWord;

// --- Writer ---------------------------------------------------------------

// Appends the fields of one block to a word stream. It never allocates a second
// buffer, so `IFields::AppendTo` can append block after block.
class FieldsWriter final : public IFieldsVisitor {
 public:
  explicit FieldsWriter(std::vector<std::uint32_t>* storage)
      : storage_(storage) {}

  void operator()(std::uint32_t& value) override { storage_->push_back(value); }

  void operator()(std::int32_t& value) override {
    // Two's complement: the word is the bit pattern, so a negative value keeps
    // its sign (docs/checkpoint-format.md section 5.2).
    storage_->push_back(std::bit_cast<std::uint32_t>(value));
  }

  void operator()(std::uint64_t& value) override {
    // Two words, the low word first and then bits 32-63 (open decision O6, and
    // the word stream comment in `nanochat/fields.h`). The order is part of the
    // format: a stream written the other way round would still decode.
    storage_->push_back(static_cast<std::uint32_t>(value & 0xffffffffu));
    storage_->push_back(static_cast<std::uint32_t>(value >> 32));
  }

  void operator()(float& value) override {
    // The bit pattern, not a conversion, so a write and a read are exact.
    if (!std::isfinite(value)) {
      return NotifyInvalid("Invalid float %f\n", static_cast<double>(value));
    }
    storage_->push_back(std::bit_cast<std::uint32_t>(value));
  }

  void operator()(std::string& value) override {
    if (SkipField()) return;
    if (value.size() > kMaxStringChars) {
      return NotifyInvalid("String too long %zu\n", value.size());
    }
    // The length word is the character count, so the reader knows where the
    // value ends and can ignore the padding (rule 3 in
    // `nanochat/checkpoint_metadata.h` caps it at `kMaxStringWords` words).
    storage_->push_back(static_cast<std::uint32_t>(value.size()));
    const std::size_t start = storage_->size();
    storage_->resize(start + DivCeil(value.size(), kCharsPerWord), 0);
    for (std::size_t i = 0; i < value.size(); ++i) {
      const std::uint32_t byte =
          static_cast<std::uint32_t>(static_cast<unsigned char>(value[i]));
      const std::size_t shift = 8 * (i % kCharsPerWord);
      (*storage_)[start + i / kCharsPerWord] |= byte << shift;
    }
  }

  void operator()(IFields& fields) override {
    if (SkipField()) return;
    // A nested block carries its own size, which does not count the size word.
    // The placeholder is patched once the payload is known, so the reader can
    // skip a block it does not know.
    const std::size_t start = storage_->size();
    storage_->push_back(0);
    fields.VisitFields(*this);
    if (AnyInvalid()) return;
    (*storage_)[start] =
        static_cast<std::uint32_t>(storage_->size() - start - 1);
  }

 private:
  std::vector<std::uint32_t>* storage_;
};

// --- Reader ---------------------------------------------------------------

// Reads the fields of one block from a word stream. `end_` is the end of the
// block being visited: the end of the span at the top level, and the end the
// nested block declared below it. A field that starts at or past `end_` is
// missing, which is the "new code, old data" direction and is not an error.
class FieldsReader final : public IFieldsVisitor {
 public:
  FieldsReader(SerializedSpan data, std::size_t pos)
      : data_(data), pos_(pos), end_(data.size()) {}

  IFields::ReadResult Result() const {
    IFields::ReadResult result(AnyInvalid() ? 0 : pos_);
    result.missing_fields = missing_;
    if (AnyInvalid()) return result;
    // Words the block carried beyond what `VisitFields` read, either inside a
    // nested block or after the last field: the data is newer than the code.
    // A read that ran out of data read everything there was, so the two
    // counters are never both nonzero.
    result.extra_u32 = extra_ + static_cast<std::uint32_t>(data_.size() - pos_);
    return result;
  }

  void operator()(std::uint32_t& value) override {
    // A field that failed earlier rejects the whole block, so no member after it
    // takes a value from a stream the read has already abandoned (rule 5 of
    // `nanochat/checkpoint_metadata.h`: no partial load). This is what makes the
    // `Recipe` block of design decision D7 all-or-nothing: one bad rate and the
    // caller keeps every member initializer instead of half a recipe.
    if (AnyInvalid()) return;
    std::uint32_t word = 0;
    if (!NextWord(&word)) return;
    value = word;
  }

  void operator()(std::int32_t& value) override {
    if (AnyInvalid()) return;
    std::uint32_t word = 0;
    if (!NextWord(&word)) return;
    value = std::bit_cast<std::int32_t>(word);
  }

  void operator()(std::uint64_t& value) override {
    // The low word first, then bits 32-63 (open decision O6). A missing field
    // leaves the member at its initializer, so neither word is written.
    if (AnyInvalid()) return;
    std::uint32_t low = 0;
    if (!NextWord(&low)) return;
    std::uint32_t high = 0;
    if (!NextWord(&high)) return;
    value = (static_cast<std::uint64_t>(high) << 32) | low;
  }

  void operator()(float& value) override {
    if (AnyInvalid()) return;
    std::uint32_t word = 0;
    if (!NextWord(&word)) return;
    const float stored = std::bit_cast<float>(word);
    // NaN and a non-finite value are not representable metadata: a bad value is
    // a hard error, and no load decision depends on a NaN
    // (design decision D7).
    if (!std::isfinite(stored)) {
      return NotifyInvalid("Invalid float %f\n", static_cast<double>(stored));
    }
    value = stored;
  }

  void operator()(std::string& value) override {
    if (SkipField()) return;
    std::uint32_t num_chars = 0;
    if (!NextWord(&num_chars)) return;
    const std::size_t num_words = DivCeil(num_chars, kCharsPerWord);
    if (num_words > kMaxStringWords) {
      return NotifyInvalid("String too long %u\n", num_chars);
    }
    // A block that ends inside a string is a corrupted or truncated file, not
    // an older one: every character the length word promises must be present.
    if (num_words > end_ - pos_) {
      return NotifyInvalid("Truncated string of %u characters\n", num_chars);
    }
    value.assign(num_chars, '\0');
    for (std::size_t i = 0; i < num_chars; ++i) {
      const std::uint32_t word = data_[pos_ + i / kCharsPerWord];
      const std::size_t shift = 8 * (i % kCharsPerWord);
      value[i] = static_cast<char>((word >> shift) & 0xffu);
    }
    // The stored length is the character count, so the zero padding of the last
    // word never reaches the value and there is nothing to trim.
    pos_ += num_words;
  }

  void operator()(IFields& fields) override {
    if (SkipField()) return;
    std::uint32_t num_words = 0;
    if (!NextWord(&num_words)) return;
    if (num_words > end_ - pos_) {
      // The block says it carries more words than the enclosing block holds,
      // so the span is shorter than the data says it should be.
      return NotifyInvalid("Block size %u exceeds the remaining %zu words\n",
                           num_words, end_ - pos_);
    }
    const std::size_t block_end = pos_ + num_words;
    const std::size_t saved_end = end_;
    end_ = block_end;
    fields.VisitFields(*this);
    end_ = saved_end;
    if (AnyInvalid()) return;
    if (pos_ < block_end) {
      // Fields the nested `VisitFields` does not know: the data is newer than
      // the code, which is valid (design decision D6).
      extra_ += static_cast<std::uint32_t>(block_end - pos_);
      pos_ = block_end;
    }
  }

 protected:
  bool SkipField() override {
    if (AnyInvalid()) return true;
    if (pos_ < end_) return false;
    // The data ended before this field. The member keeps its initializer and
    // the field is counted (design decision D5).
    ++missing_;
    return true;
  }

  bool IsReading() const override { return true; }

 private:
  // Copies the next word. Returns false, and counts the field as missing, when
  // the block ends first; the caller then leaves the member untouched.
  bool NextWord(std::uint32_t* word) {
    if (pos_ >= end_) {
      ++missing_;
      return false;
    }
    *word = data_[pos_++];
    return true;
  }

  SerializedSpan data_;
  std::size_t pos_ = 0;
  std::size_t end_ = 0;
  std::uint32_t missing_ = 0;
  std::uint32_t extra_ = 0;
};

}  // namespace

// --- IFieldsVisitor -------------------------------------------------------

IFieldsVisitor::~IFieldsVisitor() = default;

void IFieldsVisitor::NotifyInvalid(const char* fmt, ...) {
  any_invalid_ = true;
  // The header exposes the sticky flag and not the text, and the container
  // turns the flag into its own error string with the file name. The message is
  // still formatted and written, because "which field went wrong" is the one
  // fact a caller reading a corrupt file needs.
  std::va_list args;
  va_start(args, fmt);
  std::fputs("invalid field: ", stderr);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
}

// --- IFields --------------------------------------------------------------

IFields::~IFields() = default;

IFields::ReadResult IFields::Read(SerializedSpan span, std::size_t pos) {
  // A read that cannot start is a failure, not a block of missing fields: the
  // container reports an empty or truncated section itself, and a caller must
  // not mistake it for an older file.
  if (pos >= span.size()) return ReadResult(0);
  FieldsReader reader(span, pos);
  VisitFields(reader);
  return reader.Result();
}

bool IFields::AppendTo(std::vector<std::uint32_t>& storage) const {
  const std::size_t start = storage.size();
  FieldsWriter writer(&storage);
  // `VisitFields` is non-const because a read mutates the members. A write does
  // not, so the cast is safe here and nowhere else.
  const_cast<IFields*>(this)->VisitFields(writer);
  if (writer.AnyInvalid()) {
    // A partial block must never reach a file, so the words this call appended
    // are dropped (design decision D7).
    storage.resize(start);
    return false;
  }
  return true;
}

void IFields::Print() const {
  const std::vector<std::uint32_t> words = Write();
  std::fprintf(stderr, "%s: %zu word(s)\n", Name(), words.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    std::fprintf(stderr, "  [%3zu] %08x\n", i, words[i]);
  }
}

}  // namespace nanochat
