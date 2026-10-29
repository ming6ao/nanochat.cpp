#ifndef NANOCHAT_FIELDS_H_
#define NANOCHAT_FIELDS_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

// The field visitor: forward- and backward-compatible serialization for the
// run metadata block of the checkpoint container
// (docs/checkpoint-metadata-and-examples-plan.md stage A.1, and
// docs/checkpoint-format.md section 2.1).
//
// This is a port of `gemma.cpp` `io/fields.h`, which follows the libjxl
// `lib/jxl/fields.h` design. The contract names are kept verbatim so a reviewer
// can diff the two files side by side: `IFields`, `IFieldsVisitor`,
// `SerializedSpan`, `SkipField`, `IsReading`, `NotifyInvalid`, and
// `ReadResult{pos, missing_fields, extra_u32}`.
//
// This header is host-only and vendor-free: it includes no device, SIMD, or
// backend header, and no kernel header includes it (docs/build.md). The
// visit-or-skip mechanism is plain scalar code over a word stream, so a kernel
// never needs it.
//
// Design goals:
// - self-contained, so no separate schema compiler is needed (rules out
//   Protocol Buffers, FlatBuffers, and Cap'n Proto);
// - small enough to audit in one sitting (rules out Apache Fury and bitsery);
// - old code reads new data, and new code reads old data, so adding a metadata
//   field never rewrites the checkpoints that already sit on the Hub (rules out
//   yas and msgpack);
// - no user-specified version and no stored field names (rules out cereal,
//   JSON, and GGUF). A caller appends one line to `VisitFields` and nothing
//   else changes: an older reader counts the field it did not see, and a newer
//   reader counts the trailing words it does not know.
//
// Non-goals: encoding size and decode speed (the block is a few hundred words
// beside gigabytes of weights), and maps, interfaces, random access, optional
// fields, and deleted fields.
//
// The word stream, which is the contract between the writer and the reader:
// - `std::uint32_t`, `std::int32_t`, and `float` take one word each. A float
//   travels as its bit pattern (`std::bit_cast`), so a read never converts a
//   value.
// - `std::uint64_t` takes two words, the low 32 bits first and then bits 32-63
//   (open decision O6). Section 3 of the plan declares `std::uint64_t step`, so
//   the stream has to say how a 64-bit value is carried. The port keeps the
//   32-bit word stream and the full range: capping the step at `2^32 - 1` loses
//   a long run, and one word cannot hold it. The order is part of the contract,
//   because a stream written high word first would still decode, to a wrong
//   value.
// - `bool` and an enum take one word, as `std::uint32_t`.
// - `std::string` takes one length word and then that many words of characters,
//   zero-padded to the last whole word; the reader trims the trailing nulls.
// - `std::vector<T>` takes one count word and then the words of each entry.
// - A nested `IFields` takes one size word and then its payload, whose size
//   does not count the size word itself. The size prefix is why an older reader
//   can skip a block it does not know, and why a newer reader can tell how many
//   trailing words a block carried.
//
// Substitutions from the `gemma.cpp` original, one per row of the table in
// section 5 of the plan. The upstream SIMD-library span becomes
// `SerializedSpan`, which is `std::span<const std::uint32_t>`. Its
// unlikely-branch wrapper becomes `if (cond) [[unlikely]]`. Its assert macro
// becomes a plain `return`, because a container load reports an error and never
// aborts. Its `DivCeil` becomes the local `DivCeil` below. Its scalar bit-cast
// helper becomes `std::bit_cast` in the visitors in `src/fields.cc`. The plan's
// table carries the upstream spellings, so this header holds no SIMD-library
// token, and the vendor grep over `include/nanochat` finds nothing.

namespace nanochat {

struct IFields;  // Breaks the cycle with IFieldsVisitor::operator()(IFields&).

// A view of the serialized words of a metadata block. The container holds the
// words as little-endian `uint32` values on disk
// (docs/checkpoint-format.md section 5.2), and `src/data.cc` converts them
// once.
using SerializedSpan = std::span<const std::uint32_t>;

// Returns ceil(num / den). The local stand-in for the upstream `DivCeil`: a
// string of five characters needs two words.
inline constexpr std::size_t DivCeil(std::size_t num, std::size_t den) {
  return (num + den - 1) / den;
}

// The caps of the metadata block (section 3 of the plan). A longer string or
// vector is a hard error, in the same way as the container's
// `kMaxAttributes = 64`, so the block cannot become a second config file.
inline constexpr std::uint32_t kMaxStringWords = 4096;
inline constexpr std::uint32_t kMaxVectorEntries = 1024;

// The cap on the whole block, in words: 64 KiB. The container enforces it, not
// the visitor, because the block length comes from the file and not from the
// struct; it is named here so both sides use one number.
inline constexpr std::size_t kMaxBlockWords = 64 * 1024 / sizeof(std::uint32_t);

// Supported field types `T`: `std::uint32_t`, `std::int32_t`, `std::uint64_t`,
// `float`, `std::string`, an `IFields` subclass, `bool`, an enum that has an
// `EnumValid` overload, and `std::vector<T>` of the above.
//
// A visitor reads or writes one whole block. `NotifyInvalid` names the field
// that failed, and `AnyInvalid` stays true afterwards, so a caller checks once
// at the end and never half-loads a block.
class IFieldsVisitor {
 public:
  // Defined in `src/fields.cc`, like every other member of the port, so a
  // reviewer can diff the two files side by side.
  virtual ~IFieldsVisitor();

  // True once `NotifyInvalid` was called for any field of this visitor. The
  // flag is sticky for every `IFields` this visitor visits, so one bad value
  // rejects the whole block (design decision D7: a bad value is a hard error,
  // with no partial load).
  bool AnyInvalid() const { return any_invalid_; }

  // Each of these reads or writes one field. None of them throws and none of
  // them returns a status: an out-of-range value calls `NotifyInvalid`, and the
  // caller tests `AnyInvalid` once. One overload per type is required because
  // `std::vector<T>` dispatches on the element type.
  virtual void operator()(std::uint32_t& value) = 0;
  virtual void operator()(std::int32_t& value) = 0;
  virtual void operator()(std::uint64_t& value) = 0;
  virtual void operator()(float& value) = 0;
  virtual void operator()(std::string& value) = 0;
  virtual void operator()(IFields& fields) = 0;  // Recurse into a nested block.

  // A `bool` and an enum are stored as one `std::uint32_t` word. Two
  // requirements come from the format rules in
  // `nanochat/checkpoint_metadata.h`: a `bool` accepts only 0 and 1, and an
  // enum accepts only a value that `EnumValid` knows, so a new state is a new
  // field and never an added enumerator. `EnumValid` is found by
  // argument-dependent lookup, so an enum in `nanochat` declares
  // `bool EnumValid(MyEnum)` in `nanochat`.
  void operator()(bool& value) {
    if (SkipField()) [[unlikely]] {
      return;
    }

    std::uint32_t u32 = value ? 1 : 0;
    operator()(u32);
    if (u32 > 1) [[unlikely]] {
      return NotifyInvalid("Invalid bool %u\n", u32);
    }
    value = (u32 == 1);
  }

  template <typename EnumT>
    requires std::is_enum_v<EnumT>
  void operator()(EnumT& value) {
    if (SkipField()) [[unlikely]] {
      return;
    }

    std::uint32_t u32 = static_cast<std::uint32_t>(value);
    operator()(u32);
    if (!EnumValid(static_cast<EnumT>(u32))) [[unlikely]] {
      return NotifyInvalid("Invalid enum %u\n", u32);
    }
    value = static_cast<EnumT>(u32);
  }

  // `std::vector<T>` of any supported element type. `std::vector<bool>` is not
  // supported, because a `bool` element has no addressable reference to visit.
  template <typename T>
  void operator()(std::vector<T>& value) {
    if (SkipField()) [[unlikely]] {
      return;
    }

    std::uint32_t num = static_cast<std::uint32_t>(value.size());
    operator()(num);
    if (num > kMaxVectorEntries) [[unlikely]] {
      return NotifyInvalid("Vector too long %u\n", num);
    }

    if (IsReading()) {
      value.resize(num);
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
      operator()(value[i]);
    }
  }

 protected:
  // Records the message and makes `AnyInvalid` return true. The container turns
  // that flag into its own error string, so a load fails with the field that
  // went wrong. It is printf-style for the field value, like the original.
  void NotifyInvalid(const char* fmt, ...);

  // Must be checked before a field is modified: returning true means the field
  // was not read, and the member keeps its value. This is what makes "new code,
  // old data" work, because resizing a string or a vector for a field the data
  // does not carry would discard its contents.
  virtual bool SkipField() { return AnyInvalid(); }

  // True for a visitor that reads, so `operator()(std::vector<T>&)` sizes the
  // vector from the count word. The visitor itself is protected because only a
  // concrete visitor in `src/fields.cc` answers it.
  virtual bool IsReading() const { return false; }

 private:
  bool any_invalid_ = false;
};

// The base class of a serializable struct: a forward- and backward-compatible
// collection of fields. An older reader can read new data and a newer reader
// can read old data, which is what keeps a metadata field addition cheap.
//
// HOWTO:
// - declare the members with initializers, e.g. `std::uint32_t schema = 1;`;
// - define `void VisitFields(IFieldsVisitor& visitor)` and call
//   `visitor(field);` once per field, in the order the format fixes;
// - define `const char* Name()`, which is a caption for `Print` and is never
//   stored in the stream;
// - append a new field last. A field is never reordered, renamed, or removed.
struct IFields {
  // Defined in `src/fields.cc`, like the visitor's destructor.
  virtual ~IFields();

  // A caption for `Print`. It is not part of the format.
  virtual const char* Name() const = 0;

  // Called by `IFieldsVisitor::operator()(IFields&)`.
  virtual void VisitFields(IFieldsVisitor& visitor) = 0;

  // Prints the name and the fields to stderr.
  void Print() const;

  // What a read did with the stored size.
  struct ReadResult {
    ReadResult(std::size_t pos) : pos(pos), missing_fields(0), extra_u32(0) {}

    // Where to resume reading in the next `Read` call, or 0 if the read failed:
    // some field held an invalid value, or the span is shorter than the data
    // says it should be. On a zero, use neither the fields nor the stream.
    std::size_t pos;
    // How many fields `VisitFields` wanted that the data did not carry, from
    // the point of view of `VisitFields`. Non-zero means the data is older than
    // the code, which is valid, and `extra_u32` is then zero.
    std::uint32_t missing_fields;
    // How many words the block carried beyond what `VisitFields` read.
    // Non-zero means the data is newer than the code, which is valid, and
    // `missing_fields` is then zero. The container warns once.
    std::uint32_t extra_u32;
  };

  // Reads fields starting at `span[pos]`. `pos` is where the previous read
  // stopped, so one stream of back-to-back blocks reads in order.
  ReadResult Read(SerializedSpan span, std::size_t pos);

  // Writes fields to the end of `storage`, so one stream of back-to-back blocks
  // appends in order. Returns false when a field held a value the format cannot
  // carry, typically a string longer than `kMaxStringWords`; the caller then
  // discards the whole block rather than writing a partial one.
  bool AppendTo(std::vector<std::uint32_t>& storage) const;

  // The one-shot form of `AppendTo`. An invalid value yields an empty vector,
  // because a partial block must never reach a file.
  std::vector<std::uint32_t> Write() const {
    std::vector<std::uint32_t> storage;
    if (!AppendTo(storage)) storage.clear();
    return storage;
  }
};

}  // namespace nanochat

#endif  // NANOCHAT_FIELDS_H_
