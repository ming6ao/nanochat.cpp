#ifndef NANOCHAT_SRC_PARQUET_THRIFT_H_
#define NANOCHAT_SRC_PARQUET_THRIFT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// Thrift compact protocol decoder (docs/parquet-native.md). Parquet stores its
// file metadata and page headers as Thrift compact structs. The decoder reads
// the fields it needs and skips the rest, so an added upstream field does not
// break the reader.
//
// The decoder never throws. A malformed buffer sets an error and a sticky
// `ok()` false. The caller checks `ok()` at the end of a parse.

namespace nanochat {
namespace parquet {

// The Thrift compact protocol type codes.
enum class CompactType : int {
  kStop = 0,
  kBoolTrue = 1,
  kBoolFalse = 2,
  kByte = 3,
  kI16 = 4,
  kI32 = 5,
  kI64 = 6,
  kDouble = 7,
  kBinary = 8,
  kList = 9,
  kSet = 10,
  kMap = 11,
  kStruct = 12,
};

class CompactReader {
 public:
  CompactReader(const std::uint8_t* data, std::size_t size);

  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }
  std::size_t position() const { return position_; }

  // Reads the next struct field header. Returns false at STOP or on error.
  // A caller that needs to tell STOP from an error checks `ok()`.
  bool NextField(int* field_id, CompactType* type);

  // Values. Each returns a zero value and sets `ok()` false on error.
  std::int64_t ReadInt();
  bool ReadBool(CompactType type);
  double ReadDouble();
  std::string_view ReadBinary();

  struct ListHeader {
    CompactType type = CompactType::kStop;
    std::size_t size = 0;
  };
  bool ReadListHeader(ListHeader* header);

  struct MapHeader {
    CompactType key_type = CompactType::kStop;
    CompactType value_type = CompactType::kStop;
    std::size_t size = 0;
  };
  bool ReadMapHeader(MapHeader* header);

  // Consumes one value of `type`, including a nested collection or struct.
  void Skip(CompactType type);

  // Consumes one complete struct, including the closing STOP.
  void SkipStruct();

  // Saves and restores the field-id delta state around a nested struct parse.
  int field_id_state() const { return last_field_id_; }
  void set_field_id_state(int state) { last_field_id_ = state; }

  void Fail(const std::string& message);

 private:
  std::uint8_t ReadByte();
  std::uint64_t ReadVarint();
  static std::int64_t ZigZag(std::uint64_t value);

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t position_ = 0;
  bool ok_ = true;
  std::string error_;
  // The last field id of the current struct, for the delta encoding.
  int last_field_id_ = 0;
};

}  // namespace parquet
}  // namespace nanochat

#endif  // NANOCHAT_SRC_PARQUET_THRIFT_H_
