#ifndef NANOCHAT_SRC_PARQUET_FORMAT_H_
#define NANOCHAT_SRC_PARQUET_FORMAT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Parquet metadata structs and their Thrift compact parsers
// (docs/parquet-native.md). Only the fields the reader uses are kept; the
// parser skips every other field.

namespace nanochat {
namespace parquet {

// Parquet `Type` enum values, for the fields the reader checks.
enum class Type : int {
  kBoolean = 0,
  kInt32 = 1,
  kInt64 = 2,
  kInt96 = 3,
  kFloat = 4,
  kDouble = 5,
  kByteArray = 6,
  kFixedLenByteArray = 7,
};

// Parquet `FieldRepetitionType` enum values.
enum class Repetition : int {
  kRequired = 0,
  kOptional = 1,
  kRepeated = 2,
};

// Parquet `CompressionCodec` enum values.
enum class Codec : int {
  kUncompressed = 0,
  kSnappy = 1,
  kGzip = 2,
  kLzo = 3,
  kBrotli = 4,
  kLz4 = 5,
  kZstd = 6,
  kLz4Raw = 7,
};

// Parquet `Encoding` enum values.
enum class Encoding : int {
  kPlain = 0,
  kPlainDictionary = 2,
  kRle = 3,
  kBitPacked = 4,
  kDeltaBinaryPacked = 5,
  kDeltaLengthByteArray = 6,
  kDeltaByteArray = 7,
  kRleDictionary = 8,
  kByteStreamSplit = 9,
};

// Parquet `PageType` enum values.
enum class PageType : int {
  kDataPage = 0,
  kIndexPage = 1,
  kDictionaryPage = 2,
  kDataPageV2 = 3,
};

struct SchemaElement {
  int type = -1;
  int repetition = -1;
  std::string name;
  int num_children = 0;
  int converted_type = -1;
  bool has_type = false;
  bool has_repetition = false;
};

struct ColumnMetaData {
  int type = -1;
  std::vector<int> encodings;
  std::vector<std::string> path_in_schema;
  int codec = -1;
  std::int64_t num_values = 0;
  std::int64_t total_uncompressed_size = 0;
  std::int64_t total_compressed_size = 0;
  std::int64_t data_page_offset = 0;
  std::int64_t dictionary_page_offset = -1;
  bool has_data_page_offset = false;
  bool has_dictionary_page_offset = false;
};

struct ColumnChunk {
  std::string file_path;
  ColumnMetaData meta_data;
  bool has_meta_data = false;
};

struct RowGroup {
  std::vector<ColumnChunk> columns;
  std::int64_t total_byte_size = 0;
  std::int64_t num_rows = 0;
};

struct FileMetaData {
  int version = 0;
  std::vector<SchemaElement> schema;
  std::int64_t num_rows = 0;
  std::vector<RowGroup> row_groups;
  std::string created_by;
};

struct DataPageHeader {
  int num_values = 0;
  int encoding = -1;
  int definition_level_encoding = -1;
  int repetition_level_encoding = -1;
};

struct PageHeader {
  int type = -1;
  int uncompressed_page_size = 0;
  int compressed_page_size = 0;
  bool has_data_page_header = false;
  DataPageHeader data_page_header;
};

// Parses a `FileMetaData` struct from a compact-encoded buffer.
bool ParseFileMetaData(const std::uint8_t* data, std::size_t size,
                       FileMetaData* out, std::string* error);

// Parses one `PageHeader` from a compact-encoded buffer. Sets `*consumed` to
// the header size in bytes.
bool ParsePageHeader(const std::uint8_t* data, std::size_t size,
                     std::size_t* consumed, PageHeader* out,
                     std::string* error);

}  // namespace parquet
}  // namespace nanochat

#endif  // NANOCHAT_SRC_PARQUET_FORMAT_H_
