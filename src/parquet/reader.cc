#include "src/parquet/reader.h"

#include <glob.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <zstd.h>

#include "src/parquet/format.h"
#include "src/parquet/levels.h"

namespace nanochat {
namespace {

constexpr char kParquetMagic[4] = {'P', 'A', 'R', '1'};

std::uint32_t ReadLE32(const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8) |
         (static_cast<std::uint32_t>(data[2]) << 16) |
         (static_cast<std::uint32_t>(data[3]) << 24);
}

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

bool HasGlob(const std::string& path) {
  return path.find_first_of("*?[") != std::string::npos;
}

// Expands the globs and orders every file by name, then removes a duplicate.
bool ExpandPaths(const std::vector<std::string>& inputs,
                 std::vector<std::string>* files, std::string* error) {
  for (const std::string& input : inputs) {
    if (input.empty()) continue;
    if (!HasGlob(input)) {
      files->push_back(input);
      continue;
    }
    glob_t result;
    std::memset(&result, 0, sizeof(result));
    const int status = glob(input.c_str(), GLOB_NOSORT, nullptr, &result);
    if (status == GLOB_NOMATCH) {
      globfree(&result);
      continue;
    }
    if (status != 0) {
      globfree(&result);
      return Fail(error, "parquet reader: bad glob: " + input);
    }
    for (std::size_t i = 0; i < result.gl_pathc; ++i) {
      files->emplace_back(result.gl_pathv[i]);
    }
    globfree(&result);
  }
  std::sort(files->begin(), files->end());
  files->erase(std::unique(files->begin(), files->end()), files->end());
  return true;
}

// Reads the footer and parses the compact-encoded `FileMetaData`.
bool ReadFooter(const std::string& path, parquet::FileMetaData* meta,
                std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Fail(error, "parquet reader: cannot open " + path);
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 12) {
    return Fail(error, "parquet reader: file is too small: " + path);
  }
  in.seekg(size - 8, std::ios::beg);
  std::uint8_t trailer[8];
  in.read(reinterpret_cast<char*>(trailer), sizeof(trailer));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(trailer))) {
    return Fail(error, "parquet reader: truncated file trailer: " + path);
  }
  if (std::memcmp(trailer + 4, kParquetMagic, 4) != 0) {
    return Fail(error, "parquet reader: missing PAR1 magic: " + path);
  }
  const std::uint32_t footer_length = ReadLE32(trailer);
  if (footer_length == 0 ||
      static_cast<std::streamoff>(footer_length) > size - 12) {
    return Fail(error, "parquet reader: bad footer length: " + path);
  }
  std::vector<std::uint8_t> footer(footer_length);
  in.seekg(size - 8 - static_cast<std::streamoff>(footer_length),
           std::ios::beg);
  in.read(reinterpret_cast<char*>(footer.data()),
          static_cast<std::streamsize>(footer.size()));
  if (in.gcount() != static_cast<std::streamsize>(footer.size())) {
    return Fail(error, "parquet reader: truncated footer: " + path);
  }
  return parquet::ParseFileMetaData(footer.data(), footer.size(), meta, error);
}

// Checks the exact shape the reader supports: one flat OPTIONAL BYTE_ARRAY
// leaf with the requested name.
bool ValidateSchema(const parquet::FileMetaData& meta,
                    const std::string& text_column, const std::string& path,
                    std::string* error) {
  if (meta.schema.size() != 2) {
    return Fail(
        error,
        "parquet reader: only a one-column flat schema is supported: " + path);
  }
  if (meta.schema[0].num_children != 1) {
    return Fail(error, "parquet reader: expected one column: " + path);
  }
  const parquet::SchemaElement& leaf = meta.schema[1];
  if (leaf.name != text_column) {
    return Fail(
        error, "parquet reader: missing column '" + text_column + "': " + path);
  }
  if (!leaf.has_type ||
      leaf.type != static_cast<int>(parquet::Type::kByteArray)) {
    return Fail(error,
                "parquet reader: the text column is not BYTE_ARRAY: " + path);
  }
  if (!leaf.has_repetition ||
      (leaf.repetition != static_cast<int>(parquet::Repetition::kOptional) &&
       leaf.repetition != static_cast<int>(parquet::Repetition::kRequired))) {
    return Fail(
        error, "parquet reader: the text column must not be repeated: " + path);
  }
  if (leaf.num_children != 0) {
    return Fail(error,
                "parquet reader: nested columns are not supported: " + path);
  }
  return true;
}

// Reads one column chunk and appends its documents to `pending`. The chunk
// must be a single ZSTD `DATA_PAGE` v1 sequence with PLAIN values and RLE
// definition levels.
bool ReadRowGroup(std::ifstream* in, const parquet::RowGroup& row_group,
                  int max_definition_level, const std::string& path,
                  std::deque<std::string>* pending, std::string* error) {
  if (row_group.columns.size() != 1) {
    return Fail(error, "parquet reader: expected one column chunk: " + path);
  }
  const parquet::ColumnChunk& chunk = row_group.columns[0];
  if (!chunk.has_meta_data) {
    return Fail(error, "parquet reader: missing column metadata: " + path);
  }
  const parquet::ColumnMetaData& meta = chunk.meta_data;
  if (meta.codec != static_cast<int>(parquet::Codec::kZstd)) {
    return Fail(error, "parquet reader: only ZSTD is supported: " + path);
  }
  if (meta.has_dictionary_page_offset) {
    return Fail(error,
                "parquet reader: dictionary pages are not supported: " + path);
  }
  if (!meta.has_data_page_offset || meta.total_compressed_size <= 0) {
    return Fail(error, "parquet reader: missing data pages: " + path);
  }

  std::vector<std::uint8_t> raw(
      static_cast<std::size_t>(meta.total_compressed_size));
  in->clear();
  in->seekg(static_cast<std::streamoff>(meta.data_page_offset), std::ios::beg);
  in->read(reinterpret_cast<char*>(raw.data()),
           static_cast<std::streamsize>(raw.size()));
  if (in->gcount() != static_cast<std::streamsize>(raw.size())) {
    return Fail(error, "parquet reader: truncated column chunk: " + path);
  }

  std::vector<std::uint8_t> page;
  std::vector<std::uint32_t> levels;
  std::size_t position = 0;
  std::int64_t values_read = 0;
  while (values_read < meta.num_values) {
    if (position >= raw.size()) {
      return Fail(error, "parquet reader: ran out of pages: " + path);
    }
    parquet::PageHeader header;
    std::size_t header_size = 0;
    if (!parquet::ParsePageHeader(raw.data() + position, raw.size() - position,
                                  &header_size, &header, error)) {
      return false;
    }
    if (header.type != static_cast<int>(parquet::PageType::kDataPage)) {
      return Fail(error,
                  "parquet reader: only DATA_PAGE v1 is supported: " + path);
    }
    if (!header.has_data_page_header) {
      return Fail(error, "parquet reader: missing data page header: " + path);
    }
    if (header.data_page_header.encoding !=
            static_cast<int>(parquet::Encoding::kPlain) ||
        header.data_page_header.definition_level_encoding !=
            static_cast<int>(parquet::Encoding::kRle) ||
        header.data_page_header.repetition_level_encoding !=
            static_cast<int>(parquet::Encoding::kRle)) {
      return Fail(error, "parquet reader: unsupported page encoding: " + path);
    }
    const std::size_t body = position + header_size;
    const std::size_t compressed =
        static_cast<std::size_t>(header.compressed_page_size);
    if (body + compressed > raw.size()) {
      return Fail(error, "parquet reader: truncated page body: " + path);
    }

    page.resize(static_cast<std::size_t>(header.uncompressed_page_size));
    const std::size_t produced = ZSTD_decompress(page.data(), page.size(),
                                                 raw.data() + body, compressed);
    if (ZSTD_isError(produced)) {
      return Fail(error, std::string("parquet reader: zstd: ") +
                             ZSTD_getErrorName(produced));
    }
    if (produced != page.size()) {
      return Fail(error, "parquet reader: zstd size mismatch: " + path);
    }

    const int page_values = header.data_page_header.num_values;
    if (page_values < 0) {
      return Fail(error, "parquet reader: negative value count: " + path);
    }
    std::size_t cursor = 0;
    levels.clear();
    if (max_definition_level > 0) {
      if (cursor + 4 > page.size()) {
        return Fail(error, "parquet reader: truncated level length: " + path);
      }
      const std::uint32_t level_bytes = ReadLE32(page.data() + cursor);
      cursor += 4;
      if (cursor + level_bytes > page.size()) {
        return Fail(error, "parquet reader: truncated levels: " + path);
      }
      if (!parquet::DecodeRleBitPacked(page.data() + cursor, level_bytes,
                                       /*bit_width=*/1,
                                       static_cast<std::size_t>(page_values),
                                       &levels, error)) {
        return false;
      }
      cursor += level_bytes;
    }

    for (int i = 0; i < page_values; ++i) {
      if (max_definition_level > 0 &&
          levels[static_cast<std::size_t>(i)] == 0) {
        return Fail(
            error,
            "parquet reader: null text values are not supported: " + path);
      }
      if (cursor + 4 > page.size()) {
        return Fail(error, "parquet reader: truncated value length: " + path);
      }
      const std::uint32_t length = ReadLE32(page.data() + cursor);
      cursor += 4;
      if (cursor + length > page.size()) {
        return Fail(error, "parquet reader: truncated value: " + path);
      }
      pending->emplace_back(reinterpret_cast<const char*>(page.data() + cursor),
                            length);
      cursor += length;
    }
    values_read += page_values;
    position = body + compressed;
  }
  if (values_read != meta.num_values) {
    return Fail(error, "parquet reader: value count mismatch: " + path);
  }
  return true;
}

}  // namespace

struct ParquetReader::Impl {
  struct File {
    std::string path;
    std::ifstream in;
    parquet::FileMetaData meta;
    int max_definition_level = 1;
    std::size_t next_row_group = 0;
  };

  std::vector<File> files;
  std::size_t file_index = 0;
  std::deque<std::string> pending;
  std::size_t batch_size = 256;

  // Reads row groups until `pending` holds documents or the input ends.
  // Returns false at the end of the input or on an error. `error` stays empty
  // at the end of the input.
  bool Fill(std::string* error) {
    if (!pending.empty()) return true;
    while (file_index < files.size()) {
      File& file = files[file_index];
      if (file.next_row_group >= file.meta.row_groups.size()) {
        ++file_index;
        continue;
      }
      const parquet::RowGroup& row_group =
          file.meta.row_groups[file.next_row_group];
      ++file.next_row_group;
      if (!ReadRowGroup(&file.in, row_group, file.max_definition_level,
                        file.path, &pending, error)) {
        return false;
      }
      if (!pending.empty()) return true;
    }
    return false;
  }
};

ParquetReader::ParquetReader() = default;
ParquetReader::~ParquetReader() = default;

std::unique_ptr<ParquetReader> ParquetReader::Open(const Options& options,
                                                   std::string* error) {
  if (options.files.empty()) {
    Fail(error, "parquet reader: no input files");
    return nullptr;
  }
  if (options.text_column.empty()) {
    Fail(error, "parquet reader: empty text column");
    return nullptr;
  }
  std::vector<std::string> paths;
  if (!ExpandPaths(options.files, &paths, error)) return nullptr;
  if (paths.empty()) {
    Fail(error, "parquet reader: no input files matched");
    return nullptr;
  }

  // `ParquetReader` has a private constructor, so `std::make_unique` cannot
  // reach it. Wrap the raw allocation directly.
  std::unique_ptr<ParquetReader> reader(new ParquetReader());
  reader->impl_ = std::make_unique<Impl>();
  reader->impl_->batch_size = options.batch_size == 0 ? 1 : options.batch_size;
  reader->impl_->files.reserve(paths.size());
  for (const std::string& path : paths) {
    Impl::File file;
    file.path = path;
    if (!ReadFooter(path, &file.meta, error)) return nullptr;
    if (!ValidateSchema(file.meta, options.text_column, path, error)) {
      return nullptr;
    }
    file.max_definition_level =
        file.meta.schema[1].repetition ==
                static_cast<int>(parquet::Repetition::kOptional)
            ? 1
            : 0;
    file.in.open(path, std::ios::binary);
    if (!file.in) {
      Fail(error, "parquet reader: cannot open " + path);
      return nullptr;
    }
    reader->impl_->files.push_back(std::move(file));
  }
  return reader;
}

bool ParquetReader::Next(std::vector<std::string>* documents,
                         std::string* error) {
  if (documents == nullptr) return false;
  documents->clear();
  if (error != nullptr) error->clear();
  if (impl_ == nullptr) return false;

  while (documents->size() < impl_->batch_size) {
    if (impl_->pending.empty()) {
      if (!impl_->Fill(error)) {
        // The input ended, or a read failed. On a failure, discard the
        // partial batch and report the error.
        if (error != nullptr && !error->empty()) return false;
        break;
      }
    }
    documents->push_back(std::move(impl_->pending.front()));
    impl_->pending.pop_front();
  }
  return !documents->empty();
}

std::unique_ptr<DocumentSource> OpenParquetSource(
    const std::vector<std::string>& files, const std::string& text_column,
    std::size_t batch_size, std::string* error) {
  ParquetReader::Options options;
  options.files = files;
  options.text_column = text_column;
  options.batch_size = batch_size;
  return ParquetReader::Open(options, error);
}

}  // namespace nanochat
