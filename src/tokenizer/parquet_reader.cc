// The DuckDB C++ parquet reader (docs/tokenizer.md section 10). It reads one
// text column, orders the rows by file name and then by row number, and
// returns one bounded batch per `Next` call. The DuckDB library is a host
// dependency; only the tokenizer tools link it.
//
// DuckDB reports an error by throwing. The project bans exceptions in its own
// code, so every DuckDB call sits in a `try` block that turns the error into a
// string result.

#include "src/tokenizer/parquet_reader.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <duckdb.hpp>

namespace nanochat {
namespace {

// Quotes one SQL string literal. A single quote doubles.
std::string QuoteLiteral(const std::string& value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'')
      quoted += "''";
    else
      quoted += c;
  }
  quoted += "'";
  return quoted;
}

// Quotes one SQL identifier. A double quote doubles.
std::string QuoteIdentifier(const std::string& name) {
  std::string quoted = "\"";
  for (char c : name) {
    if (c == '"')
      quoted += "\"\"";
    else
      quoted += c;
  }
  quoted += "\"";
  return quoted;
}

std::string BuildFileList(const std::vector<std::string>& files) {
  std::string list = "[";
  for (std::size_t index = 0; index < files.size(); ++index) {
    if (index != 0) list += ", ";
    list += QuoteLiteral(files[index]);
  }
  list += "]";
  return list;
}

std::string BuildQuery(const ParquetReader::Options& options) {
  // The subquery projects the hidden ordering columns, so the outer `ORDER BY`
  // does not depend on the projection.
  return "SELECT doc FROM (SELECT " + QuoteIdentifier(options.text_column) +
         " AS doc, filename AS source_file, file_row_number AS source_row "
         "FROM read_parquet(" +
         BuildFileList(options.files) +
         ", filename = true, file_row_number = true)) "
         "ORDER BY source_file, source_row";
}

}  // namespace

struct ParquetReader::Impl {
  std::unique_ptr<duckdb::DuckDB> database;
  std::unique_ptr<duckdb::Connection> connection;
  std::unique_ptr<duckdb::QueryResult> result;
  std::unique_ptr<duckdb::DataChunk> chunk;
  duckdb::idx_t chunk_offset = 0;
  std::size_t batch_size = 256;
  bool finished = false;
};

ParquetReader::ParquetReader() = default;
ParquetReader::~ParquetReader() = default;

std::unique_ptr<ParquetReader> ParquetReader::Open(const Options& options,
                                                   std::string* error) {
  auto fail =
      [&](const std::string& message) -> std::unique_ptr<ParquetReader> {
    if (error != nullptr) *error = message;
    return nullptr;
  };
  if (options.files.empty()) return fail("parquet reader: no input files");
  if (options.text_column.empty()) {
    return fail("parquet reader: empty text column");
  }

  try {
    // `ParquetReader` has a private constructor, so `std::make_unique` cannot
    // reach it. Wrap the raw allocation directly, like `TokenShard::Open`.
    std::unique_ptr<ParquetReader> reader(new ParquetReader());
    reader->impl_ = std::make_unique<Impl>();
    reader->impl_->batch_size =
        options.batch_size == 0 ? 1 : options.batch_size;
    reader->impl_->database = std::make_unique<duckdb::DuckDB>(nullptr);
    reader->impl_->connection =
        std::make_unique<duckdb::Connection>(*reader->impl_->database);

    const std::string query = BuildQuery(options);
    reader->impl_->result = reader->impl_->connection->SendQuery(query);
    if (reader->impl_->result == nullptr || reader->impl_->result->HasError()) {
      const std::string message = reader->impl_->result == nullptr
                                      ? "query produced no result"
                                      : reader->impl_->result->GetError();
      return fail("parquet reader: " + message);
    }
    return reader;
  } catch (const std::exception& exception) {
    return fail(std::string("parquet reader: ") + exception.what());
  }
}

bool ParquetReader::Next(std::vector<std::string>* documents,
                         std::string* error) {
  if (documents == nullptr) return false;
  documents->clear();
  if (error != nullptr) error->clear();
  Impl* impl = impl_.get();
  if (impl == nullptr || impl->finished) return false;

  try {
    while (documents->size() < impl->batch_size) {
      if (impl->chunk == nullptr || impl->chunk_offset >= impl->chunk->size()) {
        impl->chunk = impl->result->Fetch();
        impl->chunk_offset = 0;
        if (impl->chunk == nullptr || impl->chunk->size() == 0) {
          impl->finished = true;
          break;
        }
      }
      while (impl->chunk_offset < impl->chunk->size() &&
             documents->size() < impl->batch_size) {
        const duckdb::Value value =
            impl->chunk->GetValue(0, impl->chunk_offset);
        if (value.IsNull()) {
          documents->emplace_back();
        } else {
          documents->push_back(value.GetValue<std::string>());
        }
        ++impl->chunk_offset;
      }
    }
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = std::string("parquet reader: ") + exception.what();
    impl->finished = true;
    return false;
  }
  return !documents->empty();
}

}  // namespace nanochat
