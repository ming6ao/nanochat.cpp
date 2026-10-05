// src/tokenizer/parquet_reader_test.cc -- the DuckDB parquet reader.
//
// Writes two tiny parquet files with DuckDB, then checks that the reader
// returns the text column in dataset order (file name, then row number),
// honors the text column option, and streams bounded batches. The fixture is
// built at run time, so no parquet file enters the repository.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <duckdb.hpp>
#include "src/tokenizer/parquet_reader.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

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

// Writes a two-column parquet file: `id` and `body`.
bool WriteParquet(const std::string& path,
                  const std::vector<std::string>& bodies) {
  duckdb::DuckDB database(nullptr);
  duckdb::Connection connection(database);
  auto result = connection.Query("CREATE TABLE t(id INTEGER, body VARCHAR)");
  if (result->HasError()) return false;
  for (std::size_t index = 0; index < bodies.size(); ++index) {
    result = connection.Query("INSERT INTO t VALUES (" + std::to_string(index) +
                              ", " + QuoteLiteral(bodies[index]) + ")");
    if (result->HasError()) return false;
  }
  result =
      connection.Query("COPY t TO " + QuoteLiteral(path) + " (FORMAT PARQUET)");
  return !result->HasError();
}

void CheckOrderedRead() {
  const std::string file_b = TempPath("reader_part_b.parquet");
  const std::string file_a = TempPath("reader_part_a.parquet");
  if (!WriteParquet(file_b, {"b1", "b2"})) {
    Fail("could not write parquet file b");
    return;
  }
  if (!WriteParquet(file_a, {"a1", "a2"})) {
    Fail("could not write parquet file a");
    return;
  }

  nanochat::ParquetReader::Options options;
  // The list order is reversed on purpose: the reader must order by file name.
  options.files = {file_b, file_a};
  options.text_column = "body";
  options.batch_size = 1;
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader == nullptr) {
    Fail("could not open the parquet dataset: " + error);
    return;
  }

  const std::vector<std::string> expected = {"a1", "a2", "b1", "b2"};
  for (const std::string& document : expected) {
    std::vector<std::string> batch;
    if (!reader->Next(&batch, &error)) {
      Fail("Next returned false before the end: " + error);
      return;
    }
    if (batch.size() != 1 || batch[0] != document) {
      Fail("unexpected batch: expected '" + document + "'");
    }
  }
  std::vector<std::string> end;
  if (reader->Next(&end, &error)) {
    Fail("Next returned a document after the end");
  }
  if (!error.empty()) {
    Fail("Next set an error at the end of the input: " + error);
  }
}

void CheckGlobAndBatching() {
  const std::string file_0 = TempPath("reader_shard_00.parquet");
  const std::string file_1 = TempPath("reader_shard_01.parquet");
  if (!WriteParquet(file_0, {"one"})) {
    Fail("could not write shard 00");
    return;
  }
  if (!WriteParquet(file_1, {"two"})) {
    Fail("could not write shard 01");
    return;
  }

  nanochat::ParquetReader::Options options;
  options.files = {TempPath("reader_shard_*.parquet")};
  options.text_column = "body";
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader == nullptr) {
    Fail("could not open the parquet glob: " + error);
    return;
  }
  std::vector<std::string> batch;
  if (!reader->Next(&batch, &error)) {
    Fail("glob Next returned false: " + error);
    return;
  }
  if (batch.size() != 2 || batch[0] != "one" || batch[1] != "two") {
    Fail("glob returned an unexpected batch");
  }
}

void CheckBadColumn() {
  const std::string path = TempPath("reader_bad_column.parquet");
  if (!WriteParquet(path, {"value"})) {
    Fail("could not write the bad-column fixture");
    return;
  }
  nanochat::ParquetReader::Options options;
  options.files = {path};
  options.text_column = "missing_column";
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader != nullptr) {
    Fail("Open accepted a missing text column");
  } else if (error.empty()) {
    Fail("Open did not report the missing text column");
  }
}

}  // namespace

int main() {
  CheckOrderedRead();
  CheckGlobAndBatching();
  CheckBadColumn();
  if (g_failures != 0) {
    std::fprintf(stderr, "parquet_reader_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("parquet_reader_test: ok\n");
  return 0;
}
