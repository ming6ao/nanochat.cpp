// src/parquet/reader_test.cc -- the native parquet reader, happy path.
//
// Reads the committed fixture (five row groups of two documents) and checks
// the document order, the batch sizes, the glob expansion, and the
// missing-column error.

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "src/parquet/reader.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

const std::vector<std::string>& Expected() {
  static const std::vector<std::string> kDocuments = {
      "alpha",
      "beta",
      "gamma",
      "delta",
      "epsilon",
      "",
      "a longer document with spaces",
      "unicode: caf\xc3\xa9 \xe2\x98\x95",
      "line one\nline two",
      "tab\tseparated value"};
  return kDocuments;
}

// Reads the whole dataset and returns every document, in order.
bool ReadAll(const std::string& path, std::size_t batch_size,
             std::vector<std::string>* documents, std::string* error) {
  nanochat::ParquetReader::Options options;
  options.files = {path};
  options.batch_size = batch_size;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, error);
  if (reader == nullptr) return false;
  // The reader is a `DocumentSource`, so the loader can consume it directly.
  nanochat::DocumentSource* source = reader.get();
  std::vector<std::string> batch;
  while (source->Next(&batch, error)) {
    for (const std::string& document : batch) documents->push_back(document);
  }
  return error->empty();
}

void CheckBatching(const std::string& path, std::size_t batch_size) {
  std::vector<std::string> documents;
  std::string error;
  if (!ReadAll(path, batch_size, &documents, &error)) {
    Fail("batching: " + error);
    return;
  }
  if (documents != Expected()) {
    Fail("batching: document list mismatch");
  }
}

void CheckGlob(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    Fail("glob: no directory in the fixture path");
    return;
  }
  const std::string pattern = path.substr(0, slash + 1) + "*.parquet";
  std::vector<std::string> documents;
  std::string error;
  if (!ReadAll(pattern, 4, &documents, &error)) {
    Fail("glob: " + error);
    return;
  }
  if (documents != Expected()) {
    Fail("glob: document list mismatch");
  }
}

void CheckMissingColumn(const std::string& path) {
  nanochat::ParquetReader::Options options;
  options.files = {path};
  options.text_column = "missing_column";
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader != nullptr) {
    Fail("missing column: Open accepted the wrong column");
  } else if (error.empty()) {
    Fail("missing column: Open did not report an error");
  }
}

void CheckMissingFile() {
  nanochat::ParquetReader::Options options;
  options.files = {"/nonexistent/nanochat/missing.parquet"};
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader != nullptr) {
    Fail("missing file: Open accepted a missing file");
  } else if (error.empty()) {
    Fail("missing file: Open did not report an error");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "reader_test: missing fixture path\n");
    return 2;
  }
  const std::string path = argv[1];
  CheckBatching(path, 3);
  CheckBatching(path, 1);
  CheckBatching(path, 256);
  CheckGlob(path);
  CheckMissingColumn(path);
  CheckMissingFile();
  if (g_failures != 0) {
    std::fprintf(stderr, "reader_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("reader_test: ok\n");
  return 0;
}
