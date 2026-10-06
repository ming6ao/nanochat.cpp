#ifndef NANOCHAT_SRC_PARQUET_READER_H_
#define NANOCHAT_SRC_PARQUET_READER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Streams the text column of a parquet dataset through the DuckDB C++ API
// (docs/tokenizer.md section 10). The reader selects one text column, orders
// the rows by file name and row number, and preserves the dataset order. A
// small batch bounds the documents in memory.
//
// The class is a host utility. It has no kernel dependency.

namespace nanochat {

class ParquetReader {
 public:
  struct Options {
    // One parquet glob, or a list of parquet paths.
    std::vector<std::string> files;
    // The text column to read. The default is the reference column name.
    std::string text_column = "text";
    // The largest number of documents one `Next` call returns.
    std::size_t batch_size = 256;
  };

  // Opens the dataset. Returns null and sets `error` on a bad file or a bad
  // query.
  static std::unique_ptr<ParquetReader> Open(const Options& options,
                                             std::string* error);
  ~ParquetReader();

  ParquetReader(const ParquetReader&) = delete;
  ParquetReader& operator=(const ParquetReader&) = delete;

  // Fetches up to `batch_size` documents into `documents`, in dataset order.
  // Returns false at the end of the input or on an error. `error` stays empty
  // at the end of the input.
  bool Next(std::vector<std::string>* documents, std::string* error);

 private:
  ParquetReader();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_PARQUET_READER_H_
