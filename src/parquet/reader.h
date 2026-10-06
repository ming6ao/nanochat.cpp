#ifndef NANOCHAT_SRC_PARQUET_READER_H_
#define NANOCHAT_SRC_PARQUET_READER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/dataloader.h"

// Reads the text column of a parquet dataset (docs/parquet-native.md). The
// reader supports the exact shape that `parquet-cpp-arrow` writes for the
// nanochat corpus: one OPTIONAL `BYTE_ARRAY` UTF8 column, ZSTD pages,
// `DATA_PAGE` v1, `PLAIN` values, and RLE definition levels. Every other
// feature is rejected with an error.
//
// The reader is a host utility. It has no kernel dependency. `files` may hold
// explicit paths or shell globs; the reader expands the globs and orders the
// files by name, so the document order matches the dataset order. It is a
// `DocumentSource`, so `DataLoader` can tokenize its output directly.

namespace nanochat {

class ParquetReader : public DocumentSource {
 public:
  struct Options {
    std::vector<std::string> files;
    std::string text_column = "text";
    std::size_t batch_size = 256;
  };

  // Opens the dataset. Returns null and sets `error` on a bad file, a bad
  // schema, or an unsupported feature.
  static std::unique_ptr<ParquetReader> Open(const Options& options,
                                             std::string* error);
  ~ParquetReader();

  ParquetReader(const ParquetReader&) = delete;
  ParquetReader& operator=(const ParquetReader&) = delete;

  // Fetches up to `batch_size` documents into `documents`, in dataset order.
  // Returns false at the end of the input or on an error. `error` stays empty
  // at the end of the input.
  bool Next(std::vector<std::string>* documents, std::string* error) override;

 private:
  ParquetReader();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Opens the parquet dataset as a `DocumentSource`. Returns null and sets
// `error` on failure. This is the factory body for `DataLoader` document mode,
// so `train_main` and `eval_main` do not build the reader options themselves.
std::unique_ptr<DocumentSource> OpenParquetSource(
    const std::vector<std::string>& files, const std::string& text_column,
    std::size_t batch_size, std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_PARQUET_READER_H_
