#ifndef NANOCHAT_DATA_H_
#define NANOCHAT_DATA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "nanochat/tensor.h"

// The self-describing checkpoint container (docs/model.md). Training data
// arrives as parquet documents through the native reader
// (docs/parquet-native.md); no pre-tokenized shard and no numpy is needed at
// runtime.

namespace nanochat {

// ---------------------------------------------------------------------------
// Checkpoint container
// ---------------------------------------------------------------------------
//
// A self-describing container: `name`, dtype, shape, and raw payload per
// tensor. Not a framework `state_dict`.

struct TensorRecord {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;

  std::int64_t numel() const {
    std::int64_t total = 1;
    for (std::int64_t dim : shape) total *= dim;
    return total;
  }
};

class Checkpoint {
 public:
  bool Load(const std::string& path, std::string* error = nullptr);
  bool Save(const std::string& path) const;

  const std::vector<TensorRecord>& tensors() const { return tensors_; }
  std::vector<TensorRecord>& tensors() { return tensors_; }

  const TensorRecord* Find(std::string_view name) const;
  TensorRecord* Find(std::string_view name);

  void Add(TensorRecord record);
  void Clear() { tensors_.clear(); }

 private:
  std::vector<TensorRecord> tensors_;
};

}  // namespace nanochat

#endif  // NANOCHAT_DATA_H_
