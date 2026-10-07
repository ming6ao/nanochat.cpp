#ifndef NANOCHAT_DATA_H_
#define NANOCHAT_DATA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
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
  void Clear() {
    tensors_.clear();
    optimizer_state_.clear();
  }

  // Optimizer-state records (docs/post-training.md section 7). The AdamW
  // moments and the Muon buffers use the same self-describing record layout,
  // but they live in this separate list. A loader can then tell a parameter
  // from a state buffer, because no two records share a name. `Save` writes the
  // list after the parameter records. A parameter-only NCHKPT01 file holds an
  // empty list and stays loadable.
  const std::vector<TensorRecord>& optimizer_state() const {
    return optimizer_state_;
  }
  std::vector<TensorRecord>& optimizer_state() { return optimizer_state_; }

  // Finds an optimizer-state record by name. Returns null when the file holds
  // no such record.
  const TensorRecord* FindOptimizerState(std::string_view name) const {
    for (const TensorRecord& record : optimizer_state_) {
      if (record.name == name) return &record;
    }
    return nullptr;
  }
  TensorRecord* FindOptimizerState(std::string_view name) {
    for (TensorRecord& record : optimizer_state_) {
      if (record.name == name) return &record;
    }
    return nullptr;
  }

  void AddOptimizerState(TensorRecord record) {
    optimizer_state_.push_back(std::move(record));
  }
  void ClearOptimizerState() { optimizer_state_.clear(); }

 private:
  std::vector<TensorRecord> tensors_;
  std::vector<TensorRecord> optimizer_state_;
};

}  // namespace nanochat

#endif  // NANOCHAT_DATA_H_
