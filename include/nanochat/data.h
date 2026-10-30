#ifndef NANOCHAT_DATA_H_
#define NANOCHAT_DATA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/checkpoint_metadata.h"
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

// One attribute of a checkpoint record (docs/precision.md section 5). The
// container reserves the block for int8: the `quant.scale` attribute carries
// the scale. An fp32 or bf16 record carries no attribute, and int8 is not
// implemented yet, so the block is empty in practice.
struct TensorAttribute {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;
};

struct TensorRecord {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;

  // The optional attribute block. Empty for an fp32 or bf16 record.
  std::vector<TensorAttribute> attributes;

  std::int64_t numel() const {
    std::int64_t total = 1;
    for (std::int64_t dim : shape) total *= dim;
    return total;
  }
};

// The format of a checkpoint file, so a reader routes on one value instead of a
// bool (docs/checkpoint-metadata-and-examples-plan.md section 2.2). The magic
// names the layout family and the `version` word names the layout inside it, so
// a mismatched pair is a hard error. Under O5 the magic stays `NCHKPT01` and
// only the version word moves.
enum class CheckpointFormat {
  // The legacy `NCMDL001` model file, and anything else that does not start
  // with the container magic. The legacy read reports a bad magic itself.
  kLegacyModel,
  kContainerV0,
  kContainerV1,
};

// True when `path` starts with the `NCHKPT01` magic. `Checkpointer::LoadModel`
// uses this to route a file to the container reader or the legacy `NCMDL001`
// reader (docs/model.md).
bool IsCheckpointContainer(const std::string& path);

// The format of `path`: the container magic plus the version word for
// `kContainerV0` and `kContainerV1`, and `kLegacyModel` for everything else,
// including a legacy file and a path that cannot be read. It is a total
// function, so one call covers the routing: a version word above 1 reads as
// `kContainerV1` here and is then rejected by the reader as an unsupported
// checkpoint version, which is the hard error O5 asks for.
CheckpointFormat DetectFormat(const std::string& path);

class Checkpoint {
 public:
  // Reads a container. `with_optimizer_state` selects the read mode
  // (docs/checkpoint-format.md section 9). A true value reads the parameter
  // records and the optimizer section. A false value reads the parameter
  // records only and stops before the optimizer section, so the read does not
  // parse or allocate the optimizer records. The optimizer section is last, so
  // the stop is cheap. A parameter-only read leaves `optimizer_state()` empty.
  // Both container versions are accepted: a version 1 file also fills
  // `metadata()`, and a version 0 file leaves it at the defaults. A bad value
  // anywhere in the file is a hard error, and then `error` names the field.
  bool Load(const std::string& path, std::string* error = nullptr,
            bool with_optimizer_state = true);
  // Writes a version 1 container: the header, the metadata block, the parameter
  // records, and the optimizer records (docs/checkpoint-format.md section 2.1).
  // The record framing stays byte identical to version 0. A metadata value the
  // format cannot carry is a hard error, so `Save` never writes a partial
  // block.
  bool Save(const std::string& path) const;

  const std::vector<TensorRecord>& tensors() const { return tensors_; }
  std::vector<TensorRecord>& tensors() { return tensors_; }

  const TensorRecord* Find(std::string_view name) const;
  TensorRecord* Find(std::string_view name);

  void Add(TensorRecord record);
  void Clear() {
    tensors_.clear();
    optimizer_state_.clear();
    metadata_ = CheckpointMetadata();
  }

  // The run metadata block (docs/checkpoint-metadata-and-examples-plan.md
  // sections 2.1 and 4). `Save` writes it between the header and the parameter
  // count, and `Load` reads it. A version 0 file carries no metadata section,
  // so it reads as the member initializers of `CheckpointMetadata` and every
  // field is still defined. `Clear` resets it to those defaults.
  CheckpointMetadata metadata() const { return metadata_; }
  void set_metadata(CheckpointMetadata metadata) {
    metadata_ = std::move(metadata);
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
  // Defaults until a version 1 file or a save fills it.
  CheckpointMetadata metadata_;
};

}  // namespace nanochat

#endif  // NANOCHAT_DATA_H_
