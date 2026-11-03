#ifndef NANOCHAT_SRC_DATA_CURSOR_H_
#define NANOCHAT_SRC_DATA_CURSOR_H_

#include <string>

#include "nanochat/dataloader.h"

// The data-cursor sidecar (docs/distributed-design.md section 11). The cursor
// lives beside the checkpoint, not inside it, so the parameter and optimizer
// records stay byte-identical across ranks and the smoke test keeps its hash
// check. The file holds the epoch, the source position, and the encoded, not
// yet packed buffer. The buffer is part of the cursor, because the reference
// best-fit packing reorders documents inside it.
//
// Both distribution drivers resume through this unit: the multi-process
// `TrainLoop` and the in-process `MeshTrainDriver`
// (docs/api-surface.md R13). It is the only copy of the layout.
//
// This header is private to `src/`.

namespace nanochat {

// The suffix of the sidecar file. The sidecar of `checkpoint_path` is
// `checkpoint_path + kDataCursorSuffix`, so a checkpoint alone is a complete
// record and the main file keeps its format (docs/checkpoint-format.md).
inline constexpr char kDataCursorSuffix[] = ".cursor";

// The sidecar path of `checkpoint_path`.
std::string DataCursorPath(const std::string& checkpoint_path);

// Writes `state` to the sidecar of `checkpoint_path`. Returns false on an empty
// path or a failed write.
bool WriteDataCursor(const std::string& checkpoint_path,
                     const DataLoaderState& state);

// Reads the sidecar of `checkpoint_path` into `*state`. Returns false when the
// sidecar is missing, truncated, or carries an implausible count. On false the
// caller keeps `*state` as a fresh cursor, so it resumes from the start of the
// stream instead of failing the run.
bool ReadDataCursor(const std::string& checkpoint_path, DataLoaderState* state);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_DATA_CURSOR_H_
