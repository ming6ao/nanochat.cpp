#ifndef NANOCHAT_SRC_SFT_SHARD_H_
#define NANOCHAT_SRC_SFT_SHARD_H_

#include <string>
#include <vector>

// The `NCSFT001` SFT shard (docs/training-seam.md section 6.2). The file is a
// typed-record file with two int32 records, `tokens` and `targets`, each of
// shape `[rows, seq]`. A target of -1 marks an ignored position. Python writes
// it with `nanochat_cpp.sft.pack`, and `src/record_file.{h,cc}` owns the
// layout.
//
// This header holds the typed view only. The mesh training driver and
// `sft_main` share it, so neither copies the record extraction.

namespace nanochat {

struct SftShard {
  int rows = 0;
  int seq = 0;
  std::vector<int> tokens;   // rows * seq
  std::vector<int> targets;  // rows * seq
};

// Loads the shard into memory. Returns false and writes a reason to `*error` on
// a missing file, a bad magic, or a bad record shape.
bool LoadSftShard(const std::string& path, SftShard* shard, std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_SFT_SHARD_H_
