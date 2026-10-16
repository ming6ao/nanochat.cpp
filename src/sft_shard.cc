#include "src/sft_shard.h"

#include <cstdint>

#include "src/record_file.h"

namespace nanochat {

bool LoadSftShard(const std::string& path, SftShard* shard,
                  std::string* error) {
  std::vector<Record> records;
  if (!LoadRecordFile(path, {"NCSFT001"}, &records, error)) return false;
  const Record* tokens = FindRecord(records, "tokens");
  const Record* targets = FindRecord(records, "targets");
  if (tokens == nullptr || targets == nullptr || tokens->shape.size() != 2 ||
      targets->shape.size() != 2) {
    *error = "the SFT shard must hold two rank-2 records, tokens and targets";
    return false;
  }
  if (tokens->shape != targets->shape) {
    *error = "the SFT shard records have different shapes";
    return false;
  }
  const std::int64_t rows = static_cast<std::int64_t>(tokens->shape[0]);
  const std::int64_t seq = static_cast<std::int64_t>(tokens->shape[1]);
  if (rows < 1 || seq < 1) {
    *error = "the SFT shard has a nonpositive shape";
    return false;
  }
  const std::int64_t numel = rows * seq;
  if (!ReadRecordInt32Vector(records, "tokens", numel, &shard->tokens, error) ||
      !ReadRecordInt32Vector(records, "targets", numel, &shard->targets,
                             error)) {
    return false;
  }
  shard->rows = static_cast<int>(rows);
  shard->seq = static_cast<int>(seq);
  return true;
}

}  // namespace nanochat
