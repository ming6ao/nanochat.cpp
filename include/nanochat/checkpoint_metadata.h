#ifndef NANOCHAT_CHECKPOINT_METADATA_H_
#define NANOCHAT_CHECKPOINT_METADATA_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/fields.h"
#include "nanochat/mfu.h"

// The run metadata block of the checkpoint container: one `IFields` block that
// sits between the header and the parameter count of a version 1 file
// (docs/checkpoint-metadata-and-examples-plan.md sections 2.1, 3, and 4). The
// container writes the block's word count as one `uint32` before the words, so
// a reader bounds the block without parsing it and a block above
// `kMaxBlockWords` is a hard error before any field is visited.
//
// Host-only and vendor-free, like `nanochat/fields.h`. No kernel header
// includes this one, and it includes no backend or vendor header.
//
// The rules of the block. They never change, because a checkpoint that already
// sits on the Hub has to keep loading:
// 1. The field order in `CheckpointMetadata::VisitFields` never changes, and a
//    new field is appended last. An older reader then stops at the end of the
//    block, counts the fields it did not see, and keeps the member initializer;
//    a newer reader counts the trailing words instead. Both directions are
//    accepted, and both counters reach the caller (design decisions D5 and D6).
// 2. An enum value never changes. A new state becomes a new field, never an
//    enumerator inserted into an existing enum, because an older reader rejects
//    a value its `EnumValid` does not know.
// 3. A string caps at `kMaxStringWords` (4096) words, a `std::vector` at
//    `kMaxVectorEntries` (1024) entries, and the whole block at 64 KiB
//    (`kMaxBlockWords`). A larger value is a hard error, in the same way as the
//    container's `kMaxAttributes = 64`, so the block cannot grow into a second
//    config file.
// 4. `created_utc` and `git_revision` are informational only. No load decision
//    may depend on them, so a re-save with a new timestamp or a new revision
//    cannot change how a file loads.
// 5. A bad value is a hard error: no partial load, and `Checkpoint::Load`
//    returns false with the field that failed in its error string.

namespace nanochat {

// The metadata schema version, not the file version. It moves only when the
// meaning of a field moves, which rule 1 forbids, so it stays 1.
inline constexpr std::uint32_t kMetadataSchema = 1;

// The training stage a checkpoint came from (open decision O3). The values
// match the notebook vocabulary ("base", "sft", "rl"), and the enumerator order
// is the stored value, so it never changes. There is no `kPretrain`.
enum class Stage {
  kBase = 0,
  kSft = 1,
  kRl = 2,
};

// The visitor's range check for a `Stage`. Every enum a metadata block carries
// declares one of these in namespace `nanochat`, so
// `IFieldsVisitor::operator()(EnumT&)` finds it by argument-dependent lookup.
// An unknown value is a hard error rather than a clamped one.
bool EnumValid(Stage stage);

// The same check for the recorded precision. O2 reuses `ComputePrecision` from
// `nanochat/mfu.h` instead of adding a third precision enum, so the metadata
// records the precision a report row already names.
bool EnumValid(ComputePrecision precision);

// Visits the run config of a checkpoint (open decision O4). `Config` stays the
// one declaration of the model shape; there is no `ConfigFields` mirror, so no
// field is duplicated and a drift test is unnecessary. `src/` implements this
// one function, and a new `Config` field is appended there last, inside the
// config block's own size prefix.
void VisitConfig(Config& config, IFieldsVisitor& visitor);

// The config of a metadata block, as a sized nested block: the visitor writes
// one size word and then the `VisitConfig` fields, so an older reader skips a
// config field it does not know and a newer reader learns how many words it
// carried. This is a temporary built on the stack of `VisitFields`, never a
// member, so a copy of `CheckpointMetadata` copies the config and not a pointer
// to it.
struct ConfigBlock : public IFields {
  explicit ConfigBlock(Config& config) : config(&config) {}

  const char* Name() const override { return "Config"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    VisitConfig(*config, visitor);
  }

  Config* config;  // The run config. Not owned.
};

// One checkpoint's run metadata. Every member has an initializer, because a
// field the data does not carry keeps it (rule 1), and because a version 0 file
// has no metadata block at all: it reads as exactly this default value, so a
// caller never sees an unset field.
struct CheckpointMetadata : public IFields {
  const char* Name() const override { return "CheckpointMetadata"; }

  // The field order is the format (rule 1): append a new field at the end, and
  // never reorder, rename, or remove one.
  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(schema);
    visitor(stage);
    visitor(step);
    visitor(world_size);
    visitor(precision);
    ConfigBlock config_block(config);
    visitor(config_block);
    visitor(tokenizer_id);
    visitor(git_revision);
    visitor(created_utc);
    visitor(notes);
  }

  std::uint32_t schema = kMetadataSchema;
  Stage stage = Stage::kBase;
  // Two words, low word first (open decision O6, and the word stream comment in
  // `nanochat/fields.h`). The full 64-bit range is kept: a long pretraining run
  // passes `2^32 - 1` steps, and a capped field would silently lose the step
  // that a resume schedule needs.
  std::uint64_t step = 0;
  std::uint32_t world_size = 1;
  // The recorded precision is informational. It must never change a load
  // decision, and the default is fixed rather than `BuildComputePrecision()`,
  // because a version 0 file has to report the same metadata under every
  // `--config` build (the precision gate of section 8).
  ComputePrecision precision = ComputePrecision::kFp32;
  Config config;
  std::string tokenizer_id;  // The tokenizer artifact name or its hash.
  std::string git_revision;  // Informational only (rule 4).
  std::string created_utc;   // ISO-8601, informational only (rule 4).
  std::vector<std::string> notes;
};

}  // namespace nanochat

#endif  // NANOCHAT_CHECKPOINT_METADATA_H_
