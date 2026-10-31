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
//
// The `Recipe` block declared below (design decision D7 of
// docs/sft-parity-plan.md) is the first field appended after that list was
// frozen, so it is the worked example of rule 1: a file written before it reads
// with the member defaults, and a build older than it counts the words it does
// not know and still loads the file.

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

// The training recipe a checkpoint carries (design decision D7 of
// docs/sft-parity-plan.md): the C++ equivalent of the reference `user_config`,
// which the reference SFT run reads from its base checkpoint so the finetune
// uses the batch shape and the learning rates the base run was trained with.
//
// `Recipe` is not the model configuration. `Config` and `ConfigBlock` above
// stay the one declaration of the model shape (open decision O4): layers,
// heads, widths, vocabulary, sequence length. `Recipe` carries only the run
// settings a finetune inherits, and nothing that describes the model.
//
// The block is additive, and that is why it is a sized nested block visited
// last in `CheckpointMetadata::VisitFields`. A file that predates it carries no
// block at all: the reader counts one missing field, every member below keeps
// its initializer, and the read is not an error. A block a newer release
// appended to is counted and skipped like any other nested block. That
// compatibility logic belongs to the visitor (`src/fields.cc`), so this header
// declares the block and the two cases need no code of their own.
//
// The field order is the format (rule 1): append a new member last, and never
// reorder, rename, or remove one. The order here follows the bullet list of
// D7, so a reviewer can read the two side by side.
//
// The members carry the stream's own types, `std::int32_t` for a count and
// `float` for a rate, so `VisitFields` visits each member directly and the word
// an older reader skips is the word this build wrote. Unlike `Config`, the
// field order lives here rather than in `src/`: `Config` is also visited by the
// word-count checks in `src/train.cc` and `src/data_test.cc` (open decision
// O4), and no second translation unit visits a `Recipe`.
//
// A member whose default is not a rate has a meaning of its own, so a reader
// can tell "the file recorded nothing" from "the file recorded this value":
// - `device_batch_size` and `total_batch_size` default to 0, which means the
//   recipe was not recorded. A reader then keeps its own flags instead of
//   adopting a batch shape the file never stated, and the SFT driver derives
//   `grad_accum = total_batch_size / (device_batch_size * max_seq_len *
//   world_size)` from a recorded pair only (section 5.1 of the plan).
// - `mmlu_epochs` and `gsm8k_epochs` default to the reference mixture weights,
//   3 and 4 (`MMLU_EPOCHS` and `GSM8K_EPOCHS` of
//   python/nanochat_cpp/sft_data.py, design decision D8).
//
// Every other default repeats the initializer of the struct the SFT driver
// would otherwise build: `OptimizerConfig` for the four rates, `adam_eps`, and
// `init_lr_frac`; `SchedulerConfig` for `warmup_ratio`, `warmdown_ratio`,
// `final_lr_frac`, and the four Muon momentum fields (whose
// `muon_momentum_steps` is that struct's `muon_momentum_warmup_steps`). A
// checkpoint without a recipe therefore yields exactly the values this build
// used before D7, and a file that carries one is read as written.
struct Recipe : public IFields {
  const char* Name() const override { return "Recipe"; }

  void VisitFields(IFieldsVisitor& visitor) override {
    visitor(device_batch_size);
    visitor(total_batch_size);
    visitor(embedding_lr);
    visitor(unembedding_lr);
    visitor(matrix_lr);
    visitor(scalar_lr);
    visitor(adam_eps);
    visitor(init_lr_frac);
    visitor(warmup_ratio);
    visitor(warmdown_ratio);
    visitor(final_lr_frac);
    visitor(muon_momentum_steps);
    visitor(muon_momentum_start);
    visitor(muon_momentum_peak);
    visitor(muon_momentum_final);
    visitor(mmlu_epochs);
    visitor(gsm8k_epochs);
  }

  // 0 records no batch shape; see the comment above.
  std::int32_t device_batch_size = 0;
  std::int32_t total_batch_size = 0;
  // The per-group rates, before `init_lr_frac` (the table of section 5.2).
  float embedding_lr = 0.2f;
  float unembedding_lr = 0.004f;
  float matrix_lr = 0.02f;
  float scalar_lr = 0.5f;
  float adam_eps = 1e-10f;
  // The schedule the run follows (design decisions D1 and D3).
  float init_lr_frac = 1.0f;
  float warmup_ratio = 0.0f;
  float warmdown_ratio = 0.65f;
  float final_lr_frac = 0.0f;
  float muon_momentum_steps = 400.0f;
  float muon_momentum_start = 0.85f;
  float muon_momentum_peak = 0.97f;
  float muon_momentum_final = 0.90f;
  // The mixture weights (design decision D8).
  std::int32_t mmlu_epochs = 3;
  std::int32_t gsm8k_epochs = 4;
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
    // The training recipe, appended last (design decision D7). A file that
    // predates it carries no block, and the reader keeps every `Recipe`
    // initializer.
    visitor(recipe);
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
  // The inherited run settings of design decision D7: the C++ equivalent of the
  // reference `user_config`. `config` above stays the model configuration.
  Recipe recipe;
};

}  // namespace nanochat

#endif  // NANOCHAT_CHECKPOINT_METADATA_H_
