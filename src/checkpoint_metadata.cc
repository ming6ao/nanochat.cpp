// src/checkpoint_metadata.cc -- the run metadata block of the checkpoint
// container (docs/checkpoint-metadata-and-examples-plan.md sections 3 and 4).
//
// The block itself, its field order, and its rules are declared in
// `nanochat/checkpoint_metadata.h`, which is the one declaration of
// `VisitConfig` (open decision O4). This file carries what that header cannot:
// the run config's field order in the word stream, and the range check each
// metadata enum promises.
//
// Host-only and vendor-free: `nanochat/fields.h` is plain scalar code, and
// neither `Config` nor `ComputePrecision` reaches a kernel.

#include "nanochat/checkpoint_metadata.h"

#include <cstdint>

#include "nanochat/config.h"
#include "nanochat/fields.h"
#include "nanochat/mfu.h"

namespace nanochat {
namespace {

// `Config` holds `int`, and the word stream holds `std::int32_t`. On every
// supported platform those are the same type, but the visit goes through an
// explicit `std::int32_t` so the stream's field type is the one the visitor
// sees, and the member is written back only when the value changed (a missing
// field leaves the initializer in place).
void VisitInt(int& value, IFieldsVisitor& visitor) {
  std::int32_t stored = static_cast<std::int32_t>(value);
  visitor(stored);
  value = static_cast<int>(stored);
}

}  // namespace

// The field order is the format (rule 1 in `nanochat/checkpoint_metadata.h`):
// a new `Config` field is appended last, inside the config block's own size
// prefix. The order here is the order of the fields in
// `include/nanochat/config.h`, so a reviewer can read both side by side, and
// nothing else lists the fields: there is no `ConfigFields` mirror and no drift
// test (open decision O4).
void VisitConfig(Config& config, IFieldsVisitor& visitor) {
  VisitInt(config.num_layers, visitor);
  VisitInt(config.num_heads, visitor);
  VisitInt(config.num_kv_heads, visitor);
  VisitInt(config.hidden_dim, visitor);
  VisitInt(config.seq_len, visitor);
  VisitInt(config.vocab_size, visitor);
  VisitInt(config.padded_vocab_size, visitor);
  visitor(config.rope_base);
  visitor(config.window_pattern);
  visitor(config.value_embedding);
}

// The enumerator order is the stored value, so a new state becomes a new field
// and never an inserted enumerator (rule 2). An unknown value is a hard error
// rather than a clamped one, because a load decision could depend on it.
bool EnumValid(Stage stage) {
  switch (stage) {
    case Stage::kBase:
    case Stage::kSft:
    case Stage::kRl:
      return true;
  }
  return false;
}

// The recorded precision is informational (rule 4), but it is still a range
// check: a word that names no precision means the stream is not what this build
// wrote.
bool EnumValid(ComputePrecision precision) {
  switch (precision) {
    case ComputePrecision::kFp32:
    case ComputePrecision::kBf16:
    case ComputePrecision::kFp8:
      return true;
  }
  return false;
}

}  // namespace nanochat
