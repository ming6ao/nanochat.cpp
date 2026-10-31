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

#include <cmath>
#include <cstdint>
#include <string>

#include "nanochat/config.h"
#include "nanochat/fields.h"
#include "nanochat/mfu.h"
#include "src/recipe.h"

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

// The `Recipe` block's usability rule (design decision D7 of
// docs/sft-parity-plan.md). The visitor above is the only code that touches the
// block's words, and it is deliberately word-shape-only: it cannot know that a
// ratio is a fraction of the run. This function is the second half of "reject a
// malformed block" -- the half that reads like the schedule -- and the SFT
// driver runs it on the recipe it inherited before it uses any of it.
bool RecipeUsable(const Recipe& recipe, std::string* error) {
  auto fail = [error](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };

  // 0 and 0 mean "no batch shape was recorded" (the header's sentinel, so a
  // reader keeps its own flags). One size without the other is a block that was
  // not written by this format: the accumulation of section 5.1 needs both.
  const bool has_device = recipe.device_batch_size > 0;
  const bool has_total = recipe.total_batch_size > 0;
  if (recipe.device_batch_size < 0 || recipe.total_batch_size < 0) {
    return fail("the recipe records a negative batch size");
  }
  if (has_device != has_total) {
    return fail(
        "the recipe records one of device_batch_size and total_batch_size, not "
        "both");
  }

  const struct {
    const char* name;
    float value;
  } kNonNegative[] = {
      {"embedding_lr", recipe.embedding_lr},
      {"unembedding_lr", recipe.unembedding_lr},
      {"matrix_lr", recipe.matrix_lr},
      {"scalar_lr", recipe.scalar_lr},
      {"adam_eps", recipe.adam_eps},
      {"muon_momentum_steps", recipe.muon_momentum_steps},
  };
  for (const auto& field : kNonNegative) {
    if (field.value < 0.0f) {
      if (error != nullptr) {
        *error = std::string("the recipe records a negative ") + field.name;
      }
      return false;
    }
  }
  if (recipe.mmlu_epochs < 0 || recipe.gsm8k_epochs < 0) {
    return fail("the recipe records a negative mixture weight");
  }

  const struct {
    const char* name;
    float value;
  } kFractions[] = {
      {"init_lr_frac", recipe.init_lr_frac},
      {"warmup_ratio", recipe.warmup_ratio},
      {"warmdown_ratio", recipe.warmdown_ratio},
      {"final_lr_frac", recipe.final_lr_frac},
      {"muon_momentum_start", recipe.muon_momentum_start},
      {"muon_momentum_peak", recipe.muon_momentum_peak},
      {"muon_momentum_final", recipe.muon_momentum_final},
  };
  for (const auto& field : kFractions) {
    // The visitor already rejected a NaN, so this is the plain range test; a
    // fraction outside [0, 1] is not something the schedule can use.
    if (!(field.value >= 0.0f && field.value <= 1.0f)) {
      if (error != nullptr) {
        *error = std::string("the recipe records a ") + field.name +
                 " outside [0, 1]";
      }
      return false;
    }
  }
  return true;
}

}  // namespace nanochat
