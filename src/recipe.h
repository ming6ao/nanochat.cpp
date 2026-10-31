#ifndef NANOCHAT_SRC_RECIPE_H_
#define NANOCHAT_SRC_RECIPE_H_

#include <string>

#include "nanochat/checkpoint_metadata.h"

// The `Recipe` block of the checkpoint metadata (design decision D7 of
// docs/sft-parity-plan.md), as the driver sees it.
//
// The block itself, its field order, and its defaults are declared in
// `nanochat/checkpoint_metadata.h`, and the visitor in `src/fields.cc` is what
// reads it: a file that predates the block leaves every member at its
// initializer, and a block the format cannot carry is a hard error. What
// neither of them can answer is whether the values a block *did* carry form a
// run a finetune can start: the visitor holds the word stream, not the
// schedule, so it cannot know that a warmdown ratio is a fraction or that a
// recorded batch shape needs both of its sizes.
//
// That second rule is the one this header declares. `RecipeUsable` is the check
// the SFT driver runs on the recipe it read, before it inherits a single value
// from it, so a corrupt or hand-edited block fails the run with a message
// instead of producing a schedule no reference would follow. It never rejects a
// block this build wrote: every value a writer of this format produces passes.
//
// Host-only and vendor-free, like `nanochat/checkpoint_metadata.h`.

namespace nanochat {

// True when `recipe` is a recipe a run can start from. `error` names the first
// member that is not, when it is not null.
//
// The rules, in the order they are checked:
// - the batch shape is either absent or whole: `device_batch_size` and
//   `total_batch_size` are both 0 (the file recorded no recipe, and the driver
//   keeps its own flags) or both positive. Either size alone is a block half
//   written.
// - the four rates, `adam_eps`, `init_lr_frac`, `muon_momentum_steps`, and the
//   two mixture weights are not negative.
// - the four schedule fractions, `final_lr_frac`, and the three momentum values
//   are in [0, 1], because each of them is a fraction of a rate or of the run.
bool RecipeUsable(const Recipe& recipe, std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_RECIPE_H_
