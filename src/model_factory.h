#ifndef NANOCHAT_SRC_MODEL_FACTORY_H_
#define NANOCHAT_SRC_MODEL_FACTORY_H_

#include <memory>
#include <string>

#include "nanochat/config.h"
#include "nanochat/model.h"

// The body of the two public `Model::Create` overloads (spec/api-surface.md
// R8). `src/model.cc` holds the train graph; this is the factory that turns a
// `Config` into one. The config validation and the compute-dtype rejection
// live here once, and so do the spec lookup, the config agreement check, the
// plan, and the plan log line, so the graph file reads no spec and builds no
// plan of its own.
//
// This header is private to `src/`: the rest of the tree calls `Model::Create`.

namespace nanochat {

// Which of the two public entry points is asking. They differ in one way only
// (`include/nanochat/model.h`): `Create(config)` reads no spec at all, and
// `Create(config, spec_name)` reads one, where a *null* name is still a read
// because it selects the built-in default spec. A nullable name alone cannot
// tell those three states apart, so the caller names the path it is on.
enum class SpecSource {
  // No spec is read. The block list is the one the caller's `Config` implies,
  // the plan is plan-less, and no plan line is logged. This is the historical
  // `Create(config)`.
  kConfigOnly,
  // A spec is read. `spec_name` may be null, which selects the built-in
  // default spec, so this path always builds a plan and logs it.
  kByName,
};

// Builds the training model for `config`. Returns null on any rejection and
// writes the reason to `error` when it is not null: the `Config` validation
// comes first and is the same for both sources, then the compute-dtype check,
// then - for `kByName` - the spec lookup, the agreement check, and the plan.
std::unique_ptr<Model> CreateTrainModel(const Config& config, SpecSource source,
                                        const char* spec_name,
                                        std::string* error);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MODEL_FACTORY_H_
