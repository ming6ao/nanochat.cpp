// The model factory: the body of the two public `Model::Create` overloads
// (docs/api-surface.md R8).
//
// The train graph lives in `model.cc`; this file holds everything that turns a
// `Config` into it, so the graph file reads no spec and builds no plan of its
// own. The config validation and the compute-dtype rejection are here once
// instead of twice, and the spec lookup, the config agreement check, the plan,
// and the plan log line are here rather than beside the topology.
//
// Behaviour is unchanged: the same error strings, the same log line, the same
// plan, and the same null return on a bad config.

#include "src/model_factory.h"

#include <memory>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/logger.h"
#include "nanochat/model_spec.h"
#include "nanochat/plan.h"
#include "nanochat/tensor.h"
#include "src/model_impl.h"

namespace nanochat {
namespace {

// The ordered block list a caller's own `Config` implies.
//
// The historical `Create(config)` entry point reads no spec, so its block list
// is derived from the same two facts `Config` already carries: the window
// pattern
// ('S' entries are sliding-window blocks, and the final layer is full context,
// which is exactly `Config::window_left`), and whether value embeddings exist
// at all (the ResFormer value-gated residual). That keeps the historical
// meaning of the call while making the graph read a block list: after this, the
// topology is driven by block kinds and not by an index into a pattern string.
void BlocksFromConfig(const Config& config, std::vector<BlockSpec>* blocks) {
  blocks->clear();
  blocks->reserve(static_cast<std::size_t>(config.num_layers));
  for (int i = 0; i < config.num_layers; ++i) {
    BlockSpec block;
    const bool last = i == config.num_layers - 1;
    block.attention =
        (!last && !config.window_pattern.empty() &&
         config.window_pattern[static_cast<std::size_t>(i) %
                               config.window_pattern.size()] == 'S')
            ? AttentionKind::kSlidingWindow
            : AttentionKind::kCausal;
    block.residual =
        config.value_embedding ? ResidualKind::kBlend : ResidualKind::kPlain;
    blocks->push_back(block);
  }
}

// The description a caller's own `Config` implies, for the historical entry
// point. It names no model package, because this call read none.
ModelSpec SpecFromConfig(const Config& config) {
  ModelSpec spec;
  spec.spec_version = kModelSpecVersion;
  spec.name = "config";
  // The mechanisms this Config needs, spelled the way a spec's `requires` is: a
  // pattern with an 'S' entry needs the windowed mask, and every block needs
  // the complete primitive tier.
  spec.requires_features = kFeaturePrimitiveTierComplete;
  for (char kind : config.window_pattern) {
    if (kind == 'S') spec.requires_features |= kFeatureSlidingWindow;
  }
  if (config.value_embedding) spec.requires_features |= kFeatureValueGate;
  spec.dimensions.num_heads = config.num_heads;
  spec.dimensions.num_kv_heads = config.num_kv_heads;
  spec.dimensions.hidden_dim = config.hidden_dim;
  spec.dimensions.seq_len = config.seq_len;
  spec.dimensions.rope_base = config.rope_base;
  spec.embeddings.vocab_size = config.vocab_size;
  spec.embeddings.padded_vocab_size = config.padded_vocab_size;
  spec.embeddings.value_embedding = config.value_embedding;
  spec.blocks.clear();
  BlocksFromConfig(config, &spec.blocks);
  return spec;
}

// Field-by-field agreement between the Config a caller passed and the one the
// spec derives. The selector's contract is that the two describe one model: a
// caller that passes a `Config` disagreeing with its spec is asking for two
// different models, and the error names the first field that differs.
bool ConfigsAgree(const Config& given, const Config& derived,
                  std::string* error) {
  const auto disagree = [error](const char* field, const std::string& a,
                                const std::string& b) {
    if (error != nullptr) {
      *error = std::string("the spec derives a Config whose ") + field +
               " is " + a + ", but the caller passed " + b;
    }
    return false;
  };
  if (given.num_layers != derived.num_layers) {
    return disagree("num_layers", std::to_string(derived.num_layers),
                    std::to_string(given.num_layers));
  }
  if (given.num_heads != derived.num_heads) {
    return disagree("num_heads", std::to_string(derived.num_heads),
                    std::to_string(given.num_heads));
  }
  if (given.num_kv_heads != derived.num_kv_heads) {
    return disagree("num_kv_heads", std::to_string(derived.num_kv_heads),
                    std::to_string(given.num_kv_heads));
  }
  if (given.hidden_dim != derived.hidden_dim) {
    return disagree("hidden_dim", std::to_string(derived.hidden_dim),
                    std::to_string(given.hidden_dim));
  }
  if (given.seq_len != derived.seq_len) {
    return disagree("seq_len", std::to_string(derived.seq_len),
                    std::to_string(given.seq_len));
  }
  if (given.vocab_size != derived.vocab_size) {
    return disagree("vocab_size", std::to_string(derived.vocab_size),
                    std::to_string(given.vocab_size));
  }
  if (given.padded_vocab_size != derived.padded_vocab_size) {
    return disagree("padded_vocab_size",
                    std::to_string(derived.padded_vocab_size),
                    std::to_string(given.padded_vocab_size));
  }
  if (given.rope_base != derived.rope_base) {
    return disagree("rope_base", std::to_string(derived.rope_base),
                    std::to_string(given.rope_base));
  }
  if (given.window_pattern != derived.window_pattern) {
    return disagree("window_pattern", "\"" + derived.window_pattern + "\"",
                    "\"" + given.window_pattern + "\"");
  }
  if (given.value_embedding != derived.value_embedding) {
    return disagree("value_embedding",
                    derived.value_embedding ? "true" : "false",
                    given.value_embedding ? "true" : "false");
  }
  return true;
}

// The one line the model logs for its plan. `tools/nanochat doctor` prints the
// full report from `//src:plan_main`; this is the same decision in the log a
// training run already writes, so a user sees which implementation each
// composition resolved to without reading source.
std::string DescribePlan(const Plan& plan) {
  std::string line = std::string("plan: spec \"") + plan.spec_name +
                     "\", graph " + GraphKindName(plan.graph) + ", device " +
                     std::to_string(plan.device_index);
  for (const CompositionChoice& choice : plan.choices) {
    line += std::string("\n  ") + CompositionName(choice.composition) + ": " +
            choice.implementation + (choice.fused ? " (fused)" : "");
    for (const DeclinedImpl& declined : choice.declined) {
      line += "\n      declined " + declined.implementation + ": " +
              declined.reason;
    }
  }
  line +=
      "\n  (this Plan is the device-and-graph plan; it is unrelated to "
      "python/nanochat_cpp/plan.py, the training-hyperparameter planner)";
  return line;
}

}  // namespace

std::unique_ptr<Model> CreateTrainModel(const Config& config, SpecSource source,
                                        const char* spec_name,
                                        std::string* error) {
  // The validation both entry points share, and the compute type the linked
  // backend must actually support. These lines were duplicated in the two
  // public overloads; they are written once here.
  if (!config.Validate(error)) return nullptr;
  const Caps caps = kernels::GetCaps();
  if (!caps.Supports(kComputeDType)) {
    if (error != nullptr) {
      *error =
          std::string("compute dtype is not supported by ") + caps.device_name;
    }
    return nullptr;
  }

  if (source == SpecSource::kConfigOnly) {
    // No spec is read on this path, so the caller's `Config` stays the
    // authority and the plan is plan-less: the block list is the one that
    // Config implies.
    return std::make_unique<TrainModel>(SpecFromConfig(config), config, Plan());
  }

  // 1. The description. A null name selects the built-in default, so a caller
  // with no opinion about the model definition keeps the historical behaviour.
  // (`kByName` always reads a spec: a null name is the default one, not the
  // absence of a spec.)
  ModelSpec spec;
  std::string why;
  if (!LookupModelSpec(spec_name, &spec, &why)) {
    if (error != nullptr) *error = why;
    return nullptr;
  }

  // 2. The derived struct, and the check that the caller and the spec describe
  // one model.
  Config derived;
  if (!ModelSpecToConfig(spec, &derived, &why)) {
    if (error != nullptr) {
      *error = "spec \"" + spec.name + "\" is invalid: " + why;
    }
    return nullptr;
  }
  if (!ConfigsAgree(config, derived, &why)) {
    if (error != nullptr) {
      *error =
          "spec \"" + spec.name + "\" disagrees with the given Config: " + why;
    }
    return nullptr;
  }

  // 3. The plan, built once (docs/model-and-device-bringup.md section 9.1 rule
  // 2: one caller at first, no entry point changes). Nothing re-decides at
  // launch, and a rejection is the caller's answer rather than a warning.
  PlanRequest request;
  request.model = &spec;
  request.graph = GraphKind::kTrain;
  request.device_index = caps.device_index;
  std::string why_not;
  const Plan plan = BuildPlan(request, &why_not);
  if (!plan.ok) {
    if (error != nullptr) *error = why_not;
    return nullptr;
  }

  // 4. The choice is observable: one log line per plan, through the logger the
  // run already writes. (This Plan is unrelated to
  // `python/nanochat_cpp/plan.py`.)
  Logger logger;
  logger.Info(DescribePlan(plan));

  return std::make_unique<TrainModel>(spec, derived, plan);
}

}  // namespace nanochat
