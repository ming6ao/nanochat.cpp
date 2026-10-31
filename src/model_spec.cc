// The spec reader and the derivation into `Config`
// (`include/nanochat/model_spec.h`).
//
// Three rules drive this file, and each of them is a rejection rather than a
// default:
//
//   * The field set is closed. An unknown key at any level is an error naming
//     the key, because a spec is a description of what a model *is*, and a
//     field the reader does not implement is a meaning the graph would silently
//     not run (`spec/versions.md` section 3.1).
//   * `spec_version` is exact. A file that declares a version this reader does
//     not implement is rejected by number instead of half-parsed, so a newer
//     package cannot be read as an older one.
//   * A block kind is closed in both directions. A kind name that does not
//     appear in the header's vocabulary is rejected with its name; it is never
//     rounded to the nearest kind the reader knows
//     (`spec/block_kinds.md` section 1).
//
// `ModelSpecToConfig` then derives the runtime struct. `Config` stays the
// derived struct: it keeps `head_dim()`, `mlp_dim()`, `window_left()` and
// `Validate()`, and the spec never duplicates them. The one field the spec
// *does* drive is `window_pattern`, built from the ordered block list, so the
// block list and the mask the graph runs cannot disagree.

#include "nanochat/model_spec.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "nanochat/kernels.h"
#include "src/capability_names.h"
#include "src/json.h"

namespace nanochat {
namespace {

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

std::string Quote(const std::string& value) { return "\"" + value + "\""; }

// ---------------------------------------------------------------------------
// Field access
// ---------------------------------------------------------------------------

// The required member `name` of `object`. A missing field is an error naming the
// object and the field, so no caller dereferences a null.
const json::Value* Member(const json::Value& object, const char* where,
                          const char* name, std::string* error) {
  const json::Value* value = object.Find(name);
  if (value == nullptr) {
    Fail(error, std::string(where) + " has no \"" + name + "\" field");
  }
  return value;
}

// Rejects any key of `object` that is not in `allowed`. The field set is
// closed, so this runs on every object this reader reads.
bool RejectUnknownKeys(const json::Value& object, const char* where,
                       const std::vector<const char*>& allowed,
                       std::string* error) {
  for (const auto& entry : object.object()) {
    bool known = false;
    for (const char* name : allowed) {
      if (entry.first == name) {
        known = true;
        break;
      }
    }
    if (!known) {
      std::string spelled;
      for (const char* name : allowed) {
        if (!spelled.empty()) spelled += ", ";
        spelled += name;
      }
      return Fail(error, std::string(where) + " has an unknown field " +
                            Quote(entry.first) +
                            "; the field set is closed and this reader "
                            "implements: " + spelled);
    }
  }
  return true;
}

bool WantObject(const json::Value& value, const char* where,
                std::string* error) {
  if (value.is_object()) return true;
  return Fail(error, std::string(where) + " must be an object, not " +
                        value.KindName());
}

bool WantString(const json::Value& value, const char* where, std::string* out,
                std::string* error) {
  if (!value.is_string()) {
    return Fail(error, std::string(where) + " must be a string, not " +
                          value.KindName());
  }
  *out = value.string();
  return true;
}

bool WantInt(const json::Value& value, const char* where, int* out,
             std::string* error) {
  if (!value.Integer(out)) {
    return Fail(error, std::string(where) + " must be an integer, not " +
                          value.KindName());
  }
  return true;
}

bool WantFloat(const json::Value& value, const char* where, float* out,
               std::string* error) {
  if (!value.is_number()) {
    return Fail(error, std::string(where) + " must be a number, not " +
                          value.KindName());
  }
  *out = static_cast<float>(value.number());
  return true;
}

bool WantBool(const json::Value& value, const char* where, bool* out,
              std::string* error) {
  if (!value.is_bool()) {
    return Fail(error, std::string(where) + " must be true or false, not " +
                          value.KindName());
  }
  *out = value.boolean();
  return true;
}

template <typename Row, typename Enum, std::size_t N>
bool ParseEnum(const json::Value& value, const char* where, const char* what,
               const Row (&rows)[N], Enum* out, std::string* error) {
  std::string spelling;
  if (!WantString(value, where, &spelling, error)) return false;
  for (const Row& row : rows) {
    if (spelling == row.name) {
      *out = row.value;
      return true;
    }
  }
  std::string accepted;
  for (const Row& row : rows) {
    if (!accepted.empty()) accepted += ", ";
    accepted += row.name;
  }
  return Fail(error, std::string(where) + " names an unknown " + what + " " +
                        Quote(spelling) + "; this reader accepts: " + accepted);
}

// The typed required-field readers: one per scalar type, so a section reads as
// one line per field with no null in the middle.
bool NeedInt(const json::Value& object, const char* where, const char* name,
             int* out, std::string* error) {
  const json::Value* value = Member(object, where, name, error);
  if (value == nullptr) return false;
  return WantInt(*value, (std::string(where) + "." + name).c_str(), out, error);
}

bool NeedFloat(const json::Value& object, const char* where, const char* name,
               float* out, std::string* error) {
  const json::Value* value = Member(object, where, name, error);
  if (value == nullptr) return false;
  return WantFloat(*value, (std::string(where) + "." + name).c_str(), out,
                   error);
}

bool NeedBool(const json::Value& object, const char* where, const char* name,
              bool* out, std::string* error) {
  const json::Value* value = Member(object, where, name, error);
  if (value == nullptr) return false;
  return WantBool(*value, (std::string(where) + "." + name).c_str(), out,
                  error);
}

bool NeedString(const json::Value& object, const char* where, const char* name,
                std::string* out, std::string* error) {
  const json::Value* value = Member(object, where, name, error);
  if (value == nullptr) return false;
  return WantString(*value, (std::string(where) + "." + name).c_str(), out,
                    error);
}

// The required member `name`, which must then parse as `what`.
template <typename Row, typename Enum, std::size_t N>
bool NeedEnum(const json::Value& object, const char* where, const char* name,
              const char* what, const Row (&rows)[N], Enum* out,
              std::string* error) {
  const json::Value* value = Member(object, where, name, error);
  if (value == nullptr) return false;
  return ParseEnum(*value, (std::string(where) + "." + name).c_str(), what,
                   rows, out, error);
}

// ---------------------------------------------------------------------------
// The closed value vocabularies
// ---------------------------------------------------------------------------

// Each table is one row per enum value: the JSON spelling, the enum value. The
// spellings are the ones `models/nanochat-default/spec.json` uses, which are
// the header's enum names (`include/nanochat/model_spec.h`). `spec/block_kinds.md`
// spells some of them differently (`gqa_self_attention`, `full`/`sliding`,
// `plain_residual`/`value_gated_residual`); the header is the field-list
// authority this reader implements, and the mismatch is recorded in that
// package's README and in this campaign's node report. An unknown spelling is
// rejected with the accepted set rather than mapped onto the nearest meaning.

struct NormRow { const char* name; NormKind value; };
constexpr NormRow kNormRows[] = {
    {"rms_norm", NormKind::kRmsNorm},
};

struct AttentionRow { const char* name; AttentionKind value; };
constexpr AttentionRow kAttentionRows[] = {
    {"causal", AttentionKind::kCausal},
    {"sliding_window", AttentionKind::kSlidingWindow},
};

struct VariantRow { const char* name; AttentionVariant value; };
constexpr VariantRow kVariantRows[] = {
    {"mha", AttentionVariant::kMha},
    {"gqa", AttentionVariant::kGqa},
};

struct PositionRow { const char* name; PositionKind value; };
constexpr PositionRow kPositionRows[] = {
    {"none", PositionKind::kNone},
    {"rope", PositionKind::kRope},
};

struct MlpRow { const char* name; MlpKind value; };
constexpr MlpRow kMlpRows[] = {
    {"relu_square", MlpKind::kReluSquare},
};

struct ResidualRow { const char* name; ResidualKind value; };
constexpr ResidualRow kResidualRows[] = {
    {"plain", ResidualKind::kPlain},
    {"blend", ResidualKind::kBlend},
};

struct DTypeRow { const char* name; DTypePolicy value; };
constexpr DTypeRow kDTypeRows[] = {
    {"build_default", DTypePolicy::kBuildDefault},
    {"fp32", DTypePolicy::kFp32},
    {"bf16", DTypePolicy::kBf16},
    {"fp8", DTypePolicy::kFp8},
};

struct HeadRow { const char* name; HeadKind value; };
constexpr HeadRow kHeadRows[] = {
    {"untied_linear", HeadKind::kUntiedLinear},
};

struct LossRow { const char* name; LossKind value; };
constexpr LossRow kLossRows[] = {
    {"softcap_cross_entropy", LossKind::kSoftcapCrossEntropy},
};

// ---------------------------------------------------------------------------
// The sections
// ---------------------------------------------------------------------------

bool ParseRequires(const json::Value& value, ModelSpec* spec,
                   std::string* error) {
  const char* where = "requires";
  if (!value.is_array()) {
    return Fail(error, std::string(where) + " must be an array of strings, "
                          "not " + value.KindName());
  }
  std::uint64_t mask = 0;
  for (const json::Value& entry : value.array()) {
    std::string name;
    if (!WantString(entry, "a requires entry", &name, error)) return false;
    std::uint64_t bit = 0;
    if (!ParseCapabilityName(name, &bit)) {
      return Fail(error, "requires names " + Quote(name) +
                            ", which is not in the capability vocabulary this "
                            "build defines (spec/capability_bits.md section 3)");
    }
    if (IsOverrideCapability(bit)) {
      return Fail(error, "requires names the override bit " + Quote(name) +
                            "; a model states what its semantics need, and an "
                            "implementation choice is not a semantic "
                            "(spec/capability_bits.md section 3)");
    }
    mask |= bit;
  }
  spec->requires_features = mask;
  return true;
}

bool ParseDimensions(const json::Value& value, DimensionsSpec* out,
                     std::string* error) {
  const char* where = "dimensions";
  if (!WantObject(value, where, error)) return false;
  if (!RejectUnknownKeys(value, where,
                         {"num_heads", "num_kv_heads", "hidden_dim", "seq_len",
                          "rope_base"},
                         error)) {
    return false;
  }
  return NeedInt(value, where, "num_heads", &out->num_heads, error) &&
         NeedInt(value, where, "num_kv_heads", &out->num_kv_heads, error) &&
         NeedInt(value, where, "hidden_dim", &out->hidden_dim, error) &&
         NeedInt(value, where, "seq_len", &out->seq_len, error) &&
         NeedFloat(value, where, "rope_base", &out->rope_base, error);
}

bool ParseBlock(const json::Value& value, int index, BlockSpec* out,
                std::string* error) {
  const std::string where = "blocks[" + std::to_string(index) + "]";
  if (!WantObject(value, where.c_str(), error)) return false;
  if (!RejectUnknownKeys(value, where.c_str(),
                         {"norm", "attention", "attention_variant", "position",
                          "mlp", "residual", "dtype"},
                         error)) {
    return false;
  }
  const char* w = where.c_str();
  return NeedEnum(value, w, "norm", "norm kind", kNormRows, &out->norm, error) &&
         NeedEnum(value, w, "attention", "attention kind", kAttentionRows,
                  &out->attention, error) &&
         NeedEnum(value, w, "attention_variant", "attention variant",
                  kVariantRows, &out->attention_variant, error) &&
         NeedEnum(value, w, "position", "position kind", kPositionRows,
                  &out->position, error) &&
         NeedEnum(value, w, "mlp", "mlp kind", kMlpRows, &out->mlp, error) &&
         NeedEnum(value, w, "residual", "residual kind", kResidualRows,
                  &out->residual, error) &&
         NeedEnum(value, w, "dtype", "dtype policy", kDTypeRows, &out->dtype,
                  error);
}

bool ParseBlocks(const json::Value& value, ModelSpec* spec,
                 std::string* error) {
  if (!value.is_array()) {
    return Fail(error, std::string("blocks must be an array, not ") +
                          value.KindName());
  }
  if (value.array().empty()) {
    return Fail(error, "blocks is empty; its length *is* num_layers and the "
                       "graph needs at least one block");
  }
  spec->blocks.clear();
  spec->blocks.reserve(value.array().size());
  for (std::size_t i = 0; i < value.array().size(); ++i) {
    BlockSpec block;
    if (!ParseBlock(value.array()[i], static_cast<int>(i), &block, error)) {
      return false;
    }
    spec->blocks.push_back(block);
  }
  return true;
}

bool ParseEmbeddings(const json::Value& value, EmbeddingsSpec* out,
                     std::string* error) {
  const char* where = "embeddings";
  if (!WantObject(value, where, error)) return false;
  if (!RejectUnknownKeys(value, where,
                         {"vocab_size", "padded_vocab_size", "value_embedding"},
                         error)) {
    return false;
  }
  return NeedInt(value, where, "vocab_size", &out->vocab_size, error) &&
         NeedInt(value, where, "padded_vocab_size", &out->padded_vocab_size,
                 error) &&
         NeedBool(value, where, "value_embedding", &out->value_embedding, error);
}

bool ParseHead(const json::Value& value, HeadSpec* out, std::string* error) {
  const char* where = "head";
  if (!WantObject(value, where, error)) return false;
  if (!RejectUnknownKeys(value, where, {"kind", "logit_softcap"}, error)) {
    return false;
  }
  return NeedEnum(value, where, "kind", "head kind", kHeadRows, &out->kind,
                  error) &&
         NeedFloat(value, where, "logit_softcap", &out->logit_softcap, error);
}

bool ParseLoss(const json::Value& value, LossSpec* out, std::string* error) {
  const char* where = "loss";
  if (!WantObject(value, where, error)) return false;
  if (!RejectUnknownKeys(value, where, {"kind", "ignore_index"}, error)) {
    return false;
  }
  return NeedEnum(value, where, "kind", "loss kind", kLossRows, &out->kind,
                  error) &&
         NeedInt(value, where, "ignore_index", &out->ignore_index, error);
}

bool ParsePolicy(const json::Value& value, PolicySpec* out,
                 std::string* error) {
  const char* where = "policy";
  if (!WantObject(value, where, error)) return false;
  if (!RejectUnknownKeys(value, where,
                         {"dtype", "allow_fused_overrides",
                          "require_primitive_tier"},
                         error)) {
    return false;
  }
  return NeedEnum(value, where, "dtype", "dtype policy", kDTypeRows,
                  &out->dtype, error) &&
         NeedBool(value, where, "allow_fused_overrides",
                  &out->allow_fused_overrides, error) &&
         NeedBool(value, where, "require_primitive_tier",
                  &out->require_primitive_tier, error);
}

// The block list of the built-in default spec: `Config`'s own defaults, which
// means twelve blocks tiled `SSSL` with the ResFormer residual, rotary
// positions, and the relu^2 MLP on every block.
void FillDefaultBlocks(std::vector<BlockSpec>* blocks, int num_layers) {
  blocks->clear();
  blocks->reserve(static_cast<std::size_t>(num_layers));
  const char kPattern[] = "SSSL";
  const std::size_t pattern_len = sizeof(kPattern) - 1;
  for (int i = 0; i < num_layers; ++i) {
    BlockSpec block;
    block.attention = kPattern[static_cast<std::size_t>(i) % pattern_len] == 'S'
                          ? AttentionKind::kSlidingWindow
                          : AttentionKind::kCausal;
    blocks->push_back(block);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// The built-in default and the file lookup
// ---------------------------------------------------------------------------

const ModelSpec& DefaultModelSpec() {
  static const ModelSpec kDefault = [] {
    ModelSpec spec;
    spec.spec_version = kModelSpecVersion;
    spec.name = "default";
    spec.requires_features = kFeatureRope | kFeatureSlidingWindow |
                             kFeatureValueGate | kFeaturePrimitiveTierComplete;
    // `Config`'s own defaults: twelve blocks tiled `SSSL`.
    FillDefaultBlocks(&spec.blocks, Config().num_layers);
    return spec;
  }();
  return kDefault;
}

std::string ModelSpecPath(const std::string& name) {
  return "models/" + name + "/spec.json";
}

bool ParseModelSpec(const std::string& path, ModelSpec* out,
                    std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return Fail(error, "no output spec");

  json::Value root;
  if (!json::ParseFile(path, &root, error)) return false;
  const char* where = "spec";
  if (!WantObject(root, where, error)) return false;

  ModelSpec spec;
  if (!RejectUnknownKeys(root, where,
                         {"spec_version", "name", "requires", "dimensions",
                          "blocks", "embeddings", "head", "loss", "policy"},
                         error)) {
    return false;
  }

  int version = 0;
  if (!NeedInt(root, where, "spec_version", &version, error)) return false;
  if (version != kModelSpecVersion) {
    return Fail(error, "spec_version " + std::to_string(version) +
                           " is not this reader's version (" +
                           std::to_string(kModelSpecVersion) +
                           "); a spec format this build does not implement is "
                           "refused by number, not half-parsed");
  }
  spec.spec_version = version;

  if (!NeedString(root, where, "name", &spec.name, error)) return false;
  if (spec.name.empty()) return Fail(error, "name is empty");

  // Each required section is fetched once; a missing one is an error naming
  // it, so no section parser is called with a null.
  const json::Value* requires_value = Member(root, where, "requires", error);
  if (requires_value == nullptr) return false;
  const json::Value* dimensions = Member(root, where, "dimensions", error);
  if (dimensions == nullptr) return false;
  const json::Value* blocks = Member(root, where, "blocks", error);
  if (blocks == nullptr) return false;
  const json::Value* embeddings = Member(root, where, "embeddings", error);
  if (embeddings == nullptr) return false;
  const json::Value* head = Member(root, where, "head", error);
  if (head == nullptr) return false;
  const json::Value* loss = Member(root, where, "loss", error);
  if (loss == nullptr) return false;
  const json::Value* policy = Member(root, where, "policy", error);
  if (policy == nullptr) return false;

  if (!ParseRequires(*requires_value, &spec, error)) return false;
  if (!ParseDimensions(*dimensions, &spec.dimensions, error)) return false;
  if (!ParseBlocks(*blocks, &spec, error)) return false;
  if (!ParseEmbeddings(*embeddings, &spec.embeddings, error)) return false;
  if (!ParseHead(*head, &spec.head, error)) return false;
  if (!ParseLoss(*loss, &spec.loss, error)) return false;
  if (!ParsePolicy(*policy, &spec.policy, error)) return false;

  if (!ValidateModelSpec(spec, error)) return false;
  *out = spec;
  return true;
}

bool LookupModelSpec(const char* name, ModelSpec* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) return Fail(error, "no output spec");
  if (name == nullptr || *name == '\0') {
    *out = DefaultModelSpec();
    return true;
  }
  return ParseModelSpec(ModelSpecPath(name), out, error);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool ValidateModelSpec(const ModelSpec& spec, std::string* error) {
  if (error != nullptr) error->clear();
  const DimensionsSpec& dims = spec.dimensions;
  if (spec.spec_version != kModelSpecVersion) {
    return Fail(error, "spec_version " + std::to_string(spec.spec_version) +
                           " is not this build's version (" +
                           std::to_string(kModelSpecVersion) + ")");
  }
  if (spec.blocks.empty()) {
    return Fail(error, "blocks is empty; its length *is* num_layers");
  }
  if (dims.hidden_dim <= 0 || dims.num_heads <= 0 || dims.num_kv_heads <= 0 ||
      dims.seq_len <= 0) {
    return Fail(error, "dimensions must be positive");
  }
  if (spec.embeddings.vocab_size <= 0 ||
      spec.embeddings.padded_vocab_size <= 0) {
    return Fail(error, "vocabulary sizes must be positive");
  }
  if (dims.hidden_dim % dims.num_heads != 0) {
    return Fail(error, "hidden_dim must be divisible by num_heads");
  }
  const int head_dim = dims.hidden_dim / dims.num_heads;
  if (head_dim % 2 != 0) {
    return Fail(error, "head_dim must be even for rotary positions");
  }
  if (dims.num_heads % dims.num_kv_heads != 0) {
    return Fail(error, "num_heads must be divisible by num_kv_heads");
  }
  if (spec.embeddings.padded_vocab_size < spec.embeddings.vocab_size) {
    return Fail(error, "padded_vocab_size must be >= vocab_size");
  }
  if (!(dims.rope_base > 0.0f) || !std::isfinite(dims.rope_base)) {
    return Fail(error, "rope_base must be finite and positive");
  }
  if (!std::isfinite(spec.head.logit_softcap) || spec.head.logit_softcap <= 0.0f) {
    return Fail(error, "head.logit_softcap must be finite and positive");
  }
  // The short window `Config` derives for an `S` block: a quarter of the
  // context rounded up to a 128-token attention tile. It is a derived value, not
  // a declared one, so a context shorter than one tile is clamped by the graph's
  // `EffectiveWindow` rather than rejected here.
  const int short_window = ((dims.seq_len / 4 + 127) / 128) * 128;
  (void)short_window;
  for (std::size_t i = 0; i < spec.blocks.size(); ++i) {
    const BlockSpec& block = spec.blocks[i];
    const std::string at = "blocks[" + std::to_string(i) + "]";
    if (block.attention_variant == AttentionVariant::kMha &&
        dims.num_heads != dims.num_kv_heads) {
      return Fail(error, at + " declares the mha variant but num_heads (" +
                            std::to_string(dims.num_heads) +
                            ") differs from num_kv_heads (" +
                            std::to_string(dims.num_kv_heads) + ")");
    }
    if (block.attention_variant == AttentionVariant::kGqa &&
        dims.num_kv_heads > dims.num_heads) {
      return Fail(error, at + " declares the gqa variant but num_kv_heads "
                            "exceeds num_heads");
    }
    if (block.attention == AttentionKind::kSlidingWindow &&
        dims.seq_len <= 0) {
      return Fail(error, at +
                            " declares a sliding window inside a non-positive "
                            "context");
    }
  }
  const std::uint64_t named = AllNamedCapabilityBits();
  if ((spec.requires_features & ~named) != 0) {
    return Fail(error, "requires_features carries bits this build has no "
                       "vocabulary entry for");
  }
  if (IsOverrideCapability(spec.requires_features)) {
    return Fail(error, "requires_features carries an override bit; a model "
                       "never requires an implementation choice");
  }
  return true;
}

// ---------------------------------------------------------------------------
// The derivation
// ---------------------------------------------------------------------------

bool ModelSpecToConfig(const ModelSpec& spec, Config* config,
                       std::string* error) {
  if (error != nullptr) error->clear();
  if (config == nullptr) return Fail(error, "no output config");
  if (!ValidateModelSpec(spec, error)) return false;

  Config derived;
  derived.num_layers = static_cast<int>(spec.blocks.size());
  derived.num_heads = spec.dimensions.num_heads;
  // `kMha` means one key/value head per query head, whatever the dimensions
  // section says; `kGqa` carries the declared count.
  derived.num_kv_heads =
      spec.blocks.front().attention_variant == AttentionVariant::kMha
          ? spec.dimensions.num_heads
          : spec.dimensions.num_kv_heads;
  derived.hidden_dim = spec.dimensions.hidden_dim;
  derived.seq_len = spec.dimensions.seq_len;
  derived.vocab_size = spec.embeddings.vocab_size;
  derived.padded_vocab_size = spec.embeddings.padded_vocab_size;
  derived.rope_base = spec.dimensions.rope_base;
  derived.value_embedding = spec.embeddings.value_embedding;

  // The mask the graph runs is built from the ordered block list, so the two
  // cannot disagree: 'S' for a sliding-window block, 'L' for a full-context
  // one. `Config::window_left` reads this string.
  derived.window_pattern.clear();
  derived.window_pattern.reserve(spec.blocks.size());
  for (const BlockSpec& block : spec.blocks) {
    derived.window_pattern.push_back(
        block.attention == AttentionKind::kSlidingWindow ? 'S' : 'L');
  }
  // `Config::window_pattern` is a pattern, not a per-layer record:
  // `window_left(layer)` indexes it modulo its length. The derived string is
  // therefore reduced to its shortest repeating unit, which is the same mask for
  // every layer (`"SSSLSSSLSSSL"` derives the historical `"SSSL"`) and a
  // non-periodic block list keeps its full spelling because no unit repeats.
  {
    const std::string full = derived.window_pattern;
    for (std::size_t unit = 1; unit <= full.size(); ++unit) {
      if (full.size() % unit != 0) continue;
      bool periodic = true;
      for (std::size_t i = unit; i < full.size() && periodic; ++i) {
        if (full[i] != full[i % unit]) periodic = false;
      }
      if (periodic) {
        derived.window_pattern = full.substr(0, unit);
        break;
      }
    }
  }

  // A fact with no representation in `Config` is an error rather than a
  // silent drop. Position is the live case: the derived struct has no field
  // for "no rotation", and the graph always rotates, so a `none` block would
  // run a different meaning than it declares.
  for (std::size_t i = 0; i < spec.blocks.size(); ++i) {
    if (spec.blocks[i].position == PositionKind::kNone) {
      return Fail(error, "blocks[" + std::to_string(i) +
                             "] declares no positional treatment, which Config "
                             "cannot represent: the graph always applies "
                             "rotary positions");
    }
  }

  if (!derived.Validate(error)) return false;
  *config = derived;
  return true;
}

}  // namespace nanochat
