#ifndef NANOCHAT_MODEL_SPEC_H_
#define NANOCHAT_MODEL_SPEC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/tensor.h"

// The model definition as data: one closed description of *what* the graph is,
// parsed from `models/<name>/spec.json`, plus the derivation into the runtime
// `Config`.
//
// The closed-field rule
// ---------------------
// Every field below is closed. A field is added only by bumping
// `kModelSpecVersion`, and only when a bring-up needs it; a field is never
// added because one particular model wanted one, and there is no free-form or
// string-keyed part of the description. This is the guard against the spec
// drifting into a graph IR (DESIGN.md section 6): the description holds
// identity, dimensions, one ordered block list, and policy. It holds no
// operations, no edges, no per-op shapes, no scheduling, and no fusion
// decisions. How a declared block actually runs is `nanochat/plan.h`'s
// question, asked on the device side of this description, so adding a fused
// family or a new backend changes nothing here.
//
// This header sits above the kernel seam: it includes `nanochat/config.h` and
// `nanochat/tensor.h` and nothing from `backends/`.

namespace nanochat {

// ---------------------------------------------------------------------------
// The closed vocabularies a block is built from
// ---------------------------------------------------------------------------

// The normalization inside a block.
enum class NormKind {
  kRmsNorm = 0,
};

// The attention mask a block declares.
enum class AttentionKind {
  kCausal = 0,         // full causal context
  kSlidingWindow = 1,  // causal, with a left window
};

// How many key/value heads the block's attention has. `kGqa` is the general
// case; `kMha` means one key/value head per query head (`num_kv_heads ==
// num_heads`).
enum class AttentionVariant {
  kMha = 0,
  kGqa = 1,
};

// The positional treatment.
enum class PositionKind {
  kNone = 0,
  kRope = 1,
};

// The MLP block. The width stays derived: `Config::mlp_dim()` is the authority,
// so a spec cannot declare a second answer to it.
enum class MlpKind {
  kReluSquare = 0,
};

// The residual path. `kBlend` is the ResFormer residual (the per-layer residual
// scale plus the learned embedding blend); `kPlain` is a plain add.
enum class ResidualKind {
  kPlain = 0,
  kBlend = 1,
};

// The storage/compute policy for a block, or for the whole model.
// `kBuildDefault` means "no policy": the build precision applies. It is also
// how a block declines a per-block policy, which is why the field below needs
// no separate "present" flag.
enum class DTypePolicy {
  kBuildDefault = 0,
  kFp32 = 1,
  kBf16 = 2,
  kFp8 = 3,
};

// The head the trunk projects onto.
enum class HeadKind {
  kUntiedLinear = 0,  // a separate unembedding matrix
};

// The loss the training graph minimizes.
enum class LossKind {
  // Logit softcap + cross-entropy over the vocabulary.
  kSoftcapCrossEntropy = 0,
};

// ---------------------------------------------------------------------------
// The description
// ---------------------------------------------------------------------------

// One block of the trunk. The fields say *what the block is*, in the closed
// vocabularies above; they do not say how it runs.
struct BlockSpec {
  NormKind norm = NormKind::kRmsNorm;
  AttentionKind attention = AttentionKind::kCausal;
  AttentionVariant attention_variant = AttentionVariant::kGqa;
  PositionKind position = PositionKind::kRope;
  MlpKind mlp = MlpKind::kReluSquare;
  ResidualKind residual = ResidualKind::kBlend;
  // The optional per-block dtype policy. `kBuildDefault` means the block has
  // none and the model policy applies.
  DTypePolicy dtype = DTypePolicy::kBuildDefault;
};

// The dimensions that are not implied by the block list or the embed table.
// `head_dim` is `hidden_dim / num_heads` and the MLP width is `4 * hidden_dim`;
// both stay derived, so neither appears here.
struct DimensionsSpec {
  int num_heads = 6;     // query heads
  int num_kv_heads = 6;  // key/value heads
  int hidden_dim = 768;
  int seq_len = 2048;
  float rope_base = 100000.0f;
};

// The token table and the ResFormer value embeddings.
struct EmbeddingsSpec {
  int vocab_size = 32768;
  int padded_vocab_size = 32768;  // rounded up for GEMM/classifier efficiency
  bool value_embedding = true;    // value embeddings on alternating blocks
};

// The unembedding. `Config` does not mirror these two facts (the softcap is a
// kernel default today), so the graph reads them from here.
struct HeadSpec {
  HeadKind kind = HeadKind::kUntiedLinear;
  float logit_softcap = 15.0f;  // mirrors kDefaultLogitSoftcap in kernels.h
};

// The training loss.
struct LossSpec {
  LossKind kind = LossKind::kSoftcapCrossEntropy;
  int ignore_index = -1;  // a target equal to this contributes no loss
};

// The build-and-run policy the graph is allowed to assume. These are closed
// knobs; a new knob is a spec_version bump.
struct PolicySpec {
  DTypePolicy dtype = DTypePolicy::kBuildDefault;  // the model default
  bool allow_fused_overrides = true;   // a registered fused family may win
  bool require_primitive_tier = true;  // needs the full primitive seam
};

// The schema version this header defines. A spec file that declares another
// number is rejected instead of half-parsed.
inline constexpr int kModelSpecVersion = 1;

// The whole description. `name` is the identity (`models/<name>/spec.json`),
// and `blocks` is ordered: it is the only ordering in the spec, and its length
// *is* `num_layers`, so the two can never disagree.
struct ModelSpec {
  int spec_version = kModelSpecVersion;
  std::string name;
  // The "requires" bits: the mechanism mask the bring-up must report, spelled
  // with the kFeature* constants in nanochat/tensor.h. The field is called
  // `requires_features` only because `requires` is a C++20 keyword; the JSON
  // key is plain "requires". A model asks for mechanisms; it never asks for a
  // device.
  std::uint64_t requires_features = 0;

  DimensionsSpec dimensions;
  std::vector<BlockSpec> blocks;
  EmbeddingsSpec embeddings;
  HeadSpec head;
  LossSpec loss;
  PolicySpec policy;
};

// ---------------------------------------------------------------------------
// Loading and derivation
// ---------------------------------------------------------------------------

// The built-in default: the spec that reproduces `Config`'s own defaults. It is
// what a null spec name resolves to, so `Model::Create(config)` keeps its
// historical meaning.
const ModelSpec& DefaultModelSpec();

// The file a name refers to: "models/<name>/spec.json".
std::string ModelSpecPath(const std::string& name);

// Parses a spec file into `*out`. Returns false and writes a one-line reason to
// `error` when it is not null. Unknown keys are an error, because the field set
// is closed; so is a `spec_version` this header does not define, a duplicated
// key, and a value outside its closed vocabulary.
bool ParseModelSpec(const std::string& path, ModelSpec* out,
                    std::string* error = nullptr);

// Resolves a name to a spec: a null name yields `DefaultModelSpec()`, any other
// name is parsed from `ModelSpecPath(name)`. This is the lookup behind
// `Model::Create`'s selector; it reads the host filesystem and nothing else.
bool LookupModelSpec(const char* name, ModelSpec* out,
                     std::string* error = nullptr);

// Checks the closed field set and the cross-field rules: a nonempty block list,
// a positive hidden width, heads that divide it, an even head width for rotary
// positions, padding at least the vocabulary, `kMha` only when the two head
// counts are equal, a sliding-window block only inside a positive context, and
// `requires_features` inside the defined feature bits.
bool ValidateModelSpec(const ModelSpec& spec, std::string* error = nullptr);

// Derives the runtime `Config`. `Config` stays the derived struct: it keeps
// `head_dim()`, `mlp_dim()`, `window_left()` and `Validate()`, and the spec
// never duplicates them. `window_pattern` is built from the block order ('S'
// for a sliding-window block, 'L' for a full-context one), so the block list
// cannot disagree with the window the graph runs. Returns false and writes
// `error` when a declared fact has no representation in `Config`.
bool ModelSpecToConfig(const ModelSpec& spec, Config* config,
                       std::string* error = nullptr);

}  // namespace nanochat

#endif  // NANOCHAT_MODEL_SPEC_H_
