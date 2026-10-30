#ifndef NANOCHAT_PLAN_H_
#define NANOCHAT_PLAN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/model_spec.h"
#include "nanochat/tensor.h"

// The device-and-graph plan: given a `ModelSpec` and a graph, decide which
// implementation runs each composition on the active device, and say why when
// nothing can.
//
// The name collides with an unrelated one, so the distinction is stated here
// once. *This* `Plan` is the device-and-graph plan built by `BuildPlan`.
// `python/nanochat_cpp/plan.py` is the training-hyperparameter planner: it
// chooses learning rates, batch sizes and schedules, never an implementation.
// The two do not call each other and the names are not synonyms.
//
// Purity (DESIGN.md section 9.1). `BuildPlan` is pure: it reads the spec, the
// graph kind, the device index and `kernels::GetCaps()`, and it touches nothing
// else on the device. It allocates no device memory, launches no kernel, warms
// no cache, and mutates no global state, so a bring-up check and a test can
// call it without a device side effect, and two calls with equal inputs return
// the same plan.
//
// A plan never changes the graph. It chooses an implementation for pieces the
// graph already has, so a backend cannot add a composition, reorder a block, or
// name a graph: it registers a fused family against a composition name and a
// capability bit (`FusedOverride` below), and the plan decides whether that
// entry wins or is declined (DESIGN.md section 6).

namespace nanochat {

// The graphs the runtime can ask for. Closed, and deliberately so: the topology
// of each one lives in the graph files, and a plan chooses implementations
// inside them, never new topology.
enum class GraphKind {
  kTrain = 0,
  kPrefill = 1,
  kDecode = 2,
  kEval = 3,
  kGenerate = 4,
};

// The stable spelling of a graph kind, for reports and error strings.
const char* GraphKindName(GraphKind kind);

// The closed list of compositions: one named unit of the graph that the plan
// chooses an implementation for. A composition is not a kernel name; it is the
// piece of the graph that one family (primitive or fused) implements whole. The
// names below travel, so they are part of the contract: a caps file and a
// registration both spell them exactly as `CompositionName` does.
enum class CompositionKind {
  kEmbedding = 0,
  kRmsNorm = 1,
  kQkPrep = 2,
  kAttention = 3,
  kMlp = 4,  // the relu^2 activation pair around the projection
  kClassifier = 5,
  kValueGate = 6,
  kGradNorm = 7,
  kAdamW = 8,
  kMuon = 9,
};

// The name of a composition: "embedding", "rms_norm", "qk_prep", "attention",
// "mlp", "classifier", "value_gate", "grad_norm", "adam_w", "muon".
const char* CompositionName(CompositionKind kind);

// Parses a composition name back. Returns false for an unknown name, which is
// what keeps a caps file or a registration from naming a composition that does
// not exist.
bool ParseCompositionName(const char* name, CompositionKind* out);

// One fused-family registration. A backend provides a static array of these and
// returns it from `RegisteredFusedOverrides`; the plan matches on the
// composition, requires `capability` to be set in `Caps::feature_mask`, and
// either selects the family or records it as a declined fallback.
//
// The entry names a *composition*, a family and a capability bit. It never
// names a graph, a block index, an edge or a kernel symbol, which is what stops
// a backend from adding topology through the back door (DESIGN.md section 6). A
// registration whose capability bit is clear is ignored rather than rejected,
// so a backend can land the entry before the family passes its gate; the bit,
// not the entry, is the switch.
//
// The bits a registration may take are the kFeatureFused* constants in
// nanochat/tensor.h; there is one per composition whose fusion is legal under
// the boundary rule (DESIGN.md section 3): `rms_norm` (residual add + norm),
// `qk_prep` (norm + rope + scale), `attention` (its v2 form folds qk_prep in),
// and `classifier` (softcap + cross-entropy + the logits scale).
struct FusedOverride {
  const char* composition = "";  // the CompositionName spelling
  const char* family = "";       // the family the plan names in its choice
  std::uint64_t capability = 0;  // the kFeatureFused* bit it also advertises
};

// The backend's registrations, in a device-independent order. A backend with no
// fused family returns nullptr and writes 0, so it compiles and links with no
// override code at all. The array is static and must outlive every plan.
const FusedOverride* RegisteredFusedOverrides(int* count);

// What to plan for. `device_index` selects the device: a negative value means
// the backend's default device. `model` is borrowed and must outlive the call.
struct PlanRequest {
  const ModelSpec* model = nullptr;
  GraphKind graph = GraphKind::kTrain;
  int device_index = 0;
};

// A fallback the plan considered for one composition and did not take.
struct DeclinedImpl {
  std::string implementation;
  std::string reason;
};

// One composition's decision.
struct CompositionChoice {
  CompositionKind composition = CompositionKind::kEmbedding;
  // The family that runs it, as the plan spells it in its report.
  std::string implementation;
  // True when a registered `FusedOverride` won, false when the primitive
  // family did.
  bool fused = false;
  // Every candidate considered before the chosen one, with the reason it lost.
  std::vector<DeclinedImpl> declined;
};

// A composition with no legal implementation on this device.
struct PlanGap {
  CompositionKind composition = CompositionKind::kEmbedding;
  std::string reason;
};

// The plan. `choices` has one entry per composition the graph runs, and `gaps`
// lists the compositions that could not be placed; `gaps` is empty exactly when
// `ok` is true.
struct Plan {
  bool ok = false;
  GraphKind graph = GraphKind::kTrain;
  int device_index = 0;
  std::string spec_name;  // the identity the plan was built for
  std::vector<CompositionChoice> choices;
  std::vector<PlanGap> gaps;
};

// Builds the plan. Pure: see the header comment. On rejection it returns a plan
// with `ok == false` and every unplaceable composition in `gaps`, and writes
// the same summary to `why_not` when that is not null. A null `request.model`,
// a spec that fails validation, and a device that does not report the spec's
// `requires_features` bits are all rejections rather than throws.
Plan BuildPlan(const PlanRequest& request, std::string* why_not = nullptr);

}  // namespace nanochat

#endif  // NANOCHAT_PLAN_H_
