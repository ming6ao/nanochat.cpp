#ifndef NANOCHAT_SRC_INFERENCE_VIEW_H_
#define NANOCHAT_SRC_INFERENCE_VIEW_H_

#include "nanochat/model.h"
#include "nanochat/tensor.h"
#include "src/ops.h"

// The values the inference graphs (`generate.cc`) read out of a model, and
// nothing else (spec/api-surface.md R9). Inference needs the frozen weights,
// the rotary tables, and the four mix scalars; it does not need the training
// state, so this header names no training type and includes no training header.
// `ViewForInference` builds the view, and its definition lives in
// `src/inference_view.cc`, the one translation unit that also sees the concrete
// model.
//
// The weights are plain pointers and counts, the way `src/ops.h` spells its
// operands. The mix scalars and the rotary tables stay behind a reader instead
// of a snapshot: each one is a device read the graph performs where it needs
// the value (the concrete model's `resid` and its neighbours copy one value
// back per call, and a growth of the tables moves them), so copying them here
// would move that round trip to the start of the pass.

namespace nanochat {

// Reads one global mix scalar back from the model.
using InferenceScalarReader = float (*)(const Model& model);
// Reads one per-layer mix scalar back from the model.
using InferenceLayerScalarReader = float (*)(const Model& model, int layer);
// Reads one rotary table. The table may move when it grows, so a caller reads
// it after `ensure_rope_capacity` rather than keeping the pointer.
using InferenceTableReader = const float* (*)(const Model& model);

struct InferenceView {
  // The configuration the graph shapes itself from.
  const Config* config = nullptr;

  // The frozen weight buffers.
  const ComputeType* wte = nullptr;
  const ComputeType* lm_head = nullptr;
  const ComputeType* smear_gate = nullptr;
  // One weight block per layer, in block order; `num_layers` entries.
  const ops::BlockWeights* layer_weights = nullptr;
  int num_layers = 0;

  // The model the readers below apply to.
  Model* model = nullptr;
  InferenceScalarReader smear_lambda = nullptr;
  InferenceScalarReader backout_lambda = nullptr;
  InferenceLayerScalarReader resid = nullptr;
  InferenceLayerScalarReader x0_lambda = nullptr;

  // Grows the rotary tables to hold `seq` positions. The tables are read
  // through the accessors beside it, after it runs, because it may move them.
  void (*ensure_rope_capacity)(Model& model, int seq) = nullptr;
  InferenceTableReader cos_table = nullptr;
  InferenceTableReader sin_table = nullptr;
};

// Builds the view for `model`. The definition is in `src/inference_view.cc`, so
// a caller of this header never sees the concrete model.
InferenceView ViewForInference(Model& model);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_INFERENCE_VIEW_H_
