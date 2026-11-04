// The one place an `InferenceView` is built from a concrete model
// (spec/api-surface.md R9). The builder lives here because this is the
// translation unit that also sees `TrainModel`, which is what lets
// `generate.cc` read the weights without including `src/model_impl.h`.

#include "src/inference_view.h"

#include "src/model_impl.h"

namespace nanochat {
namespace {

const TrainModel& Concrete(const Model& model) {
  return static_cast<const TrainModel&>(model);
}

TrainModel& Concrete(Model& model) { return static_cast<TrainModel&>(model); }

float ReadSmearLambda(const Model& model) {
  return Concrete(model).smear_lambda();
}

float ReadBackoutLambda(const Model& model) {
  return Concrete(model).backout_lambda();
}

float ReadResid(const Model& model, int layer) {
  return Concrete(model).resid(layer);
}

float ReadX0Lambda(const Model& model, int layer) {
  return Concrete(model).x0_lambda(layer);
}

void GrowRope(Model& model, int seq) {
  Concrete(model).EnsureRopeCapacity(seq);
}

const float* ReadCosTable(const Model& model) {
  return Concrete(model).cos_table();
}

const float* ReadSinTable(const Model& model) {
  return Concrete(model).sin_table();
}

}  // namespace

InferenceView ViewForInference(Model& model) {
  const TrainModel& train = Concrete(model);
  InferenceView view;
  view.config = &train.config();
  view.wte = train.wte();
  view.lm_head = train.lm_head();
  view.smear_gate = train.smear_gate();
  view.layer_weights = train.layer_weights();
  view.num_layers = train.config().num_layers;
  view.model = &model;
  view.smear_lambda = &ReadSmearLambda;
  view.backout_lambda = &ReadBackoutLambda;
  view.resid = &ReadResid;
  view.x0_lambda = &ReadX0Lambda;
  view.ensure_rope_capacity = &GrowRope;
  view.cos_table = &ReadCosTable;
  view.sin_table = &ReadSinTable;
  return view;
}

}  // namespace nanochat
