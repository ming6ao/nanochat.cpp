// src/mesh_runtime.cc -- the single-client multi-device training runtime
// (docs/client-runtime-design.md sections 7 and 10). One device thread per
// replica, one in-process collective, one controller. CPU-tested; the CUDA
// thread-local change is a later phase.

#include "src/mesh_runtime.h"

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace nanochat {
namespace {

int EffectiveAccum(int grad_accum) { return grad_accum > 0 ? grad_accum : 1; }

}  // namespace

// The per-replica state. The device thread owns the model, the optimizer, and
// the gradient-sync view. The controller signals work through `mutex` and
// `cv`.
struct MeshRuntime::Replica {
  std::unique_ptr<Model> model;
  std::unique_ptr<Optimizer> optimizer;
  std::unique_ptr<GradientSync> sync;
  std::thread thread;

  std::mutex mutex;
  std::condition_variable cv;
  int device = 0;
  bool ready = false;
  bool has_work = false;
  bool done = false;
  bool stop = false;
  const int* tokens = nullptr;
  const int* targets = nullptr;
  int rows = 0;
  int step = 0;
  float loss = 0.0f;
};

MeshRuntime::MeshRuntime(const MeshSpec& mesh, MeshRuntimeConfig config)
    : mesh_(mesh), config_(std::move(config)) {
  std::string error;
  if (!mesh_.Validate(&error)) {
    throw std::invalid_argument(error);
  }
  const int world = mesh_.size();
  seq_ = config_.seq > 0 ? config_.seq : config_.model.seq_len;
  if (config_.batch <= 0) {
    throw std::invalid_argument("the global batch must be positive");
  }
  if (config_.batch % world != 0) {
    throw std::invalid_argument(
        "the global batch must divide by the mesh "
        "size");
  }
  slice_rows_ = config_.batch / world;
  const int accum = EffectiveAccum(config_.grad_accum);
  if (slice_rows_ % accum != 0) {
    throw std::invalid_argument(
        "the per-replica batch must divide by grad_accum");
  }
  micro_rows_ = slice_rows_ / accum;

  collective_ = std::make_unique<InProcessCollective>(world);

  replicas_.reserve(static_cast<std::size_t>(world));
  for (int rank = 0; rank < world; ++rank) {
    replicas_.push_back(std::make_unique<Replica>());
    Replica* replica = replicas_.back().get();
    replica->device = rank < static_cast<int>(mesh_.device_indices.size())
                          ? mesh_.device_indices[static_cast<std::size_t>(rank)]
                          : rank;
    replica->sync = collective_->Participant(rank);
    replica->thread = std::thread([this, replica] { ReplicaLoop(*replica); });
  }

  // Wait until every replica has built its model, so `TrainStep` and the
  // accessors never race the construction.
  for (const auto& replica : replicas_) {
    std::unique_lock<std::mutex> lock(replica->mutex);
    replica->cv.wait(lock, [&replica] { return replica->ready; });
  }
}

MeshRuntime::~MeshRuntime() {
  for (const auto& replica : replicas_) {
    {
      std::lock_guard<std::mutex> lock(replica->mutex);
      replica->stop = true;
    }
    replica->cv.notify_all();
  }
  for (const auto& replica : replicas_) {
    if (replica->thread.joinable()) replica->thread.join();
  }
}

void MeshRuntime::ReplicaLoop(Replica& replica) {
  // Bind the thread to its device before any backend call, so the model, the
  // workspace, and the kernels all land on the right card
  // (docs/client-runtime-design.md section 8).
  SelectDevice(replica.device);
  replica.model = Model::Create(config_.model);
  replica.model->InitWeights(config_.seed);
  Scheduler scheduler(config_.scheduler);
  replica.optimizer =
      CreateOptimizer(replica.model.get(), config_.optimizer, scheduler);
  {
    std::lock_guard<std::mutex> lock(replica.mutex);
    replica.ready = true;
  }
  replica.cv.notify_all();

  for (;;) {
    std::unique_lock<std::mutex> lock(replica.mutex);
    replica.cv.wait(lock,
                    [&replica] { return replica.has_work || replica.stop; });
    if (replica.stop) break;
    replica.has_work = false;
    const int* tokens = replica.tokens;
    const int* targets = replica.targets;
    const int step = replica.step;
    lock.unlock();

    const float loss = RunReplicaStep(replica, tokens, targets, step);

    lock.lock();
    replica.loss = loss;
    replica.done = true;
    lock.unlock();
    replica.cv.notify_all();
  }
}

float MeshRuntime::RunReplicaStep(Replica& replica, const int* tokens,
                                  const int* targets, int step) {
  Model* model = replica.model.get();
  const int accum = EffectiveAccum(config_.grad_accum);
  const int world = mesh_.size();
  const float scale = 1.0f / static_cast<float>(accum * world);

  model->ZeroGrad();
  float loss_sum = 0.0f;
  for (int micro = 0; micro < accum; ++micro) {
    const std::size_t offset = static_cast<std::size_t>(micro) *
                               static_cast<std::size_t>(micro_rows_) *
                               static_cast<std::size_t>(seq_);
    loss_sum += model->ForwardLoss(tokens + offset, targets + offset,
                                   micro_rows_, seq_);
    model->BackwardAccumulate(scale);
  }

  // Reduce every gradient before the optimizer step, so the step sees the
  // global mean gradient and every replica stays equal.
  for (const ParamView& parameter : model->params()) {
    if (parameter.grad == nullptr || parameter.count <= 0) continue;
    replica.sync->AllReduceSum(parameter.grad, parameter.count);
  }
  replica.optimizer->Step(step);
  return loss_sum / static_cast<float>(accum);
}

float MeshRuntime::TrainStep(const int* tokens, const int* targets) {
  if (tokens == nullptr || targets == nullptr) {
    throw std::invalid_argument("tokens and targets must not be null");
  }
  const int world = mesh_.size();
  ++step_;

  // Dispatch one contiguous slice per replica. The slice is a sequence range,
  // so the per-sequence forward stays independent.
  for (int rank = 0; rank < world; ++rank) {
    Replica& replica = *replicas_[static_cast<std::size_t>(rank)];
    const std::size_t offset = static_cast<std::size_t>(rank) *
                               static_cast<std::size_t>(slice_rows_) *
                               static_cast<std::size_t>(seq_);
    {
      std::lock_guard<std::mutex> lock(replica.mutex);
      replica.tokens = tokens + offset;
      replica.targets = targets + offset;
      replica.rows = slice_rows_;
      replica.step = step_;
      replica.has_work = true;
    }
    replica.cv.notify_all();
  }

  float loss = 0.0f;
  for (int rank = 0; rank < world; ++rank) {
    Replica& replica = *replicas_[static_cast<std::size_t>(rank)];
    std::unique_lock<std::mutex> lock(replica.mutex);
    replica.cv.wait(lock, [&replica] { return replica.done; });
    replica.done = false;
    loss += replica.loss;
  }
  return loss / static_cast<float>(world);
}

Model* MeshRuntime::model(int replica) {
  if (replica < 0 || replica >= world_size()) return nullptr;
  return replicas_[static_cast<std::size_t>(replica)]->model.get();
}

const Model* MeshRuntime::model(int replica) const {
  if (replica < 0 || replica >= world_size()) return nullptr;
  return replicas_[static_cast<std::size_t>(replica)]->model.get();
}

int MeshRuntime::param_count(int replica) const {
  const Model* replica_model = model(replica);
  return replica_model == nullptr
             ? 0
             : static_cast<int>(replica_model->params().size());
}

void MeshRuntime::Save(const std::string& path) {
  Model* replica = model(0);
  if (replica == nullptr) {
    throw std::runtime_error("the runtime has no replica to save");
  }
  replica->Save(path);
}

void MeshRuntime::Load(const std::string& path) {
  for (int rank = 0; rank < world_size(); ++rank) {
    Model* replica = model(rank);
    replica->Load(path);
    replica->SyncMaster();
  }
}

}  // namespace nanochat
