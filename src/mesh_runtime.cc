// src/mesh_runtime.cc -- the single-client multi-device training runtime
// (docs/client-runtime-design.md sections 7 and 10). One device thread per
// replica, one in-process collective, one controller. The CUDA backend state
// is thread-local, so a device thread binds with `SelectDevice` before it
// builds the replica.

#include "src/mesh_runtime.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nanochat/kernels.h"
#include "src/train.h"

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
  float grad_norm = 0.0f;
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
  // The reduced gradient is equal on every replica, so the norm is too. Read
  // it after the step, before the controller sees `done`.
  replica.grad_norm = replica.optimizer->GradNorm();
  return loss_sum / static_cast<float>(accum);
}

float MeshRuntime::TrainStep(const int* tokens, const int* targets) {
  if (tokens == nullptr || targets == nullptr) {
    throw std::invalid_argument("tokens and targets must not be null");
  }
  const auto start = std::chrono::steady_clock::now();
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
  metrics_.loss.resize(static_cast<std::size_t>(world));
  for (int rank = 0; rank < world; ++rank) {
    Replica& replica = *replicas_[static_cast<std::size_t>(rank)];
    std::unique_lock<std::mutex> lock(replica.mutex);
    replica.cv.wait(lock, [&replica] { return replica.done; });
    replica.done = false;
    metrics_.loss[static_cast<std::size_t>(rank)] = replica.loss;
    loss += replica.loss;
  }
  metrics_.grad_norm = replicas_.empty() ? 0.0f : replicas_[0]->grad_norm;
  const auto end = std::chrono::steady_clock::now();
  metrics_.step_seconds = std::chrono::duration<float>(end - start).count();
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

std::uint64_t MeshRuntime::ParamDigest(int replica) const {
  const Model* replica_model = model(replica);
  if (replica_model == nullptr) {
    throw std::out_of_range("the replica index is out of range");
  }
  // FNV-1a over the raw parameter bytes. The replicas are byte-identical, so
  // the digest is exact. The chunk bounds the host staging buffer.
  constexpr std::size_t kChunk = 4096;
  constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::vector<ComputeType> host(kChunk);
  std::uint64_t hash = kOffsetBasis;
  for (const ParamView& parameter : replica_model->params()) {
    if (parameter.value == nullptr || parameter.count <= 0) continue;
    std::int64_t remaining = parameter.count;
    std::int64_t offset = 0;
    while (remaining > 0) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::int64_t>(remaining, static_cast<std::int64_t>(kChunk)));
      kernels::Memcpy(host.data(), parameter.value + offset,
                      count * sizeof(ComputeType), CopyDir::kDeviceToHost);
      const auto* bytes = reinterpret_cast<const unsigned char*>(host.data());
      const std::size_t byte_count = count * sizeof(ComputeType);
      for (std::size_t i = 0; i < byte_count; ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
      }
      offset += static_cast<std::int64_t>(count);
      remaining -= static_cast<std::int64_t>(count);
    }
  }
  return hash;
}

void MeshRuntime::Save(const std::string& path) {
  Replica* replica = replicas_.empty() ? nullptr : replicas_[0].get();
  if (replica == nullptr || replica->model == nullptr ||
      replica->optimizer == nullptr) {
    throw std::runtime_error("the runtime has no replica to save");
  }
  // The harness container carries the optimizer state and the step beside the
  // parameters, so a resume continues the schedule (docs/training-seam.md
  // section 10). `Checkpointer::SaveModel(model, path)` writes the parameters
  // only.
  Checkpointer::SaveModel(*replica->model, *replica->optimizer, step_, path);
}

void MeshRuntime::Load(const std::string& path) {
  int step = 0;
  for (const auto& replica : replicas_) {
    if (replica->model == nullptr || replica->optimizer == nullptr) continue;
    if (!Checkpointer::LoadModel(replica->model.get(), replica->optimizer.get(),
                                 &step, path)) {
      throw std::runtime_error("the checkpoint did not load: " + path);
    }
  }
  if (step > 0) step_ = step;
}

}  // namespace nanochat
