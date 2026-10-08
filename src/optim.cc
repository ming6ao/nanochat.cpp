// The nanochat optimizer (docs/model.md).
//
// This translation unit owns the parameter grouping and the schedules; every
// numerical update rule lives behind the kernel seam (`nanochat/kernels.h`).
// The grouping mirrors `GPT.setup_optimizer` in nanochat/gpt.py:
//
//   AdamW  lm_head, the token embedding, the value embeddings, the
//          resid/x0 scalars, and the smear/backout scalars (including the
//          1 x 24 smear gate).
//   Muon   the remaining matrix parameters, stacked by (rows, cols).
//
// The schedules mirror nanochat's training loop (`get_lr_multiplier`,
// `get_muon_momentum`, `get_weight_decay`) and live in `Scheduler`.
//
// Gradient clipping is global across every parameter. `GlobalNorm` clips one
// contiguous buffer, so `Step` gathers every gradient into a single flat
// buffer (in grouping order), clips it once, and then drives the per-group
// kernels straight from that buffer.

#include "nanochat/optim.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/data.h"
#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"
#include "src/optim_state.h"

namespace nanochat {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDmodelRef = 768.0f;

// setup_optimizer scales every AdamW learning rate by 1/sqrt(dmodel/768).
float DmodelLrScale(int hidden_dim) {
  if (hidden_dim <= 0) return 1.0f;
  return std::sqrt(kDmodelRef / static_cast<float>(hidden_dim));
}

std::size_t ElementBytes(std::int64_t count) {
  return static_cast<std::size_t>(count) * sizeof(ComputeType);
}

enum class AdamWKind {
  kLmHead = 0,
  kEmbedding,
  kValueEmbedding,
  kResid,
  kX0,
  kSmear,
};

// Stable record-name token for one AdamW group (docs/post-training.md section
// 7). The token must not change once a checkpoint exists.
const char* KindName(AdamWKind kind) {
  switch (kind) {
    case AdamWKind::kLmHead:
      return "lm_head";
    case AdamWKind::kEmbedding:
      return "embedding";
    case AdamWKind::kValueEmbedding:
      return "value_embedding";
    case AdamWKind::kResid:
      return "resid";
    case AdamWKind::kX0:
      return "x0";
    case AdamWKind::kSmear:
      return "smear";
  }
  return "unknown";
}

// Records one optimizer-state buffer as a float32 tensor record. The kernel
// seam stages a device buffer to the host.
void AddOptimizerStateRecord(Checkpoint* checkpoint, const std::string& name,
                             const float* data, std::int64_t count) {
  if (checkpoint == nullptr || data == nullptr || count <= 0) return;
  TensorRecord record;
  record.name = name;
  record.dtype = DType::kFp32;
  record.shape.assign(1, count);
  const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
  record.data.resize(bytes);
  kernels::Memcpy(record.data.data(), data, bytes, CopyDir::kDeviceToHost);
  checkpoint->AddOptimizerState(std::move(record));
}

// One AdamW parameter group: the parameters, their flat gradient offsets, and
// the first/second moment buffers (float even in an fp16 build).
struct AdamWGroup {
  AdamWKind kind = AdamWKind::kLmHead;
  std::vector<ParamView> params;
  std::vector<std::int64_t> grad_offsets;
  std::vector<std::int64_t> moment_offsets;
  float lr = 0.0f;  // nominal, pre-schedule
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-10f;
  float weight_decay = 0.0f;
  float* m = nullptr;
  float* v = nullptr;
  std::int64_t moment_count = 0;
};

// One Muon group: every parameter shares the same (rows, cols) so the group can
// be stacked into a single `[num_params, rows, cols]` update.
struct MuonGroup {
  int rows = 0;
  int cols = 0;
  int num_params = 0;
  float lr = 0.0f;  // nominal, pre-schedule, includes sqrt(max(1, rows/cols))
  int red_dim = -1;
  std::int64_t flat_offset = 0;
  std::int64_t mat = 0;
  std::int64_t buf2_len = 0;
  std::vector<ParamView> params;
  ComputeType* stacked_params = nullptr;
  float* buf1 = nullptr;
  float* buf2 = nullptr;
};

// One ANVIL group (docs/optimizer-anvil-design.md). Same shape grouping as
// Muon, but the state is the twin-rail velocity and the lane energy instead of
// the momentum and the factored second moment.
struct AnvilGroup {
  int rows = 0;
  int cols = 0;
  int num_params = 0;
  float lr = 0.0f;  // nominal, pre-schedule, includes sqrt(max(1, rows/cols))
  int red_dim = -1;
  int lane_count = 0;
  std::int64_t flat_offset = 0;
  std::int64_t mat = 0;
  std::vector<ParamView> params;
  ComputeType* stacked_params = nullptr;
  float* velocity = nullptr;     // 2 * num_params * mat floats
  float* lane_energy = nullptr;  // num_params * lane_count floats
};

class NanochatOptimizer final : public Optimizer {
 public:
  NanochatOptimizer(Model* model, const OptimizerConfig& config,
                    const Scheduler& scheduler)
      : config_(config), scheduler_(scheduler) {
    Build(model);
  }

  ~NanochatOptimizer() override { Release(); }

  NanochatOptimizer(const NanochatOptimizer&) = delete;
  NanochatOptimizer& operator=(const NanochatOptimizer&) = delete;

  void ZeroGrad() override {
    for (const ParamView& view : views_) {
      if (view.grad == nullptr || view.count <= 0) continue;
      kernels::Memset(view.grad, 0, ElementBytes(view.count));
    }
  }

  void Step(int step) override {
    GatherGradients();

    float norm = 0.0f;
    if (total_count_ > 0) {
      kernels::GlobalNorm(static_cast<int>(total_count_), config_.clip, flat_,
                          norm_dev_);
      kernels::Memcpy(&norm, norm_dev_, sizeof(float), CopyDir::kDeviceToHost);
    }
    grad_norm_ = norm;

    const float lrm = scheduler_.LrMultiplier(step);

    // AdamW cadence (docs/optimizer-anvil-design.md): the reference steps AdamW
    // on the 0-based odd iterations (`training.is_adam_step`), which are the
    // 1-based even `step`s here, so with `adam_step_period` of 2 the AdamW
    // groups update on even steps only while the matrix groups update every
    // step. The bias correction reads the Adam-update ordinal derived from the
    // 0-based counter, so a resumed run re-derives it exactly; with the default
    // period of 1 the ordinal is the global step, which keeps the Muon path
    // bit-identical.
    const int it = step - 1;
    const bool run_adam =
        config_.adam_step_period <= 1 || (it % config_.adam_step_period) == 1;
    if (run_adam) {
      const int adam_step = config_.adam_step_period > 1
                                ? it / config_.adam_step_period + 1
                                : step;
      for (AdamWGroup& group : adamw_groups_) {
        for (std::size_t i = 0; i < group.params.size(); ++i) {
          AdamWParams params;
          params.lr = group.lr * lrm;
          params.beta1 = group.beta1;
          params.beta2 = group.beta2;
          params.eps = group.eps;
          params.weight_decay = group.weight_decay;
          params.step = adam_step;
          kernels::AdamWUpdate(static_cast<int>(group.params[i].count), params,
                               group.params[i].value,
                               flat_ + group.grad_offsets[i],
                               group.m + group.moment_offsets[i],
                               group.v + group.moment_offsets[i]);
        }
      }
    }

    if (config_.matrix_optimizer == 1) {
      StepAnvil(step, lrm);
      return;
    }

    // The Muon schedules come from the shared Scheduler: the learning rate is
    // the matrix rate times the LR multiplier, momentum decays over the run,
    // and the decoupled weight decay follows the cosine schedule.
    const float momentum = scheduler_.MuonMomentum(step);
    const float muon_weight_decay = scheduler_.WeightDecay(step);

    for (MuonGroup& group : muon_groups_) {
      GatherMuonParams(group);
      MuonParams params;
      params.num_params = group.num_params;
      params.rows = group.rows;
      params.cols = group.cols;
      params.lr = group.lr * lrm;
      params.momentum = momentum;
      params.beta2 = config_.muon_beta2;
      params.weight_decay = muon_weight_decay;
      params.ns_steps = config_.muon_ns_steps;
      params.red_dim = group.red_dim;
      params.nesterov = true;
      kernels::MuonUpdate(params, flat_ + group.flat_offset,
                          group.stacked_params, group.buf1, group.buf2);
      ScatterMuonParams(group);
    }
  }

  float GradNorm() const override { return grad_norm_; }

  // Optimizer-state checkpoint records (docs/post-training.md section 7): the
  // AdamW first/second moments and the Muon momentum/second-moment buffers,
  // beside the model parameters.
  void SaveState(Checkpoint* checkpoint) const {
    for (const AdamWGroup& group : adamw_groups_) {
      const std::string name = std::string("adamw.") + KindName(group.kind);
      AddOptimizerStateRecord(checkpoint, name + ".m", group.m,
                              group.moment_count);
      AddOptimizerStateRecord(checkpoint, name + ".v", group.v,
                              group.moment_count);
    }
    for (const MuonGroup& group : muon_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      const std::string name = "muon." + std::to_string(group.rows) + "x" +
                               std::to_string(group.cols);
      AddOptimizerStateRecord(checkpoint, name + ".buf1", group.buf1,
                              mat_total);
      AddOptimizerStateRecord(checkpoint, name + ".buf2", group.buf2,
                              group.buf2_len);
    }
    for (const AnvilGroup& group : anvil_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      const std::int64_t lane_total =
          static_cast<std::int64_t>(group.num_params) * group.lane_count;
      const std::string name = "anvil." + std::to_string(group.rows) + "x" +
                               std::to_string(group.cols);
      AddOptimizerStateRecord(checkpoint, name + ".velocity", group.velocity,
                              2 * mat_total);
      AddOptimizerStateRecord(checkpoint, name + ".lane_energy",
                              group.lane_energy, lane_total);
    }
  }

  // Copies every matching record into the group buffers. Returns true when at
  // least one record was restored.
  bool LoadState(const Checkpoint& checkpoint) {
    bool restored = false;
    auto load = [&](const std::string& name, float* data, std::int64_t count) {
      if (data == nullptr || count <= 0) return;
      const TensorRecord* record = checkpoint.FindOptimizerState(name);
      if (record == nullptr || record->dtype != DType::kFp32) return;
      const std::size_t wanted =
          static_cast<std::size_t>(count) * sizeof(float);
      const std::size_t bytes = std::min(wanted, record->data.size());
      if (bytes == 0) return;
      kernels::Memcpy(data, record->data.data(), bytes, CopyDir::kHostToDevice);
      restored = true;
    };
    for (const AdamWGroup& group : adamw_groups_) {
      const std::string name = std::string("adamw.") + KindName(group.kind);
      load(name + ".m", group.m, group.moment_count);
      load(name + ".v", group.v, group.moment_count);
    }
    for (const MuonGroup& group : muon_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      const std::string name = "muon." + std::to_string(group.rows) + "x" +
                               std::to_string(group.cols);
      load(name + ".buf1", group.buf1, mat_total);
      load(name + ".buf2", group.buf2, group.buf2_len);
    }
    for (const AnvilGroup& group : anvil_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      const std::int64_t lane_total =
          static_cast<std::int64_t>(group.num_params) * group.lane_count;
      const std::string name = "anvil." + std::to_string(group.rows) + "x" +
                               std::to_string(group.cols);
      load(name + ".velocity", group.velocity, 2 * mat_total);
      load(name + ".lane_energy", group.lane_energy, lane_total);
    }
    return restored;
  }

  // Test-only introspection (declared by src/optim_test.cc). Reports the group
  // that owns `name`: kind 0 = AdamW, 1 = Muon, the matrix extents, and the
  // nominal (pre-schedule) learning rate.
  bool GroupForTest(const char* name, int* kind, int* rows, int* cols,
                    float* lr) const {
    const std::string target(name);
    for (const AdamWGroup& group : adamw_groups_) {
      for (const ParamView& view : group.params) {
        if (target != view.name) continue;
        if (kind != nullptr) *kind = 0;
        if (rows != nullptr) *rows = view.rows;
        if (cols != nullptr) *cols = view.cols;
        if (lr != nullptr) *lr = group.lr;
        return true;
      }
    }
    for (const MuonGroup& group : muon_groups_) {
      for (const ParamView& view : group.params) {
        if (target != view.name) continue;
        if (kind != nullptr) *kind = 1;
        if (rows != nullptr) *rows = view.rows;
        if (cols != nullptr) *cols = view.cols;
        if (lr != nullptr) *lr = group.lr;
        return true;
      }
    }
    for (const AnvilGroup& group : anvil_groups_) {
      for (const ParamView& view : group.params) {
        if (target != view.name) continue;
        if (kind != nullptr) *kind = 2;
        if (rows != nullptr) *rows = view.rows;
        if (cols != nullptr) *cols = view.cols;
        if (lr != nullptr) *lr = group.lr;
        return true;
      }
    }
    return false;
  }

 private:
  void Build(Model* model) {
    views_ = model->params();
    const float scale = DmodelLrScale(model->config().hidden_dim);

    struct Spec {
      AdamWKind kind;
      float lr;
      float beta1;
      float beta2;
      float weight_decay;
    };
    // The per-group learning rates, betas, and weight decays of
    // `setup_optimizer`, in its exact order.
    const Spec specs[] = {
        {AdamWKind::kLmHead, config_.unembedding_lr * scale, 0.8f, 0.96f,
         0.01f},
        {AdamWKind::kEmbedding, config_.embedding_lr * scale, 0.8f, 0.995f,
         0.001f},
        {AdamWKind::kValueEmbedding, config_.embedding_lr * scale * 0.5f, 0.8f,
         0.995f, 0.01f},
        {AdamWKind::kResid, config_.scalar_lr * 0.01f, 0.8f, 0.95f, 0.05f},
        {AdamWKind::kX0, config_.scalar_lr, 0.96f, 0.95f, 0.0f},
        {AdamWKind::kSmear, 0.2f, 0.8f, 0.95f, 0.0f},
    };
    adamw_groups_.resize(sizeof(specs) / sizeof(specs[0]));
    for (std::size_t i = 0; i < adamw_groups_.size(); ++i) {
      AdamWGroup& group = adamw_groups_[i];
      group.kind = specs[i].kind;
      group.lr = specs[i].lr;
      group.beta1 = specs[i].beta1;
      group.beta2 = specs[i].beta2;
      group.eps = config_.adam_eps;
      group.weight_decay = specs[i].weight_decay;
    }

    // Classify every parameter. The name is the only stable handle the public
    // Model API exposes, and the model reports parameters in grouping order.
    std::map<std::pair<int, int>, std::vector<ParamView>> muon_by_shape;
    for (const ParamView& view : views_) {
      const std::string name(view.name);
      int index = -1;
      if (name == "lm_head.weight") {
        index = 0;
      } else if (name == "transformer.wte.weight") {
        index = 1;
      } else if (name.rfind("value_embeds.", 0) == 0) {
        index = 2;
      } else if (name == "resid_lambdas") {
        index = 3;
      } else if (name == "x0_lambdas") {
        index = 4;
      } else if (name == "smear_gate.weight" || name == "smear_lambda" ||
                 name == "backout_lambda") {
        index = 5;
      }
      if (index >= 0) {
        adamw_groups_[static_cast<std::size_t>(index)].params.push_back(view);
        continue;
      }
      // Everything else is a matrix parameter owned by Muon, grouped by shape.
      muon_by_shape[std::make_pair(view.rows, view.cols)].push_back(view);
    }

    // `std::map` iterates in ascending shape order, matching Python's
    // `for shape in sorted({p.shape ...})`.
    for (auto& entry : muon_by_shape) {
      if (config_.matrix_optimizer == 1) {
        AnvilGroup group;
        group.rows = entry.first.first;
        group.cols = entry.first.second;
        group.params = std::move(entry.second);
        group.num_params = static_cast<int>(group.params.size());
        group.mat = static_cast<std::int64_t>(group.rows) * group.cols;
        const bool reduce_cols = group.rows >= group.cols;
        group.red_dim = reduce_cols ? -1 : -2;
        group.lane_count = reduce_cols ? group.rows : group.cols;
        group.lr =
            config_.anvil_lr *
            std::sqrt(std::max(1.0f, static_cast<float>(group.rows) /
                                         static_cast<float>(group.cols)));
        anvil_groups_.push_back(std::move(group));
        continue;
      }
      MuonGroup group;
      group.rows = entry.first.first;
      group.cols = entry.first.second;
      group.params = std::move(entry.second);
      group.num_params = static_cast<int>(group.params.size());
      group.mat = static_cast<std::int64_t>(group.rows) * group.cols;
      const bool reduce_cols = group.rows >= group.cols;
      group.red_dim = reduce_cols ? -1 : -2;
      group.buf2_len = static_cast<std::int64_t>(group.num_params) *
                       (reduce_cols ? group.rows : group.cols);
      group.lr = config_.matrix_lr *
                 std::sqrt(std::max(1.0f, static_cast<float>(group.rows) /
                                              static_cast<float>(group.cols)));
      muon_groups_.push_back(std::move(group));
    }

    // Lay out the flat gradient buffer: AdamW groups first, then the matrix
    // groups.
    std::int64_t total = 0;
    for (AdamWGroup& group : adamw_groups_) {
      for (const ParamView& view : group.params) {
        group.grad_offsets.push_back(total);
        group.moment_offsets.push_back(group.moment_count);
        total += view.count;
        group.moment_count += view.count;
      }
    }
    for (MuonGroup& group : muon_groups_) {
      group.flat_offset = total;
      for (const ParamView& view : group.params) total += view.count;
    }
    for (AnvilGroup& group : anvil_groups_) {
      group.flat_offset = total;
      for (const ParamView& view : group.params) total += view.count;
    }
    total_count_ = total;

    flat_ =
        static_cast<ComputeType*>(kernels::Alloc(ElementBytes(total_count_)));
    norm_dev_ = static_cast<float*>(kernels::Alloc(sizeof(float)));

    for (AdamWGroup& group : adamw_groups_) {
      group.m = static_cast<float*>(kernels::Alloc(
          static_cast<std::size_t>(group.moment_count) * sizeof(float)));
      group.v = static_cast<float*>(kernels::Alloc(
          static_cast<std::size_t>(group.moment_count) * sizeof(float)));
      if (group.moment_count > 0) {
        kernels::Memset(
            group.m, 0,
            static_cast<std::size_t>(group.moment_count) * sizeof(float));
        kernels::Memset(
            group.v, 0,
            static_cast<std::size_t>(group.moment_count) * sizeof(float));
      }
    }

    for (MuonGroup& group : muon_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      group.stacked_params =
          static_cast<ComputeType*>(kernels::Alloc(ElementBytes(mat_total)));
      group.buf1 = static_cast<float*>(
          kernels::Alloc(static_cast<std::size_t>(mat_total) * sizeof(float)));
      group.buf2 = static_cast<float*>(kernels::Alloc(
          static_cast<std::size_t>(group.buf2_len) * sizeof(float)));
      if (mat_total > 0) {
        kernels::Memset(group.buf1, 0,
                        static_cast<std::size_t>(mat_total) * sizeof(float));
      }
      if (group.buf2_len > 0) {
        kernels::Memset(
            group.buf2, 0,
            static_cast<std::size_t>(group.buf2_len) * sizeof(float));
      }
    }

    for (AnvilGroup& group : anvil_groups_) {
      const std::int64_t mat_total =
          static_cast<std::int64_t>(group.num_params) * group.mat;
      const std::int64_t lane_total =
          static_cast<std::int64_t>(group.num_params) * group.lane_count;
      group.stacked_params =
          static_cast<ComputeType*>(kernels::Alloc(ElementBytes(mat_total)));
      group.velocity = static_cast<float*>(kernels::Alloc(
          static_cast<std::size_t>(2 * mat_total) * sizeof(float)));
      group.lane_energy = static_cast<float*>(
          kernels::Alloc(static_cast<std::size_t>(lane_total) * sizeof(float)));
      if (mat_total > 0) {
        kernels::Memset(
            group.velocity, 0,
            static_cast<std::size_t>(2 * mat_total) * sizeof(float));
      }
      if (lane_total > 0) {
        kernels::Memset(group.lane_energy, 0,
                        static_cast<std::size_t>(lane_total) * sizeof(float));
      }
    }
  }

  void Release() {
    if (flat_ != nullptr) kernels::Free(flat_);
    if (norm_dev_ != nullptr) kernels::Free(norm_dev_);
    for (AdamWGroup& group : adamw_groups_) {
      if (group.m != nullptr) kernels::Free(group.m);
      if (group.v != nullptr) kernels::Free(group.v);
    }
    for (MuonGroup& group : muon_groups_) {
      if (group.stacked_params != nullptr) kernels::Free(group.stacked_params);
      if (group.buf1 != nullptr) kernels::Free(group.buf1);
      if (group.buf2 != nullptr) kernels::Free(group.buf2);
    }
    for (AnvilGroup& group : anvil_groups_) {
      if (group.stacked_params != nullptr) kernels::Free(group.stacked_params);
      if (group.velocity != nullptr) kernels::Free(group.velocity);
      if (group.lane_energy != nullptr) kernels::Free(group.lane_energy);
    }
    flat_ = nullptr;
    norm_dev_ = nullptr;
  }

  // Copy every gradient into the flat buffer, in grouping order.
  void GatherGradients() {
    for (AdamWGroup& group : adamw_groups_) {
      for (std::size_t i = 0; i < group.params.size(); ++i) {
        kernels::Memcpy(flat_ + group.grad_offsets[i], group.params[i].grad,
                        ElementBytes(group.params[i].count),
                        CopyDir::kDeviceToDevice);
      }
    }
    for (MuonGroup& group : muon_groups_) {
      std::int64_t offset = group.flat_offset;
      for (const ParamView& view : group.params) {
        kernels::Memcpy(flat_ + offset, view.grad, ElementBytes(view.count),
                        CopyDir::kDeviceToDevice);
        offset += view.count;
      }
    }
    for (AnvilGroup& group : anvil_groups_) {
      std::int64_t offset = group.flat_offset;
      for (const ParamView& view : group.params) {
        kernels::Memcpy(flat_ + offset, view.grad, ElementBytes(view.count),
                        CopyDir::kDeviceToDevice);
        offset += view.count;
      }
    }
  }

  void GatherMuonParams(MuonGroup& group) {
    for (int i = 0; i < group.num_params; ++i) {
      kernels::Memcpy(
          group.stacked_params + static_cast<std::int64_t>(i) * group.mat,
          group.params[static_cast<std::size_t>(i)].value,
          ElementBytes(group.mat), CopyDir::kDeviceToDevice);
    }
  }

  void ScatterMuonParams(MuonGroup& group) {
    for (int i = 0; i < group.num_params; ++i) {
      kernels::Memcpy(
          group.params[static_cast<std::size_t>(i)].value,
          group.stacked_params + static_cast<std::int64_t>(i) * group.mat,
          ElementBytes(group.mat), CopyDir::kDeviceToDevice);
    }
  }

  void GatherAnvilParams(AnvilGroup& group) {
    for (int i = 0; i < group.num_params; ++i) {
      kernels::Memcpy(
          group.stacked_params + static_cast<std::int64_t>(i) * group.mat,
          group.params[static_cast<std::size_t>(i)].value,
          ElementBytes(group.mat), CopyDir::kDeviceToDevice);
    }
  }

  void ScatterAnvilParams(AnvilGroup& group) {
    for (int i = 0; i < group.num_params; ++i) {
      kernels::Memcpy(
          group.params[static_cast<std::size_t>(i)].value,
          group.stacked_params + static_cast<std::int64_t>(i) * group.mat,
          ElementBytes(group.mat), CopyDir::kDeviceToDevice);
    }
  }

  // One ANVIL step over every stacked group. The rail schedule and the engage
  // step come from the Scheduler and the config, so the kernel sees only the
  // resolved per-step scalars. `weight_decay` carries the outer `lr`, exactly
  // as docs/optimizer-anvil-design.md specifies.
  void StepAnvil(int step, float lrm) {
    const int it = step - 1;
    const float rail_beta = scheduler_.RailBeta(step);
    const bool engaged = it >= config_.anvil_engage_step;
    const float fast_beta = engaged ? config_.anvil_fast_beta : rail_beta;
    const float fast_weight = engaged ? config_.anvil_fast_weight : 1.0f;
    const float anvil_wd = config_.anvil_weight_decay * config_.anvil_lr * lrm;

    for (AnvilGroup& group : anvil_groups_) {
      GatherAnvilParams(group);
      AnvilParams params;
      params.num_params = group.num_params;
      params.rows = group.rows;
      params.cols = group.cols;
      params.lr = group.lr * lrm;
      params.momentum = rail_beta;
      params.fast_beta = fast_beta;
      params.slow_beta = config_.anvil_slow_beta;
      params.fast_weight = fast_weight;
      params.beta2 = config_.anvil_beta2;
      params.weight_decay = anvil_wd;
      params.num_maps = config_.anvil_num_maps;
      params.red_dim = group.red_dim;
      params.nesterov = true;
      kernels::AnvilUpdate(params, flat_ + group.flat_offset,
                           group.stacked_params, group.velocity,
                           group.lane_energy);
      ScatterAnvilParams(group);
    }
  }

  OptimizerConfig config_;
  Scheduler scheduler_;
  std::vector<ParamView> views_;
  std::vector<AdamWGroup> adamw_groups_;
  std::vector<MuonGroup> muon_groups_;
  std::vector<AnvilGroup> anvil_groups_;
  ComputeType* flat_ = nullptr;
  float* norm_dev_ = nullptr;
  std::int64_t total_count_ = 0;
  float grad_norm_ = 0.0f;
};

}  // namespace

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------
//
// `step` is 1-based (Optimizer::Step); nanochat's loop counter `it` is 0-based,
// so every schedule is evaluated at `it = step - 1`.

int Scheduler::WarmdownIters() const {
  const float value =
      config_.warmdown_ratio * static_cast<float>(config_.num_iterations);
  return static_cast<int>(std::lround(value));
}

float Scheduler::LrMultiplier(int step) const {
  const int it = step - 1;
  const int warmup = config_.warmup_steps;
  const int num_iters = config_.num_iterations;
  const int warmdown = WarmdownIters();
  if (warmup > 0 && it < warmup) {
    return static_cast<float>(it + 1) / static_cast<float>(warmup);
  }
  if (warmdown <= 0 || it <= num_iters - warmdown) return 1.0f;
  const float progress =
      static_cast<float>(num_iters - it) / static_cast<float>(warmdown);
  return progress + (1.0f - progress) * config_.final_lr_frac;
}

float Scheduler::MuonMomentum(int step) const {
  const int it = step - 1;
  const int num_iters = config_.num_iterations;
  const int warmdown = WarmdownIters();
  const int warmdown_start = num_iters - warmdown;
  const float warmup = config_.muon_momentum_warmup_steps;
  if (warmup > 0.0f && static_cast<float>(it) < warmup) {
    const float frac = static_cast<float>(it) / warmup;
    return (1.0f - frac) * config_.muon_momentum_start +
           frac * config_.muon_momentum_peak;
  }
  if (warmdown > 0 && it >= warmdown_start) {
    const float progress =
        static_cast<float>(it - warmdown_start) / static_cast<float>(warmdown);
    return config_.muon_momentum_peak * (1.0f - progress) +
           config_.muon_momentum_final * progress;
  }
  return config_.muon_momentum_peak;
}

float Scheduler::WeightDecay(int step) const {
  if (config_.num_iterations <= 0) return config_.weight_decay_base;
  const int it = step - 1;
  const float ratio =
      static_cast<float>(it) / static_cast<float>(config_.num_iterations);
  return config_.weight_decay_base * 0.5f * (1.0f + std::cos(kPi * ratio));
}

// ANVIL's fast-rail beta (docs/optimizer-anvil-design.md): linear warmup from
// `rail_beta_min` to `rail_beta_max`, flat, then a linear cooldown over the
// last `rail_beta_cooldown_steps`. Mirrors modded-nanogpt's `get_rail_beta`.
float Scheduler::RailBeta(int step) const {
  const int it = step - 1;
  const float warmup = config_.rail_beta_warmup_steps;
  const float cooldown = config_.rail_beta_cooldown_steps;
  const float beta_min = config_.rail_beta_min;
  const float beta_max = config_.rail_beta_max;
  if (warmup > 0.0f && static_cast<float>(it) < warmup) {
    const float frac = static_cast<float>(it) / warmup;
    return beta_min + frac * (beta_max - beta_min);
  }
  const float cd_start = static_cast<float>(config_.num_iterations) - cooldown;
  if (cooldown > 0.0f && static_cast<float>(it) > cd_start) {
    // Clamp to [0, 1]: a run shorter than the cooldown would otherwise drive
    // the fraction past 1 and the beta below `rail_beta_min`.
    const float raw = (static_cast<float>(it) - cd_start) / cooldown;
    const float frac = std::min(1.0f, std::max(0.0f, raw));
    return beta_max - frac * (beta_max - beta_min);
  }
  return beta_max;
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

// Optimizer state (docs/post-training.md section 7). The records live in
// `Checkpoint`, not in the public `Optimizer` surface, so `src/train.cc`
// moves them across the checkpoint boundary with these two functions.
void SaveOptimizerState(const Optimizer& optimizer, Checkpoint* checkpoint) {
  const auto* impl = dynamic_cast<const NanochatOptimizer*>(&optimizer);
  if (impl == nullptr) return;
  impl->SaveState(checkpoint);
}

bool LoadOptimizerState(const Checkpoint& checkpoint, Optimizer* optimizer) {
  auto* impl = dynamic_cast<NanochatOptimizer*>(optimizer);
  if (impl == nullptr) return false;
  return impl->LoadState(checkpoint);
}

// The optimizer-step record (docs/training-seam.md section 10). The container
// has no integer dtype, so one four-byte record carries the little-endian
// int32 step. `SaveOptimizerState` writes the group buffers; this writes the
// driver's counter beside them.
void SaveOptimizerStep(int step, Checkpoint* checkpoint) {
  if (checkpoint == nullptr) return;
  TensorRecord record;
  record.name = kOptimizerStepRecordName;
  record.dtype = DType::kFp32;
  record.shape.assign(1, 1);
  const std::uint32_t raw = static_cast<std::uint32_t>(step);
  record.data.resize(sizeof(std::uint32_t));
  record.data[0] = static_cast<std::byte>(raw & 0xffu);
  record.data[1] = static_cast<std::byte>((raw >> 8) & 0xffu);
  record.data[2] = static_cast<std::byte>((raw >> 16) & 0xffu);
  record.data[3] = static_cast<std::byte>((raw >> 24) & 0xffu);
  checkpoint->AddOptimizerState(std::move(record));
}

int LoadOptimizerStep(const Checkpoint& checkpoint) {
  const TensorRecord* record =
      checkpoint.FindOptimizerState(kOptimizerStepRecordName);
  if (record == nullptr || record->data.size() < sizeof(std::uint32_t)) {
    return 0;
  }
  const std::uint32_t raw =
      static_cast<std::uint32_t>(record->data[0]) |
      (static_cast<std::uint32_t>(record->data[1]) << 8) |
      (static_cast<std::uint32_t>(record->data[2]) << 16) |
      (static_cast<std::uint32_t>(record->data[3]) << 24);
  return static_cast<int>(static_cast<std::int32_t>(raw));
}

// Test-only introspection; declared (not frozen) by src/optim_test.cc.
bool OptimizerParamGroupForTest(const Optimizer* optimizer, const char* name,
                                int* kind, int* rows, int* cols, float* lr) {
  const auto* impl = dynamic_cast<const NanochatOptimizer*>(optimizer);
  if (impl == nullptr || name == nullptr) return false;
  return impl->GroupForTest(name, kind, rows, cols, lr);
}

std::unique_ptr<Optimizer> CreateOptimizer(Model* model,
                                           const OptimizerConfig& config,
                                           const Scheduler& scheduler) {
  return std::make_unique<NanochatOptimizer>(model, config, scheduler);
}

}  // namespace nanochat
