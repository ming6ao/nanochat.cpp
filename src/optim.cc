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

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "nanochat/scheduler.h"
#include "nanochat/tensor.h"

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

    for (AdamWGroup& group : adamw_groups_) {
      for (std::size_t i = 0; i < group.params.size(); ++i) {
        AdamWParams params;
        params.lr = group.lr * lrm;
        params.beta1 = group.beta1;
        params.beta2 = group.beta2;
        params.eps = group.eps;
        params.weight_decay = group.weight_decay;
        params.step = step;
        kernels::AdamWUpdate(static_cast<int>(group.params[i].count), params,
                             group.params[i].value,
                             flat_ + group.grad_offsets[i],
                             group.m + group.moment_offsets[i],
                             group.v + group.moment_offsets[i]);
      }
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

    // Lay out the flat gradient buffer: AdamW groups first, then Muon groups.
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

  OptimizerConfig config_;
  Scheduler scheduler_;
  std::vector<ParamView> views_;
  std::vector<AdamWGroup> adamw_groups_;
  std::vector<MuonGroup> muon_groups_;
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

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

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
