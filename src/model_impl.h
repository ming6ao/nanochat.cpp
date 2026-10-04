#ifndef NANOCHAT_SRC_MODEL_IMPL_H_
#define NANOCHAT_SRC_MODEL_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/model.h"
#include "src/ops.h"
#include "src/workspace.h"

// Concrete training model: the fixed train graph (forward, backward, optimizer
// step) from docs/model.md. The topology is a plain hand-written sequence of
// `ops` calls; there is no autograd engine and no graph IR.
//
// This header is private to `src/`: the rest of the tree talks to `Model`. It
// is public here only so the oracle/gradient tests under `src/` can read the
// saved logits, inspect the parameter views, and so `generate.cc` can reach the
// frozen weights and rotary tables for the inference graphs.

namespace nanochat {

class TrainModel final : public Model {
 public:
  explicit TrainModel(const Config& config);
  ~TrainModel() override;

  void InitWeights(std::uint64_t seed) override;
  float ForwardLoss(const int* tokens, const int* targets, int batch,
                    int seq) override;
  void Backward() override;
  void ZeroGrad() override;
  void BackwardAccumulate(float scale) override;
  float TrainStep(const int* tokens, const int* targets, int batch, int seq,
                  Optimizer* optimizer) override;
  void SetGradEnabled(bool enabled) override { grad_enabled_ = enabled; }
  bool grad_enabled() const override { return grad_enabled_; }
  std::vector<ParamView> params() const override;
  void Save(const std::string& path) const override;
  void Load(const std::string& path) override;

  // --- test/debug accessors (not part of the public Model API) ---
  const ComputeType* raw_logits() const { return raw_logits_; }
  // Per-row cross-entropy (nats) saved by the most recent ForwardLoss; length
  // `last_batch() * last_seq()`. `EvalBpb` sums these before dividing by bytes.
  // The grad-mode forward writes the eval-mode buffer instead.
  const ComputeType* losses() const { return ActiveLosses(); }
  int last_batch() const { return batch_; }
  int last_seq() const { return seq_; }
  // Bytes reserved by the arena the active grad mode uses (docs/grad-mode.md).
  // The test compares the training and evaluation arenas through this accessor.
  std::size_t workspace_bytes() const {
    return grad_enabled_ ? workspace_.bytes_reserved()
                         : eval_workspace_.bytes_reserved();
  }
  // Test accessor: shrink the eval classifier chunk to exercise the tiling
  // path. The production budget is 512 MiB.
  void SetEvalLogitsBudgetForTest(std::int64_t bytes) {
    eval_logits_budget_ = bytes;
  }

  // --- inference accessors (generate.cc) -------------------------------------
  // The prefill/decode topology lives in generate.cc; these expose only the
  // frozen weights, scalars, and rotary tables it needs. No topology leaks in.
  const ComputeType* wte() const { return wte_; }
  const ComputeType* lm_head() const { return lm_head_; }
  const ComputeType* smear_gate() const { return smear_gate_; }
  const ops::BlockWeights& layer_weights(int layer) const {
    return lweights_[static_cast<std::size_t>(layer)];
  }
  float resid(int layer) const;
  float x0_lambda(int layer) const;
  float smear_lambda() const;
  float backout_lambda() const;
  const float* cos_table() const { return cos_table_dev_; }
  const float* sin_table() const { return sin_table_dev_; }
  // Inference may run past the last training sequence length, so grow the
  // rotary table on demand before reading it.
  void EnsureRopeCapacity(int seq);

 private:
  struct Param {
    std::string name;
    ComputeType* value = nullptr;
    ComputeType* grad = nullptr;
    std::int64_t count = 0;
    int rows = 0;
    int cols = 0;
  };

  Param& AddParam(const std::string& name, std::int64_t count, int rows,
                  int cols);
  void BuildTrainWorkspace(int batch, int seq);
  void BuildEvalWorkspace(int batch, int seq);
  // The shared forward topology for both grad modes (docs/grad-mode.md). When
  // `save_for_backward` is true, it writes every layer's activations to
  // `lacts_[i]` and the full classifier logits to `raw_logits_`. When false, it
  // reuses one block set and tiles the classifier.
  void RunForward(const int* tokens, const int* targets, int batch, int seq,
                  bool save_for_backward);
  // Aborts when the most recent forward did not save activations for Backward.
  void RequireGradActivations(const char* caller) const;
  const ComputeType* ActiveLosses() const {
    return grad_enabled_ ? losses_ : eval_losses_;
  }
  void BuildRope(int seq);

  // Parameters, in the optimizer's grouping order (docs/optimizer.md).
  std::vector<Param> params_;
  std::vector<ops::BlockWeights> lweights_;
  std::vector<ops::BlockGrads> lgrads_;
  std::vector<ops::BlockActivations> lacts_;

  ComputeType* wte_ = nullptr;
  ComputeType* wte_grad_ = nullptr;
  ComputeType* lm_head_ = nullptr;
  ComputeType* lm_head_grad_ = nullptr;
  ComputeType* resid_ = nullptr;
  ComputeType* resid_grad_ = nullptr;
  ComputeType* x0_lambda_ = nullptr;
  ComputeType* x0_lambda_grad_ = nullptr;
  ComputeType* smear_gate_ = nullptr;
  ComputeType* smear_gate_grad_ = nullptr;
  ComputeType* smear_lambda_ = nullptr;
  ComputeType* smear_lambda_grad_ = nullptr;
  ComputeType* backout_lambda_ = nullptr;
  ComputeType* backout_lambda_grad_ = nullptr;

  // Global activations and gradient scratch, all carved from `workspace_`.
  ComputeType* emb_raw_ = nullptr;
  ComputeType* emb_norm_ = nullptr;
  ComputeType* x0_ = nullptr;
  ComputeType* x_backout_ = nullptr;
  ComputeType* x_final_pre_ = nullptr;
  ComputeType* x_final_norm_ = nullptr;
  float* rstd_emb_ = nullptr;
  float* rstd_final_ = nullptr;
  float* smear_sig_ = nullptr;
  ComputeType* raw_logits_ = nullptr;
  ComputeType* dlogits_ = nullptr;
  ComputeType* losses_ = nullptr;
  ComputeType* dx_final_norm_ = nullptr;
  ComputeType* dx_final_pre_ = nullptr;
  ComputeType* dbackout_ = nullptr;
  ComputeType* g_a_ = nullptr;
  ComputeType* g_b_ = nullptr;
  ComputeType* de_ = nullptr;
  ComputeType* demb_ = nullptr;
  ComputeType* x0_acc_ = nullptr;
  ComputeType* scratch_a_ = nullptr;
  ComputeType* scratch_b_ = nullptr;

  ops::BlockScratch block_scratch_;

  // Grad-mode state and the evaluation workspace (docs/grad-mode.md). The
  // evaluation arena is separate from `workspace_` so an online validation
  // call does not reallocate the training arena. The eval block set is reused
  // by every layer because the eval path saves nothing for a backward pass.
  bool grad_enabled_ = true;
  bool have_grad_activations_ = false;
  Workspace eval_workspace_;
  ComputeType* eval_emb_raw_ = nullptr;
  ComputeType* eval_emb_norm_ = nullptr;
  ComputeType* eval_x0_ = nullptr;
  ComputeType* eval_x_ = nullptr;
  ComputeType* eval_x_backout_ = nullptr;
  ComputeType* eval_x_final_pre_ = nullptr;
  ComputeType* eval_x_final_norm_ = nullptr;
  ComputeType* eval_logits_ = nullptr;
  ComputeType* eval_losses_ = nullptr;
  float* eval_rstd_emb_ = nullptr;
  float* eval_rstd_final_ = nullptr;
  float* eval_smear_sig_ = nullptr;
  ops::BlockActivations eval_block_;
  int eval_batch_ = 0;
  int eval_seq_ = 0;
  std::int64_t eval_chunk_rows_ = 0;
  std::int64_t eval_logits_budget_ = static_cast<std::int64_t>(512) << 20;

  // Host staging buffers for the rotary tables; the device copies are what the
  // kernels read (`cos_table()`/`sin_table()`). The host vectors exist only so
  // `BuildRope` can compute the tables cheaply and upload them once.
  std::vector<float> cos_table_;
  std::vector<float> sin_table_;
  float* cos_table_dev_ = nullptr;
  float* sin_table_dev_ = nullptr;
  std::size_t rope_capacity_ = 0;

  // Host staging for the per-token losses so `ForwardLoss` can sum them
  // without dereferencing device memory.
  std::vector<ComputeType> losses_host_;

  // Host copy of the current batch, so Backward() (which takes no arguments)
  // sees exactly the ids ForwardLoss() ran with.
  std::vector<int> tokens_;
  std::vector<int> targets_;

  Workspace workspace_;
  int batch_ = 0;
  int seq_ = 0;
  std::int64_t rows_ = 0;
  int backout_layer_ = 0;
  // 1-based count of TrainStep() optimizer updates, so the schedules see the
  // same step numbering as an explicit training loop.
  int optimizer_step_ = 0;
};

// Internal inference entry points used by `src/generate_test.cc`. The public
// `Prefill`/`Decode` discard the logits; these expose the soft-capped
// last-position logits so the test can prove that decode matches a full
// forward. They live in generate.cc and are not part of the public Model API.
void PrefillLogits(Model* model, const int* tokens, int num_tokens, KvCache* kv,
                   float* logits_out);
void DecodeLogits(Model* model, int token, KvCache* kv, float* logits_out);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MODEL_IMPL_H_
