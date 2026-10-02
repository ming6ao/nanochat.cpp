#ifndef NANOCHAT_SRC_MODEL_IMPL_H_
#define NANOCHAT_SRC_MODEL_IMPL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "nanochat/model.h"
#include "workspace.h"

// Concrete training model: the fixed train graph (forward, backward, optimizer
// step) from docs/model.md. It is a plain hand-written kernel sequence, not an
// autograd engine and not a graph IR.
//
// This header is private to `src/`: the rest of the tree talks to `Model`. It
// is public here only so the oracle test under `src/` can read the saved logits
// and inspect the parameter views.

namespace nanochat {

class TrainModel final : public Model {
 public:
  explicit TrainModel(const Config& config);
  ~TrainModel() override;

  void InitWeights(std::uint64_t seed) override;
  float ForwardLoss(const int* tokens, const int* targets, int batch,
                    int seq) override;
  void Backward() override;
  float TrainStep(const int* tokens, const int* targets, int batch, int seq,
                  Optimizer* optimizer) override;
  std::vector<ParamView> params() const override;
  void Save(const std::string& path) const override;
  void Load(const std::string& path) override;

  // --- test/debug accessors (not part of the public Model API) ---
  const ComputeType* raw_logits() const { return raw_logits_; }
  int last_batch() const { return batch_; }
  int last_seq() const { return seq_; }

 private:
  struct Param {
    std::string name;
    ComputeType* value = nullptr;
    ComputeType* grad = nullptr;
    std::int64_t count = 0;
    int rows = 0;
    int cols = 0;
  };

  // Activations saved for the hand-written backward pass, one set per layer.
  struct LayerActs {
    ComputeType* xr = nullptr;       // resid_lambda*x + x0_lambda*x0
    ComputeType* h = nullptr;        // norm(xr), the attention input
    float* rstd1 = nullptr;          // pre-attention norm statistics
    ComputeType* q_pre = nullptr;    // saved pre-QkPrep query projection
    ComputeType* k_pre = nullptr;    // saved pre-QkPrep key projection
    ComputeType* q_final = nullptr;  // post-QkPrep query (also dq on backward)
    ComputeType* k_final = nullptr;  // post-QkPrep key (also dk on backward)
    ComputeType* ve_values = nullptr;  // gathered value embedding
    ComputeType* ve_gate = nullptr;    // 3*sigmoid(ve_gate(x[..., :12]))
    ComputeType* v_final = nullptr;    // value projection + gate * ve
    float* attn_stats = nullptr;       // (max, sum_exp) per query row
    ComputeType* attn_out = nullptr;   // attention output, flattened
    ComputeType* x_mid = nullptr;      // xr + attention output
    ComputeType* h2 = nullptr;         // norm(x_mid)
    float* rstd2 = nullptr;
    ComputeType* pre_act = nullptr;    // c_fc output (relu^2 input)
    ComputeType* act = nullptr;        // relu^2 output
    ComputeType* x_out = nullptr;      // layer output, the next layer input
    ComputeType* dq = nullptr;         // attention query gradient scratch
    ComputeType* dk = nullptr;
    ComputeType* dv = nullptr;
  };

  struct LayerParams {
    ComputeType* c_q = nullptr;
    ComputeType* c_q_grad = nullptr;
    ComputeType* c_k = nullptr;
    ComputeType* c_k_grad = nullptr;
    ComputeType* c_v = nullptr;
    ComputeType* c_v_grad = nullptr;
    ComputeType* c_proj_attn = nullptr;
    ComputeType* c_proj_attn_grad = nullptr;
    ComputeType* ve_gate = nullptr;
    ComputeType* ve_gate_grad = nullptr;
    ComputeType* value_embeds = nullptr;
    ComputeType* value_embeds_grad = nullptr;
    ComputeType* c_fc = nullptr;
    ComputeType* c_fc_grad = nullptr;
    ComputeType* c_proj_mlp = nullptr;
    ComputeType* c_proj_mlp_grad = nullptr;
    bool has_ve = false;
  };

  Param& AddParam(const std::string& name, std::int64_t count, int rows,
                  int cols);
  void BuildWorkspace(int batch, int seq);
  void BuildRope(int seq);

  // Forward sub-graphs that do not map cleanly onto a single kernel (scalar
  // gates and their broadcasting), plus the reverse-order layer body.
  void SmearForward(int batch, int seq);
  void SmearBackward(int batch, int seq, const ComputeType* d_x0);
  void VeGateForward(int layer, std::int64_t rows);
  void VeGateBackward(int layer, std::int64_t rows);
  void LayerBackward(int layer, const ComputeType* dout, ComputeType* dx_in,
                     std::int64_t rows);

  // Parameters, in the optimizer's grouping order (docs/optimizer.md ready).
  std::vector<Param> params_;
  std::vector<LayerActs> layers_;
  std::vector<LayerParams> lparams_;

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
  ComputeType* dh_ = nullptr;
  ComputeType* dxr_ = nullptr;
  ComputeType* dx_mid_ = nullptr;
  ComputeType* dh2_ = nullptr;
  ComputeType* dy_attn_ = nullptr;
  ComputeType* dact_ = nullptr;
  ComputeType* dpre_ = nullptr;
  ComputeType* de_ = nullptr;
  ComputeType* demb_ = nullptr;
  ComputeType* dve_ = nullptr;
  ComputeType* x0_acc_ = nullptr;
  ComputeType* scratch_a_ = nullptr;
  ComputeType* scratch_b_ = nullptr;

  std::vector<float> cos_table_;
  std::vector<float> sin_table_;

  // Host copy of the current batch, so Backward() (which takes no arguments)
  // sees exactly the ids ForwardLoss() ran with.
  std::vector<int> tokens_;
  std::vector<int> targets_;

  Workspace workspace_;
  int batch_ = 0;
  int seq_ = 0;
  std::int64_t rows_ = 0;
  int backout_layer_ = 0;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_MODEL_IMPL_H_
