# The kernel seam (L1)

The backend contract is a fixed set of host-callable functions. Nine compute
families, one GEMM entry point, five device utilities. A backend is complete
when every symbol resolves and passes the oracle.

The rules that keep this seam vendor-free are in
[DESIGN.md §2](../DESIGN.md); the fusion rule that constrains what may be one
kernel is in [DESIGN.md §3](../DESIGN.md). A backend supplies this header; see
[backends.md](backends.md).

## Kernel inventory

| # | Family | Access pattern | Serves | fwd/bwd |
|---|---|---|---|---|
| 1 | `RmsNorm<kFusedResidual>` | row reduction + scale | pre-attn, pre-MLP, final norm; optional fused residual | yes/yes |
| 2 | `QkPrep` | row reduction + rotate + scale | QK-norm + RoPE + `*1.2` on q and k | yes/yes |
| 3 | `Attention` | T x T pairwise + softmax | causal, sliding window, GQA | yes/yes |
| 4 | `Pointwise` | pointwise, op code | relu^2, scale-add, gate-mul, scale, softcap | yes/yes |
| 5 | `Classifier` | row reduction + CE | logit softcap + cross-entropy + vocab slice | yes/yes |
| 6 | `Embedding` | index gather / scatter-add | `wte` and every `value_embeds[i]` | yes/yes |
| 7 | `AdamW` | stateful pointwise | embedding, head, scalar params | yes |
| 8 | `Muon` | stateful + batched GEMM | matrix params (Polar Express via cuBLAS) | yes |
| 9 | `GlobalNorm` | full reduction | gradient clipping | yes |

Library-backed (not hand-written kernels):

- `Gemm(mode)` — forward, dgrad, wgrad; cuBLAS/cuBLASLt on CUDA, reference on CPU.

Device utilities: `Alloc`, `Free`, `Memcpy(direction)`, `Memset`, `Synchronize`,
`GetCaps`.

## Kernel API sketch

```cpp
// include/nanochat/kernels.h
namespace nanochat {

enum class DType { kFp32, kFp16 };

struct RmsNormParams  { int rows; int dim; float eps; };
struct AttentionParams {
  int batch, seq, num_heads, num_kv_heads, head_dim;
  bool causal; int window_left, window_right;
};

enum class PointwiseOp {
  kScale,       // out = alpha * a
  kScaleAdd,    // out = alpha * a + beta * b
  kGateMul,     // out = sigmoid(a) * b
  kReluSquare,  // out = relu(a)^2
  kSoftcap,     // out = cap * tanh(a / cap)
};

namespace kernels {

// Device / memory (backend-owned).
void* Alloc(size_t bytes);
void  Free(void* ptr);
void  Memcpy(void* dst, const void* src, size_t bytes, CopyDir dir);
void  Memset(void* ptr, int value, size_t bytes);
void  Synchronize();
Caps  GetCaps();

// GEMM: enum GemmMode { kForward, kDgrad, kWgrad }.
void Gemm(GemmMode mode, const GemmParams&, const ComputeType* a,
          const ComputeType* b, ComputeType* c);

// Compute (fwd/bwd).
void RmsNormForward(const RmsNormParams&, const ComputeType* x,
                    ComputeType* out, float* rstd);
void RmsNormBackward(const RmsNormParams&, const ComputeType* x,
                     const ComputeType* dy, const float* rstd,
                     ComputeType* dx);
void QkPrepForward(const QkPrepParams&, const float* cos, const float* sin,
                   ComputeType* q, ComputeType* k);
void QkPrepBackward(const QkPrepParams&, const float* cos, const float* sin,
                    const ComputeType* dq, const ComputeType* dk,
                    ComputeType* q, ComputeType* k);
void AttentionForward(const AttentionParams&, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats);
void AttentionBackward(const AttentionParams&, const ComputeType* q,
                       const ComputeType* k, const ComputeType* v,
                       const float* stats, const ComputeType* dout,
                       ComputeType* dq, ComputeType* dk, ComputeType* dv);
void PointwiseForward(PointwiseOp op, int n, const ComputeType* a,
                      const ComputeType* b, float alpha, float beta,
                      ComputeType* out);
void PointwiseBackward(PointwiseOp op, int n, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy,
                       float alpha, float beta,
                       ComputeType* da, ComputeType* db);
void ClassifierForward(const ClassifierParams&, const ComputeType* logits,
                       const int* targets, ComputeType* losses);
void ClassifierBackward(const ClassifierParams&, const ComputeType* logits,
                        const int* targets, ComputeType* dlogits);
void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out);
void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, ComputeType* dtable);
void AdamWUpdate(int n, const AdamWParams&, ComputeType* p,
                 const ComputeType* g, float* m, float* v);
void MuonUpdate(const MuonParams&, const ComputeType* stacked_grads,
                ComputeType* stacked_params, float* buf1, float* buf2);
void GlobalNorm(int n, float clip, ComputeType* grads, float* out_norm);

}  // namespace kernels
}  // namespace nanochat
```

## Shared device helpers (code reuse without merging kernels)

```
backends/cuda/kernels/device_utils.cuh
  RowReduceSumSq(..)      // RmsNorm, QkPrep
  RsqrtScale(..)          // RmsNorm, QkPrep
  RotatePairInPlace(..)   // QkPrep (and Attention v2)
  OnlineSoftmaxTile(..)   // Attention forward
  SoftmaxGradTile(..)     // Attention backward
  RowLogSumExp(..)        // Classifier
  ScatterAddRow(..)       // Embedding backward
```

This removes duplication **without** changing kernel boundaries.
