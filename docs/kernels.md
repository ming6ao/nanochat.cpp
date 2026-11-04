# The kernel seam (L1)

The backend contract is a fixed set of host-callable functions: device
utilities, arithmetic primitives, and stateful entries. A backend is complete
when every symbol it is required to supply resolves and passes the oracle. The
next section says which symbols a bring-up owes ([Kernel tiers](#kernel-tiers)).

The rules that keep this seam vendor-free are the [dependency
invariants](build.md#dependency-invariants). The fusion rule that constrains
what may be one kernel, and the compute strategy that follows from it, are the
two sections below. A backend supplies this header; see [build.md](build.md).
Which entry point each tier holds, and why, is
[model-and-device-bringup.md §7](model-and-device-bringup.md) with its
decision records in [../spec/decisions.md](../spec/decisions.md).

## Kernel inventory

| # | Family | Access pattern | Serves | fwd/bwd |
|---|---|---|---|---|
| 1 | `RmsNorm<kFusedResidual>` | row reduction + scale | pre-attn, pre-MLP, final norm; optional fused residual | yes/yes |
| 2 | `QkPrep` | row reduction + rotate + scale | QK-norm + RoPE + `*1.2` on q and k | yes/yes |
| 3 | `Attention` | T x T pairwise + softmax | causal, sliding window, GQA | yes/yes |
| 4 | `Pointwise` | pointwise, op code | relu^2, scale-add, gate-mul, scale, softcap, `exp`, `rsqrt`, `log`, row scale | yes/yes |
| 5 | `Classifier` | row reduction + CE | logit softcap + cross-entropy + vocab slice | yes/yes |
| 6 | `Embedding` | index gather / scatter-add | `wte` and every `value_embeds[i]` | yes/yes |
| 7 | `AdamW` | stateful pointwise | embedding, head, scalar params | yes |
| 8 | `Muon` | stateful + batched GEMM | matrix params (Polar Express via cuBLAS) | yes |
| 10 | `GradNorm` | multi-span full reduction | gradient norm and the clip scale | yes |
| 11 | `ValueGate` | row gate + per-head projection | value residual gate | yes/yes |
| 12 | `Smear` | row gate + previous-row mix | learned bigram residual on the token embedding | yes/yes |

Library-backed (not hand-written kernels):

- `Gemm(mode)` — forward, dgrad, wgrad; cuBLAS on CUDA, reference on CPU.

Helper: `ScalarDot`, a dot product over two buffers.

Device utilities: `Alloc`, `Free`, `Memcpy(direction)`, `Memset`, `Synchronize`,
`GetCaps`.

The table above is the **family** view: it names the twelve compute families and
says which ones a backend may serve with a fused kernel. Which *entry point* of
each family a device must supply is the tier question, and the answer differs
per family: a family in the composition tier is supplied once, above the seam,
for every device. The next section is that partition.

## Kernel tiers

Every entry point in `include/nanochat/kernels.h` belongs to exactly one tier,
and the tier says who supplies it. The partition is the one
[model-and-device-bringup.md §7.3](model-and-device-bringup.md) fixes. The
reasons are the decision records in [../spec/decisions.md](../spec/decisions.md),
which this table cites and does not restate.

| Entry point | Tier | Gate | Serves |
|---|---|---|---|
| `Alloc`, `Free`, `Memcpy`, `Memset`, `Synchronize`, `GetCaps` | primitive | required | The six device utilities |
| `Gemm` | primitive | required | Every projection and the classifier; cuBLAS on CUDA, a reference loop on the CPU |
| `GemmWgrad` | primitive | required | The weight gradient, accumulated into fp32 parameter storage |
| `RowReduce` | primitive | required | `RmsNorm`, `QkPrep`, and the `Classifier`'s per-row reduction. Additive at seam 1 → 2 |
| `SpanReduce` | primitive | required | `GradNorm`, `ScalarDot`, the loss reduction. Additive at seam 1 → 2 |
| `AttentionForward`, `AttentionBackward` | primitive | required | The attention graph; causal, group-query, full context and sliding window |
| `PointwiseForward`, `PointwiseBackward` | primitive | required | The op codes, elementwise; one broadcasts a per-row vector and one applies the half-split pair rotation. Three codes are additive at seam 2 → 3 |
| `EmbeddingForward` | primitive | required | The gather over `wte` and every `value_embeds[i]` |
| `EmbeddingBackward` | primitive | required | The scatter-add into the dense gradient buffer |
| `SmearForward`, `SmearBackward` | primitive | required | The bigram residual, one launch per direction on a device; the backward reduces its two parameter gradients at a phase boundary in the same launch |
| `QuantizeFp8`, `GemmScaledFp8` | primitive | optional, bits `fp8_storage` and `fp8_gemm` | The fp8 tier, [fp8-training.md](fp8-training.md) |
| `AdamWUpdate`, `MuonUpdate` | stateful | required for training, bits `stateful_adamw` and `stateful_muon` | The optimizer, [model.md](model.md) |
| `RmsNormForward`, `RmsNormBackward` | composition, plus fused override | — | `src/ops.cc`; a backend's single-kernel version is gated by `override_rms_norm` |
| `QkPrepForward`, `QkPrepBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_qk_prep` |
| `ClassifierForward`, `ClassifierBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_classifier` |
| `ValueGateForward`, `ValueGateBackward` | composition, plus fused override | — | `src/ops.cc`; no override bit exists for it yet |
| `GradNorm` | composition | — | `SpanReduce` across several spans |
| `ScalarDot` | composition | — | `SpanReduce` over two buffers, or the backend's dot |

A composition supplies its own backward as one unit, because
[model.md](model.md) § No autograd engine requires a fused backward to be
derived and tested as one unit.

**The two reductions are additive, and they raise the seam version.**
`RowReduce` and `SpanReduce` are new symbols, and `kKernelSeamVersion` in the
header records the rise. The header preamble's changelog names exactly what the
rise covers: the two reductions, and the pointwise argument list becoming
`PointwiseParams` with the `kExp`, `kRsqrt`, `kLog` and `kRowScale` codes. No
symbol was retracted and no symbol changed meaning, so an older backend adds the
two reductions and updates its two pointwise signatures, and nothing else. That
the tier gained two entries instead of one descriptor with three modes is
[D-001](../spec/decisions.md); that the per-row broadcast is an op code and not
a third entry point is [D-002](../spec/decisions.md); that the accuracy bound is
per code and lives in [pointwise_ops.md](../spec/pointwise_ops.md) is
[D-003](../spec/decisions.md).

**Required bring-up cost: the six utilities and every arithmetic primitive.** A
device at that tier runs every graph in [model.md](model.md) on the composition
path without writing one fused kernel. Training adds the two stateful entries.
The two fp8 entries are declared and never assumed: a `caps.json` carries
`fp8_storage` and `fp8_gemm` only where the silicon has the formats, and the
model then either runs the dense path or fails with the missing bit named.
Before this partition, a device implemented every entry point of the seam, and
every later fused family added a frozen-header entry point that every device
then owed.

**The required `Attention` variant is causal, group-query, full context *and*
sliding window, at one tile size.** Sliding window is required and not optional:
`models/nanochat-default/spec.json` names it on nine of its twelve blocks, and
`Config::window_pattern` derives `"SSSL"` from that list, so `window_left()` is
non-empty on every `S` layer. A device that implements full-context attention
only cannot run the one model in the tree, so it is non-conformant, not partly
conformant. The mask is a field of the block and not a second entry point: the
two variants share the arithmetic, the head mapping, the saved slots, and the
tolerance ([block_kinds.md](../spec/block_kinds.md) §2), so one kernel covers
both and the window is a parameter. *One tile size* bounds the obligation: the
tier promises the primitives at one tiling, not every tiling a future shape
might prefer, and a second tile size is device tuning rather than a contract
([capability_bits.md](../spec/capability_bits.md) §5).

## The fusion boundary rule

Fuse the ops `A -> B` with the intermediate `Y` only when all five conditions
hold:

1. **Decomposition match** — one (grid, block) configuration covers both ops,
   and the two iteration spaces map onto the same CTA and thread ownership.
2. **Residency** — the part of `Y` that `B` needs fits in registers and shared
   memory at that configuration, so the HBM round trip is really avoided.
3. **No library displacement** — neither `A` nor `B` is better served by cuBLAS
   or cuDNN, which cannot join the fusion.
4. **Occupancy survives** — the register and shared usage of the fused kernel
   does not drop occupancy past the break-even point.
5. **Derivable backward** — the fused backward is derivable and testable, under
   the policy in [model.md](model.md) § No autograd engine.

Consequences for the one model in the tree:

| Boundary | Fuse? | Reason |
|---|---|---|
| `RmsNorm -> RoPE -> scale` on q and k | **yes** (`QkPrep`) | row-local, on-chip, no library |
| `residual add -> RmsNorm` | **yes** | row-local, on-chip |
| `softcap -> cross-entropy` | **yes** (`Classifier`) | row-local over the vocabulary |
| `relu^2 -> c_proj` GEMM | no | a cuBLASLt epilogue cannot express relu^2 |
| `c_q` GEMM -> `RmsNorm(q)` | no | GEMM tiles against a whole-row reduction |
| `attention` | separate | own T x T pattern |
| `embedding gather` | separate | indexed access |
| GEMM -> GEMM | no | cuBLAS owns both |

Performance **validates** a boundary. Performance does not choose a boundary.
Structure decides which boundaries are legal: the decomposition, the residency,
and the library ownership.

### Candidate v2 fusion

`QkPrep` can fold into the `Attention` kernel, because the RMSNorm reduction
runs over `head_dim`, which one tile row holds. The reduction does not span
CTAs, so the rule permits the fold. Defer it to v2, because the fold couples the
attention backward to the norm and rope backward.
`backends/cuda/fused_overrides.cc` already declares `kFeatureFusedAttention` as
the v2 candidate, and `backends/cuda/fused_override_trace_test.cc` holds the
reference case.

## Compute strategy

### GEMM

Keep cuBLAS as the GEMM engine. It is near peak on GP102 (11.34 TFLOPS fp32), it
is tuned per shape, and it is portable across generations. Do not hand-write a
GEMM.

### Launch overhead

At `d8_s512` a full step takes about 0.9 s, and kernel launches stay under 1% of
that time. CUDA Graphs are therefore an optional internal optimization of the
CUDA backend and not an architectural element. Add them only when a profile
demands them ([performance.md](performance.md)).

### No megakernels for training

A whole-block megakernel needs a hand-written GEMM, because a device-side kernel
cannot call cuBLAS. It also collapses occupancy, it serializes the phases behind
a grid barrier, and it still pays the HBM round trip across a tile boundary.
That is the wrong trade for throughput-bound training.

**Exception:** the decode graph is batch-1 and latency-bound. A fused persistent
decode kernel is therefore a legitimate `dev/kernels/` experiment, on hardware
with matrix-multiply instructions (Turing and newer). It must beat a
`cuBLAS + CUDA Graphs` baseline before promotion, and it does not apply to
Pascal, which has no `mma` instruction.

### Direct-MMA and persistent kernels

A direct `mma` kernel and a persistent kernel do not apply to Pascal, because
sm_61 has no tensor core and no `mma` or `wmma` instruction. The gain in a
system such as Mirage MPK is whole-model inference fusion on Hopper, and not
GEMM throughput. Keep cuBLAS, and revisit this only for decode on sm_75 and
newer.

## Kernel API sketch

```cpp
// include/nanochat/kernels.h
namespace nanochat {

enum class DType { kFp32, kBf16, kInt8, kFp8E4M3, kFp8E5M2 };

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
  kSoftcap,     // out = alpha * tanh(a / alpha)
  kExp,         // out = exp(a)
  kRsqrt,       // out = 1 / sqrt(a)
  kLog,         // out = log(a)
  kRowScale,    // out[i] = alpha * a[i] * b[i % row_width]
};

enum class ReduceOp { kSumSq, kSum, kMax, kLogSumExp };  // the combine op

struct PointwiseParams {  // replaces the former scalar argument list
  PointwiseOp op = PointwiseOp::kScale;
  int count = 0;
  int row_width = 0;  // width of the row a broadcast reads; 0 => count
  float alpha = 1.0f;
  float beta = 0.0f;
};

struct RowReduceParams {
  int n_out = 0;            // segments, one output scalar each
  int count = 0;            // elements per segment
  std::int64_t stride = 1;  // element stride inside a segment
  std::int64_t offset = 0;  // element offset of the first segment
  ReduceOp op = ReduceOp::kSumSq;
};

struct SpanReduceParams {
  int num_spans = 0;
  const int* counts = nullptr;  // device array: elements per span, in order
  float scale = 1.0f;
  bool accumulate = false;
};

namespace kernels {

// Device / memory (backend-owned).
void* Alloc(std::size_t bytes);
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
void PointwiseForward(const PointwiseParams&, const ComputeType* a,
                      const ComputeType* b, ComputeType* out);
void PointwiseBackward(const PointwiseParams&, const ComputeType* a,
                       const ComputeType* b, const ComputeType* dy,
                       ComputeType* da, ComputeType* db);
void ClassifierForward(const ClassifierParams&, const ComputeType* logits,
                       const int* targets, float* losses);
void ClassifierBackward(const ClassifierParams&, const ComputeType* logits,
                        const int* targets, ComputeType* dlogits);
void EmbeddingForward(int tokens, int dim, const int* ids,
                      const ComputeType* table, ComputeType* out);
void EmbeddingBackward(int tokens, int dim, const int* ids,
                       const ComputeType* dout, float* dtable);
void AdamWUpdate(int n, const AdamWParams&, float* master, ComputeType* value,
                 const float* g, float* m, float* v);
void MuonUpdate(const MuonParams&, const float* stacked_grads,
                float* stacked_master, ComputeType* stacked_values, float* buf1,
                float* buf2);
void GradNorm(const float* const* grads, const int* counts, int num_spans,
              float clip, float* out_norm, float* out_scale);

// The two reductions.
void RowReduce(const RowReduceParams&, const ComputeType* a, float* out);
void SpanReduce(const SpanReduceParams&, const ComputeType* const* spans,
                const ComputeType* const* other, float* out);
void ScalarDot(const ComputeType* a, const ComputeType* b, int n, float* out,
               float scale, bool accumulate);

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
