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

Part 1 at the end of this page is the record of the Muon fusion: whether the
whole optimizer stage can run as one phase-parameterised kernel, what a
five-condition-legal fusion measured on this card, and why the multi-launch
cuBLAS path stays the default. The rule the result tests is the fusion boundary
rule below.

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

Helper: `ScalarDot`, a dot product over two buffers. Helper: `SumFp32`, the
sum of one fp32 buffer, which is the form the training loss needs because its
operand is fp32 in a bf16 build too.

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
| `AdamWUpdate`, `MuonUpdate` | stateful | required for training, bits `stateful_adamw` and `stateful_muon` | The optimizer, [model.md](model.md). One `MuonUpdate` call covers one shape group: 39 launches at the shipped `ns_steps` of 5 on the multi-launch path, or one cooperative launch of 84 blocks with 27 grid barriers (28 launches on the phase driver) on the fused path. The shape rule selects the multi-launch path for **every** group on this card, so the shipped launch count is 39; Part 1 has the measurement |
| `RmsNormForward`, `RmsNormBackward` | composition, plus fused override | — | `src/ops.cc`; a backend's single-kernel version is gated by `override_rms_norm` |
| `QkPrepForward`, `QkPrepBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_qk_prep` |
| `ClassifierForward`, `ClassifierBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_classifier` |
| `ValueGateForward`, `ValueGateBackward` | composition, plus fused override | — | `src/ops.cc`; no override bit exists for it yet |
| `SumFp32` | composition | — | The fp32 form of `ScalarDot` with no second operand; the training loss |
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
| the Polar Express chain: square GEMM -> `b = cb·a + cc·acc` -> apply GEMM -> `x = ca·x + prod` | **no, measured** | all five conditions hold and the kernel exists (Part 1), but the hand GEMM the fusion must carry reaches 0.68 to 0.90 of the cuBLAS rate, the whole fused step is 6 to 45 percent slower, and the worst group is 76 percent slower, so the boundary stays closed and the fold buys nothing to fold |

That last row is the one row of this table a campaign has measured rather than
argued. It is also the one where the structural verdict and the performance
verdict agree: condition 3 (no library displacement) fails, because the
arithmetic the fusion would own is arithmetic cuBLAS already wins.

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

Part 1 measured what a hand-written batched GEMM costs when a fusion needs one:
0.68 to 0.90 of the cuBLAS rate on the groups the fusion targets, on the card in
the tree. That is the number this rule stands on, and it is why the fused path
is selected for no shape.

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

**Measured corollary:** Part 1 is the optimizer-stage test of that argument. The
sixteen phases are legal, tested, and slower: one block size for every stage
holds 6 blocks per SM where the elementwise families' own kernels hold 8, so
even the `ns=0` step, which runs no GEMM at all and has nothing to fold, is 6 to
10 percent slower. That is the occupancy collapse this rule predicts, counted.

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
void SumFp32(const float* src, int n, float* out, double scale,
             bool accumulate);

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

## Part 1. The Muon fusion

_Absorbed from the design plan of record of the campaign that implemented it.
The section numbers are unchanged, so a reference to section K of that plan
resolves to section K here. The outcome, the defects, and the open items are in
[campaigns.md](campaigns.md), and the measured rows are in
[baselines.json](baselines.json) under `muon`._

Status: implemented, gated, and **not promoted**. The fused path is in
`backends/cuda/kernels/muon_fused.cu` and both drivers are tested; the shape
rule selects the multi-launch cuBLAS path for every group on the card in the
tree.

### 1. The question

One Muon step is a chain of dependent kernels. At the shipped `ns_steps` of 5
the Polar Express iteration alone is five launches -- `A = XᵀX` (or `XXᵀ`),
`A·A`, the elementwise `b = cb·a + cc·acc`, `prod = X·b` (or `b·X`), and the
elementwise `x = ca·x + prod` -- repeated five times, and the momentum, the row
equilibration, the two renormalisation statistics, and the NorMuon stages
surround it. The multi-launch path in `backends/cuda/kernels/muon.cu` issues 39
launches per shape group.

Can the whole stage be one device-side launch: the three products in one
batched GEMM of our own, both elementwise passes folded into the epilogues of
their neighbours, and a grid barrier between the stages? And is that faster on
the card in the tree?

The measured answer is two-part. The folds are legal, the arithmetic is right,
and the kernel works; it is also slower for every group at both profiles, so
the multi-launch path stays the default and the fused path stays in the tree as
the shape rule and the phase-bisection tool.

### 2. The tree it starts from

The multi-launch path runs the stage as one launch per operation: Nesterov
momentum, MuonEq row equilibration, Polar Express through strided-batched
`kernels::Gemm` (cuBLAS, one call per GEMM per group), Muon+ renormalisation,
NorMuon variance reduction with a factored second moment, and the cautious
decoupled weight decay. The math mirrors `backends/cpu/kernels.cc`.

Three properties of that path are what the fusion has to preserve, because they
are the family's contract:

- **Every whole-matrix statistic is reduced by many blocks per matrix.** The
  Frobenius norm, the equilibration divisor, and the renormalisation scale use
  `MuonPartialBlocks(mat)` blocks -- one per eight thousand elements, at least
  one and at most 32 -- and a second pass finishes the reduction, so no stage
  serialises a matrix on one block.
- **Every per-row or per-column statistic uses one block or one thread per row
  or column**, so the row equilibration and the NorMuon mean squares are also
  parallel over the matrix.
- **The two operand layouts are a shape property, not a flag fight.** Tall
  (`rows > cols`) is `A = XᵀX` of `cols x cols` and `prod = X B`; wide
  (`rows <= cols`) is `A = XXᵀ` of `rows x rows` and `prod = B X`. `A` and `b`
  therefore occupy `num_params` slices of `min(rows, cols)²`.

The GEMM is cuBLAS on that path because the compute strategy above says not to
hand-write one.

### 3. The arithmetic of the bet

The fusion trades two costs. It removes the launch gaps between the dependent
kernels and the elementwise HBM passes it folds away, and it pays for a
hand-written GEMM where cuBLAS was. A group is worth fusing when the first is
larger than the second; the design put a floor under the first, so a campaign
that finds the launch and elementwise share below **2 percent of the step**
stops and reports the negative result instead of writing the GEMM.

**The share, measured.** Three existing `muon_benchmark` rows give it without a
new kernel. `polar:<shape>:step` is the loop's fifteen batched GEMMs alone,
`muon:<shape>:step` at `ns=0` is the rest of the optimizer phase, and
`muon:<shape>:step` at `ns=5` is the whole step, so

```text
loop                 = muon:step(ns=5) - muon:step(ns=0)
launch + elementwise = loop - polar:step
share of the step    = (launch + elementwise) / muon:step(ns=5)
```

| Profile | `polar:step` | `muon:step` ns=0 | `muon:step` ns=5 | launch + elementwise | share of step |
|---|---|---|---|---|---|
| `h100` (8 layers, hidden 512, heads 8/4) | 30.93 ms | 6.29 ms | 42.84 ms | 5.62 ms | **13.1 %** |
| `big` (20 layers, hidden 1280, heads 10/5) | 1268.30 ms | 85.06 ms | 1462.76 ms | 109.41 ms | **7.5 %** |

A second session measured 3.81 ms (**9.2 %**) and 104.90 ms (**7.2 %**), so the
share is 9 to 13 percent at `h100` and 7.2 to 7.5 percent at `big`. Every value
is above the 2 percent floor, so the GEMM was written instead of the campaign
stopping on a negative result.

**The sync term, measured.** A dependent launch of a full batched grid costs
about 8.4 microseconds and a cooperative grid barrier about 0.9 microseconds at
the grid a fused kernel could hold resident (`muon_launch_sync_bench`), so one
fused iteration replaces 41.9 microseconds of launch gaps with 4.7 microseconds
of one launch plus three barriers: a **37.2 microsecond** gain per iteration.
That is 0.5 percent of an `h100` iteration and 0.01 percent of a `big` one. The
share above is therefore elementwise HBM traffic and not dispatch, which is what
fixes the shape of the bet: the folds have to remove the passes, and the hand
GEMM must not cost more than the passes save.

**Both terms, measured.** The device-side batched GEMM reached 0.68 to 0.72 of
the cuBLAS rate on the 640x1280 group the fusion targets and 0.83 to 0.90 on the
largest group, 1280x5120x1280. The gate asked for 0.85 on a targeted group, so
it failed. The whole fused step, measured against the multi-launch step, is the
same verdict: 6 to 10 percent slower at `ns=0`, 27 percent slower at `h100`
(`ns=5`), and 45 percent slower at `big` (`ns=5`). The measured loss is an order
of magnitude larger than the win, and the campaign records a negative result
rather than a promotion.

[performance.md](performance.md) section 2.1 holds the numbers under the
measurement protocol, and [baselines.json](baselines.json) holds the rows.

### 4. The design

**One device function, one block size, sixteen phases.** The fused path is one
`__device__` function parameterised by a phase index and driven from a flat
work-item index: a block strides over the items of the phase it is in, because
one cooperative grid serves every phase and fixes the block size, the grid size,
and the dynamic shared size for the whole kernel. 256 threads under
`__launch_bounds__(256, 3)`; the resident grid on this card is 84 blocks (3 per
SM on 28 SMs).

**The batched GEMM** is the measured Phase 1 schedule: block tile 128 by 64,
micro-tile 8 by 4 per thread (32 fp32 accumulators, a 16 by 16 thread grid), two
double-buffered 8-deep k slices in 12288 bytes of shared memory, and one flat
`(matrix, tile_i, tile_j)` index per block. `ptxas` reports 80 registers per
thread and no local-memory traffic in the inner loop (`#pragma unroll 2`;
unrolling the whole slice spills at this occupancy).

**All five operand forms work.** `transpose_a` and `transpose_b` select four of
them -- `XᵀX`, `XXᵀ`, `A·A`, and `X·B` -- and the wide branch `b·X` needs a
separate selector, `MuonSquareSide`, for the side the square operand sits on,
because the transpose pair cannot express it: `b·X` and `X·B` are both plain by
plain.

**The sixteen phases, and the twenty-eight executions.** Phases 1 to 7 run once,
8 to 10 run once per Polar Express iteration, and 11 to 16 run once:

| Phase | Work |
|---|---|
| 1 | Momentum and the Nesterov combination |
| 2 to 3 | Partial pass and finish for `‖x‖_F` |
| 4 | Row equilibration, one block per `(matrix, row)` |
| 5 to 6 | Partial pass and finish for `1.01‖x‖ + 1e-6` |
| 7 | Divisor scale |
| 8 | `A = XᵀX` (tall) or `XXᵀ` (wide), batched GEMM |
| 9 | `A·A`, with the `b = cb·a + cc·acc` epilogue |
| 10 | `X·b` (tall) or `b·X` (wide), with the `xnext = ca·x + prod` epilogue |
| 11 to 13 | Renormalisation: partials, finish, scale |
| 14 | NorMuon mean squares, per row or per column |
| 15 | The factored second moment and the per-index step |
| 16 | The NorMuon scale, with the cautious update as its epilogue |

Sixteen phases, 28 executions, and therefore 27 grid barriers in the cooperative
driver against 39 launches in the multi-launch path.

**Three epilogue folds, and the buffers they remove.** `b` is written from the
phase 9 accumulator, so the `a2` buffer and one pass over `min_dim²` per matrix
go. The combined iterate is written in place into the ping-pong partner by phase
10, so the `prod` buffer becomes that partner and one pass over `total` goes.
The cautious update is folded into the NorMuon scale, which is bit exact because
neither stage accumulates.

**Two drivers over one device body.** The **cooperative** driver launches one
grid per shape group, sized from the resident-block query in
`backends/cuda/device.h`, with a `grid.sync()` at each of the 27 barriers. A
launch the device refuses with `cudaErrorCooperativeLaunchTooLarge` falls back
to the phase driver and logs exactly one line. The **phase** driver launches one
grid per execution, each sized for its own phase: it is the simulator's path,
the fallback above, and the phase-bisection tool. The workspace, the arithmetic,
and the device body are identical in both drivers, so a group can move between
them without a caller noticing, and the two must agree bit for bit.

**The shape rule.** `MuonFusedSelected` keeps the multi-launch path
(`kernels::Gemm`, i.e. cuBLAS) for a group where the bet is poor and selects the
fused path where it is not. Its one number is `kMuonFusedMaxElements`, the
largest stacked group the fused path is selected for, in `ComputeType` elements.
That number is **0**, because the measurement says no group wins on this card;
promoting a group is one measurement and this constant.

### 5. The workspace

`EnsureMuonWorkspace` falls from `2·total + 3·min_total` to
`2·total + 2·min_total` `ComputeType` elements, and both paths use that one
layout (`MuonFusedWorkspaceElements`):

- two full-size slices are the iterate and its ping-pong partner -- which is why
  the multi-launch path's `prod` buffer doubles as that partner, and why
  `MuonUpdate` writes `A·A` into `b` in place, the same fold without the
  accumulator;
- two `min_dim x min_dim` slices are `A` and the folded `b`.

The `float` statistics scratch is `MuonFusedScratchFloats`: the per-matrix
statistic (`num_params`), then the NorMuon mean squares
(`num_params * max(rows, cols)`), then the NorMuon global scales
(`num_params`). The larger extent settles the size, so one allocation serves
both `red_dim` values of a group, and `MuonScratchOffset` rounds every region up
to a 16-byte group so the elementwise phases can move a whole vector of the
statistics and not only of the matrices.

The partial buffer is `num_params * MuonPartialBlocks(rows * cols)` doubles. Both
partial passes accumulate in double and both paths use the same block count and
the same chunk order, so every statistic stays within one unit in the last place
of the other path.

### 6. The gates

In the order they were run, all green:

1. **The CPU loop.** `tools/nanochat test` stays green. The CPU backend is the
   definition, and no `caps.json` changes.
2. **The multi-launch path is unchanged.** `//backends/cuda/kernels:muon_gpu_test`
   compares it with `testing/optim_ref.h` as before, now over the reduced
   workspace.
3. **Three implementations, one input, one seed.**
   `//backends/cuda/kernels:muon_fused_gpu_test` compares `optim_ref.h`, the
   shipped multi-launch `MuonUpdate`, and the fused path through each driver,
   over `4x3`, `3x6`, and `5x5` groups, both `red_dim` values, Nesterov on and
   off, `ns_steps` of 0, 1, and 5, and `num_params` of 1 and 3, at the family's
   `1e-5` tolerance. `5x5` is not a whole number of 16-byte groups, so it runs
   the scalar elementwise phases where the others run the vector ones.
4. **Both drivers, and the refusal.** The two drivers agree bit for bit (only
   the launch geometry differs); every output is finite, so a divergent stage is
   caught by name; the workspace is exactly `2·total + 2·min_total` elements with
   a 64-element sentinel tail, so a phase or a fold that reached past the last
   slice changes the tail; and a cooperative launch of eight times the resident
   grid is refused with `cudaErrorCooperativeLaunchTooLarge` and has to produce
   the phase driver's answer with `fell_back` reported.
5. **The simulator.** The mock answers 0 for `cudaDevAttrCooperativeLaunch` and
   1 for the occupancy query, so a simulated run takes the phase loop; the
   emulator runs the phase driver against the CPU reference, and cannot run the
   cooperative one, because its block-at-a-time engine would deadlock on a grid
   barrier. [simulator.md](simulator.md) section 5.2 lists the three runtime
   names.
6. **The benchmark.** `muon_benchmark` reports a `muon:` and a `muon_fused:` row
   for every group and for the `ns=0` and `ns=5` steps at both profiles, and the
   `driver=` field in the shape column names the driver the card answered with.
   That row pair is the verdict below.

### 7. What landed

`backends/cuda/kernels/muon_fused.cu` (the device function, the folds, and the
two drivers), `muon_fused.h` (the workspace helpers, the driver selection, and
`MuonFusedUpdate`), and `muon_fused_test.cc` (the gate above). `muon.cu` keeps
the one `MuonUpdate` seam entry, owns the workspace and the scratch, and selects
a path by shape. The frozen seam gains no entry point and no capability bit,
`caps.json` does not change, and `backends/cuda/kernels/BUILD.bazel` and its
`README.md` register the new test, the new source, and the benchmark rows. The
source is also compiled into the emulator library, so the phase driver runs on
the simulator.

### 8. The workstreams and the runtime names

The workstream table had one item without an owner: the device layer,
`backends/cuda`. A fused build needs three questions answered before it can
pick a driver, and all three are put to the device rather than to a capability
bit. That surface is `backends/cuda/device.h`:

| Helper | Answer |
|---|---|
| `LaunchCooperative` | The launch itself, on `Stream()`, beside `Launch` and under the same `CheckLastError` discipline |
| `CooperativeLaunch` outcome (`kLaunched`, `kTooLarge`, `kFailed`) with `ClassifyCooperativeLaunch` | A typed failure, so a caller recognises `cudaErrorCooperativeLaunchTooLarge` and falls back with one log line instead of parsing a string |
| `CooperativeLaunchSupported`, `SupportsCooperativeLaunch` | Whether the device accepts a cooperative launch at all, on the same discipline as `PrecisionFeatureBits` |
| `MultiprocessorCount`, `ActiveBlocksPerMultiprocessor`, `ResidentBlocks` | The persistent grid: active blocks per SM for that kernel, block size, and dynamic shared size, times the SM count. 84 on this card |

No capability bit, no `caps.json` field, and no change to `GetCaps`: the driver
asks the device, and a card that answers it with no takes the phase loop. The
simulator answers 0 for `cudaDevAttrCooperativeLaunch`, so it always takes it.

A fused build asks the runtime for **`cudaLaunchCooperativeKernel`,
`cudaDeviceGetAttribute`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** even when it takes the phase
driver, because the choice is made at run time and every one of the three is
reached before the choice is final. The mock must export all three or
`tools/cuda_sim/check_symbols.py` rejects it; [simulator.md](simulator.md)
section 5.2 records the answers and their consequence.

### 9. Risks and non-goals

- **One block size fits no stage.** 256 threads with `__launch_bounds__(256, 3)`
  and 15 KiB of static shared memory hold 6 blocks per SM where the elementwise
  families' own kernels hold 8, so the fused stage starts from a residence
  deficit that the folds have to pay back. They do not: the `ns=0` step, which
  runs no GEMM and has nothing to fold, is 6 to 10 percent slower. Measured.
- **The GEMM schedule is the bet, and it lost.** 80 registers per thread, 12 KB
  of shared memory, and 3 blocks per SM leave 768 of 2048 threads resident; the
  8 by 4 micro-tile reads 12 shared floats per 32 FMAs; an 8-deep k slice
  exposes one global round trip per ~1500 cycles of arithmetic per SM. An 8 by 8
  micro-tile, a 16- to 32-deep slice, or a 256 by 64 tile is a *different*
  prototype and a different measurement. Nothing here says the fusion is
  impossible, only that this schedule on this card is 30 percent off cuBLAS.
- **cuBLAS weakness masks the hand GEMM on the largest group.** At
  1280x5120x1280 the `A` stage measured 0.95 to 1.17 of cuBLAS, because that
  product is bandwidth bound and cuBLAS measured 4181 to 5608 GFLOP/s on it. A
  rate gate against cuBLAS is therefore not a selector by itself; the end-to-end
  step row is.
- **The bf16 build stays on the multi-launch path.** The fused schedule is the
  fp32 one and loads `float4` quads, so `MuonFusedSelected` reports false there.
- **Non-goals.** No capability bit, no `caps.json` change, no change to the
  frozen seam, no whole-model megakernel (see the compute strategy above), and
  no cooperative driver in the emulator, whose block-at-a-time engine would
  deadlock on a grid barrier.
