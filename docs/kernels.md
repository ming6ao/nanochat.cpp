# The kernel seam

The backend contract is a fixed set of host-callable functions: device
utilities, arithmetic primitives, and stateful entries. A backend is complete
when every symbol it is required to supply resolves and passes the oracle.
Section [Kernel tiers](#kernel-tiers) says which symbols a bring-up owes.

A backend supplies `include/nanochat/kernels.h` and nothing else on the compute
path. The header is frozen, so a change to it is a contract change with a
version bump. A device declares its capabilities in `caps.json`, and the
planner reads that file, the model description, and the seam version.

The rules that keep this seam vendor-free are the
[dependency invariants](build.md#dependency-invariants). The fusion rule that
constrains what may be one kernel, and the compute strategy that follows from
it, are the two sections below. Which entry point each tier holds, and why, is
[model-and-device-bringup.md §7](model-and-device-bringup.md) with its decision
records in [../spec/decisions.md](../spec/decisions.md).

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
for every device.

## Kernel tiers

Every entry point in `include/nanochat/kernels.h` belongs to exactly one tier,
and the tier says who supplies it. The partition is the one
[model-and-device-bringup.md §7.3](model-and-device-bringup.md) fixes. The
reasons are the decision records in [../spec/decisions.md](../spec/decisions.md).

| Entry point | Tier | Gate | Serves |
|---|---|---|---|
| `Alloc`, `Free`, `Memcpy`, `Memset`, `Synchronize`, `GetCaps` | primitive | required | The six device utilities |
| `Gemm` | primitive | required | Every projection and the classifier; cuBLAS on CUDA, a reference loop on the CPU |
| `GemmWgrad` | primitive | required | The weight gradient, accumulated into fp32 parameter storage |
| `RowReduce` | primitive | required | `RmsNorm`, `QkPrep`, and the `Classifier`'s per-row reduction. Additive at seam 1 → 2 |
| `SpanReduce` | primitive | required | `GradNorm`, `ScalarDot`, the loss reduction. Additive at seam 1 → 2 |
| `AttentionForward`, `AttentionBackward` | primitive | required | The attention graph; causal, group-query, full context and sliding window |
| `PointwiseForward`, `PointwiseBackward` | primitive | required | The op codes, elementwise. Three codes are additive at seam 2 → 3 |
| `EmbeddingForward` | primitive | required | The gather over `wte` and every `value_embeds[i]` |
| `EmbeddingBackward` | primitive | required | The scatter-add into the dense gradient buffer |
| `SmearForward`, `SmearBackward` | primitive | required | The bigram residual, one launch per direction on a device |
| `QuantizeFp8`, `GemmScaledFp8` | primitive | optional, bits `fp8_storage` and `fp8_gemm` | The fp8 tier, [precision.md](precision.md) |
| `AdamWUpdate`, `MuonUpdate` | stateful | required for training, bits `stateful_adamw` and `stateful_muon` | The optimizer, [model.md](model.md). One `MuonUpdate` call covers one shape group: 39 launches on the shipped multi-launch path, or one cooperative launch of 84 blocks with 27 grid barriers on the fused path. Part 1 has the measurement |
| `RmsNormForward`, `RmsNormBackward` | composition, plus fused override | — | `src/ops.cc`; a backend's single-kernel version is gated by `override_rms_norm` |
| `QkPrepForward`, `QkPrepBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_qk_prep`. The fused family also writes the prepared query head-major when the caller sets `QkPrepParams::q_head_major` |
| `ClassifierForward`, `ClassifierBackward` | composition, plus fused override | — | `src/ops.cc`; gated by `override_classifier` |
| `ValueGateForward`, `ValueGateBackward` | composition, plus fused override | — | `src/ops.cc`; no override bit exists for it yet |
| `SumFp32` | composition | — | The fp32 form of `ScalarDot` with no second operand; the training loss |
| `GradNorm` | composition | — | `SpanReduce` across several spans |
| `ScalarDot` | composition | — | `SpanReduce` over two buffers, or the backend's dot |

A composition supplies its own backward as one unit, because
[model.md](model.md) § No autograd engine requires a fused backward to be
derived and tested as one unit.

**Required bring-up cost: the six utilities and every arithmetic primitive.** A
device at that tier runs every graph in [model.md](model.md) on the composition
path without writing one fused kernel. Training adds the two stateful entries.
The two fp8 entries are declared and never assumed: a `caps.json` carries
`fp8_storage` and `fp8_gemm` only where the silicon has the formats.

**The required `Attention` variant is causal, group-query, full context *and*
sliding window, at one tile size.** Sliding window is required and not optional:
`models/nanochat-default/spec.json` names it on nine of its twelve blocks, and
`Config::window_pattern` derives `"SSSL"`, so `window_left()` is non-empty on
every `S` layer. A device that implements full-context attention only cannot run
the one model in the tree. The mask is a field of the block, not a second entry
point, so one kernel covers both. *One tile size* bounds the obligation: a
second tile size is device tuning rather than a contract
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
| the Polar Express chain: square GEMM -> `b = cb·a + cc·acc` -> apply GEMM -> `x = ca·x + prod` | **no, measured** | all five conditions hold and the kernel exists (Part 1), but the hand GEMM falls short of the cuBLAS rate, so the fused step is slower and the shipped multi-launch path stays |

That last row is the one row of this table a campaign measured rather than
argued. It is also the one where the structural verdict and the performance
verdict agree: condition 3 (no library displacement) fails, because the
arithmetic the fusion would own is arithmetic cuBLAS already wins.

Performance **validates** a boundary. Performance does not choose a boundary.
Structure decides which boundaries are legal: the decomposition, the residency,
and the library ownership.

### Candidate v2 fusion

`QkPrep` can fold into the `Attention` kernel, because the RMSNorm reduction
runs over `head_dim`, which one tile row holds. The reduction does not span
CTAs, so the rule permits the fold. The fold waits for v2, because it couples
the attention backward to the norm and rope backward.
`backends/cuda/fused_overrides.cc` declares `kFeatureFusedAttention` as the v2
candidate, and `backends/cuda/fused_override_trace_test.cc` holds the reference
case.

## Compute strategy

### GEMM

Keep cuBLAS as the GEMM engine. It is near peak on GP102 (11.34 TFLOPS fp32), it
is tuned per shape, and it is portable across generations. Do not hand-write a
GEMM for the dense path. Part 1 measured what a hand-written batched GEMM costs
when a fusion needs one: 0.68 to 0.90 of the cuBLAS rate on the groups the
fusion targets. That is the number this rule stands on.

Part 3 holds the design of a hand-written tensor-core GEMM for bf16 and fp8.
The tree does not implement it.

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
`cuBLAS + CUDA Graphs` baseline before promotion.

### Direct-MMA and persistent kernels

A direct `mma` kernel and a persistent kernel need a tensor core and the `mma`
or `wmma` instruction, so they belong to `sm_75` and newer only. The gain in a
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

struct PointwiseParams {
  PointwiseOp op = PointwiseOp::kScale;
  int count = 0;
  int row_width = 0;  // width of the row a broadcast reads; 0 => count
  ScalarArg alpha{1.0f};
  ScalarArg beta{0.0f};
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

The fused path is in `backends/cuda/kernels/muon_fused.cu` and both drivers are
tested. The shape rule selects the multi-launch cuBLAS path for every group. The
rows were measured on a card the tree no longer supports, so they are removed;
measure again on a supported card before you cite a number.

### 1. The question

One Muon step is a chain of dependent kernels. At the shipped `ns_steps` of 5
the Polar Express iteration alone is five launches -- `A = XᵀX` (or `XXᵀ`),
`A·A`, the elementwise `b = cb·a + cc·acc`, `prod = X·b` (or `b·X`), and the
elementwise `x = ca·x + prod` -- repeated five times, and momentum, row
equilibration, both renormalisation statistics, and the NorMuon stages surround
it. The multi-launch path issues 39 launches per shape group.

Can the whole stage be one device-side launch: the three products in one batched
GEMM of our own, both elementwise passes folded into the epilogues of their
neighbours, and a grid barrier between stages? And is that faster on the card in
the tree?

The measured answer is two-part. The folds are legal, the arithmetic is right,
and the kernel works. It is also slower for every group at both profiles, so the
multi-launch path stays the default.

### 2. The tree it starts from

The multi-launch path runs the stage as one launch per operation: Nesterov
momentum, MuonEq row equilibration, Polar Express through strided-batched
`kernels::Gemm` (cuBLAS, one call per GEMM per group), Muon+ renormalisation,
NorMuon variance reduction with a factored second moment, and the cautious
decoupled weight decay. The math mirrors `backends/cpu/kernels.cc`. Three
properties of that path are the family's contract:

- **Every whole-matrix statistic is reduced by many blocks per matrix.** The
  Frobenius norm, the equilibration divisor, and the renormalisation scale use
  `MuonPartialBlocks(mat)` blocks -- one per eight thousand elements, at least
  one and at most 32 -- and a second pass finishes the reduction.
- **Every per-row or per-column statistic uses one block or one thread per row
  or column.**
- **The two operand layouts are a shape property, not a flag fight.** Tall
  (`rows > cols`) is `A = XᵀX` of `cols x cols` and `prod = X B`; wide
  (`rows <= cols`) is `A = XXᵀ` of `rows x rows` and `prod = B X`. `A` and `b`
  therefore occupy `num_params` slices of `min(rows, cols)²`.

The GEMM is cuBLAS on that path because the compute strategy above says not to
hand-write one.

### 3. The arithmetic of the bet

The fusion trades two costs. It removes the launch gaps between dependent
kernels and the elementwise HBM passes it folds away, and it pays for a
hand-written GEMM where cuBLAS was. A group is worth fusing when the first is
larger than the second. The design put a floor under the first: a share below
**2 percent of the step** stops the campaign with a negative result.

**The share, measured.** Three `muon_benchmark` rows give it without a new
kernel. `polar:<shape>:step` is the loop's fifteen batched GEMMs alone,
`muon:<shape>:step` at `ns=0` is the rest of the optimizer phase, and
`muon:<shape>:step` at `ns=5` is the whole step, so

```text
loop                 = muon:step(ns=5) - muon:step(ns=0)
launch + elementwise = loop - polar:step
share of the step    = (launch + elementwise) / muon:step(ns=5)
```

The rows were measured on a card the tree no longer supports, so the numbers
are removed. The conclusion they carried is that the share is above the 2
percent floor at both profiles, so the bet was worth writing.

**The sync term, measured.** A dependent launch of a full batched grid costs
about one order of magnitude more than a cooperative grid barrier at the grid a
fused kernel could hold resident (`muon_launch_sync_bench`). The launch gaps one
fused iteration removes are therefore a rounding error against the elementwise
passes it also removes. The share above is elementwise HBM traffic and not
dispatch.

**Both terms, measured.** The device-side batched GEMM fell short of the cuBLAS
rate on the group the fusion targets, so the 85 percent gate failed. The whole
fused step is slower at `ns=0` and slower again at both profiles at `ns=5`. The
loss is an order of magnitude larger than the win, so the campaign records a
negative result.

[performance.md](performance.md) section 2.1 holds the protocol that produced
the removed rows.

### 4. The design

**One device function, one block size, sixteen phases.** The fused path is one
`__device__` function parameterised by a phase index and driven from a flat
work-item index. A block strides over the items of the phase it is in, because
one cooperative grid serves every phase and fixes the block size, the grid size,
and the dynamic shared size for the whole kernel. 256 threads under
`__launch_bounds__(256, 3)`; the resident grid is the active blocks per SM times
the SM count of the card, so the driver queries it and holds no constant.

**The batched GEMM** is block tile 128 by 64, micro-tile 8 by 4 per thread (32
fp32 accumulators, a 16 by 16 thread grid), two double-buffered 8-deep k slices
in 12288 bytes of shared memory, and one flat `(matrix, tile_i, tile_j)` index
per block. `ptxas` reports 80 registers per thread and no local-memory traffic
in the inner loop (`#pragma unroll 2`).

**All five operand forms work.** `transpose_a` and `transpose_b` select four of
them -- `XᵀX`, `XXᵀ`, `A·A`, and `X·B`. The wide branch `b·X` needs a separate
selector, `MuonSquareSide`, because `b·X` and `X·B` are both plain by plain.

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
launch the device refuses with `cudaErrorCooperativeLaunchTooLarge` falls back to
the phase driver and logs one line. The **phase** driver launches one grid per
execution, each sized for its own phase: it is the simulator's path, the fallback
above, and the phase-bisection tool. The workspace, the arithmetic, and the
device body are identical in both drivers, so the two must agree bit for bit.

**The shape rule.** `MuonFusedSelected` keeps the multi-launch path for a group
where the bet is poor and selects the fused path where it is not. Its one number
is `kMuonFusedMaxElements`, the largest stacked group the fused path is selected
for, in `ComputeType` elements. That number is **0**, because the measurement
says no group wins; promoting a group is one measurement and this
constant.

### 5. The workspace

`EnsureMuonWorkspace` is `2·total + 2·min_total` `ComputeType` elements, and both
paths use that one layout (`MuonFusedWorkspaceElements`):

- two full-size slices are the iterate and its ping-pong partner -- which is why
  the multi-launch path's `prod` buffer doubles as that partner, and why
  `MuonUpdate` writes `A·A` into `b` in place;
- two `min_dim x min_dim` slices are `A` and the folded `b`.

The `float` statistics scratch is `MuonFusedScratchFloats`: the per-matrix
statistic (`num_params`), then the NorMuon mean squares
(`num_params * max(rows, cols)`), then the NorMuon global scales
(`num_params`). The larger extent settles the size, so one allocation serves
both `red_dim` values of a group. `MuonScratchOffset` rounds every region up to a
16-byte group, so the elementwise phases can move a whole vector of the
statistics.

The partial buffer is `num_params * MuonPartialBlocks(rows * cols)` doubles. Both
partial passes accumulate in double and both paths use the same block count and
chunk order, so every statistic stays within one unit in the last place of the
other path.

### 6. The gates

1. **The CPU loop.** `tools/nanochat test` stays green. The CPU backend is the
   definition, and no `caps.json` changes.
2. **The multi-launch path is unchanged.**
   `//backends/cuda/kernels:muon_gpu_test` compares it with
   `testing/optim_ref.h` as before, over the reduced workspace.
3. **Three implementations, one input, one seed.**
   `//backends/cuda/kernels:muon_fused_gpu_test` compares `optim_ref.h`, the
   shipped multi-launch `MuonUpdate`, and the fused path through each driver,
   over `4x3`, `3x6`, and `5x5` groups, both `red_dim` values, Nesterov on and
   off, `ns_steps` of 0, 1, and 5, and `num_params` of 1 and 3, at the family's
   `1e-5` tolerance.
4. **Both drivers, and the refusal.** The two drivers agree bit for bit; every
   output is finite, so a divergent stage is caught by name; the workspace is
   exactly `2·total + 2·min_total` elements with a 64-element sentinel tail; and
   a cooperative launch of eight times the resident grid is refused with
   `cudaErrorCooperativeLaunchTooLarge` and produces the phase driver's answer
   with `fell_back` reported.
5. **The simulator.** The mock answers 0 for `cudaDevAttrCooperativeLaunch` and
   1 for the occupancy query, so a simulated run takes the phase loop.
   [simulator.md](simulator.md) section 5.2 lists the runtime names.
6. **The benchmark.** `muon_benchmark` reports a `muon:` and a `muon_fused:` row
   for every group and for the `ns=0` and `ns=5` steps at both profiles, and the
   `driver=` field names the driver the card answered with.

### 7. What landed

`backends/cuda/kernels/muon_fused.cu` (the device function, the folds, and the
two drivers), `muon_fused.h` (the workspace helpers, the driver selection, and
`MuonFusedUpdate`), and `muon_fused_test.cc`. `muon.cu` keeps the one
`MuonUpdate` seam entry, owns the workspace and the scratch, and selects a path
by shape. The frozen seam gains no entry point and no capability bit.

### 8. The device questions

A fused build needs three questions answered before it can pick a driver, and
all three are put to the device rather than to a capability bit. That surface is
`backends/cuda/device.h`:

| Helper | Answer |
|---|---|
| `LaunchCooperative` | The launch itself, on `Stream()`, beside `Launch` |
| `CooperativeLaunch` outcome (`kLaunched`, `kTooLarge`, `kFailed`) with `ClassifyCooperativeLaunch` | A typed failure, so a caller recognises `cudaErrorCooperativeLaunchTooLarge` and falls back with one log line |
| `CooperativeLaunchSupported`, `SupportsCooperativeLaunch` | Whether the device accepts a cooperative launch at all |
| `MultiprocessorCount`, `ActiveBlocksPerMultiprocessor`, `ResidentBlocks` | The persistent grid: active blocks per SM for that kernel, block size, and dynamic shared size, times the SM count. 84 blocks on a 28-SM card at 3 blocks per SM |

No capability bit, no `caps.json` field, and no change to `GetCaps`: the driver
asks the device, and a card that answers with no takes the phase loop. The
simulator answers 0 for `cudaDevAttrCooperativeLaunch`, so it always takes it.

A fused build asks the runtime for **`cudaLaunchCooperativeKernel`,
`cudaDeviceGetAttribute`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** even when it takes the phase
driver, because the choice is made at run time and all three are reached before
the choice is final. The mock must export all three or
`tools/cuda_sim/check_symbols.py` rejects it.

### 9. Limits

- **One block size fits no stage.** 256 threads with `__launch_bounds__(256, 3)`
  and 15 KiB of static shared memory hold 6 blocks per SM where the elementwise
  families' own kernels hold 8, so the fused stage starts from a residence
  deficit that the folds have to pay back. They do not: the `ns=0` step, which
  runs no GEMM and has nothing to fold, is slower.
- **The GEMM schedule is the bet, and it lost.** 80 registers per thread, 12 KB
  of shared memory, and 3 blocks per SM leave 768 of 2048 threads resident. An
  8 by 8 micro-tile, a 16- to 32-deep slice, or a 256 by 64 tile is a *different*
  prototype and a different measurement. Nothing here says the fusion is
  impossible, only that this schedule is well off cuBLAS.
- **cuBLAS weakness masks the hand GEMM on the largest group.** On the
  1280x5120x1280 group the `A` stage reached the cuBLAS rate, because that
  product is bandwidth bound. A rate gate against cuBLAS is therefore not a
  selector by itself; the end-to-end step row is.
- **The bf16 build stays on the multi-launch path.** The fused schedule is the
  fp32 one and loads `float4` quads, so `MuonFusedSelected` reports false there.
- **Non-goals.** No capability bit, no `caps.json` change, no change to the
  frozen seam, no whole-model megakernel, and no cooperative driver in the
  emulator, whose block-at-a-time engine would deadlock on a grid barrier.

## Part 2. Device-resident scalar arguments

A scalar that the optimizer owns must never reach the host on a training step.
`ScalarArg` is the seam form that keeps it on the device.

### 1. The rule

| Scalar kind | Examples | Form |
|---|---|---|
| A constant or a schedule value | `eps`, softcap, `kQkScale`, `lr`, `beta1`, warmup | host `float`, by value |
| **Model state** | `resid_lambdas`, `x0_lambdas`, `smear_lambda`, `backout_lambda`, the clip scale | `ScalarArg` with a device pointer |

A kernel reads a scalar argument **once, before the loop**, and keeps it in a
register:

```cpp
const float scale = grad_scale.dev != nullptr ? *grad_scale.dev
                                             : grad_scale.value;
for (long long i = ...; i < n; i += stride) { ... }
```

A read inside the element loop would turn one L2 hit into `n` loads.

**Why the host read is expensive.** `src/model.cc` read `resid_lambdas[i]` and
`x0_lambdas[i]` with `ReadHost`, which is a synchronous `cudaMemcpy`. A trace
put the `blocking_dtoh_scalars` signature at **0.3197** of a step, and the three
largest stalls followed a **0-byte** copy. The bytes are free; the queue drain
is not. The loop issues about 16100 driver calls per step at 2 to 3
microseconds each, so per-call operand validation is the wrong thing to add.

### 2. The type

```cpp
// One scalar argument to a kernel family.
//
// `value` is a host copy. When `dev` is not null the kernel loads the value
// there instead, once, before its loop, and ignores `value`. A null pointer,
// the default, keeps the host value, so every existing call site is unchanged.
//
// Use it for a scalar the optimizer owns. A constant or a schedule value stays
// a plain `float`: there is nothing to read.
struct ScalarArg {
  ScalarArg() = default;
  ScalarArg(float host) : value(host) {}  // NOLINT: implicit on purpose

  float value = 0.0f;
  const float* dev = nullptr;
};
```

`ScalarArg(float)` is implicit so that `params.alpha = 2.0f` and an aggregate
initialiser both keep compiling. The accessor `host()` returns `value`; only a
kernel wants the raw pointer.

The type is two fields and one branch. A copy of `c10::Scalar` cannot carry a
pointer to the device, because no tag selects one. An `at::Tensor` can, through
an implementation allocation, at the cost of refcounting, per-use validation,
and a dispatch layer that [../README.md](../README.md) § What this is not
forbids.

### 3. The seam

Three parameter structs carry it: `PointwiseParams::alpha`, `PointwiseParams::beta`,
`AdamWParams::grad_scale`, and `MuonParams::grad_scale`. The last two replace the
former `float` plus `const float*` pair with one field:

```cpp
struct AdamWParams {
  // The gradient clip scale, folded into the gradient read. A caller that
  // keeps the norm on the device sets `dev` and never reads the value back.
  ScalarArg grad_scale{1.0f};
};
```

The change is additive in behaviour and no enum grows a member, so the closed
`PointwiseOp` list stays closed and no backend gains a missing arm.

### 4. The backend arms

- `backends/cuda/kernels/adamw.cu` and `muon.cu` drop the `bool scale_on_device`
  argument, because the struct now says where the value lives.
- `backends/cpu/kernels.cc` keeps one `Coefficient` helper instead of two
  `GradScale` overloads.
- `backends/cuda/kernels/pointwise.cu` resolves `alpha` and `beta` in the
  prologue of each kernel that uses them.
- The simulator interposes CUDA calls, so it sees the launch and not the
  parameter struct, and it needs no arm.

### 5. The model and the ops layer

`src/ops.h` is not a frozen header, so the block entry points take device
pointers:

```cpp
void BlockForward(const BlockShape& shape, const BlockWeights& weights,
                  const BlockActivations& acts, const int* tokens,
                  const float* cos, const float* sin, const ComputeType* x,
                  const ComputeType* x0, const float* resid_dev,
                  const float* x0_lambda_dev);
```

`src/model.cc` passes the fp32 master copy, which is the authority the optimizer
updates. No new buffer and no conversion kernel is needed:

```cpp
// The fp32 view of one parameter, for a kernel that reads a device scalar. In
// a bf16 build this is the master copy, which the optimizer keeps current. In
// an fp32 build the compute buffer is already fp32.
const float* FloatView(const Param& p) {
  if (p.master != nullptr) return p.master;
  if constexpr (std::is_same_v<ComputeType, float>) return p.value;
  return nullptr;
}
```

`ops::DevScalar` is the one accessor for a device scalar in the ops layer, and
`ScalarSource` is the one place that chooses the fp32 view.

### 6. The remaining host reads

The four scalar reads are gone, and three further drains with them:

| Read | After the change |
|---|---|
| `resid_lambdas[i]`, `x0_lambdas[i]` | gone, 32 synchronous copies per step to zero |
| `smear_lambda` | gone; `SmearParams::lambda` is a `ScalarArg` |
| `backout_lambda` | gone; `TrainModel::neg_backout_dev_` holds the negated value, refreshed by `kernels::SumFp32` with a scale of minus one |
| the gradient clip scale | no traffic change; the form changed and two helpers left |
| the loss | `ForwardLossAccumulate` adds the micro-batch mean to a device total; `ConsumeLossSum` reads four bytes on a step that records the loss |
| the phase drains | `src/train.cc` synchronizes at a phase boundary only on a step that logs |

Two host round trips remain in a base step: the staged token and target
indexes, because the seam takes host pointers, and the checkpoint records, which
a save reaches on `save_every` only.

### 7. The precision consequence

The change is **not** bit identical, and the measurement found that.

The old path read `resid_` and `x0_lambda_`, which are `ComputeType`. A bf16
build therefore staged the **bf16-rounded** scalar and widened it. The new path
reads `p.master`, which is fp32. The two differ in the bits the bf16 copy drops,
so the residual blend consumes a more precise coefficient than before.

| Fact | Value |
|---|---|
| Worst relative loss difference | 4.85e-3 |
| Worst relative `grad_norm` difference | 2.5e-1 |
| First step that differs | step 1 |
| The unchanged tree against itself | 9.1e-4 loss, 3.8e-2 `grad_norm`, first difference at step 29 |

A difference at step 1 is not noise and it is not the plan cache: it is the
parameter value itself. The tree accepts it. The optimizer owns the fp32 master,
the model derives the bf16 copy from it, and a scalar that feeds float
arithmetic should carry its fp32 value. The cost is a baseline for every parity
fixture that pins the trajectory.

The CPU backend and every fp32 build take this behaviour for free, because their
`ComputeType` is `float` and `ScalarSource` returns the compute buffer itself.

### 8. Measurement and non-goals

Two untraced runs of 200 steps on `base` at the H100: 42.90 ms per step and
2.3512 percent Model FLOPs Utilization before, 41.98 ms and 2.3834 percent
after. The step fell 2.1 percent and the MFU rose 1.4 percent. The two points
overlap, so the 2.1 percent is inside the noise of that card, and the change was
kept for the seam and for the precision, not for the step time.

At the `big` profile the pair is 229.93 ms and 18.161 percent before, 229.86 ms
and 18.167 percent after. Its device work is 230 ms, and the host traffic this
change removes was under 1 ms.

The gates are `//backends/cpu:kernels_test` and the CUDA family tests for the
prologue read, `//src:model_gradient_test` for the residual path,
`//tests:oracle_cuda_test` for a whole step, and `tools/nanochat lint` for the
frozen header.

Non-goals:

- A `Tensor` or `TensorRef` type in the seam. A 24-byte borrow belongs in
  `src/ops.*` or the C ABI.
- A dtype tag on `ScalarArg`. A build has one compute type, and the master copy
  is fp32 by construction.
- Migrating constants. `eps`, the softcap, `kQkScale`, and the schedule values
  stay host floats.

## Part 3. The hand-written tensor-core GEMM (not implemented)

cuBLAS reaches 989 TFLOPS in bf16 and 1979 TFLOPS in fp8 on the H100, because it
uses WGMMA, TMA, split-K, and swizzled tiles. A hand kernel on the warp-level
matrix multiply-accumulate interface lands near Ampere-class rates. The design
therefore sets a fraction gate and not a literal match: **0.70 of the cuBLAS
rate** for a first promotion and **0.85** for a full shape replacement. The fp32
hand GEMM stayed below the first promotion on the group the Muon fusion targets
(Part 1).

**The shape set is a closed list, not a general GEMM:** the Muon Polar Express
groups of [model.md](model.md) § Optimizer, and the Linear forward, dgrad, and
wgrad shapes in `python/nanochat_cpp/pipeline.py` `PROFILES`.

**The kernel.** Generalise the `attention_bf16.cuh` microkernel into a tiled
GEMM under `backends/cuda/gemm/`. Use `cp.async` double-buffered
shared-memory stages and `wmma::mma_sync` with 16 by 16 by 16 bf16 fragments and
fp32 accumulators. Support the full `GemmParams` contract: transpose flags,
batch, leading dimensions, strides, an alpha/beta epilogue, and the fp32 output
of `GemmWgrad`. Mirror `muon_fused_gemm_test.cc` for the correctness gate.

**The fp8 path** adds the E4M3 forward and E5M2 backward fragments. The per-tensor
scale folds into alpha, because the interface takes no scale argument, and the
backward mix of both formats must work in one call. The gate is the fp8
tolerance in [precision.md](precision.md).

**Dispatch.** A predicate in `backends/cuda/gemm.cu` routes a shape to the hand
library only when it beats cuBLAS at the fraction above. cuBLAS stays the
fallback and serves every shape the library does not. The rule is: no group
slower. The frozen seam keeps its signatures and its version.

**Blocked direction.** A Hopper-native path needs WGMMA and TMA in the simulator
first, and [attention.md](attention.md) Part 5 records that as the blocker. It
is a separate effort with its own stop condition, and it does not start before
the warp-level path passes its gate.

## Part 4. Two kernel designs the tree does not implement

The arithmetic below is agreed and the tree does not carry it. Both changes keep
the frozen seam's meaning and neither adds a family.

### 1. Vectorized row kernels

Every row kernel in `backends/cuda` uses scalar strided access: the loops step
one element at a time. On every supported card the ridge point is tens of flop
for each byte, and the row kernels sit far below it. `row_benchmark.cc` reports milliseconds only, so an
achieved byte rate column comes first.

The helper is header-only in `backends/cuda/kernels/device_utils.cuh`, so it
inlines and never merges a kernel boundary. `Packed<T>` is `alignas(16)` and
holds `16 / sizeof(T)` elements, so the fp32 build packs four. `LoadPacked` and
`StorePacked` cast to `int4`, and the streaming forms use `__ldcs` and `__stcs`.

| Kernel | Operand | Change | Risk |
|---|---|---|---|
| `rms_norm.cu` | `x`, `out` | packed load and store | Low |
| `pointwise.cu` | `a`, `b`, `out` | packed load and store | Low |
| `embedding.cu` | `out`, `dtable` | packed store, scalar gather | Low |
| `grad_norm.cu` | `grads` | packed load | Low |
| `attention.cu` | softmax rows | packed load and store | Medium |
| `classifier.cu` | `logits` | packed load, scalar reduction | Medium |

A kernel takes the packed loop only when three conditions hold: the pointer is
16-byte aligned, the row width divides by the packed count, and the row count
divides by the block stride. Otherwise it takes the scalar loop, which produces
the same arithmetic. `Workspace::Reserve` gains 16-byte alignment for the slots
the packed kernels read.

A packed loop changes the order of the per-thread accumulation, so the last bits
of a sum can move. The fp32 oracle tolerance decides the result. The classifier
softmax keeps its scalar reduction, because a packed reduction can move
`row_max` by one unit in the last place.

### 2. The classifier statistics hand-off

The classifier holds `raw_logits_` and `dlogits_`, each 4.3 GB at `R = 32768`
and `padded_vocab = 32768` in fp32. The logits buffer moves six times per step,
and two of those passes rebuild statistics that the forward pass already had.

Two independent changes remove most of that traffic, and both are bit identical
to the current result.

1. **The statistics hand-off.** `ClassifierForward` writes `(row_max, sum_exp)`
   per row and `ClassifierBackward` reads the pair. The pair is `2 * rows`
   floats, about 256 KB. The value is the same bit for bit, because both calls
   use the same input, the same reduction, and the same order. Two logits reads
   and one `RowLogSumExp` call leave.
2. **The fused scale.** `ClassifierBackward` takes `dlogit_scale = scale / rows`
   and writes `dlogits[j] = (p - onehot) * sech2 * dlogit_scale`. The separate
   `PointwiseForward(kScale)` pass disappears and the arithmetic order does not
   change.

Backward logits reads fall from 3 to 1 and scale passes from 2 to 0, which is
about 17 GB less traffic per step.

The seam change is additive, and the architect owns the header:

```cpp
// Forward writes the per-row statistics when `stats` is non-null:
// stats[2*r] = row_max, stats[2*r+1] = sum_exp. Evaluation passes null.
void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses, float* stats = nullptr);

// Backward consumes the statistics when `stats` is non-null, and rebuilds them
// otherwise. `dlogit_scale` folds the loss normalization into the write.
void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        const float* stats, float dlogit_scale,
                        ComputeType* dlogits);
```

The default argument keeps every call site valid, and the null path keeps the
current behaviour, so the kernel test compares the two paths.

**The rejected option.** `llm.c` overwrites the logits buffer with the gradients,
which would remove `dlogits_` and 4.3 GB of arena. This tree cannot do it.
`BackwardAccumulate` may run more than once per forward pass, and
`src/harness_test.cc` calls it twice with a half scale. The second call would
read gradients where it expects logits.

## See also

- [model.md](model.md) — the graphs the seam serves, and the optimizer step.
- [build.md](build.md) — the layers and the dependency invariants.
- [performance.md](performance.md) — the measurement protocol and the loop.
- [precision.md](precision.md) — the fp8 tier.
- [simulator.md](simulator.md) — the interposer and the emulation engine.
