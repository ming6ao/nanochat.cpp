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

cuBLAS is the GEMM engine today. [Part 3](#part-3-the-hand-written-gemm-revision-cublas-as-the-reference-only)
sets the direction: the hand library serves every call, and cuBLAS stays only as
the measurement reference. Two measurements set the current rule, and both stay
true.

- Part 1 measured what a hand-written batched GEMM costs when a fusion needs
  one: 0.68 to 0.90 of the cuBLAS rate on the groups the fusion targets.
- Part 3 measured the bf16 tensor-core tile at 0.08 to 0.30 of the cuBLAS rate.

The first is why the design does not hand-write a GEMM to reach a fusion. The
second is why the campaign rejected the warp-level tile. Neither one is a
statement about a Hopper-native kernel, and Part 3 section 3 draws that
difference.

Do not hand-write a GEMM for the dense path until Part 3 phase 2 passes its gate.
The plan there has no code yet, and this line is the rule until it lands.

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

`backends/cuda/kernels/mma_ops.cuh` is the second such header: one micro-tile
instruction with a device arm and a scalar arm, so the emulator can run it. Its
fp8 arm arrives with that campaign, and Part 3's bf16 tile does not use it.

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

## Part 3. The hand-written GEMM (revision: cuBLAS as the reference only)

**Status.** This part replaces the promotion-gate design with a removal design.
The tile of `backends/cuda/gemm/` does not change here, and it is still slower
than cuBLAS. The end state changed instead: cuBLAS becomes a benchmark reference
and nothing else, and the hand library serves every call on every device the tree
lists. Section 1 keeps the measurement that this plan starts from, because it is
the number the work has to beat. Sections 2 to 5 size the replacement, and
sections 6 to 9 hold the plan, the gates, the files that follow, and the stop
conditions.

No frozen header changes, and the seam gains no entry point: `Gemm`,
`GemmWgrad`, and `GemmScaledFp8` already hold every signature the plan needs.

### 1. The measured starting point

The tile of `backends/cuda/gemm/` is correct against cuBLAS over the whole
`GemmParams` contract. It reaches between 0.08 and 0.30 of the cuBLAS rate over
the closed shape list. On the six shapes this part exists for, it holds 120 to
157 TFLOP/s against cuBLAS at 643 to 865. That is 14 to 16 percent of the card's
bf16 peak, against 65 to 87 percent for cuBLAS. The rows are in
[baselines.json](baselines.json) under `gemm_bf16`, the comparison is section 7
of `backends/cuda/gemm/README.md`, and the reasons are section 8 of it.

Section 8 of that README names four causes: the vendor `wmma` load path, the
48 KiB static shared window, the epilogue, and the occupancy. All four stay true.
They are also not the whole gap, because they are warp-level causes and the gap
is a warp-level ceiling. Section 3 makes that argument.

### 2. What the replacement must beat

The card declares 989 TFLOP/s of bf16 (`backends/cuda/caps.json`). All six rows
are compute bound, so the compute time is the floor for both paths. The last
column is how much of that floor cuBLAS holds today.

| Row | Work | Peak time | cuBLAS | Share of peak |
|---|---:|---:|---:|---:|
| Muon `A = XᵀX` | 168 GFLOP | 0.170 ms | 0.2259 ms | 75% |
| Muon `prod` | 336 GFLOP | 0.340 ms | 0.5219 ms | 65% |
| Linear forward `fc` | 215 GFLOP | 0.217 ms | 0.2608 ms | 83% |
| Linear dgrad `cproj` | 215 GFLOP | 0.217 ms | 0.2616 ms | 83% |
| Linear wgrad `fc` | 215 GFLOP | 0.217 ms | 0.3074 ms | 71% |
| Classifier forward | 1374 GFLOP | 1.389 ms | 1.5897 ms | 87% |

The old promotion gate asked for 0.70 of the cuBLAS rate. That is 45 to 61
percent of peak, and it is the one encouraging number in this part. The new gate
asks for parity, because a removal has no fallback to hide behind.

### 3. The instruction set decides this

Two ceilings exist, and only one of them clears the gate.

**The warp-level ceiling.** `mma.sync` and `ldmatrix` are the Ampere instruction
path. A fourth generation tensor core is fully reachable only through a
warpgroup-wide instruction, so a well written warp-level kernel lands near half
of the card's peak. On these six rows that is 0.55 to 0.72 of the cuBLAS rate. It
sits on the gate and fails it on the classifier row.

**The warpgroup ceiling.** The warpgroup matrix multiply-accumulate (WGMMA) is
the Hopper instruction. A standard Hopper kernel of this class holds 70 to 90
percent of the cuBLAS rate on shapes of this size. It clears the gate with room
left for the removal.

So tuning the tile is not the work, and a second sweep of schedules is not the
work. **The work is a new mainloop on the Hopper instruction.** The two fractions
above are estimates. Phase 0 measures them before anyone writes a mainloop.

### 4. What the Hopper mainloop must have

The reference implementation is NVIDIA CUTLASS at `b2f249a`. The seven
mechanisms below are what its Hopper path does, and what the tile of
`backends/cuda/gemm/` does instead today.

| # | Mechanism, and where CUTLASS does it | The tile today |
|---|---|---|
| 1 | WGMMA takes a 64-bit shared descriptor for each operand. No register copy, no `ldmatrix` (`cute/arch/mma_sm90_gmma.hpp`, `mma_async.m64n128k16`) | `wmma::load_matrix_sync` compiles to `LD.E` and a register permute on `sm_90`, so each operand byte moves through the register file |
| 2 | One thread issues a whole tile copy, through the Tensor Memory Accelerator (TMA) (`sm90_mma_tma_gmma_ss_warpspecialized.hpp:355`) | One thread for each 16-byte chunk, plus a scalar loop at a ragged edge |
| 3 | A 128-byte swizzled shared atom, so the descriptor encodes a conflict free read | An 8-element pad, which costs 25 percent more shared bytes |
| 4 | Four to six stages, counted from the 227 KiB device window | Two stages inside a 48 KiB static window |
| 5 | One producer warp, one consumer warpgroup, and named barriers | Every warp copies and computes, with two block barriers in each slice |
| 6 | The accumulator goes register to shared to global as one bulk store for each subtile | Eight serialized rounds of barrier, one warp store, barrier, and a strided write |
| 7 | A cluster with multicast, an L2 swizzle, and a stream-K scheduler | No cluster, no swizzle, and one fixed raster order |

Mechanism 1 is worth about half of the gap on its own. Mechanism 6 is why the
weight gradient rows lose a further fifth of their rate today.

### 5. Three mainloops, and no refusal

A removal has no library to catch the calls that a fast kernel cannot serve. The
package therefore holds three mainloops, and one resolver picks between them. The
resolver never refuses a call, and the tree never asks a library for help.

| Mainloop | Device | Precision | Role | Precondition |
|---|---|---|---|---|
| WGMMA | `sm_90a` | bf16 | The fast path on Hopper | An operand aligned to 16 bytes, and a stride in whole 16-byte units |
| `mma.sync` | `sm_80` and newer | bf16 | The general path, and the leg the emulator runs | None |
| Shared-memory tile | `sm_75`, and the fp32 build | fp32 | The chain leg on the T4, and every fp32 call | None |

Three consequences follow, and each one is a decision this plan takes.

1. **The general path is also the Turing path.** `sm_75` has no WGMMA and no TMA,
   and the T4 declares no bf16 at all (`caps.json`). The T4 runs the
   post-training chain in fp32, so the third mainloop is an fp32 shared-memory
   tile. It must fit the 64 KiB window of that card. That kernel is unusually
   well understood, and parity with cuBLAS is realistic there.
2. **A refusal becomes an error, so the resolver must be total.** Today
   `IsTileCallable` refuses only an empty extent, and the shape table sends the
   rest to cuBLAS. The resolver must serve every non-empty call on every listed
   device. An unaligned or ragged call takes the `mma.sync` mainloop.
3. **The closed shape list stops being a routing table.** It becomes the
   benchmark record: the schedule the sweep chose for each measured shape.
   `GemmTileMode` and `NANOCHAT_GEMM_LIB` go with the routing decision.
   `NANOCHAT_GEMM_SCHEDULE` stays as a benchmark-only override.

The fp8 tier is the one open decision. `GemmScaledFp8` rides `cublasLt` today,
and the T4 has no fp8 silicon. Either the WGMMA fp8 form carries the tier, or the
architect withdraws it through [../spec/versions.md](../spec/versions.md). This
part recommends the first, because the fp8 atom is the same mainloop with a
different accumulator and a scale.

### 6. The phases

**The plan of record is the CUTLASS route.** Section 6.1 states it, and
section 10 holds the resume state: which step landed, which step is next, and the
command that decides it. Phases 0 and 1 below are the record of how the tree got
there; they are not the work that remains.

| Phase | Work | Effort | Gate | State |
|---|---|---|---|---|
| 0 | A `dev/kernels/` prototype: one WGMMA atom, one hand-built descriptor, one resident block, timed over the 1280 by 1280 by 1280 shape | 1 day | 85 percent of the per-SM tensor-core rate, which is 6.4 TFLOP/s from one block on the H100 | **Done**, 96.4 percent of the card peak |
| 1 | The `mma.sync` mainloop: direct `mma.sync` with `ldmatrix`, four to six stages, dynamic shared memory, and the epilogue rewrite | 4 to 8 weeks | Every row correct, and 0.50 of the recorded cuBLAS rate | **Done and missed**, 0.27 to 0.36. Section 10 records it |
| 2 | The Hopper mainloop: tile 128 by 256 by 64, cluster 2 by 1, one producer warp, two consumer warpgroups, TMA in and out | Was 2 to 4 months; became 1 week by borrowing | Within 1.05 of the cuBLAS rate on all six rows of section 2 | **Adapter done, gate open.** It needs the card |
| 3 | The fp32 shared-memory tile for Turing and the fp32 build | 2 to 4 weeks | Within 1.05 of the recorded cuBLAS rate on the chain profiles | Not started |
| 4a | fp8 on the Hopper path | 1 to 2 weeks | The fp8 tolerance of [precision.md](precision.md), and no row above 1.05 | **Arm written and clean off a card.** The card run is the last gate. The tiny group needs no kernel of its own now: the hand tile serves it |
| 4b | Stream-K for the ragged rows | Was 1 to 2 weeks; is a different project | No row above 1.05 | **Blocked by the vendor adapter.** Section 10.3 states the evidence |
| 5 | Detach the library: cuBLAS moves out of `backends/cuda:gemm` and into the benchmark targets | 1 week | The dependency query of gate G4 is empty | Not started |

### 6.1 The CUTLASS route, and why it replaced a written mainloop

The plan sized phase 2 at 2 to 4 months of writing a mainloop. NVIDIA CUTLASS
already holds that mainloop, and [dev/cutlass/README.md](../dev/cutlass/README.md) proves
it builds here. So phase 2 became an adapter, and the estimate fell to about a
week. The seven mechanisms of section 4 are the seven things CUTLASS supplies,
one for one.

| Decision | The answer | The record |
|---|---|---|
| How the headers arrive | `third_party/cutlass.bzl` fetches commit `b2f249ac` as an external repository | [build.md](build.md) § Link dependencies |
| Why not tracked files | The lint pass checks every `git ls-files` source, and the headers break four of its five checks | `third_party/cutlass.bzl` |
| Which architecture | The adapter target states `archs = ["sm_90a"]` on itself, so `.bazelrc` gains nothing | `dev/cutlass/README.md` § 1 |
| Which tile | 128 by 256 by 64, cluster 2 by 1, three stages, 215,040 B | `dev/cutlass/README.md` § 3 |
| Which shapes | Four conditions in `gemm_shape.h`; a refused call takes the hand tile and then cuBLAS | `backends/cuda/gemm/README.md` § 16 |
| What the hand tile is for now | The emulated leg, the correctness reference, and the shapes smaller than one tile | `backends/cuda/gemm/README.md` |

Phase 1 keeps the emulated correctness leg, because the emulator maps
`ldmatrix` and `mma.sync`. Phase 2 has none, and the CUTLASS route makes that a
consequence rather than a choice. The engine maps no Tensor Memory Accelerator
and no warpgroup instruction, so the hand tile stays as the off-device leg.
Section 10.1 records it.

**Gate 0 is a rate for each SM and not for the card.** One warpgroup on one SM
cannot exceed that SM's share of the peak, which is 989 TFLOP/s over 132 SMs, or
7.49 TFLOP/s. The gate reads 85 percent of that share from the one-block row and
85 percent of the peak from the full-grid row. `dev/kernels/wgmma_probe.cu`
reports both rows, and `//dev/kernels:README.md` holds the run command. The probe
needs `sm_90a`, so the build names it by hand and the target carries `manual`.

**Phase 0's result.** `dev/kernels/wgmma_probe.cu` measures one warpgroup on a
resident 64 by 64 A tile and a 128 by 64 B tile. It reads that tile through a
hand-built descriptor, and the timed loop moves no global memory. The card job
`wgmma_probe` of
`tools/modal/app.py` builds it at `sm_90a` and runs it on the H100 leg. One
warpgroup held 7.384 TFLOP/s, which is 98.6 percent of the per-SM share of the
peak. Two warpgroups on every SM held 952.9 TFLOP/s, which is 96.4 percent of the
card peak. **The gate passed.**

The rows are under `wgmma_probe` in [baselines.json](baselines.json), and
`dev/kernels/wgmma_probe_report.md` is the record of the run.

**What the result does to section 3.** The warpgroup ceiling is higher than
the estimate there. With no data movement the instruction holds 96 percent of
the card peak, which is above every cuBLAS row this card measured. So the gap
between the shipped tile and cuBLAS is the data path alone: the copy, the shared
layout, the pipeline depth, the epilogue, and the occupancy. Mechanisms 2 to 7 of
section 4 are the whole of the work, and mechanism 1 needs no further study.

**Phase 0 of the reuse, and what it settled.** `//dev/cutlass` fetches NVIDIA
CUTLASS at `b2f249ac` through
[../third_party/cutlass.bzl](../third_party/cutlass.bzl), and its probe compiles
the SM90 collective builder and the eight seam cases. Three facts came out of it,
and each one removes a decision from the phases below.

| Fact | Consequence |
|---|---|
| The headers build here, at CUDA 12.0 and C++20 | The vendor path needs no toolkit change |
| A `cuda_library` that states `archs = ["sm_90a"]` builds beside the command-line `sm_75` | `.bazelrc` gains nothing, and the default build is untouched |
| The cooperative schedule refuses a tile narrower than 128 in M | The 64 by 256 candidate of section 4 is not a candidate |

The object holds `UTMALDG.3D`, `UTMALDG.3D.MULTICAST`, `UTMASTG.3D`, `HGMMA.64`,
and `USETMAXREG`, which are mechanisms 2 to 7 of section 4 in one build.
`dev/cutlass/README.md` is the record, and the adapter that puts them behind
`Gemm` and `GemmWgrad` is section 6.1.

**Phase 1 has started, and its first step is done.** The micro-tile of
`backends/cuda/gemm/mma_bf16.cuh` is `ldmatrix` and
`mma.sync.aligned.m16n8k16`. It holds two arms: the instruction pair in inline
assembly, and the same lane map in plain C++, which the emulation engine runs. The gate is `//backends/cuda/gemm:mma_bf16_test`, and the card
job `mma_check` runs it on the H100 leg. All four operand orientations pass on
both arms, with the same maximum error of `8.064e-05` against one scalar
reference, so the lane map is checked and not assumed.

The rest of phase 1 is the mainloop: the tile still stages and stores through
`wmma`, so the header is a verified foundation and not yet a rate. The next step
is to drive it from `gemm_bf16.cu`, then take the accumulator epilogue into
registers, which removes the shared scratch and its eight serialized rounds.

Total: six to ten months for one GPU engineer with a card in the loop. That was
the cost of the removal as sections 5 and 6 first sized it. The CUTLASS route of
section 6.1 removes phase 2 from that sum, and the arithmetic of what is left is
in section 10.1.

### 7. The gates

1. **Correctness on the emulator.** `//tests:kernel_emu_test` and the tile's own
   gates compare each mainloop with its reference over the whole `GemmParams`
   contract, at the family tolerance.
2. **Correctness on the card.** `gemm_gpu_test` compares every row against the
   recorded cuBLAS result, over the four operand forms and both stores.
3. **Rate parity.** Each row of section 2 reads at most 1.05 of its `cublas_ms`
   entry in `gemm_bf16`, and each group total reads at most 1.02.
4. **The library is gone.** `bazel query 'somepath(//src:model,
   @cuda//:cublasLt)'` returns nothing, and no shipped binary holds a `cublas`
   symbol. `tools/cuda_sim/check_symbols.py` holds the same rule for the
   interposer.
5. **The reference still measures.** `gemm_bench` reports both paths from one
   binary, and the cuBLAS rows match `gemm_bf16` within three percent. A removal
   loses the rate table when it loses the reference by accident.
6. **The step does not regress.** `tools/nanochat` reports no worse step time on
   the throughput profile than the recorded baseline.

The tiny group gets one stated exception. Its 0.005 to 0.008 ms on either path is
launch cost, so no mainloop change wins it. It passes when the difference is
0.002 ms or less, and the plan keeps one tiny kernel for it in phase 4.

### 8. What this plan changes elsewhere

| File | Change | Owner |
|---|---|---|
| `docs/kernels.md` § Kernel inventory and § Kernel tiers | The `Gemm` row names the hand library, and cuBLAS becomes the reference | Architect |
| `README.md` § What this is not | cuBLAS leaves the list of allowed third-party runtimes | Architect |
| `docs/build.md` | The `@cuda//:cublasLt` link edge moves out of the library and into the benchmark | Architect |
| `docs/precision.md` | The fp8 tier names its kernel, or the tier is withdrawn | Architect |
| `docs/model.md` § The GEMM rows | The four cuBLAS selection rows and the shape-table fall-through go | Architect |
| `backends/cuda/gemm.cu` | The library call, the plan cache, and the heuristic query go | Integrator |
| `backends/cuda/gemm/dispatch.{h,cc}` | The verdict and `GemmTileMode` go, and the table becomes a benchmark record | Kernel agent |
| `backends/cuda/gemm/**` | The two new mainloops, and the resolver that replaces the verdict | Kernel agent |
| `third_party/cutlass.bzl`, `MODULE.bazel` | The pinned CUTLASS headers as `@cutlass`; done in phase 0 | Build |
| `.bazelrc` | Unchanged: the Hopper target names `archs = ["sm_90a"]` on itself | Build |
| `tools/cuda_sim/**` | The WGMMA and TMA mapping, if phase 2 is to keep an emulated leg | Simulator |
| `backends/cuda/gemm/README.md` | The record of the removal replaces the record of the rejection | Kernel agent |
| `docs/tensor-core-gemm-plan.md` | Retired into this part, per [README.md](README.md) § How to add a document, and to retire one | Architect |

### 9. Risks and stop conditions

1. **The emulator is the schedule risk.** WGMMA, TMA, `mbarrier`, and dynamic
   shared memory are all unmapped (section 9.2 of [simulator.md](simulator.md)).
   Either the Simulator workstream lands them before phase 2, or the Hopper
   mainloop keeps a device-only gate. Decide before phase 2 starts, because the
   second choice removes the tree's cheap correctness leg.
2. **`sm_90a` is a new architecture string, and it is settled.** WGMMA is an
   architecture conditional instruction, and `.bazelrc` emits `sm_90` today.
   `//dev/cutlass:cutlass_probe_lib` closes this by naming the architecture on
   itself: `rules_cuda`'s `cuda_library` takes an `archs` attribute whose value
   replaces the command-line one for that target alone. The library builds at
   `sm_90a` and a binary that links it builds at the command line's `sm_75`, from
   one build, with no `.bazelrc` line and no extra flag. The default build of
   `//...` therefore stays untouched. `dev/cutlass/README.md` is the record.
3. **The fp8 tier may not survive.** It is an optional bit, so withdrawal is
   legal. This part recommends keeping it.
4. **The tiny group is launch bound.** Section 7 states the exception, and phase
   4 carries the one tiny kernel. Without it the group keeps a 0.003 ms loss in
   each step.
5. **The maintenance does not end.** cuBLAS gains a tuned kernel for each new
   card and generation. The hand library freezes at the generation it was
   written for, and the shape record becomes the tree's problem at each model
   change. This is the cost that the removal buys out.
6. **The estimates are estimates.** Section 3 and section 6 read from published
   results and from the instruction-set argument, not from this tree's card. A
   performance claim in this part needs a `gemm_bf16` row, exactly as the rest of
   this document requires.

**Stop condition.** A phase that misses its gate after two iterations closes the
part with a negative result a second time. Phase 0 reads below 600 TFLOP/s, or
phase 2 reads above 1.15 on any row of section 2, and cuBLAS stays the GEMM
engine.

### 10. The remaining steps

Phase 1 is a gate and not a single change, so this section is the ordered list of
what is left of it and of the phases after it. Each step names the file, the
estimated effect, and the gate that decides it. An estimate here is an estimate
and not a measurement: only a `gemm_bf16_phase1` row decides.

Step 1 of the list is done, and section 6 records it. The steps below are the
rest.

| Step | The change | File | Estimate | The gate |
|---|---|---|---|---|
| 2 | The epilogue in registers: apply alpha, beta, and the bounds through the fragment lane map and store straight to global, which removes the shared scratch and its eight serialized rounds | `gemm_bf16.cu` | The weight-gradient rows gain most, because section 9 measured the scratch at about 30 percent there | Every row correct, and `fraction_of_cublas` above 0.28 on all six |
| 3 | Four to six stages in place of two, which needs dynamic shared memory and `cudaFuncSetAttribute` | `gemm_plan.h`, `gemm_tile.cuh`, `gemm_bf16.cu` | The largest of the four causes left, because two stages expose one copy latency in each slice | Every row correct, and `fraction_of_cublas` above 0.40 on all six |
| 4 | A wider tile and fewer registers per thread, so a block holds more arithmetic for each staged byte | `gemm_plan.h` | Occupancy is what section 9 names fourth; the sweep that chose `128x128x32` predates the new operand path, so it is a second sweep and not a guess | The best schedule per row changes, and the sweep says which |
| 5 | An L2 swizzle of the block order, and stream-K for the ragged rows | `gemm_bf16.cu` | Unmeasured. The classifier is the largest shape and it is the row where the tail matters least | No row slower, and the classifier row above 0.45 |

**Step 2 is done, and its gate is not met.** The straight-to-global store that
the row above proposes lost 1.6 to 11.5 percent, because a lane holds eight
columns of two rows and one warp store therefore lands in eight rows of sixteen
bytes. The shipped form keeps the lane map for the arithmetic and passes the
accumulator through a per-warp exchange tile, so C is written in sixteen-byte
chunks. That form gains 4.6 to 6.9 percent on the five rows that store the
compute type, and it is within the run-to-run spread on the other two. The
fractions read 0.2236 to 0.3071, so one row of six reaches 0.28.
`backends/cuda/gemm/README.md` section 12 and the `gemm_bf16_phase1_step2` rows
of [baselines.json](baselines.json) are the record.

**Step 4 and step 5 are done.** Step 4 widened the tile and swept it: the wider
128 by 256 tile lost on every row, because a block of 512 threads holds one block
per multiprocessor, and the win came from a 64-deep k slice on the 128 by 128
tile instead. Step 5 gave one block a run of tiles from a grid sized to the
resident blocks, and gave the plan an order to walk them in. The persistent grid
gained 1.039 to 1.117 and the swizzle gained 1.002 to 1.049, the largest on the
classifier row. The k-split half of stream-K is recorded as not done, with the
reason: it buys the tail of a wave and the classifier row has 124 of them.
Step 5's first half holds and its second does not, because the classifier reads
0.2753 against the 0.45 it asks for. `backends/cuda/gemm/README.md` sections 14
and 15 and the `gemm_bf16_phase1_step4` and `gemm_bf16_phase1_step5` rows of
[baselines.json](baselines.json) are the record.

**The four steps together, and what they say about phase 1.** One session
measured the tree before step 2 and the tree after step 5, each with
`--schedule all` and each scored on the best schedule of its own sweep:

| Row | Before | After | Gain | Fraction before | Fraction after |
|---|---:|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.8969 ms | 0.7389 ms | 1.214 | 0.2486 | 0.3042 |
| Muon `prod` | 1.8114 ms | 1.4748 ms | 1.228 | 0.2882 | 0.3530 |
| Linear forward `fc` | 1.1211 ms | 0.9481 ms | 1.182 | 0.2318 | 0.2707 |
| Linear dgrad `cproj` | 1.1582 ms | 0.9730 ms | 1.190 | 0.2256 | 0.2671 |
| Linear wgrad `fc` | 1.3072 ms | 0.9869 ms | 1.325 | 0.2331 | 0.3079 |
| Classifier forward | 7.1397 ms | 5.7740 ms | 1.237 | 0.2229 | 0.2714 |
| Total | 13.4345 ms | 10.8957 ms | 1.233 | | |

The tile's rate rose from 164 to 192 TFLOP/s to 218 to 238, which is 22 to 24
percent of the card's bf16 peak. The phase 1 gate asks for 0.50 of the cuBLAS
rate and this works reads 0.27 to 0.35, so the gate is not met and the part takes
its own stop condition: what closed is the data path, and what remains is the
instruction. Section 3 sized the warp-level ceiling at half the card, and 0.24 of
the peak against a ceiling of about 0.5 says the same thing from the other side.
Phase 2's warpgroup mainloop is the answer, and its gate asks for 1.05 of the
cuBLAS rate.

**Step 3 is done, and its gate is not met either.** The window is now the
dynamic one, a schedule states its stage count, and every plan serves the same
source on the card and on the emulator. Four stages of the 128 by 128 tile gain
3 to 7 percent over two on five rows and nothing on the sixth. Six stages lose
1.5 times, because a 120 KiB window holds one block per multiprocessor. The
fractions read 0.2303 to 0.2968, so the step misses the 0.40 it asks for, and
the reason is arithmetic intensity and not the pipeline: 64 flops for every
staged byte is 15.5 TB/s of operand traffic at the card's peak.
`backends/cuda/gemm/README.md` section 13 and the `gemm_bf16_phase1_step3` rows
of [baselines.json](baselines.json) are the record.

**The two constraints of steps 3 to 5.**

1. **Step 3 needs the emulator.** `docs/simulator.md` section 9.2 maps no
   `extern __shared__` and no dynamic shared window, so a four-stage tile has no
   off-device leg until the Simulator workstream lands one. A static window is
   the fallback: the H100 declares 227 KiB, and the trade is one target-specific
   array size against a per-architecture constant.
2. **Steps 4 and 5 need the card.** Each one changes a number in
   `gemm_plan.h` or the grid mapping, and neither has a host reference. The card
   job `gemm_bench` with `--schedule all` is the only measurement.

**The phase 1 gate stays what section 6 says:** every row correct against the
recorded cuBLAS result, and 0.50 of the recorded cuBLAS rate. Step 3 should reach
that fraction, and step 4 decides whether it is enough for the removal or only
for the next phase.

**Phase 1 is closed, and it closes as a recorded negative.** The five steps ran
and each one states its own gate and its own miss, so the part is not abandoned
in the middle: it is finished with the answer "this instruction mix cannot reach
half of cuBLAS". The reason is arithmetic intensity and not the schedule, which
step 3's 64 flops for each staged byte states, and no further sweep of a tile,
a stage count, or a block order changes a ratio. Three facts make the close
decisive rather than a pause.

1. The seam does not need the hand tile any more. Section 11 put a warpgroup
   candidate behind the same seam, and that candidate reads 0.576 to 1.176 where
   the hand tile reads 0.147 to 0.740.
2. The hand tile's own promotion bit is off on every row of `dispatch.cc`, so no
   shipped path measures it. Tuning an arm that nothing selects is card time
   spent on a record.
3. What is left of the tile has a job the candidate cannot take: a shape below
   `kCutlassMinOutputElements` and a call whose pointers are one element off the
   sixteen-byte boundary keep the hand tile, because a tensor memory descriptor
   per call costs more than the product on a small shape. That fallback role is
   the tile's remaining purpose, and it is served at the rate it has.

A future session that wants the hand tile faster therefore has to state a
reason the candidate cannot serve, and not a schedule it has not tried.

**Then phases 2 to 5.** Section 6 holds their scope and their gates: the Hopper
mainloop with the Tensor Memory Accelerator, the fp32 tile for Turing, fp8 with
stream-K, and the detachment of the library. Section 6.1 records the decision to
borrow the mainloop instead of writing it.

### 10.1 The resume state

This table is the state of the campaign as of the CUTLASS adapter. A session
resumes here. Each row names the step, the command that decides it, and the
record to write when it lands.

| Step | State | The command that decides it | What to write |
|---|---|---|---|
| 2a. Vendor the headers | **Done** | `tools/nanochat build //dev/cutlass:cutlass_probe` | `dev/cutlass/README.md` |
| 2b. Prove the architecture and the tile | **Done** | `tools/nanochat run t0-cpu -- ./bazel-bin/dev/cutlass/cutlass_probe` | `dev/cutlass/README.md` § 3 |
| 2c. The adapter behind the seam | **Done** | `tools/nanochat build --config=hopper --config=bf16 //backends/cuda/gemm:gemm_gpu_test` | `backends/cuda/gemm/README.md` § 16 |
| 2d. Correctness on the card | **Done, passed** | `modal run tools/modal/app.py::card_job --name gemm_check` | This section |
| 2e. The rate gate, 1.05 of cuBLAS | **Done, met on 19 of the 25 callable rows** after section 11.9 | `modal run tools/modal/app.py::card_job --name gemm_bench --argv '["--rows","production","--json"]'` | The `gemm_cutlass` entry of [baselines.json](baselines.json) |
| 3. The fp32 tile for Turing | Not started | The Turing leg of `card_job` | This section |
| 4a. fp8 on the Hopper path | **Arm written, and the spill cleared off a card.** The card run is the last gate | `gemm_bench` under `--config=fp8` | The `gemm_fp8` entry of [baselines.json](baselines.json) |
| 4b. Stream-K for the ragged rows | **Blocked by the adapter**, on the evidence of section 10.3 | `gemm_bench --row linear:wgrad:cls` | This section |
| 5. Detach cuBLAS from the library | **Closed, and it does not follow yet.** The reason is in 10.2 | `bazel query 'somepath(//src:model, @cuda//:cublasLt)'` returns nothing | [build.md](build.md) |

**Step 2d passed.** The adapter matched cuBLAS on the H100 80GB HBM3 at compute
capability 9.0, over the six Hopper cases and the weight-gradient case. That run
was the first time the adapter ran on a card. It closed the step that unblocked
the rest.

**What 2e decides, and the three ways it can go.**

1. The six rows of section 2 read at most 1.05 of their `cublas_ms` entry. Then
   step 5 may follow, and cuBLAS leaves the library.
2. A row reads above 1.05. The tile is then the first thing to move, because the
   probe sized five candidates and a card has measured one of them. The tile in
   `cutlass_bf16.cu` is 128 by 256 by 64, and `dev/cutlass/README.md` § 3 holds
   the four others and their stage counts. The crossover constant
   `kCutlassMinOutputElements` in `gemm_shape.h` is the second thing, and it is
   4096 elements by argument and not by measurement.
3. A row reads above 1.15. Section 9's stop condition applies: cuBLAS stays.

**What 2e read.** The first run tripped outcome 3: every row read between 0.051
and 0.154 of its `cublas_ms` entry, which is 2.6 to 9.4 percent of the card peak.
Section 11 is the answer to that, and section 11.9 found the cause in the build.
The run after that flag reads 0.576 to 1.176, so the gate is met on 19 of the 25
callable rows and missed on six, and nine rows are promoted. The rows are the
`gemm_cutlass` entry of [baselines.json](baselines.json).

**What a resumed session should read first**, in this order.

1. This section, for the state.
2. Section 6.1, for the decisions already taken and the record of each.
3. [backends/cuda/gemm/README.md](../backends/cuda/gemm/README.md) § 16, for what
   the adapter does and what it refuses.
4. [dev/cutlass/README.md](../dev/cutlass/README.md), for the probe's three
   answers and the two defects it found.

**The one number that is not a measurement.** `kCutlassMinOutputElements` is
4096 output elements. The argument is that the launch builds a Tensor Memory
Accelerator descriptor on every call, so a shape smaller than one 64 by 64 tile
should keep the hand tile. `gemm_bench` prints the refusal reason for every row
the rule declines, so the first card run reports whether that boundary is right.

**The three open questions.**

1. Does the cooperative schedule's one-block-per-multiprocessor residency beat
   the hand tile's two on the Muon rows, whose tile count is a few waves?
2. Does `beta` cost anything on the classifier weight gradient, which is the one
   row that reads C?
3. Does the 5 by 12 group lose its 0.005 ms to the size rule, or does the hand
   tile already hold it?

### 10.2 The six rows, and why step 5 does not follow

**The measurement.** Section 11.8 reads the production family as 25 callable
rows. Nine are promoted to a candidate. Nineteen keep cuBLAS, and six of those
nineteen miss the 0.95 gate outright:

| Row | Fraction of cuBLAS | What is left there |
|---|---:|---|
| `linear:wgrad:kv` | 0.576 | The ragged tail of the largest weight gradient. Stream-K, which is item 4b of section 6 |
| `linear:fwd:cls` | 0.804 | The classifier forward at the full vocabulary |
| `muon:640x1280:A` | 0.812 | A square Muon product whose tile count is a few waves |
| `muon:640x1280:A2` | 0.941 | The square of that square |
| `linear:dgrad:cls` | 0.952 | Inside the run-to-run spread of section 11.8 |
| `linear:dgrad:cproj` | 0.952 | The same |

**The arithmetic that decides step 5, and it is not about these six.** Step 5
takes cuBLAS out of the library. It is tempting to read the six rows above as
its blocker, because they are the rows the candidate loses. That reading is
wrong. The second column of section 11.8 says why: the arm a non-promoted row
would move to is the hand tile, which reads 0.147 to 0.740 of cuBLAS.

The blocker is the thirteen rows inside the 0.04 margin. For those rows the
candidate wins by a coin flip, and the only arm behind it is the tile. A removal
would move 19 of the 25 rows to that arm. Those 19 rows hold 0.94 of the step's
time, and the tile is between 1.4 and 6.8 times slower.

**The decision.** Step 5 is a campaign gate and not a build change. It does not
follow from this revision. cuBLAS stays in the library, and the seam keeps
serving a candidate where a measurement promoted one. The gate that would
reopen it is one line: every callable row at 1.05 of cuBLAS on some arm. The
margin rule of section 11.4 retires only after a row wins in two runs.
Two consequences follow, and both are stated so a later session does not
re-derive them.

1. **Item 4b was the only planned attack on a row of this table, and it is not
   reachable.** Section 10.3 states the vendor constraint. The sentence that put
   stream-K here described an intent and not a step, so the rows below 0.95 now
   have no planned remedy inside this adapter.
2. **A wider candidate is the other half, and no ticket exists for it.** The
   probe sized five tiles and the registry carries four; `dev/cutlass/README.md`
   section 3 holds the others. A tile that wins a row by 6 percent on the
   classifier at 512 threads is a different measurement and not a re-run.

The six rows are therefore not a work item with a date. They are the evidence
for the sentence "cuBLAS stays", and the record keeps the sentence and the
rows together so the next reader can check it.

### 10.3 Stream-K is not reachable from this adapter

**The claim that had to be checked.** Section 6 priced the k-split half of
stream-K at one to two weeks, and section 11.3 listed
`sm90_tile_scheduler_stream_k.hpp` as the piece that supplies it. Both rest on
an assumption: that a scheduler is a template argument of the kernel this
adapter already builds. It is not, and the vendor states it as a compile-time
refusal.

**The evidence, from the vendored headers.** The fetch is CUTLASS 4.8.1
(`include/cutlass/version.h`). Three facts decide the question.

1. `cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` line 111 asserts
   `is_void_v<TileScheduler_> || is_same_v<TileScheduler_, PersistentScheduler>`
   and its message reads "TMA warp-specialized kernel does not support
   specializing the tile scheduler". The cooperative kernel this adapter uses is
   that kernel, so the stream-K tag cannot be its argument.
2. `sm90_gemm_tma_warpspecialized_pingpong.hpp` line 105 refuses the tag by
   name. The other SM90 TMA kernel is therefore closed too.
3. The scheduler itself exists. `sm90_tile_scheduler_stream_k.hpp` defines
   `PersistentTileSchedulerSm90StreamK`, and `tile_scheduler.hpp` maps the
   `StreamKScheduler` tag to it for `arch::Sm90`. The tag is accepted by the
   SM100 kernels and by no SM90 one.

**The one SM90 route, and why it is not a candidate.**
`cutlass/gemm/kernel/gemm_universal_streamk.h` is the CUTLASS 2.x integer API:
it takes a `ThreadblockShape` and a `WarpShape` and runs the numbered-API
mainloop, with no TMA and no warp specialization. A rate from that kernel is not
comparable with the 1.05 gate, because the gate compares against cuBLAS on a
TMA mainloop. Moving the two ragged rows onto a slower kernel to fix their tail
would measure the mainloop and not the tail.

**What this decides.** Item 4b of section 6 leaves the phase. The rows below
0.95 keep cuBLAS, and section 10.2's second column stays true. Two routes remain,
and both are written here so a later session starts from the evidence.

- **A k-split in the hand tile.** The scheduler is ours there, and step 5 of
  section 10 already names the split as the part it did not do. The tile is
  1.4 to 6.8 times below cuBLAS, so a tail fix cannot close a rate gap that
  large.
- **A wider candidate for the row that is short of the gate by 20 percent.**
  `linear:fwd:cls` at 0.804 and `linear:wgrad:kv` at 0.576 are tile problems and
  not tail problems. `dev/cutlass/README.md` section 3 holds the tiles the probe
  did not promote, and the registry holds four of five.

**The general lesson for this campaign.** The vendored headers are the authority
on what the adapter can build, and one grep answers questions that a plan
estimated in weeks. Section 11.3's table is a list of headers and not a list of
capabilities, and this section is the difference between the two.

### 10.4 The fp8 route: what the vendor supplies, and what it costs

**State.** The arm is written, and it is clean off a card. `cutlass_fp8.cu` puts
the seam's `GemmScaledFp8` on the warp-specialized mainloop with E4M3 operands.
`gemm_shape.h` holds the arm's own refusal rule beside `JudgeCutlass`, and
`gemm.cu` calls the arm behind `NANOCHAT_GEMM_FP8=cutlass`. Nothing is promoted:
an unset variable leaves every call on cublasLt, so a shipped step does not
change until a card measurement promotes a shape. `gemm_benchmark --fp8` times
the arm beside cublasLt on the same operands. The card run is the one gate that
is still open.

**The one question that decides the design.** `GemmScaledFp8` takes
`scale_a` and `scale_b` as device pointers. The epilogue must therefore multiply
the accumulator by their product. The tree's rule is that such a scalar never
reaches the host (Part 2 section 1). A host read of the two scales would add one
synchronizing copy to every eligible matrix multiply, so it is not an option.
Two vendor facts decide whether a device form exists.

1. `cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` line 220
   defines `Sm90LinearCombination` with `Sm90ScalarBroadcast` for both alpha and
   beta, so alpha is a broadcast scalar and not a fused pointer.
2. `sm90_visitor_load_tma_warpspecialized.hpp` line 1010 defines that node, and
   its `Arguments` carries `scalar_ptrs` at line 1019 beside `scalars` and
   `dScalar`. The pointer form is present, so the epilogue loads alpha from
   device memory at run time.

**The design that follows.** One single-thread launch forms
`alpha * scale_a[0] * scale_b[0]` into a four-byte device slot, and the
epilogue's `scalar_ptrs[0]` names that slot. The slot is not part of the CUTLASS
workspace, because the epilogue reads it while the kernel runs. The mainloop
takes `float_e4m3_t` operands with an fp32 accumulator, which the same
`CollectiveBuilder` supplies. Beta stays a by-value scalar, because C is stored
in the output precision and needs no dequantization. No host read, and one extra
launch of one thread.

**The tile, and the spill that a host found.** The arm states one tile,
`kFp8Tile`, because an fp8 axis over the four bf16 rows would be sixty-four
instantiations. The first tile was the probe's `wide-n` row, 128 by 256 by 64 at
three stages, and it spilled. `cuobjdump -res-usage` read 168 registers with a
256 byte stack frame, and the spill gate of section 11.4 found the accumulator
in local memory inside the k loop of both instantiations.

The cause is the vendor's fp8 accumulation path. One-byte operands accumulate in
fp32 for many k blocks, so CUTLASS builds the fp8 mainloop with
`GmmaFP8Accumulation`, which keeps two fp32 accumulators and promotes one into
the other every `mma_promotion_interval` k blocks. At 256 wide N and two
warpgroups, one accumulator is 128 registers, so two do not fit. The probe's
`balanced` row halves N, which halves the accumulator, and the fp8 operand being
one byte leaves room for five stages where bf16 has three. The rebuilt target
then read 168 registers with no stack frame, and the gate read clean over both k
loops. No card was needed for either reading.

The gate needed a fix of its own. It looked only for `HGMMA`, which is the bf16
and fp16 multiply, so an fp8 body read as "no k loop to check" and passed
unseen. Hopper spells the eight-bit multiply `QGMMA`, and the gate now accepts
both. The four bf16 candidates read clean either way, so the change adds a check
without changing a verdict.

**The cost that bounds the row.** One target, two instantiations, and a measured
build of 53 to 93 seconds. `dev/cutlass/README.md` measures one instantiation at
19 s and 1.2 GiB. The peak of the compiler process therefore bounds the row, and
not the count: one device compile held 3.3 GiB. An fp8 axis over the existing
four rows would be sixty-four instantiations. The fp8 arm therefore needs its
own rows under the build's fp8 config, and not a third template axis on the
four. The registry's count is a build budget and not a free parameter.

**What the arm does not cover.** `CallSupported` takes one operand pair, E4M3 by
E4M3, which is the form of the forward. The seam's backward states E5M2 for both
operands ([precision.md](precision.md) section 5.2), so a shipped backward call
refuses the arm and keeps cublasLt. The vendor is not the limit: the atom list
carries `MMA_64x128x32_F32E5M2E5M2_SS_TN` and the mixed
`..._F32E4M3E5M2_SS_TN` at every N. The backward is therefore two more
instantiations and one more template argument, and it is its own row of the
build budget.

The operand form is the second limit. The SM90 fp8 GMMA exists only in the
K-major pair, so a row that states another form refuses the arm. The seam then
runs cublasLt under both selectors, and the two fp8 columns of that row read the
same time. A reader must therefore compare the columns only on a row whose form
is K-major, and no report states which rows those are. The seam answers nothing,
because `GemmScaledFp8` returns void, so the honest fix is a predicate the seam
exposes and the header is contract-owned.

**What the campaign still owes.** One card run of `gemm_bench --fp8` on the
Hopper leg, and the `gemm_fp8` entry of [baselines.json](baselines.json) that its
report writes. Its gate is the tolerance of [precision.md](precision.md) and no
row above 1.05. A promotion follows only if a row clears the margin of section
11.4.

One prerequisite sits on the card route itself. `tools/modal/app.py` holds two
legs, and the H100 leg builds at bf16, so no image can run the fp8 arm today. A
third leg at fp8 is the missing piece, and the file belongs to the Python
surface workstream. A caller must not fold it into the bf16 image, because the
report's `host.precision` is a fact of the library that ran.

The E5M2 backward is a second, smaller row: it needs its own instantiations
before any call can take it.

### 11. The candidate registry

**Status.** All six steps of section 11.7 are done, and step 5 ran twice on an
H100 80GB HBM3. The first run measured spilling kernels and promoted nothing.
Section 11.9 found the cause in the build, cleared it, and the second run
promoted nine of the 28 production rows to a candidate; cuBLAS keeps the other
19 and the hand tile keeps none. Nothing here changes a frozen header, a
capability bit, or a `caps.json` field.

#### 11.1 The measurement that asks for it

One run of `gemm_bench --json` on the H100 leg timed three arms on one set of
operands: cuBLAS, the Hopper mainloop of `cutlass_bf16.cu`, and the hand tile.
The fraction is the cuBLAS time divided by the arm time, so the gate of step 2e
asks for 0.95 or more.

| Row | cuBLAS | CUTLASS | Hand tile | CUTLASS fraction | Hand fraction |
|---|---:|---:|---:|---:|---:|
| Muon `1280x1280:A` | 0.2230 ms | 1.8669 ms | 0.7414 ms | 0.119 | 0.301 |
| Linear forward `fc` | 0.2598 ms | 2.3470 ms | 0.9654 ms | 0.111 | 0.269 |
| Linear wgrad `kv` | 0.0524 ms | 1.0294 ms | 0.3779 ms | 0.051 | 0.139 |
| Classifier forward | 1.5888 ms | 14.7225 ms | 5.7837 ms | 0.108 | 0.275 |

The CUTLASS arm reads 0.051 to 0.154 on every row, at 2.6 to 9.4 percent of the
card peak. The hand tile reads 0.14 to 0.43, at 7 to 29 percent. The small rows
sit at its low end, because a launch costs more than the product. The adapter is
therefore 2.0 to 3.4 times slower than the tile it exists to retire. The full
row set is the `gemm_cutlass` entry of [baselines.json](baselines.json).

These 28 rows are the production sample. They say what the adapter does in the
one model, and not how it scales. Section 11.5 adds the row families that
measure the kernel itself.

This table is the measurement that asked for section 11. The sweep of section
11.4 replaced it. One tile became four candidates, the hand column became the
row's own schedule, the operands arrived cold, and each candidate was checked
against cuBLAS. Section 11.8 states the re-measured numbers, and the
`gemm_cutlass` entry of [baselines.json](baselines.json) carries their rows.

#### 11.2 What the generated machine code shows

Every one of the eight kernels spills the accumulator to local memory inside the
k loop. Sixty-four stores and sixty-four loads sit between the multiply
instructions, and the stack is 576 bytes. The multiply writes registers 64 to
191, and the ceiling is 168: 384 threads at one block per multiprocessor is
65536 divided by 384. The accumulator does not fit, so its tail leaves the
register file on every step.

Three experiments narrowed the cause. Each one is a local rebuild, and none of
them stays in the tree.

- The device flag `-O3` in place of `-O2` changed nothing: same 168 registers,
  same 576-byte stack, same sixty-four spills.
- A 128 by 128 by 64 tile halved the spill and kept the symptom.
- The pingpong schedule with no cluster reached 255 registers and a 2520-byte
  stack.

The local machine code comes from CUDA 12.0 on the development host, and the
container ran CUDA 12.4.1. The measured fractions are the authority, and this
inspection only explains their shape.

A 9x gap on a mature library is not the warpgroup ceiling of section 3. It is a
defect of this instantiation. The tree must find that defect before it records
the negative result of section 9. A removal that stops on a defect states the
wrong reason for a right decision.

#### 11.3 What CUTLASS supplies, and what the fetch does not

`third_party/cutlass.bzl` extracts `include/` and `LICENSE.txt` only. The
external repository holds no `tools/`, `test/`, or `examples/`, so
`cutlass_profiler` and its generated kernel list are not reachable. Widening the
fetch is not viable: the archive is 169 MB, the generated list is thousands of
translation units, and the development host holds 7.8 GiB.

The headers the fetch does hold are the pieces a registry needs.

| Header | Use |
|---|---|
| `cutlass/gemm/collective/collective_builder.hpp` | One candidate for each instantiation; the tile, the cluster, and the stage count |
| `cutlass/epilogue/collective/collective_builder.hpp` | The epilogue schedule and the fused alpha and beta |
| `cutlass/gemm/device/gemm_universal_adapter.h` | `can_implement`, `get_workspace_size`, and `run` |
| `cutlass/kernel_hardware_info.h` | The device identifier and the multiprocessor count |
| `cutlass/device_kernel.h` | The odr-use a kernel body needs, and the handle for an occupancy query |
| `cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp` | Stream-K, whose split count is a run-time value. The header is present, and section 10.3 states why no kernel of this adapter can take it |
| `cutlass/cluster_launch.hpp`, `cutlass/cuda_host_adapter.hpp` | Cluster launch and graph capture, for a later phase |

The method of CUTLASS is therefore the thing to copy, and not the tool. The
method has five parts: a kernel list, a check for each candidate, one timing
protocol, a machine-readable report, and a stored choice.

#### 11.4 The five pieces

1. **A candidate registry.** `backends/cuda/gemm/cutlass_candidates.h` holds the
   table, and one source file, `cutlass_candidate.cu`, holds every kernel body.
   A candidate is a tile, a cluster, a stage count, a kernel schedule, and an
   epilogue schedule. The table is `constexpr`, so a candidate's translation
   unit takes its own row as template arguments. Build cost bounds the count, so
   start with four candidates. One candidate is eight instantiations: four
   orientations and two store types. `dev/cutlass/README.md` measures one
   instantiation at 19 s and 1.2 GiB. The build therefore compiles that one
   source file once for each index. That keeps the memory peak inside the host
   budget.
2. **A spill gate.** A host check runs `cuobjdump -sass` over each candidate
   object and rejects a candidate with `STL` or `LDL` between its `HGMMA`
   instructions. This gate would have caught the defect above in three minutes
   with no card. The tool is `tools/gemm_spill_gate.py`, and a card is no part
   of its run.
3. **A check for each candidate.** The benchmark compares each candidate with
   cuBLAS over the operands it times, using the comparison of
   `gemm_gpu_test.cc`. A candidate that is fast and wrong must not win.
4. **One timing protocol.** The benchmark flushes the second-level cache between
   iterations, reports the median and the minimum of the rounds, and reports the
   occupancy the driver answers for the candidate. The flush stops a warm cache
   from choosing the winner: the classifier operands are 63 MB and the
   second-level cache is 50 MB.
5. **A decision table.** `backends/cuda/gemm/cutlass_rules.h` maps a shape to a
   candidate, with an explicit order of candidate, then hand tile, then cuBLAS.
   `JudgeCutlass` reads that table, and `NANOCHAT_GEMM_CANDIDATE` is the
   benchmark-only override, as `NANOCHAT_GEMM_SCHEDULE` is today. A tool under
   `tools/` writes the header from the benchmark report, so the table is data.
   The tool is `tools/gemm_rules.py`: `--write` writes the block and `--check`
   fails a hand edit.

   **A promotion needs a margin, and not a tie.** Two runs of one sweep disagree
   by up to 4 percent on a row whose arms are close. The tool therefore promotes
   an arm only when it beats cuBLAS by 0.04 of the cuBLAS time. The flag is
   `--min-margin`, and its default is that measured value. Section 11.8 states
   the spread it comes from.

The choice is offline. A training step pays one table lookup. A run-time timing
pass would put card work on the start of every step, and the shape rule would
stop being a host-only check.

#### 11.5 The measurement rows

The 28 rows of the closed list are the production sample: they are the shapes the
one model issues. They cannot say how a kernel scales, because they cluster in
one region and they all come from one profile. A candidate that wins on them can
still fall off a cliff at 4096 or at 16. So the benchmark needs a second row
family, and this design adds five families beside it.

**The production rows choose, and the measurement rows characterise.** Only the
closed list feeds the decision table of section 11.4. The extra families decide
whether a candidate is fit to enter the sweep, and they are the record that a
later tuning session reads.

**The rows.**

| Row | Shape | Work | What it measures |
|---|---|---:|---|
| `scale:16` | 16x16x16 | 8 KFLOP | The launch floor |
| `scale:64` | 64x64x64 | 524 KFLOP | A quarter of one tile |
| `scale:256` | 256x256x256 | 33 MFLOP | Below the 128 by 256 tile |
| `scale:1024` | 1024x1024x1024 | 2.1 GFLOP | One partial wave |
| `scale:2048` | 2048x2048x2048 | 17 GFLOP | One full wave |
| `scale:4096` | 4096x4096x4096 | 137 GFLOP | Four waves |
| `scale:8192` | 8192x8192x8192 | 1.1 TFLOP | An operand above the second-level cache |
| `scale:16384` | 16384x16384x16384 | 8.8 TFLOP | A large matrix, 512 MiB an operand |
| `scale:32768` | 32768x32768x32768 | 70 TFLOP | The device memory ceiling |
| `aspect:thin-m` | 128x4096x4096 | 4.3 GFLOP | Too few tiles in M |
| `aspect:thin-n` | 4096x128x4096 | 4.3 GFLOP | Too few tiles in N |
| `aspect:deep-k` | 1024x1024x32768 | 69 GFLOP | A long k loop and few tiles |
| `aspect:thin-k` | 4096x4096x64 | 2.1 GFLOP | One k slice |
| `aspect:tall` | 16384x256x2048 | 17 GFLOP | A decoder-like shape |
| `batch:1` | 512x512x512, batch 1 | 0.27 GFLOP | One slice |
| `batch:8` | 512x512x512, batch 8 | 2.1 GFLOP | A few slices |
| `batch:40` | 512x512x512, batch 40 | 11 GFLOP | The Muon batch |
| `batch:128` | 512x512x512, batch 128 | 34 GFLOP | Many slices |
| `batch:muon-tall` | 4096x4096x1024, batch 4, A transposed | 137 GFLOP | The `A = XᵀX` form |
| `batch:muon-wide` | 1024x4096x4096, batch 4 | 137 GFLOP | The `XXᵀ` and `B X` forms |
| `ragged:m` | 1000x1000x1024 | 2.0 GFLOP | A partial M tile |
| `ragged:k` | 1024x1024x1000 | 2.1 GFLOP | A partial k slice |
| `ragged:lda` | 1024x1024x1024, `lda = k + 1` | 2.1 GFLOP | A leading dimension off the tile |
| `form:nn` | 2048x2048x2048, plain A and B | 17 GFLOP | The first operand form |
| `form:tn` | 2048x2048x2048, transposed A | 17 GFLOP | The second operand form |
| `form:nt` | 2048x2048x2048, transposed B | 17 GFLOP | The third operand form |
| `form:tt` | 2048x2048x2048, both transposed | 17 GFLOP | The fourth operand form |
| `form:fp32-c` | 2048x2048x2048, fp32 C | 17 GFLOP | The weight-gradient store |
| `align:offset` | 1024x1024x1024, one element off the boundary | 2.1 GFLOP | The general path |

**The harness computes the work.** A row holds extents, a batch, two transpose
flags, and a store type. The report derives the flop count, the operand bytes,
and the tile count from those fields. A mistyped row therefore fails the host
test, and not a card run.

**One buffer set serves every row.** The largest row needs 8 GiB: two 2 GiB
operands and a 4 GiB fp32 C. The harness sizes the workspace once from the
largest row and reuses it, so no row pays for an allocation.

**The fill is on the device.** Today `Fill` in `gemm_benchmark.cc` builds the
operands on the host and copies them. That works at 60 MB and it cannot work at
2 GiB, because the sandbox caps host memory at 5 to 6 GiB in the benchmark
profile. The design therefore fills on the device with one small launch, and the
host holds no operand.

**The check on a big row is a reduction.** A full comparison of a 4 GiB C buffer
needs a second buffer and a host copy, and neither fits the budget. For a big row
the check computes the Frobenius norm of the difference and a sampled row set on
the device, and it compares those two values with cuBLAS. The small rows keep the
full element comparison of `gemm_gpu_test.cc`.

**The iteration count follows the work.** A round must run at least 50 ms. The
harness therefore sets the iteration count from the row's ideal time at the
declared peak, with a floor of 3 and a ceiling of 1000. The row reports its count, so a reader
can see that `scale:32768` ran three times and `scale:16` ran a thousand.

**A card with less memory skips a row, and says so.** The harness takes a device
budget from the card, and it reports a row that exceeds the budget as skipped,
with the byte count and the reason. A small card therefore runs a prefix of the
set, and the report names the rows it dropped.

**The extra rows do not enter the table.** They answer a question about a kernel
and not about a model, so they never select a candidate for a production shape.
They do gate one thing. A candidate whose curve falls below the hand tile's curve
at every work size leaves the registry before the card sweep of section 11.7.

**The scaling row is a curve, so its gate is monotonicity.** For two square
probes with a larger side, the rate must not fall by more than a factor of two.
The gate excepts the tiny rows, because a launch dominates their time. Section 7
already states that exception for the 5 by 12 group.

#### 11.6 How each arm gains

**The CUTLASS arm gains in five ways.**

1. The gate targets the defect. The spill is the 9x, and a candidate without it
   is a different kernel. Phase 0 measured the same instruction at 96 percent of
   the card peak with no data movement.
2. The arm stops betting on one instantiation. Today one tile serves the 5 by 12
   group, the classifier at 32768 wide, and the weight gradient with an fp32
   store. No single kernel is right for all four.
3. The crossover stops being an argument. `kCutlassMinOutputElements` is 4096
   elements by argument, and the tiny group pays about 0.003 ms per step
   because of it.
4. The flush removes the warm-cache illusion, so the winner is the winner under
   the data path a training step pays.
5. The occupancy column shows the residency cliff that a reader must otherwise
   infer.

**The hand tile gains in three ways, and its ceiling stays.**

1. Its schedule choice becomes trustworthy: the sweep gains the flush, the
   median, and the numerical check.
2. Its selection becomes per shape. All 28 rows are inert today, and the
   measurement switch is global, so a run measures an arm and not a shape.
3. The vectorized row kernels and the classifier hand-off of Part 4 each need a
   before and after row, and this protocol supplies one for both arms.

The hand tile's ceiling is the instruction set. A warp-level tile holds about
half the card, and this run reads 0.14 to 0.43 of cuBLAS for the hand tile. No
benchmark raises that ceiling, so the hand tile wins only where the other two
lose.

#### 11.7 The work order, and the limits

| Step | Work | State | The command that decides it |
|---|---|---|---|
| 1 | The measurement families of section 11.5: the rows, the device fill, and the budget skip | **Done** | `tools/nanochat test //backends/cuda/gemm:bench_shapes_test` |
| 2 | A registry of four candidates, one source compiled once for each index | **Done, and the gate fails**: every candidate spills | `tools/gemm_spill_gate.py` after a build of `//backends/cuda/gemm:gemm_benchmark` |
| 3 | The check for each candidate in the benchmark | **Done, not yet run on a card** | `gemm_benchmark --self-test` |
| 4 | The flush, the medians, and the occupancy column | **Done** | A `gemm_bench` run of the production rows |
| 5 | One card run of the sweep, on both row families | **Done twice**: the first on spilling kernels, the second after section 11.9, which promotes nine rows | `modal run tools/modal/app.py::card_job --name gemm_bench --argv '["--rows","all","--json"]'`, then the same with `--rows production` |
| 6 | The table, the seam lookup, and then step 5 of section 6 | **Done**: 28 rows, nine promoted, and the seam serves them | `tools/gemm_rules.py --report FILE --check`, then the dependency query of gate G4 |
| 7 | The shipped seam on a card, row by row | **Done, and it passed**: 28 rows, no rejection | `modal run tools/modal/app.py::card_job --name gemm_bench --argv '["--rows","production","--seam","--json"]'` |

**What landed, and where it is recorded.** `backends/cuda/gemm/README.md`
section 17 is the package's own record. It holds the four candidates, the spill
gate, the six steps of the sweep, and what is still missing. Four gates run
without a card. `candidate_registry_test` checks the table of candidates.
`bench_shapes_test` checks the rows and their arithmetic. `dispatch_test` checks
the lookup, and that every written row keys a shape of the closed list.
`gemm_rules_test` and `gemm_spill_gate_test` check the two tools.

**What the card run decides.** Step 5 is the one measurement this design exists
to take. It answers the question of section 11.2. Does a candidate without the
spill exist at all, and does it reach the 1.05 fraction of step 2e? The spill
gate answers the same question earlier and without a card. A sweep of four
candidates that all spill states a result about the tiles, and not about the
instruction.

**The spill gate's first run.** The gate ran on the four built objects at
`sm_90a` with the development host's CUDA 12.0. Every candidate fails it: each
one of the eight instantiations of each candidate touches local memory inside
its k loop.

| Candidate | Tile | Stages | Kernels that spill | The worst instantiation |
|---|---|---:|---|---|
| `wide-n` | 128x256x64 | 3 | 8 of 8 | 8 HGMMA, 64 LDL, 64 STL |
| `balanced` | 128x128x64 | 5 | 8 of 8 | 8 HGMMA, 32 LDL, 32 STL |
| `deep-k` | 128x128x128 | 2 | 8 of 8 | 16 HGMMA, 32 LDL, 32 STL |
| `tall-m` | 256x128x64 | 3 | 8 of 8 | 16 HGMMA, 64 LDL, 68 STL |

The `wide-n` row states the two numbers of section 11.2 exactly: sixty-four
loads and sixty-four stores. The gate therefore reproduces that finding, and it
is not a new instrument. The `balanced` row halves the count and keeps the
symptom, which agrees with the second local experiment of that section.

**What that decides.** The candidate list of section 11.4 was the bet of section
11.6 item 2: four tiles, and one of them without the spill. The gate says the
spill is not a property of the tile. Section 11.4 also states that a spilling
candidate leaves the registry before a sweep. The registry therefore holds no
candidate that the gate accepts, and the sweep of step 5 ran anyway. The next
section states what it read.

The design adds no fused kernel, no new instruction, and no megakernel. It does
not raise the ceiling of the hand tile. It does not fix the spill by itself: a candidate must exist
without it, and the spill gate decides whether the candidate is real.

| File | Change | Owner |
|---|---|---|
| `backends/cuda/gemm/cutlass_candidates.{h,cc}` | The registry | Kernel agent |
| `backends/cuda/gemm/cutlass_candidate.cu` | One kernel body, compiled once for each candidate | Kernel agent |
| `backends/cuda/gemm/bench_shapes.{h,cc}` | The measurement rows, the device fill, and the budget skip | Kernel agent |
| `backends/cuda/gemm/gemm_benchmark.cc` | The check, the flush, and the occupancy column | Kernel agent |
| `backends/cuda/gemm/dispatch.{h,cc}` and `cutlass_rules.h` | The table and the lookup | Kernel agent |
| `include/nanochat/bench.h` | The new report fields | Harness |
| `tools/gemm_rules.py`, `tools/gemm_spill_gate.py` | The table writer and the spill gate | Oracle |
| This document, and the `gemm_cutlass` and `gemm_scaling` entries of [baselines.json](baselines.json) | The record | Architect |

One row of that table is closed. The per-row iteration count and the occupancy
a candidate answers print in the benchmark's table. They now also travel in the
machine-readable report, as two optional row keys beside `ms` and `gflops`
(`include/nanochat/bench.h`). A row that states neither omits both, so an older
row of the same schema reads as it did.

`//src:bench_test` guards the three cases. An unstated field writes no key, a
stated one writes an integer, and the median row of a measurement carries the
same two facts as the minimum. `gemm_benchmark` states the iteration count on
every arm and the occupancy on the candidates. It states neither on the seam
row, which is cuBLAS on most shapes and a candidate on a promoted one.

#### 11.8 What the card run read

**Two runs, one revision, one card.** Both ran on an H100 80GB HBM3 at bf16
after the build flag of section 11.9. Every arm they timed was therefore a
kernel that keeps its accumulator in registers. The first run stated
`--rows all --json`. The second stated `--rows production --json`, which is the
family the decision table may read. The protocol is section 11.4: a 100 MiB flush of the second-level
cache before every timed call at 0.035 ms, three rounds, and the minimum of the
rounds. The check accepted every measured arm of both runs, and no row was
skipped for the device budget.

**The production rows, and the gate.** The fraction is the cuBLAS time over the
arm time, so the gate of section 10.1 step 2e asks for 0.95 or more.

| Arm | Old, spilling kernels | New, this revision |
|---|---|---|
| The best candidate on each row | 0.137 to 0.315 | **0.576 to 1.176** |
| The hand tile on each row's own schedule | 0.149 to 0.782 | 0.147 to 0.740 |

Twelve of the 25 callable rows are faster than cuBLAS. Six are clearly slower:
`linear:wgrad:kv` 0.576, `linear:fwd:cls` 0.804, `muon:640x1280:A` 0.812,
`muon:640x1280:A2` 0.941, and `linear:dgrad:cls` and `linear:dgrad:cproj` at
0.952. The 5 by 12 Muon group refuses every candidate by the size rule of
`gemm_shape.h`, and `align:offset` refuses them by the pointer rule.

**Two runs disagree by more than a tie.** This is why the margin of section 11.4
item 5 exists. The two runs agree to about 1 percent on most rows. Five rows
moved by more than 3 points between them.

| Row | First run | Second run |
|---|---:|---:|
| `linear:wgrad:cls` | 0.942 | 1.005 |
| `linear:fwd:q+proj` | 1.020 | 0.978 |
| `linear:fwd:kv` | 0.973 | 1.007 |
| `linear:wgrad:fc` | 0.974 | 1.016 |
| `linear:wgrad:kv` | 0.644 | 0.576 |

A promotion taken on a single run would therefore have shipped three coin flips.

**What the table promotes.** `tools/gemm_rules.py --write` read the second run
and promoted nine rows. Each one beats cuBLAS by more than 0.04 in that run, and
each one also beat it in the other run.

| Row | Candidate | Fraction of cuBLAS |
|---|---|---:|
| `muon:1280x5120:prod` | `wide-n` | 1.176 |
| `muon:5120x1280:prod` | `wide-n` | 1.130 |
| `muon:1280x5120:A2+A2` | `wide-n` | 1.127 |
| `linear:wgrad:q+proj` | `balanced` | 1.118 |
| `muon:5120x1280:A` | `wide-n` | 1.103 |
| `muon:1280x5120:A` | `wide-n` | 1.085 |
| `linear:wgrad:cproj` | `wide-n` | 1.055 |
| `linear:dgrad:fc` | `wide-n` | 1.048 |
| `linear:fwd:cproj` | `wide-n` | 1.046 |

Three rows were faster and stayed on cuBLAS, because their win is inside the
margin: `linear:wgrad:fc` 1.016, `linear:fwd:kv` 1.007, and `linear:wgrad:cls`
1.005. CuBLAS serves the other 19 rows. `deep-k` wins no row at this revision:
`wide-n` wins eight and `balanced` two.

**The measurement rows, and the curve.** cuBLAS holds 693 to 787 TFLOP/s from
4096 up, which is 70 to 80 percent of the card's declared bf16 peak. The
candidates hold 576 to 705 TFLOP/s, which is 58 to 71 percent, against 84 to 96
before the build flag. The hand tile saturates at 115 to 148, unchanged, because
no flag reaches it. At `scale:4096` the best candidate is faster than cuBLAS,
701.5 against 692.8 TFLOP/s. The monotonicity gate of section 11.5 passes.

**The shipped seam is on the gate too.** `gemm_benchmark --seam` clears
`NANOCHAT_GEMM_LIB` and calls the seam for each row. The arm it measures is
therefore the one a training step runs: a candidate on a promoted row, and
cuBLAS on every other one. Its own check compares that arm with the cuBLAS
reference, so a promoted row is gated on a card before it ships. The arm is
separate from `all`, because the table writer reads a report of the three
measured arms.

**The seam run passed.** All 28 production rows matched their reference, and no
row was rejected. On the nine promoted rows the seam read 0.87 to 0.97 of the
cuBLAS time, which is the promotion's gain in the shipped path. Two rows read
10 to 14 percent slower than the sweep did on the same shape: `linear:dgrad:cls`
and `linear:wgrad:cls`, the two largest rows in the model. Both keep cuBLAS, so
no shipped decision rests on them.

**The arm's first run found a defect in the protocol, and not in a kernel.** The
three weight-gradient rows were rejected with a relative error of exactly ten.
A weight-gradient row states `beta = 1`, so a measured arm leaves eleven products
in C: two warmup calls and nine timed ones. A run that does not measure cuBLAS
took its reference from a single call, so the two sides differed by ten products
and by nothing else. The reference is now measured under the same call count, and
the timing of that call is dropped. The defect was invisible while every run
measured cuBLAS, because then both sides accumulated the same number of times.

**What the run decides.** The gate of step 2e is met on 19 of the 25 callable
rows, and it is missed on six. The nine promotions make a real part of the model
run the candidate: one call of each production row reads 10.30 ms against 10.61
ms on cuBLAS. cuBLAS therefore stays in the library for the rows the candidate
loses. Section 6's step 5, its removal, does not follow yet. Section 9's stop
condition does not apply either, because the adapter is no longer behind by a
factor of three.

**What it also corrects.** The `gemm_cutlass` entry this replaces was measured
twice over: once on a warm cache, and once on kernels that called out of their
own k loop. The flush moved the cuBLAS column of the older entry by 0 to 12
percent. The build flag moved the candidate column by a factor of 2.6 to 3.9.
The note of the entry states both.

The rows of both families are the `gemm_cutlass` and `gemm_scaling` entries of
[baselines.json](baselines.json).

#### 11.9 The cause of the spill, and the flag that clears it

**A local rebuild found it, and no card was needed.** Section 11.8 states that
every candidate spills its accumulator. This section states why. It also states
the measurement that showed the reason without a card.

**The kernel called out of its own k loop.** The target never defined
`NDEBUG`. Every `assert` of the vendored CUTLASS headers therefore compiled into
a device call to `__assertfail`. ptxas cannot inline that call, and two defects
follow.

1. ptxas serializes the `wgmma` instructions around the call. Its warning names
the reason: `C7509: wgmma.mma_async instructions are serialized due to the
presence of Extern calls`. All eight instantiations of a candidate report it.
2. The call forces the accumulator out of the register file around every call
site. Candidate 0 measured a 576-byte stack, 2090 `LDL`, and 1124 `STL`.

**One build flag clears both.** The `cutlass_bf16` target now states
`-DNDEBUG`. One rebuilt translation unit of candidate 0 then reads:

| Row | Before | After |
|---|---:|---:|
| `CALL.ABS.NOINC` | 24 | 0 |
| Stack | 576 B | 0 B |
| `LDL` and `STL` | 2090 and 1124 | 0 and 0 |
| Registers | 168 | 168 |
| ptxas warnings | `C7509`, eight of eight | none |

**The gate accepts all four candidates.** `tools/gemm_spill_gate.py` reads clean
for `wide-n`, `balanced`, `deep-k`, and `tall-m`. Each one holds 8 of 8 kernel
bodies with no local access between the multiplies. `wide-n` and `balanced` hold
64 `HGMMA` each, and `deep-k` and `tall-m` hold 128 each. The same objects hold
no call and no stack.

**The register reallocation never worked, and it no longer matters.** The PTX of
every candidate holds 16 `setmaxnreg` instructions, and ptxas drops all of them
with `C7508: 'setmaxnreg' ignored; unable to determine register count at entry`.
The budget is therefore static at 168 registers, which is 65536 divided by 384
threads. A 128-wide tile needs 128 of them for the accumulator, and that fits.
The spill came from the calls and never from the tile.

**What this decides.** Section 11.2 read the spill as a property of the tile.
It was a property of the build. The negative result of section 11.8 therefore
rests on broken machine code, and the card measurement must run again before
section 9's stop condition stands. The decision table of section 11.4 item 5 is
inert until then, and cuBLAS keeps every shape in the meantime.

**The gate gained a second rule.** A call between two multiplies is now a
refusal of its own. `tools/gemm_spill_gate.py` reports it as `SERIALIZES`, names
the call count, and exits non-zero. Two cases of `gemm_spill_gate_test` state the
rule: a call inside the k loop refuses the body, and a call outside it does not.

**The flag is target-local.** No other device code in the tree holds a runtime
`assert`, so no other target inherits the defect. The shared option list stays
as it is.

## Part 4. Two kernel designs, and what landed

The arithmetic below is agreed. Both changes keep the frozen seam's meaning and
neither adds a family. Each section states what shipped, what the gate is, and
what is left.

### 1. Vectorized row kernels

**State.** Shipped for five families and refused for the sixth. The helper is
header-only in `backends/cuda/kernels/device_utils.cuh`, so it inlines and never
merges a kernel boundary, and it landed as `ComputeVector`, `FloatVector`,
`VectorReady`, and `VectorGrid`. This section previously named a `Packed<T>`
with `LoadPacked` and `StorePacked`; no such symbol exists, so a reader of an
older note looks for the wrong name. `ComputeVector` is `alignas(16)` and holds
`16 / sizeof(ComputeType)` elements, so the fp32 build packs four.

| Family | State | What moves in a vector |
|---|---|---|
| `pointwise.cu` | Shipped before this part | both operands and the result, for every flat code and the row-scale code |
| `attention.cu` | Shipped before this part | the streaming legs |
| `adamw.cu`, `muon.cu`, `muon_fused.cu` | Shipped before this part | the fp32 state and the compute copy beside it |
| `rms_norm.cu` | This change | `x`, `out`, `dy`, `dx`: the write pass of the forward and of the backward |
| `embedding.cu` | This change | the forward's gather of `out`, one table row per vector |
| `classifier.cu` | This change | the backward's write of `dlogits` |
| `grad_norm.cu` | Not done, on purpose | nothing: the reason is below |

**No reduction moves.** Every family in the table keeps its scalar reduction
helper. Only the element-wise traffic is packed. A packed call is therefore
identical to the scalar one in every bit, and not merely close. This is stricter
than the plan asked for: the plan accepted a moved per-thread sum and let the
oracle tolerance decide. The tree does not need to accept that, because no
reduction has to be packed to reach the bytes.

**`grad_norm.cu` is the one refusal, and the reason is arithmetic.** Its partial
kernel accumulates the sum of squares in double, and that norm is the value the
clip scale divides every gradient by. A packed load re-maps a thread onto four
consecutive elements instead of a strided set, so the double accumulation order
changes and the norm moves in its last bits. That is a change to every update of
the step, and it is not a change the plan's "Low" risk column priced in. The
read is also already coalesced: one warp reads 32 consecutive floats, so the
vector form saves one instruction per four elements and not one byte of traffic.
It buys no bandwidth and spends the reference, so it stays out until a card
measurement asks for it.

**The byte-rate column is in.** `row_benchmark.cc` reports `GB/s` beside
`ms/iter`. The count is the algorithmic traffic of one call. A kernel that reads
a row twice counts that read once, because the second pass is the one the cache
absorbs. Without the column a packed loop shows up as a time with no
denominator, which is why the plan put the column first.

**The three conditions hold in every family.** The pointer is 16-byte aligned,
the row width divides by the vector width, and the row's own base is aligned by
that divisibility. `VectorReady` states the first two for the caller, and the
scalar kernel is the fallback. A buffer the caller does not own is therefore
never read past its end, and a shape the tests use can never reach a packed loop
by accident.

**What the gates are.** The existing family tests ride the GPU tiers. They now
cover both paths: a tiny shape whose row width is 4 or 8 takes the packed loop,
and a shape whose width is odd takes the scalar one. The CPU suite is unchanged,
because the CPU backend has one loop and that loop is the reference.

The simulator leg ran and passed. `tools/nanochat simulate --device h100`
compiles the same `.cu` sources with the host compiler, and it runs each block as
a set of fibers. Three tests therefore compare every packed path with the CPU
reference: `//tests:oracle_test`, `//tests:train_parity_test`, and
`//tests:numerics_trace_test`. All three passed.

What is left is the on-card tier, which measures and does not re-derive the
arithmetic. This part is therefore `Correct off device, and the GPU tiers are
pending a card run`.

### 2. The classifier statistics hand-off

The classifier holds `raw_logits_` and `dlogits_`, each 4.3 GB at `R = 32768`
and `padded_vocab = 32768` in fp32. The logits buffer moves six times per step,
and two of those passes rebuild statistics that the forward pass already had.

Two independent changes remove most of that traffic.

1. **The statistics hand-off.** `ClassifierForward` writes `(row_max, sum_exp)`
   per row and `ClassifierBackward` reads the pair. The pair is `2 * rows`
   floats, about 256 KB. The value is the same bit for bit on a device, because
   both calls use the same input, the same reduction, and the same order. Two
   logits reads and one `RowLogSumExp` call leave.
2. **The fused scale.** `ClassifierBackward` takes a `dlogit_scale` and writes
   `dlogits[j] = (p - onehot) * sech2 * row_scale * dlogit_scale`. The separate
   `PointwiseForward(kScale)` pass disappears. The store keeps the two roundings
   that pass had: the value rounds to the compute type, and the scale multiplies
   that rounded value. A bf16 build therefore keeps the bits it shipped with,
   and an fp32 build sees one multiplication by one constant.

Backward logits reads fall from 3 to 1 and scale passes from 2 to 0, which is
about 17 GB less traffic per step.

**The count applies to the fused family, and a card runs the fused family.**
`backends/cuda/caps.json` declares `override_classifier` on both `h100` and
`t4`, so the CUDA path takes the fused kernel. `src/ops.cc` takes the
composition only on a device without the bit. The CPU backend, the simulator's
reference, and the evaluation path keep the composition, and
`src/reference_caps_test.cc` pins that the reference mask has no such bit.

The seam change is additive, and the architect owns the header. It landed with
the two new arguments trailing and defaulted, so no call site moved:

```cpp
// Forward writes the per-row statistics when `stats` is non-null:
// stats[2*r] = row_max, stats[2*r+1] = sum_exp. Evaluation passes null.
void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       float* losses, float* stats = nullptr);

// Backward consumes the statistics when `stats` is non-null, and rebuilds them
// otherwise. `dlogit_scale` folds the loss normalization into the write.
void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        ComputeType* dlogits, const float* stats = nullptr,
                        float dlogit_scale = 1.0f);
```

The defaults keep every call site valid, and the null path keeps the current
behaviour, so the kernel test compares the two paths.

**What landed, and what each piece owns.** Both backend arms state both
arguments. `backends/cuda/kernels/classifier.cu` writes the pair from the
`RowLogSumExp` it already ran, and reads it instead of rebuilding.
`backends/cpu/kernels.cc` does the same. `src/ops.h` gains `stats` and
`dlogit_scale` on `ClassifierShape`.

`ops::ClassifierBackward` returns a bool that says whether the call folded the
scale. Only the fused arm can fold it, and the composition must still see its
pass. `src/model.cc` allocates `classifier_stats_` as `2 * rows` floats. It
fills that buffer on the training forward only, and drops the full-buffer
`kScale` pass when the call folded. The weighted path states `dlogit_scale` of
one, because its per-row array already carries `scale / divisor`. It therefore
keeps the arithmetic it had.

**One caveat, stated so no one chases it.** The CPU arm accumulates `sum_exp` in
double, so the pair it stores is the float-rounded sum and the rebuild is the
double one. The two paths there agree to the last bit of a float and not past
it, which the test states as a `1e-6` tolerance. On a device both paths call the
same float reduction, so the test compares those two for exact equality.

**What this session did not measure, and what decides it.** The 17 GB and the
step time are the design's arithmetic and not a recorded run. The gate is the
step protocol of `docs/performance.md`. It takes two untraced runs of 200 steps
on `base` at the H100, one before and one after. Each run reports the step time
and the Model FLOPs share. A `base` row enters [baselines.json](baselines.json)
only when the two runs agree on a direction.

The correctness gate ran here, and it needs no card. `//backends/cpu:kernels_test`
compares the two paths. `tools/nanochat simulate --device h100` runs the fused
classifier through the emulator, because the declared `h100` mask carries
`override_classifier`, so `//tests:train_parity_test` and
`//tests:numerics_trace_test` compare the hand-off with the CPU reference and
both passed. `//backends/cuda/kernels:classifier_gpu_test` is the on-card check.
It compares the two paths for exact equality, beside a `vocab` that divides by
the vector width and one that does not, and it did not run here.

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
