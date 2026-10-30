# Model and device bring-up

This document specifies how the tree accepts more than one model and more than
one device. The goal is twofold. A bring-up costs a new directory and no shared
edit. A user on another model or another device sees no change.

Status: **phases P0 to P4 have landed; P5 has not.** What is in the tree, by
section, so a reader can check a claim against a file:

- **§7, the kernel tiers (P0, P1).** `RowReduce` and `SpanReduce` are in the
  seam at version 2, with `PointwiseParams` and the `kExp`, `kRsqrt`, `kLog` and
  `kRowScale` codes; the tier column is in [kernels.md](kernels.md) and the
  required bring-up cost is 16 of the 30 entry points. `src/ops.cc` is the
  composition tier over primitives, and the CUDA fused families are registered
  overrides in `backends/cuda/fused_overrides.cc`.
- **§8, the model description (P2).** `models/nanochat-default/spec.json`,
  `include/nanochat/model_spec.h`, `ModelSpecToConfig`, and the spec selector on
  `Model::Create`.
- **§9, capability negotiation (P3).** The capability mask in
  `nanochat/tensor.h`, `include/nanochat/plan.h` with the pure `BuildPlan`, the
  profile table derived from `backends/*/caps.json` by `src/device_profile.cc`,
  and `src/plan_main.cc`, which `tools/nanochat doctor` runs to print the plan.
- **§10.4 and §11, isolation and the matrix (P4).** `tools/gen_matrix.py`,
  `tools/isolation_gate.py`, the `--model` fixture selector, the gate
  `//tests:matrix_gate_test`, and the generated
  [conformance-matrix.md](conformance-matrix.md).
- **§17.** Every open question is answered by a record in
  [../spec/decisions.md](../spec/decisions.md), one line each, below.

Not landed: a second model, a second `backends/<name>/` directory (§12 and P5),
and the `AGENTS.md` §1 amendment §14.1 describes.

Related material:

- [../DESIGN.md](../DESIGN.md) §2.1 — the dependency invariants that this
  document extends.
- [../spec/decisions.md](../spec/decisions.md) — the decision records this
  document refers to as D-0NN. A decided question lives there and is not
  restated here.
- [../spec/block_kinds.md](../spec/block_kinds.md),
  [../spec/capability_bits.md](../spec/capability_bits.md), and
  [../spec/pointwise_ops.md](../spec/pointwise_ops.md) — the contract vocabulary
  the two axes declare against. `spec/` holds the prose and the data; this file
  holds the design.
- [conformance-matrix.md](conformance-matrix.md) — the generated matrix of §11.
- [kernels.md](kernels.md) — the current kernel seam, its inventory, and its
  tier column.
- [model.md](model.md) — the five graphs that a model spec must describe.
- [testing.md](testing.md) — the test tiers and the oracle harness.
- [simulator.md](simulator.md) — the CPU-only tier that proves a device
  bring-up without silicon.
- [precision.md](precision.md) and [fp8-training.md](fp8-training.md) — the
  precision and fp8 tier.
- [correctness-gates.md](correctness-gates.md) and [parity.md](parity.md) — the
  fixtures and the known-difference register.

## 1. Purpose

The tree must scale along two independent axes:

1. **Model axis.** The architecture that runs. Today this is one class with a
   few `Config` flags.
2. **Device axis.** The hardware that executes it. Today this is
   `backends/<name>/`.

The requirement has two halves, and the second half is the hard one:

- A new model or a new device must need minimum effort.
- A change for one model or one device must not affect any other model or
  device, in source, in build, in test, or at run time.

Both halves follow from two rules in §4. The rest of this document makes those
rules concrete.

## 2. Scope

In scope:

- The directory contract for a model and for a device.
- The kernel tiers: primitives, stateful kernels, composition, and fused
  overrides.
- The model description format and its migration from `Config`.
- The planner, which resolves declared capabilities into one plan.
- The conformance matrix that proves isolation.
- The two bring-up checklists and their definitions of done.

Out of scope:

- The executable container, the driver interface, and the compiler contract.
  Those are later work. This document stays above them.
- Serving, cluster orchestration, and observability.
- A second model implementation. This document defines the slot, not the
  occupant.
- Dynamic shapes. Per-launch selection. The plan is resolved once when it is
  built; nothing branches per launch.

## 3. The two axes and the third party

| Axis | Owner | Owns | Varies by |
|---|---|---|---|
| Model | Model owner | `models/<name>/**` | Architecture, block kinds, dtype policy |
| Device | Backend owner | `backends/<name>/**` | Arithmetic units, memory, synchronization |
| Contract | Architect | `include/nanochat/*.h`, `spec/**`, this file | Version numbers and capability bits |

The contract party exists so the other two never negotiate directly. A model
declares requirements. A device declares support. The contract defines both
vocabularies. Neither side names the other.

## 4. The two rules

**Rule 1 — no cross naming.**
Nothing above the kernel seam may name a specific model or a specific device.
Nothing below the seam may name a graph. The seam itself names neither.

**Rule 2 — a bring-up adds a directory.**
A model bring-up edits only `models/<name>/**`. A device bring-up edits only
`backends/<name>/**`. Neither edits a frozen header, a shared build file, a
shared test file, or another directory.

Rule 2 is the isolation promise in operational form. If a bring-up needs an
edit outside its directory, the contract is incomplete, and the contract owner
extends it once for every future bring-up. That extension is additive (§10.3).

## 5. Why the current tree couples them

| Coupling point | Evidence | Effect on the second model or second device |
|---|---|---|
| `include/nanochat/config.h` | One struct holds the flagship model's flags: `window_pattern = "SSSL"` (line 21), `value_embedding`, `has_value_embedding()`. | The struct becomes the union of every model's quirks. Every model carries every other model's fields. |
| `include/nanochat/kernels.h` | 28 entry points in one frozen, architect-owned header, and 30 after P0. | A new family edits a header that every target compiles, and every backend must then implement it. |
| `src/model.cc`, `src/model_impl.h` | One class owns the topology. `TrainModel` builds the graph as a plain sequence of `ops` calls. | A second model edits shared graph code, so every user of the first model rebuilds and reruns every gate. |
| `src/ops.cc` | A composition layer that calls the fused family symbols directly. | A device that lacks one fused family cannot run the graph at all. There is no fallback. |
| `include/nanochat/device_profile.h` | One hand-written table describes every target device. | A new device adds a row to a shared header, so every device bring-up touches the same file. |

The fourth row costs the most effort, and §7 fixes it. The fifth row breaks
Rule 2 on its own, and §6 fixes it.

## 6. Directory contract

```text
models/<name>/                    the model axis
  spec.json                       the architecture description (§8)
  fixtures/                       numeric fixtures for this model
  tolerances.json                 per-fixture tolerances (§11)
  block_kinds/                    optional; a block kind only this model needs
  BUILD.bazel                     exports the package label

backends/<name>/                  the device axis, unchanged in place
  caps.json                       declared capabilities (§9)
  impl/                           the primitive tier and any fused override
  BUILD.bazel                     exports the package label

spec/                             the contract vocabulary
  block_kinds.md                  the list of block kinds and their semantics
  capability_bits.md              the capability bit names and meanings
  pointwise_ops.md                the op codes and their accuracy bounds
  decisions.md                    the decision records for this document
  versions.md                     the contract version numbers (§10.3)
```

`models/` is new at the top level. `backends/` keeps its name and its
ownership, because `AGENTS.md` §1 already assigns it and every path in the tree
already uses it.

`spec/` holds prose and data, not code. It is the vocabulary both axes read. It
does not replace `docs/`, which stays the architect-owned reference.

**Device declaration has one source.** `caps.json` is the authority. The table
in `include/nanochat/device_profile.h` derives from the `caps.json` files, and
so does the profile selector behind `NANOCHAT_SIM_PROFILE`
([simulator.md](simulator.md)). A device bring-up adds `caps.json` and no row
in a header. Deriving the table is a one-time contract change, owned by the
contract owner with the Harness workstream, which owns
`src/device_profile.cc`. It happens once, before the first new device, not as a
step inside a device checklist.

`block_kinds/` is per model by default. A kind shared by two models moves to a
shared package only as a contract change, because `AGENTS.md` §6 forbids two
model owners editing one file.

## 7. Kernel tiers

Bring-up effort on the device axis is proportional to the number of entry
points a device must implement. Reduce that number.

### 7.1 The four tiers

| Tier | Content | Who supplies it | Required? |
|---|---|---|---|
| Primitives | The sixteen entry points in §7.2 | Each backend | Yes, for inference and evaluation |
| Stateful kernels | `AdamWUpdate`, `MuonUpdate` | Each backend | Yes, for training; declared by a bit |
| Composition | `RmsNorm`, `QkPrep`, `Classifier`, value residual gate, `GradNorm`, `ScalarDot` | One shared library, above the seam | Supplied once, for every device |
| Fused override | A single-kernel replacement for one composition | Optional, per backend, gated by a capability bit | No |

The tiers are separate because the ownership differs. The composition tier
belongs to the Workflow workstream, which owns `src/ops.cc`. The stateful tier
belongs to the Optimizer workstream, which owns `src/optim.cc`. A fused
override belongs to the backend that writes it.

The optimizers are not compositions. `AdamWUpdate` carries per-parameter-group
moment state, and `MuonUpdate` runs batched GEMM plus the Newton–Schulz
iterations. Both may become compositions in a later phase. Do not claim that
today.

### 7.2 The primitive tier

Six device utilities. These already exist as seam utilities.

`Alloc`, `Free`, `Memcpy(void* dst, const void* src, std::size_t bytes,
CopyDir dir)`, `Memset`, `Synchronize`, `GetCaps`.

Eight arithmetic entry points exist in the tier today, and P0 adds two. The set
is three forward and backward pairs, two GEMM entries, and two reduction
entries.

| Primitive | Entry points | Shape of the work | Serves |
|---|---|---|---|
| `Gemm` | `Gemm`, `GemmWgrad` | Dense matrix multiply, forward, dgrad, wgrad. Two entries, not a forward and a backward: `GemmWgrad` is the weight gradient, and the input gradient arrives through `Gemm(kDgrad)`. | Every linear layer, the classifier, Muon |
| `RowReduce` | `RowReduce` | Reduces each of `n_out` equally shaped segments to one scalar. The segment is a base pointer, a stride, and a length, so one call serves a contiguous row and a `head_dim` slice inside a wider row. Combine op and offset. Forward only: a backward is a different reduction, so the caller passes another combine op. | Norm sum of squares, `QkPrep` on `head_dim`, softmax statistics |
| `SpanReduce` | `SpanReduce` | Reduces a ragged list of segments to one scalar. The list lives in device memory, so `GradNorm` can read its spans and counts on the device. Takes a second operand for a dot, a scale, and an accumulate flag. | `GradNorm`, `ScalarDot`, the loss reduction |
| `Pointwise` | `PointwiseForward`, `PointwiseBackward` | One elementwise op code over a parameter block. The code table grows additively, and one code broadcasts a per-row vector. | Scale, scale-add, gate, `relu^2`, softcap, row scale, and the transcendental legs of the compositions |
| `Gather`, `ScatterAdd` | `EmbeddingForward`, `EmbeddingBackward` | Indexed read and indexed accumulate | Token and value embeddings |
| `Attention` | `AttentionForward`, `AttentionBackward` | T × T pairwise plus softmax | The attention graph |

The seam gained `RowReduce` and `SpanReduce` at P0, in seam version 1 → 2
(§10.3). Before them, no entry point reduced one segment to per-segment scalars.
`RmsNorm`, `QkPrep`, and `Classifier` all depend on that shape, so without
`RowReduce` the composition tier cannot express them, and the tier claim in
§7.5 is false.

The two reduction entries are separate on purpose. A per-segment reduce is a
coalesced scan inside one work group. A ragged reduce to one scalar is a
device-wide reduction over a list the device reads itself. They are different
algorithms with different correctness risks, and a device may run them on
different units. One descriptor primitive with three modes would hide which
one a call site uses.

The row broadcast is not a new primitive. `PointwiseForward(op, n, a, b, alpha,
beta, out)` takes two operands of the same length, so `out[r,c] = rstd[r] *
x[r,c]` is not expressible today without materializing a `rows × dim` buffer.
One additive op code, `kRowScale`, closes the gap for one line of arithmetic
inside an entry point the device already supplies. The op-code decisions are in
§7.7.

The tier boundary is **what the device must supply**, not what the device
computes by hand. A primitive that delegates to a library is still a primitive:
`Gemm` calls cuBLAS on CUDA and a reference loop on the CPU, and both are
primitives.

`Attention` stays primitive. Its iteration space is T × T, and no composition
of the other primitives reproduces its tiling without materializing a T × T
tensor.

**The required `Attention` variant is causal, group-query, full context, and
sliding window, at one tile size.** Sliding window is required, not optional,
because the only model in the tree uses it: `Config::window_pattern` defaults to
`"SSSL"`, and `Config::window_left()` computes the short window. A device that
implements full-context attention only cannot run `models/nanochat-default`, so
it is non-conformant, not partially conformant. `docs/kernels.md` family 3
already records the sliding window as part of the family.

Required bring-up cost: **16 of the 30 entry points**. Training adds the two
stateful entries.

`RowReduce` is the highest-risk line in this table, because one reduction must
serve three call sites with different combine ops and offsets. §17 records the
test that decides its descriptor.

### 7.3 Complete partition of the seam

The mapping below covers every entry point in `include/nanochat/kernels.h`. The
published partition — the same rows with the tier column and the bring-up cost
worked out — is [kernels.md](kernels.md) § Kernel tiers, and that is the copy a
backend owner implements. This table is the record of the review that produced
it.

| Entry point | Tier | Gate | Note |
|---|---|---|---|
| `Alloc`, `Free`, `Memcpy`, `Memset`, `Synchronize`, `GetCaps` | primitive | required | Six utilities |
| `Gemm` | primitive | required | cuBLAS on CUDA, reference on CPU |
| `GemmWgrad` | primitive | required | Accumulates into fp32 parameter gradients |
| `RowReduce` | primitive | required | Additive at P0. Per-segment scalars |
| `SpanReduce` | primitive | required | Additive at P0. Ragged list to one scalar |
| `AttentionForward`, `AttentionBackward` | primitive | required | Causal, group-query, full and sliding window |
| `PointwiseForward`, `PointwiseBackward` | primitive | required | The op-code table grows additively |
| `EmbeddingForward` | primitive | required | The `Gather` primitive |
| `EmbeddingBackward` | primitive | required | The `ScatterAdd` primitive |
| `QuantizeFp8`, `GemmScaledFp8` | primitive | optional, bits `fp8_storage` and `fp8_gemm` | The fp8 tier is a declared capability, never assumed |
| `AdamWUpdate`, `MuonUpdate` | stateful | required for training | Owned by the Optimizer workstream |
| `RmsNormForward`, `RmsNormBackward` | composition, plus fused override | | Sum of squares through a row reduction, then pointwise |
| `QkPrepForward`, `QkPrepBackward` | composition, plus fused override | | Row reduction on `head_dim`, rotate, scale |
| `ClassifierForward`, `ClassifierBackward` | composition, plus fused override | | Softcap and cross-entropy through pointwise and a row reduction with a vocabulary offset |
| `ValueGateForward`, `ValueGateBackward` | composition, plus fused override | | A small gate network over pointwise |
| `GradNorm` | composition | | `SpanReduce` across several spans |
| `ScalarDot` | composition | | `SpanReduce` with a second operand |

Total: 30 entry points after P0. Six utilities and 24 arithmetic entries. A
composition supplies its own backward as one unit, because `DESIGN.md` §5
requires a fused backward to be derived and tested as a unit. The rule that
keeps a wrong composition out of the tier is in §7.6.

### 7.4 The fallback chain

For each composition, the planner resolves one implementation when it builds the
plan:

```text
if the device declares the fused override bit and the shapes match:
    use the override
else:
    use the composition
```

Three conditions keep this honest:

1. **The composition defines the semantics.** A fused override is an
   optimization. Both paths must satisfy the same golden trace within the
   committed tolerance.
2. **`src/ops.cc` is the composition tier.** It already claims this role. The
   work is to make every function there expressible over primitives alone, and
   to move each fused family down into the backend as an override.
3. **The resolution happens once.** The planner records the choice. A launch
   reads the plan; it never re-decides.

The consequence for the CPU backend is deliberate. It becomes primitives plus
compositions, and its fused families stop being reachable: `caps.json` declares
no `override_*` bit, so the planner resolves every composition to the
composition path there. The fused symbols that remain in
`backends/cpu/kernels.cc` are a follow-up removal, deleted once `src/` calls the
composition instead of the fused symbol; until then the composition is what the
plan chooses and what the fixtures gate. The trade is acceptable because the
CPU backend is the oracle, and the oracle needs clarity, not speed. The
`tests/data/*.bin` fixtures prove that the values stay inside the committed
tolerance, and `TestRmsNormComposition` compares the two paths in the same test.

### 7.5 Why this reduces effort

| Bring-up | Entry points before P0 (seam 1) | Entry points now (seam 2) |
|---|---|---|
| New device, inference | 28, plus every fused family as it appeared | 16 |
| New device, training | 28 | 18 |
| Each later fused family | A new frozen-header entry point that every device must then implement | One optional override, behind a bit |

A new device reaches a full training step at the primitive and stateful tiers.
Fused overrides then arrive one at a time, each gated by a capability bit, each
measured against the composition it replaces.

### 7.6 When a composition cannot express its backward

A composition becomes the definition of a fused family only if its backward is
writable over primitives. Today the backward saves state inside the kernel:
`QkPrepBackward` recovers the norm statistics from the saved forward `q` and
`k`, and `ClassifierBackward` reads the saved logits. A composition must make
the same decision at the composition level.

The rule:

> When a composition's backward needs an operation the primitive set lacks, add
> the missing primitive. Do not keep a fused family. A new primitive serves
> every future composition. A fused family serves one composition and must be
> reimplemented on every device.

Two conditions bound the rule:

1. **A composition declares its saved slots.** The slots are static and named
   in `docs/model.md` today. A composition backward reads only its declared
   slots, so a fused override and a composition can share a workspace.
2. **`Attention` is the exception, and it is already primitive.** A backward
   that needs a T × T intermediate is not a composition.

Before a fused override may be removed, its composition backward must match the
fused backward's trace at `T1 GPU correctness` shapes, within tolerance.

### 7.7 Op codes and their accuracy

`PointwiseOp` in `include/nanochat/kernels.h` is a closed C++ enum with five
codes. The compositions need four more: `kExp`, `kRsqrt`, `kLog`, and the row
broadcast `kRowScale`.

Decisions:

1. **Keep the closed C++ enum.** One build links the runtime and every backend,
   so a code cannot skew between them. A device that misses a code fails to
   compile, which is the cheapest safety net available. A stable numeric table
   is needed only when a program can arrive from a different build than the
   runtime, and that belongs to the executable format.
2. **Add the four codes in one contract change at P1**, with a seam version bump
   (§10.3).
3. **Replace the scalar argument list with a `PointwiseParams` struct** in the
   same change. The row broadcast needs the row width, and `alpha`/`beta`
   already carry three meanings: `kSoftcap` reads its cap from `alpha`.
4. **Declare an accuracy bound per code** in `spec/pointwise_ops.md`, in units in
   the last place or in absolute terms. `exp`, `tanh`, and `rsqrt` are not
   bit-exact across devices. Without a bound, a device's fast `expf` shifts the
   loss and the tolerance gate blames the wrong layer. The fp8 rounding rule in
   `docs/correctness-gates.md` §3.2 is the precedent: share the rule, not the
   implementation.

## 8. Model description

`Config` is the wrong place for model identity, because it is one struct. It
stays the right place for derived dimensions, because `head_dim()`,
`mlp_dim()`, `window_left()`, and `Validate()` are used everywhere.

The split:

- `ModelSpec` is the description. It is data. A model package ships one.
- `Config` becomes the derived runtime struct. The runtime computes it from a
  `ModelSpec` at load.

`ModelSpec` fields:

| Group | Fields |
|---|---|
| Identity | `spec_version`, `name`, `requires` (capability bits) |
| Dimensions | layers, hidden, heads, kv heads, vocabulary, padded vocabulary, sequence length |
| Blocks | An ordered list. Each entry names a norm kind, an attention kind, an attention variant (full or sliding), a position kind, an MLP kind, a residual kind, and an optional per-block dtype policy |
| Embeddings | Learned or tied. Value embeddings present or absent, and on which layers |
| Head | Softcap constant, weight tying, vocabulary padding rule |
| Loss | Cross-entropy, row weighting, masking rule |
| Policy | Default dtype per region, and the numeric contract version |

The field list is closed. It grows only by a version bump in
`spec/versions.md`, and a new field needs a stated meaning that the existing
blocks cannot express. This is the guard against drift toward the graph
Intermediate Representation that `DESIGN.md` §6 rejects.

The block list is the topological part. A block kind is a named meaning, not a
code path: `spec/block_kinds.md` defines what each kind computes, and the model
package that needs the kind holds the implementation.

Migration path, in order:

1. Write `models/nanochat-default/spec.json` so that it reproduces today's
   defaults exactly.
2. Add the derivation `ModelSpec -> Config`, and one test that compares it with
   the current `Config` defaults. Nothing else changes.
3. Replace `window_pattern` and `has_value_embedding()` with reads of the
   block list. Keep the derived accessors on `Config`.
4. Move the specialized graph builder to `models/nanochat-default/` only when a
   second model needs the generic builder.

Step 2 is the whole migration cost for the first model. It changes no numerics.

### 8.1 One definition of semantics

The flagship model may keep a hand-written fast path. It must match the generic
path's fixture. State this as a rule and gate it:

> One definition of semantics. N implementations. One golden trace.

Without this rule, model owners silently diverge, and a shared fixture becomes
ambiguous. With it, a fast path is always an optimization of a known reference.

### 8.2 The factory needs a selector

`Model::Create(const Config&, std::string*)` in `include/nanochat/model.h` takes
no selector, so a registry needs one additive argument. That edits a frozen
header, and this document requires it. Sequence it as one contract change at
P2:

```cpp
static std::unique_ptr<Model> Create(const Config& config,
                                     const char* spec_name = nullptr,
                                     std::string* error = nullptr);
```

A null name selects the built-in default. The overload keeps every existing
call site unchanged.

## 9. Capability negotiation

Both sides declare. The planner resolves. Nothing else decides.

```text
model requires:  gqa, rope, sliding_window, value_gate, bf16
device offers:   gqa, rope, sliding_window, value_gate, fp32
resolution:      rejected — missing bf16
                 consequence: the model declares no fp32 fallback
```

Rules:

1. **Capability bits, not booleans.** `Caps` in `tensor.h` gains one 64-bit
   feature mask, additive. Booleans do not compose, and each new boolean
   touches a shared header. `Caps` already carries `supports_fp32`,
   `supports_bf16`, `supports_fp8`, and `Supports(DType)`; the mask names those
   as bits and adds the rest.
2. **Precision is declared like anything else.** A device that offers no fp8
   path declares no `fp8_storage` and no `fp8_gemm` bit. The model then either
   runs the dense path or fails with a named gap.
3. **Bits are named in `spec/capability_bits.md`.** One table holds the name,
   the meaning, the numeric obligation, and the owning layer. The per-device
   declaration file is `backends/<name>/caps.json` (§6). A numeric obligation
   that belongs to an op code lives in `spec/pointwise_ops.md` (§7.7).
4. **Resolution happens once, when the planner builds the plan.** The result is
   a plan: the chosen implementation for each composition, plus the fallbacks
   it declined.
5. **A rejection names the gap.** The message lists the missing bits and the
   consequence. It never reports "unsupported device".
6. **The plan is observable.** The planner logs it, and `tools/nanochat doctor`
   prints it. A user proves why their path changed without reading source.
7. **The default device is explicit.** No device becomes the default as a side
   effect of registration.

### 9.1 The planner interface

The plan has one dimension more than a model: `(model, device, graph)`. Decode
and training want different choices on one device, and a mesh run wants one plan
per rank. So the planner is not a member of the model.

```cpp
// include/nanochat/plan.h (new, additive)
struct PlanRequest {
  const ModelSpec* model;  // requirements
  GraphKind graph;         // train, prefill, decode, eval, generate
  int device_index;
};

// Pure. Calls GetCaps() and nothing else on the device.
Plan BuildPlan(const PlanRequest& request, std::string* why_not);
```

Four rules make it worth its own file:

1. **Pure.** A test builds a plan for a fabricated capability mask, so a device
   can be tested under `T0 CPU` before it exists.
2. **One caller at first.** `Model::Create` calls it and stores the result. No
   entry point changes at P3.
3. **`tools/nanochat doctor` calls it directly** with a probe spec, so a user
   can see the plan before the model loads.
4. **The planner never mutates a device.** It answers a question; something else
   acts on the answer. This is what keeps the resolution out of the hot path.

## 10. Isolation

Isolation has four parts. Each needs a gate, because each is otherwise a claim.

### 10.1 Source isolation

A model package exports exactly one symbol: its factory (§8.2). The runtime
never includes a model header. A backend never includes a model header. The
build graph enforces both with visibility rules, the same mechanism
`DESIGN.md` §2.1 already uses.

### 10.2 Build and test isolation

Each model package and each backend package has its own target and its own
tests. A diff inside one package rebuilds and retests only that package's
targets, plus the conformance matrix row for that package. Slow or flagship
suites carry `manual`, as the GPU suites do today.

### 10.3 Contract versioning

Four independent version numbers, one per contract:

| Contract | Version lives in | Bumped when |
|---|---|---|
| Spec format | `spec/versions.md` and `spec.json` | A field appears or a meaning changes |
| Capability vocabulary | `spec/capability_bits.md` | A bit appears or a bit is retired |
| Kernel seam | `include/nanochat/kernels.h` header comment | An entry point appears, changes, or retires |
| Fixture format | `tests/oracle_fixture.h` | The record layout changes |

Rules: additions are free, removals need one deprecation release, and a
meaning change needs a new name. A version number rises only with a change that
a reader can detect. The `NCHKPT01` policy in
[checkpoint-format.md](checkpoint-format.md) is the model to copy: writers write
the current version and readers accept the current and the previous one.

### 10.4 Cross-package non-interference

The gate is a regenerate-and-diff check. For a diff that touches
`models/<name>/**`:

1. Regenerate that model's fixtures.
2. Diff them against the committed fixtures within tolerance.
3. Confirm every other model's fixtures are unchanged.

Step 3 is the isolation proof. It costs nothing, because both the fixture
comparison and the reference trace run on the CPU.

A contract extension is the one legitimate exception. When a change moves every
model (§10.3), the commit declares the affected rows in the matrix and states
the reason. The contract owner reviews it and records it in `spec/versions.md`.
Without the exception the gate either blocks every contract change or gets
ignored.

## 11. The conformance matrix

The matrix is the one place where a model and a device meet.

| Axis | Values | Source |
|---|---|---|
| Model | Each directory under `models/` | Enumerated by a generator |
| Tier | `T0 CPU`, `S0 simulator`, `T1 GPU correctness`, `T2 GPU parity` | [testing.md](testing.md) |
| Graph | train step, prefill, decode, eval, generate | [model.md](model.md) |

Tier names are the ones the tree already uses: `T0 CPU`, `S0 simulator`,
`T1 GPU correctness`, `T2 GPU parity` in [testing.md](testing.md), and the
profiles `t0-cpu`, `t1-gpu`, `t2-parity` in `tools/sandbox/profiles.json`.

Tier choice per device, so the matrix does not explode:

- `T0 CPU` for every model. Always. This is the definition.
- `S0 simulator` for a device profile, when the device has no physical sample.
- `T1 GPU correctness` for correctness on real silicon, tiny shapes.
- `T2 GPU parity` only for the flagship model, at wave boundaries.

Graph coverage per row. A row states which of the five graphs it proves. The
minimum for a passing row is the train step and decode. A complete row covers
all five, and the flagging row names any it skips. Without this column, a
model that runs its train step and breaks in prefill looks green.

Fixtures live under `models/<name>/fixtures/`. Tolerances live beside them in
`tolerances.json`, so a model owns its own numeric contract. A tolerance change
is a contract-owner review, not a model-owner decision, because a loosened
tolerance turns the isolation gate into self-certification.

The matrix is generated, not maintained. The generator is owned by the Oracle
workstream, because it produces the data that the gates compare. The committed
rendering is [conformance-matrix.md](conformance-matrix.md): it is reproduced
with `python3 tools/gen_matrix.py --out docs/conformance-matrix.md` and
validated with `python3 tools/gen_matrix.py --check --matrix
docs/conformance-matrix.md`, so a reviewer sees the matrix in the file and sees
it move in the diff.

## 12. Bring-up checklists

### 12.1 A new device

1. Write `backends/<name>/caps.json`. Declare the primitive tier complete and
   name the precision bits this device supports.
2. Implement the six utilities and the ten arithmetic primitives, including
   the sliding-window attention variant and the two reductions.
3. Reach `tools/nanochat simulate --suite correctness` with the CPU reference
   values, and `--suite kernel` with the emulated values.
4. Reach `T1 GPU correctness` on one tiny shape per primitive.
5. Add the two stateful entries if the device claims training.
6. Add the fp8 pair only if `caps.json` declares the fp8 bits.
7. Add fused overrides one at a time. Each needs a benchmark and the same
   trace as the composition it replaces, within the committed tolerance, as
   `DESIGN.md` §3.1 requires for `QkPrep`.
8. Add the matrix rows for this device, with their graph coverage. Do not touch
   another backend's directory.

Definition of done: every model in `models/` runs the five graphs, or the row
names the graphs it skips, on the composition path. Each fused override
improves a measured number.

### 12.2 A new model

1. Write `models/<name>/spec.json`.
2. Confirm that the required block kinds exist in `spec/block_kinds.md`. A
   missing kind is a contract addition, not a private implementation.
3. Generate fixtures through the fixture generator. The generator is
   Oracle-owned, so it gains a `--model` selector once, at P4, and a model owner
   then adds no code to it.
4. Record tolerances in `tolerances.json`.
5. Run the matrix rows for `T0 CPU` and one device tier.
6. Add a fast path only after the composition path agrees.

Definition of done: the new model passes its own matrix rows, and every other
model's fixtures are unchanged.

## 13. What a bring-up may not change

Reject a change under review if it does any of these, without a contract
extension first:

- Edits `include/nanochat/config.h`, `kernels.h`, `tensor.h`, or
  `device_profile.h` for one model or one device.
- Edits `src/model.cc`, `src/model_impl.h`, `src/ops.cc`, `src/optim.cc`, or
  `src/generate.cc` to add a model-specific path.
- Edits another model's or another backend's directory.
- Adds a field to `Config` that only one model reads.
- Adds a capability as a boolean instead of a named bit.
- Puts a model name or a device name in a preprocessor condition.

## 14. Ownership

| Party | Owns | Reviews |
|---|---|---|
| Model owners | `models/<name>/**` | Their own matrix rows |
| Backend owners | `backends/<name>/**` | Their own matrix rows |
| Contract owner | `include/nanochat/*.h`, `spec/**`, this file | Every contract extension, every tolerance change |
| Oracle workstream | `tests/`, `tests/data/*.bin`, the fixture generator, the matrix generator | Every fixture change |
| Workflow workstream | `src/ops.cc`, `src/model.cc`, `src/generate.cc` | The composition tier and its fallbacks |
| Optimizer workstream | `src/optim.cc` | The stateful kernel tier |
| Harness workstream | `src/device_profile.cc` | The `caps.json` derivation into the profile table |

The contract owner answers one question and only one question: does this
extension serve many future bring-ups? If yes, it lands once, additively.

### 14.1 Amendment to `AGENTS.md` §1

This document adds two directories and two roles to the ownership map. The
amendment lands with P0, in one edit to `AGENTS.md`, so the map never disagrees
with the tree:

| Row to add | Owns |
|---|---|
| **Model owners** (one per model) | `models/<name>/**` |
| **Contract** (the architect) | `spec/**` |
| **Oracle** (unchanged) | `tests/`, the fixture generator, the matrix generator |

`src/device_profile.cc` stays with the Harness workstream.

Status of the amendment: the three rows are **not** in `AGENTS.md` §1 yet. The
edit is the contract owner's, it is still owed, and until it lands the ownership
map and this document disagree about `models/` and `spec/`. Nothing else in this
document depends on the edit: the directories exist and are owned by the
workstreams named above.

## 15. Anti-patterns

- A preprocessor condition on a model name or a device name.
- One operation enumeration with one large switch in a shared file. Each new
  model or device edits it, so every user is affected.
- "Just add a field to `Config`."
- A device that implements fused families before it passes the primitive tier.
- A device that implements full-context attention only and calls itself
  conformant.
- A model-specific tuning constant inside a backend.
- A fixture keyed only by shape, when two models share the shape.
- A matrix row added by hand.
- A device that becomes the default through registration.
- A model owner loosens a tolerance to pass a gate.

## 16. Phases

| Phase | Work | Owner | Gate |
|---|---|---|---|
| P0 | Classify all 30 entry points into a tier. Add `RowReduce` and `SpanReduce` with a seam version bump. Publish the tier column in [kernels.md](kernels.md). Start `spec/decisions.md`. Amend `AGENTS.md` §1. | Contract | Review only |
| P1 | Make every composition in `src/ops.cc` expressible over primitives. Add `kExp`, `kRsqrt`, `kLog`, `kRowScale` and the `PointwiseParams` block. Write `spec/pointwise_ops.md`. Move fused families into the CUDA backend as overrides. | Workflow, Backend | The parity tests pass at the committed tolerance |
| P2 | Write `models/nanochat-default/spec.json`, add the selector on `Model::Create`, add `ModelSpec -> Config`, prove the defaults match. | Contract, Workflow | One equality test |
| P3 | Derive `device_profile.h` from `caps.json`. Add `plan.h` and the pure `BuildPlan`, and the capability mask. Log the plan, print it in `doctor`. | Contract, Harness, Backend | A rejection test names its gap, and a fabricated capability mask builds a plan under `T0 CPU` |
| P4 | Add the matrix generator, the `--model` fixture selector, and the regenerate-and-diff gate. | Oracle | Two models, two tiers |
| P5 | Add the second model and the second device on their own. | Model, Backend | P4 stays green |

P1 pays back before any second model or second device exists, because the CPU
backend becomes a pure composition oracle. P2 costs one equality test. P5 is
the proof that P0 to P4 worked, and it should feel trivial.

P0 to P4 are landed, so the table above is a record rather than a queue; P5 is
the open phase, and it is the one that adds a directory instead of editing one.
The status paragraph at the top of this file maps each landed phase to the files
it produced.

## 17. Risks and open questions

| Risk | Effect | Response |
|---|---|---|
| The composition path is slower than a fused family on the CPU backend. | The oracle slows down. | Accepted. The oracle trades speed for clarity. Keep one benchmark to track it. |
| `ModelSpec` grows into a general graph language. | The tree drifts toward the graph Intermediate Representation that `DESIGN.md` §6 rejects. | The field list and the block kind list are closed (§8). |
| The matrix grows as models × tiers × graphs. | Slow gates. | Tiers per device (§11), `manual` for flagship rows. |
| Two models share a block kind and then diverge. | A shared fixture becomes ambiguous. | A block kind has one meaning. A divergence is a new kind. |
| The primitive tier turns out to be under-specified. | A device needs an extra primitive, so `kernels.h` changes. | Expected. The tier is a target, reviewed at P0. The change is additive and lands once. |
| A tolerance drifts upward until the gate proves nothing. | The isolation gate becomes decoration. | Every tolerance change is a contract-owner review (§11). |

One risk in the table above has already materialized, and it is the one to read
twice: the P0 comparison found that `RowReduce`'s descriptor lacks two fields a
composition wants — a second operand (or a dot combine op) for the per-row
`sum_c x * dy`, and a per-segment input pitch for a padded vocabulary row. That
is an additive seam change, it is reported to the contract owner, and it is not
patched from a backend — which is the rule this document exists to keep.

Open questions. Each one was posed with the phase that answers it. All four are
now answered, and the answer is a record in
[../spec/decisions.md](../spec/decisions.md) that holds the options, the
evidence, and what would reopen it. One line here; the record is the authority.

1. **P0: the descriptor of `RowReduce`.** The declared fields carry the forward
   cases, and the experiment named the two fields the descriptor lacks — a
   second operand (or a dot combine op) for the per-row `sum_c x * dy`, and a
   per-segment input pitch for the `Classifier`'s padded vocabulary row — which
   are reported to the contract owner rather than patched into the header. The
   experiment, its table, and the workaround it measures are in
   `backends/cpu/kernels_test.cc` (`TestRmsNormComposition`) and
   `backends/cpu/README.md`. [D-006](../spec/decisions.md)
2. **P1: the accuracy bound per call to `kExp` and `kRsqrt`.** An absolute bound
   per code over a stated operand envelope, tabulated and gated; a bounds table
   with no envelope is not a bound. [D-007](../spec/decisions.md), and the table
   is [../spec/pointwise_ops.md](../spec/pointwise_ops.md) §4.
3. **P2: does a composition whose backward needs a T x T pattern become a second
   primitive?** Yes, by rule, and no second primitive is added today: `Attention`
   is the only case and it is already primitive. [D-008](../spec/decisions.md)
4. **P2: where a block kind lives.** The composition tier supplies every kind by
   default; `models/<name>/block_kinds/` exists only for a kind with no shared
   implementation, and sharing a kind across models is a contract change.
   [D-009](../spec/decisions.md)

### 17.1 Decision records

A new device team reads `spec/decisions.md` before it starts. Each record holds
the question, the options, the evidence that decided it, and the phase it lands
in. Five decisions are decided and four carry a binding working answer, and the
records exist so that a review does not re-litigate them:

| Decision | Answer | Section |
|---|---|---|
| The shape of the reduction | Two entries: `RowReduce` and `SpanReduce` | §7.2 |
| The row broadcast | One `Pointwise` op code, `kRowScale`, not a new entry point | §7.2, §7.7 |
| Op codes and accuracy | Closed C++ enum, one additive change at P1, with a declared bound per code | §7.7 |
| A composition that cannot express its backward | Add the missing primitive; do not keep a fused family | §7.6 |
| Where the plan is resolved | A pure planner, `BuildPlan`, with `Model::Create` as its first caller | §9.1 |

Order the remaining work by dependency. The `RowReduce` descriptor test comes
first, because §7.6 and §7.7 depend on the primitive set it produces. The
planner comes last, because it consumes the interfaces of the other three.

## Appendix A. The bring-up contract in one page

A model brings:

- `models/<name>/spec.json`
- `models/<name>/fixtures/*.bin`
- `models/<name>/tolerances.json`
- `models/<name>/BUILD.bazel`

A device brings:

- `backends/<name>/caps.json` with its precision bits
- the six utilities and the ten arithmetic primitives, sliding window
  included
- the two stateful entries, if it claims training
- `backends/<name>/BUILD.bazel`

Neither edits a frozen header, a shared source file, another model's directory,
or another backend's directory. If a bring-up needs such an edit, the contract
is incomplete. The contract owner extends it once, additively, and every later
bring-up stays inside its own directory.
