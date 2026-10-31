# Decision records

One record per decision. A record holds the question, the options, the answer,
the evidence that decided it, and the phase it lands in. The purpose is narrow
and worth stating plainly: a settled question is not re-litigated in review, and
a new device team reads nine records instead of reconstructing an argument from
the design document.

The records below belong to
[docs/model-and-device-bringup.md](../docs/model-and-device-bringup.md) §17 and
§17.1. They are the vocabulary's memory: every version rise in
[versions.md](versions.md) cites one, and every meaning in
[capability_bits.md](capability_bits.md), [block_kinds.md](block_kinds.md), and
[pointwise_ops.md](pointwise_ops.md) is downstream of one.

## 1. How to use this file

- **Read it before a bring-up.** A device bring-up that reopens D-001 has
  already been answered in §7.2; the review cites the record.
- **A decided record is closed.** D-001 to D-005 carry **Decided**. They are not
  reopened by preference, by a new team's taste, or by a benchmark. They are
  reopened only by new evidence of the kind named in the record, and only by the
  contract owner.
- **An open record carries a working answer.** D-006 to D-009 carry **Open** and
  a working answer that is binding until a later record supersedes it. The
  working answer is what the tree builds today; the record says what evidence
  will settle it and at which phase.
- **A record is never edited to say something else.** A superseded record keeps
  its text and gains one line: `Superseded by D-0NN`. A record is never
  renumbered, because the version changelog and the review history cite the
  numbers.
- **A new record takes the next free number** and names its phase. A decision
  with no record is a decision that was not reviewed.

## 2. The nine records at a glance

Dependency order. D-006 comes first because the primitive set it produces is
what D-007, D-008, and every composition depend on; D-005 comes last among the
decided ones because the planner consumes the interfaces the others define.

| Id | Question | Status | Phase | Lands as |
|---|---|---|---|---|
| D-001 | One reduction entry point or two? | Decided | P0 | `RowReduce` + `SpanReduce`, seam 1 |
| D-002 | How is a per-row scalar broadcast expressed? | Decided | P0 declared, P1 used | `kRowScale`, an op code |
| D-003 | How do op codes and their accuracy work? | Decided | P0 declared, P1 used | Closed enum, one additive change, a bound per code |
| D-004 | What if a composition cannot express its backward? | Decided | standing from P0 | Add the missing primitive |
| D-005 | Where is the model/device resolution made? | Decided | P3 | A pure `BuildPlan`, `Model::Create` first caller |
| D-006 | Is one `RowReduce` descriptor enough? | Open | P0 | The declared descriptor, pending the CPU comparison |
| D-007 | Absolute bounds or units in the last place? | Open | P1 | Absolute per code, table in [pointwise_ops.md](pointwise_ops.md) |
| D-008 | Does a `T x T` backward need a second primitive? | Open | P2 | No second primitive; a primitive if a real case appears |
| D-009 | Where does a block kind live? | Open | P2 | The model package by default |

---

## D-001 — The shape of the reduction

**Status:** Decided. **Phase:** P0. **Bumps:** kernel seam 0 → 1.

**Question.** The primitive tier has no reduction that produces per-segment
scalars, and three compositions need one. Does the tier gain one reduction
entry point with modes, or two entry points?

**Options.**

1. One entry point with a mode field covering segmented, ragged, and
   span-reduce.
2. Two entry points: `RowReduce` (each of `n_out` equally shaped segments to one
   scalar, with a combine op and an offset) and `SpanReduce` (a ragged list of
   segments in device memory to one scalar, with a second operand, a scale, and
   an accumulate flag).
3. Keep the fused families and let the composition tier call them.

**Answer.** Option 2.

**Evidence.**

- Three call sites need the segmented shape and only that shape: the norm's sum
  of squares over a row, `QkPrep`'s reduction over a `head_dim` slice inside a
  wider row, and the `Classifier`'s row reduction with a vocabulary offset. A
  single call must serve all three, so the descriptor must carry the combine op
  and the offset.
- Two call sites need the ragged shape and only that shape: `GradNorm` over
  several gradient spans, and `ScalarDot` over two buffers. Their list lives in
  device memory, because a host-side list is a round trip per step.
- The two are different algorithms with different correctness risks and
  potentially different units: a coalesced scan inside one work group, and a
  device-wide reduction over a list the device reads itself. Design
  `docs/model-and-device-bringup.md` §7.2.
- A mode field would hide which algorithm a call site uses, which is the one
  thing a reviewer of a numeric change needs to see.
- Option 3 is the state the tree is in and it is what the tier exists to end: a
  device without the fused family cannot run the graph at all (§7.4).

**Consequences.** `RowReduce` is the highest-risk line in the tier, because one
descriptor serves three call sites with different combine ops and offsets; D-006
is the test that decides whether the descriptor's fields suffice.
`SpanReduce` is the lower-risk line and no test decides it because nothing in the
tree is ambiguous about a ragged list.

---

## D-002 — The row broadcast

**Status:** Decided. **Phase:** declared at P0 with the seam, used at P1.
**Bumps:** kernel seam 0 → 1.

**Question.** `RmsNorm` needs `out[r, c] = rstd[r] * x[r, c]` and `ValueGate`
needs `gate[r] * ve[r, c]` over the head width. `PointwiseForward` takes two
operands of the same length, so neither is expressible. New entry point, new
op code, or a materialized operand?

**Options.**

1. A new entry point, `RowScaleForward`/`RowScaleBackward`.
2. One additive op code, `kRowScale`, inside the pointwise family, with the row
   width in the parameter block.
3. Materialize a `rows x width` operand on the host or in the workspace and use
   `kScaleAdd`.

**Answer.** Option 2.

**Evidence.**

- Option 3 costs a buffer and a pass that grow with the sequence length, for one
  line of arithmetic. §7.2 records the reason the broadcast exists at all.
- Option 1 raises the required bring-up cost: every entry point is a thing every
  device implements forever, and §7.5 counts the tier in entry points. The
  broadcast has no different access pattern, no different units, and no
  different correctness argument, which is exactly the test §7.2 uses to decide
  that the two *reductions* are separate entries.
- Option 2 keeps the closed enum as the safety net: a backend that misses the
  code fails to compile, which is cheaper than a link error and much cheaper than
  a run-time fallback.
- The backward is a per-row reduction (`da[r] = sum_c dy[r,c] * b[r,c]`), so it
  reuses the primitive D-001 added instead of growing the tier again.

**Consequences.** `row_width` joins the parameter block, which is why the block
replaces the scalar argument list (D-003). The two call sites and the full
statement of the width's meaning are in
[pointwise_ops.md](pointwise_ops.md) §3.1.

---

## D-003 — Op codes and their accuracy

**Status:** Decided. **Phase:** declared at P0 with the seam, bounds used at
P1. **Bumps:** kernel seam 0 → 1.

**Question.** The compositions need four more codes (`kExp`, `kRsqrt`, `kLog`,
`kRowScale`). Should the enum become an open, stably numbered table? And what is
a device's numeric obligation for a code?

**Options.**

1. An open enum plus a stable numeric table and a run-time error on a code the
   build does not know.
2. A closed C++ enum, one additive change, and a declared accuracy bound per
   code.
3. Separate entry points per arithmetic operation, with no codes at all.

**Answer.** Option 2. The closed enum is
[pointwise_ops.md](pointwise_ops.md) §1; the bounds are its §4.

**Evidence.**

- One build links the runtime and every backend, so a code cannot skew between
  them, and a missing code can be made to fail to compile. That is the cheapest
  safety net available (§7.7 decision 1).
- A stable numeric table is needed only when a program carrying a code can
  arrive from a different build than the runtime. That is the executable
  container's problem and it is out of scope (§2).
- `exp`, `tanh`, and `rsqrt` are not bit-exact across devices, so without a
  declared bound a device's fast `expf` shifts the loss and the tolerance gate
  blames the wrong layer (§7.7 decision 4). `docs/correctness-gates.md` §3.2 is
  the precedent: share the rule, not the implementation.
- `alpha` and `beta` already carry three meanings across five codes, and the row
  broadcast needs a width, so the scalar argument list has to become a block
  (§7.7 decision 3).
- Option 3 is what the tier exists to avoid: it multiplies the entry-point count
  by the number of arithmetic legs, and every device would implement every leg.

**Consequences.** The bound's *form* is left open and is D-007. The codes' names
and the numeric values' non-contractual status are in
[pointwise_ops.md](pointwise_ops.md).

---

## D-004 — A composition that cannot express its backward

**Status:** Decided. **Phase:** standing rule from P0; applied whenever a
composition hits a gap.

**Question.** A composition's backward sometimes needs an operation the
primitive set lacks. `QkPrepBackward` today recovers the norm statistics from the
saved forward `q` and `k`; `ClassifierBackward` reads the saved logits. Keep the
fused family, stage on the host, or extend the primitive set?

**Options.**

1. Keep the fused family when the composition's backward is not writable over
   primitives.
2. Add the missing primitive.
3. Stage the missing step through the host, so the composition is expressible
   without a new entry point.

**Answer.** Option 2. The rule, verbatim from §7.6:

> When a composition's backward needs an operation the primitive set lacks, add
> the missing primitive. Do not keep a fused family. A new primitive serves every
> future composition. A fused family serves one composition and must be
> reimplemented on every device.

**Evidence.**

- Option 1 is the coupling §5 exists to break: a device that lacks the one fused
  family cannot run the graph at all, and every future device pays for it.
- Option 3 is the current tree's habit and it is why the CPU backend is slow
  today. A host round trip is a synchronisation point per step, and it makes the
  composition's numeric behaviour depend on the host's libm, which is the
  opposite of a declared bound.
- Two conditions bound the rule and both are properties of this directory: a
  composition declares its saved slots ([block_kinds.md](block_kinds.md) §3), so
  an override and a composition can share one workspace; and `Attention` is the
  documented exception, because it is already primitive.

**Consequences.** A composition that hits a gap reports the missing primitive
instead of reintroducing a fused family. The report goes to the contract owner,
because a new primitive is a seam change. Before a fused override is removed,
its composition's backward must match the fused backward's trace at
`T1 GPU correctness` shapes, within tolerance (§7.6).

---

## D-005 — Where the model/device resolution is made

**Status:** Decided. **Phase:** P3.

**Question.** A model declares requirements and a device declares support. Where
do the two meet, and when?

**Options.**

1. Per launch, inside the kernel dispatch: pick the implementation each time a
   kernel is called.
2. Once, at plan time: a pure `BuildPlan` resolves each composition to an
   override or to the composition, records the declined fallbacks, and
   `Model::Create` stores the result.
3. At build time: a compile-time macro selects the implementation per device.

**Answer.** Option 2.

**Evidence.** §9.1's four rules, each of which excludes one of the others:

- **Pure.** A plan is built for a fabricated capability mask, so a device can be
  tested under `T0 CPU` before the silicon exists. Option 1 cannot be tested that
  way, because its decision is distributed through the launch path.
- **One caller at first.** `Model::Create` calls it and stores the plan; no entry
  point changes. Nothing else re-decides at launch (option 1's failure mode).
- **Observable.** `tools/nanochat doctor` prints the plan from a probe spec, so a
  user proves why their path changed without reading source (§9 rule 6).
- **Never mutates the device.** The planner answers a question; something else
  acts on it.
- The plan has one dimension more than a model — `(model, device, graph)` —
  because decode and training want different choices on one device and a mesh
  run wants one plan per rank, so the planner cannot be a member of the model.
- Option 3 cannot express "this device, this graph" and it puts a device name in
  a preprocessor condition, which Rule 1 forbids.

**Consequences.** `include/nanochat/plan.h` and a pure `BuildPlan`; the rejection
message names the missing bits and the consequence and never says "unsupported
device"; a backend registers a fused family against a composition name and a
capability bit, never against a graph.

---

## D-006 — Is one `RowReduce` descriptor enough?

**Status:** **Open.** **Phase:** P0. **Would bump:** nothing if the answer is
"the declared fields suffice"; a second seam change if it is not.

**Question.** D-001 fixed two entries. The remaining question is the per-segment
descriptor itself: is one call carrying a combine op and an offset enough for all
three call sites, or does a call site need a field the descriptor does not have —
a per-segment output stride, a second operand, a mask, or a third entry point?

**Options.**

1. Combine op plus offset only, as `include/nanochat/kernels.h` declares.
2. Add a per-segment output stride, so a call can write every k-th scalar.
3. Add a second operand, so one call can do a dot instead of a sum.
4. A third entry point for the segmented case, on the grounds that three call
   sites with three combine ops are three meanings.

**Working answer.** Option 1. It is what the header declares and what the
compositions are written against today.

**Evidence that will settle it.** The CPU backend's `RmsNorm` composition
comparison, at `B=2, T=8`:

1. Write `RmsNormForward` and `RmsNormBackward` as compositions over `RowReduce`
   and `Pointwise` alone, with no call to the fused family.
2. Compare them with the current fused path at the committed fp32 tolerance
   (`tests/numerics_trace.h`: `kLossRtol = 1e-5`, `kParamL2Rtol = 1e-4`).
3. Record which descriptor fields the composition needed, which it never
   touched, and which it worked around.

The diff answers the question: a field the composition needs and does not have
is a field to add; a field it never touches is a field to consider dropping; a
workaround is a missing field wearing a disguise. Two further call sites are part
of the evidence and not a separate record: the `Classifier` row reduction with a
vocabulary offset (the offset case) and the `kRowScale` backward (the dot case,
[pointwise_ops.md](pointwise_ops.md) §3.1).

**What happens with the answer.** "The declared fields suffice" closes the
record with no header change. "A field is missing, or is superfluous" is a
follow-up report to the contract owner, who owns `include/nanochat/kernels.h`:
the CPU node that runs the experiment does not edit the header, and a silent
second edit is exactly what §7.2 forbids. The edit lands as one additive seam
change with its own version rise in [versions.md](versions.md).

**Why this record is first.** The primitive set that D-006 produces is what
D-002, D-003, and every composition in `src/ops.cc` are written over, so
settling it first is what keeps the other records from moving.

---

## D-007 — Absolute bounds, or units in the last place?

**Status:** **Open.** **Phase:** P1. **Would bump:** the kernel seam when a
bound moves (a tightening is free, a loosening needs the §11 review).

**Question.** D-003 decided that every op code carries a declared accuracy
bound. In what form: units in the last place, or an absolute bound per code?

**Options.**

1. A per-code bound in units in the last place, relative to the reference.
2. A per-code absolute bound, asserted over a stated operand envelope.
3. No per-code bound: one graph-level tolerance for the whole model.

**Working answer.** Option 2, implemented by the table in
[pointwise_ops.md](pointwise_ops.md) §4 with one bound and one envelope per
code.

**Evidence that will settle it.** The loss-curve comparison at the tolerance the
tree already commits, `kLossRtol = 1e-5` in `tests/numerics_trace.h`: substitute
one device code's implementation for the reference one at a time and require the
trace's `train_loss` and `train_grad_norm` records to stay inside per step.

Three arguments already point the same way and are recorded here so the
experiment is a confirmation and not a survey:

- A per-code bound has to *compose* into a graph bound. An absolute bound adds
  up with one multiplication per layer; a relative bound does not compose without
  knowing every intermediate magnitude, which is what a reviewer cannot get from
  a table.
- `kLog` decides it on its own: near one, `log(1 + e)` has an unbounded relative
  error and a tiny absolute one. A last-place bound would either pass a
  catastrophically wrong logarithm or fail a correct one.
- A last-place bound is a statement about the reference's rounding and changes
  meaning when the reference changes; an absolute bound is a statement about the
  value.

**Cost of the answer, stated so it is not a surprise.** Every bound needs an
operand envelope, and an envelope is a claim about the graph. The claim is
checked: the trace asserts every operand stays inside its envelope and fails with
the record name when it does not.

**What happens with the answer.** If the comparison holds, the record closes with
the table as written. If one substitution leaves `kLossRtol`, the answer is not
"loosen the tolerance" (§11: a loosened tolerance turns the gate into
self-certification) — it is either a tighter device implementation or a bound
that says what is true, recorded here first.

---

## D-008 — A composition whose backward needs a `T x T` pattern

**Status:** **Open.** **Phase:** P2.

**Question.** §7.6 condition 2 exempts `Attention`, because its backward needs a
`T x T` intermediate and it is already primitive. If a second composition turns
up whose backward needs the same pattern, does it become a second primitive?

**Options.**

1. Yes: a `T x T` backward makes the operation a primitive entry point, and the
   composed form is not offered.
2. No: the composition allocates a `T x T` workspace slot and stays a
   composition.
3. Keep the fused family for it, as an override with no composition behind it.

**Working answer.** Option 1, with the rule stated as a rule and not as a list:

> A composition whose backward needs a `T x T` intermediate is not a composition.
> It is a primitive, because a `T x T` intermediate is a slot that grows with the
> square of the sequence length and the composition tier has no way to declare a
> slot the workspace cannot size.

Option 1 is the answer to the question as asked. It does not *add* a primitive
today: `Attention` is the only case, and it is already one. The answer is what
will happen the first time a second case appears.

**Evidence that will settle it.** Attempt every composition's backward over the
P0 primitive set in a scratch branch and count the gaps (§7.6): a gap that is a
`T x T` pattern is option 1's first real case; a gap that is anything else is
D-004's missing primitive. Today the count of `T x T` gaps beyond `Attention` is
zero, which is why no second primitive is added in this campaign.

**Why not options 2 or 3.** Option 2 buys a workspace buffer that scales with
`seq^2` and a second tiling convention to keep in agreement with the primitive's,
for an operation the device implements natively anyway. Option 3 reintroduces the
fused family that D-004 forbids, and it is the configuration §7.4 condition 1
cannot check: with no composition there is nothing for the override to match.

---

## D-009 — Where does a block kind live?

**Status:** **Open.** **Phase:** P2.

**Question.** A block kind is a named meaning ([block_kinds.md](block_kinds.md)).
Does its implementation live in the model package that needs it, or in a shared
package from the start?

**Options.**

1. Per model package: `models/<name>/block_kinds/`, created only when a kind has
   no shared implementation.
2. A shared package from the start: one directory with every kind's
   implementation, read by every model.
3. No model-side implementation at all: every kind is served by the composition
   tier in `src/ops.cc`.

**Working answer.** Option 1, and specifically "option 1 with option 3 as the
default implementation": the composition tier defines every kind's semantics and
serves it for every model; `models/<name>/block_kinds/` exists only for a kind
with no shared implementation, and sharing a kind across models is a contract
change.

**Evidence.**

- `AGENTS.md` §6 forbids two owners editing one file, so a shared kind directory
  would have two model owners writing one source file. Option 2 is excluded by
  the ownership map before it is a technical question (§14.1 of the bring-up
  document).
- The composition tier already owns the definition of every kind's semantics
  (§7.4 condition 2: `src/ops.cc` *is* the composition tier). So the default
  implementation is not in a model package at all, and a per-model directory is
  an exception rather than a plan.
- `models/nanochat-default` is the proof: it creates `block_kinds/` only if its
  spec needs a kind with no shared implementation, and it leaves the directory
  out rather than committing an empty placeholder.
- A shared kind moves to a shared package only as a contract change, recorded
  here and in [block_kinds.md](block_kinds.md) §5, for the same reason a
  tolerance change is a contract-owner review: one owner's edit would otherwise
  reach another model.

**What happens with the answer.** A divergence is a new kind, not a flag on a
shared one ([block_kinds.md](block_kinds.md) §5). The first model that needs a
kind with no shared implementation creates its own directory and cites this
record.

---

## 3. What is deliberately not decided here

- **A tolerance value.** Per-model tolerances live in
  `models/<name>/tolerances.json`. The *review rule* is decided (a tolerance
  change is a contract-owner review, §11); the numbers are the model's.
- **A tile size, a thread count, a cache hint.** Device tuning, invisible to a
  caller, not a contract ([capability_bits.md](capability_bits.md) §5).
- **The numeric values of the op-code enum, the bit values in the mask, and the
  magic words.** These are build artifacts, not contracts
  ([pointwise_ops.md](pointwise_ops.md) §1, [capability_bits.md](capability_bits.md)
  §1).
- **The executable container.** Out of scope by §2 of the bring-up document, and
  the reason the numeric tables above are not contractual.
