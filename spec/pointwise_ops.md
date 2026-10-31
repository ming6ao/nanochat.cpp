# Pointwise op codes

`PointwiseOp` is one closed C++ enum, one entry point pair, and one declared
accuracy bound per code. The codes are the transcendental legs of the
composition tier: without them `RmsNorm`, `QkPrep`, `Classifier`, and
`ValueGate` cannot be written over primitives, and with them they can
(`docs/model-and-device-bringup.md` §7.2, §7.7).

This file is the authority for the code meanings and for the accuracy bounds.
The code names live in `include/nanochat/kernels.h` and must match this table
name for name; the parameter block lives beside them. The version this table
belongs to is **kernel seam 1** ([versions.md](versions.md)).

Related material:

- [capability_bits.md](capability_bits.md) — the bit that says a device supplies
  the pointwise family, and the rule that a bound belongs to a code and not to a
  bit.
- [block_kinds.md](block_kinds.md) — the kinds whose compositions are built from
  these codes, and the slots they declare.
- [decisions.md](decisions.md) — D-006, which settles the absolute-per-code form
  of the bound below.
- [../docs/kernels.md](../docs/kernels.md) — the tier the family sits in. This
  file does not restate the inventory.

## 1. Why the enum stays closed

One build links the runtime and every backend into one binary, so an op code
cannot skew between the caller and the implementation: the caller's constant and
the backend's `switch` label are the same enumerator. A device that misses a code
fails to **compile**, which is the cheapest safety net available and the only one
that costs no run time.

A stable numeric table would be needed only if a program carrying an op code
could arrive from a different build than the runtime — a serialized kernel
descriptor, an executable format, a plugin boundary. That belongs to the
executable container, which is out of scope (`docs/model-and-device-bringup.md`
§2). The enum's underlying values (0 to 8 in declaration order) are therefore
**not** a contract: no file, fixture, or `caps.json` records one, and a reader
must never compare a stored integer against a code. What is a contract is the
name and the meaning.

The same code drives forward and backward, so the two cannot drift apart
(`DESIGN.md` §5).

## 2. The parameter block

`PointwiseForward` and `PointwiseBackward` take a `PointwiseParams` block
instead of the scalar argument list they take today. The block is:

| Field | Meaning |
|---|---|
| `op` | The code. Closed enum, above |
| `count` | The number of elements in `a`, `b`, `out`, and in the gradient tensors. One flat count: the family is elementwise and never sees a shape |
| `row_width` | The number of elements per row, for the codes that read a per-row operand. **Read only by `kRowScale`**; every other code ignores it, and 0 means "not a row-broadcast call" |
| `alpha` | The first scalar. Its meaning is per code: the multiplier for `kScale`/`kScaleAdd`, the cap for `kSoftcap`. Unused by the codes that read no scalar |
| `beta` | The second scalar. Its meaning is per code: the second multiplier for `kScaleAdd`. Unused elsewhere |

The block exists because the argument list had already grown one scalar beyond
its meaning: `kSoftcap` reads its cap from `alpha`, so `alpha` carries three
different jobs across five codes, and the row broadcast needs a width that has
no place in a scalar list. A struct with one documented field per meaning is the
only shape that survives the next four codes.

Rules that apply to every code:

- **No allocation and no host round trip.** Both entry points run entirely on
  the device over the pointers they are given.
- **`out` may alias `a`.** Call sites rely on it: the residual accumulation
  passes the same buffer as input and output, and every composition reuses its
  buffers in place. A backend that stages through a temporary must reproduce the
  aliased answer, not the staged one.
- **`b` may be null** for a code that does not read it, and `da`/`db` may be null
  in the backward, which means "this operand's gradient is not wanted".
- **The backward reads `a` and `b` only.** It never reads `out`, because a
  composition's declared saved slot for a pointwise leg is its input `a`
  ([block_kinds.md](block_kinds.md) §3). A backend that wants to reconstruct
  from `out` may, but the declaration stays `a`, so a fused override and a
  composition can share one workspace.

## 3. The codes

Nine codes: five that exist today, four added by this contract extension.

| Code | Forward | Backward (`da`, `db`) | `PointwiseParams` fields read |
|---|---|---|---|
| `kScale` | `out = alpha * a` | `da = alpha * dy` | `alpha` |
| `kScaleAdd` | `out = alpha * a + beta * b` | `da = alpha * dy`, `db = beta * dy` | `alpha`, `beta` |
| `kGateMul` | `out = sigmoid(a) * b` | `da = dy * b * s * (1 - s)`, `db = dy * s`, with `s = sigmoid(a)` | none |
| `kReluSquare` | `out = relu(a)^2` | `da = dy * 2 * a` when `a > 0`, else 0 | none |
| `kSoftcap` | `out = alpha * tanh(a / alpha)` | `da = dy * (1 - t^2)`, with `t = tanh(a / alpha)` | `alpha` (the cap) |
| `kExp` | `out = exp(a)` | `da = dy * out` (recomputed from `a`, not read from `out`) | none |
| `kRsqrt` | `out = 1 / sqrt(a)`, `a > 0` | `da = -0.5 * dy * out^3` (recomputed from `a`) | none |
| `kLog` | `out = log(a)`, `a > 0` | `da = dy / a` | none |
| `kRowScale` | `out[i] = a[i / row_width] * b[i]` | `da[r] = sum_c dy[r,c] * b[r,c]`, `db[i] = dy[i] * a[i / row_width]` | `row_width` |

`kGateMul` reads no scalar on purpose. The ResFormer value gate multiplies by a
gate factor `m` (`gate = m * sigmoid(dot)`, [block_kinds.md](block_kinds.md) §2),
and that factor is applied by a following `kScale` with `alpha = m` rather than
by giving `kGateMul` a third meaning for `alpha`. A code that quietly reads a
field for one call site is how an enum loses its meaning.

### 3.1 `kRowScale`: the row width, and why a code and not an entry point

**What `row_width` means.** `row_width` is the number of elements in one row of
the dense operand, so for the flat element index `i`:

- the row is `r = i / row_width`,
- the column is `c = i % row_width`,
- `a` holds exactly `count / row_width` values, the per-row scalars in row
  order, and `b` and `out` hold `count` values.

`count % row_width != 0`, a `row_width` of 0 on a `kRowScale` call, or a
`row_width` greater than `count` is a **declared error**: the call rejects with
a message naming the count and the width. It is not rounded, clamped, or guessed,
because a wrong width silently reads the wrong row and every number downstream
still looks plausible.

The code computes `out[r, c] = a[r] * b[r, c]` with no materialized
`rows x width` operand. Two call sites depend on it in this release:

1. **`RmsNorm`.** `out[r, c] = rstd[r] * x[r, c]`, then the per-channel weight.
   Without the broadcast this needs a `rows x dim` copy of `rstd`, which is a
   second buffer, a second pass, and a workspace slot that grows with the
   sequence length — for one line of arithmetic.
2. **`ValueGate`.** The saved gate is `[rows, num_kv_heads]` and the value rows
   are `[rows, num_kv_heads, head_dim]`, so `gate * ve` is the same broadcast
   with `row_width = head_dim`. Without it the gate is expanded by hand, which
   the CPU backend does today through a host staging helper.

**Why it is an op code and not a new entry point.**

- The access pattern is the one the family already implements: one elementwise
  loop over `count` elements, with a shape parameter. A primitive tier is
  counted in entry points, and every entry point is a thing every device must
  implement forever (§7.5: the required bring-up cost is 16 entry points, and
  each new one raises it for inference and for training).
- The row broadcast is one multiplicand's index expression. It needs no
  different units, no different accumulation, and no different correctness
  argument, which is the test §7.2 uses to decide that the two reductions are
  *separate* entries. By that same test, this is not a new entry: it is the same
  kernel with one extra shape word.
- A new entry point is a frozen-header change that every backend must satisfy
  before it can link. An op code is one `switch` label in a function the device
  already writes, and the closed enum makes a missing label a compile error
  rather than a link error.
- Its backward is a per-row reduction (`da[r] = sum_c dy[r,c] * b[r,c]`), which
  is the `RowReduce` primitive with a multiply combine op — the same shape the
  `Classifier` composition needs with a vocabulary offset. So the broadcast
  reuses the primitive the tier already gained, and it is a third piece of
  evidence for the descriptor question in D-006.

## 4. The accuracy bounds

Every code carries **one declared absolute bound per element**, asserted over a
stated operand envelope. This is the absolute-per-code form, taken as the working
answer to `docs/model-and-device-bringup.md` §17 open question 2 (record D-007).

| Code | Declared bound (absolute, per element) | Operand envelope the bound is asserted over |
|---|---|---|
| `kScale` | `8e-6` | `abs(alpha) <= 4`, `abs(a) <= 8` |
| `kScaleAdd` | `2e-5` | the above, plus `abs(beta) <= 4`, `abs(b) <= 8` |
| `kGateMul` | `4e-6` | `abs(a) <= 40`, `abs(b) <= 8` (so the gate is in `(0, 1)`) |
| `kReluSquare` | `6e-5` | `abs(a) <= 16` |
| `kSoftcap` | `8e-6` | `abs(a) <= 64`, `alpha` in `[4, 64]` |
| `kExp` | `3e-7` | `a` in `[-88, 0]`, so the output is in `(0, 1]` |
| `kRsqrt` | `1e-6` | `a` in `[0.25, 4]` |
| `kLog` | `2e-5` | `a` in `[1e-30, 1e6]` |
| `kRowScale` | `8e-6` | `abs(a) <= 4` per row scalar, `abs(b) <= 8` |

Each bound is roughly two units in the last place at the envelope's largest
magnitude, which is what one correctly rounded arithmetic step costs plus the
allowance for a device that contracts a multiply and an add, or that computes a
transcendental in an approximation rather than in a library call. The envelope is
part of the declaration: an absolute bound with no envelope is meaningless,
because the smallest difference a float can represent scales with the magnitude,
and a bound tight enough to be useful at `1e-3` is impossible at `1e3`.

Outside the envelope a caller owns the bound, and the rule is `2 ulp` of its
output magnitude. Two of the envelopes are load bearing and worth stating
explicitly:

- **`kExp`.** The bound holds because the compositions that call `kExp` shift by
  the row maximum first (`Classifier`'s log-sum-exp) or feed a negated dot
  (`ValueGate`'s sigmoid), so the operand is non-positive and the output is at
  most one. A caller that passes a positive argument gets an output that can
  exceed one by many orders of magnitude, and no fixed absolute bound covers
  that; the shift is not an optimization, it is what makes the bound
  expressible.
- **`kRsqrt`.** The bound holds on the mean square of a normalized row, which
  sits near one. A mean square far outside `[0.25, 4]` means the norm's input was
  not normalized, which is an upstream bug and not a bound failure; the eps in
  the norm keeps the operand off zero, so the reciprocal square root never sees
  an infinite result.

### 4.1 How a gate asserts a bound

A test for a code asserts the bound, never bit equality:

```text
for each operand set in the code's envelope, over the committed fixture inputs
and the tier's shapes:
    ref = the CPU backend's evaluation of the code's meaning in double precision
    got = the implementation under test
    assert abs(got - ref) <= the declared bound for that code
```

Bit equality is the wrong gate for a transcendental: a device `expf` and a CPU
`std::exp` are both correct and disagree in the last place. It is the right gate
for the codes that contain no transcendental (`kScale`, `kScaleAdd`,
`kReluSquare`, `kRowScale` contain only multiplies and adds), and the table above
is loose enough to admit both a contracted and an uncontracted implementation of
them, which is deliberate: the gate is a numeric contract, not a build flag.

### 4.2 Why absolute per code, and not units in the last place

The evidence is a loss-curve comparison at the tolerance the tree already
commits: `kLossRtol = 1e-5`, `kParamL2Rtol = 1e-4`, `kMarginAtol = 1e-5` in
`tests/numerics_trace.h`. Substitute one code's device implementation for the
reference one at a time and compare the trace's `train_loss` and
`train_grad_norm` records per step. If every substitution stays inside
`kLossRtol`, the set of per-code bounds is admissible.

That comparison is the reason the form is absolute:

- **A per-code bound has to compose into a graph bound.** The trace compares a
  loss, which is an absolute quantity on a known scale (a few units per token).
  An absolute per-code bound adds up into an absolute graph bound with one
  multiplication per layer. A per-code relative bound does not compose without
  knowing the magnitude of every intermediate, which is exactly what a reviewer
  cannot get from the table.
- **`kLog` decides it.** Near one, `log(1 + e)` has an unbounded *relative*
  error and a tiny absolute one. A units-in-the-last-place bound would either
  pass a catastrophically wrong logarithm or fail a correct one, depending on how
  the last-place unit is defined. The absolute form states the truth: the number
  is right to within `2e-5`, and that is all a downstream sum needs.
- **A device cannot be held to a bit pattern it cannot see.** A last-place bound
  is a relative statement about the reference's rounding, so it silently changes
  meaning when the reference changes; an absolute bound is a statement about the
  value.

The cost of the absolute form is the envelope, which is why each bound names
one. An envelope is a claim about the graph, and a claim about the graph is
checkable: the trace asserts every operand stays inside it, and a violation
fails with the record name rather than quietly widening the bound.

### 4.3 Moving a bound

A tightening is free and still rises the seam version, because a reader can
detect it: the tighter bound is what will fail a device that used to pass. A
loosening rises the version **and** needs the contract-owner review of
`docs/model-and-device-bringup.md` §11, because a loosened numeric obligation is
how a gate turns into decoration. Both movements are recorded in
[versions.md](versions.md), and a bound change never renames a code: the code's
meaning is unchanged, only its numeric promise moved.
