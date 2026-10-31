# backends/cpu

The reference implementation of every symbol in `nanochat/kernels.h`. It is the
correctness baseline, the oracle-on-CPU path, and the backend that lets the
workflow, optimizer, and model run before any GPU kernel exists.

It is also the proof of the primitive and composition tier
(`docs/model-and-device-bringup.md` section 7.4): the loops in `kernels.cc` are
the primitives, `src/ops.cc` is the definition of the compositions over them, and
this backend trades speed for clarity on purpose.

- `kernels.cc` — all eleven compute families, `Gemm`, the two reductions, and the
  device utilities. `--config=fp32` selects `ComputeType` (docs/precision.md).
- `caps.json` — the reference capability declaration (see below).
- `caps_names.h` — the one table where a vocabulary name meets a `kFeature*`
  constant, so a declared string with no constant fails loudly.
- `kernels_test.cc` — analytic checks, a finite-difference gradient check for
  every family with a backward, the declared pointwise accuracy bounds, and the
  composition experiment of section 17. Run with `tools/nanochat test
  //backends/cpu:kernels_test`.
- `caps_test.cc` — the gate on the declaration: `format` and vocabulary, every
  feature name resolved through `caps_names.h`, and no fused-override bit.

The library target is `//backends/cpu:cpu`.

## The reference declaration, `caps.json`

`backends/cpu/caps.json` is the declaration every other backend is reviewed
against. JSON holds no comments, so the reasoning lives here.

**The format is shared.** The file carries `"format": "nanochat.caps/1"`, the
capability vocabulary number (`spec/versions.md`), and a `devices` array with one
entry, so the same derivation in `src/device_profile.cc` reads every backend's
file. The entry's field names mirror `include/nanochat/device_profile.h` one for
one; the performance fields are zero because a host device has no launch cost, no
HBM rate and no tensor cores, and `is_device` is false for the same reason.

**What the entry declares.** The six device utilities (`Alloc`, `Free`, `Memcpy`,
`Memset`, `Synchronize`, `GetCaps`), the arithmetic primitives with both
reductions (`row_reduce`, `span_reduce`) and both attention variants at one tile
size, the nine pointwise op codes, the four combine ops, the two stateful entries
(`adamw`, `muon`), and the capability bits in the spelling of
`spec/capability_bits.md` — nine of the fifteen:

```text
gqa  rope  sliding_window  value_gate
fp32  bf16
primitive_tier_complete
stateful_adamw  stateful_muon
```

**It declares no fused-override bit, and that is the point.** The four
`override_*` bits are absent, so on this backend the planner resolves every
composition to the composition path: `RmsNorm`, `QkPrep`, `Classifier`, and
`ValueGate` are served by `src/ops.cc` here and never by a single-kernel
replacement. That is section 7.4's consequence expressed as data rather than as
prose, and `caps_test.cc` fails if an `override_*` name appears in the file.

**The fp8 pair is absent, and it is not an omission.** This backend emulates fp8
(`QuantizeFp8`, `GemmScaledFp8` with the shared rounding header) because that
emulation is the oracle a GPU's fp8 bits are measured against, not because the
host has an fp8 path: `supports_fp8` is false, and a bit says what the device can
run. Declaring `fp8_storage` here would claim a mechanism this device is never
asked to execute. The two fp8 entry points are still listed under `primitives`,
because they resolve and pass their gate; the `features` list, not the entry-point
list, is the promise the planner reads.

**The strings are the spec's; the mapping to the frozen header is explicit.**
`spec/capability_bits.md` section 2 makes the vocabulary name the JSON string and
`kFeature` + upper camel case the header constant, mechanically. The two lists
drift, so `caps_names.h` holds the mapping as data - one row per name, carrying
the `kFeature*` constant `include/nanochat/tensor.h` actually defines - and
`caps_test.cc` looks every declared string up through it. A declared string with
no constant is a hard failure naming the string, never a silent drop. The frozen
header is the authority here, so five rows carry its spelling and add no alias:
`kFeatureAdamW` and `kFeatureMuon` for `stateful_adamw`/`stateful_muon`, and
`kFeatureFusedResidualRmsNorm`, `kFeatureFusedQkPrep`, `kFeatureFusedClassifier`
for `override_rms_norm`/`override_qk_prep`/`override_classifier`.
`override_value_gate` has no constant in this build, so the table has no row for
it and this declaration must never name it; `kFeatureFusedAttention` has a
constant but no vocabulary name and no row either. `backends/cuda/caps_names.h`
carries the same drift from its own side.

**No fused override is registered.** This directory registers nothing against a
composition name: not `value_gate`, not `grad_norm`, not `attention`. The
`value_gate` bit below is the mechanism the declaration offers (the gate is
expressible from `kGateMul` and `kRowScale` here); it is not an override
registration, and `grad_norm` and `attention` have no bit to register against.

**Fused families still in the tree, and why.** `kernels.cc` still implements
`RmsNormForward`/`RmsNormBackward`, `QkPrepForward`/`QkPrepBackward`,
`ClassifierForward`/`ClassifierBackward`, and `ValueGateForward`/
`ValueGateBackward`, because `src/ops.cc` and `src/model.cc` call those symbols
today and the tree has to keep building. They are **follow-up removals**: each
one is deleted once `src/` calls the composition instead, in the same change set
that stops needing the symbol. The declaration already says what the planner will
choose, so the removal changes no plan and no trace — the compositions that
replace them are checked against them by `TestRmsNormComposition` in
`kernels_test.cc` at the committed tolerance. `GradNorm` and `ScalarDot` are
compositions with no fused counterpart (`spec/capability_bits.md` section 5) and
stay entry points until `src/` rebuilds them over `SpanReduce`.

**`attention.tile_rows`.** The reference has no key tiling: one query row scans
its whole visible window in one pass. So the declared tile is one query row, and
both mask variants (`full`, `sliding`) run at that same tile, which is what
`capability_bits.md` bit 3 requires. A tile size is device tuning, not a bit
(section 5 of that file).

**What is deliberately absent.** No model name, no model constant, and no model
size appears anywhere in this directory, and no preprocessor condition names a
device: a bit and a shape relation are the only vocabulary a backend is allowed
to know (Rule 1, section 4 of the bring-up document).

**The contract policy this directory follows**, so a reviewer can check the rules
against the code rather than against an argument:

1. `caps.json` declares only bits that map to a constant in the frozen
   `include/nanochat/tensor.h`, spelled exactly as `spec/capability_bits.md`
   spells them. The mapping is `caps_names.h`; an unmapped name is a hard failure
   in `caps_test.cc`. `override_value_gate` is omitted because no constant exists
   for it, and no `override_grad_norm` exists to declare.
2. The header's constant names win over the mechanical rule where they drift
   (`kFeatureAdamW`, `kFeatureMuon`, and the `kFeatureFused*` rows). No alias is
   added to either side.
3. `kRowScale` follows `spec/pointwise_ops.md`: `a` is the per-row scalars, `b` is
   dense, `alpha` is not read, and a width of zero is a declared error.
4. `kGateMul` is gated by its `4e-6` bound and not by bit equality, although
   `include/nanochat/kernels.h` lists it among the bit-exact codes.
5. No fused override is registered: not `value_gate`, not `grad_norm`, not
   `attention`. This directory has no registration path at all, and
   `include/nanochat/tensor.h` is frozen for the campaign, so neither the header
   nor `spec/` is edited from here.

## The primitive tier

**The two reductions (D-001, D-006).** `RowReduce` writes one scalar per segment
of `count` elements `stride` apart, starting at `offset`; the combine op is
`kSumSq`, `kSum`, `kMax`, or the max-stable `kLogSumExp`. `SpanReduce` reduces a
ragged span list to one scalar, with an optional second list (the per-span dot),
a `scale`, and an `accumulate` flag. Both are plain loops: no host round trip, no
hidden allocation, and the span list is read where the caller put it.

**The pointwise block (D-003).** `PointwiseForward`/`PointwiseBackward` take a
`PointwiseParams` block and cover all nine codes. `kRowScale` is the row
broadcast, and it follows `spec/pointwise_ops.md` section 3.1 exactly: `a` holds
the `count / row_width` per-row scalars, `b` and `out` hold `count` values,
`out[r, c] = a[r] * b[r, c]`, and the code reads no scalar of the parameter block
- `row_width` is its whole field list, so a stale `alpha` or `beta` cannot move
the answer. The forward is one loop with no `rows x width` copy of the row
scalars; the backward is the per-row reduction `da[r] = sum_c dy[r,c] * b[r,c]`
plus `db[i] = dy[i] * a[i / row_width]`.

**`kRowScale`'s declared errors.** `row_width == 0`, a width larger than the
count, and a count that is not a multiple of the width are declared errors
(`spec/pointwise_ops.md` section 3.1), because a wrong width silently reads the
wrong row and every number downstream still looks plausible. The seam has no
error return, so this backend writes nothing and prints one line naming the count
and the width; a loud no-op beats a plausible wrong answer, and the caller-side
rejection belongs to the composition tier.

**The accuracy bounds.** Every code is gated against the absolute per-element
bound of `spec/pointwise_ops.md` section 4 over the envelope stated there, never
by bit equality: a device `expf` and a host `std::exp` are both correct and
disagree in the last place. `kernels_test.cc` has one bound test per new code
(`TestPointwiseExpBound`, `TestPointwiseRsqrtBound`, `TestPointwiseLogBound`,
`TestPointwiseRowScale`) plus one for `kGateMul`
(`TestPointwiseGateMulBound`, bound `4e-6`), and a finite-difference check of the
three transcendental backwards. `kGateMul` is gated by its bound and not by bit
equality even though `include/nanochat/kernels.h` groups it with the bit-exact
codes: its sigmoid is a transcendental, and a bound is what a downstream sum
actually receives. The bound gates run in the fp32 build, which is the
correctness gate (`docs/precision.md` section 8); a bf16 store has about three
decimal digits of its own and cannot hold an absolute bound of `1e-6`.

## The section 17 experiment (record D-006), as run here

`TestRmsNormComposition` writes `RmsNorm` forward and backward as compositions
over `RowReduce` and `Pointwise` alone, at `B=2, T=8`, and compares them with the
fused family at the committed fp32 tolerance (`tests/numerics_trace.h`:
`kLossRtol = 1e-5`, and `kParamL2Rtol = 1e-4` is looser, so one bound covers both
records). What it found, per call site:

| Call site | Fields needed | Fields untouched | Worked around |
|---|---|---|---|
| `RmsNorm` forward (sum of squares per row) | `n_out`, `count`, `op` | `stride`, `offset` | — |
| `RmsNorm` backward (dot per row) | `n_out`, `count`, `op` | `stride`, `offset` | the dot: two `kSumSq` calls and the polarization identity |
| `Classifier` row log-sum-exp (vocabulary offset) | `n_out`, `count`, `offset`, `op` | `stride` | the segment pitch: one call per row |
| `kRowScale` backward (dot per row) | — | — | computed inside the pointwise family, since the dot case does not exist in the descriptor |

Two findings are reported to the contract owner rather than patched here, because
`include/nanochat/kernels.h` is not this directory's file:

1. **No second operand and no dot combine op.** `RowReduceParams`
   (`include/nanochat/kernels.h:205`) has one input pointer and `ReduceOp`
   (`include/nanochat/kernels.h:81`) has no dot code, so `sum_c x * dy` — the
   `RmsNorm` backward's statistic and `kRowScale`'s `da` — cannot be one call.
   The composition measures the workaround instead: `sum x*dy = (sum (x+dy)^2 -
   sum (x-dy)^2)/4`, two `kSumSq` calls over one added and one subtracted pair.
   It is accurate enough for the trace (the measured drift here is below `1e-7`
   relative, because both sums are accumulated in double), but a workaround is a
   missing field wearing a disguise (D-006), and the field is
   `ReduceOp::kDot`/a second operand pointer.
2. **No segment pitch distinct from the extent.** A segment advances by
   `count * stride`, so the `Classifier`'s shape — `vocab_size` elements of a
   `padded_vocab_size`-wide row — is not one call: it is `rows` calls of
   `n_out = 1` with a per-row `offset`. `TestRowReduceClassifierOffset` asserts
   both the form that works and the fact that the single-call form cannot, so the
   limit is a test and not a sentence. The missing field is a per-segment input
   pitch, separate from `count`.

Both findings are reported, not patched: this node does not own
`include/nanochat/kernels.h`, and a silent second edit is what section 7.2
forbids.

Three details of the pointwise family are settled the spec's way here and
recorded in the same place:

- `spec/pointwise_ops.md` section 3.1 says `kRowScale`'s `a` holds `count /
  row_width` per-row scalars and `b` holds `count` values. The comment on
  `PointwiseOp::kRowScale` (`include/nanochat/kernels.h:76`) and the block comment
  at lines 278 to 286 state the opposite roles
  (`out[i] = alpha * a[i] * b[i % row_width]`, with `b` replicated down the
  rows). This backend follows the spec, because that is what the two call sites
  need (`out[r, c] = rstd[r] * x[r, c]`); the header's comment is the part that
  disagrees with the header's own code list (`row_width` is the only field the
  spec gives the code).
- The same header comment makes `alpha` a multiplier for `kRowScale`. This
  backend ignores `alpha` for that code, per the spec's field list; a stale value
  is not asserted, it is simply not read.
- A `row_width` that is zero on a `kRowScale` call, one larger than the count, or
  one that does not divide the count, is a declared error in
  `spec/pointwise_ops.md` section 3.1. The seam has no error channel, so the call
  prints one line naming the count and the width and writes nothing (see above);
  `include/nanochat/kernels.h:282`'s "0 means count" reading is superseded by the
  spec here.

## Contracts worth knowing

- **Layouts.** Q/K/V are `[batch, seq, heads, head_dim]` row-major (the layout
  nanochat's SDPA fallback uses). Attention stats are
  `[batch, num_heads, seq, 2]` holding `(max, sum_exp)`.
- **GQA head mapping.** Query head `h` uses key/value head
  `h / (num_heads / num_kv_heads)`, matching `enable_gqa`.
- **`QkPrep` backward.** The `q`/`k` buffers passed to `QkPrepBackward` are the
  *saved pre-norm projection outputs*; they are overwritten with the gradient
  with respect to those projection outputs. The upstream `dq`/`dk` are taken
  with respect to the final (RMSNorm -> RoPE -> scale) activation. Saving the
  pre-norm rows is what makes the RMSNorm statistic recoverable. The forward
  writes the final activation into whatever buffer it is given, so a model that
  needs both must keep a copy of the pre-norm rows (for example
  `q_proj` for backward and a `q_prep` copy for attention).
- **Write-not-accumulate.** `RmsNorm*` and `Pointwise*` write their outputs.
  `Classifier*` writes its outputs. `EmbeddingBackward` scatter-adds onto the
  dense gradient buffer, so the caller zeroes that buffer once with `ZeroGrad`
  and repeated backward calls sum. `AttentionBackward` zeroes `dq`/`dk`/`dv`
  before accumulating. `SpanReduce` accumulates only when `accumulate` is set,
  and the reduction entries write their `out` otherwise.
- **Loss scaling.** `ClassifierForward` writes per-row cross-entropy and
  `ClassifierBackward` writes the gradient of the *sum* of those rows
  (`softmax - onehot` through the softcap). The caller divides by the number of
  non-ignored rows to get a mean-loss gradient.
- **Gemm.** Universal row-major `C = alpha * op(A) * op(B) + beta * C`; the
  `GemmMode` enum is advisory and the transpose flags decide the operands.
