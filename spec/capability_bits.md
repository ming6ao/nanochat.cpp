# Capability bits

A capability bit is a named promise a device makes and a named requirement a
model states. The two axes never talk to each other and never name each other
(`docs/model-and-device-bringup.md` §4, Rule 1). They declare against this
vocabulary, and the planner resolves the two declarations into one plan
(§9 of the same document).

Version: **capability vocabulary 1** (the format is
[versions.md](versions.md); the numbers of every contract live there).

Related material:

- [versions.md](versions.md) — the four contract versions and the rule that a bit
  name never changes meaning.
- [pointwise_ops.md](pointwise_ops.md) — where the numeric obligation of an op
  code lives, one bound per code.
- [block_kinds.md](block_kinds.md) — the kinds a model's block list names, and
  the bits a kind requires.
- [decisions.md](decisions.md) — D-001, D-002, and D-003 are the decisions this
  table is downstream of.
- [../docs/kernels.md](../docs/kernels.md) — the entry-point inventory the
  `primitive_tier_complete` bit refers to. This file does not restate it.

## 1. The four places a bit name appears

| Place | Holds | Authority for |
|---|---|---|
| This file | The name, the meaning, the numeric obligation, the owning layer | The **names** |
| `include/nanochat/tensor.h` | `Caps::feature_mask` (`std::uint64_t`) and one named constant per bit, queried through `Caps::HasFeature(uint64_t)` | The **values** |
| `backends/<name>/caps.json` | The bits a device offers, as strings, one per bit | The **declaration** of one device |
| `models/<name>/spec.json` | The bits a model requires, in `requires`, as strings | The **requirement** of one model |

`Caps` keeps its existing fields: `supports_fp32`, `supports_bf16`,
`supports_fp8`, `Supports(DType)`, `has_cublas`, `has_tensor_cores`,
`total_memory_bytes`, `device_name`, `device_index`, `compute_major`,
`compute_minor`, `is_device`. The mask names the ones that are contractual
promises and adds the rest. A backend fills the mask from its `caps.json`-derived
profile entry plus what the device runtime reports, never from a second
hand-written list.

## 2. The naming rule

A bit has exactly one spelling everywhere it appears:

| Form | Spelling | Example |
|---|---|---|
| Vocabulary name, used verbatim as the JSON string in `caps.json` and in `spec.json` `requires` | `lower_snake_case` | `sliding_window` |
| C++ constant in `include/nanochat/tensor.h` | `kFeature` + upper camel case of the same name | `kFeatureSlidingWindow` |

The mapping is mechanical: split the vocabulary name on `_`, upper-case the
first letter of each part, join, prefix `kFeature`. No aliases, no abbreviations,
no second spelling for a bit whose meaning grew.

This matters more than it looks. A bit name that differs by one character
between this file and `tensor.h` is a compile-time silence — the header's bit is
never set, nobody notices — and a run-time lie: the planner reads a device that
offers a capability as a device that does not, or the reverse. Both failures are
silent and both are diagnosed as "the wrong implementation was chosen". A review
compares the list below against the header constant by constant, and a name that
does not match is a blocker, not a nit.

## 3. The bits

Fifteen bits, in five groups. Every bit is listed once.

| # | Name | Group | One-line meaning | Offered by |
|---|---|---|---|---|
| 1 | `gqa` | Architecture | Attention with fewer key/value heads than query heads | device |
| 2 | `rope` | Architecture | Rotary position embedding on q and k | device |
| 3 | `sliding_window` | Architecture | Causal attention restricted to a left window | device |
| 4 | `value_gate` | Architecture | A gate-scaled value embedding added into the value rows | device |
| 5 | `fp32` | Precision | Arithmetic in single precision | device |
| 6 | `bf16` | Precision | Arithmetic in bfloat16 | device |
| 7 | `fp8_storage` | Precision | One-byte quantization with a tensorwise scale | device |
| 8 | `fp8_gemm` | Precision | Scaled fp8 matmul with an fp32 accumulator | device |
| 9 | `primitive_tier_complete` | Tier | Every required primitive of the tier, at one attention tile size | device |
| 10 | `stateful_adamw` | Stateful | The AdamW update with its moment state | device |
| 11 | `stateful_muon` | Stateful | The Muon update with its factored state | device |
| 12 | `override_rms_norm` | Override | A single-kernel replacement for the `RmsNorm` composition | device |
| 13 | `override_qk_prep` | Override | A single-kernel replacement for the `QkPrep` composition | device |
| 14 | `override_classifier` | Override | A single-kernel replacement for the `Classifier` composition | device |
| 15 | `override_value_gate` | Override | A single-kernel replacement for the `ValueGate` composition | device |

A model states requirements (`requires`). A device states offerings
(`caps.json`). A model never requires an `override_*` bit and never requires a
stateful bit it does not train with: a model states what its semantics need, and
an implementation choice is not a semantic. The planner is the only place the
two lists meet.

Every numeric obligation below is stated as a relation over shapes, never as a
constant of the flagship model. The contract does not know the model (Rule 1).

---

### 1. `gqa`

| | |
|---|---|
| **Name** | `gqa` — `Caps::kFeatureGqa` |
| **Meaning** | Attention whose key and value projections carry fewer heads than the query projection, so a group of query heads shares one key/value head. `num_kv_heads == num_heads` is the degenerate 1:1 case and needs no separate bit |
| **Numeric obligation** | `num_heads % num_kv_heads == 0`; query head `h` reads key/value head `h / (num_heads / num_kv_heads)` for both `k` and `v`. The grouping is a mapping and not an average, so the result carries no arithmetic the 1:1 case does not: the attention output matches the committed attention fixture within the model's `tolerances.json` |
| **Owning layer** | Backend — the attention primitive (`AttentionForward`, `AttentionBackward`) |
| **Declared by** | model `requires`; device `caps.json` |

### 2. `rope`

| | |
|---|---|
| **Name** | `rope` — `Caps::kFeatureRope` |
| **Meaning** | Rotary position embedding applied to the query and key rows of one head before attention, with the position taken from the row's index in the sequence |
| **Numeric obligation** | For head width `D` and half width `H = D/2`, the angle of table row `t` and channel `d` is `theta = t * rope_base ** (-2d/D)`, `d` in `[0, H)`. The device reads fp32 `cos`/`sin` tables of shape `[seq, H]` from device memory; the host builds them, so trig precision is above the seam and is not the device's obligation. The device's obligation is the rotation of the half-split pair `(x[d], x[d+H])` to `(x[d]*cos + x[d+H]*sin, -x[d]*sin + x[d+H]*cos)`, followed by the qk scale, at one rounding per output element. The result matches the committed q/k fixture within the model's `tolerances.json` |
| **Owning layer** | Backend arithmetic. The rotate-pair arithmetic runs inside the pointwise family (see [pointwise_ops.md](pointwise_ops.md)); the position block kind that uses it is in [block_kinds.md](block_kinds.md). No entry point exists for `rope` alone, and none is added |
| **Declared by** | model `requires`; device `caps.json` |

### 3. `sliding_window`

| | |
|---|---|
| **Name** | `sliding_window` — `Caps::kFeatureSlidingWindow` |
| **Meaning** | Causal attention in which a query row attends only to the `window_left` keys preceding it and to itself. `window_left < 0` means full context and is the other variant of the same kind ([block_kinds.md](block_kinds.md)) |
| **Numeric obligation** | Key position `j` is visible to query `i` iff `i - window_left <= j <= i`; the right extent is 0, so the mask is causal and windowed at once. Out-of-window contributions are excluded *before* the softmax normalization, never zeroed after it, so the per-row sum in the saved statistics is the windowed sum. Full and windowed variants must run at one tile size, and the tile size is a device tuning parameter and not a bit. The result matches the committed attention fixture within the model's `tolerances.json` |
| **Owning layer** | Backend — the attention primitive, the same entry points as `gqa`. A device that implements full context only cannot run a model whose spec names a sliding variant, so it is non-conformant and not partially conformant (§7.2) |
| **Declared by** | model `requires`; device `caps.json` |

### 4. `value_gate`

| | |
|---|---|
| **Name** | `value_gate` — `Caps::kFeatureValueGate` |
| **Meaning** | A gate that scales a per-row value embedding before it is added into the value rows: for each row and key/value head, `gate = m * sigmoid(dot(h[row, :G], gate_w[kvh, :]))` and `v[row, kvh, :] += gate * ve[row, kvh, :]`, where `G` is the leading gate width of the hidden row and `m` is the gate multiplier |
| **Numeric obligation** | The dot runs over the leading `G` channels of the hidden row (`G` is `kVeGateChannels` in `src/ops.h`, 12 in this release) and is a per-segment reduce with a dot combine op, not a host round trip. The sigmoid and the multiply run through `kGateMul` (or `kExp` where the composition forms the sigmoid itself), the gate factor `m` through `kScale`, and the per-head broadcast of the gate over the value rows through `kRowScale`, so the declared bound of each of those codes in [pointwise_ops.md](pointwise_ops.md) is the numeric obligation here; the gate itself is saved and read by the backward, so forward and backward cannot drift. The result matches the committed value-gate fixture within the model's `tolerances.json` |
| **Owning layer** | Composition tier (`src/ops.cc`) over the device's primitives; an optional backend override behind bit 15 |
| **Declared by** | model `requires`; device `caps.json` |

### 5. `fp32`

| | |
|---|---|
| **Name** | `fp32` — `Caps::kFeatureFp32` |
| **Meaning** | The device computes arithmetic in single precision, which is the `ComputeType` of an `--config=fp32` build |
| **Numeric obligation** | Every fixture in `models/<name>/tolerances.json` passes at the committed fp32 tolerance — the loss-curve groups of `tests/numerics_trace.h` (`kLossRtol = 1e-5`, `kParamL2Rtol = 1e-4`, `kMarginAtol = 1e-5`). Persisted state (norm statistics, optimizer moments, master weights) stays `float` in this build and in every other, so the bit is about arithmetic and not about storage. A build whose precision is fp32 and a `Caps` without this bit is a startup failure, which `Caps::Supports(DType)` already enforces and the mask keeps enforcing |
| **Owning layer** | Backend, plus the build precision selected in `nanochat.bzl` (`--config=fp32`) |
| **Declared by** | device `caps.json`; a model may state it in `requires` when its numeric contract assumes it |

### 6. `bf16`

| | |
|---|---|
| **Name** | `bf16` — `Caps::kFeatureBf16` |
| **Meaning** | The device computes arithmetic in bfloat16, which is the `ComputeType` of an `--config=bf16` build. bfloat16 is always available as a *storage* type in every build, so the bit is about compute |
| **Numeric obligation** | The rounding is the shared one: round-to-nearest-even of the upper 16 bits of the fp32 pattern (`FloatToBf16Bits` in `include/nanochat/tensor.h`), inf and NaN keeping their top bits. A device that rounds differently breaks the shared rule and fails the bf16 leg of the numerics trace, which compares against the committed trace at that leg's tolerance. One precision per build and no runtime dispatch; persistent state stays fp32 |
| **Owning layer** | Backend, plus the build precision (`--config=bf16`) |
| **Declared by** | model `requires` (the common case); device `caps.json` |

### 7. `fp8_storage`

| | |
|---|---|
| **Name** | `fp8_storage` — `Caps::kFeatureFp8Storage` |
| **Meaning** | The device can quantize a tensor to one byte per element with one tensorwise scale and report the dequantization value: `QuantizeFp8` with E4M3 for forward operands and E5M2 for backward operands |
| **Numeric obligation** | The rounding is the shared header `include/nanochat/fp8.h`, so the device and the CPU emulator agree **exactly** and not within a tolerance (`docs/correctness-gates.md` §3.2: share the rule, not the implementation). The amax floor applies, because an all-zero tensor would otherwise produce an infinite scale. The declared dequantization value is the reciprocal of the scale the quantizer chose. The quantization bytes match the emulator bit for bit |
| **Owning layer** | Backend — `QuantizeFp8`, the optional half of the fp8 pair |
| **Declared by** | device `caps.json`; model `requires` only when the model's numeric contract is an fp8 contract |

### 8. `fp8_gemm`

| | |
|---|---|
| **Name** | `fp8_gemm` — `Caps::kFeatureFp8Gemm` |
| **Meaning** | A scaled fp8 matrix multiply, `C = alpha * op(A) * op(B) + beta * C`, with one-byte operands, per-operand dequantization values, and an fp32 accumulator |
| **Numeric obligation** | The accumulator is fp32. A device that reduces in lower precision declares it through `fast_accum`, which is forward only, and must then still hold the graph leg at the fp8 tolerance the trace table commits (`docs/correctness-gates.md` §2: 1e-2 on the loss and gradient leg). `fp8_gemm` implies `fp8_storage`: a device that can multiply fp8 but cannot produce it declares neither bit |
| **Owning layer** | Backend — `GemmScaledFp8` |
| **Declared by** | device `caps.json` |

### 9. `primitive_tier_complete`

| | |
|---|---|
| **Name** | `primitive_tier_complete` — `Caps::kFeaturePrimitiveTierComplete` |
| **Meaning** | The device supplies every required primitive of §7.2 of `docs/model-and-device-bringup.md` — the six utilities, both GEMM entries, both reductions, the pointwise family with all nine op codes, gather and scatter-add, and causal group-query attention in both the full and the sliding variant at one tile size. The inventory is [docs/kernels.md](../docs/kernels.md); this file does not restate it |
| **Numeric obligation** | Every primitive passes its own correctness test at the tier the device claims (`T0 CPU`, `S0 simulator`, or `T1 GPU correctness`), and the composition tier built over those primitives passes the five graphs of `docs/model.md`. A device that omits one required primitive declares **no** bit of this name: the tier is complete or it is absent, and there is no partial claim. Its absence is a named plan rejection, because no plan can be built at all — the planner cannot resolve a composition to a primitive it does not have |
| **Owning layer** | Backend, and the planner that reads it |
| **Declared by** | device `caps.json` |

### 10. `stateful_adamw`

| | |
|---|---|
| **Name** | `stateful_adamw` — `Caps::kFeatureStatefulAdamw` |
| **Meaning** | The AdamW update, carrying one first-moment and one second-moment buffer per parameter, bias correction from a 1-based step, an optional weight decay, and the gradient clip scale folded into the read |
| **Numeric obligation** | The update is stateful, so it is exact rather than tolerance-gated: for the same inputs it produces the same master weight within one rounding per element, because it contains no transcendental and no order-dependent reduction. The committed optimizer-state fixture pins the bytes, and the graph leg pins the loss. The folded clip scale is 1 when clipping is off or inactive, so an unscaled read stays bit-identical |
| **Owning layer** | Backend — the stateful tier (`AdamWUpdate`), owned by the Optimizer workstream below the seam |
| **Declared by** | device `caps.json`, and only when the device claims training |

### 11. `stateful_muon`

| | |
|---|---|
| **Name** | `stateful_muon` — `Caps::kFeatureStatefulMuon` |
| **Meaning** | The Muon update for stacked matrix parameters: batched GEMM plus the Newton–Schulz iteration, with a momentum buffer and a factored second moment reduced along one axis |
| **Numeric obligation** | The iteration count, the reduction axis (`columns` or `rows`), and the shape of the factored second moment (`[num_params, rows, 1]` for a column reduction, `[num_params, 1, cols]` for a row reduction) are part of the meaning. Newton–Schulz amplifies rounding, so the polar factor is compared against the reference within the committed fp32 tolerance rather than bitwise, and the resulting graph leg within `kLossRtol` |
| **Owning layer** | Backend — the stateful tier (`MuonUpdate`) |
| **Declared by** | device `caps.json`, and only when the device claims training |

### 12. `override_rms_norm`

| | |
|---|---|
| **Name** | `override_rms_norm` — `Caps::kFeatureOverrideRmsNorm` |
| **Meaning** | The device supplies a single-kernel implementation of the `RmsNorm` composition, resolvable as a fused override when the planner builds a plan |
| **Numeric obligation** | The composition defines the semantics and the override is an optimization (§7.4 condition 1). The override matches the composition's trace within the model's `tolerances.json` at the `T1 GPU correctness` shapes, and it reads only the saved slots the `rms_norm` kind declares in [block_kinds.md](block_kinds.md), so the override and the composition can share one workspace |
| **Owning layer** | The backend that writes it, registered against the composition name and this bit. The composition stays in the tree either way |
| **Declared by** | device `caps.json` only. A model never requires an override |

### 13. `override_qk_prep`

| | |
|---|---|
| **Name** | `override_qk_prep` — `Caps::kFeatureOverrideQkPrep` |
| **Meaning** | A single-kernel implementation of the `QkPrep` composition (norm on `head_dim`, rotate, scale), resolvable as a fused override |
| **Numeric obligation** | As bit 12: the composition's trace within tolerance at `T1` shapes, and only the saved slots the `rms_norm` and `rope` kinds declare in [block_kinds.md](block_kinds.md), which is the union a composition spanning two kinds declares. `QkPrep` is the reference case for the whole override tier — `DESIGN.md` §3.1 already requires the fused pair to carry the same golden trace as the composition |
| **Owning layer** | The backend that writes it |
| **Declared by** | device `caps.json` only |

### 14. `override_classifier`

| | |
|---|---|
| **Name** | `override_classifier` — `Caps::kFeatureOverrideClassifier` |
| **Meaning** | A single-kernel implementation of the `Classifier` composition (logit softcap, cross-entropy, unpadded-vocabulary slice, optional row weighting), resolvable as a fused override |
| **Numeric obligation** | As bit 12, with one addition that is the composition's own contract: the backward zeroes every ignored row and the padded vocabulary tail, and applies the per-row weight after the softcap chain rule. The override matches that trace within tolerance, including the padded tail's zeros and every ignored row — a tolerance on the loss does not cover a wrong zero |
| **Owning layer** | The backend that writes it |
| **Declared by** | device `caps.json` only |

### 15. `override_value_gate`

| | |
|---|---|
| **Name** | `override_value_gate` — `Caps::kFeatureOverrideValueGate` |
| **Meaning** | A single-kernel implementation of the `ValueGate` composition, resolvable as a fused override |
| **Numeric obligation** | As bit 12, and the gate it saves must be the gate the composition's backward would have computed, because the backward reads the saved gate rather than recomputing it. A device that saves a gate under a different sigmoid approximation fails the value-gate fixture even when its `v` rows are right |
| **Owning layer** | The backend that writes it |
| **Declared by** | device `caps.json` only |

## 4. Precision is declared like anything else

A device that offers no fp8 path **declares neither `fp8_storage` nor
`fp8_gemm`** and accepts a named rejection. It does not declare a bit it cannot
honour, it does not declare a "partial" fp8, and nothing in the tree assumes the
pair is present because a neighbouring device has it.

The rejection has a fixed shape (`docs/model-and-device-bringup.md` §9 rule 5):

```text
model requires:  fp32, bf16, fp8_gemm
device offers:   fp32, bf16, fp8_storage
resolution:      rejected — missing fp8_gemm
consequence:     the fp8 layer set has no fp8 matmul on this device; the
                 model declares no dense fallback for it
```

Three things a rejection must always do:

- Name the missing bit by its vocabulary name, so a reader can look it up here.
- Name the consequence in the model's own terms, so the reader learns what will
  not run rather than which device failed.
- Never say "unsupported device". A device that lacks one bit is conformant for
  every other bit, and the sentence hides which promise is missing.

Where the model *does* declare a dense fallback, resolution succeeds and the
plan records the declined path instead of a rejection. The plan is observable
(`tools/nanochat doctor`), so the user sees the same sentence whether the
resolution succeeded or failed.

## 5. What is not a bit

- **A tuning parameter.** Tile size, tile shape, thread count, cache hint. These
  vary per device and per launch and no caller can detect them. A parameter that
  a caller *can* detect belongs in the descriptor that carries it, not in the
  mask.
- **A numeric obligation of an op code.** The declared accuracy bound of
  `kExp`, `kRsqrt`, `kLog`, and the rest lives in
  [pointwise_ops.md](pointwise_ops.md), one bound per code. The pointwise
  primitive is one bit (`primitive_tier_complete`), and its nine codes carry
  nine obligations in that file. This split is deliberate: a bound changes far
  more often than a bit does, and a bound is read per code while a bit is read
  per device.
- **A fact about the device.** `total_memory_bytes`, `compute_major`,
  `compute_minor`, `device_name`, `has_cublas`, `has_tensor_cores`. These are
  reported, not promised. A boolean that is a fact stays a boolean.
- **A build precision.** A bit says what the device *can* do; the build says
  what it is *doing*. An `fp32` bit on a bf16 build is correct and expected, so
  no code may infer the running precision from the mask.
- **A model field.** `num_heads`, `window_left`, the softcap constant. A model
  states these in its spec and the planner compares them to the device's bits;
  they are never bits themselves.
- **`grad_norm` and `scalar_dot`.** Both are compositions with no override bit
  in vocabulary 1. `docs/model-and-device-bringup.md` §7.3 partitions
  `GradNorm` and `ScalarDot` as compositions with no fused counterpart, so there
  is nothing to gate. A backend that wants to fuse either one asks for a bit
  through the addition path (a new name here plus the version rise of
  [versions.md](versions.md)); it does not register an override under a bit that
  does not exist, because the planner would never resolve it and the backend
  would ship dead code that looks live.

## 6. Changing this vocabulary

The vocabulary is closed. A new bit is an addition and costs one deprecation
release only if it later retires; a bit's meaning never changes, because a
meaning change is a new name. Both are recorded in [versions.md](versions.md),
which rises the capability vocabulary version on the change. A bit that is
retired keeps its constant in `tensor.h` for one release, unused, so a stale
`caps.json` in the field is rejected by name rather than silently re-pointed.
