# `models/nanochat-default`

The model axis for the flagship model. This package is data plus a build label:
it declares *what the model is* and owns the model's numeric contract. It
contains no graph builder, no fast path, and no backend tuning constant.

The spec format is version 1 (`include/nanochat/model_spec.h`, `kModelSpecVersion`).
The field set is closed: this file's `spec.json` uses only the fields that
header declares, and every value below is traceable to a default in
`include/nanochat/config.h` or to a constant the current graph reads.

```
spec.json        the definition of the flagship model's semantics
tolerances.json  one bound per fixture record, by record-name selector
fixtures/        the committed golden-trace bytes and their manifest
BUILD.bazel      the package's filegroup and its (empty) test_suite
```

`block_kinds/` is deliberately absent; see "Where a block kind lives" below.

## Provenance: every field against today's `Config`

`spec.json` reproduces the defaults of `nanochat::Config` exactly. The line
references are to `include/nanochat/config.h` unless another file is named.

| `spec.json` field | Value | Reproduces | Authority |
|---|---|---|---|
| `spec_version` | `1` | The only spec format ever written | `include/nanochat/model_spec.h:156`; `spec/versions.md` §2 |
| `name` | `nanochat-default` | The package identity | `include/nanochat/model_spec.h:161-170` |
| `requires` | `rope`, `sliding_window`, `value_gate`, `primitive_tier_complete` | The mechanisms the block list and policies need: rotary positions, the `S` mask, the value gate, and the complete primitive seam | `spec/capability_bits.md` §3 |
| `dimensions.num_heads` | `6` | `Config::num_heads` | `config.h:14` |
| `dimensions.num_kv_heads` | `6` | `Config::num_kv_heads` | `config.h:15` |
| `dimensions.hidden_dim` | `768` | `Config::hidden_dim` | `config.h:16` |
| `dimensions.seq_len` | `2048` | `Config::seq_len` | `config.h:17` |
| `dimensions.rope_base` | `100000.0` | `Config::rope_base` | `config.h:20` |
| `embeddings.vocab_size` | `32768` | `Config::vocab_size` | `config.h:18` |
| `embeddings.padded_vocab_size` | `32768` | `Config::padded_vocab_size` | `config.h:19`, validation at `config.h:95-97` |
| `embeddings.value_embedding` | `true` | `Config::value_embedding` | `config.h:22-24`, alternation at `config.h:36-42` |
| `head.kind` | `untied_linear` | The separate `lm_head.weight` and `transformer.wte.weight` matrices | `src/model.cc:187-198`; `include/nanochat/model_spec.h:87-90` |
| `head.logit_softcap` | `15.0` | `kDefaultLogitSoftcap` / `ops::kLogitSoftcap` | `include/nanochat/kernels.h:93`; `src/ops.h:28`; used at `src/model.cc:859,960` |
| `loss.kind` | `softcap_cross_entropy` | Logit softcap + cross-entropy over the vocabulary | `include/nanochat/model.h:73-100`; `include/nanochat/kernels.h:305-320` |
| `loss.ignore_index` | `-1` | A target of `-1` contributes no loss and is excluded from the mean; the backward zeroes ignored rows and the padded vocabulary tail | `include/nanochat/model.h:73-100`; `include/nanochat/kernels.h:305-320` |
| `policy.dtype` | `build_default` | No model-level dtype override: the build precision applies | `include/nanochat/model_spec.h:148-155`; `include/nanochat/tensor.h:121-140`; `nanochat.bzl:78-90` |
| `policy.allow_fused_overrides` | `true` | A registered fused family may win; the composition remains the definition | `include/nanochat/model_spec.h:152-153`; `include/nanochat/plan.h:16-27` |
| `policy.require_primitive_tier` | `true` | The model needs the complete primitive seam | `include/nanochat/model_spec.h:154` |

The `requires` list names mechanisms, never a device or an implementation. It
does not name `gqa`, because this model has one key/value head per query head
(the `mha` variant); it does not name a stateful bit, because the optimizer is
not part of the model definition; and it does not name an `override_*` bit,
because a model never requires an implementation choice. The
`primitive_tier_complete` entry repeats `policy.require_primitive_tier` so the
mechanism mask the bring-up reports carries the same claim as the policy.

The phase-1 contract policy that governs this list is:

- a `requires` string is spelled exactly as `spec/capability_bits.md` spells
  it; that file is the vocabulary authority, verbatim;
- a model requires only semantic bits — `gqa`, `rope`, `sliding_window`,
  `value_gate`, a precision bit (`fp32` or `bf16`), `fp8_storage` only when the
  model's numeric contract needs it, and `primitive_tier_complete` for the tier
  promise. It never requires an `override_*`, a `stateful_*`, or `fp8_gemm`
  bit;
- the header constant names and the spec strings drift (the AdamW, Muon, and
  fused-override families are the planted examples). The capability mask is
  read from this package's `spec.json` strings and never from a hard-coded
  constant name in this directory; mapping the strings to constants belongs to
  the reader, not to the model package.

`tensor.h`, `spec/`, and `include/` are frozen for this campaign and were not
edited here. The tree's phase-2 build adaptation is also not this package's
work, so no package build was attempted for this data-only node.

### Dimensions

`hidden_dim = 768` and `num_heads = 6` make `head_dim()` `128`
(`config.h:27`), and `num_heads == num_kv_heads` makes the attention variant
`mha` (`AttentionVariant::kMha`, `model_spec.h:51-55`). `Config`'s comment calls
the field "group-query attention" (`config.h:15`), but this default has one
key/value head per query head, so the general GQA mapping degenerates to MHA.
The `gqa` capability bit is defined for *fewer* key/value heads than query heads
(`spec/capability_bits.md` §3, bit 1), so `spec.json` does not require it.

The MLP width and `head_dim` stay derived, so neither appears in the spec:
`Config::mlp_dim()` is `4 * hidden_dim` (`config.h:34`) and
`Config::head_dim()` is `hidden_dim / num_heads` (`config.h:27`).

### The ordered block list

`blocks` has twelve entries because its length *is* `num_layers`;
`Config::num_layers` is `12` (`config.h:13`). Every block declares:

- `norm = rms_norm` (`NormKind::kRmsNorm`, `model_spec.h:38-41`);
- `position = rope` (`PositionKind::kRope`, `model_spec.h:57-62`);
- `mlp = relu_square` (`MlpKind::kReluSquare`, `model_spec.h:64-68`);
- `residual = blend`, the ResFormer residual: the per-layer residual scale plus
  the learned embedding blend (`ResidualKind::kBlend`, `model_spec.h:70-76`);
- `dtype = build_default`, so every block inherits the model policy
  (`DTypePolicy::kBuildDefault`, `model_spec.h:79-85`).

The attention entries reproduce `Config::window_pattern = "SSSL"`
(`config.h:21`) tiled across the layers: short, short, short, full, repeated
three times. A short block declares `attention = sliding_window`
(`AttentionKind::kSlidingWindow`); a full block declares
`attention = causal` (`AttentionKind::kCausal`). This is the same order
`Config::window_left()` reads (`config.h:49-57`); its "last layer is always
full" rule is already satisfied because the twelfth block is `L`.

The value-embedding rule is not a second per-block field. `Config` puts value
embeddings on alternating layers and always on the last layer
(`has_value_embedding`, `config.h:36-42`). With twelve layers that is layers 1,
3, 5, 7, 9, and 11. `EmbeddingsSpec::value_embedding = true`
(`model_spec.h:127-132`) is the only declared bit; the alternation stays derived
in `Config`. The block residual is the ResFormer blend on every block; the value
gate is present only where the embedding table is present, exactly as the
current graph reads `has_ve`.

The short window is derived, not declared: a quarter of the context rounded up
to a 128-token attention tile (`config.h:44-47`), with a causal right extent
(`config.h:59`).

### Vocabulary padding

`padded_vocab_size` is a declared size, not a derived round-up. Today both
sizes are `32768`, so the padded vocabulary adds no rows. The validation rule
that padding is at least the vocabulary is `Config::Validate()`
(`config.h:95-97`); the classifier reads only the first `vocab_size` columns and
the backward zeroes the padded tail (`include/nanochat/kernels.h:305-320`).

### Weight tying

The default is untied. `src/model.cc:187-198` allocates `lm_head.weight` and
`transformer.wte.weight` as two parameters, and `HeadKind` currently declares
only `kUntiedLinear` (`include/nanochat/model_spec.h:87-90`), so
`head.kind = untied_linear` records today's behaviour. A tied head would be a
new `HeadKind` name under the meaning-change rule (`spec/versions.md` §3.3), not
a flag on this one.

### Loss row weighting and masking

`LossKind::kSoftcapCrossEntropy` is the classifier composition: logit softcap,
cross-entropy over the unpadded vocabulary, `ignore_index = -1`, an optional
per-row weight applied after the softcap chain rule, and a backward that zeroes
every ignored row and the padded tail (`include/nanochat/kernels.h:305-320`).
`ForwardLoss` excludes `-1` targets from the mean (`include/nanochat/model.h:73-80`);
`BackwardAccumulate` divides by the valid-target count and `BackwardWeighted`
applies per-row weights (`include/nanochat/model.h:85-100`).

Row weighting is a call-time option of the same loss kind, not a fact about the
model's identity, so `spec.json` has no field for it. `LossSpec` declares
`kind` and `ignore_index` and nothing else (`model_spec.h:141-144`). If a
future model needs a different *default* row-weight rule, that is a new
`LossKind` name with its own meaning, not a new field bolted onto this one.

## One definition of semantics, N implementations, one golden trace

The statement this package exists to carry, from section 8.1 of the bring-up
design:

> The spec is the definition of the flagship model's semantics; implementations
> of it are optimizations. One definition of semantics, N implementations, one
> golden trace.

`src/ops.cc` is the composition tier and the default implementation of every
declared kind. A backend's fused override is an optimization that must satisfy
the same golden trace at the tolerances below. Nothing in this package names an
implementation, and nothing here branches on a backend, a device, or a model
name in a preprocessor condition.

There is no graph builder and no hand-written fast path in this package. A
hand-written fast path waits until a second model needs the generic builder
(migration step 4); section 2 keeps a second model out of scope. No numeric
value in this package is a tuning constant.

## Fixtures and the numeric contract

`fixtures/` holds the model release's committed reference bytes. They are
copied, not regenerated: no torch-dependent command was run here, and
`tests/data/*.bin` was not moved, rewritten, or deleted.

| File | Source | Records | Bytes | sha256 | FNV-1a 64 |
|---|---|---:|---:|---|---|
| `fixtures/debug_state.bin` | `tests/data/debug_state.bin` | 156 | 652387 | `06a4959cd83b8b4efe89a170c641ee66754d3f840e1396a634e7ff38f25b5407` | `0x4d43417f1969b020` |
| `fixtures/core_eval.bin` | `tests/data/core_eval.bin` | 43 | 107247 | `28f6f8f174a471fd6772c0ab947bde02a282ce307acf7687d27a6f9bbf7d4e6e` | `0xf06b593b962f2f2a` |
| `fixtures/train_parity.bin` | `tests/data/train_parity.bin` | 993 | 161760 | `006d6aaa62393cb95e641765d954e45950653f45082a644a62c5b75cf8caca23` | `0xbaee7504867988fb` |
| `fixtures/generate_parity.bin` | `tests/data/generate_parity.bin` | 39 | 106368 | `65af290d95eeaeefcc60fbb7ec8c75b3fa274a22cd8808d1c5199354bf092db3` | `0x6089b933917d2abb` |
| `fixtures/rl_parity.bin` | `tests/data/rl_parity.bin` | 69 | 108062 | `42735fc17fcea63ac6335ef78b952ba8ed83ee30fabf59335f39406debed12f2` | `0xfbc260e97c743092` |
| `fixtures/numerics_golden_10l.bin` | `tests/data/numerics_golden_10l.bin` | 1500 | 118370 | `2af2a12c26b8b9bbf76eb70e40afa3b3c0f1306b8a4aa58150ccbe630705edad` | `0xd7be5aebca44d775` |
| `fixtures/numerics_golden_10l_fp8.bin` | `tests/data/numerics_golden_10l_fp8.bin` | 1500 | 118368 | `5b30b15f05c027e0780d9d69f960977ff76167af2402d646097f368a37d8cfa8` | `0x06a057c9a57237a8` |

These are the model-dependent reference fixtures the tree already uses: the
PyTorch oracle (`debug_state.bin`), the evaluation/score reference
(`core_eval.bin`), the training trajectory (`train_parity.bin`), the greedy
generation fixture (`generate_parity.bin`), the RL rollout fixture
(`rl_parity.bin`), and the fp32 and fp8 legs of the staged golden trace
(`numerics_golden_10l*.bin`). The tokenizer, loader, chat, plan, API, and
checkpoint fixtures are not model definitions and stay in `tests/data/` with
their owning harnesses.

`fixtures/manifest.json` records, per file, every record name in container
order, the source path it was copied from, the magic and format version, the
size, the SHA-256, and the FNV-1a 64 checksum. The checksums prove each copy is
byte-identical to the source. When the tools node's `--model` selector lands it
becomes the generating authority for `fixtures/` and must reproduce the bytes
above exactly.

### `tolerances.json`

`tolerances.json` gives one bound per fixture record, for every one of the
seven files above. Because the trace files have 1500 records per precision and
the records repeat by step and parameter name, the file states the bounds as
ordered record-name selectors: the first selector that matches a record name is
that record's bound, and the file's `coverage` note says every record in
`fixtures/manifest.json` resolves to exactly one selector. The selectors cover
every record; none falls through.

The golden-trace fp32 bounds are the tolerance table of
`docs/numerics-integration.md` §4: loss and gradient norm `1e-5` relative,
parameter L2 `1e-4` relative, evaluation bits per byte and per-batch loss `1e-5`
relative, logit margins `1e-5` absolute, integer identifiers and masks exact.
The parameter and gradient hash records are exact on the CPU and simulator legs
and skipped on the device leg, where atomic scatter-adds change the bytes by
design. The fp8 file has its own golden and its own `1e-2` bounds, as the fp8
rows of the same section say; its margins are reported and its greedy
identifiers are exact.

The parity fixtures carry the bounds their existing tests already enforce:
`tests/oracle_test.cc` for `debug_state.bin` (forward and backward `1e-5`
absolute, the tight optimizer step `1e-5`, the trajectory loss `1e-5` and
parameters `1e-4`), `tests/train_parity_test.cc` for `train_parity.bin` (loss
atol/rtol `2e-3`, gradient norm `1e-3`/`1e-2`, parameter and gradient L2
`1e-3`/`5e-3`), `tests/score_parity_test.cc` for `core_eval.bin` (NLL and
focused logits `1e-4` absolute plus `1e-4` relative, argmax and offsets exact),
and exact input/reference comparisons for `generate_parity.bin` and
`rl_parity.bin`.

A tolerance change is a **contract-owner review**, not a model-owner decision.
A loosened tolerance turns the isolation gate into self-certification
(`docs/model-and-device-bringup.md` §11), so loosening one is not an edit this
package's owner may make alone.

## Build

`BUILD.bazel` exports the package as section 6 requires: a `filegroup` naming
`spec.json`, `tolerances.json`, and `fixtures/`, and an `all` `test_suite` that
is present even while empty, so the matrix generator and `//:all_tests` can
enumerate the package without a special case. The common options are loaded
from `//:nanochat.bzl` like the other packages. The root aggregation label is
owned by the root build file and is not edited here.

## Where a block kind lives

`block_kinds/` is not created, and the directory is left out rather than
committed as an empty placeholder. Every kind in this block list has a shared
implementation in the composition tier (`src/ops.cc`): `rms_norm`, attention,
`rope`, the relu-squared MLP, and the ResFormer residual blend. A model package
creates `block_kinds/` only for a kind with no shared implementation
(`spec/block_kinds.md` §4; `spec/decisions.md` D-009).

The rule is: per model by default, shared only as a contract change. Sharing a
kind across models means one owner's edit reaches another model, and
`AGENTS.md` §6 forbids two owners editing one file. A divergence is a new kind,
not a flag on a shared one, and a shared kind moves above the model packages
only as a recorded contract change.

## Contract gaps reported, not resolved here

This package edits only its own directory. Two vocabulary gaps belong to the
contract-header and spec-vocabulary owners:

1. **The block-kind spellings disagree.** `spec/block_kinds.md` §1-2 names the
   attention kind `gqa_self_attention` with an `attention_variant` of `full` /
   `sliding`, and names the residual kinds `plain_residual` /
   `value_gated_residual`. `include/nanochat/model_spec.h` instead declares
   `AttentionKind { kCausal, kSlidingWindow }`,
   `AttentionVariant { kMha, kGqa }`, and
   `ResidualKind { kPlain, kBlend }`. The header's split is what reproduces
   today's model: the mask varies per block while the head mapping is fixed at
   `mha`, and `kBlend` is the ResFormer residual scale that the prose's
   `value_gated_residual` does not carry. `spec.json` therefore uses the
   header's enum spellings (`causal` / `sliding_window`, `mha`, `blend`). The
   mismatch is reported to the spec-vocabulary and contract-header nodes; this
   package does not edit either contract.
2. **Loss row weighting has no field.** `LossSpec` has `kind` and
   `ignore_index` only. Row weighting is an optional behaviour of the
   classifier composition, so it needs no model-identity field. If the
   vocabulary later decides that a model must be able to declare a default row
   weighting, that is a new `LossKind` meaning (or a versioned field), not a
   value added here by hand.
