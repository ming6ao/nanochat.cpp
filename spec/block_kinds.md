# Block kinds

A block kind is a **named meaning**, not a code path. It says what a position in
the model's ordered block list computes. It says nothing about which function
runs there. That distinction is the whole point of the list: the composition
tier, a backend's fused override, and a model package's own implementation may
all serve one kind, and all of them must satisfy the same golden trace
(`docs/model-and-device-bringup.md` §8.1: one definition of semantics, N
implementations, one golden trace).

This file is the authority for the kind names. The field list that carries them
is `include/nanochat/model_spec.h` (§8 of the same document), and the names are
consumed verbatim as strings by `models/<name>/spec.json`.

Related material:

- [capability_bits.md](capability_bits.md) — the bits a kind may require.
- [pointwise_ops.md](pointwise_ops.md) — the accuracy bound of the op codes a
  kind's composition is built from.
- [../docs/model.md](../docs/model.md) — the five graphs, the workspace, and
  the slot names as they exist in the tree today.
- [../docs/kernels.md](../docs/kernels.md) — the primitive inventory. This file
  refers to entry points by family name and does not restate the inventory.

## 1. The block entry

`ModelSpec`'s block list is ordered, and each entry is a record with one kind per
group:

| Group | Field | What the group decides |
|---|---|---|
| Norm | `norm` | The normalization applied before attention and before the MLP |
| Attention | `attention` | The pairwise operator over the sequence |
| — | `attention_variant` | `full` or `sliding`: the mask applied to that operator |
| Position | `position` | How position enters q and k |
| MLP | `mlp` | The feed-forward transform |
| Residual | `residual` | How a sublayer's output rejoins the residual stream |

Plus an optional per-block dtype policy. That is the whole entry list. A field
outside it is a spec format change and needs the version rise of
[versions.md](versions.md) plus a stated meaning the existing groups cannot
express (§8). This is the guard against the block list drifting into a graph
intermediate representation.

The list is closed in both directions: every kind a block may name appears
below, and a kind that does not appear is rejected at spec load with its name in
the error — never defaulted, never rounded to the nearest kind the reader knows.

## 2. The kinds

Six kind names across five groups. Every kind is listed once.

### Norm

| Kind | Computes | Saved slots its backward reads |
|---|---|---|
| `rms_norm` | `out[r, c] = x[r, c] * rstd[r] * weight[c]` where `rstd[r] = rsqrt(mean_c(x[r, c]^2) + eps)`. The mean is over the full row width of the block's residual stream | The input rows `x` and the per-row `rstd`. In the tree's workspace these are `xr`/`x_mid` and `rstd1`/`rstd2` (`BlockActivations` in `src/ops.h`). `weight` is a parameter, not a slot |

### Attention

| Kind | Computes | Saved slots its backward reads |
|---|---|---|
| `gqa_self_attention` | Causal softmax attention over the sequence, with the group-query head mapping of [capability_bits.md](capability_bits.md) bit 1, `scale = 1 / sqrt(head_dim)` unless the block overrides it, and the residual-stream rows split into query, key and value heads | `q_final`, `k_final`, `v_final`, and `attn_stats` — the per-query-row `(max, sum_exp)` pair, layout `[batch, num_heads, seq, 2]` (`AttentionStatsCount` in `include/nanochat/kernels.h`). The saved statistics are what let the backward skip a second softmax pass; a kind that does not save them would have to recompute them, so the slot is part of the declaration |

**The attention variant is a field, not a kind.** `full` and `sliding` differ
only in the mask: query row `i` sees every key in `[0, i]`, or every key in
`[i - window_left, i]`. The arithmetic, the head mapping, the saved slots, and
the tolerance are identical, so one kind carries both and the variant decides
the mask. A different *pairwise pattern* — anything that is not masked softmax
attention over the same rows — is not a variant; it is a new kind.

`Attention` is the one primitive with no composition behind it
(`docs/model-and-device-bringup.md` §7.2): its iteration space is `T x T`, and no
composition of the other primitives reproduces its tiling without materializing
a `T x T` tensor. So the attention kind is served by the primitive directly.
That is not an exception to "a kind is a meaning, not a code path": the meaning
is fixed and the primitive is one implementation of it.

### Position

| Kind | Computes | Saved slots its backward reads |
|---|---|---|
| `rope` | Rotates the half-split pair `(q[d], q[d + D/2])` of every query and key row by the angle of the row's sequence index, then applies the qk scale. `D` is the head width and the tables are fp32 `[seq, D/2]` cosine and sine, read from device memory | The pre-rotation rows `q_pre` and `k_pre` (`BlockActivations`), plus the `cos`/`sin` tables. The tables are constants, not slots: they are rebuilt only when the sequence length grows |

### MLP

| Kind | Computes | Saved slots its backward reads |
|---|---|---|
| `relu2_mlp` | `out = (relu(h @ c_fc))^2 @ c_proj`: one expansion to the MLP width, `relu^2` applied elementwise, one projection back | `pre_act`, the pre-activation rows. `d/dx relu(x)^2 = 2 * relu(x) * 1{x > 0}` needs only the sign and the value of the pre-activation, so `act` is derivable from it and is deliberately not a declared slot: a composition may recompute it, and a fused override may keep it |

### Residual

| Kind | Computes | Saved slots its backward reads |
|---|---|---|
| `plain_residual` | `out = x + f(x)` where `f` is the sublayer's output; the residual stream is the sum of every sublayer output in list order | **None.** An addition's backward is the identity on both branches, so a kind that reads nothing shares a workspace with anything |
| `value_gated_residual` | As `plain_residual`, with a ResFormer value gate added into the value rows before attention: `gate = m * sigmoid(dot(h[row, :G], gate_w[kvh, :]))`, `v[row, kvh, :] += gate * ve[row, kvh, :]`. Requires the `value_gate` bit | `ve_values` (the per-row value embedding rows), `ve_gate` (the saved gate, `[rows, num_kv_heads]`), and the hidden rows `h` that fed the gate. The backward reads the saved gate rather than recomputing it, so an override that saves a differently-rounded gate fails even when its `v` rows are right |

`value_gated_residual` is a second kind and not a flag on `plain_residual`
because it changes what the block computes: it reads an extra embedding, writes
the value rows, adds a parameter, and saves two slots. Two blocks that differ in
their slot declaration cannot share one kind, because the slot declaration is
what a shared workspace is built from (§3).

## 3. Saved slots and the workspace-sharing rule

`docs/model-and-device-bringup.md` §7.6 condition 1:

> A composition declares its saved slots. The slots are static and named. A
> composition backward reads only its declared slots, so a fused override and a
> composition can share a workspace.

Three consequences bind this file:

1. **The slots column above is the declaration.** A composition's backward reads
   exactly the slots its kinds declare and nothing else. A read outside the
   declaration is a bug even when it produces the right number, because it makes
   the workspace requirement invisible to the allocator and to the override that
   wants to share it.
2. **A composition spanning more than one kind declares the union.** The
   tree's `QkPrep` composition is norm on `head_dim` followed by `rope` and the
   scale, so it declares `rstd` over the head-width rows together with
   `q_pre` and `k_pre`. A composition that declares less than the union cannot
   be replaced by a fused override without the override secretly allocating.
3. **Shrinking a declaration is a contract change.** Dropping a slot because an
   implementation found it derivable is a detectable change: the workspace
   shrinks, and the override that shared the slot may now be reading freed
   memory. It follows the additive-with-a-version-rise rule of
   [versions.md](versions.md), and the consuming override is re-gated in the same
   change.

## 4. Where a kind's implementation lives

A kind name does not name an owner:

| Path | Who supplies it | When |
|---|---|---|
| The composition tier, `src/ops.cc` | The Workflow workstream, once, for every device | The default. Every kind has a composition, and the composition is the definition of the semantics |
| A backend's fused override, gated by an `override_*` bit | The backend | When the backend writes a faster single-kernel version of the same meaning |
| `models/<name>/block_kinds/` | The model owner | Only for a kind that has no shared implementation, and only with a README stating why |

The block list names a kind for every block; nothing in the list names a
composition, a kernel, a symbol, or a model-package file. A branch on a model
name, or a shared `switch` that every new model must edit, is the anti-pattern
of `docs/model-and-device-bringup.md` §15 and is rejected in review.

## 5. Sharing, divergence, and addition

- **Sharing a kind across models is a contract change.** Two models naming
  `rms_norm` means one implementation serves both, so one owner's edit reaches
  the other model. `AGENTS.md` §6 forbids two owners editing one file, so a
  shared kind lives above the model packages (the composition tier or a backend
  override), the sharing is recorded here, and it passes the contract owner.
  §17 open question 4 is record D-009 in [decisions.md](decisions.md); the
default is the model package.
- **A divergence is a new kind.** If two models need `rms_norm` to compute
  different things — a different epsilon rule, a per-channel scale, a norm over a
  different width — they need two kinds. Re-pointing one name at two meanings
  makes every fixture that used the name ambiguous, and the ambiguity surfaces
  as a tolerance argument rather than as a bug. The old name keeps its meaning
  or retires under the deprecation rule.
- **Adding a kind is an addition.** It costs a name here, a version rise of the
  spec format contract in [versions.md](versions.md), and an implementation on
  whichever path above owns it. It breaks no reader, because an unknown kind is
  rejected by name.
- **Retiring a kind needs one deprecation release**, and the name is never
  reused.

The list is closed: six kinds today, and a model that needs a seventh asks for
it rather than embedding it in a model package under a private name.

## 6. What is not a block kind

- **A composition that is not a block.** `GradNorm` and `ScalarDot` are
  compositions (§7.1 of the bring-up document) but no block list names them;
  they belong to the graph, not to a block. A composition becomes a kind only
  when the block list needs a slot for it.
- **A kind outside the block list.** The embedding lookup, the value-embedding
  lookup, the smear, the backout, the final norm, the logit softcap and the
  cross-entropy are model-level steps: they appear in the `Embeddings`, `Head`,
  and `Loss` groups of `ModelSpec`, not as block entries. They are graph steps
  today (`ops::SmearForward`, `ops::BackoutForward` are called once around the
  block loop in `src/model.cc`). A model that needs one of them *per block* asks
  for a kind here instead of adding a field.
- **A dtype policy.** The optional per-block dtype policy is a field of the
  entry, not a kind. `bf16` and the fp8 pair are precision declarations, and they
  live in [capability_bits.md](capability_bits.md).
- **A variant of a kind.** `full` and `sliding` are the two values of the
  attention variant, and they describe a mask over the same meaning. Any other
  difference is a new kind.
