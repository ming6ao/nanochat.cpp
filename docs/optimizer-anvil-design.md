# ANVIL optimizer (Phase 1)

ANVIL is a Muon successor used by the reference `modded-nanogpt` run. This
document describes the C++ port: the algorithm, the configuration surface, the
grouping, and the checkpoints. It is opt-in; the default matrix optimizer stays
Muon.

Reference sources (read-only, outside this tree):

- `modded-nanogpt/track_1_short/optim/anvil.py` — the algorithm and constants.
- `modded-nanogpt/track_1_short/training.py` — the parameter table, the Adam
  cadence, and the rail schedule wiring.
- `modded-nanogpt/track_1_short/schedule.py` — `get_rail_beta`.

## 1. What ANVIL replaces

Muon applies Nesterov momentum, row equilibration, Polar Express
orthogonalization, a Frobenius renorm, NorMuon variance reduction, and a
cautious weight decay. ANVIL keeps the "update the matrix, then scale it"
shape but changes every stage:

1. **Twin-rail momentum.** Two velocity EMAs of the gradient. Rail 0 is fast,
   rail 1 is slow. Before the engage step the update reads the fast rail alone
   on the scheduled beta; after it, the two rails are blended.
2. **Whitening cascade.** Instead of Polar Express, six quintic spectral maps
   applied to the Gram of a Frobenius-normalized matrix.
3. **Per-lane energy equalizer.** Each lane (row or column, whichever is
   longer) is rescaled by the inverse root of an EMA of its mean squared
   update, then the matrix is rescaled back to its pre-equalization Frobenius
   norm. This is NorMuon's low-rank variance estimate.
4. **Cautious weight decay.** The decay is gated on the *slow rail's* sign, not
   the update's.

Per-matrix learning-rate multipliers and frozen matrices are deliberately not
implemented in this phase.

## 2. Algorithm

The host resolves every schedule and hands the kernel one `AnvilParams`. The
kernel then runs, per stacked matrix:

```
V[0] += (1 - fast_beta) * (G - V[0])          # fast rail
V[1] += (1 - slow_beta) * (G - V[1])          # slow rail
blend = fast_weight * V[0] + (1 - fast_weight) * V[1]
g     = (1 - momentum) * G + momentum * blend # Nesterov lookahead

X = ComputeType(g)
A = X^T X (tall) or X X^T (wide)
d = sqrt(trace(A)) * 1.05 + 1e-6
X = X / d ; A = A / (d*d)

for k in 0..num_maps-1:
  if k > 0: A = Gram(X)
  B = b_k * A + c_k * (A @ A)
  X = a_k * X + X @ B          # wide: X = a_k * X + B @ X

lane_power = mean(X*X over red_dim)
pre_norm   = sqrt(sum(lane_power) * lane_len)
E          = beta2 * E + (1-beta2) * lane_power
gain       = rsqrt(max(E, 1e-10))
post_norm  = sqrt(sum(lane_power * lane_len * gain * gain))
eq_scale   = gain * (pre_norm / max(post_norm, 1e-10))
X          = X * eq_scale

aligned = (V[1] * p) >= 0
p = p - p * aligned * lr * weight_decay - lr * X
```

`momentum` is the rail beta (`Scheduler::RailBeta`); `fast_beta` and
`fast_weight` are resolved by the host from the engage step. The cascade
coefficient table `kAnvilMaps[6][3]` is copied verbatim from `anvil.py:36-43`.

The `red_dim` convention matches Muon: `-1` reduces over columns (one lane per
row, `lane_len = cols`), `-2` over rows (one lane per column,
`lane_len = rows`). The default follows the longer extent, so a tall matrix
gets row lanes.

## 3. Weight decay carries `lr` squared

The reference computes
`p -= p * aligned * (wd_mul * wd * lr) * (lr_mul * lr) - update * (lr_mul * lr)`.
The kernel applies `decay = lr * weight_decay * p`, so the host passes the
*unshaped* scheduled learning rate in `weight_decay`:

```
params.lr           = anvil_lr * sqrt(max(1, rows/cols)) * lr_multiplier
params.weight_decay = anvil_weight_decay * anvil_lr * lr_multiplier
```

The product is then `wd * shape_mult * (anvil_lr * lrm)^2`, exactly the
reference's `lr^2` decay. The shape multiplier appears once, through `lr`.

## 4. Rail schedule

`Scheduler::RailBeta(step)` mirrors `get_rail_beta` and, like every other
schedule in this tree, evaluates at the 0-based loop counter `it = step - 1`:

- linear warmup from `rail_beta_min` to `rail_beta_max` over
  `rail_beta_warmup_steps`;
- flat at `rail_beta_max`;
- linear cooldown back to `rail_beta_min` over the last
  `rail_beta_cooldown_steps`.

The cooldown fraction is clamped to `[0, 1]`, so a run shorter than
`rail_beta_cooldown_steps` (reachable with the CAPI default `num_iterations`
of `0`) holds `rail_beta_min` instead of decaying past it; the reference never
reaches that regime.

Before `anvil_engage_step` the kernel sees `fast_beta = RailBeta(step)` and
`fast_weight = 1`; at and after it, `anvil_fast_beta` and
`anvil_fast_weight`. `momentum` is always `RailBeta(step)`.

## 5. Configuration

`OptimizerConfig` gains (additive; the ANVIL numeric defaults match the
reference, while the two selectors stay off so the Muon path is unchanged):

| Field | Default | Meaning |
|---|---|---|
| `matrix_optimizer` | `0` | `0` = Muon, `1` = ANVIL |
| `adam_step_period` | `1` | `1` = AdamW every step, `2` = the reference cadence |
| `anvil_lr` | `0.023` | nominal matrix rate |
| `anvil_weight_decay` | `2.25` | base decay; `lr` carries the outer factor |
| `anvil_momentum` | `0.95` | unused: the rail beta overwrites it |
| `anvil_beta2` | `0.9` | lane-energy EMA decay |
| `anvil_fast_beta` | `0.85` | fast rail beta after the engage step |
| `anvil_slow_beta` | `0.98` | slow rail beta |
| `anvil_fast_weight` | `0.4385` | fast rail weight in the blend |
| `anvil_engage_step` | `514` | 0-based step the blend turns on |
| `anvil_num_maps` | `6` | cascade maps (clamped to 6) |

`SchedulerConfig` gains `rail_beta_warmup_steps` (`240`),
`rail_beta_cooldown_steps` (`50`), `rail_beta_min` (`0.85`), and
`rail_beta_max` (`0.93`).

The default `adam_step_period` of `1` keeps the Muon path bit-identical. A
faithful ANVIL run needs `--adam-step-period 2`: the reference steps AdamW on
its 0-based odd iterations only, which this tree's 1-based `step` sees as the
even steps. The struct default stays `1` because changing it would move the
Muon path's AdamW cadence.

## 6. Grouping and state

There is one optimizer class. It populates either `muon_groups_` or
`anvil_groups_`, never both, from the same ascending shape map. Every matrix
parameter (everything not claimed by an AdamW group) lands in a matrix group.

Each ANVIL group owns:

- `stacked_params` — `[num_params, rows, cols]` `ComputeType`;
- `velocity` — `2 * num_params * rows * cols` floats, fast rail then slow rail;
- `lane_energy` — `num_params * lane_count` floats, where `lane_count` is
  `rows` for `red_dim == -1` and `cols` for `red_dim == -2`.

Both float buffers are allocated with `kernels::Alloc`, zeroed once, and freed
in the destructor.

The AdamW cadence guard evaluates at the 0-based counter,
`adam_step_period <= 1 || ((step - 1) % adam_step_period) == 1`, so the
reference's `is_adam_step` (0-based odd) is the 1-based even `step` here. With
the default period of 1 the bias correction reads the global step, so the Muon
path is bit-identical. With a period above 1 it reads the Adam-update ordinal
`(step - 1) / adam_step_period + 1`, which is a pure function of the global
step, so a resumed run re-derives it exactly.

## 7. Checkpoints

The optimizer-state section gains two records per ANVIL group, fp32, beside the
model parameters:

```
anvil.<rows>x<cols>.velocity     # 2 * num_params * rows * cols
anvil.<rows>x<cols>.lane_energy  # num_params * lane_count
```

`SaveState`, `LoadState`, and `python/nanochat_cpp/checkpoint.py` know the
names. A Muon checkpoint and an ANVIL checkpoint are disjoint: the group sets
never coexist.

## 8. Kernels

`AnvilUpdate` is one seam symbol with three implementations that must agree:

- `backends/cpu/kernels.cc` — the reference (float working state, `Gemm` in
  fp32, a double-accumulator fallback in fp16);
- `backends/cuda/kernels/anvil.cu` — the CUDA family: momentum, the first Gram
  and its normalization, the cascade (batched cuBLAS GEMMs), the equalizer, and
  the apply;
- `backends/cuda/kernels/testing/optim_ref.h` — the host reference the GPU test
  compares against.

The CUDA family caches a workspace of `2 * total + 3 * min_total` `ComputeType`
plus `num_params * lane_count` floats, so a per-step `cudaMalloc` never
dominates the update.

The GPU test `//backends/cuda/kernels:anvil_gpu_test` covers tall, wide, and
square matrices, both `red_dim` values, `num_maps` 0 and 3, both rail states,
Nesterov on and off, and the `rows <= 0` no-op. In fp32 it holds a 1e-5
tolerance; the fp16 build records a looser tolerance because the working matrix
is stored as half. Both builds pass.

The reference casts the working matrix to bfloat16 before the first Gram and
re-rounds it at every cascade map. The fp32 build here deliberately keeps the
whole cascade in fp32 (only the fp16 build rounds through half), so an fp32
curve comparison against `anvil.py` shows small systematic differences that are
not defects.

## 9. Deferred

- The ANVIL oracle fixture. `tools/dump_oracle.py` is untouched in this phase;
  the fixture lands after the first training comparison.
- Per-matrix learning-rate multipliers (the reference's 2x on MLP `c_proj`) and
  frozen matrices.
- Mantissa tracking for the parameter update; the port updates the parameter in
  fp32 (`ComputeType` storage) directly.
