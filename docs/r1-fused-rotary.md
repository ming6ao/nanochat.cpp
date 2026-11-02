# R1: the fused rotary leg, and the parity gap in its backward

Status: landed for R1.1 to R1.4, owed for R1.5. The fused rotary leg is the
executed path on a CUDA device that carries the capability bit, and it now
carries the composition's arithmetic operation for operation. Three commits
carry the work:

- `37a3ddc0ea780fc87897bedaea0f8f1cb83414b1`, the campaign base, where the red
  line of section 4.1 was taken;
- `ff3bf0def961000a2cea5c2aec98234a2c5d0cc0`, the model-shape case, the named
  bound, and the kernel's arithmetic;
- `a65e03e9a55eb63b35723dbcdcd79a8e8dba5bf8`, the flip of the gate, the
  deletion of the constant, and the deletion of its parity note.

R1.5, the H100 phase split, has not run: section 6.1 records it as owed, with
the command.

The plan works test first: every change starts with a test that fails, then the
smallest change that passes it. Section 4.2 records the one place where the
predicted red run did not fail, and why the row went ahead anyway.

Revision note: an oracle review revised this document. The first draft named a
scale-squared factor in the backward statistics. Section 5.1 records why the
code rules that factor out, and section 5.2 names the candidates that remained.
A second revision followed the rows: section 4 now carries the commands and the
commits of each capture, section 5.2 records that H1 was half the answer,
section 6 marks what landed, and sections 7 and 8 separate what is measured from
what still rests on the model.

## 1. Goal

Make `kernels::QkPrepForward` and `kernels::QkPrepBackward` carry the same
golden trace as `ops::QkPrepForward` and `ops::QkPrepBackward`, then dispatch
the fused pair. The measured prize is 1.65 s of forward and 4.37 s of backward
per step at the 20-layer shape on an H100.

The method is test driven. A work row writes a test, watches it fail for the
named reason, and then makes the smallest change that passes. A row that cannot
show a red run is not ready to start.

## 2. The working rules

| # | Rule |
|---|---|
| R1-T1 | A work row starts with a test that fails on the parent commit. The review reads the failing line, not a description of it. |
| R1-T2 | The change that passes the test lands in the same row. A row with no red test is allowed only for a deletion and for the measurement. |
| R1-T3 | Red evidence carries its command, its commit hash, and the failing line. A number without those three is a memory. |
| R1-T4 | A gate is narrowed, never widened. R1-D2 is the same rule stated for tolerances. |
| R1-T5 | The gate constant was the only selector while the campaign ran. A red run therefore needed a temporary local flip, and the row records the flip. |

The last rule is the awkward one, and it held until R1.4. The constant
`kFusedQkPrepMatchesComposition` selected the composition on every device. A
test could not select the fused pair without a local edit until R1.4 deleted the
constant. The plan kept the edit explicit, in a scratch worktree, so no commit
carries a flipped gate. R1.4 deleted the constant, so no row needs a flip now;
the rule stays as the record of what R1.1 and R1.4 required.

### 2.1 The decisions that hold

| # | Decision |
|---|---|
| R1-D1 | The composition stays the definition. The fused kernel moves to it, never the reverse. |
| R1-D2 | No tolerance changes. A parity gate is not widened to fit a kernel. |
| R1-D3 | The scale stays on both q and k, because the reference does that. |
| R1-D4 | The gate flips only when `//tests:oracle_cuda_test` passes with it on. |

## 3. What the tree holds

| Part | Path | What it does |
|---|---|---|
| Seam | `include/nanochat/kernels.h` | `QkPrepParams` holds batch, seq, heads, key/value heads, head dimension, epsilon, and scale. The tables are fp32 of shape `[seq, head_dim / 2]`. |
| Fused | `backends/cuda/kernels/qk_prep.cu` | One kernel per row for the forward and one for the backward. The backward accumulates `sum_sq` and `dot` in double, both kernels compute the norm scale as `1.0f / std::sqrt` of the float sum, and every product of the rotation is written with `__fmul_rn` / `__fadd_rn` so nvcc cannot contract it into an FMA. Commit `ff3bf0def961000a2cea5c2aec98234a2c5d0cc0` is the change that made the arithmetic the composition's. |
| Composition | `src/ops.cc`, `QkPrepForward` and `QkPrepBackward` | Norm over `RowReduce`, then `RopeForwardHost` and `RopeNormBackwardHost`, which copy the whole activation to the host and back. It accumulates the backward statistics in double, and it computes the norm scale with `1.0f / std::sqrt`. |
| CPU backend | `backends/cpu/kernels.cc`, `QkPrepApplyBackward` | The naive loop. It accumulates in double, like the composition. |
| Kernel test | `backends/cuda/kernels/qk_prep_test.cc` | Compares the fused pair against `dev::rowref` at `kQkPrepPathAtol`, which is `1e-6` in an fp32 build and `1e-4` in a bf16 one. Three shapes: two tiny ones (`B=2, T=8, head_dim=4`, plus a grouped-query variant) and one model shape (`head_dim` 128, sequence 64, batch 2, the real rotary base). The reference accumulates in double, so it carries the composition's arithmetic. |
| Gate | `src/ops.cc` | The constant `kFusedQkPrepMatchesComposition` is gone. `a65e03e9a55eb63b35723dbcdcd79a8e8dba5bf8` deleted it, so `QkPrepForward` and `QkPrepBackward` dispatch the fused pair on the capability bit alone, and the composition is what a device without the bit runs. |
| Scale | `src/ops.h`, `kQkScale = 1.2f` | Every model shape uses it, so every trajectory run already runs at 1.2. |
| Reference | `nanochat/gpt.py:57, 101-104` | A device op on device tensors, then `q * 1.2` and `k * 1.2`. The scale goes on both tensors, which the seam already matches. |

Two facts follow from the table, and both changed the plan.

1. The saved buffers hold the pre-norm projection outputs on every path. The
   seam contract in `qk_prep.cu`, the comment above `RopeNormBackwardHost`, and
   the CPU backend all say so. No path reads a scaled or rotated value in the
   backward.
2. Every implementation now accumulates the backward's row statistics in double
   and takes the norm scale from `1.0f / std::sqrt`. The fused kernel was the
   only one that did not: it summed in float and called `rsqrtf` until commit
   `ff3bf0def961000a2cea5c2aec98234a2c5d0cc0`. The composition, the CPU
   backend, and the test's own reference always used double. The forward's
   `sum_sq` is the one reduction still in float, and section 5.2 records it as a
   candidate and not a measurement.

## 4. The evidence

Every block below carries its command and its commit, because a number without
them is a memory (R1-T3).

### 4.1 The gate failure at the campaign base (R1.1)

R1-T5's flip, in a scratch worktree that no commit carries and that is removed
after the capture:

```
git worktree add ../r1-scratch-37a3ddc 37a3ddc0ea780fc87897bedaea0f8f1cb83414b1
# src/ops.cc, the one edit, never committed:
#   constexpr bool kFusedQkPrepMatchesComposition = true;
git rev-parse HEAD
37a3ddc0ea780fc87897bedaea0f8f1cb83414b1
tools/nanochat test --gpu //tests:oracle_cuda_test
```

The worktree is detached at the campaign base, so `git rev-parse HEAD` prints
that hash and the tree is the base tree plus the one flipped constant. The run
failed, one line, verbatim:

```
FAIL: step/3/param/backout_lambda: max |diff| = 0.000122 at 0 (got 0.199736997 want 0.199858665)
```

with the summary the test prints above it:

```
oracle_test: max forward error 3.73e-09, max backward error 5.96e-08, max optimizer-step-1 error 2.38e-07, max trajectory loss error 2.38e-07, max trajectory parameter error 0.000122
oracle_test: 1 check(s) failed
```

The tolerance is `kTrajectoryParamTolerance = 1e-4`. Every other number agrees
to `6e-8`. The base commit's own value of the constant is `false`, which is the
green `main` head; the flip is the only difference between that tree and the
one above, and no commit carries it.

The failing case is already a scale of 1.2, because `kQkScale` is 1.2 for every
model shape. The plan needed no new fixture and no new case to name this
failure. `backout_lambda` is a scalar of the smear group, and its true gradient
is zero at this initialization.

### 4.2 The model-shape kernel case (R1.2, R1.3)

R1.2 added a case to `backends/cuda/kernels/qk_prep_test.cc` at the model shape:
head dimension 128, sequence 64, batch 2, two query heads and one key/value
head, the real rotary base 100000, and `scale = 1.2`. It also added a named
bound, `kQkPrepPathAtol`, derived from the storage precision as
`docs/testing.md` requires: `1e-6` for fp32 and `1e-4` for a bf16 build.

R1.2 predicted the case would be red at that bound. It was not. Reproduced at
the base commit with only the new case in the tree (the scratch worktree of
4.1, which is removed after these two captures):

```
git worktree add ../r1-scratch-37a3ddc 37a3ddc0ea780fc87897bedaea0f8f1cb83414b1
# the one edit: backends/cuda/kernels/qk_prep_test.cc, taken from
#   ff3bf0def961000a2cea5c2aec98234a2c5d0cc0
tools/nanochat test --gpu //backends/cuda/kernels:qk_prep_gpu_test
```

At the model shape, against `dev::rowref`, the base kernel printed:

```
  qk fwd q                 max abs error 4.77e-07
  qk fwd k                 max abs error 7.15e-07
  qk bwd dq                max abs error 7.15e-07
  qk bwd dk                max abs error 7.15e-07
qk_prep: all checks passed
```

Every check passed, so the bound could not be the row's red line. The reason is
that `CheckVectorClose` is a relative test: it fails only when a difference
exceeds `tol * (1 + |want|)`, so a nominal `1e-6` is about `2.2e-6` on values
near 1.2, which is above the accumulator difference. The statement H1 needed is
in the other direction. At `ff3bf0def961000a2cea5c2aec98234a2c5d0cc0` the same
four lines read:

```
  qk fwd q                 max abs error 7.15e-07
  qk fwd k                 max abs error 7.15e-07
  qk bwd dq                max abs error 0
  qk bwd dk                max abs error 0
qk_prep: all checks passed
```

The double accumulators moved the backward from `7.15e-07` to bit-identical, and
they did not move the forward at all. H1's confirmation is that pair of numbers,
not a failing check: this is what R1-T3 is for, since a verdict from a bound
that cannot see the difference would have read as "no difference".

The forward's `sum_sq` is still a float reduction (`RowReduceSumSq`) where
`dev::rowref` sums in double, so this case's forward number measures the width
of the forward's row reduction and not the rotation rounding. That is why the
forward's change shows in the trajectory (5.2) and nowhere in this case.

### 4.3 The green trajectory (R1.4)

At `a65e03e9a55eb63b35723dbcdcd79a8e8dba5bf8`, with the fused pair dispatched
and no local edit anywhere, `tools/nanochat test --gpu //tests:oracle_cuda_test`
printed:

```
oracle_test: max forward error 3.73e-09, max backward error 5.96e-08, max optimizer-step-1 error 2.38e-07, max trajectory loss error 2.38e-07, max trajectory parameter error 5.64e-05
oracle_test: all checks passed
```

Each number against its tolerance:

| Quantity | Measured | Tolerance | Fraction of the budget |
|---|---|---|---:|
| Max forward error | 3.73e-09 | `kForwardTolerance = 1e-5` | 0.0004 |
| Max backward error | 5.96e-08 | `kBackwardTolerance = 1e-5` | 0.006 |
| Max optimizer-step-1 error | 2.38e-07 | `kOptimizerTolerance = 1e-5` | 0.024 |
| Max trajectory loss error | 2.38e-07 | `kTrajectoryLossTolerance = 1e-5` | 0.024 |
| Max trajectory parameter error | 5.64e-05 | `kTrajectoryParamTolerance = 1e-4` | 0.56 |

The line that failed at the base commit is the last one:
`step/3/param/backout_lambda` went from `0.000122` to `5.64e-05`, from over the
gate to just over half of it.
The CPU backend on the same fixture, `//tests:oracle_test`, prints
`max trajectory parameter error 1.9e-05` where it runs the composition, so the
fused leg carries about three times the CPU leg's trajectory error and still
sits inside the gate. The retirement of the gate has its evidence beside the
failure that closed it.

This section also records two earlier probes, so a later reader does not repeat
them.

1. Rounding the forward's normalized value to the compute type before the
   rotation changed the error by nothing. The value was `0.000122` before and
   after, digit for digit. The kernel did that rounding already, so the
   composition's rounding point was not the cause; section 5.2 names the two
   operations that were.
2. `backends/cuda/kernels/qk_prep_test.cc` passed at `scale = 1.2` under its
   original `1e-4` bound, which is the size of the divergence: that bound could
   not separate the two arithmetics. R1.2 replaced it with `kQkPrepPathAtol`,
   and section 4.2 shows what the separation looks like at that bound.

The cost of the host leg, measured on the H100:

| Shape | Forward | Backward | Step |
|---|---|---|---|
| 8 layers, hidden 512, sequence 256, batch 16 | 69 ms | 135 ms | 535 ms |
| 20 layers, hidden 1280, sequence 1024, batch 16 | 1648 ms | 4371 ms | 9037 ms |

## 5. The analysis

### 5.1 What the code rules out

The scale-squared factor. Both paths accumulate the sum of squares from the
saved pre-norm rows. Both paths multiply the upstream gradient by `scale` before
they accumulate the dot product. A scale that is present on one side and absent
on the other would show in a single step, and the single-step backward error is
`5.96e-08`. The factor is not there.

### 5.2 The candidates, and what each one cost

| # | Candidate | Why it was first | Probe | Outcome |
|---|---|---|---|---|
| H1 | The accumulator width and the reciprocal square root. The composition accumulates in double and computes `1.0f / std::sqrt`. The fused kernel accumulated in float and computed `rsqrtf`. | It was the one arithmetic difference in the family, and the difference was `6e-08` on one step. `backout_lambda` has a true gradient of zero, so an AdamW step has a size of about the group rate whatever the gradient size. Three steps then show a parameter move of about one rate, which is the size of the gap. | A kernel case at the model shape, compared against `dev::rowref` at a bound of `1e-6` (section 4.2). | Half the answer. The backward moved to bit-identical; the trajectory did not move. |
| H2 | A second difference, if H1 does not close the gap. | Not known yet. | A per-row dump of `r` and `coeff` from both paths at one batch. | Not needed: the numbers named the forward first. No dump was taken. |

H1 was right about the backward and wrong about the trajectory. The two
operations it named moved `r` and `coeff` by about one ulp per row: at the model
shape the backward's agreement with `dev::rowref` went from `7.15e-07` to
bit-identical (4.2). The trajectory stayed at `0.000122`, which is what a
one-ulp change in a near-zero direction looks like after AdamW's `adam_eps`
turns it into a full-size step.

The residual was the forward, and it had two parts:

1. `RsqrtScale`'s `rsqrtf`, which nvcc emits as `rsqrt.approx.f32`, against
   `kRsqrt`'s `1.0f / std::sqrt`, which is the pointwise code the composition
   runs (`backends/cpu/kernels.cc`). The kernels now compute
   `1.0f / std::sqrt(sum_sq * (1 / dim) + eps)`, the composition's operation for
   operation, and the emitted PTX is `rcp.rn` / `mul.rn` / `add.rn` / `sqrt.rn`
   / `rcp.rn`.
2. nvcc contracting `x1 * c + x2 * s` into an FMA. `RopeForwardHost` writes two
   products and an add as plain expressions, so it rounds both products;
   contraction leaves one of them unrounded, and the difference is an ulp at
   every pair. `RotateForwardPair` in `qk_prep.cu` now writes the products with
   `__fmul_rn` and the sum with `__fadd_rn`, and `RotateBackwardFirst`,
   `RotateBackwardSecond`, and `CombineBackward` hold the backward's rotation
   and combine to one rounding per operation for the same reason.

With both in place the trajectory parameter error is `5.64e-05` (4.3), and that
is what let R1.4 delete the gate. Both parts land in commit
`ff3bf0def961000a2cea5c2aec98234a2c5d0cc0`, which is why R1.3's row touched the
forward as well (section 6).

What is left is not measured to be anything yet. The forward's `sum_sq` is still
a float reduction (`RowReduceSumSq`) where the composition sums that row in
double, so the `5.64e-05` that remains is consistent with the forward's
accumulator width. It has not been shown to be that, and section 8 keeps it as
an open risk rather than a conclusion.

## 6. Work rows

Each row reads left to right: the failing test first, then the change, then the
green gate. The last column is the one the campaign added when the rows landed:
what actually happened, in a clause.

| # | Red test first | Then the change | Green gate | Landed |
|---|---|---|---|---|
| R1.1 | Record the gate failure. Flip `kFusedQkPrepMatchesComposition` in a scratch worktree, run `tools/nanochat test --gpu //tests:oracle_cuda_test`, and paste the failing line and the commit hash into section 4. No change to the tree. | Nothing. R1.4 closes this red line. | The failing line is in section 4, with its command. | Landed: the line, its command, and the base commit are section 4.1. |
| R1.2 | Add a case to `qk_prep_test.cc` at the model shape: head dimension 128, sequence 64, batch 2, the real rotary base, `scale = 1.2`. Add a named bound `kQkPrepPathAtol = 1e-6`, derived from the storage precision as `docs/testing.md` requires, and pass it to the four checks. | Nothing. The row is red on purpose, and the printed maximum difference names the arithmetic that differs. | The case fails, and the failure is the accumulator difference of H1. If the case does not fail, H1 is wrong: stop, take H2, and amend this section before any kernel change. | Landed: the case and `kQkPrepPathAtol` are in the tree, and the gate column above did not hold: the case passed at the base commit and H1's difference printed as `7.15e-07` rather than as a failure (4.2). H1 was still right about the backward. |
| R1.3 | R1.2, unchanged. | Accumulate `sum_sq` and `dot` in double in `QkPrepBackwardKernel`, and compute the norm scale as `1.0f / sqrt(...)` of the float sum, which is what `RopeNormBackwardHost` does. | `//backends/cuda/kernels:qk_prep_gpu_test` passes at `kQkPrepPathAtol`. | Landed, and larger than written: `ff3bf0def961000a2cea5c2aec98234a2c5d0cc0` changes both kernels, the backward to double accumulators and the forward to the composition's `rstd` and its one-rounding-per-operation rotation (see below and 5.2). |
| R1.4 | R1.1, with the local flip still in place. | Confirm the trajectory passes, then flip the constant in the tree, delete it, and delete the parity note above it in `src/ops.cc`, which still says the kernel keeps the value in fp32 across the rotation. | `//tests:oracle_cuda_test`, `//tests:oracle_test`, and `//backends/cuda/kernels:qk_prep_gpu_test` pass with no local edit. | Landed: `a65e03e9a55eb63b35723dbcdcd79a8e8dba5bf8` deletes the constant and the note, and section 4.3 is the green run. |
| R1.5 | None. A measurement has no test, which R1-T2 allows for this row only. | Re-measure the phase split on the H100 at the 20-layer shape. | `--split-job` reports `fwd_ms` and `bwd_ms` within 1.5 times the roofline rows for the rotary leg. | Owed: no card that can run it was reachable from this campaign. Section 6.1. |

The claim that R1.3 would not touch the forward kernel is false, and the
`3.73e-09` forward agreement of section 4 was never evidence that the forward
sat at the composition's order. The forward change is real, it lands in the
cuda-kernels row (`ff3bf0def961000a2cea5c2aec98234a2c5d0cc0`,
`backends/cuda/kernels/qk_prep.cu`), and it has two parts: the norm scale
becomes `1.0f / std::sqrt` where `RsqrtScale` called `rsqrtf`, and the pair
rotation is written with `__fmul_rn` / `__fadd_rn` where `RopeForwardHost`
rounds both products and the compiler had contracted one of them into an FMA.
`3.73e-09` is the oracle's whole-model forward check against a fixture at
`1e-5`: a value far below a tolerance says the trajectory was inside the gate,
not that one kernel's rounding matched another's operation for operation. The
number that moved when the forward changed is the trajectory, `0.000122` to
`5.64e-05` (5.2).

### 6.1 R1.5 is owed

The phase split has not run, so this document carries no `fwd_ms`, no `bwd_ms`,
no `opt_ms`, and no `sync_ms` for the rotary leg. The command is:

```
modal run tools/modal/app.py::pipeline_demo \
    --split-job '{"profile":"h100","batch":16,"grad_accum":1,"devices":1}' \
    --split-steps 6
```

with a corpus, by the protocol of `docs/performance.md` section 2 (an idle
device, a warm-up, then the reported statistic named with its configuration);
`--split-corpus` is the argument that supplies it, and
`tools/modal/app.py::pipeline_demo` documents its form.

It cannot run from the host this campaign ran on, and R1-T3 is why that is
written down instead of estimated.

- The host has one GTX 1080 Ti with 11 GB (`sm_61`, fp32), which is not an H100
  and does not fit the shape.
- The Modal client is not installed on the host (`import modal` fails), so the
  entry point that submits the job cannot start.
- The 20-layer shape at batch 16 needs about 80 GB of device memory (section 8),
  which the local card does not have by a factor of seven.

The comparison the gate names is `fwd_ms` and `bwd_ms` against the
`norms_and_pointwise` row of `tools/nanochat roofline` at the same shape -- the
row the rotary leg lands in (`src/performance_model.cc`) -- within 1.5 times it.
`docs/performance-model.md` section 7.3 carries the command form for one row.

## 7. The rate this unlocks

What is measured now:

- The parity gap is closed inside the gate's own tolerance: `5.64e-05` against
  `1e-4`, with the forward at `3.73e-09`, the backward at `5.96e-08`, and the
  first optimizer step at `2.38e-07` (4.3).
- The fused pair is the executed path on a CUDA device with the capability bit,
  so the composition's per-call host staging is out of that path.
- The staging costs what section 4.3's table says on the H100 at the 20-layer
  shape: 1648 ms of forward, 4371 ms of backward, 9037 ms of the step.

What still rests on the model, and not on a measurement:

- The rotary leg is 20 launches, and its device bound is 5.5 ms at the 20-layer
  sequence 2048 shape, so the fused pair is a device cost and not a host one.
- Removing the host staging therefore takes about 6 s out of the 9.04 s step.
- The step becomes about 3 s, which gives 1.4 percent MFU.

No paired step time with the fused pair on has been measured on an H100, so
every number in the second list is a prediction. R1.5 is the row that replaces
them, it is owed (6.1), and this section keeps its prediction standing rather
than borrowing a number from the model and calling it a measurement. R2 is the
next limit.

## 8. Risks

Each risk below is either retired on evidence or kept open with what would close
it.

- **H1 may be wrong.** Retired, and half wrong in a direction the plan did not
  expect: H1 named the backward and the trajectory moved only with the forward
  (5.2). The cheap probe was still worth more than the guess, and no action is
  left.
- **A tiny shape may hide the difference.** Retired. The model-shape case
  exists, at head dimension 128, the real rotary base, and grouped-query
  geometry (4.2), and its printed numbers are what named the two arithmetics.
- **The new bound may need a bf16 arm.** Open. `kQkPrepPathAtol` follows the
  `kComputeIsBf16` pattern of `docs/testing.md` today, `1e-4` in a bf16 build
  and `1e-6` in fp32, but no bf16 build has run the case: this host is `sm_61`
  at fp32, and the bf16 build is the H100 leg. The arm is written, not measured.
- **The composition reads the tables through `HostTable`, one copy per call.**
  Retired. The fused branch passes the device tables straight to the kernel, so
  the copy is out of the executed path, and the case of 4.2 compares the output
  tensors `q`, `k`, `dq`, and `dk` rather than the tables, so a wrong statistic
  cannot hide behind a correct table.
- **The 20-layer shape needs 80 GB of device memory at batch 16.** Open, and it
  is one of the reasons R1.5 is owed (6.1). Confirm the memory plan before the
  measurement run.
- **The forward's `sum_sq` is still a float reduction.** Open, and unmeasured.
  The composition sums that row in double and the fused forward does not, so the
  `5.64e-05` that remains is consistent with the width (5.2). Closing it means
  either measuring it with H2's dump or moving the forward's reduction to
  double as the backward's already is, and neither belongs to a row that has
  landed.

## 9. Non-goals

- Folding the rotary leg into the attention kernel. That is
  `kFeatureFusedAttention`, a different row with its own golden trace.
- Changing the rotary convention, the base, or the scale.
- Removing the composition. It is the definition and the CPU path.
