# R1: the fused rotary leg, and the parity gap in its backward

Status: proposed. The composition is the executed path today. This plan works
test first: every change starts with a test that fails, then the smallest change
that passes it.

Revision note: an oracle review revised this document. The first draft named a
scale-squared factor in the backward statistics. Section 5.1 records why the
code rules that factor out, and section 5.2 names the candidates that remain.

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
| R1-T5 | The gate constant is the only selector today. A red run therefore needs a temporary local flip, and the row records the flip. |

The last rule is the awkward one. `kFusedQkPrepMatchesComposition` selects the
composition on every device. A test cannot select the fused pair without a local
edit until R1.4 deletes the constant. The plan keeps the edit explicit, in a
scratch worktree, so no commit carries a flipped gate.

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
| Fused | `backends/cuda/kernels/qk_prep.cu` | One kernel per row for the forward and one for the backward. It accumulates `sum_sq` and `dot` in float, and it computes the norm scale with `rsqrtf`. |
| Composition | `src/ops.cc`, `QkPrepForward` and `QkPrepBackward` | Norm over `RowReduce`, then `RopeForwardHost` and `RopeNormBackwardHost`, which copy the whole activation to the host and back. It accumulates the backward statistics in double, and it computes the norm scale with `1.0f / std::sqrt`. |
| CPU backend | `backends/cpu/kernels.cc`, `QkPrepApplyBackward` | The naive loop. It accumulates in double, like the composition. |
| Kernel test | `backends/cuda/kernels/qk_prep_test.cc` | Compares the fused pair against `dev::rowref`, at `1e-4`. The reference accumulates in double, so it carries the composition's arithmetic. |
| Gate | `src/ops.cc` | `kFusedQkPrepMatchesComposition = false` keeps the composition on every device. |
| Scale | `src/ops.h`, `kQkScale = 1.2f` | Every model shape uses it, so every trajectory run already runs at 1.2. |
| Reference | `nanochat/gpt.py:57, 101-104` | A device op on device tensors, then `q * 1.2` and `k * 1.2`. The scale goes on both tensors, which the seam already matches. |

Two facts follow from the table, and both change the plan.

1. The saved buffers hold the pre-norm projection outputs on every path. The
   seam contract in `qk_prep.cu`, the comment above `RopeNormBackwardHost`, and
   the CPU backend all say so. No path reads a scaled or rotated value in the
   backward.
2. The fused kernel is the only implementation that uses float accumulators.
   The composition, the CPU backend, and the test's own reference use double.

## 4. The evidence

`//tests:oracle_cuda_test` with a local flip of the gate constant:

```
FAIL: step/3/param/backout_lambda: max |diff| = 0.000122 at 0
      (got 0.199736997 want 0.199858665)
oracle_test: max forward error 3.73e-09, max backward error 5.96e-08,
             max optimizer-step-1 error 2.38e-07,
             max trajectory loss error 2.38e-07,
             max trajectory parameter error 0.000122
```

The tolerance is `kTrajectoryParamTolerance = 1e-4`. Every other number agrees
to `6e-8`. With the gate off the same test passes.

The failing case is already a scale of 1.2, because `kQkScale` is 1.2 for every
model shape. The plan needs no new fixture and no new case to name this failure.
`backout_lambda` is a scalar of the smear group, and its true gradient is zero
at this initialization.

This section records two earlier probes, so a later reader does not repeat them.

1. Rounding the forward's normalized value to the compute type before the
   rotation changed the error by nothing. The value was `0.000122` before and
   after, digit for digit. The kernel does that rounding today
   (`QkPrepForwardKernel`), so the forward is already at the composition's
   order.
2. `backends/cuda/kernels/qk_prep_test.cc` passes at `scale = 1.2`. Its bound is
   `1e-4`, which is the size of the divergence. The test cannot separate the two
   arithmetics at that bound.

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

### 5.2 The candidates that remain, in order

| # | Candidate | Why it is the first one | Probe |
|---|---|---|---|
| H1 | The accumulator width and the reciprocal square root. The composition accumulates in double and computes `1.0f / std::sqrt`. The fused kernel accumulates in float and computes `rsqrtf`. | The fused kernel is the only implementation that differs, and the difference is `6e-08` on one step. `backout_lambda` has a true gradient of zero, so an AdamW step has a size of about the group rate whatever the gradient size. Three steps then show a parameter move of about one rate, which is the size of the gap. | A kernel case at the model shape, compared against `dev::rowref` at a bound of `1e-6`. |
| H2 | A second difference, if H1 does not close the gap. | Not known yet. | A per-row dump of `r` and `coeff` from both paths at one batch. |

The test's own reference already accumulates in double. The kernel test can
therefore see H1 today, and only its `1e-4` bound hides it. That is why the
first row of section 6 narrows a bound instead of adding a fixture.

## 6. Work rows

Each row reads left to right: the failing test first, then the change, then the
green gate.

| # | Red test first | Then the change | Green gate |
|---|---|---|---|
| R1.1 | Record the gate failure. Flip `kFusedQkPrepMatchesComposition` in a scratch worktree, run `tools/nanochat test --gpu //tests:oracle_cuda_test`, and paste the failing line and the commit hash into section 4. No change to the tree. | Nothing. R1.4 closes this red line. | The failing line is in section 4, with its command. |
| R1.2 | Add a case to `qk_prep_test.cc` at the model shape: head dimension 128, sequence 64, batch 2, the real rotary base, `scale = 1.2`. Add a named bound `kQkPrepPathAtol = 1e-6`, derived from the storage precision as `docs/testing.md` requires, and pass it to the four checks. | Nothing. The row is red on purpose, and the printed maximum difference names the arithmetic that differs. | The case fails, and the failure is the accumulator difference of H1. If the case does not fail, H1 is wrong: stop, take H2, and amend this section before any kernel change. |
| R1.3 | R1.2, unchanged. | Accumulate `sum_sq` and `dot` in double in `QkPrepBackwardKernel`, and compute the norm scale as `1.0f / sqrt(...)` of the float sum, which is what `RopeNormBackwardHost` does. | `//backends/cuda/kernels:qk_prep_gpu_test` passes at `kQkPrepPathAtol`. |
| R1.4 | R1.1, with the local flip still in place. | Confirm the trajectory passes, then flip the constant in the tree, delete it, and delete the parity note above it in `src/ops.cc`, which still says the kernel keeps the value in fp32 across the rotation. | `//tests:oracle_cuda_test`, `//tests:oracle_test`, and `//backends/cuda/kernels:qk_prep_gpu_test` pass with no local edit. |
| R1.5 | None. A measurement has no test, which R1-T2 allows for this row only. | Re-measure the phase split on the H100 at the 20-layer shape. | `--split-job` reports `fwd_ms` and `bwd_ms` within 1.5 times the roofline rows for the rotary leg. |

R1.3 must not change the forward kernel. The forward already agrees to `3.73e-9`
and it already rounds at the composition's point.

## 7. The rate this unlocks

The rotary leg is 20 launches. Its device bound is 5.5 ms at the 20-layer
sequence 2048 shape, so the fused pair is a device cost and not a host one.
Removing the host staging takes about 6 s out of the 9.04 s step. The plan
predicts about 3 s per step, which gives 1.4 percent MFU. R2 is the next limit.

This section predicts. It sets no gate, and R1.5 reports the measurement that
confirms or rejects it.

## 8. Risks

- H1 may be wrong. The plan spends one cheap test on it, and R1.2 says to stop
  if that test stays green. A probe is cheaper than a guess.
- A tiny shape may hide the difference. R1.2 therefore uses head dimension 128
  and the real rotary base, not the `B=2, T=8, head_dim=4` default of a kernel
  test. The cost stays under 100 ms.
- The new bound may need a bf16 arm. In a bf16 build every golden fixture test
  skips, so `kQkPrepPathAtol` must follow the `kComputeIsBf16` pattern of
  `docs/testing.md` and stay honest in both builds.
- The composition also reads the tables through `HostTable`, one copy per call.
  A fused run removes that copy, so a test that compares only the tables will
  pass while the statistics stay wrong.
- The 20-layer shape needs 80 GB of device memory at batch 16. Confirm the
  memory plan before the measurement run.

## 9. Non-goals

- Folding the rotary leg into the attention kernel. That is `kFeatureFusedAttention`,
  a different row with its own golden trace.
- Changing the rotary convention, the base, or the scale.
- Removing the composition. It is the definition and the CPU path.
