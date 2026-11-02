# R1: the fused rotary leg, and the parity gap in its backward

Status: proposed. The composition is the executed path today. The fused kernel
is one arithmetic difference away from it, and that difference costs seconds per
step at the shapes this chain now runs.

## 1. Goal

Make `kernels::QkPrepForward` and `kernels::QkPrepBackward` carry the same
golden trace as `ops::QkPrepForward` and `ops::QkPrepBackward`, then dispatch
the fused pair. The measured prize is 1.65 s of forward and 4.37 s of backward
per step at the 20-layer shape on an H100.

## 2. What the tree holds

| Part | Path | What it does |
|---|---|---|
| Seam | `include/nanochat/kernels.h` | `QkPrepParams` holds batch, seq, heads, key/value heads, head dimension, epsilon, and scale. The tables are fp32 of shape `[seq, head_dim / 2]`. |
| Fused | `backends/cuda/kernels/qk_prep.cu` | One kernel per row for the forward and one for the backward. |
| Composition | `src/ops.cc`, `QkPrepForward` and `QkPrepBackward` | Norm over `RowReduce`, then `RopeForwardHost` and `RopeNormBackwardHost`, which copy the whole activation to the host and back. |
| Gate | `src/ops.cc` | `kFusedQkPrepMatchesComposition = false` keeps the composition on every device. |
| Reference | `nanochat/gpt.py:57, 101-104` | A device op on device tensors, then `q * 1.2` and `k * 1.2`. The scale goes on both tensors, which the seam already matches. |

## 3. The evidence

`//tests:oracle_cuda_test` with the gate on:

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

Three facts narrow the cause.

1. Rounding the forward's normalized value to the compute type before the
   rotation, which is the composition's order, changed the error by **nothing**.
   The value was `0.000122` before and after, digit for digit.
2. `backends/cuda/kernels/qk_prep_test.cc` passes at `scale = 1.2`, so the
   kernel test does not see the difference. Its tolerance is looser than a
   three-step trajectory.
3. The composition and the fused kernel both read the saved output, which is
   scaled and rotated.

The cost of the host leg, measured on the H100:

| Shape | Forward | Backward | Step |
|---|---|---|---|
| 8 layers, hidden 512, sequence 256, batch 16 | 69 ms | 135 ms | 535 ms |
| 20 layers, hidden 1280, sequence 1024, batch 16 | 1648 ms | 4371 ms | 9037 ms |

## 4. The hypothesis

The backward recounts two statistics from the saved output: the sum of squares
and the dot product with the upstream gradient. That output is the rotated and
**scaled** value, so the sum of squares carries a factor of `scale` squared that
the normalized vector does not have.

`RopeNormBackwardHost` divides the scale out before it accumulates. The fused
kernel accumulates from the scaled values. A scale of 1.2 puts a factor of 1.44
into the reciprocal square root and into the coefficient. That is the right
order for the overshoot on a scalar that moved `1.4e-4`.

The kernel test cannot see it because its oracle is the same sequence reference
for both sides, and the tolerance is a kernel tolerance rather than a trajectory
tolerance.

## 5. Decisions

| # | Decision |
|---|---|
| R1-D1 | The composition stays the definition. The fused kernel moves to it, never the reverse. |
| R1-D2 | No tolerance changes. A parity gate is not widened to fit a kernel. |
| R1-D3 | The scale stays on both q and k, because the reference does that. |
| R1-D4 | The gate flips only when `//tests:oracle_cuda_test` passes with it on. |

## 6. Work rows

| # | Work | Gate |
|---|---|---|
| R1.1 | Add a trajectory case at `scale = 1.2` to the oracle suite, so the divergence is caught by a test that names it rather than by a parameter name in a log | the new case fails with the gate on and passes with it off |
| R1.2 | Divide the scale out of the backward's statistics, in the fused kernel | the new case passes with the gate on |
| R1.3 | Flip `kFusedQkPrepMatchesComposition`, and delete the constant | `//tests:oracle_cuda_test`, `//backends/cuda/kernels:qk_prep_gpu_test` |
| R1.4 | Re-measure the phase split on the H100 at the 20-layer shape | `--split-job` reports `fwd_ms` and `bwd_ms` within 1.5 times the roofline rows for the rotary leg |

## 7. The rate this unlocks

The rotary leg is 20 launches. Its device bound is 5.5 ms at the 20-layer
sequence 2048 shape, so the fused pair is a device cost and not a host one.
Removing the host staging takes about 6 s out of the 9.04 s step. The expected
step is then about 3 s, which is 1.4 percent MFU, and R2 is the next limit.

## 8. Risks

- The backward statistics may not be the whole difference. The forward rounding
  test was cheap and negative, so the next probe must be a comparison of the two
  kernels' `dq` and `dk` on one batch at `scale = 1.2`, not another guess.
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
