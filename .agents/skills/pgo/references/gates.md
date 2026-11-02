# The correctness gate of one pair

An optimization must not change the numbers. The loop exists to make a step
faster, not to make a different run. Every pair therefore passes one gate before
its measurement counts.

## The gate per stage

| Stage | The gate |
|---|---|
| `base` | the loss trace and `grad_norm` of the steady window agree with the before run; `//tests:oracle_cuda_test` passes |
| `sft` | the loss trace agrees; `//src:sft_test` passes |
| `rl` | the mean reward and the loss agree; `//tests:oracle_cuda_test` passes |
| `eval` | `bpb_base` and `bpb_rl` agree |

Add the T0 tier of the family the patch touched: `tools/nanochat test` for a
workstream, or the single target beside the kernel. When a patch touches a
kernel family, the backend seam, or the device profile, add
`tools/nanochat simulate --device h100`.

## The tolerance comes before the patch

Two of the likely ideas change the order of a floating point reduction. A
reduction on the device sums in a different order than a serial host loop, so the
last bits move. The loop allows that, and the idea declares it first:

- `ideas.json` carries `numeric_tolerance` for each idea, and the analyst
  writes it before the patch exists;
- the gate compares against that tolerance, not against an exact match;
- a change that cannot state a tolerance is not an optimization, and it does not
  enter the loop.

State the tolerance as the tree's own run-to-run spread, and measure that spread
before the pair. The cuBLASLt plan cache times its candidate algorithms and keeps
the fastest (`R2-D4`). The reduction order of a GEMM therefore depends on the
timing of the run, and an unchanged tree differs from itself. On the reference
H100 run the spread over 200 steps was 9.1e-4 relative on the loss and 3.8e-2 on
`grad_norm`. The first step that differed was step 29.

A tolerance of 1e-4 sits below that floor, and no patch can pass it. The gate
reads the floor instead. A pair passes when its drift stays inside the spread of
two runs of the unchanged tree. Its first few steps must also be bit identical.

A change of arithmetic, such as a wider accumulation or a different rounding,
needs its own rung and does not belong in this loop.

## Run it

```bash
.agents/skills/pgo/scripts/compare_runs.py \
    <before>/run-0/demo/base/metrics.jsonl <after>/run-0/demo/base/metrics.jsonl
```

It prints the loss, the gradient norm, and the step time of both runs side by
side, and the ratio of the means. Read the difference against the noise floor
below before you call it a change.

## What the gate reads

The gate compares these fields of `measurement.json` and of the engine's own
records:

- `runs[].step_ms`, against the before point, for a spread the noise rule
  already allows;
- the loss of the last steady step, and the mean loss of the window;
- `grad_norm` of the same window;
- for `eval`, `bpb`.

The first two steps of the before run and the first two of the after run are not
comparable, because they warm up different caches.

## Reverting

When the gate fails, the loop reverts both patches. It does not keep one and drop
the other, because the pair is the unit. The pair joins `loop.json.rejected`, so
a later iteration cannot pick the same two signatures again.

When the gate passes and the measurement misses the bar, the loop reverts both
as well, and records the measured gain with the pair. That record is the evidence
that the ceiling was an upper bound and that this idea is spent.

## What the operator approves

The loop stops at a reviewed working tree. A commit is the operator's decision,
and the message states the pair, the measured gain, and the gate.
