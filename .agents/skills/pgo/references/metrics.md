# The metric of a stage, and the protocol that makes it decidable

The loop scores one number, and it scores it on an untraced run. The trace is the
instrument that names the cause, never the ruler.

## The metric per stage

| Stage | Metric | Direction | Where it comes from |
|---|---|---|---|
| `base` | `step_ms` | lower is better | `metrics.jsonl`, `step_seconds`, median of the steady window |
| `sft` | `step_ms` | lower is better | the driver's own seconds over the step count, until work row 4 lands |
| `rl` | `step_ms` | lower is better | the gaps between the `on_step` callbacks |
| `eval` | `samples_per_second` | higher is better | the stage's own seconds over the step count |

The loop records `tokens_per_second` for every stage. It records MFU where the
stage trains, and it records `null` for `eval`, which trains nothing.

Two stages need a word of care:

- `rl` mixes a Python rollout with device work inside a worker. Its `step_ms` is
  the whole step, and its MFU describes the optimizer phase alone. Compare an
  `rl` MFU with another `rl` MFU and with nothing else.
- `eval` has no step. Its unit is one evaluated batch, and its metric is the
  rate of those batches.

## The two windows

The trace takes 50 steps, because the structure of a step does not change and a
short trace keeps the profiler cost small.

The scored run takes 200 steps, which is about 65 seconds of the 180 second cap.
Fifty steps is about 20 seconds, and its median moves by more than the bar.

The first steps of a run carry the library load and the first batch. The last
step carries the checkpoint write. The median leaves both out: `WARMUP_STEPS` is
two, and the metric is the median of the rest.

A stage that reports only its own total seconds has no window to take. It divides
the total by the step count, and `measurement.json` says so through
`source: "stage_seconds"`. Land work row 4 before trusting a small difference on
such a stage.

## Two runs, and the better one stands

Every measurement point is two runs of the same stage in the same container
image, and the value is the better of the two. Interference only ever slows a run,
so the minimum is the better estimate.

The before and the after must then be disjoint: every after run better than every
before run. The reference run measured steady steps from 0.397 to 0.443 seconds
on one configuration, which is 11 percent, and the bar is 5 percent. One run
cannot carry a 5 percent decision, and `compare.py` reverts a gain inside the
noise even when the point estimate clears the bar.

## Why the trace is not the ruler

The profiler costs about 3 microseconds per instrumented driver call. A step
that issues 16,123 calls therefore pays about 48 ms. On the reference H100 run
that was 1.7 times the measured time: 116.7 ms traced against 69.2 ms. On an
older tree the same call count cost only 28 percent, because that step was
322 ms long.

The cost therefore scales with the call count and not with the step time, and it
grows as a step gets faster. A count, a size, and a ratio survive it. An absolute
millisecond does not.

So `measure.py` never passes `nsys`, and `analysis.json` is never compared with
`measurement.json`.

## What stays fixed

Across the iterations of one target, four things stay fixed, so the numbers stay
comparable:

- the job file, which `pgo init` copies into the target's folder;
- the corpus it names;
- the seed;
- the model shape.

A change to any of the four is a new target, and it starts a new loop.

## One run of the loop, once more

1. `measure --point before` runs the stage twice, untraced, and keeps the better.
2. `trace` runs the same stage with the profiler, 50 steps or 180 seconds.
3. `analyze` writes the signatures and the pair.
4. The analyst names two ideas, and the implementers write them.
5. `measure --point after` runs the stage twice on the changed tree.
6. `compare` decides. The pair stays when the gain is at least 5 percent and the
   two points are disjoint.
