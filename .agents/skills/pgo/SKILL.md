---
name: pgo
description: Run the profile-guided optimization loop on one training or evaluation stage of nanochat.cpp on one Modal GPU. Use when a stage's step time is slower than expected, when MFU is low, when a change needs a before/after measurement, or when the operator asks to profile, trace, or optimize a stage on H100 or T4.
---

# PGO: the profile-guided optimization loop

One target is one stage on one device: `base`, `sft`, `rl`, or `eval` on `h100`
or `t4`. The loop measures the stage, collects a CUDA activity trace, ranks the
costs the trace measured, lands the two largest, and measures the pair.

One target is one folder under `profiles/pgo/`, and its name carries the shape,
the stage, and the device: `nanochat-d20-base-h100` is the twenty-layer shape on
one H100. Pass that name in `--name`. Without it the folder is
`<stage>-<device>`.

The design, the work rows, and the reasoning are in
[`docs/performance.md`](../../../docs/performance.md) Part 1.
Read it before the first iteration of a new target.

## When to use it

- "Base training on the H100 has low MFU. Find the overhead and fix it."
- "The SFT step is slower than the base step. Why?"
- "Profile `eval` on the T4 and land the two biggest wins."
- "Is this change faster? Measure it before and after."

Do not use it for a single kernel. Use `tools/nanochat profile` and
`docs/performance.md` section 4 for one kernel family.

## Run it

Four phases, one command each. The scripts forward to
`python -m nanochat_cpp.pgo`, so the numbers never come from prose.

```bash
# 0. one target, one frozen job file
tools/nanochat pgo init --name nanochat-d20-base-h100 --stage base \
    --device h100 --iterations 5 --job job.json

# 1. the baseline: two untraced runs, the better of the two stands
.agents/skills/pgo/scripts/measure.py \
    --stage base --device h100 --point before

# 2. the trace: 50 steps or 180 seconds, then the artifacts come home
.agents/skills/pgo/scripts/trace.py \
    --stage base --device h100

# 3. the analysis: one steady step, its stalls, and the signature ceilings
.agents/skills/pgo/scripts/analyze.py \
    --stage base --device h100

# 4. ... the analyst and the implementers do their work here (see below) ...

# 5. the pair: apply both patches, then measure and decide
.agents/skills/pgo/scripts/measure.py \
    --stage base --device h100 --point after
.agents/skills/pgo/scripts/compare.py \
    --stage base --device h100
```

`tools/nanochat pgo status --name nanochat-d20-base-h100 --stage base --device
h100` prints what the loop needs next, so an interrupted session resumes in one
command. `tools/nanochat pgo report` renders `RESULT.md`, which `measure` and
`analyze` refresh on their own.

The state lives in `profiles/pgo/<name>/loop.json`, and one iteration writes
`profiles/pgo/<name>/iter-NN/`.

## The three agent roles

The four scripts are exact. Three decisions are not, and each belongs to one
role:

| Role | Reads | Writes | Bound |
|---|---|---|---|
| analyst | `before/analysis.json`, the source | `iter-NN/ideas.json`: exactly two ideas, each with `files`, `mechanism`, `ceiling`, `numeric_tolerance`, `test_plan` | read only, no GPU |
| implementer | `ideas.json`, one idea | `iter-NN/idea-K/patch.diff`, `evidence.json` | its own worktree, T0 tests only |
| verifier | the patch, its idea | `iter-NN/gate.json` | read only, no GPU |

Rules that the loop depends on:

- The pair is the unit. Both ideas go in together, and one measurement decides
  both. Do not split the pair to attribute a regression; revert both, then take
  the next two candidates.
- An implementer never calls Modal, never touches a GPU, and never edits
  `include/nanochat/kernels.h`, `MODULE.bazel`, `.bazelrc`, or another
  workstream's directory. A patch that needs one comes back rejected.
- When the two ideas share a file, one implementer writes both as one patch.
  Two writers never share a worktree.
- No role commits. The operator approves a commit.

## The bar

The pair must move the metric by at least 5 percent, and the two runs before and
the two runs after must not overlap. One idea of the pair may be smaller than 5
percent. The loop stops at the iteration cap, or when the best remaining pair
has a ceiling below 5 percent, or when two consecutive pairs each gained less
than 5 percent.

A ceiling is a share of one step, measured from the trace. It is an upper bound.
The measurement decides, and a missed forecast is not a failure.

## What to save

| Path | Holds |
|---|---|
| `profiles/pgo/<name>/loop.json` | the state: baseline, history, rejected ideas |
| `.../iter-NN/before/measurement.json` | two untraced runs |
| `.../iter-NN/before/trace/` | the report, the SQLite copy, the timeline CSVs |
| `.../iter-NN/before/chrome.json` | the Chrome trace. Open it in Perfetto |
| `.../iter-NN/before/analysis.json` | the signatures and their ceilings |
| `.../iter-NN/ideas.json` | the pair |
| `.../iter-NN/idea-K/` | one patch, its evidence, its gate |
| `.../iter-NN/after/measurement.json` | two untraced runs |
| `.../iter-NN/verdict.json` | kept or reverted, with the noise check |
| `.../iter-NN/DECISION.md` | one page: the pair, the numbers, the gate |
| `profiles/pgo/<name>/RESULT.md` | the status page: the last number, the history, the command, and the longest kernels |
| `profiles/pgo/<name>/REPORT.md` | every iteration, both metrics |

Git ignores the trace artifacts, because they are large. Keep `profile.log`:
it holds the engine's own per-step lines for the traced run.

## Read the analysis

`analysis.json` holds one steady step, cut out of the trace by the trace's own
step marks. Read `signatures` in order:

- `host_own_code` is host time inside a stall with no driver call in flight.
  The preceding copy in `evidence.stalls` names the buffer that the host then
  works on.
- `blocking_dtoh_scalars` is a scalar read that drains the queue.
- `dtoh_bytes`, `dtod_gather`, `launches_under_3us`, `tiny_gemm_tile` are
  device costs.
- `idle_unexplained` is the part the trace cannot attribute. Collect the next
  trace with `--sample cpu` and read `references/trace-signatures.md`.

## Test it

```bash
cd .agents/skills/pgo/scripts
python3 -m unittest pgo_cli_test

cd ../../../python
PYTHONPATH=. python3 -m unittest tests.pgo_test
```

`tests/pgo_test.py` is the gate of the analysis: it reproduces the ceilings from
a synthetic trace, and it pins the noise rule of the comparison.

## Files

| File | Purpose |
|---|---|
| `scripts/trace.py` | collect one trace on Modal and bring it home |
| `scripts/measure.py` | measure one point, untraced, two runs |
| `scripts/analyze.py` | write `analysis.json` and `chrome.json` |
| `scripts/compare.py` | decide the pair, write `verdict.json`, advance the loop |
| `scripts/pgo_cli_test.py` | the wrapper test |
| `scripts/analyze_step.py` | the same step budget, cut at the optimizer kernel cluster instead of at a loss copy, so it also measures a stage whose step marks nothing |
| `scripts/compare_runs.py` | the numeric gate: loss, gradient norm, and step time of two `metrics.jsonl` files side by side |
| `scripts/summarize_trace.py` | the standalone form of the analyzer that came before `pgo.py analyze`, kept for a trace the loop did not write |
| `references/trace-signatures.md` | each signature, its detection rule, and its fix |
| `references/metrics.md` | the per-stage metric, the noise protocol, the sources |
| `references/gates.md` | the correctness gate for each stage |
