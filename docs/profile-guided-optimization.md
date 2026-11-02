# Profile-guided optimization: the loop, the artifacts, and the work rows

Status: work rows 1, 2, 3, 6, 9, and 10 have landed, and section 12 records
what each one left in the tree. Rows 4, 5, 7, and 8 remain open work.

The first idea that this loop found on the `base`/`h100` target has landed too, as
commit `8b71b0c`, and section 8 records the measurement. The Python surface owns
`pgo.py` and the Modal functions. The Harness owns the two stage loggers of row
Row 4. The architect owns this document, the skill, and the index entry.

This document specifies a loop that measures one stage on one card, reads a CUDA
activity trace, and lands the two largest optimizations that the trace supports.

## 1. Goal

Two facts start the loop. Before the gradient norm moved to the device, the card
was idle for 81 percent of a base step on the H100, so that step was not
arithmetic bound. The record is `profiles/h100-base-20step/REPORT.md`, a local
artifact.
CUPTI does not initialise on the local card under WSL2 and it does initialise in
the Modal container, so the trace must come from the container
([performance.md](performance.md) section 5).

Read [performance.md](performance.md) for the measurement protocol and the
root-cause checklist. Read [pipeline-scaleup-plan.md](pipeline-scaleup-plan.md)
for the two Modal legs and gates G2 and G3.

One command runs one loop for one target. A target is a stage and a device: the
stage is `base`, `sft`, `rl`, or `eval`, and the device is `h100` or `t4`. One
run uses one card on Modal.

The loop does five things. It runs the stage for 50 steps or 3 minutes,
whichever ends first, and it collects a Nsight Systems trace of that run. It
exports the trace to Chrome trace JSON and analyzes it. It names the two
optimizations with the largest measured ceiling, implements both, and measures
the pair. It repeats.

The pair must account for at least 5 percent of the step time, or 5 percent of
the throughput, together. One idea of the pair may be smaller than the bar. The
loop stops at the iteration cap, or when the best remaining pair falls below the
bar.

## 2. What the tree holds today

| Part | Path | State |
|---|---|---|
| One H100 base trace, traced and analyzed | `profiles/h100-base-20step/` | A local artifact, and git ignores it. 20 steps, 190317 device operations |
| The analyzer that produced it | `.agents/skills/pgo/scripts/summarize_trace.py` | The standalone form. `nanochat_cpp.pgo analyze` is the tested form |
| The Modal function that produced it | `tools/modal/app.py`, `profile_h100` | Done for `base` on `h100` only |
| The uniform step record | `include/nanochat/logger.h`, `LogRecord` | Done. `step_ms`, `mfu`, `forward_ms`, `backward_ms`, `optim_ms`, `sync_ms`, `data_ms` |
| The predicted step time | `tools/nanochat roofline` | Done. CPU only, no queue |
| The stage loggers | `src/sft_main.cc`, `src/rl_worker_main.cc` | **Gap.** Both print free text, so only `base` writes `metrics.jsonl` |
| The deterministic loop | `python/nanochat_cpp/pgo.py`, `python/tests/pgo_test.py` | Done. Five commands, five schemas, 29 T0 tests |
| The stage-generic tracer and measurer | `tools/modal/app.py`, `profile_stage` and `measure_stage` | Done. Four stages, both devices, the 180 second cap, one driver |
| The procedure | `.agents/skills/pgo/` | Done. `SKILL.md`, three references, four wrappers, one wrapper test |
| The entry point | `tools/nanochat pgo` | Done. A forwarder, so the loop obeys the one-command rule |
| A Bazel disk cache for the image | `.bazelrc`, `tools/modal/app.py` | **Not possible as first written.** Modal builds an image without a Volume, so the cache can accelerate a run-time rebuild only. Section 2.1 records the correction |

### 2.1 The traps this work already hit

| Trap | Fact |
|---|---|
| A bare `--` before the application | Version 2023.4.4 reads it as an empty long option and rejects the whole line as ambiguous. Put the application directly after the options |
| `--duration` | Absent in 2023.4.4. Enforce the 3 minute cap from the host and accept a non-zero exit that still wrote a trace |
| `cublas_api_sum` | Not a report name in 2023.4.4, and the database holds no cuBLAS table. A host question needs `--sample=cpu --backtrace=fp` |
| The profiler itself | It adds about 28 percent to the step time. Score on the untraced run only |
| `add_local_dir` | It hashes the tree, so each patch invalidates the image. A disk cache Volume is required for this loop to be affordable |
| CUPTI on the local card | Not available under WSL2. Every trace comes from the container |
| A Volume inside an image build | Modal attaches a Volume to a function, not to an image, so `--disk_cache` cannot reach the one Bazel build that an image performs. Row PGO.5 is therefore a run-time cache and not an image cache, and the tree hash still decides when the image rebuilds |
| `--sample` in 2023.4.4 | Valid, and it needs `--backtrace` to name a function. The default trace takes no sample |
| A stage that needs a warm start | `sft`, `rl`, and `eval` run here on a fresh model. The trace measures a throughput, and the shapes do not change |

## 3. The loop

### 3.1 The phases

| Phase | Command | Artifact |
|---|---|---|
| 0 baseline | `tools/nanochat pgo measure --stage base --device h100 --point before` | `before/measurement.json` |
| 1 trace | `tools/nanochat pgo trace --stage base --device h100` | `before/trace/` |
| 2 analyze | `tools/nanochat pgo analyze --stage base --device h100` | `before/analysis.json`, `before/chrome.json` |
| 3 rank | the `pgo-analyst` subagent reads `analysis.json` and the source | `ideas.json`, exactly two ideas |
| 4 implement | two `pgo-implementer` subagents, one worktree each, or one when the file sets overlap | `idea-K/patch.diff`, `evidence.json` |
| 5 verify | one `pgo-verifier` on the patch, then the combined gate on the merged tree | `gate.json` |
| 6 apply and measure | apply both patches, then `measure --point after` | `after/measurement.json` |
| 7 compare and record | `tools/nanochat pgo compare --stage base --device h100` | `verdict.json`, `loop.json` |

`tools/nanochat pgo status --stage S --device D` prints the phase the loop needs
next, so an interrupted session resumes in one command. The `pgo` rows forward
to `python -m nanochat_cpp.pgo`, and the skill's four wrappers forward there too.
`pgo trace` runs `modal run tools/modal/app.py::profile_stage`, and `pgo measure`
runs `::measure_stage`; each one then copies the artifacts home with
`modal volume get`.

The orchestrator runs phases 0, 1, 2, 6, and 7 as scripts, so a number in the
state file never comes from a language model. Phases 3, 4, and 5 are the three
agent roles of section 5.

### 3.2 The measurement protocol

The trace window and the scored window are different, on purpose:

- **The trace** runs 50 steps or 180 seconds, whichever ends first. Fifty steps
  is enough to find the structure, and a short trace keeps the profiler cost and
  the Modal bill small.
- **The scored run** uses the full 180 seconds of budget, untraced, with no
  profiler. Fifty steps is about 20 seconds here, and its median is noise.

The step median is the median of the steady window: the driver's first two steps
and its last one stay out, because they carry start-up and the checkpoint write.

Two runs per measurement point. Take the minimum, because interference only ever
slows a run. The pair wins only when both conditions hold:

1. the gain is at least 5 percent on `step_ms` or on `tokens/s`;
2. the two runs before and the two runs after do not overlap.

Condition 2 is not decoration. The measured steady steps of the reference run
spread from 0.397 to 0.443 seconds, which is 11 percent, and the decision bar is
5 percent. A single run cannot carry a 5 percent decision.

### 3.3 The pair goes in together

The loop applies both ideas as one change and measures once. It does not measure
one idea at a time, so the verdict is the verdict of the pair.

This has three consequences. They are the price of the shorter loop, and each
one has a rule:

- **A regression names no culprit.** The loop reverts both patches, records the
  pair in `loop.json.rejected` with the measured number, and takes the next two
  candidates. It does not split the pair to find the culprit.
- **The two patches must merge.** The analyst states the file set of each idea.
  When the two sets intersect, one implementer writes both ideas as one patch in
  one worktree, and the loop loses nothing but wall clock.
- **The gate reads the merged tree.** The correctness check of section 6 runs
  once, on the tree that carries both patches, and its numbers compare against
  the `before` run.

### 3.4 The ranking and the stop rule

The analyzer emits a signature table with a ceiling for each signature. The
ceiling is the share of the step time that the signature owns, measured from the
trace:

| Signature | Idea shape | Ceiling |
|---|---|---|
| `host_own_code` at a named function | move the work to the device | that host time divided by the step |
| `launches_under_3us` | fuse the pointwise chain, or batch the GEMM | launch host time plus the removed kernel time |
| `blocking_dtoh_scalars` | keep the value on the device | the sum of the drain windows |
| `dtoh_bytes_per_parameter` | reduce on the device | the copy time plus the stall it induces |
| `tiny_gemm_tile` | retile, or one batched call | device time times the efficiency gap |
| `idle_without_host_signature` | none. The trace cannot see it | zero, and the next iteration adds `--sample=cpu` |

The ranking rule:

1. Sort the signatures by ceiling, largest first, and drop any that a previous
   iteration already rejected.
2. Take the top two. Their **pair ceiling** is the sum, clamped to 0.9, because
   no pair of ideas removes a whole step. The measurement decides the rest.
3. Proceed when the pair ceiling is at least 5 percent. An individual ceiling
   may be smaller.

A signature that no idea answers never takes a slot. `idle_unexplained` is the
device time that no stall accounts for, so the next trace must sample the CPU.
`pgo.py` keeps it in the table, reports it as unactionable, and leaves it out of
the ranking.

The ceilings are not additive in truth. They are an upper bound, and the
measurement decides. The loop therefore records both numbers, the predicted pair
ceiling and the measured gain, so a reader sees how good the forecast was.

The loop stops when any of these is true:

- `iteration == max_iterations`;
- the best remaining pair ceiling is below 5 percent;
- two consecutive iterations each gained less than 5 percent.

## 4. The artifact contract

```
profiles/pgo/<stage>-<device>/
  loop.json                     the state: metric history, rejected ideas, spend
  iter-01/
    before/measurement.json     two untraced runs
    before/trace/               base.nsys-rep, timeline CSVs, sums
    before/analysis.json        the signature table with ceilings
    before/chrome.json          the Chrome trace, for Perfetto
    ideas.json                  the pair, with files and ceilings
    idea-1/{patch.diff,evidence.json}
    idea-2/{patch.diff,evidence.json}
    gate.json                   the combined correctness gate
    after/measurement.json      two untraced runs
    verdict.json                kept or reverted, with the noise check
    DECISION.md                 one page: the pair, the numbers, the gate
  REPORT.md                     every iteration, both metrics, one table
```

| Produced by | Path | Consumed by |
|---|---|---|
| `pgo measure` | `measurement.json` | `pgo compare`, the analyst |
| `profile_stage` | `trace/` | `pgo analyze` |
| `pgo analyze` | `analysis.json` | the analyst, `pgo compare` |
| the analyst | `ideas.json` | both implementers, the verifier |
| an implementer | `patch.diff`, `evidence.json` | the verifier, the orchestrator |
| the verifier and the gate | `gate.json` | `pgo compare` |
| `pgo compare` | `verdict.json` | `pgo record` |
| `pgo record` | `loop.json` | the next iteration, and the next session |

The schemas carry a version, and the tests read them:

```json
{"schema": "nanochat.pgo.analysis.v1", "stage": "base", "device": "h100",
 "steps": 50, "wall_ms": 21800.0, "gpu_idle_percent": 81.3,
 "step": {"wall_ms": 436.4, "device_busy_ms": 81.8, "driver_calls": 16178,
          "host_in_cuda_api_ms": 103.6, "host_own_code_ms": 187.9},
 "signatures": [{"id": "host_own_code", "evidence": {"stalls_ms": [51.5, 41.6]},
                 "ceiling": 0.58, "hint": "src/ops.cc:1344"}]}
```

```json
{"schema": "nanochat.pgo.ideas.v1", "iteration": 1, "pair_ceiling": 0.70,
 "ideas": [{"id": "pgo-1", "title": "reduce the gradient norm on the device",
            "signature": "host_own_code", "ceiling": 0.58,
            "files": ["src/ops.cc"],
            "mechanism": "the bf16 branch stages every span to the host",
            "numeric_tolerance": "loss and grad_norm within 1e-4 relative",
            "test_plan": ["//src:optim_test"], "risk": "reduction order"}]}
```

```json
{"schema": "nanochat.pgo.loop.v1",
 "target": {"stage": "base", "device": "h100"},
 "budget": {"trace_steps": 50, "seconds": 180, "measure_runs": 2},
 "threshold": 0.05, "max_iterations": 5, "iteration": 2,
 "baseline": {"step_ms": 0.322, "tokens_per_second": 12700, "mfu": 0.0032},
 "history": [{"iteration": 1, "pair": ["pgo-1", "pgo-2"], "pair_ceiling": 0.70,
              "before": {"step_ms": 0.322}, "after": {"step_ms": 0.114},
              "gain": 0.646, "mfu_after": 0.009, "verdict": "kept"}],
 "rejected": [{"id": "pgo-9", "pair_ceiling": 0.003, "why": "below the bar"}]}
```

## 5. The scheduling graph

```
        measure, two runs ---+
                             +--> trace --> analyze --> rank (the analyst)
        roofline, CPU only --+                             |
                                                          v  ideas.json
                          +-------------------------------+-----------------+
                          v                                                 v
                 impl-1, worktree A                            impl-2, worktree B
                          | patch.diff                                      | patch.diff
                          v                                                 v
                      verify-1                                          verify-2
                          +-------------------------------+-----------------+
                                                          v
                                         merge both, then the combined gate
                                                          v
                                          measure, two runs, then verdict
                                                          v
                                                      loop.json
```

| Role | Kind | Writes | Returns |
|---|---|---|---|
| orchestrator | the main agent | the working tree, `loop.json` | the decision, and the report |
| `pgo-analyst` | `bg_delegate`, inspect only | nothing | `ideas.json`, with exactly two ideas and their file sets |
| `pgo-implementer` | `subagent`, one per idea, `worktree: true` | its own worktree only | `patch.diff`, `evidence.json` |
| `pgo-verifier` | `bg_delegate`, inspect only | nothing | `gate.json` for the patch, and for the merged tree |

The rules that keep the graph safe in this repository:

- An implementer never touches a GPU and never calls Modal. It builds its
  worktree and runs the T0 tier. Measurement stays in the orchestrator, so the
  loop keeps the single-GPU discipline of `AGENTS.md` section 5.

  It never edits a frozen header, `MODULE.bazel`, `.bazelrc`, or another
  workstream's directory. A patch that needs one comes back rejected.
- One writer per worktree, per `AGENTS.md` section 2. Two implementers run in
  parallel only when the analyst's file sets do not intersect.
- The two verifiers run in parallel on the two patches. The combined gate is
  one run, in the orchestrator, because it reads the merged tree.
- No role commits. The orchestrator applies the patches to the working tree and
  asks the operator before a commit.

## 6. The per-stage contract

| Stage | Launch | Scored metric | Correctness gate |
|---|---|---|---|
| `base` | `pipeline.stage_base`, `base_steps=50` | `step_ms`, `tokens/s`, MFU | loss trace and `grad_norm` within tolerance, `//tests:oracle_cuda_test` |
| `sft` | `sft.run(num_iterations=50)` | `step_ms`, `tokens/s`, MFU | loss trace, `//src:sft_test` |
| `rl` | `rl.run(steps=50)` | `rollout_ms`, `optim_ms`, `tokens/s`; MFU of the optimizer phase, named as such | mean reward and loss, `//tests:oracle_cuda_test` |
| `eval` | `pipeline.stage_eval`, in process | `samples/s` and `tokens/s`. No MFU, because no training | `bpb_base` and `bpb_rl` within tolerance |

Two stages need words. `rl` mixes a Python rollout with device work in a worker,
so its trace needs `--sample=cpu` and its MFU describes one phase only. `eval`
has no MFU at all, so the 5 percent bar applies to `samples/s` and the report
must say so.

The gate is a tolerance, not an equality, because two of the likely ideas change
a reduction order. The idea record states its tolerance before the patch exists.
A change that cannot state a tolerance is not an optimization, and it does not
enter the loop.

## 7. Work rows

| # | Work | Gate | Owner |
|---|---|---|---|
| PGO.1 | `python/nanochat_cpp/pgo.py` with `measure`, `analyze`, `compare`, `record`, and the four schemas, plus `python/tests/pgo_test.py` over a synthetic trace | the test passes under `tools/nanochat test`, and the analyzer reproduces the numbers of a trace the loop wrote | Python surface. **Landed** |
| PGO.2 | `profile_stage(stage, device, steps, seconds)` in `tools/modal/app.py`, one image per device, the Chrome trace export, and the 180 second cap | `modal_app_test` passes, and one `base`/`h100` call writes the eight artifacts the analyzer reads | Python surface. **Landed**, for four stages and two devices |
| PGO.3 | The skill: `SKILL.md`, `references/trace-signatures.md`, `references/metrics.md`, `references/gates.md`, and the four thin CLIs | the skill runs one full iteration from a clean checkout | architect. **Landed** |
| PGO.4 | A `LogRecord` per step from `sft_main` and `rl_worker_main`, so all four stages share one metric source | `metrics.jsonl` holds `step_ms` and `mfu` for both, and `//src:harness_test` passes | Harness. **Open.** Until it lands, `sft` is measured by its own seconds and `rl` by its step callbacks |
| PGO.5 | A Bazel disk cache. Modal builds an image without a Volume, so the achievable half is a run-time cache under `--disk_cache` | a one-line source change costs seconds of build inside a function, and the doc records why the image half is out of reach | build, architect. **Open**, and narrowed |
| PGO.6 | `tools/nanochat pgo <action>`, a forwarder to `nanochat_cpp.pgo` | the entry point runs every phase, and a bare `python -m` still works | architect. **Landed** |
| PGO.7 | The first full iteration on `base`/`h100`: the gradient norm on the device, and the blocking scalar reads | the pair gains at least 5 percent on `step_ms`, the gate passes, and `REPORT.md` holds both numbers | the operator. **Half landed:** the gradient norm landed as `8b71b0c`, the scalar reads are open |
| PGO.8 | Widen to `t4`, then to `sft`, `rl`, and `eval` | one iteration each, with the stage's own metric and gate | the operator. **Open** |
| PGO.9 | The index entry in `docs/README.md`, and a cross-reference from `performance.md` section 4 | the two links resolve | architect. **Landed** |
| PGO.10 | A reference report, regenerated by the skill, beside the trace of record | a reader reproduces the numbers from the artifact paths alone | the operator. **Landed** for the trace of record |

Rows 1 to 6 are prerequisites. Row 7 proves the loop on the target whose trace
already exists. Rows 8 to 10 widen it.

`.agents/skills/**` is not in the ownership map of `AGENTS.md` section 1, and
this plan adds it. It holds no compute code: it states the measurement policy
and calls the scripts of row 1, so the architect owns it with this document.

## 8. The measurement

The first trace gives the forecast the loop must beat. One steady base step on
the H100, traced:

| Fact | Value |
|---|---|
| Wall time of one step, traced | 436.4 ms |
| Wall time of one step, untraced | 322 ms, from the engine's own record |
| Device busy | 81.8 ms, 18.7 percent |
| Driver calls | 16178 |
| Host inside a CUDA call | 103.6 ms |
| Host in its own code | 187.9 ms |
| Device to host bytes | 374 MB |
| Kernels shorter than 3 us | 94049 of 190317 in the whole trace |

The analyzer reports this signature table for that step, and
`docs/profile-guided-optimization.md` section 12 records the command that writes
it again:

| Signature | Ceiling |
|---|---|
| `host_own_code` | 0.4306 |
| `driver_call_overhead` | 0.2374 |
| `dtoh_bytes` | 0.0996 |
| `blocking_dtoh_scalars` | 0.0958 |
| `idle_unexplained` | 0.0890 |

The pair ceiling is 0.668, so the loop proceeds. The pair is the host side of the
step twice over: the code the host runs while the device waits, and the driver
calls it sits inside. Those two are disjoint, so their ceilings add.

The first idea of that pair has landed: commit `8b71b0c` moved the gradient norm
to `kernels::GradNorm`, so the 374 MB of host staging per step left the tree. The
second idea the loop took from the same trace is the per-layer blocking scalar
read, and it stays open. Iteration 2 of this target needs a fresh trace of the
current tree, because the signature table has moved.

## 8.1 The first iteration, measured

The loop ran on `base` on the H100 with a cap of one iteration. The tree before
the iteration measured 69.18 ms per step, 59209 tokens per second, and 1.4845
percent MFU.

The trace put the host side of the step first: `driver_call_overhead` 0.3241,
`blocking_dtoh_scalars` 0.3197, `host_own_code` 0.2726, and `tiny_gemm_tile`
0.2172. The pair held two ideas:

1. one staged read per scalar array per pass, in `src/model.cc`, replacing 32
   synchronous copies per step with two;
2. one batched GEMM per head in the grouped-query attention backward, in
   `backends/cuda/kernels/attention.cu`, replacing 2048 launches per step with a
   call per head.

The measurement returned 42.90 ms per step, 95496 tokens per second, and 2.3512
percent MFU. The gain was 38.0 percent of the step and 58.4 percent of the MFU.
The two points of the comparison do not overlap. Idea 1 alone measured 64.76
ms and 1.5704 percent, which is 6.4 percent of the step.

Two things about the tree came out of the iteration and belong in the plan.

The first is the noise floor of the correctness gate. Two runs of the unchanged
tree differ by 9.1e-4 relative on the loss over 200 steps. The cublasLt plan
cache picks its algorithm by timing (`R2-D4`), so the reduction order follows the
run. A declared tolerance of 1e-4 therefore sits below the tree's own spread, and
it fails for every patch. The gate now compares against the measured floor, which
`references/gates.md` of the skill records.

The second is the cost of the profiler. It was 28 percent of an older 322 ms
step, and it was 1.7 times a 69 ms step. The step issues about 16000 driver calls
either way, and each call costs about 3 microseconds to instrument. The overhead
therefore follows the call count, so a faster step carries a larger multiple.
Nothing in the loop compares a traced millisecond with a measured one, which is
why section 3.2 holds.

`iter-01/ideas.json` records the candidate the pair did not take. Both
`ops::SmearForward` and `ops::SmearBackward` run on the host and stage the whole
activation. They own the 25 ms stall of the trace, and their ceiling is 0.2726. A
device path needs a new seam entry or a new `PointwiseOp` code. Such a change
edits `include/nanochat/kernels.h`, a frozen header, so the architect must
approve it before iteration 2 takes it. [scalar-arguments.md](scalar-arguments.md)
carries the related proposal, which removes the two per-layer scalar readbacks
without touching the Smear path.

## 9. Cost and wall clock

| Item | Per iteration |
|---|---|
| Trace run | one container, 180 seconds of budget, about 3 minutes wall |
| Measure runs | two containers, about 3 minutes each |
| GPU time | about 9 to 12 minutes on one H100 SXM, about 1.2 US dollars |
| Build | dominated by the image rebuild until row PGO.5 lands |
| Agent work | one analyst, two implementers, two verifiers, one gate |

Five iterations cost about six dollars of GPU time and one working day of agent
time. The build cost, not the GPU cost, is the reason row 5 is a
prerequisite.

## 10. Risks

- **The 5 percent bar is not above the noise on one run.** Mitigated by the two
  runs, the minimum, and the no-overlap condition of section 3.2. A reader who
  skips those gets a random verdict.
- **A pair hides a regression.** One idea gains 12 percent and the other loses 4
  percent. The pair still passes, and nobody learns. Accepted: the operator
  asked for the shorter loop. The verdict keeps the two predicted ceilings, so a
  reader sees a pair that underperformed its forecast, and `DECISION.md` names
  the pair as the unit.
- **Two patches fight.** Mitigated by the file-set check before the launch, and
  by one implementer when the sets intersect.
- **The ceiling is optimistic.** It is an upper bound from the trace, so the
  measured gain is usually smaller. The loop records both, and it does not treat
  a missed forecast as a failure.
- **A change drifts the numbers.** Mitigated by the tolerance in the idea record
  and the gate of section 6.
- **The image rebuild dominates the loop.** See row 5. Until it lands, the
  loop is build bound, and a reader should expect it.
- **Eval has no MFU.** The bar reads `samples/s` for that stage, and the report
  says so, so a reader does not compare two different numbers.

## 11. Non-goals

- Attribution between two ideas of one pair.
- A search over hyperparameters, batch sizes, or model shapes. The job file, the
  corpus, the seed, and the shape stay fixed across iterations, so the numbers
  stay comparable.
- Multi-device targets. One card, one stage, one Modal container.
- Local GPU measurement. CUPTI does not work there.
- Automatic commits. The loop stops at a reviewed working tree.
- A scheduler for several loops at once. Two loops for two targets may run in
  parallel on Modal, and that is all this document allows.

## 12. What landed

| Path | What it does |
|---|---|
| `python/nanochat_cpp/pgo.py` | The deterministic loop: `init`, `measure`, `trace`, `analyze`, `compare`, `status` |
| `python/tests/pgo_test.py` | 29 T0 tests over a synthetic trace pair: the ceilings, the ranking, the noise rule, the state machine, the commands |
| `tools/modal/app.py` | `profile_stage` and `measure_stage`, the four remote functions, and one stage driver for all four stages |
| `.agents/skills/pgo/` | The procedure, the three references, and the four wrappers |
| `python/nanochat_cpp/_entry.py`, `cli.py` | The `pgo` row of the command table |
| `python/BUILD.bazel` | The `pgo_test` target |

The analyzer reproduces the trace of record, and not only the synthetic
fixtures. Create a target with `pgo init --job <the r2after job>` and copy the
trace into `iter-01/before/trace`:

```bash
tools/nanochat pgo init --stage base --device h100 --job /tmp/mrun/r2after-job.json
tools/nanochat pgo analyze --stage base --device h100
```

It reports `device_busy_percent` 18.7, `host_own_code` 0.4306, and a pair ceiling
of 0.668. Those are the numbers of section 8, from the same CSV files a second
tool wrote, which is the gate of row 1.

Two corrections came out of the implementation, and section 2.1 carries both. A
Volume cannot join an image build, so row 5 is a run-time cache. And a stage
that the chain normally runs after another one runs on a fresh model here,
because the trace measures a throughput and the shapes do not move.

## 13. The measurement of iteration 1

One target, one iteration, one H100 SXM card on Modal:

| Point | Step time | Throughput | MFU | Two runs |
|---|---|---|---|---|
| Before | 69.18 ms | 59211 tokens/s | 1.4845 percent | 70.38, 69.18 ms |
| After, both ideas | 42.90 ms | 95488 tokens/s | 2.3512 percent | 42.90, 43.71 ms |
| After, idea 1 alone | 64.76 ms | 63249 tokens/s | 1.5704 percent | 66.63, 64.76 ms |

The pair gained 38.0 percent of the step and 58.4 percent of the MFU. The
artifacts are in `profiles/pgo/base-h100/`, and `iter-01/DECISION.md` is the one
page record. The trace of record, the pair, the gate, and the verdict all sit
under `iter-01/`.
