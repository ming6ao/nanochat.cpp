# Performance: measurement, the optimization loop, and the CPU backend

This document describes how to measure a performance gap in `nanochat.cpp`, how
to run the optimization loop that follows a measurement, and what the CPU
backend measured. The general method is in the `performance-investigation`
skill. Part 1 is the loop. Part 2 is the CPU backend record.

The host is one GTX 1080 Ti (sm_61) under WSL2. CUPTI is not available. Kernel
counters do not work here. Device-event timing through the project benchmarks is
the primary instrument. Kernel counters need native Linux.

## 1. Measure through the entry point

Run every timing through `tools/nanochat`. The entry point applies the sandbox
and submits the job to the single-GPU queue. Do not run a benchmark binary
directly.

- `tools/nanochat doctor` reports the revision, the toolchain, the device, the
  sandbox limits, and the queue state. Use it before a measurement.
- `tools/nanochat profile [opts]` builds and submits the attention benchmark
  under the `t3-bench` profile. The worker holds the GPU for the job. The
  options are `--json`, `--out PATH`, `--warmup N`, `--iters N`, and
  `--rounds N`.
- `tools/nanochat bench -- <binary> ...` runs another benchmark under the
  `t3-bench` profile.

Record the revision, the build configuration, the precision, and the device
with each number. `tools/nanochat doctor` prints most of them.

## 2. The measurement protocol

1. Confirm the device is idle (`tools/nanochat doctor`).
2. Warm up. Then time a batch with one synchronization at the end.
3. Report the best of several rounds for throughput. Report the median across
   runs for latency. State which statistic you report.
4. Do not set `CUDA_LAUNCH_BLOCKING` for timing.
5. Keep the before and after numbers with their configuration.

## 3. The debugging interface

### 3.1 The attention benchmark

`backends/cuda/kernels/attention_benchmark.cc` times `AttentionForward` and
`AttentionBackward` with CUDA events. The shapes are the `d8_s512` training
shape, a sliding-window variant, a grouped-query variant, the production `SSSL`
mixed-window pattern, and a head-dimension sweep.

The head-dimension sweep shows the redundancy in `OnlineSoftmaxTile` and
`SoftmaxGradTile`. The tile recomputes the query-key dot product for each owned
output dimension. The issued work grows with the square of the head dimension.
The necessary work grows with the head dimension.

A steeper exponent in the sweep confirms the redundancy. A flatter exponent
after a change confirms the fix. See the `roofline` reference in the skill. The
benchmark reports the rate against the necessary operation count. A correct
implementation approaches the device ceiling.

### 3.2 Validate a candidate fix

Do not edit the shipped tile first. Build the candidate under `dev/kernels` with
the same seam signatures. Check it against
`backends/cuda/kernels/testing/sequence_ref.h`. Time it with
`backends/cuda/kernels/testing/bench_utils.h` at the same shapes. Promote it
into `device_utils.cuh` after the correctness check passes and the sweep
exponent flattens. The `summarize_times.py` script and the exponent snippet in
the skill produce the numbers.

### 3.3 Add a benchmark

Follow `backends/cuda/kernels/README.md`. Use
`backends/cuda/kernels/testing/bench_utils.h`:

- `BenchOptions` sets the warm-up, the iterations, the rounds, and the JSON
  output.
- `EventTimer::Time` returns the best mean milliseconds per call. It uses
  device events on the default stream.
- `BenchReport::Add` collects a row.
- `BenchReport::UseDevicePeak` states the denominator of the model FLOPs
  utilization of every row: the running card's declared rate at a precision.
  One call, and no benchmark names a device or a table.
- `BenchReport::SetPeakFlops` states the denominator by hand, in FLOP/s, for a
  card the table does not hold.
- `BenchReport::Print` prints a table. It writes JSON to `options.out` when
  set.

Build the target as a `cuda_binary` next to the code it measures. Do not add a
benchmark to the `all` test suite.

### 3.4 Machine-readable output

`BenchReport` emits the `nanochat.bench.v1` schema:

```json
{
  "schema": "nanochat.bench.v1",
  "device": "NVIDIA H100 80GB HBM3",
  "peak_flops": 989000000000000.0,
  "rows": [
    {"name": "attention_fwd:d8_s512", "shape": "B=8 T=512 H=4 KV=4 D=128 wl=-1", "ms": 12.3, "gflops": 45.6, "work_ratio": 2.0, "mfu": 0.33}
  ]
}
```

`gflops` is the rate against the necessary operation count. `work_ratio` is the
issued work over the necessary work. `device` and `peak_flops` appear only
when the report states a denominator, and `mfu` is the row's `gflops` over that
`peak_flops`, clamped to `[0, 1]`. The share belongs to the precision the
benchmark names, so a bf16 build divides by the tensor-core rate and an fp32
build by the CUDA-core rate ([precision.md](precision.md)). A row that issues
no arithmetic states `0`, and the table prints `-` for it.

In the table, the denominator stands above the header as one comment line, and
the `MFU` column appears only when a row reports a share. A benchmark that
states no peak prints the three-column table of the earlier reports. Feed the
file to the skill's `summarize_times.py`.

## 4. Root-cause checklist

Split the step before you choose a fix:

- Attention forward and backward
  (`backends/cuda/kernels/attention_benchmark.cc`).
- Memory access in the attention tile: uncoalesced key/value loads, a serial
  dot product, and no key/value reuse across query rows.
- Backward key/value atomics, especially grouped-query attention on Pascal.
- GEMM (cuBLAS). Confirm the rate. Do not assume it.
- QkPrep, Pointwise, Classifier, and Embedding.
- The optimizer and the global-norm clip.
- Launch overhead and the backward `Memset` of the query/key/value gradients.
- Host-side gaps: data loading, logging, checkpointing, and evaluation.

Benchmarks to use: `row_benchmark`, `qk_prep_bench`, `decode_fused_bench`,
`attention_benchmark`, and `eval_bench`.

The checklist orders the parts by hand. For the loop that measures the split
from a device trace and lands the two largest items, read
[performance.md](performance.md). The loop writes
its artifacts under `profiles/`, which git ignores: a trace is a measurement of
one host on one day, and it is hundreds of megabytes.

## 5. Profiling limits

CUPTI does not initialize under WSL2. `torch.profiler`, Nsight Systems, and
Nsight Compute cannot attribute device time on this host. The reference profile
report (`profiles/d8_s512/PROFILE_REPORT.md`) contains host operator time only.

Use the CUDA event benchmarks for device time. Profile on native Linux for
occupancy, bandwidth, and warp stall reasons. Record the revision.

## 6. Eval GEMM attribution

A measured eval gap shows one dominant cause. The workload is d12, `B=16`,
`T=1024`, and fp32. The reference is PyTorch nanochat.

| Part | PyTorch | nanochat.cpp | Delta |
|---|---|---|---|
| lm_head GEMM | 92.9 ms | 229.1 ms | +136.2 ms |
| Classifier softcap and cross-entropy | 61.2 ms | 18.5 ms | -42.7 ms |
| Full forward | 686.6 ms | 882.7 ms | +196.1 ms |

The language-model head GEMM was the cause. The shape is `M=16384`, `N=32768`,
and `K=768`.

`cublasGemmStridedBatchedEx` with `CUBLAS_GEMM_DEFAULT` reached 3.6 TFLOP/s.
`cublasSgemm` reached 8.8 TFLOP/s. The two calls take the same operands and the
same flags.

`backends/cuda/gemm.cu` now uses `cublasSgemm` for a single fp32 GEMM. The full
forward dropped to 679 ms. The reference is 687 ms. Reproduce the numbers with
`eval_bench`:

```bash
tools/nanochat build --config=cuda //src:eval_bench
tools/nanochat bench -- bazel-bin/src/eval_bench --batch 16 --seq 1024 \
  --layers 12 --heads 6 --kv-heads 6 --hidden 768 \
  --vocab 32768 --padded-vocab 32768
```

## 7. Baselines

Keep a committed baseline of the `nanochat.bench.v1` output for the production
shape. Then a regression is visible without a new measurement. Update the
baseline only with the revision and the configuration that produced it. The
merge-gate benchmark is one `tools/nanochat profile --json` run at a wave
boundary.

---

## Part 1. The profile-guided optimization loop

_Absorbed from the former `profile-guided-optimization.md`. Its section numbers
are unchanged, so a reference to a section of that document resolves to the
same number here. The execution record is in [campaigns.md](campaigns.md)._

Status: work rows 1, 2, 3, 6, 9, and 10 have landed, and section 12 records
what each one left in the tree. Rows 4, 5, 7, and 8 remain open work.

The first idea that this loop found on the `base`/`h100` target has landed too, as
commit `8b71b0c`, and section 8 records the measurement. The Python surface owns
`pgo.py` and the Modal functions. The Harness owns the two stage loggers of row
Row 4. The architect owns this document, the skill, and the index entry.

This document specifies a loop that measures one stage on one card, reads a CUDA
activity trace, and lands the two largest optimizations that the trace supports.

### 1. Goal

Two facts start the loop. Before the gradient norm moved to the device, the card
was idle for 81 percent of a base step on the H100, so that step was not
arithmetic bound. The record is `profiles/h100-base-20step/REPORT.md`, a local
artifact.
CUPTI does not initialise on the local card under WSL2 and it does initialise in
the Modal container, so the trace must come from the container
([performance.md](performance.md) section 5).

Read [performance.md](performance.md) for the measurement protocol and the
root-cause checklist. Read [post-training.md](post-training.md)
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

### 2. What the tree holds today

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

#### 2.1 The traps this work already hit

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

### 3. The loop

#### 3.1 The phases

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

#### 3.2 The measurement protocol

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

#### 3.3 The pair goes in together

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

#### 3.4 The ranking and the stop rule

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

### 4. The artifact contract

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

#### Retention

A trace is hundreds of megabytes, and it describes one host on one day. Keep
the argument and drop the trace. The record is a few kilobytes, and it is what
the next session reads.

- Keep every iteration's `loop.json`, `REPORT.md`, `DECISION.md`, `ideas.json`,
  `gate.json`, `verdict.json`, `measurement.json`, and `analysis.json`. Never
  delete a record.
- Keep the trace of the two latest iterations of a target. Delete the trace of
  an older iteration: the `before/trace/` and `after/trace/` directories, and
  `before/chrome.json`.
- Delete a target that no report cites.

This command keeps the two latest iterations of one target and deletes the
traces of the others:

```bash
cd profiles/pgo/<target>
ls -d iter-* | sort | head -n -2 |
  xargs -r -I{} rm -rf {}/before/trace {}/before/chrome.json {}/after/trace
```

The Chrome trace is the largest file: `profiles/pgo/base-h100/iter-01` held
150 MB in `before/chrome.json`. The directory `profiles/` held 915 MB in three
targets. The budget for the whole directory is 2 GB. Run the command when the
directory passes that budget, or when a campaign closes.

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

### 5. The scheduling graph

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

### 6. The per-stage contract

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

### 7. Work rows

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

### 8. The measurement

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
`docs/performance.md` Part 1, section 12 records the command that writes
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

### 8.1 The first iteration, measured

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

### 9. Cost and wall clock

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

### 10. Risks

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

### 11. Non-goals

- Attribution between two ideas of one pair.
- A search over hyperparameters, batch sizes, or model shapes. The job file, the
  corpus, the seed, and the shape stay fixed across iterations, so the numbers
  stay comparable.
- Multi-device targets. One card, one stage, one Modal container.
- Local GPU measurement. CUPTI does not work there.
- Automatic commits. The loop stops at a reviewed working tree.
- A scheduler for several loops at once. Two loops for two targets may run in
  parallel on Modal, and that is all this document allows.

### 12. What landed

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

### 13. The measurement of iteration 1

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


---

## Part 2. CPU backend performance

_Absorbed from the former `cpu-performance.md`. The three sections below were
its three parts, and the section numbers inside them are unchanged._

### The measured baseline

This document records a baseline for base training on the CPU. The baseline
compares the PyTorch nanochat reference with the `nanochat.cpp` CPU backend.
Use this baseline to measure a future change to the CPU backend.

The CPU backend in `backends/cpu/kernels.cc` started as a correctness
reference. Phases 1 to 3 of this document added a blocked and
threaded GEMM. They also vectorized the optimizer and the elementwise
families. The scalar baseline and the promoted baseline are both below.

#### Host and configuration

- Host: 12 vCPU under WSL2, 7.8 GiB RAM, one GTX 1080 Ti (not used).
- Sandbox profile: `train` (800% CPU, 8 cores, 6 GiB).
- Model: depth 4, dimension 256, 2 heads, 2 key/value heads, head dimension 128.
- Context length: 512. Window pattern: `L` (full attention).
- Device batch: 2. Total batch: 1024. Gradient accumulation: 1.
- Steps: 10. Warmup: 1. Evaluation, sampling, and saving disabled.
- Parameter count: 36,700,242. Estimated operations per token: 7.195896e7.
- Training tokens per run: 10,240.

#### Procedure

1. Start from the PyTorch reference directory.

   ```bash
   cd ~/repos/nanochat
   ```

2. Set the environment for the PyTorch reference.

   ```bash
   export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
   export TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32
   export WANDB_RUN=dummy WANDB_MODE=disabled
   ```

3. Run the PyTorch reference on the CPU. The sandbox wrapper supplies the
   resource limits.

   ```bash
   ~/repos/nanochat.cpp/tools/nanochat run train -- \
     bash -c 'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; \
     exec .venv/bin/python -m scripts.base_train --device-type=cpu \
       --depth=4 --max-seq-len=512 --window-pattern=L \
       --device-batch-size=2 --total-batch-size=1024 --num-iterations=10 \
       --warmup-steps=1 --eval-every=-1 --core-metric-every=-1 \
       --sample-every=-1 --save-every=-1 --run=dummy --model-tag=cpu_bench_torch'
   ```

4. Run the `nanochat.cpp` CPU backend. Build `//src:train_main` and run the
   binary under the sandbox. Derive the model shape and the horizon with
   `nanochat_cpp.plan` ([python.md](python.md) section 4).

   ```bash
   cd ~/repos/nanochat.cpp
   tools/nanochat build //src:train_main
   tools/nanochat run train -- bazel-bin/src/train_main \
     --train-parquet <train-parquet> --tokenizer <tokenizer.nctoken> \
     --layers 4 --heads 4 --kv-heads 4 --hidden 512 --seq 512 --vocab <vocab> \
     --window-pattern L --batch 2 --grad-accum 1 --num-iterations 10 \
     --warmup-steps 1 --eval-every -1 --save-every -1 \
     --checkpoint <checkpoint-dir>/model.ckpt --log <checkpoint-dir>/train.log
   ```

#### Results

The table gives the mean rate for steps 1 to 9. Step 0 contains the first-run
cost and is not part of the mean. The promoted row gives the best mean rate
over steps 2 to 10 of three measured runs. See "Promoted baseline" below.

| Configuration | Revision | Threads | tokens/s | GFLOP/s | seconds per 10k tokens |
|---|---|---:|---:|---:|---:|
| PyTorch CPU, 8 threads | reference | 8 | 1,299 | 93.5 | 7.7 |
| PyTorch CPU, 1 thread | reference | 1 | 454 | 32.7 | 22.0 |
| `nanochat.cpp` CPU, scalar | `759083e` | 1 | 7.2 | 0.51 | 1,398 |
| `nanochat.cpp` CPU, promoted | `124ceb0` | 8 | 591.1 | 42.5 | 16.9 |

The ratios are:

- PyTorch at 8 threads against the scalar `nanochat.cpp`: about 182 times.
- PyTorch at 1 thread against the scalar `nanochat.cpp`: about 63 times.
- PyTorch at 8 threads against the promoted `nanochat.cpp`: about 2.2 times.
- The promoted `nanochat.cpp` against the scalar `nanochat.cpp`: about 82
  times.

#### Promoted baseline (Phase 4)

Phase 4 of this document promotes the Phase 1 to 3 changes and
records the new baseline. The measurement follows the protocol in Cause, design, and plan, section 9.
The device was idle. The GPU had no compute process. The `nanochat.slice`
sandbox had no task.

One warmup run precedes three measured runs. Each run is the bridge command in
step 4 above under the `train` profile. The profile gives 8 CPU threads. The
rate is the mean over steps 2 to 10. Step 1 is not part of the mean, because
it contains the first-run cost.

| Run | tokens/s | seconds per step |
|---|---:|---:|
| warmup | 588.3 | 1.741 |
| run 1 | 582.3 | 1.759 |
| run 2 | 591.1 | 1.732 |
| run 3 | 585.4 | 1.749 |

The best-of-N rate is 591.1 tokens/s at 8 threads. The step time is 1.732
seconds. The gate is at most 5 seconds per step. The gate result is pass.

The end-to-end speedup against the 7.2 tokens/s baseline is 82.1 times.

The revision is `124ceb0` on branch `sliceme/agent-cpu-backend`. The raw log is
`/tmp/cpu-gap-p4b-run2.log`. The machine-readable value is
`/tmp/cpu-gap-p4b-e2e.json`.

The three correctness tests in section 8 pass:

```bash
tools/nanochat test --nocache_test_results \
  //backends/cpu:kernels_test //tests:oracle_test //tests:train_parity_test
```

The log is `/tmp/cpu-gap-p4-section8.log`.

#### Findings

The findings below describe the scalar baseline at revision `759083e`.

1. `Gemm` in `backends/cpu/kernels.cc` is a scalar triple loop with `double`
   accumulation. It has no tiling, no single-instruction-multiple-data
   vectorization, and no threading. The whole run uses one core. The measured
   `user` time is 23m35s and the measured `real` time is 23m48s.
2. The build already uses `-O2` (`nanochat.bzl`). The gap is algorithmic. It is
   not a missing compiler flag.
3. The 63-times single-thread gap comes from the vectorized and cache-blocked
   linear algebra library in PyTorch. That library uses a scalar loop as its
   comparison.
4. The remaining 2.86 times comes from PyTorch thread scaling across 8 cores.
5. Set `MKL_NUM_THREADS` together with `OMP_NUM_THREADS`. Without that setting,
   the first run reached only 1,049 tokens/s, because the math library started
   more threads than the 8-core cgroup allows.

#### Revisions

Record the revision of this repository and the revision of the reference
repository with every new measurement. The scalar baseline comes from
`nanochat.cpp` revision `759083e` on branch `cpu`. The promoted baseline comes
from `nanochat.cpp` revision `124ceb0` on branch `sliceme/agent-cpu-backend`.
The reference is the PyTorch nanochat checkout.

#### Raw data

The summary data is the `cpu_train` entry of [baselines.json](baselines.json).
The raw logs are
in `/tmp/`:

| File | Content |
|---|---|
| `cpp_10step.log` | The scalar `nanochat.cpp` CPU run. |
| `torch_t1*.log` | The PyTorch runs with one thread. |
| `torch_t8*.log` | The PyTorch runs with eight threads. |
| `torch_10step.log` | The first PyTorch run, with `MKL_NUM_THREADS` unset. |
| `cpu-gap-p4-run*.log` | The promoted `nanochat.cpp` runs. |
| `cpu-gap-p4b-run*.log` | The Phase 4 re-measurement runs. |
| `cpu-gap-p4-section8.log` | The section 8 correctness tests. |

### Cause, design, and plan

This document turns the CPU gap analysis into a work order. It states the
measured problem, the design of the fix, the work phases, the owner of each
phase, and the numeric gate that closes it.

Status at writing: a proposal. No code change existed then, and the analysis had
measurements but no measured predictions. The phase results record them below.

Related documents:

- The measured baseline, above.
- Sections 1 to 7 above: the measurement protocol.
- [testing.md](testing.md): the test tiers and the definition of done.
- [build.md](build.md): backend selection.
- [DESIGN.md](../DESIGN.md) section 2: the dependency rules.
- [AGENTS.md](../AGENTS.md): the ownership map.

---

#### 1. Purpose

The CPU backend is the correctness reference. It is also 182 times slower than
the PyTorch reference for base training. This plan closes most of that gap
without a third-party runtime dependency.

The plan has stages. Phase 1 is the critical change and carries a kill gate.
The later phases are optional.

---

#### 2. The measured problem

The gap and the configuration are in The measured baseline, above.

| Stack | tokens/s | seconds per step |
|---|---:|---:|
| PyTorch reference, 8 threads | 1,299 | 0.79 |
| nanochat.cpp, CPU backend | 7.2 | 143 |

The ratio is 182 times.

The table gives the cost of each family for one training step. Each number is a
measured best-of-N.

| Component | ms per step | Share | Rate |
|---|---:|---:|---:|
| lm_head dgrad GEMM | 72,604 | 51% | 0.24 GFLOP/s |
| lm_head wgrad GEMM | 22,953 | 16% | 0.75 GFLOP/s |
| lm_head forward GEMM | 17,598 | 12% | 0.98 GFLOP/s |
| Muon optimizer | 12,673 | 9% | |
| MLP GEMMs, forward and backward | ~13,000 | 9% | ~1 GFLOP/s |
| Classifier softcap and cross-entropy | 2,304 | 1.6% | 0.17 GFLOP/s |
| Attention, forward and backward | ~1,550 | 1.1% | ~3 GFLOP/s |
| AdamW | 135 | 0.1% | 2.2 GFLOP/s |
| Total | ~143,000 | 100% | |

The three language-model head GEMMs alone are 113 s. The shapes are:

| GEMM | M | N | K | Note |
|---|---:|---:|---:|---|
| forward | 1024 | 32768 | 256 | |
| wgrad | 32768 | 256 | 1024 | transposed A |
| dgrad | 1024 | 256 | 32768 | worst access pattern |

---

#### 3. The cause

`Gemm` in `backends/cpu/kernels.cc` is a scalar triple loop. The loop order is
`i`, `j`, `l`. The code reads operand `b` as `b[l * ldb + j]`. The inner loop
steps `l`, so each load lands on a different cache line. The `dgrad` shape has
the worst pattern and runs at 0.24 GFLOP/s.

The same code has no threads and no single-instruction-multiple-data
vectorization.

A first try used `gprof`. The flat profile blamed `AttentionForward` for 89% of
the time. The direct measurement showed that attention is 1% of the step. The
`-O2` inlining breaks the `gprof` call graph. Do not use `gprof` for this
backend. Use the microbenchmark in section 12.

---

#### 4. Design

##### 4.1 The GEMM change

Change only the body of `kernels::Gemm`. Keep the signature and the operand
semantics. The new body does this:

1. Walk the output rows `i` in the outer loop.
2. Hold one accumulator row of `n` values in `double`.
3. For each reduction step `l`, broadcast `a` and accumulate the whole row of
   `b` into the accumulator.
4. Write the accumulator row to `c` at the end.

This order makes the `b` read contiguous in `j`. It removes the cache-line
stride of the shipped order. The `a` read becomes a scalar broadcast.

The `double` accumulator stays. It is the arithmetic of the shipped code. It
avoids the parity risk in section 4.3.

##### 4.2 Threading contract

Parallelize the output-row loop with OpenMP. `docs/build.md` already lists
OpenMP as an allowed CPU dependency.

The thread count comes from the environment:

- `python/nanochat_cpp/sandbox.py` sets `OMP_NUM_THREADS` for every profile.
- The same wrapper sets `NANOCHAT_NUM_THREADS` for library thread pools.
- If neither variable exists, OpenMP uses the hardware concurrency.

Do not hardcode a thread count. Do not create threads per `Gemm` call. Let
OpenMP manage the pool.

The build must add `-fopenmp` to the CPU backend. The file is
`backends/cpu/BUILD.bazel`, which the Runtime workstream owns.

##### 4.3 Numerics

The `double` accumulator keeps the summation precision of the shipped code.
The reordered summation changes the rounding slightly, but at `double`
precision the difference is negligible.

Do not switch to an fp32 accumulator in Phase 1. At `K=4096` the fp32
accumulator differs from the shipped code by 5.7e-5. The oracle tolerance is
1e-5. An fp32 accumulator needs a full parity check first.

##### 4.4 Later stages

Phase 2 adds register blocking and operand packing. This is the path to the
OpenBLAS rate.

Phase 3 accelerates `MuonUpdate`, the classifier, and attention. These become
the largest items after Phase 1.

---

#### 5. Ownership and interfaces

The frozen seam does not change. `include/nanochat/kernels.h` keeps every
signature. The public API and the model do not change.

| Item | Owner (directory) |
|---|---|
| `kernels::Gemm` body and `backends/cpu/BUILD.bazel` | Runtime (`backends/cpu/`) |
| `MuonUpdate` body | Runtime (`backends/cpu/`) |
| Classifier and attention bodies | Runtime (`backends/cpu/`) |
| CPU microbenchmark and baseline JSON | Kernel agents (`dev/kernels/`) |
| This document and the baseline documents | Architect (`docs/`) |

One writer per file. Coordinate before a change to a shared file. See
[AGENTS.md](../AGENTS.md) section 1.

---

#### 6. Plan

Every phase runs through `tools/nanochat`. Tests use the T0 CPU profile.
Benchmarks use the `train` or `t3-bench` profile. See
[performance.md](performance.md) and [testing.md](testing.md).

##### Phase 0 — benchmark harness (0.5 session)

- Promote the microbenchmark to `dev/kernels/cpu_gemm_bench.cc`. Time the four
  dominant shapes. Report the `nanochat.bench.v1` schema.
- Add a `cc_binary` target to `dev/kernels/BUILD.bazel`.
- Record the shipped numbers in the `cpu_train` entry of `docs/baselines.json`.
- **Gate:** the benchmark reproduces the section 2 numbers within 10%.

##### Phase 1 — reordered, threaded GEMM (1 session, the critical gate)

- Rewrite the `kernels::Gemm` body as in section 4.1.
- Add OpenMP to `backends/cpu/BUILD.bazel`.
- Keep the `double` accumulator.
- **Kernel gate:** each of the four shapes must reach at least 15 GFLOP/s. The
  measured prediction is 23.8 to 28.0.
- **Correctness gate:** the three tests in section 8 pass.
- **End-to-end gate:** the 10-step CPU run must improve by at least 5 times
  (143 s per step to 29 s or less). The prediction is 19.4 s.
- If the kernel gate fails, stop and report the negative result. Do not
  continue to Phase 2.

##### Phase 2 — blocking and packing (2 sessions)

- Add register blocking and operand packing to `Gemm`.
- Keep the parity gate green after each change.
- **Kernel gate:** the `lm_head` `dgrad` shape must reach at least 80 GFLOP/s.
  The measured OpenBLAS ceiling is 414.
- Consider an fp32 accumulator only with the full parity check. Keep the
  `double` path if parity fails.

##### Phase 3 — optimizer and elementwise families (2 sessions)

- Thread and vectorize `MuonUpdate`. **Gate:** at most 2 s per step. The
  baseline is 12.7 s.
- Vectorize `ClassifierForward` and `ClassifierBackward`. **Gate:** at most
  0.5 s per step. The baseline is 2.3 s.
- Vectorize `AttentionForward` and `AttentionBackward`. **Gate:** at most
  0.4 s per step. The baseline is 1.6 s.
- **End-to-end gate:** the 10-step CPU run must reach at most 5 s per step.

##### Phase 4 — promote and record (0.5 session)

- Re-measure with the protocol in section 9.
- Update `docs/baselines.json` and `docs/performance.md` with the revision.
- **Gate:** the end-to-end step is at most 5 s, and every correctness test in
  section 8 passes.

---

#### 7. Acceptance criteria

| Criterion | Baseline | Target | Stretch |
|---|---:|---:|---:|
| lm_head dgrad rate | 0.24 GFLOP/s | 15 | 80 |
| lm_head forward rate | 0.98 GFLOP/s | 15 | 80 |
| lm_head wgrad rate | 0.75 GFLOP/s | 15 | 80 |
| Muon per step | 12.7 s | 12.7 (unchanged) | 2 |
| Classifier per step | 2.3 s | 2.3 (unchanged) | 0.5 |
| Attention per step | 1.6 s | 1.6 (unchanged) | 0.4 |
| Step time | 143 s | 29 | 5 |
| End-to-end speedup | 1x | 5x | 28x |

The PyTorch reference is 0.79 s per step. The stretch target is 6.3 times the
reference.

---

#### 8. Correctness gates

Run these tests after every phase:

- `tools/nanochat test //backends/cpu:cpu_kernels_test`
- `tools/nanochat test //tests:oracle_test`
- `tools/nanochat test //tests:train_parity_test`

The `train_parity` gate is the strict one. It compares the loss, the gradient,
and the parameter trajectory against the PyTorch reference. A fast wrong answer
is not progress.

Also run the full CPU suite before a merge:

- `tools/nanochat test`

---

#### 9. Measurement protocol

1. Record the revision with `tools/nanochat doctor`.
2. Confirm the device is idle and no other CPU job runs.
3. Warm up. Then time a batch with the best-of-N rule.
4. Report the best round for throughput. State the thread count.
5. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` together for the PyTorch side.
   Without the second setting the first run reached only 1,049 tokens/s.
6. Keep the before and after numbers with the revision and the configuration.

---

#### 10. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| The reordered `Gemm` is slower than the shipped loop | The shipped loop may vectorize in a rare case | Phase 1 kernel gate; revert if it fails. |
| The fp32 accumulator breaks parity | The oracle tolerance is 1e-5 | Keep the `double` accumulator; validate before any fp32 change. |
| OpenMP oversubscribes the cgroup | The sandbox caps the CPU budget | Use the wrapper's thread count; keep `OMP_NUM_THREADS` as the source. |
| A hand-written GEMM never reaches BLAS | Register blocking and packing are hard | Phase 2 target is 80 GFLOP/s, not 414; a BLAS link is the fallback. |
| Muon and the elementwise families dominate after Phase 1 | The step stays slow | Phase 3 covers them; the Phase 1 end-to-end gate shows the size. |
| The CPU backend is a correctness reference | A change may weaken the oracle | Every phase keeps the section 8 tests green. |
| Concurrent agents edit the same files | Conflicting changes | One writer per file; the ownership map in section 5. |

---

#### 11. Alternatives considered and rejected

- **Link a CPU BLAS as the first step.** A BLAS reaches 258 to 414 GFLOP/s.
  It also adds a third-party runtime dependency, which the design avoids.
  Keep it as the Phase 2 fallback, not the first step.
- **Use an fp32 accumulator immediately.** It is faster on paper. It changes
  the rounding by up to 5.7e-5 and risks the oracle. Rejected for Phase 1.
- **Parallelize without reordering the loops.** Threads alone give only 3.4
  times at `dgrad`. The loop order is the main cause. Rejected.
- **Use `gprof` for the attribution.** It misattributes the time under `-O2`.
  Rejected. Use the direct microbenchmark.
- **Add a framework or a code generator.** The design forbids a framework
  dependency and an autograd engine. Rejected.

---

#### 12. Reproduction

The microbenchmark sources are in `/tmp/nanochat-cpu-bench/`. Compile and run
them against the CPU backend:

```bash
cd ~/repos/nanochat.cpp
g++ -O2 -std=c++20 -fopenmp -DNANOCHAT_PRECISION=0 -Iinclude -I. \
  backends/cpu/kernels.cc /tmp/nanochat-cpu-bench/gemm_omp.cc \
  -o /tmp/nanochat-cpu-bench/gemm_omp
tools/nanochat run train -- /tmp/nanochat-cpu-bench/gemm_omp
```

The end-to-end run is the bridge command in The measured baseline, step 4.

---

#### 13. References

- The shipped GEMM: `backends/cpu/kernels.cc`, function `Gemm`.
- The kernel seam: `include/nanochat/kernels.h`, function `Gemm`.
- The measurement protocol: [performance.md](performance.md).
- The test tiers: [testing.md](testing.md).
- The backend rules: [build.md](build.md), [DESIGN.md](../DESIGN.md).
- The ownership map: [AGENTS.md](../AGENTS.md).
- The measured baseline, above.

### Phase results

Status: **Phase 1 PASS, Phase 2 PASS, Phase 3 PASS.** Phase 1 passes the kernel
gate, the correctness gate, and the end-to-end gate. The 10-step CPU run reaches
22.3 seconds per step. The target is 29 seconds. The speedup is 6.4 times.
Phase 2 passes the kernel gate and keeps the parity gate green (section 7).
Phase 3 passes the three family gates, the end-to-end gate at 1.733 seconds per
step, and the full CPU suite (section 8).

This part records the Phase 1 kill gate from the plan in Cause, design, and plan, section 6. The gate has three parts: the four kernel rates, the three
correctness tests, and the end-to-end step time. The gate also covers the
model-layout forward shape (`transpose_b = true`).

#### Configuration

| Item | Value |
|---|---|
| Revision | `0bc2fad`, branch `sliceme/agent-cpu-backend` |
| Host | 12 vCPU under WSL2, 7.8 GiB RAM, one GTX 1080 Ti (not used) |
| Kernel profile | `t3-bench` (12 threads) |
| Bridge profile | `train` (8 threads) |
| Precision | fp32 |
| Build | OpenMP enabled in `backends/cpu/BUILD.bazel` and the CPU `Gemm` body |
| Model | depth 4, dimension 256, 2 heads, 2 key/value heads, head dimension 128 |
| Context | 512, window pattern `L`, device batch 2, total batch 1024 |
| Steps | 10, warmup 1, evaluation, sampling, and saving disabled |

#### 1. Kernel gate — PASS

Command:

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gap-p1.json
```

The table gives the acceptance run (`--rounds 3 --warmup 1`). The gate is
15 GFLOP/s. The model-layout forward row (`gemm_forward_tb1`) and the
model-layout MLP row (`mlp_gemm_tb1`) use `transpose_b = true`, the layout that
`ops::LinearForward` stores.

| Shape | Layout | Rate (GFLOP/s) | Result |
|---|---|---:|---|
| `gemm_forward` | `tb=0` | 19.29 | pass |
| `gemm_forward_tb1` | `tb=1` (model) | 18.97 | pass |
| `gemm_wgrad` | `ta=1` | 17.37 | pass |
| `gemm_dgrad` | `tb=0` | 19.73 | pass |
| `mlp_gemm` | `tb=0` | 22.90 | pass |
| `mlp_gemm_tb1` | `tb=1` (model) | 22.32 | pass |

A warmed run confirms the numbers. The command is `--rounds 10 --warmup 2`.
The warmed rates are `gemm_forward` 19.42, `gemm_forward_tb1` 19.34,
`gemm_wgrad` 17.32, `gemm_dgrad` 19.36, `mlp_gemm` 21.13, and `mlp_gemm_tb1`
21.22 GFLOP/s.

The operand-layout probe shows that the two layouts now agree:

| Shape | Layout | Rate (GFLOP/s) |
|---|---|---:|
| `lm_head` forward | `transpose_b = true` (model) | 19.35 |
| `lm_head` forward | `transpose_b = false` | 19.55 |
| MLP forward | `transpose_b = true` (model) | 21.90 |
| MLP forward | `transpose_b = false` | 21.28 |

The probe command writes `/tmp/cpu-gap-p1-layout.json`. The warmed benchmark
writes `/tmp/cpu-gap-p1-kernel.json`.

#### 2. Correctness gate — PASS

Command:

```bash
tools/nanochat test --nocache_test_results \
  //backends/cpu:kernels_test //tests:oracle_test //tests:train_parity_test
```

All three tests pass. `train_parity_test` is the strict gate. It compares the
loss, the gradient, and the parameter trajectory against the PyTorch
reference.

The header `include/nanochat/kernels.h` is unchanged. The read-only check
`git diff --quiet -- include/nanochat/kernels.h` returns 0.

#### 3. End-to-end gate — PASS

Command (the bridge in The measured baseline, step 4):

```bash
tools/nanochat run train -- bazel-bin/src/train_main \
  --train-parquet <train-parquet> --tokenizer <tokenizer.nctoken> \
  --layers 4 --heads 4 --kv-heads 4 --hidden 512 --seq 512 --vocab <vocab> \
  --window-pattern L --batch 2 --grad-accum 1 --num-iterations 10 \
  --warmup-steps 1 --eval-every -1 --save-every -1 \
  --checkpoint <checkpoint-dir>/model.ckpt --log <checkpoint-dir>/train.log
```

The bridge runs under the `train` profile (8 cores). The mean rate over steps 2
to 10 is 45.91 tokens/s. Step 1 is not part of the mean, because it contains
the first-run cost.

| Metric | Baseline (`759083e`) | Phase 1 (`0bc2fad`) | Target | Result |
|---|---:|---:|---:|---|
| tokens/s | 7.2 | 45.91 | — | — |
| seconds per step | 143 | 22.3 | 29 | pass |
| speedup | 1.0 | 6.4 | 5.0 | pass |

The machine-readable value is `/tmp/cpu-gap-p1-e2e.json`:
`{"seconds_per_step": 22.3}`.

The loss values match the earlier Phase 1 run step for step. The model-layout
fix changes only the iteration order, so the numerics are identical.

#### 4. Root cause and fix

The first Phase 1 run failed the end-to-end gate at 44.1 seconds per step. The
`Gemm` reorder read `b` contiguously only when `transpose_b` is false. The
model does not use that layout. `ops::LinearForward` in `src/ops.cc` stores
weights as `[out, in]` and sets `params.transpose_b = true`. The
`transpose_b` branch read `b[j * ldb + l]`, which strides by `k` elements. That
stride is the cache-line stride of the shipped loop.

The fix rewrites the `transpose_b = true` branch as a dot product. The loop
walks `b[j, l]` and `a[i, l]` along `l`, so both reads are contiguous. The
`double` accumulator, the signature, and the operand semantics stay the same.

The `transpose_b = true` model forward rate rose from 0.69 to 19.35 GFLOP/s.
The first failed run is recorded in the revision history of this file.

#### 5. Verdict

**Phase 1 kill gate: PASS.**

- Kernel gate: pass. All four design shapes and both model-layout shapes
  exceed 15 GFLOP/s.
- Correctness gate: pass. The three tests pass and the frozen seam is
  unchanged.
- End-to-end gate: pass. 22.3 seconds per step against the 29-second target.
  The speedup is 6.4 times, not 5 times.

Promote the Phase 1 `Gemm` and continue to Phase 2. Phase 2 must keep the
model-layout rows in `cpu_gemm_bench` and keep the section 8 tests green.

#### 6. Reproduction

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench \
  //dev/kernels:cpu_linear_layout_probe
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gap-p1.json
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_linear_layout_probe \
  --rounds 5 --warmup 2 --json --out /tmp/cpu-gap-p1-layout.json
tools/nanochat test --nocache_test_results //backends/cpu:kernels_test \
  //tests:oracle_test //tests:train_parity_test
```

The raw bridge log is `/tmp/cpu-gap-p1-e2e.log`. The kernel reports are
`/tmp/cpu-gap-p1.json` (acceptance run), `/tmp/cpu-gap-p1-kernel.json` (warmed
run), and `/tmp/cpu-gap-p1-layout.json` (operand-layout probe).

---

#### 7. Phase 2 — blocking and packing — PASS

The Phase 2 change adds register blocking and operand packing to the CPU
`Gemm`. The output is tiled into 4 by 16 register tiles. The `b` operand is
packed per column block into a contiguous panel. The `a` operand is packed per
row block. An AVX2 plus fused-multiply-add micro-kernel runs on a supported
host. A portable micro-kernel is the fallback. OpenMP parallelizes the output
row blocks.

The Phase 2 kernel gate is 80 GFLOP per second for the language-model head
`dgrad` shape. The measured OpenBLAS ceiling is 414 GFLOP per second.

##### Kernel gate — PASS

Command:

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gap-p2.json
```

The run uses the `t3-bench` profile with 12 threads and fp32 precision. The
table gives the four dominant shapes from Cause, design, and plan, section 2.

| Shape | Layout | Rate (GFLOP/s) | Target | Result |
|---|---|---:|---:|---|
| `gemm_forward` | `tb=0` | 133.15 | 15 | pass |
| `gemm_wgrad` | `ta=1` | 120.19 | 15 | pass |
| `gemm_dgrad` | `tb=0` | 130.85 | 80 | pass |
| `mlp_gemm` | `tb=0` | 161.36 | 15 | pass |

The `gemm_dgrad` rate is 130.85 GFLOP per second against the 80 GFLOP per
second target. The rate is 1.6 times the target, and it is 31 percent of the
414 GFLOP per second OpenBLAS ceiling.

The model-layout rows also pass:

| Shape | Layout | Rate (GFLOP/s) |
|---|---|---:|
| `gemm_forward_tb1` | `tb=1` (model) | 141.38 |
| `mlp_gemm_tb1` | `tb=1` (model) | 172.07 |

##### Accumulator type

The reduction stays in one `double` accumulator. The factory packs the whole
reduction into a single `double` sum over `k`. This matches the rounding of the
shipped scalar loop. Phase 2 adds no `float` accumulator path. The comments in
`backends/cpu/kernels.cc` state the `double` accumulator.

##### Correctness gate — PASS

The parity gate stays green. The three tests pass after the Phase 2 change:

```bash
tools/nanochat test --nocache_test_results \
  //backends/cpu:kernels_test //tests:oracle_test //tests:train_parity_test
```

`train_parity_test` is the strict gate. It compares the loss, the gradient, and
the parameter trajectory against the PyTorch reference. The frozen seam
`include/nanochat/kernels.h` is unchanged.

##### Verdict

**Phase 2 kernel gate: PASS.** The `gemm_dgrad` rate is 130.85 GFLOP per
second, above the 80 GFLOP per second target. The OpenBLAS ceiling is 414
GFLOP per second. The parity gate passes.

##### Reproduction

The raw benchmark report is `/tmp/cpu-gap-p2.json`. The report uses the
`nanochat.bench.v1` schema with `threads=12`.

---

#### 8. Phase 3 — optimizer and elementwise families — PASS

Revision `283a511`, branch `sliceme/agent-cpu-backend`. The Phase 3 change
threads and vectorizes `MuonUpdate`, `ClassifierForward`,
`ClassifierBackward`, `AttentionForward`, and `AttentionBackward`.

##### End-to-end gate — PASS

Command (the bridge in The measured baseline, step 4):

```bash
tools/nanochat run train -- bazel-bin/src/train_main \
  --train-parquet <train-parquet> --tokenizer <tokenizer.nctoken> \
  --layers 4 --heads 4 --kv-heads 4 --hidden 512 --seq 512 --vocab <vocab> \
  --window-pattern L --batch 2 --grad-accum 1 --num-iterations 10 \
  --warmup-steps 1 --eval-every -1 --save-every -1 \
  --checkpoint <checkpoint-dir>/model.ckpt --log <checkpoint-dir>/train.log
```

The bridge runs under the `train` profile. The profile gives 8 CPU threads
(`cpus=0-7`, `cpu.max=800000 100000`). The run uses fp32.

The mean rate over steps 2 to 10 is 591.0 tokens/s. Step 1 is not part of the
mean, because it contains the first-run cost. The mean step time is 1.733
seconds.

| Metric | Phase 2 | Phase 3 | Target | Result |
|---|---:|---:|---:|---|
| tokens/s | 45.91 | 591.0 | — | — |
| seconds per step | 22.3 | 1.733 | 5 | pass |
| threads | 8 | 8 | — | — |

The machine-readable value is `/tmp/cpu-gap-p3-e2e.json`:
`{"seconds_per_step": 1.733}`. The raw log is `/tmp/cpu-gap-p3-e2e.log`.

##### Family gates — PASS

The three Phase 3 family gates pass. Each value is the mean seconds per step
for one family.

| Family | Baseline (s/step) | Phase 3 (s/step) | Target (s/step) | Result |
|---|---:|---:|---:|---|
| Muon | 12.673 | 0.2179 | 2 | pass |
| Classifier | 2.304 | 0.2676 | 0.5 | pass |
| Attention | 1.550 | 0.2889 | 0.4 | pass |

The family values are in `/tmp/cpu-gap-p3-muon.json`,
`/tmp/cpu-gap-p3-classifier.json`, and `/tmp/cpu-gap-p3-attention.json`. The
three families use 0.7744 seconds of the 1.733-second step.

##### Correctness gate — PASS

The full CPU suite passes. All 16 tests pass, including the strict
`train_parity_test`.

```bash
tools/nanochat test
```

##### Verdict

**Phase 3 end-to-end gate: PASS.** The step time is 1.733 seconds against the
5-second target. The three family gates pass. The full CPU suite passes.
