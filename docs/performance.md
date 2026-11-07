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

### 2.1 The launch, barrier, and elementwise budget of one stage

The protocol above applied to a stage: before a campaign fuses an optimizer
stage into one kernel, it measures three things separately -- what the dispatch
costs, what an elementwise pass costs, and what the arithmetic it displaces
costs in a hand-written kernel. The Muon optimizer stage is the worked example
(`docs/kernels.md` Part 1 section 3 has the arithmetic of the bet, and
[baselines.json](baselines.json) holds the rows).

Device: **NVIDIA GeForce GTX 1080 Ti (sm_61, 28 SMs, 11 GB, driver 536.99,
CUDA 12.0)**, declared fp32 rate 11340 GFLOP/s. Every number below is the best
of three rounds from a `tools/nanochat bench` job on this host, with device-event
timing and one synchronization at the end of the batch, as the protocol asks.
The command is named with each table. The revisions are the `muon` entry's in
[baselines.json](baselines.json): the `muon:` and `muon_fused:` rows are at
`6675c5d`, and `phase0_share`, `sync`, and `phase1_gemm` are at the prototype
revision `98fe9f2`.

**Dispatch, in microseconds.** Empty launches, a cooperative launch, and one
cooperative grid barrier at the grids a fused kernel would hold resident
(`bazel-bin/dev/kernels/muon_launch_sync_bench --json`):

| Row | Shape | us |
|---|---|---|
| `launch:empty:1` | 1 block, 256 threads | 1.64 |
| `launch:empty:84` | 84 blocks (the resident grid of a 256-thread, 12 KB block) | 1.88 |
| `launch:empty:4000` | 4000 blocks (a full batched-GEMM grid) | 8.37 |
| `launch:coop:84` | cooperative launch, 84 blocks, no barrier | 1.92 |
| `sync:barrier:84` | one `grid.sync()`, 84 blocks | 0.92 |
| `sync:barrier:224` | one `grid.sync()`, 224 blocks | 2.19 |

A dependent launch of a full batched grid costs 8.4 us and a grid barrier
0.9 us. One fused iteration therefore replaces 41.9 us of launch gaps with
4.7 us of one launch plus three barriers: **37.2 us saved per iteration**, which
is 0.5 percent of an `h100` iteration (7.3 ms) and 0.01 percent of a `big` one
(275 ms). Launch fusion is a rounding error against the elementwise passes it
also removes, which is the measurement that decided where the rest of the effort
went.

**The elementwise passes, as a share of the step.** From three existing rows of
`bazel-bin/backends/cuda/kernels/muon_benchmark --profile h100|big --muon
--json`, with `loop = muon:step(ns=5) - muon:step(ns=0)` and
`launch + elementwise = loop - polar:step`:

| Profile | `polar:step` | `muon:step` ns=0 | `muon:step` ns=5 | launch + elementwise | share of step |
|---|---|---|---|---|---|
| `h100` | 30.93 ms | 6.29 ms | 42.84 ms | 5.62 ms | **13.1 %** |
| `big` | 1268.30 ms | 85.06 ms | 1462.76 ms | 109.41 ms | **7.5 %** |

A second session measured 3.81 ms (9.2 percent) and 104.90 ms (7.2 percent), so
the share is 9 to 13 percent at `h100` and 7 to 7.5 percent at `big`.

**The pass savings, measured end to end.** The fusion folded three elementwise
passes into their neighbours and dropped two buffers, which takes the stage
workspace from `2·total + 3·min_total` to `2·total + 2·min_total`. What that is
worth is the `muon_fused:` row against the `muon:` row over the same buffers:

| Profile | Group or step | `muon:` ms | `muon_fused:` ms | Fused |
|---|---|---|---|---|
| `h100` | 4x12, batch 4 | 0.11-0.17 | 0.124-0.125 | noise, both directions across runs |
| `h100` | 256x512, batch 16 | 3.065 | 3.472 | +13 % |
| `h100` | 512x512, batch 16 | 9.181 | 12.170 | +33 % |
| `h100` | 512x2048, batch 8 | 15.388 | 18.486 | +20 % |
| `h100` | 2048x512, batch 8 | 14.860 | 18.903 | +27 % |
| `h100` | step, ns=0 | 6.242 | 6.869 | +10 % |
| `h100` | step, ns=5 | 41.746 | 53.192 | +27 % |
| `big` | 5x12, batch 10 | 0.160 | 0.133 | -17 %, inside the run-to-run jitter of a 0.13 ms row |
| `big` | 640x1280, batch 40 | 72.588 | 88.017 | +21 % |
| `big` | 1280x1280, batch 40 | 335.215 | 387.420 | +16 % |
| `big` | 1280x5120, batch 20 | 583.223 | 1025.325 | +76 % |
| `big` | 5120x1280, batch 20 | 436.526 | 584.754 | +34 % |
| `big` | step, ns=0 | 84.986 | 90.179 | +6 % |
| `big` | step, ns=5 | 1426.937 | 2073.520 | +45 % |

The `ns=0` rows are the control: no GEMM runs and no fold has anything to fold,
so those 6 to 10 percent are the cost of one block size for sixteen phases,
not of the fusion's arithmetic. The folded passes do not recover it.

**The hand GEMM, against cuBLAS.** `dev/kernels/muon_fused_gemm_bench` times the
three products of one iteration -- `A`, `A·A`, and `prod` -- as hand GEMM and as
`kernels::Gemm`, at the two groups the campaign reports:

```bash
tools/nanochat build //dev/kernels:muon_fused_gemm_bench
tools/nanochat bench -- bazel-bin/dev/kernels/muon_fused_gemm_bench --json
```

| Group | Stage | hand (GFLOP/s) | cuBLAS (GFLOP/s) | hand / cuBLAS |
|---|---|---|---|---|
| 640x1280, batch 40 | `A` 640x640x1280 | 5373-5484 | 6690-6762 | 0.80 |
| | `A2` 640x640x640 | 5234-5371 | 8933-9484 | 0.55-0.60 |
| | `prod` 640x1280x640 | 5443-5526 | 8958-9116 | 0.60-0.62 |
| | `iter` | 5725-5933 | 8205-8318 | **0.68-0.72** |
| 1280x5120, batch 20 | `A` 1280x1280x5120 | 4549-5341 | 4181-5608 | 0.95-1.17 |
| | `A2` 1280x1280x1280 | 5993-6307 | 8786-9027 | 0.68-0.70 |
| | `prod` 1280x5120x1280 | 6037-6047 | 8779-8799 | 0.69 |
| | `iter` | 5149-5612 | 6192-6311 | **0.83-0.90** |

The gate asked for 85 percent of the cuBLAS rate on a targeted group; the
targeted 640x1280 group reaches 0.68 to 0.72. The one shape where the hand
schedule meets the gate is the largest group, and it meets it because cuBLAS is
weak there (4181 to 5608 GFLOP/s) and not because the hand kernel is fast. A
rate gate against cuBLAS is therefore a poor selector on its own; the end-to-end
`muon_fused:` row against the `muon:` row is the measurement that decides.

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

`BenchReport` in `nanochat/bench.h` is the one writer of the
`nanochat.bench.v3` schema:

```json
{
  "schema": "nanochat.bench.v3",
  "device": "h100",
  "precision": "bf16",
  "peak_flops": 989000000000000.0,
  "rows": [
    {"name": "attention_fwd:d8_s512", "family": "attention", "phase": "forward",
     "precision": "bf16", "shape": "B=8 T=512 H=4 KV=4 D=128 wl=-1",
     "ms": 12.3, "gflops": 45.6, "work_ratio": 2.0, "mfu": 0.33}
  ]
}
```

`gflops` is the rate against the necessary operation count. `work_ratio` is the
issued work over the necessary work. `device`, `precision`, and `peak_flops`
appear only when the report states them, and `mfu` is the row's `gflops` over
that `peak_flops`, clamped to `[0, 1]`. The share belongs to the precision the
report names, so a bf16 run divides by the tensor-core rate and an fp32 run by
the CUDA-core rate ([precision.md](precision.md)). A row that issues no
arithmetic states `0`, and the table prints `-` for it.

One header writes the document for every benchmark, whether it links the CUDA
runtime (`backends/cuda/kernels/testing/bench_utils.h`) or a host clock
(`src/eval_bench_command.cc`, `dev/kernels/`). A benchmark adds rows. Where
its report carries a field that is not a row, the benchmark adds one extra
top-level field with `SetExtra`. It does not format JSON itself.

In the table, the denominator stands above the header as one comment line, and
the `MFU` column appears only when a row reports a share. A benchmark that
states no peak prints the three-column table of the earlier reports. Feed the
file to the skill's `summarize_times.py`.

The four identity keys join a measured row to a predicted one: `family`,
`phase`, `precision`, and `shape`. The model names the family and the phase, and
the benchmark names the same family and adds the shape
([performance-model.md](performance-model.md) section 7.4). Both sides state
`ms` in milliseconds and both state `work_ratio`, so a predicted row and a
measured row compare field by field with no conversion step:

```bash
tools/nanochat roofline --device h100 --json > roof.json
tools/nanochat profile --json --out bench.json
tools/nanochat attain --bench bench.json --roofline roof.json
```

`attain` prints the fraction of the bound each family reached, the binding
limit, and the gap. Its output is the attainment record of
[performance-model.md](performance-model.md) section 10.

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
`attention_benchmark`, and `nanochat_cli eval_bench`.

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
`nanochat_cli eval_bench`:

```bash
tools/nanochat build --config=cuda //src:nanochat_cli
tools/nanochat bench -- bazel-bin/src/nanochat_cli eval_bench --batch 16 --seq 1024 \
  --layers 12 --heads 6 --kv-heads 6 --hidden 768 \
  --vocab 32768 --padded-vocab 32768
```

## 7. Baselines

Keep a committed baseline of the `nanochat.bench.v3` output for the production
shape. Then a regression is visible without a new measurement. Update the
baseline only with the revision and the configuration that produced it. The
merge-gate benchmark is one `tools/nanochat profile --json` run at a wave
boundary.

---

## Part 1. The profile-guided optimization loop

_Absorbed from the former `profile-guided-optimization.md`. Its section numbers
are unchanged, so a reference to a section of that document resolves to the
same number here._

One stage runs on one card, one iteration produces a pair of measurements, and
the pair decides the next guess. The Python surface owns `pgo.py` and the Modal
functions, the Harness owns the stage loggers, and the architect owns this
document and the skill.

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
| The stage loggers | `src/sft_command.cc`, `src/rl_worker_command.cc` | **Gap.** Both print free text, so only `base` writes `metrics.jsonl` |
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

One target is one folder directly under the root, and its name carries the
shape, the stage, and the device. The name is one path component, such as
`nanochat-d20-base-h100`, so a path tells a reader which model it measures. The
`--name` option sets it, and `<stage>-<device>` is the default.

```
profiles/pgo/<name>/
  loop.json                     the state: metric history, rejected ideas, spend
  RESULT.md                     the status page: the last number, the history,
                                the command, and the longest kernels
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

The Chrome trace is the largest file: `iter-01` of `nanochat-d8-base-h100` held
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
approve it before iteration 2 takes it. [kernels.md](kernels.md) Part 2
carries the related design, which removes the two per-layer scalar readbacks
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

4. Run the `nanochat.cpp` CPU backend. Build `//src:nanochat_cli` and run the
   binary under the sandbox. Derive the model shape and the horizon with
   `nanochat_cpp.plan` ([python.md](python.md) section 4).

   ```bash
   cd ~/repos/nanochat.cpp
   tools/nanochat build //src:nanochat_cli
   tools/nanochat run train -- bazel-bin/src/nanochat_cli train \
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

The revision is `124ceb0`. The raw log is
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
from `nanochat.cpp` revision `124ceb0`. The reference is the PyTorch nanochat
checkout.

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
- [build.md](build.md) § Dependency invariants: the dependency rules.
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
- The backend rules: [build.md](build.md), [kernels.md](kernels.md).
- The ownership map: [AGENTS.md](../AGENTS.md).
- The measured baseline, above.
