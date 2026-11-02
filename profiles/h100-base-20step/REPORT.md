# The H100 base-training trace: where one step goes

Status: measured. One Nsight Systems trace of the base stage on the H100 leg.
The trace shows that the card is idle for 81 percent of a step. It also shows
that the gradient norm is the largest single cost, because it runs on the CPU
in a bf16 build.

The run is the chain's own base stage: profile `h100` (8 layers, hidden 512,
8 heads, 4 kv heads, sequence 256, vocabulary 32768), batch 16, the real
FineWeb-Edu shard, 20 steps. The engine reports 0.322 s per step, 12700 tok/s,
and 0.32 percent MFU without the profiler. With the profiler attached it
reports 0.413 s per step, 9900 tok/s, and 0.25 percent MFU.

## 1. The artifacts

All paths are relative to this directory, which holds the trace of record:

| File | What it holds |
|---|---|
| `trace/base.nsys-rep` | The Nsight Systems report. Open it with `nsys-ui`. |
| `trace/base.sqlite` | The same report as a database, for a query. |
| `h100-base-trace.json` | The Chrome trace, 75 MB. Open it in Perfetto. |
| `report.json` | The numbers of section 3 and section 4. |
| `trace/timeline_cuda_gpu_trace.csv` | Every device operation with its start, end, size, and grid. |
| `trace/timeline_cuda_api_trace.csv` | Every driver call with its host interval. |
| `trace/base_cuda_gpu_kern_sum.csv` | The device time per kernel. |
| `trace/base_cuda_api_sum.csv` | The host time per driver call. |
| `metrics.jsonl`, `summary.json` | The engine's own records of the traced run. |
| `profile.log` | The traced run's stdout, one line per step. |

The trace also sits on the `nanochat-runs` Volume at
`profile-base/profile/`. A second trace sits at `profile-cublas/profile/` and
takes the cuBLAS activity domain. Its CUDA records agree with this one. Version
2023.4.4 writes no report for that domain, so section 6 says what it did not
answer.

Section 1.1 shows the two commands. Section 1.2 shows how to read the trace.

### 1.1 Collect the trace

```bash
modal run tools/modal/app.py::profile_demo \
  --job /tmp/mrun/r2after-job.json --run-id profile-base --steps 20
modal run tools/modal/app.py::profile_demo \
  --job /tmp/mrun/r2after-job.json --run-id profile-cublas --steps 20 \
  --domains "cuda,cublas,osrt"
modal volume get nanochat-runs profile-base/profile/base.nsys-rep ./base.nsys-rep
```

`profile_h100` layers Nsight Systems on the built H100 image, so the trace is
of the library the chain runs and not of a second build. It traces the chain's
own `nanochat_cpp.pipeline.stage_base`, so the shape, the batch, and the driver
are the ones of the run being explained. `docs/performance.md` section 5 says
that CUPTI does not work on the local 1080 Ti host. It works in the H100
container, which is why this trace exists.

### 1.2 Read the trace

```bash
python profiles/h100-base-20step/summarize_trace.py \
  --dir profiles/h100-base-20step/trace --steps 20 --step 6 \
  --json profiles/h100-base-20step/h100-base-trace.json
```

The script cuts the trace at the engine's own step marks, so a reader gets the
step the machine ran. `--step 6` names the seventh steady step. The Chrome
trace opens in Perfetto at <https://ui.perfetto.dev>.

## 2. The instrument is sound

The profiler costs 28 percent of the step rate: 0.413 s per step traced against
0.322 s untraced. The cost is the instrumentation of 16178 driver calls per
step. A count, a size, and a ratio survive that cost. An absolute millisecond
does not, and section 3 states which column is which.

The trace's own tail is not a step. The checkpoint write copies 22016 buffers
of 8 KB to the host after the last step. That is 386 ms of the trace's 12.8 s
span, and it is a start-up and shut-down cost, not a step cost.

## 3. One steady step

The window is step 7 of the sequence, 436.4 ms wall time with the profiler
attached.

| Fact | Value | Share of the step |
|---|---|---|
| Device operations queued | 8352 | - |
| Driver calls | 16178 | - |
| Device busy | 81.8 ms | 18.7 percent |
| Host inside a CUDA call | 103.6 ms | 23.7 percent |
| Host in its own code, inside a stall | 187.9 ms | 43.1 percent |
| Device to host bytes | 374 MB | - |
| Device to device bytes | 382 MB | - |
| Host to device bytes | 9 MB | - |

The operations are 7812 kernels, 226 device-to-device copies, 115
device-to-host copies, 82 host-to-device copies, and 117 memsets. The card has
nothing to run for 81 percent of the step, so the step is not arithmetic bound.
The host is the bottleneck.

## 4. The four causes

### 4.1 The gradient norm runs on the CPU (`src/ops.cc:1344`)

`ops::GradNormForward` reduces the gradients on the device only in an fp32
build. In the bf16 build the `else` branch of the `if constexpr` stages
**every gradient span to the host** with a blocking copy, and then runs a
serial scalar loop over every element:

```cpp
for (int s = 0; s < num_spans; ++s) {
  std::vector<float> span(static_cast<std::size_t>(host_counts[s]));
  kernels::Memcpy(span.data(), host_grads[s], span.size() * sizeof(float),
                  CopyDir::kDeviceToHost);
  for (float value : span) acc += static_cast<double>(value) * value;
}
```

The trace shows the six largest stalls of the step, and each one starts at the
end of one of these copies:

| Stall | Copy before it | Host in a CUDA call | Host in its own loop |
|---|---|---|---|
| 51.5 ms | 67.1 MB | 17.3 ms | 34.2 ms |
| 41.6 ms | 67.1 MB | 12.8 ms | 28.8 ms |
| 28.7 ms | a span the report records as 0 KB | 1.5 ms | 27.2 ms |
| 24.7 ms | 33.6 MB | 8.8 ms | 15.9 ms |
| 23.6 ms | 33.6 MB | 8.7 ms | 14.9 ms |
| 22.9 ms | 33.6 MB | 8.7 ms | 14.2 ms |

The loop costs between 1.7 and 3.2 nanoseconds per element. The 16.78 million
elements of one 67 MB span take 34.2 ms, and the 8.39 million elements of a
33.6 MB span take about 27 ms. That is the latency of one dependent double
precision accumulation per element, and it happens once per step.

The host is inside its own code for 187.9 ms of the traced step while it waits
nowhere: that is this loop. It is CPU work, so the profiler does not inflate it,
and the same loop runs in the untraced step. That is about 58 percent of the
0.322 s step. The 374 MB of device-to-host traffic is the same mechanism. Two
spans are 67.1 MB each and four are 33.6 MB each; the rest is one span per
parameter.

**The fix.** The `SpanReduce` primitive already walks a ragged span list on the
device (`include/nanochat/kernels.h`). The bf16 build needs the reduce whose
operand is the fp32 gradient type, or one cast into a bf16 scratch, which is
the entry the node report names as missing. Then the step reads one scalar, and
the clip scale already stays on the device. Expected effect: the step falls from
0.322 s to about 0.12 s, and the reported MFU rises from 0.32 to about 0.9
percent.

### 4.2 Every device read is a blocking copy of pageable memory

`backends/cuda/device.cu:107` maps `kernels::Memcpy` to `cudaMemcpy`. The seam
exposes no events, so the copy is synchronous on purpose. The destination is a
`std::vector`, so the host buffer is pageable, and the driver stages it through
an internal pinned buffer.

The model reads scalars back to the host with `ReadHost` for `resid`,
`x0_lambda`, `smear_lambda`, and `backout_lambda`. That is about 36 reads per
step, and the trace shows 51 zero-byte device-to-host copies per step. Each one
drains the whole queue, so the card goes idle, and then the host pays the
staging copy.

**The fix.** Keep the scalars on the device. A scalar that a kernel needs is a
kernel argument or a device buffer, and neither needs a host read. Where a read
is unavoidable, use one pinned buffer and one asynchronous copy at a phase
boundary.

### 4.3 The step queues 7812 launches

The trace holds 94049 kernels shorter than 3 microseconds. They own 12.8
percent of the device time and 49 percent of the launches.

Two families own the count:

| Family | Launches per step | Device time per step | Mean |
|---|---|---|---|
| Attention backward GEMMs, 32x32 tile | 2048 | 26.6 ms | 2.7 us |
| Muon and pointwise kernels | about 500 | 9.9 ms | 20 us |

The attention backward issues one cuBLAS call per batch and head pair: 16 rows
times 8 heads is 128 problems, times two GEMMs, times 8 layers. The tile is
32x32, so each call leaves the card almost empty.

**The fix.** One batched or grouped GEMM for the per-head backward, with the
same seam signature. The Muon benchmark measured 75 grouped GEMM calls at
1.01 ms, so a grouped call is not free, but 2048 launches at 30 ms of host
launch time is a different scale.

### 4.4 The optimizer gathers 382 MB per step

The Muon step gathers each shape group into one buffer before the update. The
count is 58 copies of 4 MB, 48 of 1 MB, 24 of 2 MB, and 64 of 512 KB. That is
device to device traffic at about 1 TB/s, so it costs 1.1 ms. It is a cost of
the algorithm, and it is not a stall. This document names it so a reader does
not mistake the 382 MB for the cause.

## 5. The order of work

| # | Work | Expected effect | Gate |
|---|---|---|---|
| 1 | Reduce the gradient norm on the device in the bf16 build | step 0.322 to about 0.12 s, MFU 0.32 to about 0.9 percent | `//src:optim_test`, `//tests:oracle_cuda_test` and the loss trace unchanged |
| 2 | Remove the per-layer `ReadHost` scalar reads | about 51 drains per step | the same tests, plus one `tools/nanochat profile` before and after |
| 3 | Batch the attention backward GEMMs | about 26 ms of device time and 30 ms of host launch time | the finite difference test of the attention family |
| 4 | Batch the checkpoint write, which copies 22016 buffers of 8 KB | a faster shut-down, no step effect | the checkpoint round trip test |

## 6. What this does not measure

- The trace has no CPU sample, so the driver calls and the code name the host
  time inside the trainer, not a stack. Section 4.1 matches the
  code to the trace: the loop is `GradNormForward`, and its 1.7 to 3.2
  nanoseconds per element explain the six stall windows.
- The second trace asked for the cuBLAS activity domain, to confirm that the
  host is not inside a library call. Nsight Systems 2023.4.4 writes no report
  for that domain. Its database holds no cuBLAS table either. The confirmation
  therefore rests on the measured in-code time and on the source, not on a
  second instrument. The artifacts of that trace still serve: its CUDA
  records agree with the first trace.
- One card, one container, one job file. The ratios belong to this shape at
  batch 16. A shape that fills the card carries a different mix.
