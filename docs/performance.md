# Performance measurement and debugging

How to investigate a performance gap in `nanochat.cpp`, and what the project
provides to make that investigation cheap. The general, project-independent
method is also captured in the `performance-investigation` skill; this document
is the `nanochat.cpp`-specific interface and protocol.

The host is a single GTX 1080 Ti (sm_61) under WSL2 with no CUDA Profiling Tools
Interface, so kernel-counter profiling is unavailable here. Device-event timing
through this project's benchmarks is the primary instrument. Kernel-level
counters require native Linux.

---

## 1. Measure through the entry point

Every timing run goes through `tools/nanochat` so the sandbox and the single-GPU
broker are applied. Do not run a benchmark binary directly.

- `tools/nanochat doctor` — read-only diagnostics before measuring. It reports
  the revision, toolchain, device, sandbox limits, whether the GPU broker lock
  is free, and the tail of the broker and sandbox logs. Use it to tell slow code
  from a busy or misconfigured host.
- `tools/nanochat profile [opts]` — builds and runs the attention benchmark
  under the `t3-bench` profile with the GPU broker held. Options pass through:
  `--json`, `--out PATH`, `--warmup N`, `--iters N`, `--rounds N`.
- `tools/nanochat bench -- <binary> ...` — run any other benchmark under the
  `t3-bench` resource profile (no broker; use `profile` or
  `tools/nanochat gpu --profile t3-bench -- <binary>` when exclusivity matters).

Record the revision, build configuration (`--config=cuda` or `--config=sm_75`,
precision), and device with every number. `tools/nanochat doctor` prints most
of it.

## 2. The measurement protocol

1. Confirm the device is idle (`tools/nanochat doctor`).
2. Warm up, then time a batch with one synchronization at the end, or use
   device events (the event benchmark does this).
3. Report the best of several rounds for throughput and the median across runs
   for latency; say which.
4. Never set `CUDA_LAUNCH_BLOCKING` for timing.
5. Keep the before and after numbers together with their configuration.

## 3. The debugging interface

### 3.1 The attention benchmark

`dev/kernels/attention_bench.cc` times the shipped `AttentionForward` and
`AttentionBackward` with CUDA events at:

- the `d8_s512` training shape (batch 8, sequence 512, 4 heads, head dimension
  128, full context);
- a sliding-window variant;
- a grouped-query variant;
- a head-dimension sweep over 32, 64, 128, and 256.

The head-dimension sweep is the diagnostic for the known redundancy in
`OnlineSoftmaxTile` and `SoftmaxGradTile`
(`backends/cuda/kernels/device_utils.cuh`): the tile recomputes the query-key
dot product once per owned output dimension, so the issued work grows about
quadratically with the head dimension while the necessary work grows linearly.
If the measured time across the sweep grows at a steeper exponent than the
necessary operation count, the redundancy is confirmed; if a fixed variant
removes the exponent, the fix is validated. See the `roofline` reference in the
skill.

The benchmark reports the achieved rate against the *necessary* floating-point
operations, so a correct implementation should approach the device ceiling and
the current implementation should sit far below it.

### 3.1.1 Validating a candidate fix locally

Do not edit the shipped tile first. Build the candidate as a local mirror under
`dev/kernels` with the same seam signatures, check it against the host reference
in `dev/kernels/sequence_ref.h`, and time it with `bench_utils.h` at the same
shapes. Promote it into `device_utils.cuh` only after the correctness check
passes and the head-dimension exponent from the sweep has flattened relative to
the shipped tile. The `summarize_times.py` script and the exponent snippet in
the `performance-investigation` skill produce both.

### 3.2 Adding a benchmark

Follow `dev/kernels/README.md`. Use `dev/kernels/bench_utils.h`:

- `BenchOptions` sets warm-up, iterations, rounds, and JSON output.
- `EventTimer::Time(callable, options)` returns the best mean milliseconds per
  call, measured with device events on the default stream.
- `BenchReport::Add(name, shape, ms, gflops)` collects rows.
- `BenchReport::Print(options)` prints a human table, and writes JSON to
  `options.out` when set.

Build the target as a `cuda_binary` (it uses the CUDA runtime through
`bench_utils.h`) and add it next to the other benchmark binaries in
`dev/kernels/BUILD.bazel`. Do not add a benchmark to the `all` test suite.

### 3.3 Machine-readable output

`BenchReport` emits the `nanochat.bench.v1` schema:

```json
{
  "schema": "nanochat.bench.v1",
  "rows": [
    {"name": "attention_fwd:d8_s512", "shape": "B=8 T=512 H=4 KV=4 D=128 wl=-1", "ms": 12.3, "gflops": 45.6}
  ]
}
```

`gflops` is the achieved rate in GFLOP/s against the necessary operation count.
Feed the file to the skill's `summarize_times.py` for a per-feature budget.

## 4. Root-cause checklist for this project

When the full step is slower than expected, split it before choosing a fix:

- Attention forward and backward (`dev/kernels/attention_bench.cc`). Known
  per-owned-dimension redundancy; see section 3.1.
- Memory access in the attention tile: uncoalesced key/value loads, a serial
  single-accumulator dot product, and no key/value reuse across query rows.
- Backward key/value atomics, especially grouped-query attention and the fp16
  compare-and-swap fallback on Pascal.
- GEMM (cuBLAS) — expected near peak on Pascal; confirm rather than assume.
- QkPrep (RMSNorm plus RoPE plus scale), Pointwise, Classifier, Embedding.
- Optimizer (AdamW and Muon) and global-norm clipping.
- Launch overhead and the backward `Memset` of the query/key/value gradients.
- Host-side gaps: data loading, logging, checkpointing, evaluation.

Existing benchmarks to reach for: `row_bench`, `qk_prep_bench`,
`decode_fused_bench`, and now `attention_bench`.

## 5. Profiling limitations

CUPTI cannot initialize under WSL2, so `torch.profiler`, Nsight Systems, and
Nsight Compute cannot attribute device time on this host, and the PyTorch
profile report (`profiles/d8_s512/PROFILE_REPORT.md` in the reference checkout)
contains host operator time only. To attribute device time here, use the CUDA
event benchmarks. For streaming-multiprocessor occupancy, achieved bandwidth,
and warp stall reasons, profile on native Linux (T3), and record the revision.

## 6. Baselines

Keep a committed baseline of the `nanochat.bench.v1` output for the production
shape so a regression is visible without re-deriving it. Update the baseline
only with the revision and configuration that produced it. The merge-gate
benchmark is a single `tools/nanochat profile --json` run at a wave boundary,
not per commit.
