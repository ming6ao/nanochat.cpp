# Performance measurement and debugging

This document describes how to measure a performance gap in `nanochat.cpp`. The
general method is in the `performance-investigation` skill. This document covers
the project tools.

The host is one GTX 1080 Ti (sm_61) under WSL2. CUPTI is not available. Kernel
counters do not work here. Device-event timing through the project benchmarks is
the primary instrument. Kernel counters need native Linux.

## 1. Measure through the entry point

Run every timing through `tools/nanochat`. The entry point applies the sandbox
and the single-GPU broker. Do not run a benchmark binary directly.

- `tools/nanochat doctor` reports the revision, the toolchain, the device, the
  sandbox limits, and the broker state. Use it before a measurement.
- `tools/nanochat profile [opts]` builds and runs the attention benchmark under
  the `t3-bench` profile. The GPU broker stays held. The options are `--json`,
  `--out PATH`, `--warmup N`, `--iters N`, and `--rounds N`.
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
- `BenchReport::Print` prints a table. It writes JSON to `options.out` when
  set.

Build the target as a `cuda_binary` next to the code it measures. Do not add a
benchmark to the `all` test suite.

### 3.4 Machine-readable output

`BenchReport` emits the `nanochat.bench.v1` schema:

```json
{
  "schema": "nanochat.bench.v1",
  "rows": [
    {"name": "attention_fwd:d8_s512", "shape": "B=8 T=512 H=4 KV=4 D=128 wl=-1", "ms": 12.3, "gflops": 45.6, "work_ratio": 2.0}
  ]
}
```

`gflops` is the rate against the necessary operation count. `work_ratio` is the
issued work over the necessary work. Feed the file to the skill's
`summarize_times.py`.

## 4. Root-cause checklist

Split the step before you choose a fix:

- Attention forward and backward
  (`backends/cuda/kernels/attention_benchmark.cc`).
- Memory access in the attention tile: uncoalesced key/value loads, a serial
  dot product, and no key/value reuse across query rows.
- Backward key/value atomics, especially grouped-query attention and the fp16
  compare-and-swap fallback on Pascal.
- GEMM (cuBLAS). Confirm the rate. Do not assume it.
- QkPrep, Pointwise, Classifier, and Embedding.
- The optimizer and the global-norm clip.
- Launch overhead and the backward `Memset` of the query/key/value gradients.
- Host-side gaps: data loading, logging, checkpointing, and evaluation.

Benchmarks to use: `row_benchmark`, `qk_prep_bench`, `decode_fused_bench`,
`attention_benchmark`, and `eval_bench`.

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
