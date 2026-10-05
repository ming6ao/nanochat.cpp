# CPU base-training baseline

This document records a baseline for base training on the CPU. The baseline
compares the PyTorch nanochat reference with the `nanochat.cpp` CPU backend.
Use this baseline to measure a future change to the CPU backend.

The CPU backend in `backends/cpu/kernels.cc` started as a correctness
reference. Phases 1 to 3 of [cpu-gap.md](cpu-gap.md) added a blocked and
threaded GEMM. They also vectorized the optimizer and the elementwise
families. The scalar baseline and the promoted baseline are both below.

## Host and configuration

- Host: 12 vCPU under WSL2, 7.8 GiB RAM, one GTX 1080 Ti (not used).
- Sandbox profile: `train` (800% CPU, 8 cores, 6 GiB).
- Model: depth 4, dimension 256, 2 heads, 2 key/value heads, head dimension 128.
- Context length: 512. Window pattern: `L` (full attention).
- Device batch: 2. Total batch: 1024. Gradient accumulation: 1.
- Steps: 10. Warmup: 1. Evaluation, sampling, and saving disabled.
- Parameter count: 36,700,242. Estimated operations per token: 7.195896e7.
- Training tokens per run: 10,240.

## Procedure

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

4. Run the `nanochat.cpp` CPU backend. The Python bridge tokenizes the data,
   builds `//src:train_main`, and runs the binary under the sandbox.

   ```bash
   cd ~/repos/nanochat.cpp
   tools/nanochat_cpp base_train --backend cpu \
     --depth=4 --max-seq-len=512 --window-pattern=L \
     --device-batch-size=2 --total-batch-size=1024 --num-iterations=10 \
     --warmup-steps=1 --eval-every=-1 --core-metric-every=-1 \
     --sample-every=-1 --save-every=-1 --model-tag=cpu_bench_cpp
   ```

## Results

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

## Promoted baseline (Phase 4)

Phase 4 of [cpu-gap.md](cpu-gap.md) promotes the Phase 1 to 3 changes and
records the new baseline. The measurement follows section 9 of that document.
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

## Findings

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

## Revisions

Record the revision of this repository and the revision of the reference
repository with every new measurement. The scalar baseline comes from
`nanochat.cpp` revision `759083e` on branch `cpu`. The promoted baseline comes
from `nanochat.cpp` revision `124ceb0` on branch `sliceme/agent-cpu-backend`.
The reference is the PyTorch nanochat checkout.

## Raw data

The summary data is in [cpu-baseline.json](cpu-baseline.json). The raw logs are
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
