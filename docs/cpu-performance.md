# CPU backend performance

The CPU backend is the correctness reference for `nanochat.cpp`. It is also the
slow path. This document records the measured baseline, the cause of the gap,
the design of the fix, and the result of every phase. Read
[performance.md](performance.md) for the measurement protocol and
[testing.md](testing.md) for the gates.

This document replaces the former `cpu-baseline.md`, `cpu-gap.md`, and
`cpu-gap-results.md`.

## Part 1. The measured baseline

This document records a baseline for base training on the CPU. The baseline
compares the PyTorch nanochat reference with the `nanochat.cpp` CPU backend.
Use this baseline to measure a future change to the CPU backend.

The CPU backend in `backends/cpu/kernels.cc` started as a correctness
reference. Phases 1 to 3 of this document added a blocked and
threaded GEMM. They also vectorized the optimizer and the elementwise
families. The scalar baseline and the promoted baseline are both below.

### Host and configuration

- Host: 12 vCPU under WSL2, 7.8 GiB RAM, one GTX 1080 Ti (not used).
- Sandbox profile: `train` (800% CPU, 8 cores, 6 GiB).
- Model: depth 4, dimension 256, 2 heads, 2 key/value heads, head dimension 128.
- Context length: 512. Window pattern: `L` (full attention).
- Device batch: 2. Total batch: 1024. Gradient accumulation: 1.
- Steps: 10. Warmup: 1. Evaluation, sampling, and saving disabled.
- Parameter count: 36,700,242. Estimated operations per token: 7.195896e7.
- Training tokens per run: 10,240.

### Procedure

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

### Results

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

### Promoted baseline (Phase 4)

Phase 4 of this document promotes the Phase 1 to 3 changes and
records the new baseline. The measurement follows Part 2, section 9.
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

### Findings

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

### Revisions

Record the revision of this repository and the revision of the reference
repository with every new measurement. The scalar baseline comes from
`nanochat.cpp` revision `759083e` on branch `cpu`. The promoted baseline comes
from `nanochat.cpp` revision `124ceb0` on branch `sliceme/agent-cpu-backend`.
The reference is the PyTorch nanochat checkout.

### Raw data

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

## Part 2. Cause, design, and plan

This document turns the CPU gap analysis into a work order. It states the
measured problem, the design of the fix, the work phases, the owner of each
phase, and the numeric gate that closes it.

Status: proposal. No code change exists yet. The analysis has measurements.
The predictions do not.

Related documents:

- Part 1: the measured baseline.
- [performance.md](performance.md): the measurement protocol.
- [testing.md](testing.md): the test tiers and the definition of done.
- [build.md](build.md): backend selection.
- [DESIGN.md](../DESIGN.md) section 2: the dependency rules.
- [AGENTS.md](../AGENTS.md): the ownership map.

---

### 1. Purpose

The CPU backend is the correctness reference. It is also 182 times slower than
the PyTorch reference for base training. This plan closes most of that gap
without a third-party runtime dependency.

The plan has stages. Phase 1 is the critical change and carries a kill gate.
The later phases are optional.

---

### 2. The measured problem

The gap and the configuration are in Part 1.

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

### 3. The cause

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

### 4. Design

#### 4.1 The GEMM change

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

#### 4.2 Threading contract

Parallelize the output-row loop with OpenMP. `docs/build.md` already lists
OpenMP as an allowed CPU dependency.

The thread count comes from the environment:

- `tools/sandbox.sh` sets `OMP_NUM_THREADS` for every profile.
- The same wrapper sets `NANOCHAT_NUM_THREADS` for library thread pools.
- If neither variable exists, OpenMP uses the hardware concurrency.

Do not hardcode a thread count. Do not create threads per `Gemm` call. Let
OpenMP manage the pool.

The build must add `-fopenmp` to the CPU backend. The file is
`backends/cpu/BUILD.bazel`, which the Runtime workstream owns.

#### 4.3 Numerics

The `double` accumulator keeps the summation precision of the shipped code.
The reordered summation changes the rounding slightly, but at `double`
precision the difference is negligible.

Do not switch to an fp32 accumulator in Phase 1. At `K=4096` the fp32
accumulator differs from the shipped code by 5.7e-5. The oracle tolerance is
1e-5. An fp32 accumulator needs a full parity check first.

#### 4.4 Later stages

Phase 2 adds register blocking and operand packing. This is the path to the
OpenBLAS rate.

Phase 3 accelerates `MuonUpdate`, the classifier, and attention. These become
the largest items after Phase 1.

---

### 5. Ownership and interfaces

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

### 6. Plan

Every phase runs through `tools/nanochat`. Tests use the T0 CPU profile.
Benchmarks use the `train` or `t3-bench` profile. See
[performance.md](performance.md) and [testing.md](testing.md).

#### Phase 0 — benchmark harness (0.5 session)

- Promote the microbenchmark to `dev/kernels/cpu_gemm_bench.cc`. Time the four
  dominant shapes. Report the `nanochat.bench.v1` schema.
- Add a `cc_binary` target to `dev/kernels/BUILD.bazel`.
- Record the shipped numbers as `docs/cpu-gap-baseline.json`.
- **Gate:** the benchmark reproduces the section 2 numbers within 10%.

#### Phase 1 — reordered, threaded GEMM (1 session, the critical gate)

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

#### Phase 2 — blocking and packing (2 sessions)

- Add register blocking and operand packing to `Gemm`.
- Keep the parity gate green after each change.
- **Kernel gate:** the `lm_head` `dgrad` shape must reach at least 80 GFLOP/s.
  The measured OpenBLAS ceiling is 414.
- Consider an fp32 accumulator only with the full parity check. Keep the
  `double` path if parity fails.

#### Phase 3 — optimizer and elementwise families (2 sessions)

- Thread and vectorize `MuonUpdate`. **Gate:** at most 2 s per step. The
  baseline is 12.7 s.
- Vectorize `ClassifierForward` and `ClassifierBackward`. **Gate:** at most
  0.5 s per step. The baseline is 2.3 s.
- Vectorize `AttentionForward` and `AttentionBackward`. **Gate:** at most
  0.4 s per step. The baseline is 1.6 s.
- **End-to-end gate:** the 10-step CPU run must reach at most 5 s per step.

#### Phase 4 — promote and record (0.5 session)

- Re-measure with the protocol in section 9.
- Update `docs/cpu-baseline.json` and `docs/cpu-performance.md` with the revision.
- **Gate:** the end-to-end step is at most 5 s, and every correctness test in
  section 8 passes.

---

### 7. Acceptance criteria

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

### 8. Correctness gates

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

### 9. Measurement protocol

1. Record the revision with `tools/nanochat doctor`.
2. Confirm the device is idle and no other CPU job runs.
3. Warm up. Then time a batch with the best-of-N rule.
4. Report the best round for throughput. State the thread count.
5. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` together for the PyTorch side.
   Without the second setting the first run reached only 1,049 tokens/s.
6. Keep the before and after numbers with the revision and the configuration.

---

### 10. Risks and mitigations

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

### 11. Alternatives considered and rejected

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

### 12. Reproduction

The microbenchmark sources are in `/tmp/nanochat-cpu-bench/`. Compile and run
them against the CPU backend:

```bash
cd ~/repos/nanochat.cpp
g++ -O2 -std=c++20 -fopenmp -DNANOCHAT_PRECISION_FP32 -Iinclude -I. \
  backends/cpu/kernels.cc /tmp/nanochat-cpu-bench/gemm_omp.cc \
  -o /tmp/nanochat-cpu-bench/gemm_omp
tools/nanochat run train -- /tmp/nanochat-cpu-bench/gemm_omp
```

The end-to-end run is the bridge command in Part 1, step 4.

---

### 13. References

- The shipped GEMM: `backends/cpu/kernels.cc`, function `Gemm`.
- The kernel seam: `include/nanochat/kernels.h`, function `Gemm`.
- The measurement protocol: [performance.md](performance.md).
- The test tiers: [testing.md](testing.md).
- The backend rules: [build.md](build.md), [DESIGN.md](../DESIGN.md).
- The ownership map: [AGENTS.md](../AGENTS.md).
- The measured baseline: Part 1.

## Part 3. Phase results

Status: **Phase 1 PASS, Phase 2 PASS, Phase 3 PASS.** Phase 1 passes the kernel
gate, the correctness gate, and the end-to-end gate. The 10-step CPU run reaches
22.3 seconds per step. The target is 29 seconds. The speedup is 6.4 times.
Phase 2 passes the kernel gate and keeps the parity gate green (section 7).
Phase 3 passes the three family gates, the end-to-end gate at 1.733 seconds per
step, and the full CPU suite (section 8).

This part records the Phase 1 kill gate from the plan in Part 2, section 6. The gate has three parts: the four kernel rates, the three
correctness tests, and the end-to-end step time. The gate also covers the
model-layout forward shape (`transpose_b = true`).

### Configuration

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

### 1. Kernel gate — PASS

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

### 2. Correctness gate — PASS

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

### 3. End-to-end gate — PASS

Command (the bridge in Part 1, step 4):

```bash
tools/nanochat_cpp base_train --backend cpu \
  --depth=4 --max-seq-len=512 --window-pattern=L \
  --device-batch-size=2 --total-batch-size=1024 --num-iterations=10 \
  --warmup-steps=1 --eval-every=-1 --core-metric-every=-1 \
  --sample-every=-1 --save-every=-1 --model-tag=cpu_bench_cpp
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

### 4. Root cause and fix

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

### 5. Verdict

**Phase 1 kill gate: PASS.**

- Kernel gate: pass. All four design shapes and both model-layout shapes
  exceed 15 GFLOP/s.
- Correctness gate: pass. The three tests pass and the frozen seam is
  unchanged.
- End-to-end gate: pass. 22.3 seconds per step against the 29-second target.
  The speedup is 6.4 times, not 5 times.

Promote the Phase 1 `Gemm` and continue to Phase 2. Phase 2 must keep the
model-layout rows in `cpu_gemm_bench` and keep the section 8 tests green.

### 6. Reproduction

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

### 7. Phase 2 — blocking and packing — PASS

The Phase 2 change adds register blocking and operand packing to the CPU
`Gemm`. The output is tiled into 4 by 16 register tiles. The `b` operand is
packed per column block into a contiguous panel. The `a` operand is packed per
row block. An AVX2 plus fused-multiply-add micro-kernel runs on a supported
host. A portable micro-kernel is the fallback. OpenMP parallelizes the output
row blocks.

The Phase 2 kernel gate is 80 GFLOP per second for the language-model head
`dgrad` shape. The measured OpenBLAS ceiling is 414 GFLOP per second.

#### Kernel gate — PASS

Command:

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gap-p2.json
```

The run uses the `t3-bench` profile with 12 threads and fp32 precision. The
table gives the four dominant shapes from Part 2, section 2.

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

#### Accumulator type

The reduction stays in one `double` accumulator. The factory packs the whole
reduction into a single `double` sum over `k`. This matches the rounding of the
shipped scalar loop. Phase 2 adds no `float` accumulator path. The comments in
`backends/cpu/kernels.cc` state the `double` accumulator.

#### Correctness gate — PASS

The parity gate stays green. The three tests pass after the Phase 2 change:

```bash
tools/nanochat test --nocache_test_results \
  //backends/cpu:kernels_test //tests:oracle_test //tests:train_parity_test
```

`train_parity_test` is the strict gate. It compares the loss, the gradient, and
the parameter trajectory against the PyTorch reference. The frozen seam
`include/nanochat/kernels.h` is unchanged.

#### Verdict

**Phase 2 kernel gate: PASS.** The `gemm_dgrad` rate is 130.85 GFLOP per
second, above the 80 GFLOP per second target. The OpenBLAS ceiling is 414
GFLOP per second. The parity gate passes.

#### Reproduction

The raw benchmark report is `/tmp/cpu-gap-p2.json`. The report uses the
`nanochat.bench.v1` schema with `threads=12`.

---

### 8. Phase 3 — optimizer and elementwise families — PASS

Revision `283a511`, branch `sliceme/agent-cpu-backend`. The Phase 3 change
threads and vectorizes `MuonUpdate`, `ClassifierForward`,
`ClassifierBackward`, `AttentionForward`, and `AttentionBackward`.

#### End-to-end gate — PASS

Command (the bridge in Part 1, step 4):

```bash
tools/nanochat_cpp base_train --backend cpu \
  --depth=4 --max-seq-len=512 --window-pattern=L \
  --device-batch-size=2 --total-batch-size=1024 --num-iterations=10 \
  --warmup-steps=1 --eval-every=-1 --core-metric-every=-1 \
  --sample-every=-1 --save-every=-1 --model-tag=cpu_bench_cpp
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

#### Family gates — PASS

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

#### Correctness gate — PASS

The full CPU suite passes. All 16 tests pass, including the strict
`train_parity_test`.

```bash
tools/nanochat test
```

#### Verdict

**Phase 3 end-to-end gate: PASS.** The step time is 1.733 seconds against the
5-second target. The three family gates pass. The full CPU suite passes.
