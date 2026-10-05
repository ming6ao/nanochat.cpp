# CPU backend gap-phase results

Status: **Phase 1 PASS, Phase 2 PASS, Phase 3 PASS.** Phase 1 passes the kernel
gate, the correctness gate, and the end-to-end gate. The 10-step CPU run reaches
22.3 seconds per step. The target is 29 seconds. The speedup is 6.4 times.
Phase 2 passes the kernel gate and keeps the parity gate green (section 7).
Phase 3 passes the three family gates, the end-to-end gate at 1.733 seconds per
step, and the full CPU suite (section 8).

This document records the Phase 1 kill gate from [cpu-gap.md](cpu-gap.md)
section 6. The gate has three parts: the four kernel rates, the three
correctness tests, and the end-to-end step time. The gate also covers the
model-layout forward shape (`transpose_b = true`).

## Configuration

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

## 1. Kernel gate — PASS

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

## 2. Correctness gate — PASS

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

## 3. End-to-end gate — PASS

Command (the bridge in [cpu-baseline.md](cpu-baseline.md) step 4):

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

## 4. Root cause and fix

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

## 5. Verdict

**Phase 1 kill gate: PASS.**

- Kernel gate: pass. All four design shapes and both model-layout shapes
  exceed 15 GFLOP/s.
- Correctness gate: pass. The three tests pass and the frozen seam is
  unchanged.
- End-to-end gate: pass. 22.3 seconds per step against the 29-second target.
  The speedup is 6.4 times, not 5 times.

Promote the Phase 1 `Gemm` and continue to Phase 2. Phase 2 must keep the
model-layout rows in `cpu_gemm_bench` and keep the section 8 tests green.

## 6. Reproduction

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

## 7. Phase 2 — blocking and packing — PASS

The Phase 2 change adds register blocking and operand packing to the CPU
`Gemm`. The output is tiled into 4 by 16 register tiles. The `b` operand is
packed per column block into a contiguous panel. The `a` operand is packed per
row block. An AVX2 plus fused-multiply-add micro-kernel runs on a supported
host. A portable micro-kernel is the fallback. OpenMP parallelizes the output
row blocks.

The Phase 2 kernel gate is 80 GFLOP per second for the language-model head
`dgrad` shape. The measured OpenBLAS ceiling is 414 GFLOP per second.

### Kernel gate — PASS

Command:

```bash
tools/nanochat build //dev/kernels:cpu_gemm_bench
tools/nanochat bench -- ./bazel-bin/dev/kernels/cpu_gemm_bench \
  --json --out /tmp/cpu-gap-p2.json
```

The run uses the `t3-bench` profile with 12 threads and fp32 precision. The
table gives the four dominant shapes from [cpu-gap.md](cpu-gap.md) section 2.

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

### Accumulator type

The reduction stays in one `double` accumulator. The factory packs the whole
reduction into a single `double` sum over `k`. This matches the rounding of the
shipped scalar loop. Phase 2 adds no `float` accumulator path. The comments in
`backends/cpu/kernels.cc` state the `double` accumulator.

### Correctness gate — PASS

The parity gate stays green. The three tests pass after the Phase 2 change:

```bash
tools/nanochat test --nocache_test_results \
  //backends/cpu:kernels_test //tests:oracle_test //tests:train_parity_test
```

`train_parity_test` is the strict gate. It compares the loss, the gradient, and
the parameter trajectory against the PyTorch reference. The frozen seam
`include/nanochat/kernels.h` is unchanged.

### Verdict

**Phase 2 kernel gate: PASS.** The `gemm_dgrad` rate is 130.85 GFLOP per
second, above the 80 GFLOP per second target. The OpenBLAS ceiling is 414
GFLOP per second. The parity gate passes.

### Reproduction

The raw benchmark report is `/tmp/cpu-gap-p2.json`. The report uses the
`nanochat.bench.v1` schema with `threads=12`.

---

## 8. Phase 3 — optimizer and elementwise families — PASS

Revision `283a511`, branch `sliceme/agent-cpu-backend`. The Phase 3 change
threads and vectorizes `MuonUpdate`, `ClassifierForward`,
`ClassifierBackward`, `AttentionForward`, and `AttentionBackward`.

### End-to-end gate — PASS

Command (the bridge in [cpu-baseline.md](cpu-baseline.md) step 4):

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

### Family gates — PASS

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

### Correctness gate — PASS

The full CPU suite passes. All 16 tests pass, including the strict
`train_parity_test`.

```bash
tools/nanochat test
```

### Verdict

**Phase 3 end-to-end gate: PASS.** The step time is 1.733 seconds against the
5-second target. The three family gates pass. The full CPU suite passes.
