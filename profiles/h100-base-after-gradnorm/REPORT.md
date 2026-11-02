# The gradient-norm fix: one device reduce instead of a host pass

Status: measured. One change to `src/ops.cc` and the base stage's MFU rises
from 0.24 to 1.30 percent. The traced step falls from 414.6 to 76.9 ms, and the
gradient-norm block stops being 70 percent of the step.

The change is the one `profiles/h100-base-20step/REPORT.md` section 4.1 asks
for, taken through the seam entry that already existed rather than through a new
primitive.

## 1. The change

`ops::GradNormForward` summed the squares of every gradient span on the host in
a bf16 build. It did so because `kernels::SpanReduce` takes `ComputeType`
operands, which are bf16 in a bf16 build, while the gradients are fp32.

`kernels::GradNorm` takes an fp32 span list, reads the span table on the device,
and accumulates in double. The seam declared it and the backend implemented it
before this work; nothing in `src/` called it.

`src/ops.cc` now calls it for every precision:

```cpp
void GradNormForward(const float* const* grads, const int* counts,
                     int num_spans, float clip, float* out_norm,
                     float* out_scale) {
  if (grads == nullptr || counts == nullptr || num_spans <= 0) return;
  kernels::GradNorm(grads, counts, num_spans, clip, out_norm, out_scale);
}
```

The `if constexpr`, the `RaggedSumOfSquares` helper, and the host loop are gone.
`src/ops.h` and the comment above the call in `src/optim.cc` now describe this
call. The clip scale still stays on the device for the whole step.

## 2. The run

| | Before | After |
|---|---|---|
| Run | `profile-base` | `after-gradnorm` |
| Job | base stage, 20 steps, batch 16, profile `h100` | identical |
| Cards | one H100, bf16 | identical |
| Instrument | Nsight Systems 2023.4.4, `--trace=cuda,osrt` | identical |

```bash
modal run tools/modal/app.py::profile_demo \
  --job /tmp/mrun/after-job.json --run-id after-gradnorm --steps 20
```

## 3. The result

Per-step numbers come from `metrics.jsonl`. Trace numbers are the median of the
19 steady step windows, cut at the optimizer phase by `analyze_step.py`.

| Per step | Before | After | Change |
|---|---|---|---|
| Step wall, engine mean (traced) | 447.3 ms | 102.3 ms | 4.4 times |
| Step wall, trace median (steady) | 414.6 ms | 76.9 ms | 5.4 times |
| MFU, engine mean | 0.243 percent | 1.295 percent | 5.3 times |
| MFU, `summary.json` steady | 0.251 percent | 1.391 percent | 5.6 times |
| Model rate | 2.5 TFLOP/s | 13.7 TFLOP/s | 5.5 times |
| Tokens per second | 9 567 | 50 903 | 5.3 times |

The trace budget of one steady step:

| Component | Before | After |
|---|---|---|
| Device busy, union | 75.4 ms (18.2%) | 38.7 ms (50.3%) |
| Kernels | 36.9 ms (8.9%) | 36.6 ms (47.6%) |
| Device-to-host copies | 37.3 ms (9.0%) | 0.9 ms (1.2%) |
| Host inside driver calls | 92.1 ms (22.2%) | 33.3 ms (43.3%) |
| Host in its own code | 323.8 ms (78.1%) | 42.6 ms (55.5%) |

The host code breaks down by the call that precedes it:

| Host-only work after | Before | After |
|---|---|---|
| A device-to-host copy of 20 MB or more | 182.8 ms | 2.4 ms |
| A device-to-host copy of a span | 98.2 ms | 13.1 ms |
| The window opens on a copy | 14.0 ms | 0.0 ms |
| A kernel launch | 27.9 ms | 23.3 ms |
| A kernel-function lookup | 7.4 ms | 6.1 ms |

The gradient-norm block was 295.0 ms of host-only time plus 60 ms inside the
copies: 86 percent of the step. It is now under 3 ms.

Traffic tells the same story: 115 device-to-host copies and 374 MB per step
before, 50 copies and 12.9 MB after.

## 4. Parity

The same seed and the same corpus give the same numbers.

| | Value |
|---|---|
| Worst loss difference over 20 steps | 0.0 |
| Worst gradient-norm difference over 20 steps | 5.96e-08 |
| Mean loss, before and after | 8.086975169181823 |

The gradient norm agrees to one float ulp, which is the difference between a
double accumulation on the host and a double accumulation on the device. The
loss is unchanged, so the clip scale's low bits do not move the run.

## 5. Gates

| Gate | Result |
|---|---|
| Build, `//src:model //src:optim_test` | pass |
| Build, `--config=hopper` (sm_90, bf16) | pass |
| `//src:optim_test` | pass |
| `//backends/cpu:kernels_test` | pass |
| `tools/nanochat lint` on the changed files | pass |
| `simulate --precision bf16 --suite correctness` | `sim_numerics_test` segfaults, on the parent tree too |
| `simulate --precision fp32 --suite correctness` | `numerics_trace_test` fails 762 gradient-hash records, on the parent tree too |

Both simulator failures reproduce with `src/ops.cc` restored from `HEAD`, so
neither belongs to this change. The fp32 exact-golden failure is the gradient
hashes of a tree that carries the uncommitted R2 work; the bf16 leg crashes
before it compares anything.

## 6. What is left

The step is still host-bound: 76.9 ms of wall time against 38.7 ms of device
busy. Two costs remain, in order.

1. **The row-scalar staging in `src/ops.cc` (`RowScalars`).** In a bf16 build
   `ToCompute` and `ToFloat` convert through the host, one device-to-host copy
   and one host loop per call. It is the same gap as the gradient norm, at a
   smaller size: 12.9 MB and 13.1 ms of host-only time per step, 17 percent of
   the step.
2. **The launch path.** 7 882 launches, 7 281 kernel-function lookups, and 107
   `cudaMalloc` pairs per step. The lookups are one per launch and a cached
   handle removes them. `cudaMalloc` now costs 8.1 ms per step against 2.6 ms
   before, because it waits for a device that is no longer idle.

Kernel work is now worth optimizing, for the first time. The device spends
36.6 ms per step, and 25.2 ms of it is 6 821 cuBLAS GEMMs on 32x32 tiles that
fill 16 to 64 of 132 SMs. Grouped GEMMs would remove most of that 25.2 ms.

## 7. Instruments

```bash
python profiles/h100-base-after-gradnorm/analyze_step.py <trace_dir> 20
python profiles/h100-base-after-gradnorm/compare_runs.py \
  profiles/h100-base-20step/metrics.jsonl \
  profiles/h100-base-after-gradnorm/metrics.jsonl
```

`analyze_step.py` cuts every trace at the optimizer phase, so it measures a run
whose step stages its gradients and one whose step does not. `compare_runs.py`
holds the numeric gate.
