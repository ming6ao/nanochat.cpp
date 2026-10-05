# CPU backend performance: design and plan

This document turns the CPU gap analysis into a work order. It states the
measured problem, the design of the fix, the work phases, the owner of each
phase, and the numeric gate that closes it.

Status: proposal. No code change exists yet. The analysis has measurements.
The predictions do not.

Related documents:

- [cpu-baseline.md](cpu-baseline.md): the measured baseline.
- [performance.md](performance.md): the measurement protocol.
- [testing.md](testing.md): the test tiers and the definition of done.
- [backends.md](backends.md): backend selection.
- [DESIGN.md](../DESIGN.md) section 2: the dependency rules.
- [AGENTS.md](../AGENTS.md): the ownership map.

---

## 1. Purpose

The CPU backend is the correctness reference. It is also 182 times slower than
the PyTorch reference for base training. This plan closes most of that gap
without a third-party runtime dependency.

The plan has stages. Phase 1 is the critical change and carries a kill gate.
The later phases are optional.

---

## 2. The measured problem

The gap and the configuration are in [cpu-baseline.md](cpu-baseline.md).

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

## 3. The cause

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

## 4. Design

### 4.1 The GEMM change

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

### 4.2 Threading contract

Parallelize the output-row loop with OpenMP. `docs/backends.md` already lists
OpenMP as an allowed CPU dependency.

The thread count comes from the environment:

- `tools/sandbox.sh` sets `OMP_NUM_THREADS` for every profile.
- The same wrapper sets `NANOCHAT_NUM_THREADS` for library thread pools.
- If neither variable exists, OpenMP uses the hardware concurrency.

Do not hardcode a thread count. Do not create threads per `Gemm` call. Let
OpenMP manage the pool.

The build must add `-fopenmp` to the CPU backend. The file is
`backends/cpu/BUILD.bazel`, which the Runtime workstream owns.

### 4.3 Numerics

The `double` accumulator keeps the summation precision of the shipped code.
The reordered summation changes the rounding slightly, but at `double`
precision the difference is negligible.

Do not switch to an fp32 accumulator in Phase 1. At `K=4096` the fp32
accumulator differs from the shipped code by 5.7e-5. The oracle tolerance is
1e-5. An fp32 accumulator needs a full parity check first.

### 4.4 Later stages

Phase 2 adds register blocking and operand packing. This is the path to the
OpenBLAS rate.

Phase 3 accelerates `MuonUpdate`, the classifier, and attention. These become
the largest items after Phase 1.

---

## 5. Ownership and interfaces

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

## 6. Plan

Every phase runs through `tools/nanochat`. Tests use the T0 CPU profile.
Benchmarks use the `train` or `t3-bench` profile. See
[performance.md](performance.md) and [testing.md](testing.md).

### Phase 0 — benchmark harness (0.5 session)

- Promote the microbenchmark to `dev/kernels/cpu_gemm_bench.cc`. Time the four
  dominant shapes. Report the `nanochat.bench.v1` schema.
- Add a `cc_binary` target to `dev/kernels/BUILD.bazel`.
- Record the shipped numbers as `docs/cpu-gap-baseline.json`.
- **Gate:** the benchmark reproduces the section 2 numbers within 10%.

### Phase 1 — reordered, threaded GEMM (1 session, the critical gate)

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

### Phase 2 — blocking and packing (2 sessions)

- Add register blocking and operand packing to `Gemm`.
- Keep the parity gate green after each change.
- **Kernel gate:** the `lm_head` `dgrad` shape must reach at least 80 GFLOP/s.
  The measured OpenBLAS ceiling is 414.
- Consider an fp32 accumulator only with the full parity check. Keep the
  `double` path if parity fails.

### Phase 3 — optimizer and elementwise families (2 sessions)

- Thread and vectorize `MuonUpdate`. **Gate:** at most 2 s per step. The
  baseline is 12.7 s.
- Vectorize `ClassifierForward` and `ClassifierBackward`. **Gate:** at most
  0.5 s per step. The baseline is 2.3 s.
- Vectorize `AttentionForward` and `AttentionBackward`. **Gate:** at most
  0.4 s per step. The baseline is 1.6 s.
- **End-to-end gate:** the 10-step CPU run must reach at most 5 s per step.

### Phase 4 — promote and record (0.5 session)

- Re-measure with the protocol in section 9.
- Update `docs/cpu-baseline.json` and `docs/cpu-baseline.md` with the revision.
- **Gate:** the end-to-end step is at most 5 s, and every correctness test in
  section 8 passes.

---

## 7. Acceptance criteria

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

## 8. Correctness gates

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

## 9. Measurement protocol

1. Record the revision with `tools/nanochat doctor`.
2. Confirm the device is idle and no other CPU job runs.
3. Warm up. Then time a batch with the best-of-N rule.
4. Report the best round for throughput. State the thread count.
5. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` together for the PyTorch side.
   Without the second setting the first run reached only 1,049 tokens/s.
6. Keep the before and after numbers with the revision and the configuration.

---

## 10. Risks and mitigations

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

## 11. Alternatives considered and rejected

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

## 12. Reproduction

The microbenchmark sources are in `/tmp/nanochat-cpu-bench/`. Compile and run
them against the CPU backend:

```bash
cd ~/repos/nanochat.cpp
g++ -O2 -std=c++20 -fopenmp -DNANOCHAT_PRECISION_FP32 -Iinclude -I. \
  backends/cpu/kernels.cc /tmp/nanochat-cpu-bench/gemm_omp.cc \
  -o /tmp/nanochat-cpu-bench/gemm_omp
tools/nanochat run train -- /tmp/nanochat-cpu-bench/gemm_omp
```

The end-to-end run is the bridge command in
[cpu-baseline.md](cpu-baseline.md).

---

## 13. References

- The shipped GEMM: `backends/cpu/kernels.cc`, function `Gemm`.
- The kernel seam: `include/nanochat/kernels.h`, function `Gemm`.
- The measurement protocol: [performance.md](performance.md).
- The test tiers: [testing.md](testing.md).
- The backend rules: [backends.md](backends.md), [DESIGN.md](../DESIGN.md).
- The ownership map: [AGENTS.md](../AGENTS.md).
- The measured baseline: [cpu-baseline.md](cpu-baseline.md).
