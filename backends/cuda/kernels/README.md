# backends/cuda/kernels — the CUDA kernel families

This package holds one translation unit per kernel family. Each family keeps
its correctness test and its micro-benchmark next to the kernel, so a reader
finds all three in one place. Host-only test scaffolding lives in `testing/`.

See [docs/kernels.md](../../../docs/kernels.md) for the kernel inventory and
[docs/testing.md](../../../docs/testing.md) for the test tiers.

## Layout

| File | What |
|---|---|
| `<family>.cu` | The production kernel. |
| `<family>_test.cc` | Tiny-shape correctness and finite differences against the host reference. |
| `<family>_benchmark.cc` | Device-event timing. A benchmark, not a test. |
| `muon_fused.h`, `muon_fused.cu` | The fused Muon update: one phase-parameterised device function, its 16 phases and three epilogue folds, and the two drivers over one device body. |
| `muon_fused_test.cc` | Three implementations of one Muon step on one input: the host reference, the shipped multi-launch `MuonUpdate`, and the fused path through both drivers, plus the workspace check and the forced cooperative refusal. |
| `row_kernels_test.cc` | Smoke test for the row families over the sealed seam. |
| `precision_test.cc` | Float reference for the elementwise, norm, attention, and embedding families, in both precisions. |
| `sequence_oracle_test.cc` | Classifier and Embedding against `tests/data/debug_state.bin`. |
| `testing/row_ref.h` | Host reference for the row and elementwise families. |
| `testing/sequence_ref.h` | Host reference for the attention, classifier, and embedding families. |
| `testing/optim_ref.h` | Host reference for AdamW and Muon. |
| `testing/gpu_test_utils.h` | Device buffer, deterministic random source, tolerance check, finite-difference helper. |
| `testing/bench_utils.h` | CUDA-event timer, and the `BenchReport` alias for the `nanochat.bench.v3` report in `nanochat/bench.h`. |

## The fused Muon path

`muon_fused.cu` runs the whole `MuonUpdate` stage as one phase-parameterised
device function instead of the thirty-nine launches the multi-launch path in
`muon.cu` issues at the shipped `ns_steps` of 5. Section 4 of
[kernels.md](../../../docs/kernels.md) is the design; this
note records what landed, what it costs, and why it is not the shipped default.

The stage becomes sixteen phases, which run as 28 executions (seven before the
Polar Express loop, three per iteration, six after) and therefore 27 grid
barriers in the cooperative driver:

| Phase | Work |
|---|---|
| 1 | Momentum and the Nesterov combination |
| 2 to 3 | Partial pass and finish for `‖x‖_F` |
| 4 | Row equilibration, one block per `(matrix, row)` |
| 5 to 6 | Partial pass and finish for `1.01‖x‖ + 1e-6` |
| 7 | Divisor scale |
| 8 | `A = XᵀX` (tall) or `XXᵀ` (wide), batched GEMM |
| 9 | `A·A`, with the `b = cb·a + cc·acc` epilogue |
| 10 | `X·b` (tall) or `b·X` (wide), with the `xnext = ca·x + prod` epilogue |
| 11 to 13 | Renormalisation: partials, finish, scale |
| 14 | NorMuon mean squares, per row or per column |
| 15 | The factored second moment and the per-index step |
| 16 | The NorMuon scale, with the cautious update as its epilogue |

Three epilogue folds are what the fusion buys beyond the launches: `b` is
written from the phase 9 accumulator (the `a2` buffer and one pass over
`min_dim²` per matrix go), the combined iterate is written into the ping-pong
partner by phase 10 (the `prod` buffer becomes that partner, and one pass over
`total` goes), and the cautious update is folded into the NorMuon scale, which
is bit exact because neither stage accumulates. The workspace falls from
`2·total + 3·min_total` to `2·total + 2·min_total`, and both paths now use that
one layout: `MuonUpdate` writes `A·A` into `b` in place, which is the same fold
without the accumulator.

Both drivers run one device body over one workspace, so a group can move between
them without a caller noticing:

* the **cooperative** driver launches one grid per group, sized from the
  resident-block query in `backends/cuda/device.h` (the active blocks per SM
  times the SM count of the card it runs on), and puts a `grid.sync()` between
  the executions. A launch the device refuses falls back to the phase driver and
  logs one line;
* the **phase** driver launches one grid per execution, each sized for its own
  phase. It is the simulator's path (`tools/cuda_sim` answers 0 for
  `cudaDevAttrCooperativeLaunch`, and a grid barrier would deadlock in its
  block-at-a-time engine), the fallback above, and the phase-bisection tool.

### The gate

```bash
tools/nanochat test --gpu //backends/cuda/kernels:muon_fused_gpu_test
```

`muon_fused_test.cc` runs three implementations of one Muon step on one input
and one seed -- the host reference `testing/optim_ref.h`, the shipped
multi-launch `MuonUpdate`, and the fused path through each driver -- over
`4x3`, `3x6`, and `5x5` groups, both `red_dim` values, Nesterov on and off,
`ns_steps` of 0, 1, and 5, and `num_params` of 1 and 3, at the family's 1e-5
tolerance. `5x5` is not a whole number of 16-byte groups, so it runs the scalar
elementwise phases and the others run the vector ones. On top of the tolerance
comparison:

* the two drivers have to agree with each other **exactly** (bit for bit), since
  only the launch geometry differs;
* every output has to be finite. A `CheckVectorClose` comparison against a NaN
  is false for every tolerance, so a divergent stage is caught by name;
* the workspace check hands the fused path exactly
  `2·total + 2·min_total` elements with a 64-element sentinel tail, so a phase
  or a fold that reached past the last slice changes the tail;
* a cooperative launch of eight times the resident grid is refused with
  `cudaErrorCooperativeLaunchTooLarge`, and the fallback has to produce the
  phase driver's answer and report `fell_back`.

### The fused path, measured

The rows were measured on a card the tree no longer supports, so the numbers are
removed. Measure them again on a supported card before you cite one:

```bash
tools/nanochat build //backends/cuda/kernels:muon_benchmark
tools/nanochat bench -- bazel-bin/backends/cuda/kernels/muon_benchmark \
  --profile h100 --muon --json
tools/nanochat bench -- bazel-bin/backends/cuda/kernels/muon_benchmark \
  --profile big --muon --json
```

The `muon:` rows are the shipped multi-launch path and the `muon_fused:` rows are
the fused path over the same buffers, both at the shipped `ns_steps` of 5 unless
the row says otherwise.

**Verdict: implemented, gated, and not promoted.** The fused path was slower for
every large group of both profiles, so `MuonFusedSelected` -- the shape rule of
section 4, whose one number is `kMuonFusedMaxElements` in `muon_fused.cu` --
selects the multi-launch path (`kernels::Gemm`, i.e. cuBLAS) for every shape and
the constant is 0. Two of the three costs are the ones section 3 predicted and
`dev/kernels/README.md` measured at Phase 1: the hand GEMM falls short of the
cuBLAS rate on the groups the fusion targets, and the elementwise passes the
folds remove are worth less than that gap. The third is new here: even the
`ns=0` step, where no GEMM runs and the folds have nothing to fold, was slower,
because one kernel serves sixteen phases and its 15 KiB of static shared memory
holds six blocks per SM where the elementwise families' own kernels hold eight.
One block size fits no stage, as section 9's risk table says.

The fused path stays in the tree, tested through both drivers and measured group
by group, because it is the shape-rule and phase-bisection tool the design keeps
and because a different card, a different GEMM schedule, or a group above the
one-block-size cost is one measurement away: raising `kMuonFusedMaxElements`
above a group's element count is the whole promotion.

## Conventions

- **Tiny shapes by default.** The attention-shaped kernels use `B=2, T=8`,
  `head_dim=4`, `num_heads=2`. They cost microseconds, so they are gates rather
  than iteration tools.
- **Compare against a host reference.** The `testing/*_ref.h` headers are the
  device-independent specification. A correctness test compares the device
  output to them. This keeps the oracle torch-free and the test deterministic.
- **Finite-difference every backward.** Check the analytic gradients against
  central differences of the reference forward (`h = 1e-2`, relative tolerance
  `2e-3`). Finite differences run in fp32.
- **Fixed seeds.** The `Rng` type is a deterministic xorshift, so a failure
  reproduces.
- **Tag GPU tests `gpu`.** `tools/nanochat test` filters them out with
  `--test_tag_filters=-gpu`. `tools/nanochat test --gpu //...` runs only them.
  Do not add the `manual` tag. Bazel excludes `manual` targets from wildcard
  expansion, so the acceptance command would miss them.
- **Benchmarks are not tests.** Use `testing/bench_utils.h`. Run them through
  `tools/nanochat bench` or `tools/nanochat profile`. Do not add a benchmark to
  the `all` test suite.
- **A `wmma` fragment stays opaque.** Pass a fragment only to `fill_fragment`,
  `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`. The emulator of
  [simulator.md](../../docs/simulator.md) section 9.4 holds one whole micro-tile
  in every fiber, so a kernel that reads one element of a fragment depends on the
  vendor layout and the emulator would agree with it by accident. That rule is
  what lets `//tools/cuda_sim/emu:gemm_emu_test` gate the hand GEMM off device.

## Run the tests

```bash
tools/nanochat test                                    # T0 CPU tests
tools/nanochat test --gpu //backends/cuda/kernels/...  # T1 GPU tests
tools/nanochat bench -- ./bazel-bin/backends/cuda/kernels/row_benchmark
tools/nanochat profile                                 # the attention benchmark
```

## Add a family

1. Add the CUDA translation unit under `backends/cuda/kernels/` and its
   `cuda_library` target (see `BUILD.bazel`).
2. Add or extend the host reference under `testing/`.
3. Add `<family>_test.cc` next to the kernel. Include a correctness check and,
   when the family has a backward, a finite-difference check. Add the target to
   the `all` test suite.
4. Optionally add `<family>_benchmark.cc`.
