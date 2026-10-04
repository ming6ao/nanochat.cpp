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
| `row_kernels_test.cc` | Smoke test for the row families over the sealed seam. |
| `precision_test.cc` | Float reference for the elementwise, norm, attention, and embedding families, in both precisions. |
| `sequence_oracle_test.cc` | Classifier and Embedding against `tests/data/debug_state.bin`. |
| `testing/row_ref.h` | Host reference for the row and elementwise families. |
| `testing/sequence_ref.h` | Host reference for the attention, classifier, and embedding families. |
| `testing/optim_ref.h` | Host reference for AdamW and Muon. |
| `testing/gpu_test_utils.h` | Device buffer, deterministic random source, tolerance check, finite-difference helper. |
| `testing/bench_utils.h` | CUDA-event timer and the `nanochat.bench.v1` report. |

## Conventions

- **Tiny shapes by default.** The attention-shaped kernels use `B=2, T=8`,
  `head_dim=4`, `num_heads=2`. They cost microseconds, so they are gates rather
  than iteration tools.
- **Compare against a host reference.** The `testing/*_ref.h` headers are the
  device-independent specification. A correctness test compares the device
  output to them. This keeps the oracle torch-free and the test deterministic.
- **Finite-difference every backward.** Check the analytic gradients against
  central differences of the reference forward (`h = 1e-2`, relative tolerance
  `2e-3`). Finite differences run in fp32 only. The fp16 storage cannot resolve
  the perturbation.
- **Fixed seeds.** The `Rng` type is a deterministic xorshift, so a failure
  reproduces.
- **Tag GPU tests `gpu`.** `tools/nanochat test` filters them out with
  `--test_tag_filters=-gpu`. `tools/nanochat test --gpu //...` runs only them.
  Do not add the `manual` tag. Bazel excludes `manual` targets from wildcard
  expansion, so the acceptance command would miss them.
- **Benchmarks are not tests.** Use `testing/bench_utils.h`. Run them through
  `tools/nanochat bench` or `tools/nanochat profile`. Do not add a benchmark to
  the `all` test suite.

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
