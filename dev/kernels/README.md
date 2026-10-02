# dev/kernels — standalone per-kernel tests and benchmarks

This directory follows the `llm.c` `dev/cuda` convention: each kernel family
gets its own small, self-contained test and (optionally) a benchmark, separate
from the full-graph tests in `//tests`. The point is to make a single family
fast to reason about and cheap to gate.

The canonical tiers, tagging, and resource rules are in
[docs/testing.md](../../docs/testing.md) and [AGENTS.md](../../AGENTS.md) §5.
Run everything through the project entry point:

```bash
tools/nanochat test                      # T0 CPU tests (excludes tag `gpu`)
tools/nanochat test --gpu //...          # T1 GPU tests (tag `gpu`)
tools/nanochat bench -- <row_bench>      # T3-style micro-benchmark under the broker
```

## Layout

| File | What |
|---|---|
| `row_ref.h` | Host reference for the row/elementwise families, ported line-for-line from `backends/cpu/kernels.cc`. |
| `gpu_test_utils.h` | Storage conversion, deterministic RNG, `DevBuf`, tolerance checks, central finite-difference helper. |
| `rms_norm_test.cc` | RmsNorm forward/backward correctness and finite differences, plus the fused residual variant. |
| `qk_prep_test.cc` | QkPrep forward/backward correctness and finite differences, including a grouped-query case. |
| `pointwise_test.cc` | All five pointwise op codes, forward/backward, plus finite differences. |
| `global_norm_test.cc` | GlobalNorm norm/clip correctness, including a multi-block reduction. |
| `row_bench.cc` | Micro-benchmark for every family on tiny and medium shapes. |
| `pascal_spike.cu`, `pascal_spike_test.cc` | The P0 toolchain spike. |

## Conventions

- **Tiny shapes by default.** The attention-shaped kernels use `B=2, T=8`,
  `head_dim=4`, `num_heads=2`. They cost microseconds, so they are proper
  gates rather than iteration tools.
- **Compare against a host reference.** `row_ref.h` is the device-independent
  specification; a correctness test compares device output to it. This keeps
  the oracle torch-free and the test deterministic.
- **Finite-difference every backward.** Analytic gradients are checked against
  central differences of the reference forward (`h = 1e-2`, relative tolerance
  `2e-3`). Finite differences run in fp32 only; fp16 storage cannot resolve
  the perturbation.
- **Fixed seeds.** `Rng` is a deterministic xorshift, so failures reproduce.
- **Tag GPU tests `gpu`.** `tools/nanochat test` filters them out with
  `--test_tag_filters=-gpu`; `tools/nanochat test --gpu //...` runs only them.
  Do **not** add the `manual` tag to a test that the `--gpu //...` acceptance
  command must discover, because Bazel excludes `manual` targets from wildcard
  expansion.

## Adding a family

1. Add the CUDA translation unit under `backends/cuda/kernels/` and its
   `cuda_library` target (see `backends/cuda/kernels/BUILD.bazel`).
2. Port the reference math into `row_ref.h` (or a sibling reference header).
3. Add a `<family>_gpu_test.cc` here with a correctness check and, when the
   family has a backward, a finite-difference check. Add it to the `all`
   test suite.
4. Optionally add the family to `row_bench.cc`.
