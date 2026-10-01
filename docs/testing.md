# Testing

## Test tiers

| Tier | What | Cost | Scheduling |
|---|---|---|---|
| T0 CPU | reference kernels, workflow, oracle on CPU | ms–s | parallel, each under `t0-cpu` |
| T1 GPU correctness | tiny shapes (`B=2, T=8`), single kernel | < 100 ms | serialized via broker |
| T2 GPU parity | full oracle, loss curve, small training | seconds–minutes | serialized, exclusive |
| T3 GPU benchmark/profile | throughput, Nsight | minutes+ | scheduled, exclusive, native Linux |

Rule: **the inner development loop is T0.** GPU tiers are gates, not iteration
tools. If you are waiting on the GPU to test logic, that logic should have been
tested on CPU. All tiers run under a resource profile; see
[sandbox.md](sandbox.md).

## Definition of done

Every workstream must pass, in order:

1. `bazel build //...` — no warnings.
2. `bazel test //... --test_tag_filters=-gpu` — CPU tests (fast, hermetic).
3. `bazel test //<family>:<family>_gpu_test` — small-shape GPU correctness.
4. Finite-difference gradient check (kernel families with a backward).
5. Oracle fixture match within the backend tolerance.
6. `clang-format` clean.

A workstream is not done until 1–6 pass on the **CPU** backend and at least
small-shape GPU correctness passes.

## Oracle fixtures

`tests/debug_state.bin` is produced offline by nanochat's `gpt.py` (see
[data.md](data.md)). `tests/oracle_test.cc` compares logits, loss, and
gradients, then runs a few optimizer steps and matches losses. Tolerances are
per backend: fp32 CUDA ~1e-5; fp16 Turing looser.

Fixtures are data, so GPU tests never need torch at runtime. Use fixed seeds.

## Finite-difference checks

Kernel families with a backward are checked with a finite-difference gradient
test in `dev/kernels/`. The fused backward is derived and tested as a unit; see
[DESIGN.md §5](../DESIGN.md). `dev/kernels/README.md` describes the standalone
per-kernel test and benchmark convention.

## Bazel tagging

```python
cc_test(
    name = "rms_norm_gpu_test",
    srcs = ["rms_norm_test.cu"],
    tags = ["gpu", "manual"],   # excluded from default runs
    ...
)
```

- `-gpu` filter for the fast default test loop.
- `--local_test_jobs=1` for GPU runs.
- `manual` keeps GPU targets out of `bazel test //...` churn.
