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

1. `tools/nanochat build` — no warnings.
2. `tools/nanochat test` — CPU tests.
3. `tools/nanochat test --gpu <target>` — small-shape GPU correctness.
4. Finite-difference gradient check (kernel families with a backward).
5. Oracle fixture match within the backend tolerance.
6. `clang-format` clean.

A workstream is not done until 1–6 pass on the **CPU** backend and at least
small-shape GPU correctness passes.

## How tests are sandboxed

`tools/nanochat test` runs Bazel with
`--run_under='tools/sandbox.sh --profile=… --'`, so every test action gets its
own cgroup, and forces `--spawn_strategy=local` so the wrapper can reach the
systemd user manager. GPU suites hold the broker lock for the whole invocation
with `tools/gpu.sh --lock-only`. Sandboxing per test action means the result
does not depend on whether a Bazel server is already running. The trade-off is
that Bazel's own filesystem sandbox is disabled for these runs; the resource
sandbox is the priority. See [sandbox.md](sandbox.md).

## Oracle fixtures

`tests/debug_state.bin` is produced offline by nanochat's `gpt.py` (see
[data.md](data.md)). `tests/oracle_test.cc` compares logits, loss, and
gradients, then runs a few optimizer steps and matches losses. Tolerances are
per backend: fp32 CUDA ~1e-5; fp16 Turing looser.

The multi-step trajectory is a strict gate (loss `1e-5`, parameter `1e-4`).
`tools/dump_oracle.py` records the AdamW `eps` it used as
`config/opt/adam_eps` (currently `1e-4`) and the C++ test matches it: several
fixture parameters have a true gradient of exactly zero at this initialization,
so at nanochat's default `eps=1e-10` their fp32 roundoff would be amplified
into unreproducible updates. Regenerate the fixture with
`python tools/dump_oracle.py --adam-eps 1e-4`.

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
