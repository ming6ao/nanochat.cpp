# nanochat.cpp

A minimal, hardware-portable, from-scratch training and inference stack for the
nanochat architecture, in the spirit of [llm.c](https://github.com/karpathy/llm.c)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). No PyTorch, no autograd
engine, no graph compiler. One model definition, swappable hardware backends.

Status: implemented and in use. The CPU reference backend and the CUDA backend
(Pascal sm_61 and Turing sm_75) run the full training and inference graph. The
oracle and training-parity fixtures pin the model, the backward pass, and the
optimizer against the PyTorch reference. [docs/parity.md](docs/parity.md) lists
the remaining known differences.

## What this is

- The **nanochat architecture** (RMSNorm, RoPE, QK-norm, GQA, relu^2 MLP,
  sliding-window attention, value residual, Muon + AdamW) with no framework
  dependency.
- **Hardware portability by replacement**: adding a backend means adding a
  directory that implements one header. Public API and model definition are
  stable by default.
- **First-class on Pascal (GTX 1080 Ti, sm_61)** and **Turing (T4, sm_75)**;
  later NVIDIA generations are a backend variant, not a rewrite.
- Every operator independently testable against a fixed oracle.

## What this is not

- A general-purpose tensor library or model framework.
- An autograd engine, graph IR, or compiler.
- A home for dynamic shapes or runtime kernel selection.

## Quickstart

```bash
# Build everything.
tools/nanochat build

# Build, then run every CPU test. Each test runs under the resource sandbox.
tools/nanochat check

# One GPU correctness test; the broker serializes access to the card.
tools/nanochat test --gpu //backends/cuda/kernels:rms_norm_gpu_test

# Derive a training plan in Python and read the two model-derived numbers.
PYTHONPATH=python python3 -c "import nanochat_cpp as nc; \
    print(nc.plan.compute_plan(depth=4, seq_len=512, vocab_size=32768, \
    device_batch_size=2, window_pattern='L', num_iterations=10))"
```

`tools/nanochat help` lists all commands. Training, evaluation, generation, and
scoring each have a binary: `//src:train_main`, `//src:eval_main`,
`//src:generate_main`, and `//src:score_main`. The Python packages
`nanochat_cpp.api`, `nanochat_cpp.plan`, `nanochat_cpp.chat`, and
`nanochat_cpp.toolchain` drive them in one process
([docs/python.md](docs/python.md)).

- [docs/build.md](docs/build.md) — toolchains and flags.
- [docs/testing.md](docs/testing.md) — the test tiers and Definition of Done.
- [docs/sandbox.md](docs/sandbox.md) — the resource budgets every job runs
  under.

## Repository layout

```
include/nanochat/   public headers; the frozen backend seam
src/                backend-agnostic workflow, model, optimizer, harness, CLI
backends/cpu/       reference implementation of every kernel
backends/cuda/      CUDA implementation; cuBLAS for GEMM
backends/cuda/kernels/  kernels, per-family tests, benchmarks, test scaffolding
dev/kernels/        unpromoted prototypes and the toolchain spike
tests/              oracle fixtures and parity tests
bindings/           the C ABI for the in-process Python API
python/             the single Python surface: compute, planning, orchestration
notebooks/          the Kaggle setup module and the T4 notebook
tools/              entry point, GPU broker, resource sandbox, fixtures
tools/kaggle/       Kaggle Notebook bootstrap
docs/               reference and how-to
```

## Documentation

- [DESIGN.md](DESIGN.md) — design rationale, invariants, and evolution rules.
- [docs/](docs/README.md) — reference and how-to. Start with
  [build.md](docs/build.md), [testing.md](docs/testing.md), and
  [sandbox.md](docs/sandbox.md).
- [docs/parity.md](docs/parity.md) — known differences from the PyTorch
  reference.
- [docs/python.md](docs/python.md) — the nanochat-compatible
  Python entry points.
- [docs/host-portability.md](docs/host-portability.md) — the second host,
  including the Kaggle procedure.
- [CONTRIBUTING.md](CONTRIBUTING.md) — style, definition of done, and workflow.
- [AGENTS.md](AGENTS.md) — multi-agent coordination on the shared host.
