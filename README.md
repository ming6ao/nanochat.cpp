# nanochat.cpp

A minimal, hardware-portable, from-scratch training and inference stack for the
nanochat architecture, in the spirit of [llm.c](https://github.com/karpathy/llm.c)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). No PyTorch, no autograd
engine, no graph compiler. One model definition, swappable hardware backends.

Status: design. Nothing built yet.

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
bazel build //...

# CPU tests: fast and hermetic.
bazel test //... --test_tag_filters=-gpu

# One GPU correctness test, through the sandbox gateway.
tools/gpu.sh --profile=t1-gpu -- ./bazel-bin/.../rms_norm_dev
```

See [docs/build.md](docs/build.md) for toolchains and flags, and
[docs/sandbox.md](docs/sandbox.md) for the host resource limits every job runs
under.

## Repository layout

```
include/nanochat/   public headers; the frozen backend seam
src/                backend-agnostic workflow, model, optimizer, CLI
backends/cpu/       reference implementation of every kernel
backends/cuda/      CUDA implementation; cuBLAS for GEMM
dev/kernels/        standalone per-kernel test + benchmark
tests/              oracle fixture tests
tools/              GPU broker and resource sandbox
docs/               reference and how-to
```

## Documentation

- [DESIGN.md](DESIGN.md) — design rationale, invariants, and evolution rules.
- [docs/](docs/README.md) — reference and how-to.
- [CONTRIBUTING.md](CONTRIBUTING.md) — style, definition of done, and workflow.
- [AGENTS.md](AGENTS.md) — multi-agent coordination on the shared host.
