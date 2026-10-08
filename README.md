# nanochat.cpp

A minimal, hardware-portable, from-scratch training and inference stack for the
nanochat architecture, in the spirit of [llm.c](https://github.com/karpathy/llm.c)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). No PyTorch, no autograd
engine, no graph compiler. One model definition, swappable hardware backends.

Status: implemented and in use. The CPU reference backend and the CUDA backend
run the full training and inference graph. Pascal (GTX 1080 Ti, sm_61) is the
measured device. The Turing (T4, sm_75) build configs exist, but the project has
not recorded the Turing correctness gates.

The oracle and training-parity fixtures pin the model, the backward pass, and
the AdamW and Muon optimizers against the PyTorch reference. The tree also
carries data-parallel training on a host reference and an opt-in ANVIL
optimizer. It carries a native byte pair encoding tokenizer, a parquet reader, a
CPU-only CUDA simulator, and a Python surface. Supervised fine-tuning and
reinforcement learning are partly implemented, and the ANVIL oracle is still
open.
[docs/parity.md](docs/parity.md) lists the remaining known differences.

## What this is

- The **nanochat architecture** (RMSNorm, RoPE, QK-norm, GQA, relu^2 MLP,
  sliding-window attention, value residual, AdamW, Muon, and the opt-in ANVIL
  matrix optimizer) with no framework dependency.
- **Hardware portability by replacement**: adding a backend means adding a
  directory that implements one header. Public API and model definition are
  stable by default.
- **First-class on Pascal (GTX 1080 Ti, sm_61)**. **Turing (T4, sm_75)** build
  configs exist; its correctness gates are not recorded. Later NVIDIA
  generations are a backend variant, not a rewrite.
- **Data-parallel training** behind a host-reference gradient-sync seam. The
  NCCL transport remains future work.
- **A native data path**: a byte pair encoding tokenizer and a parquet reader,
  with on-the-fly tokenization during training.
- **A CPU-only CUDA simulator** that checks the backend seam and the staged
  numeric trace without a GPU.
- **A Python surface** for in-process compute, planning, chat evaluation, and
  the `tools/nanochat` toolchain.
- Every operator independently testable against a fixed oracle.

## What this is not

- A general-purpose tensor library or model framework.
- An autograd engine, graph IR, or compiler.
- A home for dynamic shapes or runtime kernel selection.
- A finished post-training product. The supervised fine-tuning and
  reinforcement learning loops are incomplete.
- A distributed-training product beyond the host reference. The NCCL transport
  is not implemented.

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
src/                backend-agnostic workflow, model, optimizer, harness, CLI,
                    distributed sync, tokenizer, parquet reader
backends/cpu/       reference implementation of every kernel
backends/cuda/      CUDA implementation; cuBLAS for GEMM
backends/cuda/kernels/  kernels, per-family tests, benchmarks, test scaffolding
dev/kernels/        unpromoted prototypes and the toolchain spike
tests/              oracle fixtures, parity tests, and the staged numeric trace
bindings/           the C ABI for the in-process Python API
python/             the single Python surface: compute, planning, chat,
                    orchestration
notebooks/          the Kaggle setup module, the T4 notebook, the trial runner
tools/              entry point, GPU broker, resource sandbox, fixtures
tools/cuda_sim/     the CPU-only CUDA simulator
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
