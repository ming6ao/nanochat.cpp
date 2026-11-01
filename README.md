# nanochat.cpp

A minimal, hardware-portable, from-scratch training and inference stack for the
nanochat architecture, in the spirit of [llm.c](https://github.com/karpathy/llm.c)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). No PyTorch, no autograd
engine, no graph compiler. One model definition, swappable hardware backends.

Status: implemented and in use. The CPU reference backend and the CUDA backend
run the full training and inference graph.

The oracle and training-parity fixtures pin the model, the backward pass, and
the AdamW and Muon optimizers against the PyTorch reference. The tree also
carries data-parallel training on a host reference. It carries a native byte
pair encoding tokenizer, a parquet reader, a CPU-only CUDA simulator, and a
Python surface. Supervised fine-tuning and reinforcement learning are partly
implemented. At the `demo` scale one job file drives both post-training stages
end to end ([the one-job chain](#the-one-job-chain)).
[docs/parity.md](docs/parity.md) lists the remaining known differences.

## What this is

- The **nanochat architecture** (RMSNorm, RoPE, QK-norm, GQA, relu^2 MLP,
  sliding-window attention, value residual, AdamW, and Muon) with no framework
  dependency.
- **Hardware portability by replacement**: adding a backend means adding a
  directory that implements one header. Public API and model definition are
  stable by default.
- **Portable across NVIDIA generations.** The project builds and tests on the
  GPUs that are available, which today are Pascal (GTX 1080 Ti, sm_61) and
  Turing (T4, sm_75). The deployment target is Hopper and later. A new
  generation is a backend variant, not a rewrite.
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

## Models and devices

A model is a package under `models/`, described by `spec.json`; a device is a
package under `backends/`, declared by `caps.json`. Both sides declare against
the shared vocabulary in `spec/`, and the plan resolves them. A bring-up adds a
directory and edits no frozen header. See
[docs/model-and-device-bringup.md](docs/model-and-device-bringup.md) and the
flagship package in
[models/nanochat-default](models/nanochat-default/README.md).

## Install

The base install is the whole Python surface with no third-party dependency:
`pip install nanochat-cpp` carries both import packages, `nanochat_cpp` imports
only the standard library, and `nanochat_cpp_bridge` imports its heavy package
inside the function that needs it, so the base stays inert until a task adds one
extra. `nanochat-cpp[hub]` reads a stage from a notebook,
`nanochat-cpp[reference,nanochat]` produces a parity fixture, and
`nanochat-cpp[reference,nanochat,hfdata]` builds SFT data from the Hub; the
`nanochat-cpp-convert` console script converts a reference checkpoint. The
per-architecture binary wheel that carries the shared library is not published
yet, so a host that needs the C++ core builds the source tree below, which is
also the workstation path.

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

## The one-job chain

One job file drives post-training end to end: base training, then supervised
fine-tuning, then reinforcement learning, each stage warm starting from the
checkpoint the stage before it wrote. The chain is
[`python/nanochat_cpp/pipeline.py`](python/nanochat_cpp/pipeline.py), the
committed example job is
[`notebooks/pipeline_job.json`](notebooks/pipeline_job.json), and the stages are
the two binaries `//src:sft_main` and `//src:rl_worker`. A run ends in
`<out_root>/<run_id>/deliver/`, where `out_root` is the job's own value or
`$NANOCHAT_RUNS_DIR`: `chatrl.ckpt` beside `pipeline_report.json`, the review
artifact that carries the stage order, the step counts, the loss traces, the
metadata blocks, the digests, and the host block. The same job runs on three
kinds of host: this workstation's CPU under the `train` profile, a Colab T4
that builds from source, and Modal, where two legs run the job, a T4 at fp32
and then an H100 at bf16, each from its own image;
[docs/host-portability.md](docs/host-portability.md) section 9 has one row per
host. `//python:pipeline_test` is the gate of the chain in the default CPU test
loop and `//python:pipeline_gpu_test` is its CUDA twin
([docs/pipeline-plan.md](docs/pipeline-plan.md) section 7). Every run is wrapped
by the entry point, because both stage executables refuse to start outside it:

```bash
# Build the two stage binaries, then run the example job: base@200, sft@200,
# rl@8, the in-process eval on one batch, and the two files of deliver/.
tools/nanochat build //src:sft_main //src:rl_worker
PYTHONPATH=python tools/nanochat train -- \
    python -m nanochat_cpp.pipeline notebooks/pipeline_job.json
```

The Modal run repeats that job as two legs, and the entry point is what compares
them. `pipeline_demo` runs the T4 leg first and the H100 leg second, prints one
document with three keys -- `legs` (each leg's report summary), `comparison`
(the cross-leg step-time ratio beside the recorded MFU and speed ratio), and
`gates` (G1 to G3, each pass or fail with the numbers it read) -- and exits
non-zero when a gate fails. [docs/pipeline-plan.md](docs/pipeline-plan.md) has
the chain and the gates, and
[docs/host-portability.md](docs/host-portability.md) the two legs, the two
images, and the two precisions.

```bash
# Both legs of one job. The T4 leg writes <run_id>-t4 first; the H100 leg
# writes <run_id>-h100 second, so a failed H100 leg leaves the T4 report
# readable. Both reports land on the nanochat-runs Volume.
modal run tools/modal/app.py::pipeline_demo
```

## Examples

`examples/` carries the two smallest programs that use the public API, and
[docs/examples.md](docs/examples.md) is their reference. Each one is a whole
workflow in one file, with a tiny config, a fixed seed, and no model file:

```bash
# Inference end to end: config, model, generate, print the ids.
tools/nanochat run t0-cpu -- bazel-bin/examples/hello_world/hello_world

# The checkpoint container: save, load, print the metadata and records.
tools/nanochat run t0-cpu -- \
    bazel-bin/examples/checkpoint_roundtrip/checkpoint_roundtrip
```

Both are `cc_test`s as well as binaries, so `tools/nanochat test` runs them
with the rest of the CPU tests. [docs/examples.md](docs/examples.md) says what
each one proves, the one line to change for a larger model, and the line budget
that keeps them readable.

## The performance model

The performance model reports the fastest possible step at one shape. It also
reports the operation that binds and the distance from the measured step. The
model uses the device peaks and the operation counters. It holds no fitted
value. It needs no GPU and no broker.

The `roofline` command builds the estimator and runs it under the CPU sandbox:

```bash
# The depth-24 H100 bound from docs/performance-model.md section 7.2.
tools/nanochat roofline --device h100 --precision fp8 \
    --layers 24 --heads 12 --kv-heads 12 --hidden 1536 --seq 2048 \
    --batch 16 --grad-accum 4 --devices 8 --tokens 5.84e9 --link-gbps 600

# The same report as `nanochat.roofline.v1` JSON.
tools/nanochat roofline --json --device h100 --precision fp8 \
    --layers 24 --heads 12 --kv-heads 12 --hidden 1536 --seq 2048 \
    --batch 16 --grad-accum 4 --devices 8 --tokens 5.84e9 --link-gbps 600
```

The output is one row per operation with the compute time, the memory time,
the bound, and the binding limit. The last line gives the step bound and the
run bound.

- `--device` selects a device profile: `h100`, `h200`, `a100`, `t4`, or
  `gtx1080ti`.
- `--precision` selects the modeled precision: `fp32`, `bf16`, `fp8`, or
  `auto`. The `fp8` tier applies the fp8 conversion rule in
  [docs/performance-model.md](docs/performance-model.md) section 5.1.
- `--batch`, `--grad-accum`, and `--devices` set the training plan.
- `--tokens` sets the run length. The value `0` models one optimizer step.
- `--link-gbps` sets the per-rank link bandwidth for the gradient sync.
- The model flags (`--layers`, `--heads`, `--kv-heads`, `--hidden`, `--seq`,
  `--vocab`, `--padded-vocab`, `--window-pattern`, `--rope-base`) set the
  architecture.

Run `tools/nanochat roofline --help` for the full option list. See
[docs/performance-model.md](docs/performance-model.md) for the model, the
counter table, and the device limit table.

## Repository layout

```
include/nanochat/   public headers; the frozen backend seam
src/                backend-agnostic workflow, model, optimizer, harness, CLI,
                    distributed sync, tokenizer, parquet reader
backends/cpu/       reference implementation of every kernel
backends/cuda/      CUDA implementation; cuBLAS for GEMM
backends/cuda/kernels/  kernels, per-family tests, benchmarks, test scaffolding
dev/kernels/        unpromoted prototypes and the toolchain spike
examples/           the two smallest programs that use the public API
models/             one package per model: its spec, tolerances, and fixtures
spec/               the contract vocabulary both axes declare against
tests/              oracle fixtures, parity tests, and the staged numeric trace
bindings/           the C ABI for the in-process Python API
python/             the single Python surface: compute, planning, chat,
                    orchestration
notebooks/          the Kaggle setup module, the T4 notebook, the trial runner
tools/              entry point, GPU job queue, resource sandbox, fixtures
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
- [docs/examples.md](docs/examples.md) — the two smallest examples of the
  public API, and how to run them.
- [docs/python.md](docs/python.md) — the nanochat-compatible
  Python entry points.
- [docs/host-portability.md](docs/host-portability.md) — the second host,
  including the Kaggle procedure.
- [CONTRIBUTING.md](CONTRIBUTING.md) — style, definition of done, and workflow.
- [AGENTS.md](AGENTS.md) — multi-agent coordination on the shared host.
