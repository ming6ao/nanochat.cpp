# nanochat.cpp

A from-scratch training and inference stack for the nanochat language model.
It uses C++ and CUDA. It needs no PyTorch, no autograd engine, and no graph
compiler. One model definition runs on a CPU reference backend or a CUDA
backend. The design follows the small, auditable scope of
[llm.c](https://github.com/karpathy/llm.c) and
[llama.cpp](https://github.com/ggml-org/llama.cpp).

Status: the CPU and CUDA backends run the full training and inference graph.
The post-training chain runs end to end at demo scale. The supervised
fine-tuning (SFT) and reinforcement learning (RL) loops are not finished.
[docs/parity.md](docs/parity.md) lists the known differences from the
reference.

## What this is

- The nanochat architecture: RMSNorm, rotary position embedding (RoPE),
  QK-norm, grouped-query attention, a relu squared multilayer perceptron,
  sliding-window attention, and value residual.
- AdamW and Muon, with a hand-written backward pass and a fixed workspace.
- Two backends behind one kernel header: a CPU reference and a CUDA
  implementation. A backend is a build and link decision.
- Base pretraining from parquet, with a native tokenizer and on-the-fly
  tokenization.
- Data-parallel training on one host, behind a gradient-sync seam.
- Supervised fine-tuning and reinforcement learning, driven by one job file.
- Batched inference with a key-value (KV) cache, bits-per-byte and chat
  evaluation, and a CPU-only CUDA simulator.
- A Python surface for compute, planning, and orchestration.

## What this is not

- A general tensor library, a model framework, or a framework dependency.
- An autograd engine, a graph intermediate representation, or a compiler.
- A runtime precision switch. Precision is a build and link decision
  ([docs/precision.md](docs/precision.md)).
- A runtime kernel registry, a virtual dispatch table, or a memory planner.
- A user of a third-party runtime beyond CUDA, cuBLAS, the ZSTD
  decompressor in `src/parquet/`, and the CUTLASS headers, which are a
  build-time dependency and link nothing ([docs/build.md](docs/build.md#third-party-code)).
- A multi-node trainer. Data parallel today works inside one host, and the
  NCCL transport is not complete.
- A finished post-training product.
- An HTTP inference server or a serving framework.

## Requirements

- Linux, or WSL2 on Windows.
- GCC 12, or a host compiler that CUDA 12.0 supports. Bazel 9.2.0 arrives
  through Bazelisk.
- CUDA 12.0 and an NVIDIA card for the CUDA backend.
- Python 3.10 or newer for the Python surface.

The development host card is below `sm_75`, so the tree does not support it and
no device target names it. The tree targets the T4 (`sm_75`) and Hopper
(`sm_90`).

The Python package lives in `python/`. Run it from the tree with
`PYTHONPATH=python`. A binary wheel is not published yet.

## Build

Use `tools/nanochat` for every build and every run. The binaries refuse to
start outside it. `tools/nanochat help` lists every command.

```bash
# CPU reference backend, the default. This builds all targets.
tools/nanochat build

# CUDA backend for the default architecture, sm_75.
tools/nanochat build --config=cuda

# The Hopper hand-run target.
tools/nanochat build --config=hopper

# One target.
tools/nanochat build //src:nanochat_cli
```

The static library and the stage binary land in `bazel-bin/`. The entry
point serializes builds, caps the Bazel heap, and applies the host budget.
[docs/build.md](docs/build.md#layers-and-the-one-seam) gives the layers and the
dependency rules.

## Test

```bash
# Build, then run every CPU test under the resource sandbox.
tools/nanochat check

# All CPU tests. GPU tests carry the `gpu` tag and stay out.
tools/nanochat test

# One GPU correctness test. The queue serializes access to the card.
tools/nanochat test --gpu //backends/cuda/kernels:rms_norm_gpu_test

# Style gate.
tools/nanochat lint

# H100 and H200 behaviour without a Hopper card. CPU only.
tools/nanochat simulate --device h100
```

`T0 CPU` runs the reference kernels, the workflow, and the oracle checks. It
is fast and parallel, so use it for the inner loop. `T1 GPU` checks one tiny
shape per kernel. `T2 GPU` runs the full oracle and a small training run. Both
use the GPU queue. `S0` is the CPU-only simulator for a device that is not on
this host. [docs/testing.md](docs/testing.md) gives the tiers and the gate list.

## Train a base model

The training path reads parquet documents and tokenizes during the run. Train
the tokenizer on the same corpus first.

```bash
tools/nanochat build //src/tokenizer:tok_train_main //src:nanochat_cli

# Write an NCTOKEN1 tokenizer.
tools/nanochat run t0-cpu -- \
    bazel-bin/src/tokenizer/tok_train_main \
    --parquet 'data/*.parquet' --vocab-size 32768 \
    --out tokenizer.nctoken

# Train the base model.
tools/nanochat train -- \
    bazel-bin/src/nanochat_cli train \
    --train-parquet 'data/*.parquet' \
    --val-parquet 'data/val-*.parquet' \
    --tokenizer tokenizer.nctoken \
    --layers 12 --heads 6 --kv-heads 6 --hidden 768 --seq 2048 \
    --vocab 32768 --padded-vocab 32768 --window-pattern SSSL \
    --batch 8 --grad-accum 4 --num-iterations 1000 \
    --checkpoint base.ckpt
```

The `--vocab` value must match the tokenizer artifact. Keep the model flags.
The inference path uses the same flags to build the model before it loads the
weights.

For two cards on one host, start one process per rank:

```bash
tools/nanochat dist --nproc-per-node 2 -- \
    bazel-bin/src/nanochat_cli train --train-parquet 'data/*.parquet' ...
```

`dist` sets the rank variables and submits one queue job for the group. See
[docs/distributed.md](docs/distributed.md).

## Post-train: SFT and RL

SFT trains on a packed conversation shard. RL samples rollouts and applies a
weighted policy-gradient backward pass. The Python facades own the chat
template and the loss mask.

The one-job chain is the easiest entry. A job file names the profile, the step
counts, and the output root:

```bash
cat > /tmp/job.json <<'JSON'
{
  "run_id": "demo",
  "profile": "demo",
  "base_steps": 100,
  "sft_steps": 100,
  "rl_steps": 5
}
JSON

tools/nanochat build //src:nanochat_cli
PYTHONPATH=python tools/nanochat train -- \
    python3 -m nanochat_cpp.pipeline /tmp/job.json
```

The `standard` profile uses a larger model. The `demo` profile keeps the CPU
loop short. Each stage warm starts from the checkpoint of the stage before it.
The run ends in `<out_root>/<run_id>/deliver/`. Treat the demo as a wiring
check, not a quality result.

`out_root` comes from the job, then `NANOCHAT_RUNS_DIR`, then the system
temporary directory. The report records the stage order, the step counts, the
loss traces, the metadata, and the file digests. `//src:nanochat_cli` is the
one stage binary, and a subcommand names the stage.

[docs/post-training.md](docs/post-training.md) gives the design, the parity
entries, and the stages that are not implemented.

## Inference and deployment

Inference loads a checkpoint and samples tokens. Pass the model flags that
match the checkpoint. A mismatch fails at load, before a kernel reads a wrong
shape.

```bash
tools/nanochat build //src:nanochat_cli

tools/nanochat eval -- \
    bazel-bin/src/nanochat_cli generate \
    --model base.ckpt --tokenizer tokenizer.nctoken \
    --prompt "Name a prime number." \
    --max-tokens 64 --temperature 0.8 --top-k 50 \
    --layers 12 --heads 6 --kv-heads 6 --hidden 768 --seq 2048 \
    --vocab 32768 --padded-vocab 32768 --window-pattern SSSL
```

The same path in Python:

```python
import nanochat_cpp as nc

config = nc.Config(num_layers=12, num_heads=6, num_kv_heads=6,
                   hidden_dim=768, seq_len=2048, vocab_size=32768,
                   window_pattern="SSSL")
model = nc.Model(config, device="cuda")  # omit `device` on a CPU host
model.load("base.ckpt")
tokenizer = nc.tokenizer("tokenizer.nctoken")

prompt = tokenizer.encode(
    "<|bos|><|user_start|>Name a prime number.<|user_end|>"
    "<|assistant_start|>")
rows = model.generate(prompt, max_tokens=64, temperature=0.8, top_k=50)
print(tokenizer.decode(rows[0].tokens))
```

The two smallest complete programs are in `examples/`:

```bash
tools/nanochat run t0-cpu -- bazel-bin/examples/hello_world/hello_world
tools/nanochat run t0-cpu -- \
    bazel-bin/examples/checkpoint_roundtrip/checkpoint_roundtrip
```

### Deployment hosts

The tree has no HTTP server. It exposes a batch binary, a Python API, and a
single-client multi-device runtime. Use one of those as the serving surface,
and choose a host:

- **Modal**: `modal run tools/modal/app.py::pipeline_demo` runs one job on a
  T4 at fp32 and then on an H100 at bf16. `modal deploy tools/modal/app.py`
  deploys the app. See [docs/host-portability.md](docs/host-portability.md)
  section 9.
- **Kaggle**: two T4 cards, `--config=cuda --@rules_cuda//cuda:archs=sm_75`,
  and the sandbox backend set to `none`. See [docs/host-portability.md](docs/host-portability.md) section 8.
- **H100 node**: the memory plan, the batch plan, and the launch are in
  [docs/host-portability.md](docs/host-portability.md) Part 1.

The design in [docs/distributed.md](docs/distributed.md) covers data-parallel
training and replicated inference across local devices.

## Evaluate

`nanochat_cli eval` computes bits per byte on the validation split. Pass the
model flags that match the checkpoint:

```bash
tools/nanochat eval -- \
    bazel-bin/src/nanochat_cli eval \
    --model base.ckpt \
    --val-parquet 'data/val-*.parquet' \
    --tokenizer tokenizer.nctoken \
    --layers 12 --heads 6 --kv-heads 6 --hidden 768 --seq 2048 \
    --vocab 32768 --padded-vocab 32768 --window-pattern SSSL \
    --steps 20
```

Chat evaluation runs in Python and covers ARC, MMLU, GSM8K, and ChatCORE.
Pass the model shape with the `--layers`, `--aspect-ratio`, `--head-dim`,
`--max-seq-len`, `--vocab`, and `--window-pattern` flags:

```bash
PYTHONPATH=python tools/nanochat eval -- \
    python3 -m nanochat_cpp.chat --model deliver/chatrl.ckpt --tasks arc,gsm8k \
    --layers 12 --aspect-ratio 64 --head-dim 128 \
    --max-seq-len 2048 --vocab 32768 --window-pattern SSSL
```

[docs/eval.md](docs/eval.md) gives the metrics and the compute seam.

## Models and devices

A model is a package under `models/`, described by `spec.json`. A device is a
package under `backends/`, declared by `caps.json`. Both declare against the
shared vocabulary in `spec/`, and the planner resolves them. The flagship
model is `models/nanochat-default/`. `backends/cpu/caps.json` declares the
reference device, and `backends/cuda/caps.json` declares the served devices.

A bring-up adds a directory. It does not edit a frozen header. See
[docs/model-and-device-bringup.md](docs/model-and-device-bringup.md) and
[models/nanochat-default](models/nanochat-default/README.md).

## Repository layout

```text
include/nanochat/   public headers and the frozen kernel seam
src/                model, workflow, optimizer, training harness, tokenizer,
                    parquet reader, and the stage binaries
backends/cpu/       reference implementation of every kernel
backends/cuda/      CUDA kernels, cuBLAS GEMM, per-family tests, benchmarks
models/             one package per model
spec/               the contract vocabulary
tests/              oracle fixtures, parity tests, and the numeric trace
examples/           the two smallest programs that use the public API
bindings/           the C ABI for the Python surface
python/             compute, planning, chat evaluation, and orchestration
notebooks/          Kaggle setup, the pipeline job, and host helpers
tools/              entry point, GPU queue, sandbox, fixtures, simulator
docs/               reference and how-to
```

## Documentation

- [docs/README.md](docs/README.md): the full reference index.
- [docs/build.md](docs/build.md): what the build produces, the layers, the
  dependency invariants, the build configurations, the resource bounds, the
  third-party code, the backends and their precision, the Python distribution,
  the parity seam, and the log of measured runs.
- [docs/testing.md](docs/testing.md): test tiers, the oracle, and the gates.
- [docs/sandbox.md](docs/sandbox.md): the resource budgets, the agent session
  scope, and the GPU queue.
- [docs/post-training.md](docs/post-training.md): SFT, RL, and the chain.
- [docs/python.md](docs/python.md): the Python API and the toolchain.
- [docs/host-portability.md](docs/host-portability.md): the second host and
  the Modal legs.
- [docs/parity.md](docs/parity.md): the known differences from the reference.
- [CONTRIBUTING.md](CONTRIBUTING.md): style and the Definition of Done.
- [AGENTS.md](AGENTS.md): multi-agent coordination on the shared host.
