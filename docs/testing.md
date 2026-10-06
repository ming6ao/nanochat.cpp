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
6. `tools/nanochat lint` — the Google C++ Style gate.

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
[model.md](model.md)). `tests/oracle_test.cc` compares logits, loss, and
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

## Training-parity harness

`tools/dump_train_fixture.py` records a multi-step *training trajectory* from
nanochat's PyTorch `gpt.py`: the initial parameters, the exact token/target
batch at every step, and, after every step, the loss, the global gradient norm,
and the L2 norm of every parameter (and optionally the final parameter tensors).
`tests/train_parity_test.cc` drives `nanochat.cpp`'s `TrainStep` from the same
initial parameters over the same batches with the same optimizer and
learning-rate schedule, and compares the loss, gradient, and parameter
trajectories. This is the scaled-up version of the oracle trajectory gate: it
is what catches a bug that only appears after the first update, when the
zero-initialized output projections activate the attention and MLP backward.

The committed fixture `tests/data/train_parity.bin` is the compact CPU case
(`--nonzero-projections` so the whole graph is active at step 0) and runs as
`//tests:train_parity_test` (CPU) and `//tests:train_parity_cuda_test` (GPU).
Because the fixture shares the exact batches, this gate covers the model,
backward pass, optimizer, and schedules but **not** the data pipeline; the
loader difference is tracked as entry D1 in [parity.md](parity.md).
The same source builds as the standalone `//tests:train_parity` binary for an
arbitrary production fixture:

```bash
python3 tools/dump_train_fixture.py --out /tmp/train_parity_d8_s512_50.bin \
    --layers 8 --heads 4 --kv-heads 4 --embd 512 --vocab 32768 --pad-to 32768 \
    --seq 512 --batch 8 --steps 50 --window L --adam-eps 1e-4
tools/nanochat gpu --profile t2-parity -- \
    ./bazel-bin/tests/train_parity /tmp/train_parity_d8_s512_50.bin
```

A 50-step run at `d8_s512` matches the reference: loss `6e-6`, parameter L2
`1.7e-3`, gradient L2 `4.3e-5`.

`TRAIN_PARITY_STOP=1` stops at the first divergent step, which is useful for
bisecting a shape- or schedule-dependent bug. Like the oracle, the fixture is
generated on the CPU with torch and the runtime tests never import torch.

## Tokenizer tests

The tokenizer, the trainer, and the parquet reader are host
utilities. Every test is tier T0. The tests need no GPU and no broker. The exact
commands are:

```bash
# The tokenizer package: round trip, splitter, reference parity, trainer,
# reader. The suite label runs all five targets.
tools/nanochat test //src/tokenizer:all

# The `LoadTokenizer` contract and the `DataLoader` shift.
tools/nanochat test //src:data_test

# The bridge `NCTOKEN1` reader and merge reconstruction, torch-free.
tools/nanochat test //tools:nanochat_cpp_selftest

# The whole CPU suite and the style gate.
tools/nanochat test
tools/nanochat lint
```

`//src/tokenizer:tokenizer_parity_test` and `//src/tokenizer:bpe_trainer_test`
read the committed fixture `tests/data/tokenizer_fixture.bin`, so they never
import Python. The native command lines build and run through the sandbox:

```bash
tools/nanochat build //src/tokenizer:tok_train_main
tools/nanochat run t0-cpu -- <tok_train_main> --text corpus.txt \
    --vocab-size 512 --out /tmp/tokenizer.nctoken
tools/nanochat run t0-cpu -- <tok_train_main> --parquet 'data/*.parquet' \
    --vocab-size 512 --max-chars 2000000 --out /tmp/tokenizer.nctoken
```

The bridge self-test needs the system `python3` only. A full bridge run needs
the reference virtual environment (torch, pyarrow) and the parquet dataset; see
[python-bridge.md](python-bridge.md).

## Finite-difference checks

Kernel families with a backward are checked with a finite-difference gradient
test next to the kernel in `backends/cuda/kernels/`. The fused backward is
derived and tested as a unit; see [DESIGN.md §5](../DESIGN.md).
`backends/cuda/kernels/README.md` describes the per-kernel test and benchmark
convention.

## Bazel tagging

```python
cc_test(
    name = "rms_norm_gpu_test",
    srcs = ["rms_norm_test.cc"],
    tags = ["gpu"],   # excluded from the default CPU loop
    ...
)
```

- `-gpu` filter for the fast default test loop.
- `--local_test_jobs=1` for GPU runs.
- `tools/nanochat test --gpu //...` discovers every `gpu`-tagged test. Do not
  add the `manual` tag to a test that this command must discover, because Bazel
  excludes `manual` targets from wildcard expansion.
