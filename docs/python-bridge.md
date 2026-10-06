# Python bridge

The `python/nanochat_cpp` package has two entry points to the C++ runtime:

1. **The process bridge** (this document). It launches the C++ binaries as
   subprocesses, so the command line can match the PyTorch nanochat scripts. It
   supplies the `NCTOKEN1` artifact, the dataset paths, and the
   training-horizon math.
2. **The in-process Python API** ([python-api.md](python-api.md)). It loads the
   shared library with `ctypes` and trains, evaluates, and generates inside one
   Python process. The notebook uses this entry point.

Both entry points read the parquet documents and tokenize during the run
([parquet-native.md](parquet-native.md)).

## Why entry points, not bindings

The bridge launches the C++ binary as a subprocess. It does not use pybind11.

- The C++ entry points call `RequireSandboxOrDie`, which reads `NANOCHAT_SANDBOX`
  (set by `tools/sandbox.sh`). A subprocess inherits that environment and the
  cgroup; in-process bindings would bypass the gate.
- pybind11 is a third-party build dependency the design avoids, and it would
  duplicate the CLI.
- The clean seam is the process boundary: Python does the orchestration, the
  binary does compute.

## Commands

The module names mirror the PyTorch scripts, so the command looks the same:

```bash
# PyTorch reference
python -m scripts.base_train --depth=8 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=8 --total-batch-size=4096 --num-iterations=50 \
    --warmup-steps=3 --eval-every=-1 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 --run=dummy --model-tag=demo50

# nanochat.cpp bridge
tools/nanochat_cpp base_train --depth=8 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=8 --total-batch-size=4096 --num-iterations=50 \
    --warmup-steps=3 --eval-every=-1 --save-every=-1 --model-tag=demo50
```

`tools/nanochat_cpp` is a thin wrapper that puts `python/` on `PYTHONPATH` and
picks the reference virtual environment. The equivalent explicit form is:

```bash
PYTHONPATH=python python3 -m nanochat_cpp.base_train --depth=8 ...
```

Useful bridge flags: `--backend cpu|cuda`, `--profile <sandbox profile>`,
`--binary <path>`, `--force-data`, and `--dry-run` (print the mapped
`train_main` command and exit).

## What it does

1. **Configuration and horizon.** `nanochat_cpp/config.py` reproduces
   `scripts/base_train.py`: `model_dim = ceil(depth * aspect_ratio / head_dim) *
   head_dim`, `num_heads = model_dim / head_dim`, and `n_kv_head = num_heads`.
   It also computes the padded vocabulary, the optimal token horizon, the auto
   batch size, the `sqrt(B / B_ref)` learning-rate scale, and the scaled weight
   decay. The parameter counts come from the reference
   `GPT.num_scaling_params()`, so they match exactly.
2. **Data paths.** `nanochat_cpp/data.py` lists the parquet files for a split
   (`train` is all but the last file, `val` is the last file) and resolves the
   `NCTOKEN1` artifact. It passes both to `train_main`/`eval_main`, which read
   the parquet documents and tokenize during the run
   ([parquet-native.md](parquet-native.md)). The bridge writes no shard.
3. **Launch.** `nanochat_cpp/launcher.py` finds (and builds if needed) the
   binary and runs it under the sandbox. A CUDA run goes through
   `tools/nanochat gpu --profile t2-parity`, which holds the GPU broker; a CPU
   run uses the `train` profile. When the bridge is already inside a sandbox
   (for example `tools/nanochat gpu -- python -m nanochat_cpp.base_train`), the
   binary is exec'd directly so the existing cgroup covers the whole tree.

## The `NCTOKEN1` artifact and the cache key

The bridge prefers the portable `NCTOKEN1` artifact over the reference pickle.
`read_nctoken1` parses the little-endian container ([tokenizer.md](tokenizer.md)
section 5). `mergeable_ranks` rebuilds the token-bytes-to-rank map from the
ordered merge pairs, and `build_tiktoken_encoding` builds a `tiktoken.Encoding`
from the pattern and the map. `Nctoken1Tokenizer` supplies `get_vocab_size`,
`get_bos_token_id`, and `encode`. The same object drives the reference
dataloader in the loader-parity fixture, so the portable path needs no `torch`.

The bridge looks for the artifact at
`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.nctoken` (`NCTOKEN1_NAME`). Training and
bit-per-byte evaluation require it; the bridge stops with a message when it is
absent. The bridge imports `tiktoken` only on use, so the torch-free self-test
can import the module.

Create the artifact with the native trainer `tok_train_main`:

```bash
tools/nanochat run t0-cpu -- <tok_train_main> --parquet 'data/*.parquet' \
    --vocab-size 32768 --out tokenizer.nctoken
```

## Flag mapping

The bridge passes the derived values to `train_main`:

| nanochat flag | C++ flag |
|---|---|
| `--depth` | `--layers` |
| `--aspect-ratio`, `--head-dim` | `--hidden`, `--heads`, `--kv-heads` |
| `--max-seq-len` | `--seq` |
| `--window-pattern` | `--window-pattern` |
| `--device-batch-size` | `--batch` |
| `--total-batch-size` | `--grad-accum` (and `--batch`) |
| `--num-iterations` | `--num-iterations` |
| `--embedding-lr` etc. (already scaled) | `--embedding-lr` etc. |
| `--weight-decay` (already scaled) | `--weight-decay`, `--weight-decay-base` |
| `--warmup-steps`, `--warmdown-ratio`, `--final-lr-frac` | same |
| `--eval-every`, `--eval-tokens` | `--eval-every`, `--eval-steps` |
| `--save-every`, `--model-tag` | `--save-every`, `--checkpoint` |

`--clip` is always `0` because `scripts/base_train.py` does no gradient
clipping. `--core-metric-*`, `--sample-every`, `--fp8`, and `--run` are
accepted for compatibility and ignored.

## Parity

The C++ `DataLoader` uses the reference BOS-aligned best-fit packing
([dataloader.h](../include/nanochat/dataloader.h)). This packing closes entry D1
in [parity.md](parity.md). The training-parity harness
(`tools/dump_train_fixture.py`, `//tests:train_parity`) is the exact gate for
the model and optimizer.

## Tests

`tools/nanochat test //tools:nanochat_cpp_selftest` runs the torch-free logic
(configuration math, the token-count estimate, and the `NCTOKEN1` reader) with the system
`python3`. A full run needs the reference virtual environment (torch, pyarrow)
and the parquet dataset.
