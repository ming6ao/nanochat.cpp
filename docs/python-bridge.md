# Python bridge

The C++ runtime reads only pre-tokenized `NANO` shards and has no tokenizer or
parquet dependency (docs/model.md). The Python bridge supplies the missing
pieces — tokenization, dataset access, and the training-horizon math — and
drives the C++ binaries, so the command line can match the PyTorch nanochat
scripts.

## Why entry points, not bindings

The bridge launches the C++ binary as a subprocess. It does not use pybind11.

- The C++ entry points call `RequireSandboxOrDie`, which reads `NANOCHAT_SANDBOX`
  (set by `tools/sandbox.sh`). A subprocess inherits that environment and the
  cgroup; in-process bindings would bypass the gate.
- pybind11 is a third-party build dependency the design avoids, and it would
  duplicate the CLI.
- The clean seam is the process boundary: Python does data preparation (which
  needs torch and pyarrow), the binary does compute.

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
   head_dim`, `num_heads = model_dim / head_dim`, `n_kv_head = num_heads`, the
   padded vocabulary, the optimal token horizon, the auto batch size, the
   `sqrt(B / B_ref)` learning-rate scale, and the scaled weight decay. The
   parameter counts come from the reference `GPT.num_scaling_params()`, so they
   match exactly.
2. **Just-in-time data.** `nanochat_cpp/data.py` tokenizes only the tokens the
   run needs (`total_batch_size * num_iterations`), reading the parquet dataset
   in small `pyarrow` record batches so a short run touches only the first few.
   It writes a `NANO` shard plus the `<shard>.bytes` sidecar that
   `DataLoader` reads for bits-per-byte, and caches the result under
   `~/.cache/nanochat_cpp/shards` keyed by dataset and tokenizer fingerprints.
   The tokenizer comes from the portable `NCTOKEN1` artifact when present,
   otherwise from the reference pickle; see the next section.
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
`get_bos_token_id`, and `encode`. `token_byte_lengths` derives the
`<shard>.bytes` sidecar from the same merges, so the portable path needs no
`torch`.

The bridge looks for the artifact at
`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.nctoken` (`NCTOKEN1_NAME`). When the file
is present it wins; otherwise `_load_bridge_tokenizer` falls back to
`nanochat.tokenizer.get_tokenizer()` in the reference checkout. `tiktoken` is
imported lazily, so the torch-free self-test can import the module.

The shard cache key (`_fingerprint`) is a 16-character SHA-256 prefix. It hashes
the run shape (`split`, `max_tokens`, `width`) and, for every parquet file and
every tokenizer file, the name, the size, and the time of the last change. The
three tokenizer files are `tokenizer.pkl`, `token_bytes.pt`, and
`tokenizer.nctoken`.
A new or changed artifact therefore invalidates the cached shard. The full key is
part of the shard file name: `<split>_<max_tokens>_<width>_<key>.bin`.

Create the artifact with `tools/convert_tokenizer.py` inside the reference
environment, or with the native trainer `tok_train_main`:

```bash
python3 tools/convert_tokenizer.py --tokenizer <dir> --out tokenizer.nctoken
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

## Known difference

nanochat packs BOS-aligned best-fit batches in Python; the C++ `DataLoader`
reads a contiguous random window over the flat token stream. The two are close
but not bit-identical, so a loss curve from the bridge will not match
`scripts.base_train` step for step. This is tracked as entry D1 in
[parity.md](parity.md). The training-parity harness (`tools/dump_train_fixture.py`,
`//tests:train_parity`) is the exact gate for the model and optimizer; the
bridge is for training.

## Tests

`tools/nanochat test //tools:nanochat_cpp_selftest` runs the torch-free logic
(configuration math, token-count estimate, shard round-trip) with the system
`python3`. A full run needs the reference virtual environment (torch, pyarrow)
and the parquet dataset.
