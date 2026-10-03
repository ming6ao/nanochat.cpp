#!/usr/bin/env python3
"""Generate the *generation-parity* fixture, ``tests/data/generate_parity.bin``.

This is the inference analogue of ``tools/dump_oracle.py`` (one forward/backward)
and ``tools/dump_train_fixture.py`` (a training trajectory). It records a fixed
greedy-decoding trace from nanochat's PyTorch ``GPT``:

* the architecture and generation hyperparameters,
* every model parameter, named as in ``gpt.py``,
* the reference prompt,
* the generated token ids (the effective prompt followed by the sampled ids),
* the per-position mask (``0`` for a prompt id, ``1`` for a sampled id).

``tests/generate_parity_test.cc`` rebuilds the same model from the recorded
parameters and drives ``nanochat.cpp``'s ``GenerateBatch`` in greedy mode. The
expected ids and mask are compared exactly. Because the fixture is data, the
test never imports torch; the same source runs against the CPU reference backend
by default and the CUDA backend under the ``gpu`` tag.

Two details make the discrete comparison robust. First, the recorded parameters
are *trained* for ``--train-steps`` steps on a learnable bigram stream before the
trace is taken. Random initialization leaves the classifier almost uniform, so
two implementations that agree to within fp32 roundoff can still disagree on an
argmax; a short, deterministic PyTorch training run sharpens the distribution
and the top-1 margin by orders of magnitude. Second, the trace is produced by the
reference ``GPT.generate`` with ``temperature=0`` (greedy), which is exactly the
single-row loop ``GenerateBatch`` is built to reproduce.

Output is the same little-endian ``NANOORC1`` container as the oracle and
training fixtures (see ``tools/dump_oracle.py``), so the C++ side reuses
``tests/oracle_fixture.h``.

The fixture is generated on the CPU and deterministically. Run it from
anywhere::

    python3 tools/dump_generate_fixture.py --out tests/data/generate_parity.bin
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# Force the CPU reference before importing torch. The fixture is the oracle and
# the host has one shared GPU that must stay free for the candidate run.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# The reference optimizer is compiled with @torch.compile in gpt.py; run it
# eagerly so the fixture is deterministic and needs no C++ compiler.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _find_reference_repo() -> Path:
    """Locate the PyTorch nanochat checkout that holds the `nanochat` package."""
    candidates = []
    if os.environ.get("NANOCHAT_REPO"):
        candidates.append(Path(os.environ["NANOCHAT_REPO"]))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "nanochat")  # sibling checkout
    candidates.append(Path.home() / "repos" / "nanochat")
    for candidate in candidates:
        if (candidate / "nanochat" / "gpt.py").is_file():
            return candidate
    raise SystemExit(
        "could not locate the PyTorch nanochat checkout; set NANOCHAT_REPO to "
        "the directory that contains the `nanochat` package")


REFERENCE_REPO = _find_reference_repo()
if str(REFERENCE_REPO) not in sys.path:
    sys.path.insert(0, str(REFERENCE_REPO))

from nanochat.gpt import GPT, GPTConfig  # noqa: E402


# --- fixture container (kept in sync with tools/dump_oracle.py) -------------

_DTYPE_CODES = {
    np.dtype(np.float32): 0,
    np.dtype(np.int32): 1,
    np.dtype(np.float64): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.uint8): 4,
}


class FixtureWriter:
    """Append-only writer for the little-endian NANOORC1 container."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: list[bytes] = []

    def add(self, name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        dtype = array.dtype
        if dtype not in _DTYPE_CODES:
            raise TypeError(f"unsupported dtype {dtype} for record {name!r}")
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"record name too long: {name!r}")
        if array.ndim > 0xFF:
            raise ValueError(f"record rank too high: {name!r}")
        parts = [
            struct.pack("<H", len(name_bytes)),
            name_bytes,
            struct.pack("<BB", _DTYPE_CODES[dtype], array.ndim),
        ]
        for extent in array.shape:
            parts.append(struct.pack("<Q", int(extent)))
        parts.append(array.tobytes(order="C"))
        self._records.append(b"".join(parts))

    def write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as handle:
            handle.write(b"NANOORC1")
            handle.write(struct.pack("<II", 1, len(self._records)))
            for record in self._records:
                handle.write(record)


def _as_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().numpy()


def _scalar(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float32)


# A compact but structurally complete default: grouped-query attention, value
# embeddings on odd layers, a sliding window pattern. Matches the oracle and
# training fixtures (2 layers, 8 query heads, 2 key/value heads, hidden 32,
# vocabulary 64) apart from `sequence_len`, which is 16 here so the trace can
# extend past the 8-token training fixtures without leaving the context.
DEFAULT_SEED = 1234
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_EMBD = 32
DEFAULT_VOCAB = 64
# `sequence_len` is 16 and the committed trace is `prompt_len + max_tokens = 14`
# tokens, so the run stays inside one training context. That matters: the
# reference sets the "L" (full-context) window to `sequence_len` rather than
# unlimited, so once generation runs past `sequence_len` its L layers attend to
# only the last `sequence_len` positions, while nanochat.cpp's L layers are
# unlimited. Keeping the trace within `sequence_len` makes the two attend the
# same keys. `generate` below enforces this.
DEFAULT_SEQ = 16
DEFAULT_WINDOW = "SL"
DEFAULT_PAD_TO = 64
DEFAULT_PROMPT_LEN = 6
DEFAULT_MAX_TOKENS = 8
DEFAULT_TRAIN_STEPS = 300
DEFAULT_BATCH = 2


def build_config(args: argparse.Namespace) -> GPTConfig:
    return GPTConfig(
        sequence_len=args.seq,
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
        window_pattern=args.window,
    )


def make_stream(count: int, vocab: int) -> torch.Tensor:
    """A deterministic, learnable bigram stream (``next = 7*token + 3 mod V``).

    The model can drive the loss near zero on it, which is what sharpens the
    classifier and makes the greedy trace robust to roundoff.
    """
    stream = torch.empty(count, dtype=torch.long)
    value = 0
    for i in range(count):
        stream[i] = value
        value = (value * 7 + 3) % vocab
    return stream


def train_reference(model: GPT, args: argparse.Namespace) -> float:
    """Runs a short deterministic training loop to sharpen the classifier."""
    if args.train_steps <= 0:
        return float("nan")
    rows = args.batch * args.seq
    stream = make_stream(rows * (args.train_steps + 1) + 1, args.vocab)
    optimizer = model.setup_optimizer()
    model.train()
    loss_value = float("nan")
    for it in range(args.train_steps):
        window = stream[it * rows:it * rows + rows + 1]
        tokens = window[:-1].reshape(args.batch, args.seq)
        targets = window[1:].reshape(args.batch, args.seq)
        loss = model(tokens, targets=targets)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        loss_value = float(loss)
    return loss_value


def generate(args: argparse.Namespace) -> Path:
    if args.prompt_len + args.max_tokens > args.seq:
        raise SystemExit(
            "prompt_len + max_tokens must not exceed seq: the reference caps "
            "its L layers at sequence_len, so a longer trace would compare "
            "two different attention windows")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    model = GPT(config, pad_vocab_size_to=args.pad_to)
    model.init_weights()
    model.to(torch.float32)

    train_reference(model, args)
    model.eval()

    padded_vocab = model.lm_head.weight.shape[0]

    prompt_generator = torch.Generator().manual_seed(args.seed + 1)
    prompt = torch.randint(0, args.vocab, (args.prompt_len,),
                           generator=prompt_generator, dtype=torch.long)
    prompt_ids = prompt.tolist()

    # The reference greedy decode. `generate` is the batch-1 streaming loop in
    # gpt.py: temperature <= 0 selects an argmax over the soft-capped logits at
    # the last position, exactly the `GenerateBatch` greedy contract.
    generated = list(model.generate(prompt_ids, max_tokens=args.max_tokens,
                                    temperature=0.0, top_k=None,
                                    seed=args.seed))
    effective = prompt_ids + generated
    mask = [0] * len(prompt_ids) + [1] * len(generated)

    writer = FixtureWriter(Path(args.out))

    # --- configuration -----------------------------------------------------
    writer.add("config/version", np.asarray([1], dtype=np.int32))
    writer.add("config/seed", np.asarray([args.seed], dtype=np.int64))
    writer.add("config/layers", np.asarray([args.layers], dtype=np.int32))
    writer.add("config/heads", np.asarray([args.heads], dtype=np.int32))
    writer.add("config/kv_heads", np.asarray([args.kv_heads], dtype=np.int32))
    writer.add("config/embd", np.asarray([args.embd], dtype=np.int32))
    writer.add("config/vocab", np.asarray([args.vocab], dtype=np.int32))
    writer.add("config/padded_vocab", np.asarray([padded_vocab], dtype=np.int32))
    writer.add("config/seq", np.asarray([args.seq], dtype=np.int32))
    writer.add("config/train_steps",
               np.asarray([args.train_steps], dtype=np.int32))
    writer.add(
        "config/window_pattern",
        np.frombuffer(args.window.encode("ascii"), dtype=np.uint8).copy(),
    )
    writer.add("config/prompt_len", np.asarray([args.prompt_len],
                                               dtype=np.int32))
    writer.add("config/max_tokens", np.asarray([args.max_tokens],
                                               dtype=np.int32))
    writer.add("config/bos_id", np.asarray([-1], dtype=np.int32))
    writer.add("config/temperature", _scalar(0.0))

    # --- trained parameters, prompt, and the greedy trace ------------------
    for name, parameter in model.named_parameters():
        writer.add(f"param/{name}", _as_f32(parameter))
    writer.add("prompt/tokens",
               np.asarray(prompt_ids, dtype=np.int32))
    writer.add("generate/tokens", np.asarray(effective, dtype=np.int32))
    writer.add("generate/mask", np.asarray(mask, dtype=np.uint8))

    writer.write()
    return Path(args.out)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .bin path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--window", default=DEFAULT_WINDOW)
    parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
    parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if torch.cuda.is_available():
        raise RuntimeError(
            "the generation fixture must be generated on the CPU; "
            "set CUDA_VISIBLE_DEVICES='' and retry")
    path = generate(args)
    size = path.stat().st_size
    print(
        f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
        f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
        f"seq={args.seq}, prompt_len={args.prompt_len}, "
        f"max_tokens={args.max_tokens}, train_steps={args.train_steps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
