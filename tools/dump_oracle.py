#!/usr/bin/env python3
"""Generate the CPU oracle fixture, ``tests/data/debug_state.bin``.

The fixture is *data*: it is produced once, offline, from nanochat's reference
``gpt.py`` (PyTorch) and committed to the tree. Runtime tests never import
torch; they parse this file with the dependency-free reader in
``tests/oracle_fixture.h``.

What it contains
----------------
For one fixed seed and one tiny configuration:

* ``config/*``       the exact architecture and optimizer hyperparameters;
* ``input/tokens``   the int32 token ids fed to the model;
* ``input/targets``  the int32 next-token targets;
* ``forward/raw_logits``   the pre-softcap classifier output;
* ``forward/logits``       the post-softcap output (what ``gpt.py`` returns);
* ``forward/loss``         the batch-mean cross-entropy;
* ``grad/raw_logits``      d(loss)/d(raw logits), analytic;
* ``grad/logits``          d(loss)/d(post-softcap logits), analytic;
* ``param/<name>``         every initial parameter, named as in ``gpt.py``;
* ``grad/<name>``          the autograd gradient of every parameter;
* ``step/<k>/loss``        the loss after ``k`` optimizer steps (``k`` = 0..N);
* ``step/<k>/param/<name>`` every parameter after ``k`` optimizer steps.

The optimizer trajectory uses nanochat's canonical ``setup_optimizer``
grouping, with *no* gradient clipping and a constant learning rate, so a C++
optimizer can reproduce it exactly. It overrides the AdamW ``eps`` (default
``1e-4`` instead of nanochat's ``1e-10``): several fixture parameters have a
true gradient of exactly zero at this initialization, so their fp32 backward
is pure roundoff at the ``1e-10`` scale, which ``eps=1e-10`` amplifies into
unreproducible updates. The chosen ``eps`` is recorded as
``config/opt/adam_eps`` so the C++ test can match it.

Binary format (little-endian)
-----------------------------
::

    header   : magic[8] = b"NANOORC1", uint32 version, uint32 num_records
    record   : uint16 name_len, name bytes,
               uint8 dtype (0=fp32, 1=int32, 2=fp64, 3=int64, 4=uint8),
               uint8 ndim, uint64 shape[ndim],
               payload (product(shape) * dtype_size bytes)

Run it from anywhere; it locates the repository root from ``__file__``::

    python3 tools/dump_oracle.py
    python3 tools/dump_oracle.py --seed 1234 --steps 3 --out tests/data/debug_state.bin

The script forces the CPU and disables ``torch.compile`` so the output is
deterministic and never touches the local GPU.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# Force CPU *before* importing torch: the fixture is the CPU oracle and the
# host has one shared GPU that must not be touched here.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# The reference optimizer is compiled with @torch.compile in gpt.py; run it
# eagerly so the fixture is deterministic and needs no C++ compiler.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# The repository root holds the `nanochat` package; this file lives at
# <root>/nanochat.cpp/tools/dump_oracle.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nanochat.gpt import GPT, GPTConfig  # noqa: E402

# Tiny but structurally complete: two blocks, grouped-query attention, value
# embeddings on the last layer, sliding-window pattern. Small enough that the
# whole fixture is a few hundred kilobytes.
# n_embd must be at least 24: gpt.py's smear gate reads x[..., :24] and the
# value-embedding gate reads x[..., :12]. head_dim = n_embd / n_head = 4.
DEFAULT_SEED = 1234
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_EMBD = 32
DEFAULT_VOCAB = 64
DEFAULT_SEQ = 8
DEFAULT_BATCH = 2
DEFAULT_STEPS = 3
DEFAULT_WINDOW = "SL"
DEFAULT_PAD_TO = 64
SOFTCAP = 15.0

# Dtype codes shared with tests/oracle_fixture.h.
_DTYPE_CODES = {
    np.dtype(np.float32): 0,
    np.dtype(np.int32): 1,
    np.dtype(np.float64): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.uint8): 4,
}


class FixtureWriter:
    """Append-only writer for the little-endian record container."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: list[bytes] = []

    def add(self, name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        # Normalise to little-endian so the file is portable.
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


def _as_i32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.int32).cpu().numpy()


def _scalar(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float32)


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


def generate(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    model = GPT(config, pad_vocab_size_to=args.pad_to)
    model.init_weights()
    model.to(torch.float32)
    model.eval()

    padded_vocab = model.lm_head.weight.shape[0]

    generator = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(
        0, args.vocab, (args.batch, args.seq), generator=generator, dtype=torch.long
    )
    targets = torch.randint(
        0, args.vocab, (args.batch, args.seq), generator=generator, dtype=torch.long
    )

    # Capture the pre-softcap classifier output from the lm_head forward hook.
    raw_holder: dict[str, torch.Tensor] = {}

    def capture_raw(_module, _inputs, output):
        raw_holder["raw"] = output

    handle = model.lm_head.register_forward_hook(capture_raw)

    model.zero_grad(set_to_none=True)
    capped = model(tokens)  # post-softcap logits, requires grad
    loss = F.cross_entropy(
        capped.reshape(-1, args.vocab), targets.reshape(-1), reduction="mean"
    )
    loss.backward()
    handle.remove()

    raw = raw_holder["raw"][..., : args.vocab]
    with torch.no_grad():
        recapped = SOFTCAP * torch.tanh(raw / SOFTCAP)
    if not torch.allclose(capped.detach(), recapped, atol=1e-6, rtol=1e-5):
        raise AssertionError("softcap reconstruction does not match gpt.py forward")

    # Analytic classifier gradients: softmax cross-entropy through the softcap.
    with torch.no_grad():
        probs = F.softmax(capped.detach(), dim=-1)
        one_hot = F.one_hot(targets, num_classes=args.vocab).to(probs.dtype)
        count = float(args.batch * args.seq)
        grad_capped = (probs - one_hot) / count
        softcap_derivative = 1.0 - (capped.detach() / SOFTCAP).square()
        grad_raw = grad_capped * softcap_derivative

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
    writer.add("config/batch", np.asarray([args.batch], dtype=np.int32))
    writer.add("config/steps", np.asarray([args.steps], dtype=np.int32))
    writer.add(
        "config/window_pattern",
        np.frombuffer(args.window.encode("ascii"), dtype=np.uint8).copy(),
    )
    # Optimizer hyperparameters (nanochat setup_optimizer defaults, no clipping).
    writer.add("config/opt/unembedding_lr", _scalar(0.004))
    writer.add("config/opt/embedding_lr", _scalar(0.2))
    writer.add("config/opt/matrix_lr", _scalar(0.02))
    writer.add("config/opt/scalar_lr", _scalar(0.5))
    writer.add("config/opt/weight_decay", _scalar(0.0))
    writer.add("config/opt/clip", _scalar(0.0))
    writer.add("config/opt/adam_eps", _scalar(args.adam_eps))

    # --- inputs ------------------------------------------------------------
    writer.add("input/tokens", _as_i32(tokens))
    writer.add("input/targets", _as_i32(targets))

    # --- reference forward and backward ------------------------------------
    writer.add("forward/raw_logits", _as_f32(raw))
    writer.add("forward/logits", _as_f32(capped))
    writer.add("forward/loss", _scalar(float(loss)))
    writer.add("grad/raw_logits", _as_f32(grad_raw))
    writer.add("grad/logits", _as_f32(grad_capped))

    # Initial parameters and their gradients, named as in gpt.py.
    for name, parameter in model.named_parameters():
        writer.add(f"param/{name}", _as_f32(parameter))
        if parameter.grad is None:
            raise AssertionError(f"parameter {name!r} has no gradient")
        writer.add(f"grad/{name}", _as_f32(parameter.grad))

    # --- a few optimizer steps --------------------------------------------
    writer.add("step/0/loss", _scalar(float(loss)))
    for name, parameter in model.named_parameters():
        writer.add(f"step/0/param/{name}", _as_f32(parameter))

    optimizer = model.setup_optimizer()
    # `setup_optimizer` hardcodes eps=1e-10. Several fixture parameters have a
    # true gradient of exactly zero at this initialization (the zero-initialized
    # output projections plus RMSNorm scale invariance), so their fp32 backward
    # is pure roundoff at the ~1e-10 scale. At eps=1e-10 AdamW amplifies that
    # roundoff into large spurious updates and the trajectory is not
    # reproducible. A larger eps freezes those directions while leaving the
    # smallest real gradient (mlp.c_proj, ~1.7e-4) essentially intact.
    for group in optimizer.param_groups:
        if group.get("kind") == "adamw":
            group["eps"] = args.adam_eps
    for step in range(1, args.steps + 1):
        optimizer.step()
        model.zero_grad(set_to_none=True)
        step_loss = model(tokens, targets=targets)
        step_loss.backward()
        writer.add(f"step/{step}/loss", _scalar(float(step_loss)))
        for name, parameter in model.named_parameters():
            writer.add(f"step/{step}/param/{name}", _as_f32(parameter))

    writer.write()
    return Path(args.out)


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_out = REPO_ROOT / "nanochat.cpp" / "tests" / "data" / "debug_state.bin"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(default_out), help="output .bin path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--embd", type=int, default=DEFAULT_EMBD)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--window", default=DEFAULT_WINDOW)
    parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if torch.cuda.is_available():
        raise RuntimeError(
            "the oracle fixture must be generated on the CPU; "
            "set CUDA_VISIBLE_DEVICES='' and retry"
        )
    path = generate(args)
    size = path.stat().st_size
    print(
        f"wrote {path} ({size} bytes, seed={args.seed}, "
        f"layers={args.layers}, heads={args.heads}, kv_heads={args.kv_heads}, "
        f"embd={args.embd}, vocab={args.vocab}, seq={args.seq}, "
        f"batch={args.batch}, steps={args.steps})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
