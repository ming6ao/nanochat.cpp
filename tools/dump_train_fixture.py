#!/usr/bin/env python3
"""Generate a multi-step *training-parity* fixture.

This is the scaled-up cousin of ``tools/dump_oracle.py``. Where the oracle
fixture records one forward/backward and a couple of optimizer steps, this
fixture records a fixed *training trajectory* of ``N`` steps: the initial
parameters, the exact token/target batch used at every step, and, after every
step, the loss, the global gradient norm, and the L2 norm of every parameter.

Both implementations are then run from the *same* initial parameters over the
*same* batches with the *same* optimizer and learning-rate schedule:

* the reference side is this script, which drives nanochat's PyTorch ``GPT``
  with ``setup_optimizer`` and the same scheduler formulas as
  ``scripts/base_train.py``;
* the candidate side is ``tests/train_parity_test.cc``, which drives
  ``nanochat.cpp``'s ``Model::TrainStep``.

Because the initial weights and the data are shared, the two runs are
comparable step by step. The per-step parameter L2 norm is a compact trajectory
checksum that catches divergence without storing hundreds of megabytes of
weights; pass ``--final-params`` to also record the final parameter tensors.

Output is the same little-endian ``NANOORC1`` container the oracle fixture uses
(see ``tools/dump_oracle.py``), so the C++ side reuses
``tests/oracle_fixture.h``.

The fixture is generated on the CPU and deterministically, like the oracle.
Run it from anywhere::

    python3 tools/dump_train_fixture.py --out /tmp/train_parity.bin

For the production comparison against the measured ``d8_s512`` profile::

    python3 tools/dump_train_fixture.py \
        --out /tmp/train_parity_d8_s512.bin \
        --layers 8 --heads 4 --kv-heads 4 --embd 512 --vocab 32768 \
        --pad-to 32768 --seq 512 --batch 8 --steps 50 --window L \
        --adam-eps 1e-4 --final-params
"""

from __future__ import annotations

import argparse
import math
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


def _as_i32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.int32).cpu().numpy()


def _scalar(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float32)

# A compact but structurally complete default: grouped-query attention, value
# embeddings on odd layers, a sliding window pattern. Small enough that the
# committed fixture is well under a megabyte and the CPU test runs in ms.
DEFAULT_SEED = 1234
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_EMBD = 32
DEFAULT_VOCAB = 64
DEFAULT_SEQ = 8
DEFAULT_BATCH = 2
DEFAULT_STEPS = 20
DEFAULT_WINDOW = "SL"
DEFAULT_PAD_TO = 64
# See the oracle note: at eps=1e-10 the fp32 roundoff of exactly-zero gradient
# directions is amplified into unreproducible updates at this initialization.
DEFAULT_ADAM_EPS = 1e-4


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


def make_stream(count: int, vocab: int, seed: int, mode: str) -> torch.Tensor:
    """A deterministic token stream.

    ``cycle`` has a learnable bigram structure (``next = 7*token + 3 mod V``)
    so the loss descends and the optimizer schedules are actually exercised;
    ``random`` is uniform noise, useful when only numerical parity matters.
    """
    if mode == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (count,), generator=generator, dtype=torch.long)
    stream = torch.empty(count, dtype=torch.long)
    value = 0
    for i in range(count):
        stream[i] = value
        value = (value * 7 + 3) % vocab
    return stream


# --- scheduler mirrors of scripts/base_train.py -----------------------------


def _warmdown_iters(num_iterations: int, warmdown_ratio: float) -> int:
    return int(round(warmdown_ratio * num_iterations))


def get_lr_multiplier(it: int, num_iterations: int, warmup_steps: int,
                      warmdown_ratio: float, final_lr_frac: float) -> float:
    warmdown_iters = _warmdown_iters(num_iterations, warmdown_ratio)
    if warmup_steps > 0 and it < warmup_steps:
        return (it + 1) / warmup_steps
    if warmdown_iters <= 0 or it <= num_iterations - warmdown_iters:
        return 1.0
    progress = (num_iterations - it) / warmdown_iters
    return progress + (1.0 - progress) * final_lr_frac


def get_muon_momentum(it: int, num_iterations: int, warmdown_ratio: float,
                      warmup_steps: float, start: float, peak: float,
                      final: float) -> float:
    warmdown_iters = _warmdown_iters(num_iterations, warmdown_ratio)
    warmdown_start = num_iterations - warmdown_iters
    if warmup_steps > 0 and it < warmup_steps:
        frac = it / warmup_steps
        return (1.0 - frac) * start + frac * peak
    if warmdown_iters > 0 and it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return peak * (1.0 - progress) + final * progress
    return peak


def get_weight_decay(it: int, num_iterations: int, base: float) -> float:
    if num_iterations <= 0:
        return base
    return base * 0.5 * (1.0 + math.cos(math.pi * it / num_iterations))


def total_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def param_l2(parameter: torch.Tensor) -> float:
    return float(parameter.detach().double().pow(2).sum()) ** 0.5


def generate(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    model = GPT(config, pad_vocab_size_to=args.pad_to)
    model.init_weights()
    model.to(torch.float32)
    model.train()
    padded_vocab = model.lm_head.weight.shape[0]

    # `gpt.py` zero-initializes both output projections and the smear scalar,
    # which hides the attention and MLP backward behind a zero path at step 0.
    # For a stronger single-step parity check, optionally give them small
    # nonzero values so the whole graph participates from the first step.
    if args.nonzero_projections:
        generator = torch.Generator().manual_seed(args.seed + 7)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("c_proj.weight"):
                    noise = torch.rand(parameter.shape, generator=generator)
                    parameter.copy_(0.05 * (noise * 2.0 - 1.0))
                elif name == "smear_lambda":
                    parameter.fill_(0.7)
                elif name == "backout_lambda":
                    parameter.fill_(0.3)

    rows = args.batch * args.seq
    stream = make_stream(rows * args.steps + 1, args.vocab, args.seed + 1,
                         args.data)
    tokens = []
    targets = []
    for step in range(args.steps):
        window = stream[step * rows:step * rows + rows + 1]
        tokens.append(window[:-1].reshape(args.batch, args.seq))
        targets.append(window[1:].reshape(args.batch, args.seq))

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
    writer.add("config/opt/unembedding_lr", _scalar(args.unembedding_lr))
    writer.add("config/opt/embedding_lr", _scalar(args.embedding_lr))
    writer.add("config/opt/matrix_lr", _scalar(args.matrix_lr))
    writer.add("config/opt/scalar_lr", _scalar(args.scalar_lr))
    writer.add("config/opt/weight_decay", _scalar(args.weight_decay))
    writer.add("config/opt/clip", _scalar(args.clip))
    writer.add("config/opt/adam_eps", _scalar(args.adam_eps))
    writer.add("config/opt/muon_ns_steps", np.asarray([args.muon_ns_steps],
                                                      dtype=np.int32))
    writer.add("config/opt/muon_beta2", _scalar(args.muon_beta2))
    writer.add("config/sched/num_iterations",
               np.asarray([args.steps], dtype=np.int32))
    writer.add("config/sched/warmup_steps",
               np.asarray([args.warmup_steps], dtype=np.int32))
    writer.add("config/sched/warmdown_ratio", _scalar(args.warmdown_ratio))
    writer.add("config/sched/final_lr_frac", _scalar(args.final_lr_frac))
    writer.add("config/sched/weight_decay_base", _scalar(args.weight_decay_base))
    writer.add("config/sched/muon_momentum_warmup_steps",
               _scalar(args.muon_momentum_warmup_steps))
    writer.add("config/sched/muon_momentum_start", _scalar(args.muon_momentum_start))
    writer.add("config/sched/muon_momentum_peak", _scalar(args.muon_momentum_peak))
    writer.add("config/sched/muon_momentum_final", _scalar(args.muon_momentum_final))

    # --- initial parameters and the exact batches --------------------------
    parameters = list(model.named_parameters())
    for name, parameter in parameters:
        writer.add(f"param/{name}", _as_f32(parameter))
    for step in range(args.steps):
        writer.add(f"batch/{step}/tokens", _as_i32(tokens[step]))
        writer.add(f"batch/{step}/targets", _as_i32(targets[step]))

    # --- optimizer and schedule -------------------------------------------
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
        scalar_lr=args.scalar_lr,
    )
    # setup_optimizer hardcodes a 1e-10 eps and a 0.95 Muon momentum; make both
    # configurable so the recorded trajectory is reproducible and matched.
    for group in optimizer.param_groups:
        if group.get("kind") == "adamw":
            group["eps"] = args.adam_eps
        if group.get("kind") == "muon":
            group["ns_steps"] = args.muon_ns_steps
            group["beta2"] = args.muon_beta2

    def record_trajectory(step: int, loss_value: float) -> None:
        writer.add(f"step/{step}/loss", _scalar(float(loss_value)))
        for name, parameter in parameters:
            writer.add(f"step/{step}/param_l2/{name}", _scalar(param_l2(parameter)))

    # Step 0: the initial loss and the initial parameter trajectory point. The
    # backward also produces the gradients the first update will consume.
    model.zero_grad(set_to_none=True)
    loss = model(tokens[0], targets=targets[0])
    loss.backward()
    record_trajectory(0, float(loss))

    for it in range(args.steps):
        # The gradients in `.grad` were produced by the forward/backward on
        # `tokens[it]` at the parameters after `it` updates: exactly what
        # `TrainStep` consumes on its `it + 1`-th call.
        assert all(p.grad is not None for _, p in parameters)
        writer.add(f"step/{it + 1}/grad_norm", _scalar(total_grad_norm(model)))
        for name, parameter in parameters:
            writer.add(f"step/{it + 1}/grad_l2/{name}",
                       _scalar(param_l2(parameter.grad)))

        lrm = get_lr_multiplier(it, args.steps, args.warmup_steps,
                                args.warmdown_ratio, args.final_lr_frac)
        momentum = get_muon_momentum(
            it, args.steps, args.warmdown_ratio,
            args.muon_momentum_warmup_steps, args.muon_momentum_start,
            args.muon_momentum_peak, args.muon_momentum_final)
        weight_decay = get_weight_decay(it, args.steps, args.weight_decay_base)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get("kind") == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

        optimizer.step()
        model.zero_grad(set_to_none=True)

        # Post-update loss and parameter checksum on the same batch. No
        # gradient is needed, so run it under `no_grad`.
        with torch.no_grad():
            eval_loss = model(tokens[it], targets=targets[it])
        record_trajectory(it + 1, float(eval_loss))

        # The next update consumes the next batch, so build its gradients now.
        if it + 1 < args.steps:
            next_loss = model(tokens[it + 1], targets=targets[it + 1])
            next_loss.backward()

    if args.final_params:
        for name, parameter in parameters:
            writer.add(f"final/param/{name}", _as_f32(parameter))

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
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--window", default=DEFAULT_WINDOW)
    parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
    parser.add_argument("--data", choices=["cycle", "random"], default="random")
    parser.add_argument("--nonzero-projections", action="store_true",
                        help="give c_proj/mlp.c_proj/smear_lambda nonzero "
                             "values so the whole graph is active at step 0")
    parser.add_argument("--final-params", action="store_true",
                        help="also record the final parameter tensors")
    # Optimizer.
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=DEFAULT_ADAM_EPS)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-beta2", type=float, default=0.9)
    # Scheduler.
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.1)
    parser.add_argument("--weight-decay-base", type=float, default=0.05)
    parser.add_argument("--muon-momentum-warmup-steps", type=float, default=4.0)
    parser.add_argument("--muon-momentum-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-peak", type=float, default=0.97)
    parser.add_argument("--muon-momentum-final", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if torch.cuda.is_available():
        raise RuntimeError(
            "the training fixture must be generated on the CPU; "
            "set CUDA_VISIBLE_DEVICES='' and retry")
    path = generate(args)
    size = path.stat().st_size
    print(
        f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
        f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
        f"seq={args.seq}, batch={args.batch}, steps={args.steps}, "
        f"data={args.data}, adam_eps={args.adam_eps:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
