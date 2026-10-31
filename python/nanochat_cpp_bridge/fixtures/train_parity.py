"""The training-parity producer (D11; section 7 of the layering plan).

This module writes ``models/<name>/fixtures/train_parity.bin``: a fixed
training trajectory of ``N`` steps. It records the initial parameters, the
exact token/target batch used at every step, and, after every step, the loss,
the global gradient norm, and the L2 norm of every parameter. Pass
``final_params=True`` to record the final tensors too.

Both implementations then run from the *same* initial parameters over the
*same* batches with the *same* optimizer and schedule:

* the reference side is this module, which drives the reference PyTorch
  ``GPT`` with ``setup_optimizer`` and the scheduler formulas of
  ``scripts/base_train.py``;
* the candidate side is ``tests/train_parity_test.cc``, which drives
  ``nanochat.cpp``'s ``Model::TrainStep``.

Because the weights and the data are shared, the two runs are comparable step
by step, and the per-step parameter L2 norm is a compact trajectory checksum
that catches divergence without storing hundreds of megabytes of weights.

The manifest of ``models/<name>/fixtures/manifest.json`` names this module in
``generated_by`` (section 7.2)::

    "generated_by": "nanochat_cpp_bridge.fixtures.train_parity"

Run it through the one producer command (section 7.3)::

    tools/nanochat parity produce train_parity

or directly, from a virtual environment that holds the extras::

    python3 -m nanochat_cpp_bridge.fixtures.train_parity --out /tmp/t.bin

The run prints the environment it ran in and the hash of the bytes it wrote,
so a Model owner stamps provenance from a run rather than from memory::

    producer {"generated_by": "...", "producer": {"python": "3.12.4", ...}}
    fixture {"path": "...", "sha256": "...", "size_bytes": 161760}

The fixture is generated on the CPU and deterministically: the reference
optimizer is compiled with ``@torch.compile`` in ``gpt.py``, so the run sets
``TORCHDYNAMO_DISABLE`` and runs eagerly, and it hides every GPU before torch
imports so the host's one device stays free for the candidate run.

The container is the pinned ``NANOORC1`` format of
:mod:`nanochat_cpp_bridge.fixtures._container`, which the C++ side reads with
``tests/oracle_fixture.h``.

Extras: ``reference`` (torch and numpy) and ``nanochat``.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .. import _extras
from . import _provenance
from ._container import FixtureWriter

__all__ = [
    "DEFAULT_MODEL",
    "FILENAME",
    "GENERATED_BY",
    "build_parser",
    "check",
    "generate",
    "main",
    "produce",
]

#: The module name a manifest records in ``generated_by`` (section 7.2).
GENERATED_BY = "nanochat_cpp_bridge.fixtures.train_parity"

#: The fixture file this producer writes inside a model package.
FILENAME = "train_parity.bin"

#: The model package selected when none is named.
DEFAULT_MODEL = "nanochat-default"

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
# At eps=1e-10 the fp32 roundoff of exactly-zero gradient directions is
# amplified into unreproducible updates at this initialization.
DEFAULT_ADAM_EPS = 1e-4


def _numpy():
    """The ``numpy`` module, imported where it is used."""
    try:
        import numpy  # noqa: PLC0415 - the lazy import of section 2.1
    except ImportError as error:
        raise _extras.missing("numpy") from error
    return numpy


def _torch():
    """The ``torch`` module, imported where it is used."""
    try:
        import torch  # noqa: PLC0415 - the lazy import of section 2.1
    except ImportError as error:
        raise _extras.missing("torch") from error
    return torch


def _as_f32(tensor):
    """The tensor as a contiguous fp32 NumPy array."""
    torch = _torch()
    return tensor.detach().to(torch.float32).cpu().numpy()


def _as_i32(tensor):
    """The tensor as a contiguous int32 NumPy array."""
    torch = _torch()
    return tensor.detach().to(torch.int32).cpu().numpy()


def _scalar(value: float):
    """One float as a one-element fp32 NumPy array."""
    numpy = _numpy()
    return numpy.asarray([value], dtype=numpy.float32)


def _gpt_classes():
    """The reference ``GPT`` and ``GPTConfig``, imported where they are used."""
    _extras.reference_on_path()
    gpt = _extras.import_package("nanochat.gpt")
    return gpt.GPT, gpt.GPTConfig


def _repository_root() -> Path:
    """The nanochat.cpp repository root, as the core resolves it."""
    from nanochat_cpp.reference import repository_root  # noqa: PLC0415

    return repository_root()


def _default_out(model: str) -> Path:
    """The committed path of the fixture, ``models/<model>/fixtures/``."""
    root = _repository_root()
    spec = root / "models" / model / "spec.json"
    if not spec.is_file():
        raise FileNotFoundError(
            f"unknown model {model!r}: {spec} is missing (the model axis names "
            "a directory under models/); pass out= to write elsewhere")
    return spec.parent / "fixtures" / FILENAME


def build_config(args: argparse.Namespace, config_class):
    """The reference ``GPTConfig`` from the parsed arguments."""
    return config_class(
        sequence_len=args.seq,
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
        window_pattern=args.window,
    )


def make_stream(count: int, vocab: int, seed: int, mode: str):
    """A deterministic token stream.

    ``cycle`` has a learnable bigram structure (``next = 7*token + 3 mod V``)
    so the loss descends and the optimizer schedules are actually exercised;
    ``random`` is uniform noise, useful when only numerical parity matters.
    """
    torch = _torch()
    if mode == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (count,), generator=generator,
                             dtype=torch.long)
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
    """The reference learning-rate multiplier at iteration ``it``."""
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
    """The reference Muon momentum at iteration ``it``."""
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
    """The reference cosine weight-decay schedule at iteration ``it``."""
    if num_iterations <= 0:
        return base
    return base * 0.5 * (1.0 + math.cos(math.pi * it / num_iterations))


def total_grad_norm(model) -> float:
    """The global L2 norm of every gradient on the model."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def param_l2(parameter) -> float:
    """The L2 norm of one tensor."""
    return float(parameter.detach().double().pow(2).sum()) ** 0.5


def _force_cpu() -> None:
    """Hide every GPU and run the reference eagerly, before torch imports.

    The fixture is the oracle, and the host has one shared device that must
    stay free for the candidate run. ``gpt.py`` compiles the reference
    optimizer with ``@torch.compile``, so the run is eager too: the fixture
    must be deterministic and must need no C++ compiler.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def generate(args: argparse.Namespace) -> Path:
    """Write the fixture named by ``args.out`` and return its path."""
    _force_cpu()

    numpy = _numpy()
    torch = _torch()
    GPT, GPTConfig = _gpt_classes()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    config = build_config(args, GPTConfig)
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
    writer.add("config/version", numpy.asarray([1], dtype=numpy.int32))
    writer.add("config/seed", numpy.asarray([args.seed], dtype=numpy.int64))
    writer.add("config/layers", numpy.asarray([args.layers], dtype=numpy.int32))
    writer.add("config/heads", numpy.asarray([args.heads], dtype=numpy.int32))
    writer.add("config/kv_heads",
               numpy.asarray([args.kv_heads], dtype=numpy.int32))
    writer.add("config/embd", numpy.asarray([args.embd], dtype=numpy.int32))
    writer.add("config/vocab", numpy.asarray([args.vocab], dtype=numpy.int32))
    writer.add("config/padded_vocab",
               numpy.asarray([padded_vocab], dtype=numpy.int32))
    writer.add("config/seq", numpy.asarray([args.seq], dtype=numpy.int32))
    writer.add("config/batch", numpy.asarray([args.batch], dtype=numpy.int32))
    writer.add("config/steps", numpy.asarray([args.steps], dtype=numpy.int32))
    writer.add(
        "config/window_pattern",
        numpy.frombuffer(args.window.encode("ascii"), dtype=numpy.uint8).copy(),
    )
    writer.add("config/opt/unembedding_lr", _scalar(args.unembedding_lr))
    writer.add("config/opt/embedding_lr", _scalar(args.embedding_lr))
    writer.add("config/opt/matrix_lr", _scalar(args.matrix_lr))
    writer.add("config/opt/scalar_lr", _scalar(args.scalar_lr))
    writer.add("config/opt/weight_decay", _scalar(args.weight_decay))
    writer.add("config/opt/clip", _scalar(args.clip))
    writer.add("config/opt/adam_eps", _scalar(args.adam_eps))
    writer.add("config/opt/muon_ns_steps",
               numpy.asarray([args.muon_ns_steps], dtype=numpy.int32))
    writer.add("config/opt/muon_beta2", _scalar(args.muon_beta2))
    writer.add("config/sched/num_iterations",
               numpy.asarray([args.steps], dtype=numpy.int32))
    writer.add("config/sched/warmup_steps",
               numpy.asarray([args.warmup_steps], dtype=numpy.int32))
    writer.add("config/sched/warmdown_ratio", _scalar(args.warmdown_ratio))
    writer.add("config/sched/final_lr_frac", _scalar(args.final_lr_frac))
    writer.add("config/sched/weight_decay_base",
               _scalar(args.weight_decay_base))
    writer.add("config/sched/muon_momentum_warmup_steps",
               _scalar(args.muon_momentum_warmup_steps))
    writer.add("config/sched/muon_momentum_start",
               _scalar(args.muon_momentum_start))
    writer.add("config/sched/muon_momentum_peak",
               _scalar(args.muon_momentum_peak))
    writer.add("config/sched/muon_momentum_final",
               _scalar(args.muon_momentum_final))

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
            writer.add(f"step/{step}/param_l2/{name}",
                       _scalar(param_l2(parameter)))

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


def build_parser() -> argparse.ArgumentParser:
    """The producer's argument parser."""
    parser = argparse.ArgumentParser(
        prog="train_parity",
        description="Produce the training-parity fixture "
                    f"({FILENAME}).")
    parser.add_argument("--out", default=None,
                        help="output .bin path (default: "
                             f"models/<model>/fixtures/{FILENAME})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model package under models/ (default: "
                             f"{DEFAULT_MODEL})")
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temporary directory and "
                             "compare the bytes with the committed fixture")
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
    return parser


def _options(out: str | os.PathLike[str] | None, model: str,
             options: dict[str, Any]) -> argparse.Namespace:
    """The parser defaults with ``options`` applied and the output resolved."""
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in options.items():
        if not hasattr(args, key):
            known = sorted(vars(args))
            raise TypeError(
                f"unknown option {key!r}; the options are "
                + ", ".join(known))
        setattr(args, key, value)
    args.model = model
    args.out = str(Path(out) if out is not None else _default_out(model))
    return args


def _emit_records(path: Path) -> None:
    """Print the section 7.2 producer record and the fixture record."""
    _provenance.emit("producer", _provenance.producer_record(GENERATED_BY))
    _provenance.emit("fixture", _provenance.fixture_record(path))


def produce(out: str | os.PathLike[str] | None = None, *,
            model: str = DEFAULT_MODEL, **options: Any) -> Path:
    """Produce the fixture, print its provenance, and return its path.

    ``options`` are the long-option names of :func:`build_parser` with
    underscores (``final_params=True``, ``steps=4``), so a host forwards what
    its command line parsed without translating anything.
    """
    args = _options(out, model, options)
    if args.check:
        return check(out=out, model=model, **options)
    _force_cpu()
    torch = _torch()
    if torch.cuda.is_available():
        raise RuntimeError(
            "the training fixture must be generated on the CPU; "
            "set CUDA_VISIBLE_DEVICES='' and retry")
    path = generate(args)
    _emit_records(path)
    return path


def check(out: str | os.PathLike[str] | None = None, *,
          model: str = DEFAULT_MODEL, **options: Any) -> Path:
    """Regenerate into a temporary directory and compare the committed bytes.

    The comparison is the fixture hash of the manifest, computed from the two
    files rather than read from a table, so a drift between this producer and
    the committed bytes fails here with both digests named. Nothing is written
    into ``models/``.
    """
    committed = Path(out) if out is not None else _default_out(model)
    if not committed.is_file():
        raise FileNotFoundError(f"no committed fixture at {committed}")
    with tempfile.TemporaryDirectory(prefix="train_parity_") as directory:
        args = _options(Path(directory) / FILENAME, model, options)
        args.check = False
        produced = generate(args)
        produced_hash = _provenance.sha256(produced)
    committed_hash = _provenance.sha256(committed)
    _provenance.emit("check", {
        "path": str(committed),
        "committed_sha256": committed_hash,
        "produced_sha256": produced_hash,
        "match": committed_hash == produced_hash,
    })
    if committed_hash != produced_hash:
        raise SystemExit(
            f"the producer no longer reproduces {committed}: "
            f"{produced_hash} != {committed_hash}")
    return committed


def main(argv: Sequence[str] | None = None) -> int:
    """The producer command line; return the process exit code."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    options = vars(args)
    out = options.pop("out")
    model = options.pop("model")
    produce(out, model=model, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
