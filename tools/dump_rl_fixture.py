#!/usr/bin/env python3
"""Generate the reinforcement-learning parity fixture,
``tests/data/rl_parity.bin``.

This is the RL cousin of ``tools/dump_train_fixture.py``. It records the inputs
of one short RL run so the file-driven ``rl_step`` binary and
``//tests:rl_parity_test`` can replay it:

* the model configuration and the optimizer/schedule configuration,
* the initial parameter tensors (one ``param/<name>`` record per parameter),
* per step, the recorded rollout (``rollout/<step>/tokens``), the targets with
  ``-1`` at every ignored position (``rollout/<step>/targets``), and the
  per-position advantage weights (``rollout/<step>/advantages``), and
* the model-independent normalization the RL step must use, computed by the
  pure-Python reference below: ``ref/<step>/valid_targets`` and
  ``ref/<step>/divisor`` = ``valid * num_passes * examples_per_rank``
  (docs/training-seam.md section 5.7).

The reference side of a full parity run is the PyTorch ``scripts/chat_rl.py``
rollout and objective. **This host has neither torch nor numpy installed**, so
the script falls back to a deterministic pure-Python reference: it draws the
parameters and the rollout from a fixed 64-bit LCG, and it computes the
normalization reference (the valid-target count and the divisor) in pure
Python. The model trajectory itself is then checked by
``//tests:rl_parity_test`` against the shared C++ step, and the recorded
``ref/`` records pin the normalization that the C++ step owns. The torch path
that records a reference trajectory is a documented future extension, not a
requirement for this fixture.

The container is the same little-endian ``NANOORC1`` format the oracle and
training fixtures use (see ``tools/dump_oracle.py``), so the C++ side reads it
with ``tests/oracle_fixture.h``.

Run it from anywhere::

    python3 tools/dump_rl_fixture.py --out tests/data/rl_parity.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# --- fixture container (kept in sync with tools/dump_oracle.py) -------------

_MAGIC = b"NANOORC1"
_VERSION = 1

_DTYPE_CODES = {
    "f4": 0,  # fp32
    "i4": 1,  # int32
    "f8": 2,  # fp64
    "i8": 3,  # int64
    "u1": 4,  # uint8
}


class FixtureWriter:
    """Append-only writer for the little-endian NANOORC1 container."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: list[bytes] = []

    def add(self, name: str, dtype: str, shape: tuple[int, ...],
            payload: bytes) -> None:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"record name too long: {name!r}")
        if len(shape) > 0xFF:
            raise ValueError(f"record rank too high: {name!r}")
        parts = [
            struct.pack("<H", len(name_bytes)),
            name_bytes,
            struct.pack("<BB", _DTYPE_CODES[dtype], len(shape)),
        ]
        for extent in shape:
            parts.append(struct.pack("<Q", int(extent)))
        parts.append(payload)
        self._records.append(b"".join(parts))

    def write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as handle:
            handle.write(_MAGIC)
            handle.write(struct.pack("<II", _VERSION, len(self._records)))
            for record in self._records:
                handle.write(record)


def _f32(values) -> bytes:
    values = list(values)
    return struct.pack("<%df" % len(values), *values)


def _i32(values) -> bytes:
    values = list(values)
    return struct.pack("<%di" % len(values), *values)


def _i64(values) -> bytes:
    values = list(values)
    return struct.pack("<%dq" % len(values), *values)


def _f32_scalar(value: float) -> bytes:
    return _f32([float(value)])


def _i32_scalar(value: int) -> bytes:
    return _i32([int(value)])


# --- deterministic pure-Python reference RNG --------------------------------


class Rng:
    """A tiny deterministic 64-bit linear congruential generator.

    It is deliberately not `random`: the fixture must be byte-identical across
    Python builds, and the LCG is small enough to audit.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 +
                      1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * (self.next_u64() / float(1 << 64))

    def randrange(self, count: int) -> int:
        return self.next_u64() % count


# --- fixture configuration --------------------------------------------------

# A compact but structurally complete model: grouped-query attention, value
# embeddings on the odd layer, a sliding-window pattern. Small enough that the
# committed fixture is a few tens of kilobytes and the CPU test runs in ms.
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_KV_HEADS = 2
DEFAULT_EMBD = 32
DEFAULT_VOCAB = 64
DEFAULT_PAD_TO = 64
DEFAULT_SEQ = 8
DEFAULT_BATCH = 2
DEFAULT_STEPS = 3
DEFAULT_WINDOW = "SL"
DEFAULT_SEED = 1234
DEFAULT_NUM_PASSES = 2
DEFAULT_EXAMPLES_PER_RANK = 1
# The prompt positions are ignored; the sampled positions carry the advantage.
DEFAULT_PROMPT_LEN = 2

# `Config::has_value_embedding` (include/nanochat/config.h): value embeddings
# are on the layers whose parity matches the last layer's.
_SMEAR_CHANNELS = 24
_VE_GATE_CHANNELS = 12


def has_value_embedding(layer: int, layers: int) -> bool:
    return layer % 2 == (layers - 1) % 2


def parameter_specs(layers: int, heads: int, kv_heads: int, embd: int,
                    padded_vocab: int) -> list[tuple[str, tuple[int, ...]]]:
    """The parameter name and shape list, in the optimizer grouping order.

    It mirrors the constructor order in ``src/model.cc`` so every
    ``param/<name>`` record the C++ model looks up is present.
    """
    head_dim = embd // heads
    query_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    mlp_dim = 4 * embd
    specs: list[tuple[str, tuple[int, ...]]] = []
    specs.append(("lm_head.weight", (padded_vocab, embd)))
    specs.append(("transformer.wte.weight", (padded_vocab, embd)))
    for layer in range(layers):
        if has_value_embedding(layer, layers):
            specs.append((f"value_embeds.{layer}.weight", (padded_vocab,
                                                           kv_dim)))
    specs.append(("resid_lambdas", (layers,)))
    specs.append(("x0_lambdas", (layers,)))
    specs.append(("smear_gate.weight", (_SMEAR_CHANNELS, 1)))
    specs.append(("smear_lambda", (1,)))
    specs.append(("backout_lambda", (1,)))
    for layer in range(layers):
        specs.append((f"transformer.h.{layer}.attn.c_q.weight",
                      (query_dim, embd)))
        specs.append((f"transformer.h.{layer}.attn.c_k.weight",
                      (kv_dim, embd)))
        specs.append((f"transformer.h.{layer}.attn.c_v.weight",
                      (kv_dim, embd)))
        specs.append((f"transformer.h.{layer}.attn.c_proj.weight",
                      (embd, embd)))
        if has_value_embedding(layer, layers):
            specs.append((f"transformer.h.{layer}.attn.ve_gate.weight",
                          (kv_heads, _VE_GATE_CHANNELS)))
        specs.append((f"transformer.h.{layer}.mlp.c_fc.weight",
                      (mlp_dim, embd)))
        specs.append((f"transformer.h.{layer}.mlp.c_proj.weight",
                      (embd, mlp_dim)))
    return specs


def _parameter_values(name: str, shape: tuple[int, ...], rng: Rng) -> bytes:
    count = 1
    for extent in shape:
        count *= extent
    if name == "resid_lambdas":
        return _f32([1.0] * count)
    if name == "x0_lambdas":
        return _f32([0.2] * count)
    if name == "smear_lambda":
        return _f32([0.0] * count)
    if name == "backout_lambda":
        return _f32([0.2] * count)
    if name.endswith("ve_gate.weight"):
        return _f32([rng.uniform(0.0, 0.02) for _ in range(count)])
    if name == "smear_gate.weight":
        return _f32([rng.uniform(0.0, 0.02) for _ in range(count)])
    return _f32([rng.uniform(-0.1, 0.1) for _ in range(count)])


def _advantages(batch: int, values: list[float]) -> list[float]:
    """The per-row advantage: the reward minus the batch mean reward."""
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def generate(args: argparse.Namespace) -> Path:
    layers = args.layers
    heads = args.heads
    kv_heads = args.kv_heads
    embd = args.embd
    vocab = args.vocab
    padded_vocab = args.pad_to
    seq = args.seq
    batch = args.batch
    steps = args.steps
    prompt_len = args.prompt_len
    if padded_vocab < vocab:
        raise SystemExit("--pad-to must be >= --vocab")
    if prompt_len < 0 or prompt_len >= seq:
        raise SystemExit("--prompt-len must be in [0, seq)")

    writer = FixtureWriter(Path(args.out))

    # --- configuration -----------------------------------------------------
    writer.add("config/version", "i4", (1,), _i32_scalar(1))
    writer.add("config/seed", "i8", (1,), _i64([args.seed]))
    writer.add("config/layers", "i4", (1,), _i32_scalar(layers))
    writer.add("config/heads", "i4", (1,), _i32_scalar(heads))
    writer.add("config/kv_heads", "i4", (1,), _i32_scalar(kv_heads))
    writer.add("config/embd", "i4", (1,), _i32_scalar(embd))
    writer.add("config/vocab", "i4", (1,), _i32_scalar(vocab))
    writer.add("config/padded_vocab", "i4", (1,), _i32_scalar(padded_vocab))
    writer.add("config/seq", "i4", (1,), _i32_scalar(seq))
    writer.add("config/batch", "i4", (1,), _i32_scalar(batch))
    writer.add("config/steps", "i4", (1,), _i32_scalar(steps))
    writer.add("config/num_passes", "i4", (1,), _i32_scalar(args.num_passes))
    writer.add("config/examples_per_rank", "i4", (1,),
               _i32_scalar(args.examples_per_rank))
    writer.add("config/prompt_len", "i4", (1,), _i32_scalar(prompt_len))
    writer.add("config/window_pattern", "u1", (len(args.window),),
               args.window.encode("ascii"))
    writer.add("config/opt/unembedding_lr", "f4", (1,),
               _f32_scalar(args.unembedding_lr))
    writer.add("config/opt/embedding_lr", "f4", (1,),
               _f32_scalar(args.embedding_lr))
    writer.add("config/opt/matrix_lr", "f4", (1,),
               _f32_scalar(args.matrix_lr))
    writer.add("config/opt/scalar_lr", "f4", (1,),
               _f32_scalar(args.scalar_lr))
    writer.add("config/opt/weight_decay", "f4", (1,),
               _f32_scalar(args.weight_decay))
    writer.add("config/opt/clip", "f4", (1,), _f32_scalar(args.clip))
    writer.add("config/opt/adam_eps", "f4", (1,),
               _f32_scalar(args.adam_eps))
    writer.add("config/opt/muon_ns_steps", "i4", (1,),
               _i32_scalar(args.muon_ns_steps))
    writer.add("config/opt/muon_beta2", "f4", (1,),
               _f32_scalar(args.muon_beta2))
    writer.add("config/sched/num_iterations", "i4", (1,), _i32_scalar(steps))
    writer.add("config/sched/warmup_steps", "i4", (1,),
               _i32_scalar(args.warmup_steps))
    writer.add("config/sched/warmdown_ratio", "f4", (1,),
               _f32_scalar(args.warmdown_ratio))
    writer.add("config/sched/final_lr_frac", "f4", (1,),
               _f32_scalar(args.final_lr_frac))
    writer.add("config/sched/weight_decay_base", "f4", (1,),
               _f32_scalar(args.weight_decay_base))
    writer.add("config/sched/muon_momentum_warmup_steps", "f4", (1,),
               _f32_scalar(args.muon_momentum_warmup_steps))
    writer.add("config/sched/muon_momentum_start", "f4", (1,),
               _f32_scalar(args.muon_momentum_start))
    writer.add("config/sched/muon_momentum_peak", "f4", (1,),
               _f32_scalar(args.muon_momentum_peak))
    writer.add("config/sched/muon_momentum_final", "f4", (1,),
               _f32_scalar(args.muon_momentum_final))

    # --- initial parameters ------------------------------------------------
    rng = Rng(args.seed)
    for name, shape in parameter_specs(layers, heads, kv_heads, embd,
                                       padded_vocab):
        writer.add(f"param/{name}", "f4", shape,
                   _parameter_values(name, shape, rng))

    # --- recorded rollouts and the pure-Python normalization reference -----
    rows = batch * seq
    for step in range(steps):
        tokens = [rng.randrange(vocab) for _ in range(rows)]
        targets = [-1] * rows
        advantages = [0.0] * rows
        rewards = [rng.uniform(0.0, 1.0) for _ in range(batch)]
        row_advantage = _advantages(batch, rewards)
        for row in range(batch):
            for position in range(seq):
                index = row * seq + position
                if position < prompt_len:
                    continue
                # A sampled position: a valid target and the row's advantage.
                targets[index] = tokens[index]
                advantages[index] = row_advantage[row]
        valid = sum(1 for target in targets if target != -1)
        divisor = valid * args.num_passes * args.examples_per_rank
        writer.add(f"rollout/{step}/tokens", "i4", (batch, seq),
                   _i32(tokens))
        writer.add(f"rollout/{step}/targets", "i4", (batch, seq),
                   _i32(targets))
        writer.add(f"rollout/{step}/advantages", "f4", (batch, seq),
                   _f32(advantages))
        # The model-independent reference the C++ step must reproduce.
        writer.add(f"ref/{step}/valid_targets", "i4", (1,),
                   _i32_scalar(valid))
        writer.add(f"ref/{step}/divisor", "i8", (1,), _i64([divisor]))

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
    parser.add_argument("--pad-to", type=int, default=DEFAULT_PAD_TO)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--window", default=DEFAULT_WINDOW)
    parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--num-passes", type=int, default=DEFAULT_NUM_PASSES)
    parser.add_argument("--examples-per-rank", type=int,
                        default=DEFAULT_EXAMPLES_PER_RANK)
    # Optimizer.
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-beta2", type=float, default=0.9)
    # Scheduler.
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--warmdown-ratio", type=float, default=1.0)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--weight-decay-base", type=float, default=0.0)
    parser.add_argument("--muon-momentum-warmup-steps", type=float, default=0.0)
    parser.add_argument("--muon-momentum-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-peak", type=float, default=0.97)
    parser.add_argument("--muon-momentum-final", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    path = generate(args)
    size = path.stat().st_size
    print(
        f"wrote {path} ({size} bytes, layers={args.layers}, heads={args.heads}, "
        f"kv_heads={args.kv_heads}, embd={args.embd}, vocab={args.vocab}, "
        f"seq={args.seq}, batch={args.batch}, steps={args.steps}, "
        f"num_passes={args.num_passes}, "
        f"examples_per_rank={args.examples_per_rank})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
