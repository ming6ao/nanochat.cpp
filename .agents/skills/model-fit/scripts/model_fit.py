#!/usr/bin/env python3
"""Fit nanochat model architectures to hardware and topology.

This skill answers one question: for a given accelerator, a given device count,
and a given interconnect, which nanochat architectures are the largest, the
longest-context, and the fastest that fit?

The engine has three inputs:

* a :class:`Hardware` profile (memory, fp32/fp16/bf16 rates);
* a :class:`Topology` (device count, interconnect, sync overlap);
* a :class:`ModelSpec` (the nanochat shape and the sequence length).

The memory model is exact. It mirrors the nanochat.cpp source. The throughput
model is an estimate. The ``references/`` directory records the sources.

Run ``model_fit.py --help`` for the command line.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from typing import Iterable

__all__ = [
    "DEVICES",
    "INTERCONNECTS",
    "PRECISIONS",
    "BEST_FIT_T4",
    "Candidate",
    "Hardware",
    "ModelSpec",
    "Topology",
    "answer",
    "best_batch",
    "card_bytes",
    "flops_per_token",
    "frontier",
    "hardware",
    "memory_breakdown",
    "optimizer_bytes",
    "parameter_counts",
    "recommend",
    "search",
    "shape_flags",
    "step_seconds",
    "tokens_per_second",
    "train_arena_bytes",
]

GIB = 1024**3

#: The channels of the value gate and the smear gate in nanochat.cpp.
_VE_GATE_CHANNELS = 12
_SMEAR_CHANNELS = 24


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """One accelerator type.

    ``memory_bytes`` is the total device memory. ``reserve_bytes`` is the
    overhead of the CUDA context and the library workspaces. The model sees the
    difference. ``fp32_flops`` is the CUDA-core rate. ``fp16_flops`` and
    ``bf16_flops`` are the tensor-core rates, or zero when the device has no
    path at that precision.
    """

    name: str
    memory_bytes: int
    fp32_flops: float
    fp16_flops: float = 0.0
    bf16_flops: float = 0.0
    tensor_cores: bool = False
    reserve_bytes: int = int(1.5 * GIB)
    gemm_efficiency: float = 0.8
    attention_efficiency: float = 0.25
    #: An explicit attention rate. Zero means "derive from the fp32 rate".
    attention_flops: float = 0.0

    def budget_bytes(self) -> int:
        """The memory the model may use on one device."""
        return max(0, self.memory_bytes - self.reserve_bytes)

    def peak_flops(self, precision: str) -> float:
        """The dense peak rate for ``precision``, or zero when unsupported."""
        if precision == "fp32":
            return self.fp32_flops
        if precision == "fp16":
            return self.fp16_flops
        if precision == "bf16":
            return self.bf16_flops
        raise ValueError(f"unknown precision: {precision}")

    def gemm_rate(self, precision: str) -> float:
        """The estimated achieved GEMM rate for ``precision``."""
        return self.peak_flops(precision) * self.gemm_efficiency

    def attention_rate(self) -> float:
        """The estimated attention rate. The kernel uses no tensor cores."""
        if self.attention_flops > 0:
            return self.attention_flops
        return self.fp32_flops * self.attention_efficiency

    def supports(self, precision: str) -> bool:
        """True when the device has a path at ``precision``."""
        return self.peak_flops(precision) > 0.0


#: The device table. The rates and the memory come from nanochat.cpp
#: ``src/device_profile.cc``. See references/devices.md.
DEVICES: dict[str, Hardware] = {
    "h200": Hardware("h200", 141 * GIB, 67.0e12, 989e12, 989e12,
                     tensor_cores=True),
    "h100": Hardware("h100", 80 * GIB, 66.9e12, 989e12, 989e12,
                     tensor_cores=True),
    "a100": Hardware("a100", 80 * GIB, 19.5e12, 312e12, 312e12,
                     tensor_cores=True),
    "t4": Hardware("t4", 16 * GIB, 8.1e12, 65e12, 0.0, tensor_cores=True),
    "gtx1080ti": Hardware("gtx1080ti", 11 * GIB, 11.34e12, 0.177e12, 0.0),
}


def hardware(name: str) -> Hardware:
    """Look up a device, or raise ``KeyError`` with the known names."""
    key = name.strip().lower()
    if key not in DEVICES:
        raise KeyError(
            f"unknown device {name!r}; known: {', '.join(sorted(DEVICES))}")
    return DEVICES[key]


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

#: Per-rank unidirectional bandwidth in gigabytes per second (1e9 bytes/s).
#: See references/topology.md.
INTERCONNECTS: dict[str, float] = {
    "nvlink1": 160.0,
    "nvlink2": 300.0,
    "nvlink3": 600.0,
    "nvlink4": 900.0,
    "pcie3": 16.0,
    "pcie4": 32.0,
    "pcie5": 64.0,
    "shared-memory": 20.0,
    "ethernet-100g": 12.5,
    "ethernet-400g": 50.0,
    "infiniband-ndr": 50.0,
}


@dataclass(frozen=True)
class Topology:
    """The device group and its interconnect.

    The parallelism is data parallel: every device holds the full model and
    processes its own micro-batch. The devices all-reduce the gradients once per
    optimizer step. ``overlap`` is the fraction of the reduction that hides
    behind the backward pass.
    """

    world_size: int = 1
    interconnect: str = "pcie3"
    bandwidth_gbps: float | None = None
    latency_us: float = 5.0
    overlap: float = 0.0

    def bandwidth(self) -> float:
        """The per-rank bandwidth in bytes per second."""
        if self.bandwidth_gbps is not None:
            return self.bandwidth_gbps * 1e9
        if self.interconnect not in INTERCONNECTS:
            raise KeyError(
                f"unknown interconnect {self.interconnect!r}; known: "
                f"{', '.join(sorted(INTERCONNECTS))}")
        return INTERCONNECTS[self.interconnect] * 1e9

    def sync_seconds(self, payload_bytes: float) -> float:
        """The time the ring all-reduce adds to one step.

        A ring all-reduce moves ``2 * (N - 1) / N`` of the payload on each
        rank. ``overlap`` removes a fraction of the result.
        """
        if self.world_size <= 1 or payload_bytes <= 0:
            return 0.0
        n = self.world_size
        moved = 2.0 * (n - 1) / n * payload_bytes
        full = moved / self.bandwidth() + self.latency_us * 1e-6
        return (1.0 - self.overlap) * full


#: The Kaggle two-T4 group. The cards share a host and have no NVLink.
KAGGLE_T4X2 = Topology(world_size=2, interconnect="pcie3")


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

#: Bytes for the value, the gradient, and the fp32 master weight. nanochat.cpp
#: keeps the master separate only in a reduced-precision build. See
#: references/devices.md.
PRECISIONS: dict[str, dict[str, int]] = {
    "fp32": {"value": 4, "grad": 4, "master": 0},
    "fp16": {"value": 2, "grad": 2, "master": 4},
    "bf16": {"value": 2, "grad": 4, "master": 4},
}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A nanochat architecture and a sequence length."""

    layers: int
    hidden: int
    heads: int
    kv_heads: int
    seq: int
    vocab: int = 32768
    window: str = "SSSL"
    value_embedding: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def query_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def mlp_dim(self) -> int:
        return 4 * self.hidden

    def has_value_embedding(self, layer: int) -> bool:
        if not self.value_embedding:
            return False
        return layer % 2 == (self.layers - 1) % 2

    def short_window(self) -> int:
        return ((self.seq // 4 + 127) // 128) * 128

    def window_left(self, layer: int) -> int:
        """The left attention extent, or -1 for full context."""
        if layer >= self.layers - 1:
            return -1
        if not self.window:
            return -1
        kind = self.window[layer % len(self.window)]
        return self.short_window() if kind == "S" else -1

    def validate(self) -> None:
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.heads < 1 or self.hidden % self.heads != 0:
            raise ValueError("hidden must divide by heads")
        if self.kv_heads < 1 or self.heads % self.kv_heads != 0:
            raise ValueError("heads must divide by kv_heads")
        if self.seq < 1:
            raise ValueError("seq must be positive")


def parameter_counts(spec: ModelSpec) -> dict[str, int]:
    """The parameter count by group. Mirrors ``CountParams`` in the source."""
    spec.validate()
    hidden = spec.hidden
    vocab = spec.vocab
    matrices = 0
    embeddings = vocab * hidden  # transformer.wte.weight
    for layer in range(spec.layers):
        matrices += spec.query_dim * hidden
        matrices += spec.kv_dim * hidden
        matrices += spec.kv_dim * hidden
        matrices += hidden * hidden
        if spec.has_value_embedding(layer):
            matrices += spec.kv_heads * _VE_GATE_CHANNELS
            embeddings += vocab * spec.kv_dim
        matrices += spec.mlp_dim * hidden
        matrices += hidden * spec.mlp_dim
    lm_head = vocab * hidden
    scalars = 2 * spec.layers + _SMEAR_CHANNELS + 1 + 1
    return {
        "matrices": matrices,
        "lm_head": lm_head,
        "embeddings": embeddings,
        "scalars": scalars,
        "total": matrices + lm_head + embeddings + scalars,
    }


def train_arena_bytes(spec: ModelSpec, batch: int, value_bytes: int) -> int:
    """The training activation arena. Mirrors ``BuildTrainWorkspace``.

    ``value_bytes`` is the size of one compute element. The float statistics
    buffers stay four bytes.
    """
    hidden = spec.hidden
    vocab = spec.vocab
    rows = batch * spec.seq
    stats = batch * spec.heads * spec.seq * 2
    per_layer = (6 * hidden + 2 * spec.query_dim + 4 * spec.kv_dim
                 + 2 * spec.mlp_dim + spec.kv_heads)
    glob = 16 * hidden + 2 * vocab + 1
    scratch = spec.query_dim + 3 * spec.kv_dim + 5 * hidden + 2 * spec.mlp_dim
    units = spec.layers * per_layer + glob + scratch
    compute = value_bytes * units * rows
    floats = 4 * (spec.layers * (2 * rows + stats) + 4 * rows)
    return compute + floats + 256


def optimizer_bytes(spec: ModelSpec) -> int:
    """The optimizer state. Mirrors ``NanochatOptimizer::Build``.

    AdamW holds a first and a second moment in fp32. Muon holds one momentum
    buffer of the matrix size and one factored buffer of the long side. The
    state stays fp32 at every precision.
    """
    counts = parameter_counts(spec)
    adamw = counts["lm_head"] + counts["embeddings"] + counts["scalars"]
    total = 8 * adamw  # m and v, 4 bytes each
    shapes: dict[tuple[int, int], int] = {}
    for layer in range(spec.layers):
        for shape in ((spec.query_dim, spec.hidden),
                      (spec.kv_dim, spec.hidden),
                      (spec.kv_dim, spec.hidden),
                      (spec.hidden, spec.hidden),
                      (spec.mlp_dim, spec.hidden),
                      (spec.hidden, spec.mlp_dim)):
            shapes[shape] = shapes.get(shape, 0) + 1
        if spec.has_value_embedding(layer):
            shape = (spec.kv_heads, _VE_GATE_CHANNELS)
            shapes[shape] = shapes.get(shape, 0) + 1
    for (rows, cols), number in shapes.items():
        total += 4 * number * (rows * cols)  # buf1
        total += 4 * number * max(rows, cols)  # buf2
    return total


def memory_breakdown(spec: ModelSpec, batch: int,
                     precision: str) -> dict[str, int]:
    """The memory one device needs, by group.

    Data parallel keeps the full model on each device. The value, the gradient,
    the master weight, and the optimizer state do not divide with the device
    count.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")
    layout = PRECISIONS[precision]
    count = parameter_counts(spec)["total"]
    return {
        "params": count * layout["value"],
        "grads": count * layout["grad"],
        "master": count * layout["master"],
        "optimizer": optimizer_bytes(spec),
        "arena": train_arena_bytes(spec, batch, layout["value"]),
    }


def card_bytes(spec: ModelSpec, batch: int, precision: str) -> int:
    """The total memory one device needs for training at ``batch``."""
    return sum(memory_breakdown(spec, batch, precision).values())


def flops_per_token(spec: ModelSpec) -> float:
    """The training arithmetic for one token. Includes forward and backward.

    Mirrors ``EstimateFlopsPerToken``. The value embeddings are lookups, so
    they do not enter the matmul term.
    """
    counts = parameter_counts(spec)
    matmul = counts["total"] - counts["embeddings"]
    attention = 0.0
    for layer in range(spec.layers):
        left = spec.window_left(layer)
        effective = min(left, spec.seq) if left >= 0 else spec.seq
        attention += 12.0 * spec.heads * spec.head_dim * effective
    return 6.0 * matmul + attention


def step_seconds(spec: ModelSpec, batch: int, hw: Hardware,
                 topo: Topology, precision: str) -> float:
    """The estimated time of one optimizer step on one rank.

    The step has three parts: the GEMM work, the attention work, and the
    gradient reduction. The reduction amortizes over ``grad_accum`` if the
    caller sets one; this function models one micro-batch.
    """
    if not hw.supports(precision):
        raise ValueError(f"{hw.name} has no {precision} path")
    counts = parameter_counts(spec)
    matmul = counts["total"] - counts["embeddings"]
    tokens = batch * spec.seq
    gemm_flops = 6.0 * matmul * tokens
    attention_flops = 0.0
    for layer in range(spec.layers):
        left = spec.window_left(layer)
        effective = min(left, spec.seq) if left >= 0 else spec.seq
        attention_flops += (12.0 * spec.heads * spec.head_dim * effective
                            * tokens)
    compute = (gemm_flops / hw.gemm_rate(precision)
               + attention_flops / hw.attention_rate())
    payload = counts["total"] * PRECISIONS[precision]["grad"]
    return compute + topo.sync_seconds(payload)


def tokens_per_second(spec: ModelSpec, batch: int, hw: Hardware,
                      topo: Topology, precision: str) -> float:
    """The estimated global tokens per second for one micro-batch."""
    seconds = step_seconds(spec, batch, hw, topo, precision)
    if seconds <= 0:
        return 0.0
    return topo.world_size * batch * spec.seq / seconds


def scaling_efficiency(spec: ModelSpec, batch: int, hw: Hardware,
                       topo: Topology, precision: str) -> float:
    """The achieved rate over the ideal rate.

    The ideal rate is ``world_size`` times the one-device rate. The gap is the
    gradient reduction.
    """
    if topo.world_size <= 1:
        return 1.0
    single = tokens_per_second(spec, batch, hw, Topology(world_size=1),
                               precision)
    if single <= 0:
        return 0.0
    return tokens_per_second(spec, batch, hw, topo, precision) \
        / (topo.world_size * single)


def best_batch(spec: ModelSpec, hw: Hardware, precision: str,
               max_batch: int = 256) -> int:
    """The largest micro-batch that fits one device."""
    budget = hw.budget_bytes()
    best = 0
    for batch in range(1, max_batch + 1):
        if card_bytes(spec, batch, precision) <= budget:
            best = batch
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One feasible point."""

    spec: ModelSpec
    precision: str
    batch: int
    params: int
    bytes: int
    tokens_per_second: float
    efficiency: float
    sync_fraction: float
    breakdown: dict[str, int]

    @property
    def gib(self) -> float:
        return self.bytes / GIB

    @property
    def rows(self) -> int:
        return self.batch * self.spec.seq

    def label(self) -> str:
        spec = self.spec
        return (f"L{spec.layers} C{spec.hidden} H{spec.heads} "
                f"KV{spec.kv_heads} T{spec.seq} b{self.batch}")


def _candidate(spec: ModelSpec, batch: int, hw: Hardware, topo: Topology,
               precision: str) -> Candidate:
    counts = parameter_counts(spec)
    compute = step_seconds(spec, batch, hw, replace(topo, overlap=topo.overlap),
                           precision)
    sync = topo.sync_seconds(counts["total"] * PRECISIONS[precision]["grad"])
    return Candidate(
        spec=spec, precision=precision, batch=batch,
        params=counts["total"],
        bytes=card_bytes(spec, batch, precision),
        tokens_per_second=tokens_per_second(spec, batch, hw, topo, precision),
        efficiency=scaling_efficiency(spec, batch, hw, topo, precision),
        sync_fraction=(sync / compute) if compute > 0 else 0.0,
        breakdown=memory_breakdown(spec, batch, precision))


def grid(presets: str = "default") -> Iterable[ModelSpec]:
    """The architecture grid to search."""
    if presets == "quick":
        layers = (12, 16, 24)
        hidden_heads = ((768, 6), (1024, 8), (1536, 12))
        seqs = (2048, 4096)
    elif presets == "wide":
        layers = (8, 12, 16, 20, 24, 32, 40, 48)
        hidden_heads = ((512, 4), (768, 6), (1024, 8), (1280, 10),
                        (1536, 12), (2048, 16), (2560, 20), (3072, 24))
        seqs = (1024, 2048, 4096, 8192, 16384)
    else:
        layers = (12, 16, 20, 24, 32)
        hidden_heads = ((768, 6), (1024, 8), (1280, 10), (1536, 12),
                        (2048, 16), (2560, 20))
        seqs = (1024, 2048, 4096, 8192)
    for layer in layers:
        for hidden, heads in hidden_heads:
            for kv_div in (1, 2, 4):
                kv_heads = heads // kv_div
                if kv_heads < 1:
                    continue
                for seq in seqs:
                    yield ModelSpec(layer, hidden, heads, kv_heads, seq)


def search(hw: Hardware, topo: Topology, precision: str,
           specs: Iterable[ModelSpec] | None = None,
           min_seq: int = 1, min_params: int = 0,
           presets: str = "default") -> list[Candidate]:
    """Every feasible candidate that passes the filters."""
    out: list[Candidate] = []
    if not hw.supports(precision):
        return out
    for spec in (specs if specs is not None else grid(presets)):
        if spec.seq < min_seq:
            continue
        batch = best_batch(spec, hw, precision)
        if batch < 1:
            continue
        row = _candidate(spec, batch, hw, topo, precision)
        if row.params < min_params:
            continue
        out.append(row)
    return out


def _corner(rows: list[Candidate], key, name: str) -> dict | None:
    if not rows:
        return None
    row = max(rows, key=key)
    return _row_dict(row, name)


def _row_dict(row: Candidate, name: str) -> dict:
    return {
        "name": name,
        "label": row.label(),
        "spec": asdict(row.spec),
        "precision": row.precision,
        "batch": row.batch,
        "rows": row.rows,
        "params": row.params,
        "card_bytes": row.bytes,
        "card_gib": round(row.gib, 2),
        "tokens_per_second": row.tokens_per_second,
        "scaling_efficiency": round(row.efficiency, 3),
        "sync_fraction": round(row.sync_fraction, 3),
        "breakdown": row.breakdown,
    }


def answer(hw: Hardware, topo: Topology, precision: str = "fp32",
           min_seq: int = 1, min_params: int = 0,
           presets: str = "default") -> dict:
    """The full answer for one hardware and topology.

    The result holds the three corners, the balanced pick, and the caveats.
    """
    rows = search(hw, topo, precision, min_seq=min_seq,
                  min_params=min_params, presets=presets)
    if not rows:
        return {
            "hardware": asdict(hw), "topology": asdict(topo),
            "precision": precision, "corners": {}, "balanced": None,
            "reason": "no architecture fits the memory budget",
        }
    balanced_rows = [row for row in rows
                     if row.params >= max(min_params, 250_000_000)
                     and row.spec.seq >= max(min_seq, 4096)]
    balanced = max(balanced_rows, key=lambda row: row.tokens_per_second) \
        if balanced_rows else max(rows, key=lambda row: row.tokens_per_second)
    return {
        "hardware": asdict(hw),
        "topology": asdict(topo),
        "precision": precision,
        "budget_bytes": hw.budget_bytes(),
        "corners": {
            "biggest": _corner(rows, lambda row: (row.params, -row.spec.seq),
                               "biggest"),
            "longest": _corner(rows, lambda row: (row.spec.seq, row.params),
                               "longest"),
            "fastest": _corner(rows, lambda row: row.tokens_per_second,
                               "fastest"),
        },
        "balanced": _row_dict(balanced, "balanced"),
        "caveats": [
            "The memory model is exact. The throughput model is an estimate.",
            "The sync cost uses a ring all-reduce and the interconnect table.",
            "Data parallel keeps the full model on each device.",
            "The balanced pick needs 250M parameters and 4096 tokens.",
        ],
    }


#: The balanced fit for the two Kaggle T4 cards at fp32.
BEST_FIT_T4 = ModelSpec(layers=16, hidden=1024, heads=8, kv_heads=4, seq=4096)


def recommend(hw: Hardware, topo: Topology, precision: str = "fp32",
              min_seq: int = 4096, min_params: int = 250_000_000) -> Candidate:
    """The balanced pick for the hardware and topology."""
    rows = search(hw, topo, precision, min_seq=min_seq, min_params=min_params)
    if not rows:
        raise ValueError("no architecture fits the filters")
    return max(rows, key=lambda row: row.tokens_per_second)


def shape_flags(spec: ModelSpec) -> tuple[str, ...]:
    """The ``train_main`` model flags for ``spec``.

    Pass the ``track3`` preset first, so the value embeddings stay off and the
    optimizer and schedule come from the preset. These flags then override the
    baseline shape. ``batch`` and ``grad-accum`` stay out, because the launch
    plan owns them.
    """
    return ("--layers", str(spec.layers),
            "--heads", str(spec.heads),
            "--kv-heads", str(spec.kv_heads),
            "--hidden", str(spec.hidden),
            "--seq", str(spec.seq),
            "--window-pattern", spec.window)


def frontier(rows: list[Candidate]) -> list[Candidate]:
    """The Pareto set on (params, seq, tokens/s)."""
    ordered = sorted(rows, key=lambda row: (row.params, row.spec.seq),
                     reverse=True)
    out: list[Candidate] = []
    best_speed = -1.0
    for row in ordered:
        if row.tokens_per_second > best_speed:
            out.append(row)
            best_speed = row.tokens_per_second
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _print_table(rows: list[Candidate]) -> None:
    print(f"{'config':40} {'paramsM':>8} {'GiB':>6} {'kTok/s':>8} "
          f"{'eff':>5} {'sync':>5}")
    for row in rows:
        print(f"{row.label():40} {row.params/1e6:8.0f} {row.gib:6.2f} "
              f"{row.tokens_per_second/1e3:8.1f} {row.efficiency:5.2f} "
              f"{row.sync_fraction:5.2f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit nanochat architectures to hardware and topology.")
    parser.add_argument("--hardware", default="t4",
                       help="a device name, or 'custom' with the rate flags")
    parser.add_argument("--devices", type=int, default=1,
                       help="the device count (data parallel)")
    parser.add_argument("--interconnect", default="pcie3",
                       help="the link between the devices")
    parser.add_argument("--bandwidth-gbps", type=float, default=None,
                       help="override the interconnect bandwidth")
    parser.add_argument("--latency-us", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.0,
                       help="the fraction of the sync hidden by the backward")
    parser.add_argument("--precision", default="fp32",
                       choices=sorted(PRECISIONS))
    parser.add_argument("--min-seq", type=int, default=1)
    parser.add_argument("--min-params", type=float, default=0.0,
                       help="the minimum parameter count, or 250e6 for 250M")
    parser.add_argument("--presets", default="default",
                       choices=("quick", "default", "wide"))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-interconnects", action="store_true")
    # Custom hardware overrides.
    parser.add_argument("--memory-gib", type=float, default=None)
    parser.add_argument("--fp32-tflops", type=float, default=None)
    parser.add_argument("--fp16-tflops", type=float, default=None)
    parser.add_argument("--bf16-tflops", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list_devices:
        for name, hw in sorted(DEVICES.items()):
            print(f"{name:12} {hw.memory_bytes/GIB:5.0f} GiB  "
                  f"fp32 {hw.fp32_flops/1e12:6.1f} TFLOP/s  "
                  f"fp16 {hw.fp16_flops/1e12:6.1f}  "
                  f"bf16 {hw.bf16_flops/1e12:6.1f}")
        return 0
    if args.list_interconnects:
        for name, gbps in sorted(INTERCONNECTS.items()):
            print(f"{name:16} {gbps:6.1f} GB/s")
        return 0

    if args.hardware == "custom":
        if args.memory_gib is None or args.fp32_tflops is None:
            print("custom hardware needs --memory-gib and --fp32-tflops")
            return 2
        hw = Hardware(
            "custom", int(args.memory_gib * GIB), args.fp32_tflops * 1e12,
            (args.fp16_tflops or 0.0) * 1e12,
            (args.bf16_tflops or 0.0) * 1e12,
            tensor_cores=bool(args.fp16_tflops or args.bf16_tflops))
    else:
        hw = hardware(args.hardware)
    topo = Topology(world_size=args.devices, interconnect=args.interconnect,
                    bandwidth_gbps=args.bandwidth_gbps,
                    latency_us=args.latency_us, overlap=args.overlap)

    result = answer(hw, topo, args.precision, min_seq=args.min_seq,
                    min_params=args.min_params, presets=args.presets)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"hardware {hw.name} x{topo.world_size} over {topo.interconnect} "
          f"at {args.precision}")
    print(f"budget   {hw.budget_bytes()/GIB:.1f} GiB per device")
    print()
    for name in ("biggest", "longest", "fastest"):
        row = result["corners"].get(name)
        if row is None:
            continue
        print(f"{name:9} {row['label']:40} {row['params']/1e6:7.0f}M "
              f"{row['card_gib']:5.2f} GiB {row['tokens_per_second']/1e3:7.1f}k "
              f"tok/s eff {row['scaling_efficiency']:.2f}")
    if result["balanced"]:
        row = result["balanced"]
        print(f"{'balanced':9} {row['label']:40} {row['params']/1e6:7.0f}M "
              f"{row['card_gib']:5.2f} GiB {row['tokens_per_second']/1e3:7.1f}k "
              f"tok/s eff {row['scaling_efficiency']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
