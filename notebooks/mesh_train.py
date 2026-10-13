"""The single-client multi-device training loop for the two-card notebook.

One :class:`MeshTrainPlan` holds the run. :func:`run` builds a device mesh, one
``MeshRuntime``, and a parquet token loader, then calls :func:`iterate`. The
runtime takes one global batch per step and reduces the gradients in process,
so the notebook names no rank.

The module imports ``nanochat_cpp`` inside :func:`run`, not at module load. The
pure helpers then import without the shared library. See
``docs/client-runtime-design.md`` and ``docs/notebook-workflow.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import nanochat_cpp as nc

__all__ = [
    "MeshRunResult",
    "MeshTrainPlan",
    "global_batch",
    "iterate",
    "mfu",
    "run",
]


@dataclass(frozen=True)
class MeshTrainPlan:
    """The inputs of one mesh run.

    ``per_rank_batch`` is the number of sequences one device consumes in one
    micro-batch. ``grad_accum`` is the micro-batch count per device per step.
    The runtime takes the global batch, so :func:`global_batch` multiplies the
    three factors. ``seq`` of zero selects ``config.seq_len``.

    ``flops_per_token`` and ``peak_flops`` set the model FLOPs utilization. A
    zero ``peak_flops`` keeps it at zero.
    """

    train_parquet: str
    tokenizer: "nc.Tokenizer"
    out_dir: str
    config: "nc.Config"
    per_rank_batch: int
    grad_accum: int
    world_size: int
    steps: int
    seq: int = 0
    seed: int = 42
    data_seed: int = 42
    log_every: int = 1
    save_every: int = 0
    preset: str = "track3"
    flops_per_token: float = 0.0
    peak_flops: float = 0.0

    @property
    def sequence_length(self) -> int:
        """The sequence length of one step."""
        return int(self.seq) if self.seq > 0 else int(self.config.seq_len)

    @property
    def global_sequences(self) -> int:
        """The global number of sequences in one step."""
        return global_batch(self.per_rank_batch, self.grad_accum,
                            self.world_size)

    @property
    def tokens_per_step(self) -> int:
        """The global number of tokens in one step."""
        return self.global_sequences * self.sequence_length

    def checkpoint_path(self) -> Path:
        """The canonical checkpoint path."""
        return Path(self.out_dir) / "base.ncmdl"

    def metrics_path(self) -> Path:
        """The per-step metrics path."""
        return Path(self.out_dir) / "metrics.jsonl"


@dataclass
class MeshRunResult:
    """The outcome of one mesh run."""

    plan: MeshTrainPlan
    records: list[dict] = field(default_factory=list)
    checkpoint: Path | None = None
    replica_digests: tuple[str, ...] = ()

    @property
    def mean_loss(self) -> float:
        """The mean loss over every logged step, or zero when none."""
        return mean_loss(self.records)

    @property
    def steady_mfu(self) -> float:
        """The mean model FLOPs utilization over the second half of the run."""
        return steady_mfu(self.records)


def global_batch(per_rank_batch: int, grad_accum: int, world_size: int) -> int:
    """The global sequence count of one step.

    The multi-process path used a per-rank batch. The mesh runtime takes the
    global batch instead, so this function rebuilds the same total:
    ``per_rank_batch * grad_accum * world_size``.
    """
    per_rank = int(per_rank_batch)
    accum = int(grad_accum)
    world = int(world_size)
    if per_rank < 1 or accum < 1 or world < 1:
        raise ValueError("per_rank_batch, grad_accum, and world_size must be "
                         "positive")
    return per_rank * accum * world


def mfu(tokens_per_sec: float, flops_per_token: float,
        peak_flops: float) -> float:
    """The model FLOPs utilization, clamped to ``[0, 1]``.

    A nonpositive ``peak_flops`` returns zero, so an unknown device reports no
    utilization instead of a division by zero.
    """
    peak = float(peak_flops)
    if peak <= 0.0:
        return 0.0
    value = float(tokens_per_sec) * float(flops_per_token) / peak
    return min(max(value, 0.0), 1.0)


def mean_loss(records) -> float:
    """The mean of the ``loss`` field over ``records``, or zero when none."""
    rows = list(records)
    if not rows:
        return 0.0
    return sum(float(row["loss"]) for row in rows) / len(rows)


def steady_mfu(records) -> float:
    """The mean ``mfu`` over the second half of ``records``, or zero."""
    rows = list(records)
    if not rows:
        return 0.0
    tail = rows[len(rows) // 2:]
    return sum(float(row.get("mfu", 0.0)) for row in tail) / len(tail)


def _preset_config(plan: MeshTrainPlan):
    """The optimizer config: the named preset with the run length."""
    import nanochat_cpp as nc

    config = nc.preset_optimizer_config(plan.preset)
    config.num_iterations = int(plan.steps)
    return config


def iterate(runtime, data, plan: MeshTrainPlan, *,
            progress: "Callable[[dict, int], None] | None" = None,
            clock: Callable[[], float] = time.perf_counter) -> list[dict]:
    """Run ``plan.steps`` steps and write one record per step.

    ``runtime`` is any object with ``train_step``, ``metrics``, and ``save``.
    ``data`` is any object with ``next`` and ``reset``. The function is pure
    except for the runtime, the clock, and the metrics file, so a CPU test can
    drive a fake runtime.
    """
    tokens_per_step = plan.tokens_per_step
    plan.metrics_path().parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    start = clock()
    previous = start
    with plan.metrics_path().open("w", encoding="utf-8") as sink:
        for step in range(1, int(plan.steps) + 1):
            batch = data.next()
            if batch is None and hasattr(data, "reset"):
                data.reset()
                batch = data.next()
            if batch is None:
                raise RuntimeError("the data loader returned no batch")
            tokens, targets = batch
            loss = runtime.train_step(tokens, targets)
            now = clock()
            step_seconds = max(now - previous, 1e-9)
            previous = now
            metrics = runtime.metrics()
            tokens_per_sec = tokens_per_step / step_seconds
            row = {
                "step": step,
                "loss": float(loss),
                "grad_norm": float(metrics.grad_norm),
                "tokens_per_sec": tokens_per_sec,
                "mfu": mfu(tokens_per_sec, plan.flops_per_token, plan.peak_flops),
                "step_seconds": step_seconds,
                "seconds": now - start,
                "device_loss": [float(value) for value in metrics.loss],
            }
            records.append(row)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            if plan.save_every and step % int(plan.save_every) == 0:
                runtime.save(str(plan.checkpoint_path()))
            if progress is not None and (
                    step == 1 or step == int(plan.steps) or
                    step % max(1, int(plan.log_every)) == 0):
                progress(row, int(plan.steps))
    return records


def run(plan: MeshTrainPlan, *,
        progress: "Callable[[dict, int], None] | None" = None) -> MeshRunResult:
    """Build the mesh runtime and the loader, then run the loop.

    The mesh uses the first ``world_size`` local devices. The loader reads the
    train parquet and yields one global batch per ``next()``. The run writes a
    metrics record per step and a canonical checkpoint at the end.
    """
    import nanochat_cpp as nc
    from nanochat_cpp import mesh, runtime as rt

    devices = mesh.devices()
    if not devices:
        raise RuntimeError("no local device is available")
    if plan.world_size < 1 or plan.world_size > len(devices):
        raise RuntimeError(
            f"the mesh needs {plan.world_size} devices, but the host has "
            f"{len(devices)}")
    device_mesh = mesh.Mesh(tuple(devices[:plan.world_size]))

    optimizer = _preset_config(plan)
    data = nc.TokenData(plan.train_parquet, plan.tokenizer,
                        seq_len=plan.sequence_length,
                        batch=plan.global_sequences, seed=plan.data_seed)
    checkpoint = plan.checkpoint_path()
    with rt.MeshRuntime(device_mesh, plan.config, optimizer,
                        batch=plan.global_sequences,
                        grad_accum=plan.grad_accum,
                        seq=plan.sequence_length, seed=plan.seed) as runtime:
        records = iterate(runtime, data, plan, progress=progress)
        digests = tuple(f"{runtime.replica_digest(rank):016x}"
                        for rank in range(runtime.world_size))
        runtime.save(str(checkpoint))
    return MeshRunResult(plan=plan, records=records, checkpoint=checkpoint,
                         replica_digests=digests)
