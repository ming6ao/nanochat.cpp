"""The single-client multi-device training loop for the two-card notebook.

One :class:`MeshTrainPlan` holds the run. :func:`run` builds a device mesh, one
``MeshRuntime``, and a parquet token loader, then calls :func:`iterate`. The
runtime takes one global batch per step and reduces the gradients in process,
so the notebook names no rank.

:func:`launch` runs :func:`run` in a child process through
``tools/nanochat client``. The notebook cell returns, so the cell status stays
correct, and the parent polls ``metrics.jsonl`` and ``worker.log`` instead. See
``docs/notebook-workflow.md``.

The module imports ``nanochat_cpp`` inside the functions, not at module load.
The pure helpers then import without the shared library.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import nanochat_cpp as nc

__all__ = [
    "MeshLiveRun",
    "MeshRunResult",
    "MeshTrainPlan",
    "format_row",
    "global_batch",
    "iterate",
    "launch",
    "log_progress",
    "mfu",
    "read_metrics",
    "read_summary",
    "run",
]


@dataclass(frozen=True)
class MeshTrainPlan:
    """The inputs of one mesh run.

    ``per_rank_batch`` is the number of sequences one device consumes in one
    micro-batch. ``grad_accum`` is the micro-batch count per device per step.
    The runtime takes the global batch, so :func:`global_batch` multiplies the
    three factors. ``seq`` of zero selects ``config.seq_len``.

    ``tokenizer`` is the ``NCTOKEN1`` path, so the whole plan is serializable
    for the child process. ``flops_per_token`` and ``peak_flops`` set the model
    FLOPs utilization. A zero ``peak_flops`` keeps it at zero.
    """

    train_parquet: str
    tokenizer: str
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

    def summary_path(self) -> Path:
        """The run summary path."""
        return Path(self.out_dir) / "summary.json"

    def job(self) -> dict:
        """A JSON-serializable job for the child process."""
        config = self.config
        return {
            "train_parquet": self.train_parquet,
            "tokenizer_path": self.tokenizer,
            "out_dir": self.out_dir,
            "config": {
                "num_layers": int(config.num_layers),
                "num_heads": int(config.num_heads),
                "num_kv_heads": int(config.num_kv_heads),
                "hidden_dim": int(config.hidden_dim),
                "seq_len": int(config.seq_len),
                "vocab_size": int(config.vocab_size),
                "padded_vocab_size": int(config.padded_vocab_size),
                "rope_base": float(config.rope_base),
                "window_pattern": str(config.window_pattern),
                "value_embedding": bool(config.value_embedding),
            },
            "per_rank_batch": self.per_rank_batch,
            "grad_accum": self.grad_accum,
            "world_size": self.world_size,
            "steps": self.steps,
            "seq": self.seq,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "log_every": self.log_every,
            "save_every": self.save_every,
            "preset": self.preset,
            "flops_per_token": self.flops_per_token,
            "peak_flops": self.peak_flops,
        }

    @classmethod
    def from_job(cls, data: dict) -> "MeshTrainPlan":
        """Rebuild a plan from :meth:`job` output. The child process uses this."""
        import nanochat_cpp as nc

        return cls(
            train_parquet=data["train_parquet"],
            tokenizer=data["tokenizer_path"],
            out_dir=data["out_dir"],
            config=nc.Config(**data["config"]),
            per_rank_batch=int(data["per_rank_batch"]),
            grad_accum=int(data["grad_accum"]),
            world_size=int(data["world_size"]),
            steps=int(data["steps"]),
            seq=int(data.get("seq", 0)),
            seed=int(data.get("seed", 42)),
            data_seed=int(data.get("data_seed", 42)),
            log_every=int(data.get("log_every", 1)),
            save_every=int(data.get("save_every", 0)),
            preset=str(data.get("preset", "track3")),
            flops_per_token=float(data.get("flops_per_token", 0.0)),
            peak_flops=float(data.get("peak_flops", 0.0)),
        )


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

    def to_dict(self) -> dict:
        """The serializable summary that :func:`run` writes."""
        return {
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "steps": len(self.records),
            "mean_loss": self.mean_loss,
            "steady_mfu": self.steady_mfu,
            "replica_digests": list(self.replica_digests),
        }


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


def format_row(row: dict) -> str:
    """One compact log line for a step record.

    The line reports the loss, the gradient norm, the wall step time, the
    compute time alone, the throughput, and the model FLOPs utilization. The
    difference between the two times is the data cost.
    """
    return (f"loss {float(row['loss']):.4f}  "
            f"grad_norm {float(row['grad_norm']):.4g}  "
            f"step {float(row['step_seconds']):.2f}s  "
            f"(compute {float(row.get('compute_seconds', 0.0)):.2f}s)  "
            f"{float(row['tokens_per_sec']):,.0f} tok/s  "
            f"mfu {float(row['mfu']) * 100:.2f}%")


def log_progress(row: dict, total: int) -> None:
    """A `progress` callback that prints one log line per step."""
    print(f"[{int(row['step']):5d}/{int(total)}] {format_row(row)}",
          flush=True)


def read_metrics(path) -> list[dict]:
    """Read the per-step JSON lines. A missing file returns an empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def read_summary(out_dir) -> dict:
    """Read the run summary. A missing or bad file returns an empty dict."""
    path = Path(out_dir) / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


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
                "compute_seconds": float(metrics.step_seconds),
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
    metrics record per step, a canonical checkpoint, and a summary. An
    interrupt saves the checkpoint before it propagates.
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
    tokenizer = nc.Tokenizer.load(plan.tokenizer)
    data = nc.TokenData(plan.train_parquet, tokenizer,
                        seq_len=plan.sequence_length,
                        batch=plan.global_sequences, seed=plan.data_seed)
    checkpoint = plan.checkpoint_path()
    with rt.MeshRuntime(device_mesh, plan.config, optimizer,
                        batch=plan.global_sequences,
                        grad_accum=plan.grad_accum,
                        seq=plan.sequence_length, seed=plan.seed) as runtime:
        try:
            records = iterate(runtime, data, plan, progress=progress)
        except BaseException:
            runtime.save(str(checkpoint))
            raise
        digests = tuple(f"{runtime.replica_digest(rank):016x}"
                        for rank in range(runtime.world_size))
        runtime.save(str(checkpoint))
    result = MeshRunResult(plan=plan, records=records, checkpoint=checkpoint,
                           replica_digests=digests)
    plan.summary_path().write_text(
        json.dumps(result.to_dict(), indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


class MeshLiveRun:
    """A mesh worker that runs in a child process.

    The parent starts the child with :func:`launch` and polls it. The child
    writes ``metrics.jsonl`` and ``worker.log`` in the plan directory, so the
    parent never reads a pipe and the notebook cell never blocks.
    """

    def __init__(self, plan: MeshTrainPlan, process: subprocess.Popen, *,
                 log_path, timeout: float | None = None) -> None:
        self.plan = plan
        self.process = process
        self.log_path = Path(log_path)
        self._timeout = timeout

    def running(self) -> bool:
        """True while the child process is alive."""
        return self.process.poll() is None

    def tail(self, lines: int = 15) -> str:
        """The last ``lines`` lines of the child log. Missing returns ""."""
        if not self.log_path.is_file():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def rows(self) -> list[dict]:
        """The per-step metrics written so far."""
        return read_metrics(self.plan.metrics_path())

    def summary(self) -> dict:
        """The final summary, or an empty dict before the child finishes."""
        return read_summary(self.plan.out_dir)

    def wait(self, timeout: float | None = None) -> int:
        """Block until the child exits and return its return code."""
        timeout = self._timeout if timeout is None else timeout
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.terminate()
            raise

    def terminate(self, timeout: float = 30.0) -> None:
        """Stop the child group. Send SIGINT first, so the worker can save."""
        if self.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            self.process.terminate()
            return
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.process.kill()


def launch(plan: MeshTrainPlan, root: str | Path = ".", *,
           entry: str = "tools/nanochat", profile: str = "train",
           device: str | None = None, timeout: float | None = None,
           extra_env: dict | None = None) -> MeshLiveRun:
    """Start :func:`run` in a child process and return a pollable handle.

    The child runs through ``tools/nanochat client``, so one broker lock and
    one sandbox scope cover the whole mesh. The child stdout and stderr go to
    ``worker.log``, so the parent never blocks on a full pipe.
    """
    root = Path(root)
    out_dir = Path(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(plan.job(), indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")

    worker = Path(__file__).with_name("mesh_worker.py")
    env = dict(os.environ)
    env.update(extra_env or {})
    paths = [str(root / "python"), str(root / "notebooks")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    command = [str(root / entry), "client", "--profile", profile]
    if device:
        command += ["--device", str(device)]
    command += ["--", sys.executable, str(worker), str(job_path)]

    log_path = out_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(root), env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return MeshLiveRun(plan, process, log_path=log_path, timeout=timeout)
