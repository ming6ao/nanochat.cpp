"""The notebook launcher for the C++ mesh training driver.

One :class:`MeshTrainPlan` holds the run. It is a thin wrapper over the C++
driver config and the data paths. :func:`run` builds the driver config and
calls one C++ function, ``nanochat_mesh_train_run``. The driver owns the step
loop, the metric record, the summary, the checkpoint interval, and the resume
(docs/client-runtime-design.md section 10.1 and docs/training-seam.md
section 6.6).

:func:`launch` runs :func:`run` in a child process through
``tools/nanochat client``. The notebook cell returns, so the cell status stays
correct, and the parent polls ``metrics.jsonl`` and ``worker.log`` instead. See
``docs/client-runtime-design.md``.

The module imports ``nanochat_cpp`` inside the functions, not at module load.
The pure helpers then import without the shared library.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import nanochat_cpp as nc

__all__ = [
    "MeshLiveRun",
    "MeshRunResult",
    "MeshTrainPlan",
    "global_batch",
    "launch",
    "read_metrics",
    "read_summary",
    "run",
]


@dataclass(frozen=True)
class MeshTrainPlan:
    """The inputs of one mesh run.

    ``per_rank_batch`` is the number of sequences one device consumes in one
    micro-batch. ``grad_accum`` is the micro-batch count per device per step.
    The driver takes the global batch, so :func:`global_batch` multiplies the
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
    resume: bool = False
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
        return Path(self.out_dir) / "checkpoint.ckpt"

    def metrics_path(self) -> Path:
        """The per-step metrics path."""
        return Path(self.out_dir) / "metrics.jsonl"

    def summary_path(self) -> Path:
        """The run summary path."""
        return Path(self.out_dir) / "summary.json"

    def train_config(self):
        """The C++ driver config for this plan.

        The optimizer and the schedule come from ``preset``, so the plan does
        not carry optimizer fields.
        """
        from nanochat_cpp import runtime as rt

        return rt.MeshTrainConfig(
            self.config,
            preset=self.preset,
            device_count=self.world_size,
            tokenizer=self.tokenizer,
            train_parquet=self.train_parquet,
            batch=self.global_sequences,
            grad_accum=self.grad_accum,
            steps=self.steps,
            seq=self.seq,
            seed=self.seed,
            data_seed=self.data_seed,
            log_every=self.log_every,
            save_every=self.save_every,
            resume=self.resume,
            out_dir=self.out_dir,
            flops_per_token=self.flops_per_token,
            peak_flops=self.peak_flops,
        )

    def job_json(self) -> str:
        """The JSON job the child process reads."""
        config = self.train_config()
        try:
            return config.job_json()
        finally:
            config.close()


@dataclass
class MeshRunResult:
    """The outcome of one mesh run."""

    plan: MeshTrainPlan
    records: list[dict] = field(default_factory=list)
    checkpoint: Path | None = None
    replica_digests: tuple[str, ...] = ()
    mean_loss: float = 0.0
    steady_mfu: float = 0.0

    def to_dict(self) -> dict:
        """The serializable summary of the run."""
        return {
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "steps": len(self.records),
            "mean_loss": self.mean_loss,
            "steady_mfu": self.steady_mfu,
            "replica_digests": list(self.replica_digests),
        }


def global_batch(per_rank_batch: int, grad_accum: int, world_size: int) -> int:
    """The global sequence count of one step.

    The driver takes the global batch, so this function rebuilds the total:
    ``per_rank_batch * grad_accum * world_size``.
    """
    per_rank = int(per_rank_batch)
    accum = int(grad_accum)
    world = int(world_size)
    if per_rank < 1 or accum < 1 or world < 1:
        raise ValueError("per_rank_batch, grad_accum, and world_size must be "
                         "positive")
    return per_rank * accum * world


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


def run(plan: MeshTrainPlan) -> MeshRunResult:
    """Build the C++ driver config and run the whole job.

    The driver writes ``metrics.jsonl``, ``summary.json``, and the canonical
    checkpoint in the plan output directory. With ``plan.resume`` set and a
    checkpoint on disk, the driver loads the parameters, the optimizer state,
    and the step, then continues from the next step.
    """
    config = plan.train_config()
    try:
        result = config.run()
    finally:
        config.close()
    summary = read_summary(plan.out_dir)
    checkpoint = Path(result.checkpoint) if result.checkpoint else None
    return MeshRunResult(
        plan=plan,
        records=read_metrics(plan.metrics_path()),
        checkpoint=checkpoint,
        replica_digests=tuple(summary.get("replica_digests", ())),
        mean_loss=float(result.mean_loss),
        steady_mfu=float(result.steady_mfu),
    )


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
        """Stop the child group. Send SIGINT first, so the worker can save.

        Further SIGINT delivery is ignored until the stop finishes, so a second
        cancel click does not break the cleanup.
        """
        if self.process.poll() is not None:
            return
        previous = None
        try:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            previous = None
        try:
            self._stop(timeout)
        finally:
            if previous is not None:
                try:
                    signal.signal(signal.SIGINT, previous)
                except (ValueError, OSError):
                    pass

    def _stop(self, timeout: float) -> None:
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
    job_path.write_text(plan.job_json() + "\n", encoding="utf-8")

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
