"""Two-card data-parallel launch for the Kaggle notebook.

One :class:`ParallelPlan` holds the run. :func:`command_for` builds the
``train_main`` argument list for one rank. :func:`launch` starts one process per
card, waits, and returns one :class:`RankResult` per rank.

The gradient sync is the host reference in ``src/distributed.cc``. It sends
each gradient over loopback TCP, so it needs no NCCL. The CUDA backend stages
each gradient through host memory. See ``docs/distributed-design.md``.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ParallelError",
    "ParallelPlan",
    "ParallelResult",
    "RankResult",
    "assert_complete",
    "command_for",
    "launch",
    "loss_rows",
    "parse_log",
]

#: The one log line that ``Logger::Log`` writes (``src/train.cc``). The phase
#: fields after ``mfu`` are optional, so an older log still parses.
_LOG_LINE = re.compile(
    r"step\s+(?P<step>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>\S+)\s*\|\s*"
    r"lr\s+(?P<lr>\S+)\s*\|\s*"
    r"grad_norm\s+(?P<grad_norm>\S+)\s*\|\s*"
    r"tok/s\s+(?P<tokens_per_sec>\S+)\s*\|\s*"
    r"mfu\s+(?P<mfu>\S+)%"
    r"(?:\s*\|\s*step_ms\s+(?P<step_ms>\S+)\s*\|\s*"
    r"data_ms\s+(?P<data_ms>\S+)\s*\|\s*"
    r"fwd_ms\s+(?P<forward_ms>\S+)\s*\|\s*"
    r"bwd_ms\s+(?P<backward_ms>\S+)\s*\|\s*"
    r"sync_ms\s+(?P<sync_ms>\S+)\s*\|\s*"
    r"opt_ms\s+(?P<optim_ms>\S+)\s*\|\s*"
    r"eval_ms\s+(?P<eval_ms>\S+))?")

#: The extra ``train_main`` flags a small smoke run needs before ``--preset``
#: is absent. The vocab must match the tokenizer.
SMOKE_FLAGS = (
    "--layers", "4", "--heads", "4", "--kv-heads", "4",
    "--hidden", "256", "--seq", "512",
    "--vocab", "32768", "--padded-vocab", "32768",
    "--window-pattern", "L",
)


class ParallelError(RuntimeError):
    """A rank failed, or the two ranks disagree."""


@dataclass(frozen=True)
class ParallelPlan:
    """The inputs of one two-card run.

    ``batch`` is the number of sequences in one micro-batch, so one forward
    pass sees ``batch`` sequences. One rank consumes ``batch * grad_accum``
    sequences per step. The global batch is
    ``batch * grad_accum * world_size * seq_len`` tokens. ``flags`` carries
    extra ``train_main`` flags and comes last, so it can override a preset
    field.
    """

    train_parquet: str
    val_parquet: str
    tokenizer: str
    out_dir: str
    batch: int
    grad_accum: int = 1
    num_iterations: int = 50
    entry: str = "tools/nanochat"
    binary: str = "bazel-bin/src/train_main"
    preset: str | None = "track3"
    flags: tuple[str, ...] = ()
    seed: int = 42
    log_every: int = 10
    eval_every: int = 0
    eval_steps: int = 8
    save_every: int = 0
    world_size: int = 2
    master: str = "127.0.0.1"
    port: int = 29500
    device_name: str = "t4"

    def log_path(self, rank: int) -> Path:
        """The structured log file of ``rank``."""
        return Path(self.out_dir) / f"rank{rank}.log"

    def stdio_path(self, rank: int) -> Path:
        """The captured stdout and stderr file of ``rank``."""
        return Path(self.out_dir) / f"rank{rank}.stdio.log"

    def checkpoint_path(self, rank: int) -> Path:
        """The final checkpoint path of ``rank``.

        Each rank writes its own file. The two checkpoints are equal, because
        the reduced update is equal. Rank 0 is the canonical one.
        """
        return Path(self.out_dir) / f"rank{rank}.nchkpt01"


@dataclass
class RankResult:
    """The outcome of one rank."""

    rank: int
    returncode: int
    log_path: Path
    stdio_path: Path
    metrics: list[dict] = field(default_factory=list)


@dataclass
class ParallelResult:
    """The outcome of one two-card run."""

    out_dir: Path
    ranks: tuple[RankResult, ...] = ()

    @property
    def returncodes(self) -> tuple[int, ...]:
        """The exit code of each rank, in rank order."""
        return tuple(rank.returncode for rank in self.ranks)

    @property
    def ok(self) -> bool:
        """True when every rank exited with status 0."""
        return all(code == 0 for code in self.returncodes)


def command_for(rank: int, plan: ParallelPlan) -> list[str]:
    """The full argument list that runs ``rank`` of ``plan``.

    The list starts with the ``tools/nanochat`` entry point, so the executable
    sees the sandbox variable. Each rank gets its own log and checkpoint path,
    so the two processes never write one file.
    """
    if plan.world_size < 1:
        raise ValueError(f"world_size must be positive: {plan.world_size}")
    if not 0 <= rank < plan.world_size:
        raise ValueError(f"rank {rank} is outside [0, {plan.world_size})")
    if plan.batch < 1 or plan.grad_accum < 1:
        raise ValueError("batch and grad_accum must be positive")

    argv = [plan.entry, "train", "--", plan.binary,
            "--train-parquet", str(plan.train_parquet),
            "--tokenizer", str(plan.tokenizer)]
    if plan.val_parquet:
        argv += ["--val-parquet", str(plan.val_parquet)]
    if plan.preset:
        argv += ["--preset", plan.preset]
    argv += ["--batch", str(plan.batch),
             "--grad-accum", str(plan.grad_accum),
             "--num-iterations", str(plan.num_iterations),
             "--log-every", str(plan.log_every),
             "--eval-every", str(plan.eval_every),
             "--eval-steps", str(plan.eval_steps),
             "--save-every", str(plan.save_every),
             "--seed", str(plan.seed),
             "--device", plan.device_name,
             "--rank", str(rank),
             "--world-size", str(plan.world_size),
             "--master", plan.master,
             "--port", str(plan.port),
             "--log", str(plan.log_path(rank)),
             "--checkpoint", str(plan.checkpoint_path(rank))]
    argv += list(plan.flags)
    return argv


def parse_log(path: str | Path) -> list[dict]:
    """Read the structured log file into one dict per logged step.

    The parser keeps only the ``step ... | loss ...`` lines. An information
    line is ignored. A missing file returns an empty list.
    """
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOG_LINE.search(line)
        if match is None:
            continue
        try:
            row = {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "lr": float(match.group("lr")),
                "grad_norm": float(match.group("grad_norm")),
                "tokens_per_sec": float(match.group("tokens_per_sec")),
                "mfu": float(match.group("mfu")) / 100.0,
            }
            # The phase split is optional: an older log line has none.
            for name in ("step_ms", "data_ms", "forward_ms", "backward_ms",
                         "sync_ms", "optim_ms", "eval_ms"):
                value = match.group(name)
                if value is not None:
                    row[name] = float(value)
            rows.append(row)
        except ValueError:
            # A field that is not a number, for example a corrupted line.
            continue
    return rows


def _wait(processes: list[subprocess.Popen], timeout: float | None,
          poll_seconds: float) -> None:
    """Wait for every process, or kill them after ``timeout`` seconds."""
    start = time.monotonic()
    while True:
        if all(process.poll() is not None for process in processes):
            return
        if timeout is not None and time.monotonic() - start > timeout:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.wait()
            return
        time.sleep(poll_seconds)


def launch(plan: ParallelPlan, root: str | Path = ".",
           extra_env: dict | None = None, poll_seconds: float = 1.0,
           timeout: float | None = None) -> ParallelResult:
    """Start one process per card, wait, and collect the outcome.

    Rank 0 binds the port. Rank 1 connects. Start rank 0 first, so the bind
    wins the race; the client retries for about five seconds in any case.
    ``extra_env`` overrides the child environment. ``CUDA_VISIBLE_DEVICES`` is
    set to the rank, so each process sees one card as device 0.
    """
    out_dir = Path(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processes: list[subprocess.Popen] = []
    streams = []
    for rank in range(plan.world_size):
        env = os.environ.copy()
        env.update(extra_env or {})
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        stream = open(plan.stdio_path(rank), "w", encoding="utf-8")
        streams.append(stream)
        processes.append(subprocess.Popen(
            command_for(rank, plan), cwd=str(root), env=env,
            stdout=stream, stderr=subprocess.STDOUT))

    try:
        _wait(processes, timeout, poll_seconds)
    finally:
        for stream in streams:
            stream.close()

    ranks = []
    for rank, process in enumerate(processes):
        ranks.append(RankResult(
            rank=rank,
            returncode=process.returncode if process.returncode is not None
            else -1,
            log_path=plan.log_path(rank),
            stdio_path=plan.stdio_path(rank),
            metrics=parse_log(plan.log_path(rank))))
    return ParallelResult(out_dir=out_dir, ranks=tuple(ranks))


def assert_complete(result: ParallelResult) -> None:
    """Raise :class:`ParallelError` when a rank failed or logged nothing.

    The message carries the last lines of the failing rank, so the notebook
    cell shows the cause without a second lookup.
    """
    if result.ok:
        return
    details = []
    for rank in result.ranks:
        if rank.returncode == 0:
            continue
        tail = ""
        if rank.stdio_path.is_file():
            lines = rank.stdio_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-15:])
        details.append(f"rank {rank.rank} exited {rank.returncode}:\n{tail}")
    raise ParallelError("\n".join(details))


def loss_rows(result: ParallelResult) -> list[dict]:
    """One row per step: ``{"step": s, "rank0": l0, "rank1": l1}``.

    The row count is the shortest rank log, so a partial log does not invent a
    value. The ranks must agree on the loss; a difference points to a broken
    all-reduce.
    """
    by_rank = {rank.rank: {row["step"]: row["loss"] for row in rank.metrics}
               for rank in result.ranks}
    if not by_rank:
        return []
    steps = sorted(set.intersection(*(set(values) for values in by_rank.values())))
    return [{"step": step, **{f"rank{rank}": values[step]
                              for rank, values in by_rank.items()}}
            for step in steps]
