"""Two-card data-parallel launch for the Kaggle notebook.

One :class:`DistributedPlan` holds the run. :func:`command_for` builds the
``train_main`` argument list for one rank. :func:`launch` calls
``tools/nanochat dist``, which starts one process per card under one broker
lock, waits, and returns one :class:`RankResult` per rank.

The gradient sync is the native collective
(``backends/cuda/collective/native_sync.cu``, selected with
``--config=native``). See ``docs/distributed-native-plan.md`` and
``docs/distributed-launch.md``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import toolchain

__all__ = [
    "DistributedError",
    "DistributedPlan",
    "DistributedResult",
    "LiveRun",
    "RankResult",
    "assert_complete",
    "command_for",
    "launch",
    "launch_live",
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


class DistributedError(RuntimeError):
    """A rank failed, or the two ranks disagree."""


@dataclass(frozen=True)
class DistributedPlan:
    """The inputs of one two-card run.

    ``batch`` is the number of sequences in one micro-batch. One rank consumes
    ``batch * grad_accum`` sequences per step. The global batch is
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
    device: str | None = None
    device_name: str = "t4"
    profile: str = "train"

    def log_path(self, rank: int) -> Path:
        """The structured log file of ``rank``."""
        return Path(self.out_dir) / f"rank{rank}.log"

    def stdio_path(self, rank: int) -> Path:
        """The captured stdout and stderr file of ``rank``."""
        return Path(self.out_dir) / f"rank{rank}.stdio.log"

    def checkpoint_path(self) -> Path:
        """The canonical checkpoint path.

        Rank 0 writes one file. The other ranks write nothing, because a
        data-parallel rank holds a full replica of the model
        (docs/precision.md section 5).
        """
        return Path(self.out_dir) / "checkpoint.ckpt"


@dataclass
class RankResult:
    """The outcome of one rank."""

    rank: int
    returncode: int
    log_path: Path
    stdio_path: Path
    metrics: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class DistributedResult:
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


def command_for(rank: int, plan: DistributedPlan) -> list[str]:
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
             "--checkpoint", str(plan.checkpoint_path())]
    argv += list(plan.flags)
    return argv


def _shared_argv(plan: DistributedPlan) -> list[str]:
    """The ``train_main`` argument list, with ``{rank}`` in the per-rank paths.

    ``tools/nanochat dist`` expands ``{rank}`` per process, so one command
    serves every rank. The rank values themselves travel in the ``NANOCHAT_*``
    environment.
    """
    argv = [plan.binary,
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
             "--log", str(Path(plan.out_dir) / "rank{rank}.log"),
             "--checkpoint", str(plan.checkpoint_path())]
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
            for name in ("step_ms", "data_ms", "forward_ms", "backward_ms",
                         "sync_ms", "optim_ms", "eval_ms"):
                value = match.group(name)
                if value is not None:
                    row[name] = float(value)
            rows.append(row)
        except ValueError:
            continue
    return rows


def launch(plan: DistributedPlan, root: str | Path = ".",
           extra_env: dict | None = None,
           timeout: float | None = None) -> DistributedResult:
    """Start one process per card through ``tools/nanochat dist``.

    The shell launcher holds one broker lock for the group, sets the
    ``NANOCHAT_*`` rank variables, and expands ``{rank}`` in the command. It
    returns the parsed JSON summary. ``timeout`` bounds the whole group.
    """
    out_dir = Path(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = plan.entry
    env = dict(os.environ)
    env.update(extra_env or {})
    tool = toolchain.Toolchain(entry=entry, repo_root=root, env=env,
                               timeout=timeout)
    result = tool.dist(_shared_argv(plan), nproc_per_node=plan.world_size,
                       device=plan.device, master=plan.master, port=plan.port,
                       profile=plan.profile, log_dir=out_dir, json=True)
    ranks = _parse_summary(result.stdout, plan)
    return DistributedResult(out_dir=out_dir, ranks=tuple(ranks))


def launch_live(plan: DistributedPlan, root: str | Path = ".",
                extra_env: dict | None = None,
                timeout: float | None = None) -> "LiveRun":
    """Start one process per card and return a pollable :class:`LiveRun`.

    This is :func:`launch` without the block. Call :meth:`LiveRun.wait` to
    collect the final :class:`DistributedResult`; the parsed records are
    identical. The launcher's JSON summary stays on the pipe until then.
    """
    out_dir = Path(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(extra_env or {})
    tool = toolchain.Toolchain(entry=plan.entry, repo_root=root, env=env,
                               timeout=timeout)
    argv = tool.dist_argv(
        _shared_argv(plan), nproc_per_node=plan.world_size,
        device=plan.device, master=plan.master, port=plan.port,
        profile=plan.profile, log_dir=out_dir, json=True)
    process = subprocess.Popen(
        [str(tool.entry), *argv], cwd=str(tool.repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    return LiveRun(plan=plan, out_dir=out_dir, process=process,
                   timeout=timeout)


class LiveRun:
    """A running two-card launch that the caller can poll.

    ``launch_live`` returns this handle as soon as the launcher starts. The
    caller reads the per-rank structured logs while the run continues. The
    training logger flushes each record, so a live tail shows every completed
    step. Call :meth:`wait` to collect the final result.
    """

    def __init__(self, plan: DistributedPlan, out_dir: Path,
                 process: subprocess.Popen, timeout: float | None = None) -> None:
        self.plan = plan
        self.out_dir = Path(out_dir)
        self.process = process
        self._timeout = timeout

    def running(self) -> bool:
        """True while at least one launcher process is alive."""
        return self.process.poll() is None

    def tail(self, rank: int = 0, lines: int = 15) -> str:
        """The last ``lines`` lines of one rank structured log.

        The logger flushes each record, so the tail shows every completed step
        and every information line. A missing file returns an empty string.
        """
        path = self.plan.log_path(rank)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def rows(self, field: str = "loss") -> list[dict]:
        """The live merged metric rows for ``field``, one column per rank."""
        by_rank = {
            rank: {row["step"]: row.get(field)
                   for row in parse_log(self.plan.log_path(rank))}
            for rank in range(self.plan.world_size)}
        return _merged_rows(by_rank)

    def wait(self, timeout: float | None = None) -> DistributedResult:
        """Block until the group exits and return the final result.

        ``timeout`` overrides the value from :func:`launch_live`. On timeout
        the process group is terminated and
        :class:`subprocess.TimeoutExpired` propagates.
        """
        if timeout is None:
            timeout = self._timeout
        try:
            stdout, _ = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.terminate()
            raise
        return DistributedResult(
            out_dir=self.out_dir,
            ranks=tuple(_parse_summary(stdout or "", self.plan)))

    def terminate(self) -> None:
        """Stop the launcher and every rank process under it."""
        if self.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), 15)
        except (ProcessLookupError, PermissionError):
            self.process.terminate()


def _parse_summary(stdout: str, plan: DistributedPlan) -> list[RankResult]:
    """Turn the launcher's JSON summary into one :class:`RankResult` per rank.

    A launcher that printed no JSON (an older shell tool) falls back to the
    known per-rank paths, so the notebook still shows the loss curves.
    """
    payload = None
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except ValueError:
                payload = None
            if payload is not None:
                break
    by_rank: dict[int, dict] = {}
    if isinstance(payload, dict):
        for entry in payload.get("ranks", []):
            if isinstance(entry, dict) and "rank" in entry:
                by_rank[int(entry["rank"])] = entry
    ranks = []
    for rank in range(plan.world_size):
        entry = by_rank.get(rank, {})
        stdio = Path(entry.get("stdio", plan.stdio_path(rank)))
        ranks.append(RankResult(
            rank=rank,
            returncode=int(entry.get("returncode", 0)),
            log_path=plan.log_path(rank),
            stdio_path=stdio,
            metrics=parse_log(plan.log_path(rank)),
            seconds=float(entry.get("seconds", 0.0) or 0.0)))
    return ranks


def assert_complete(result: DistributedResult) -> None:
    """Raise :class:`DistributedError` when a rank failed or logged nothing.

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
    raise DistributedError("\n".join(details))


def _merged_rows(by_rank: dict[int, dict[int, float]]) -> list[dict]:
    """One row per step, one column per rank, from ``{rank: {step: value}}``.

    The row count is the shortest rank log, so a partial log does not invent a
    value. The ranks must agree on a reduced quantity; a difference points to a
    broken all-reduce.
    """
    if not by_rank:
        return []
    steps = sorted(set.intersection(
        *(set(values) for values in by_rank.values())))
    return [{"step": step, **{f"rank{rank}": values[step]
                              for rank, values in by_rank.items()}}
            for step in steps]


def loss_rows(result: DistributedResult) -> list[dict]:
    """One row per step: ``{"step": s, "rank0": l0, "rank1": l1}``."""
    by_rank = {rank.rank: {row["step"]: row["loss"]
                           for row in rank.metrics}
               for rank in result.ranks}
    return _merged_rows(by_rank)
