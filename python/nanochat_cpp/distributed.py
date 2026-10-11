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
from dataclasses import dataclass, field
from pathlib import Path

from . import toolchain

__all__ = [
    "DistributedError",
    "DistributedPlan",
    "DistributedResult",
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
             "--checkpoint", str(plan.checkpoint_path(rank))]
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
             "--checkpoint", str(Path(plan.out_dir) / "rank{rank}.nchkpt01")]
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


def loss_rows(result: DistributedResult) -> list[dict]:
    """One row per step: ``{"step": s, "rank0": l0, "rank1": l1}``.

    The row count is the shortest rank log, so a partial log does not invent a
    value. The ranks must agree on the loss; a difference points to a broken
    all-reduce.
    """
    by_rank = {rank.rank: {row["step"]: row["loss"] for row in rank.metrics}
               for rank in result.ranks}
    if not by_rank:
        return []
    steps = sorted(set.intersection(
        *(set(values) for values in by_rank.values())))
    return [{"step": step, **{f"rank{rank}": values[step]
                              for rank, values in by_rank.items()}}
            for step in steps]
