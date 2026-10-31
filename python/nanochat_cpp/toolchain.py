"""The orchestration layer for ``nanochat.cpp`` (docs/python.md section 6).

The layer exposes every ``tools/nanochat`` command as a Python function. It
calls the shell tool through ``subprocess`` and never calls Bazel, the sandbox,
or the GPU queue directly. :class:`CommandResult` carries the result and
:class:`ToolError` reports a failure.

The command table in :mod:`nanochat_cpp._entry` is the single source of truth.
A new shell subcommand is one table row.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from . import _entry

__all__ = [
    "CommandResult",
    "DoctorReport",
    "ToolError",
    "Toolchain",
    "bench",
    "build",
    "check",
    "client",
    "commands",
    "dispatch",
    "dist",
    "doctor",
    "eval",
    "gpu",
    "gpuq",
    "lint",
    "prune",
    "profile",
    "rl",
    "run",
    "shutdown",
    "test",
    "train",
    "verify",
]


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """The result of one shell command.

    ``argv`` is the argument list after the entry script, so it is identical to
    what the shell tool receives. ``seconds`` is the wall-clock duration.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


class ToolError(RuntimeError):
    """A shell command that returned a nonzero status."""

    def __init__(self, result: CommandResult) -> None:
        self.argv = result.argv
        self.returncode = result.returncode
        self.stderr = result.stderr
        self.stdout = result.stdout
        message = (f"{' '.join(result.argv)} exited with "
                   f"{result.returncode}")
        if result.stderr.strip():
            message += "\n" + result.stderr.rstrip()
        super().__init__(message)


@dataclasses.dataclass(frozen=True)
class DoctorReport:
    """The parsed ``tools/nanochat doctor --json`` result.

    ``gpu_queue`` holds the queue object: ``worker_id``, ``alive``,
    ``devices``, ``depth``, ``running_job``, and ``queued``. ``raw`` holds the
    unparsed output so a caller can show it when the shell tool is older and
    prints text only.
    """

    repo: str = ""
    revision: str = ""
    branch: str = ""
    nvcc: str = ""
    bazel: str = ""
    gpu: str = ""
    gpu_procs: str = ""
    sandbox: str = ""
    sandbox_backend: str = ""
    gpu_queue: dict = dataclasses.field(default_factory=dict)
    raw: str = ""


def _default_root() -> Path:
    """The repository root: ``python/nanochat_cpp`` is two levels down."""
    return Path(__file__).resolve().parents[2]


class Toolchain:
    """Builds ``tools/nanochat`` argument lists and runs them.

    ``entry`` overrides the shell tool path (the test uses a fake script).
    ``env`` overrides the child environment; the default is a copy of
    ``os.environ``. ``check`` controls whether a nonzero status raises
    :class:`ToolError`; the per-call ``check`` argument wins.

    A command that runs a child executable has two shapes (see
    :meth:`_dispatch`): through ``tools/nanochat`` when this process holds no
    scope, and as a direct child of this process when it already does. The
    second shape is what keeps the three stages of the chain inside the one
    scope the run was given (docs/pipeline-plan.md B5).
    """

    def __init__(self, entry: str | os.PathLike | None = None,
                 repo_root: str | os.PathLike | None = None,
                 env: Mapping[str, str] | None = None,
                 timeout: float | None = None,
                 check: bool = True) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None \
            else _default_root()
        self.entry = Path(entry) if entry is not None \
            else self.repo_root / "tools" / "nanochat"
        self.env = dict(os.environ) if env is None else dict(env)
        self.timeout = timeout
        self._check = check

    def _run(self, argv: Sequence[str], check: bool | None = None) -> CommandResult:
        """Run ``self.entry`` with ``argv`` and capture the result."""
        arguments = [str(value) for value in argv]
        return self._capture([str(self.entry), *arguments], arguments, check)

    def _run_direct(self, argv: Sequence[str],
                    check: bool | None = None) -> CommandResult:
        """Run one child executable itself and capture the result.

        The child inherits ``self.env``, which holds the ``NANOCHAT_SANDBOX``
        or ``NANOCHAT_GPU_QUEUED`` sentinel that says a scope or a queued job
        is already active, so the child runs inside the scope this process
        already holds and no second profile is created.
        """
        arguments = [str(value) for value in argv]
        return self._capture(arguments, arguments, check)

    def _capture(self, command: Sequence[str], arguments: Sequence[str],
                 check: bool | None = None) -> CommandResult:
        """Run ``command`` and capture it; ``arguments`` is the reported argv.

        The entry point and a direct child share this path, so the two arms
        report one :class:`CommandResult` and fail the same way.
        """
        reported = [str(value) for value in arguments]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                [str(part) for part in command], cwd=str(self.repo_root),
                env=self.env, capture_output=True, text=True,
                timeout=self.timeout, check=False)
            seconds = time.monotonic() - start
        except OSError as error:
            result = CommandResult(tuple(reported), 127, "", str(error),
                                   time.monotonic() - start)
            raise ToolError(result) from error
        result = CommandResult(argv=tuple(reported),
                               returncode=completed.returncode,
                               stdout=completed.stdout,
                               stderr=completed.stderr,
                               seconds=seconds)
        if (self._check if check is None else check) and result.returncode != 0:
            raise ToolError(result)
        return result

    # -- The in-scope branch ----------------------------------------------

    def _dispatch(self, direct: Sequence[str] | None,
                  wrapped: Sequence[str]) -> CommandResult:
        """Run a command directly inside a scope, and via the entry outside.

        When ``NANOCHAT_SANDBOX`` or ``NANOCHAT_GPU_QUEUED`` is set this
        process already holds a resource scope or a queued job, so ``direct``
        (one child executable and its arguments) runs as a child of this
        process and inherits that scope. Otherwise ``wrapped`` runs through
        ``tools/nanochat``, which starts the sandbox scope, the GPU queue, and
        the resource profile.

        ``direct`` is ``None`` for a command whose argv carries flags and not
        a child executable (:meth:`profile`): the entry point owns that
        command, so both arms run ``wrapped``.

        Two facts about the tree make this the only working shape, and both
        are one command each to check. First, the entry point is not a
        passthrough: ``tools/nanochat /bin/echo hi`` prints ``unknown command
        '/bin/echo'`` and exits 2, so handing a raw child argv to ``_run``
        cannot be the direct arm (the pre-B5 branch of :meth:`gpu` did exactly
        that, and so never ran its child). Second, the entry point already
        skips its own scope when a sentinel is set: ``cli.Context.submit``
        execs the child under ``in_sandbox()``, which
        ``NANOCHAT_SANDBOX=t0-cpu tools/nanochat train -- /bin/echo ok`` shows
        by printing ``ok`` and starting no profile. So the defect of B5 is
        narrower than "a nested tools/nanochat starts a nested scope": what
        still re-enters is the *queue* (:meth:`gpu` under a bare sandbox
        sentinel, and ``profile``, whose handler submits a job). This helper is
        the facade-level statement of one rule: a child launched from a
        process that already holds a scope runs in that scope, and nothing
        asks for a second one (docs/pipeline-plan.md B5).
        """
        if direct is not None and self._in_scope():
            return self._run_direct(direct)
        return self._run(wrapped)

    def _in_scope(self) -> bool:
        """True when this process already runs inside a scope or a queued job."""
        return bool(self.env.get("NANOCHAT_SANDBOX")
                    or self.env.get("NANOCHAT_GPU_QUEUED"))

    # -- Commands ---------------------------------------------------------

    def build(self, targets: Sequence[str] | None = None) -> CommandResult:
        return self._run(["build", *(targets or [])])

    def test(self, targets: Sequence[str] | None = None, *,
             gpu: bool = False) -> CommandResult:
        argv = ["test"]
        if gpu:
            argv.append("--gpu")
        argv.extend(targets or [])
        return self._run(argv)

    def check(self, *, gpu: bool = False) -> CommandResult:
        return self._run(["check", "--gpu"] if gpu else ["check"])

    def lint(self, paths: Sequence[str] | None = None) -> CommandResult:
        return self._run(["lint", *(paths or [])])

    def shutdown(self) -> CommandResult:
        return self._run(["shutdown"])

    def prune(self, *, apply: bool = False,
              worktree: str | os.PathLike | None = None) -> CommandResult:
        argv = ["prune"]
        if apply:
            argv.append("--apply")
        if worktree is not None:
            argv.extend(["--worktree", str(worktree)])
        return self._run(argv)

    def run(self, profile: str,
            argv: Sequence[str] | None = None) -> CommandResult:
        return self._run(["run", str(profile), "--", *(argv or [])])

    def dist_argv(self, argv: Sequence[str], *, nproc_per_node: int = 2,
                  device: str | None = None, master: str = "127.0.0.1",
                  port: int = 29500, profile: str = "train",
                  log_dir: str | os.PathLike | None = None,
                  json: bool = False) -> list[str]:
        """Build the ``tools/nanochat dist`` argument list without running it.

        ``launch_live`` uses the list to start the launcher itself, so the
        caller can poll the run. :meth:`dist` runs the same list.
        """
        arguments = ["dist", "--nproc-per-node", str(nproc_per_node)]
        if device is not None:
            arguments += ["--device", str(device)]
        arguments += ["--profile", str(profile), "--master", str(master),
                      "--port", str(port)]
        if log_dir is not None:
            arguments += ["--log-dir", str(log_dir)]
        if json:
            arguments.append("--json")
        arguments += ["--", *(str(value) for value in argv)]
        return arguments

    def dist(self, argv: Sequence[str], *, nproc_per_node: int = 2,
             device: str | None = None, master: str = "127.0.0.1",
             port: int = 29500, profile: str = "train",
             log_dir: str | os.PathLike | None = None,
             json: bool = False) -> CommandResult:
        """Launch one process per rank through ``tools/nanochat dist``.

        The rank values travel in the ``NANOCHAT_*`` environment, so ``argv``
        is the same for every rank; a ``{rank}`` token in an argument expands
        to the rank index. See ``docs/distributed-launch.md``.
        """
        return self._run(self.dist_argv(
            argv, nproc_per_node=nproc_per_node, device=device,
            master=master, port=port, profile=profile, log_dir=log_dir,
            json=json))

    def client(self, argv: Sequence[str], *, device: str | None = None,
               profile: str = "train") -> CommandResult:
        """Run one client process under one queue job for the device set.

        The client enumerates every local device itself, builds one mesh, and
        trains in one process. See ``docs/client-runtime-design.md`` section
        14.
        """
        arguments = ["client"]
        if device is not None:
            arguments += ["--device", str(device)]
        arguments += ["--profile", str(profile), "--",
                      *(str(value) for value in argv)]
        return self._run(arguments)

    def train(self, argv: Sequence[str] | None = None) -> CommandResult:
        """Run one child executable under the ``train`` profile.

        Inside a scope or a queued job the child itself runs, so a stage of the
        chain keeps the one scope it was given and starts no second profile
        (docs/pipeline-plan.md B5).
        """
        arguments = [*(argv or [])]
        return self._dispatch(arguments, ["train", "--", *arguments])

    def rl(self, argv: Sequence[str] | None = None) -> CommandResult:
        """Run the persistent RL worker.

        The RL entry in :mod:`nanochat_cpp._entry` is an alias of ``train``
        (``docs/rl-notebook.md`` section 4), and the shell routes the RL worker
        through the ``train`` profile. The method emits that profile so the
        alias works without a separate shell subcommand.
        """
        arguments = [*(argv or [])]
        return self._dispatch(arguments, ["train", "--", *arguments])

    def eval(self, argv: Sequence[str] | None = None) -> CommandResult:
        """Run one evaluation child, directly inside an active scope."""
        arguments = [*(argv or [])]
        return self._dispatch(arguments, ["eval", "--", *arguments])

    def bench(self, argv: Sequence[str] | None = None) -> CommandResult:
        """Run one benchmark child, directly inside an active scope."""
        arguments = [*(argv or [])]
        return self._dispatch(arguments, ["bench", "--", *arguments])

    def verify(self, argv: Sequence[str] | None = None) -> CommandResult:
        """Run one verification child, directly inside an active scope."""
        arguments = [*(argv or [])]
        return self._dispatch(arguments, ["verify", "--", *arguments])

    def profile(self, argv: Sequence[str] = ()) -> CommandResult:
        """Run the attention-benchmark profile.

        ``argv`` carries flags (``--json``), not a child executable: the entry
        point builds the benchmark and submits it to the queue itself, so this
        method has no direct arm and calls the entry point in both branches.
        """
        arguments = [*argv]
        return self._dispatch(None, ["profile", *arguments])

    def gpu(self, argv: Sequence[str], *,
            profile: str | None = None) -> CommandResult:
        """Submit one GPU job to the queue, then wait for it.

        When the process already runs inside the sandbox or a queued job, the
        command itself runs and no second job is submitted. This mirrors the
        shell entry point and avoids a nested job.
        """
        prefix = ["gpu"]
        if profile is not None:
            prefix.extend(["--profile", str(profile)])
        return self._dispatch(argv, [*prefix, "--", *argv])

    def gpuq(self, argv: Sequence[str]) -> CommandResult:
        """Run one ``tools/nanochat gpuq`` action.

        ``argv`` is the action and its arguments, for example
        ``["status", "--json"]`` or ``["submit", "--", "binary"]``.
        """
        return self._run(["gpuq", *(str(value) for value in argv)])

    def doctor(self) -> DoctorReport:
        """Return the environment report from the tool's JSON output."""
        result = self._run(["doctor", "--json"], check=False)
        report = _parse_doctor_json(result.stdout)
        if report is None:
            raise ToolError(result)
        return report

    def dispatch(self, name: str, *args: object) -> CommandResult:
        """Run any command by name, with the given arguments."""
        _entry.command(name)
        return self._run([name, *(str(value) for value in args)])

    def commands(self) -> tuple[str, ...]:
        """The command list from the table."""
        return _entry.names()


def _parse_doctor_json(text: str) -> DoctorReport | None:
    """Parse a JSON doctor report, or return None for a non-JSON report."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    fields = {}
    for field in dataclasses.fields(DoctorReport):
        if field.name == "raw":
            continue
        value = payload.get(field.name, "")
        if field.name == "gpu_queue":
            fields[field.name] = value if isinstance(value, dict) else {}
        else:
            fields[field.name] = "" if value is None else str(value)
    return DoctorReport(raw=text, **fields)


# -- Module-level surface over one default toolchain ---------------------

_default: Toolchain | None = None


def _toolchain() -> Toolchain:
    global _default
    if _default is None:
        _default = Toolchain()
    return _default


def build(targets: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().build(targets)


def test(targets: Sequence[str] | None = None, *,
         gpu: bool = False) -> CommandResult:
    return _toolchain().test(targets, gpu=gpu)


def check(*, gpu: bool = False) -> CommandResult:
    return _toolchain().check(gpu=gpu)


def lint(paths: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().lint(paths)


def shutdown() -> CommandResult:
    return _toolchain().shutdown()


def prune(*, apply: bool = False,
          worktree: str | os.PathLike | None = None) -> CommandResult:
    return _toolchain().prune(apply=apply, worktree=worktree)


def run(profile: str, argv: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().run(profile, argv)


def dist(argv: Sequence[str], *, nproc_per_node: int = 2,
         device: str | None = None, master: str = "127.0.0.1",
         port: int = 29500, profile: str = "train",
         log_dir: str | os.PathLike | None = None,
         json: bool = False) -> CommandResult:
    return _toolchain().dist(argv, nproc_per_node=nproc_per_node,
                             device=device, master=master, port=port,
                             profile=profile, log_dir=log_dir, json=json)


def client(argv: Sequence[str], *, device: str | None = None,
           profile: str = "train") -> CommandResult:
    return _toolchain().client(argv, device=device, profile=profile)


def train(argv: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().train(argv)


def rl(argv: Sequence[str] | None = None) -> CommandResult:
    """Run the persistent RL worker through the ``train`` profile.

    The RL entry in :mod:`nanochat_cpp._entry` is an alias of ``train``
    (``docs/rl-notebook.md`` section 4).
    """
    return _toolchain().rl(argv)


def eval(argv: Sequence[str] | None = None) -> CommandResult:  # noqa: A001
    return _toolchain().eval(argv)


def bench(argv: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().bench(argv)


def verify(argv: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().verify(argv)


def profile(argv: Sequence[str] = ()) -> CommandResult:
    return _toolchain().profile(argv)


def gpu(argv: Sequence[str], *,
        profile: str | None = None) -> CommandResult:
    return _toolchain().gpu(argv, profile=profile)


def gpuq(argv: Sequence[str]) -> CommandResult:
    return _toolchain().gpuq(argv)


def doctor() -> DoctorReport:
    return _toolchain().doctor()


def dispatch(name: str, *args: object) -> CommandResult:
    return _toolchain().dispatch(name, *args)


def commands() -> tuple[str, ...]:
    return _toolchain().commands()
