"""The orchestration layer for ``nanochat.cpp`` (docs/python.md section 6).

The layer exposes every ``tools/nanochat`` command as a Python function. It
calls the shell tool through ``subprocess`` and never calls Bazel, the sandbox,
or the GPU broker directly. :class:`CommandResult` carries the result and
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
    "commands",
    "dispatch",
    "doctor",
    "eval",
    "gpu",
    "lint",
    "prune",
    "profile",
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

    ``raw`` holds the unparsed output so a caller can show it when the shell
    tool is older and prints text only.
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
    gpu_lock: str = ""
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
        command = [str(self.entry), *arguments]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=str(self.repo_root), env=self.env,
                capture_output=True, text=True, timeout=self.timeout,
                check=False)
            seconds = time.monotonic() - start
        except OSError as error:
            result = CommandResult(tuple(arguments), 127, "", str(error),
                                   time.monotonic() - start)
            raise ToolError(result) from error
        result = CommandResult(argv=tuple(arguments),
                               returncode=completed.returncode,
                               stdout=completed.stdout,
                               stderr=completed.stderr,
                               seconds=seconds)
        if (self._check if check is None else check) and result.returncode != 0:
            raise ToolError(result)
        return result

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

    def train(self, argv: Sequence[str] | None = None) -> CommandResult:
        return self._run(["train", "--", *(argv or [])])

    def eval(self, argv: Sequence[str] | None = None) -> CommandResult:
        return self._run(["eval", "--", *(argv or [])])

    def bench(self, argv: Sequence[str] | None = None) -> CommandResult:
        return self._run(["bench", "--", *(argv or [])])

    def verify(self, argv: Sequence[str] | None = None) -> CommandResult:
        return self._run(["verify", "--", *(argv or [])])

    def profile(self, argv: Sequence[str] = ()) -> CommandResult:
        return self._run(["profile", *argv])

    def gpu(self, argv: Sequence[str], *,
            profile: str | None = None) -> CommandResult:
        """Acquire the GPU broker, then run ``argv``.

        When the process already runs inside the sandbox, the broker is
        already held, so the command runs directly. This mirrors
        ``launcher.launch`` and avoids a nested lock.
        """
        if self.env.get("NANOCHAT_SANDBOX"):
            return self._run(argv)
        prefix = ["gpu"]
        if profile is not None:
            prefix.extend(["--profile", str(profile)])
        return self._run([*prefix, "--", *argv])

    def doctor(self) -> DoctorReport:
        """Return the environment report, from JSON when available."""
        result = self._run(["doctor", "--json"], check=False)
        report = _parse_doctor_json(result.stdout)
        if report is not None:
            return report
        # An older shell tool prints text only. Parse what is present.
        report = _parse_doctor_text(result.stdout)
        if result.returncode != 0 and not result.stdout.strip() and \
                not result.stderr.strip():
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
    """Parse a JSON doctor report, or return None for a non-JSON tool."""
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
        fields[field.name] = "" if value is None else str(value)
    return DoctorReport(raw=text, **fields)


def _parse_doctor_text(text: str) -> DoctorReport:
    """Parse the older text doctor output into a report (best effort)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized = key.strip().lower().replace(" ", "_")
        if normalized in {field.name for field in
                          dataclasses.fields(DoctorReport)}:
            values[normalized] = value.strip()
    return DoctorReport(raw=text, **values)


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


def train(argv: Sequence[str] | None = None) -> CommandResult:
    return _toolchain().train(argv)


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


def doctor() -> DoctorReport:
    return _toolchain().doctor()


def dispatch(name: str, *args: object) -> CommandResult:
    return _toolchain().dispatch(name, *args)


def commands() -> tuple[str, ...]:
    return _toolchain().commands()
