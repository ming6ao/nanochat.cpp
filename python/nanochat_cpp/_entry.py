"""The command table for the toolchain layer (docs/python.md section 6).

Every shell subcommand of ``tools/nanochat`` appears here once. The functions
in :mod:`nanochat_cpp.toolchain` read this table, so a new shell subcommand is
one table row. The ``rl`` row is the reinforcement-learning entry: an alias of
``train`` for the persistent worker (``docs/rl-notebook.md`` section 4). The
module imports only the Python standard library.
"""

from __future__ import annotations

import dataclasses

__all__ = ["Command", "COMMANDS", "command", "names"]


@dataclasses.dataclass(frozen=True)
class Command:
    """One ``tools/nanochat`` subcommand.

    ``alias`` marks a convenience alias of ``run``; the shell table in
    ``tools/nanochat`` defines the profile.
    """

    name: str
    summary: str
    alias: bool = False


#: The command list, in help order. This tuple is the single source of truth.
COMMANDS = (
    Command("build", "bazel build (default //...)"),
    Command("test", "run tests, sandboxed per action"),
    Command("check", "build, then the CPU tests (and GPU tests with --gpu)"),
    Command("lint", "style gate: clang-format and the text checks"),
    Command("shutdown", "stop this workspace's Bazel server now"),
    Command("prune", "reap orphaned Bazel output bases"),
    Command("run", "run a command under a sandbox profile"),
    Command("train", "alias for: run train -- <cmd...>", alias=True),
    Command("rl", "alias for: train -- <rl_worker...>", alias=True),
    Command("eval", "alias for: run eval -- <cmd...>", alias=True),
    Command("bench", "alias for: run t3-bench -- <cmd...>", alias=True),
    Command("profile", "build and run the attention benchmark"),
    Command("doctor", "print environment, device, sandbox, and lock state"),
    Command("verify", "alias for: run t2-parity -- <cmd...>", alias=True),
    Command("gpu", "acquire the GPU broker, then run"),
)

_BY_NAME = {entry.name: entry for entry in COMMANDS}


def command(name: str) -> Command:
    """The table entry for ``name``. Raise ``KeyError`` for an unknown name."""
    return _BY_NAME[name]


def names() -> tuple[str, ...]:
    """Every command name, in help order."""
    return tuple(entry.name for entry in COMMANDS)
