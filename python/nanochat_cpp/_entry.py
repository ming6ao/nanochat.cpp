"""The command table for the host tool (docs/python.md section 6).

This module is the single source of truth for the ``tools/nanochat`` command
list and for its help text. ``python/nanochat_cpp/cli.py`` renders the usage
block from :func:`usage` and validates the requested command against
:func:`names`, and :mod:`nanochat_cpp.toolchain` exposes one function per row.
A new subcommand is one row.

The module imports only the Python standard library.
"""

from __future__ import annotations

import dataclasses

__all__ = ["Command", "COMMANDS", "command", "names", "usage"]


@dataclasses.dataclass(frozen=True)
class Command:
    """One ``tools/nanochat`` subcommand.

    ``help_lines`` holds the exact lines that the usage block shows for the
    command, including the aligned summary. ``alias`` marks a convenience alias
    of ``run``; the profile is in the handler.
    """

    name: str
    help_lines: tuple[str, ...]
    alias: bool = False


#: The command list, in help order. This tuple is the single source of truth.
COMMANDS = (
    Command("build", (
        "  build [targets...]         bazel build (default //...; serialized)",
    )),
    Command("test", (
        "  test [--gpu] [--config NAME] [targets...]",
        "                             run tests; every test is sandboxed per action,",
        "                             and every GPU action is submitted to the queue",
    )),
    Command("check", (
        "  check [--gpu]              build, then CPU tests (and GPU tests with --gpu)",
    )),
    Command("lint", (
        "  lint [file...]             style gate: clang-format + the text checks",
    )),
    Command("shutdown", (
        "  shutdown                   stop this workspace's Bazel server now",
    )),
    Command("prune", (
        "  prune [--apply] [--worktree DIR]",
        "                             reap orphaned Bazel output bases (dry run by",
        "                             default); see docs/sandbox.md",
    )),
    Command("gpuq", (
        "  gpuq <action> [opts]       the GPU job queue: submit, list, status, wait,",
        "                             logs, cancel, start, stop; see",
        "                             docs/gpu-queue-plan.md",
    )),
    Command("run", (
        "  run <profile> -- <cmd...>  run a command under a sandbox profile; a GPU",
        "                             profile is submitted to the queue",
    )),
    Command("dist", (
        "  dist [opts] -- <cmd...>    one process per rank; one queue job for the",
        "                             group; sets the NANOCHAT_* rank variables and",
        "                             expands a `{rank}` token in the command. See",
        "                             docs/distributed-launch.md",
    )),
    Command("client", (
        "  client [opts] -- <cmd...>  one client process; one queue job for the full",
        "                             device set; the client sees all local devices as",
        "                             one mesh. See docs/client-runtime-design.md",
    )),
    Command("train", (
        "  train -- <cmd...>          alias for: run train -- <cmd...>",
    ), alias=True),
    Command("rl", (
        "  rl    -- <cmd...>          alias for: run train -- <cmd...>",
    ), alias=True),
    Command("eval", (
        "  eval  -- <cmd...>          alias for: run eval  -- <cmd...>",
    ), alias=True),
    Command("bench", (
        "  bench -- <cmd...>          alias for: run t3-bench -- <cmd...>",
    ), alias=True),
    Command("profile", (
        "  profile [opts]             build + run the attention benchmark in the queue",
    )),
    Command("roofline", (
        "  roofline [opts]            build + run the performance model (roofline bound)",
        "                             CPU only, no GPU queue; see",
        "                             docs/performance-model.md",
    )),
    Command("doctor", (
        "  doctor [--json]            print environment, device, sandbox, and queue state",
    )),
    Command("simulate", (
        "  simulate [--device NAME] [--suite S] [--mode compile] [--precision P]",
        "                             S0 CPU-only simulator (no GPU, no queue); see",
        "                             docs/simulator.md. Suites: correctness fp8-fast",
        "                             api collective kernel profile all (default all)",
        "                             `correctness` also carries the staged numeric",
        "                             trace (docs/numerics-integration.md)",
        "                             `fp8-fast` is the fast fp8 core",
        "                             (docs/correctness-gates.md section 5.1)",
        "                             Precision: fp32 (default) bf16 fp8",
    )),
    Command("verify", (
        "  verify -- <cmd...>         alias for: run t2-parity -- <cmd...>",
    ), alias=True),
    Command("gpu", (
        "  gpu [--profile P] -- <cmd> submit one job to the GPU queue",
    )),
    Command("help", (
        "  help                       this message",
    )),
)

#: The usage header and the trailing profile lines.
_HEADER = "tools/nanochat <command> [args...]"
_FOOTER = (
    "Profiles: t0-cpu t1-gpu t2-parity t3-bench train eval",
    "Profile budgets: tools/sandbox/profiles.json",
)

_BY_NAME = {entry.name: entry for entry in COMMANDS}


def command(name: str) -> Command:
    """The table entry for ``name``. Raise ``KeyError`` for an unknown name."""
    return _BY_NAME[name]


def names() -> tuple[str, ...]:
    """Every command name, in help order."""
    return tuple(entry.name for entry in COMMANDS)


def usage() -> str:
    """The full ``tools/nanochat help`` text."""
    lines = [_HEADER, ""]
    for entry in COMMANDS:
        lines.extend(entry.help_lines)
    lines.append("")
    lines.extend(_FOOTER)
    return "\n".join(lines) + "\n"
