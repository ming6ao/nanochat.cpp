#!/usr/bin/env python3
"""The resource sandbox: the launcher and the limit verifier.

Every test, training run, evaluation, and benchmark starts through this program.
It applies a cgroup v2 budget (memory, CPU, process count, wall clock) through
the unprivileged systemd user manager, plus CPU affinity and hard rlimits. It
needs no root and no container runtime.

The program has two subcommands.

- ``launch`` (the default) reads the profile, starts the scope, and execs the
  job. See :func:`launch`.
- ``verify`` runs *inside* the scope, before the job. It reads the limits back
  from its own process cgroup. It fails when the budget was silently not
  applied, so an unsandboxed run cannot pass as a sandboxed one. See
  :func:`verify`.

The two subcommands must stay separate processes. The verifier reads the cgroup
of its own process, so a merged program would read the parent cgroup, which
sits outside the scope. ``taskset`` stays external for the same reason: it sets
the affinity mask, and a verifier that sets its own mask cannot check it. The
core limit uses ``resource.setrlimit``, because no read-back check covers it.

The observable contract is frozen and ``python/tests/contract_test.py`` pins it:
the environment variable names, the ``/tmp/nanochat-sandbox.log`` line formats,
the exit codes, the ``NANOCHAT_SANDBOX`` sentinel, and the option interface
``[--profile NAME|--profile=NAME] [--backend auto|systemd|none] [--no-slice]
[--] <cmd...>``.

The module imports only the Python standard library.

    tools/nanochat run t0-cpu -- ./bazel-bin/train/train_main --config d8
    python/nanochat_cpp/sandbox.py --profile=t0-cpu -- ./bench
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "load_profiles", "to_bytes", "cpu_count", "check_limits"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROFILES = REPO_ROOT / "tools" / "sandbox" / "profiles.json"
SLICE_SRC = REPO_ROOT / "tools" / "sandbox" / "nanochat.slice"

#: The frozen sandbox log. Decision 4 of the consolidation plan pins the path.
SANDBOX_LOG = Path("/tmp/nanochat-sandbox.log")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
PROC_SELF_STATUS = Path("/proc/self/status")

#: The default profile, and the profile for a CPU test action.
DEFAULT_PROFILE = "t0-cpu"


class UnknownProfile(Exception):
    """The profile name is not in the table."""


# -- The profile table -----------------------------------------------------


def load_profiles(path: Path | None = None) -> dict:
    """The profile table. ``path`` overrides the default, for a test."""
    with open(path or PROFILES, encoding="utf-8") as handle:
        return json.load(handle)


def profile_values(profiles: dict, name: str) -> dict:
    """The values of one profile, or :class:`UnknownProfile`."""
    values = profiles.get("profiles", {}).get(name)
    if values is None:
        raise UnknownProfile(name)
    return values


def to_bytes(value: str) -> int | str:
    """The byte count of a systemd size string.

    The parse mirrors the earlier shell verifier. A trailing ``K``, ``M``, or
    ``G`` scales the head. A non-numeric head without a unit is returned
    unchanged, so the comparison reports it as the wanted value.
    """
    head, unit = value, ""
    if value and value[-1] in "KkMmGg":
        head, unit = value[:-1], value[-1]
    if not head.isdigit():
        if unit:
            raise ValueError(value)
        return value
    number = int(head)
    if unit in ("K", "k"):
        return number * 1024
    if unit in ("M", "m"):
        return number * 1024 * 1024
    if unit in ("G", "g"):
        return number * 1024 * 1024 * 1024
    return number


def cpu_count(spec: str) -> int:
    """The number of CPUs in a ``taskset`` list such as ``0-1`` or ``0,2``."""
    total = 0
    for part in spec.split(","):
        if "-" in part:
            low, _, high = part.partition("-")
            total += int(high) - int(low) + 1
        else:
            total += 1
    return max(total, 1)


# -- The limit check -------------------------------------------------------


def check_limits(expected: dict, actual: dict) -> str:
    """The mismatch text, or ``""`` when every limit matches.

    The keys are ``mem``, ``cpu``, ``pids``, and ``cpus``. The text matches the
    earlier shell verifier byte for byte.
    """
    failure = ""
    if actual["mem"] != expected["mem"]:
        failure += f" memory.max={actual['mem']}(want {expected['mem']})"
    if actual["cpu"] != expected["cpu"]:
        failure += f" cpu.max='{actual['cpu']}'(want '{expected['cpu']}')"
    if actual["pids"] != expected["pids"]:
        failure += f" pids.max={actual['pids']}(want {expected['pids']})"
    if actual["cpus"] != expected["cpus"]:
        failure += f" cpus={actual['cpus']}(want {expected['cpus']})"
    return failure


def read_text(path: Path) -> str:
    """The trimmed file text, or ``missing`` when the read fails."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return "missing"


def read_cgroup_path() -> Path:
    """This process's cgroup directory under ``/sys/fs/cgroup``."""
    try:
        with open(PROC_SELF_CGROUP, encoding="utf-8",
                  errors="replace") as handle:
            line = handle.readline().rstrip("\n")
    except OSError:
        line = ""
    fields = line.split(":", 2)
    relative = fields[2] if len(fields) == 3 else ""
    return CGROUP_ROOT / relative.lstrip("/")


def read_cpus_allowed() -> str:
    """The ``Cpus_allowed_list`` value of this process, or ``""``."""
    try:
        with open(PROC_SELF_STATUS, encoding="utf-8",
                  errors="replace") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def read_limits() -> dict:
    """The four observed limits of this process."""
    cgroup = read_cgroup_path()
    return {
        "mem": read_text(cgroup / "memory.max"),
        "cpu": read_text(cgroup / "cpu.max"),
        "pids": read_text(cgroup / "pids.max"),
        "cpus": read_cpus_allowed(),
    }


# -- Small helpers ---------------------------------------------------------


def error(message: str) -> int:
    """Print one command-line error and return the argument exit code."""
    print(message, file=sys.stderr)
    return 2


def append_log(line: str) -> None:
    """Append one line to the frozen sandbox log."""
    with open(SANDBOX_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def quiet() -> bool:
    """True when the caller asked for no progress line."""
    return os.environ.get("NANOCHAT_SANDBOX_QUIET") == "1"


def exec_replace(argv: list[str]) -> int:
    """Replace this process with ``argv``. It does not return on success."""
    try:
        os.execvp(argv[0], argv)
    except OSError as failure:
        print(f"sandbox: cannot run {argv[0]}: {failure.strerror}",
              file=sys.stderr)
        return 127
    return 127


def visible_cpus() -> int:
    """The CPUs available to this process, like ``nproc``."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


def systemd_available() -> bool:
    """True when the unprivileged systemd user manager can start a scope."""
    try:
        result = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return False
    return result.returncode == 0


def slice_destination() -> Path:
    """The user drop-in path for the aggregate slice."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home)
    else:
        base = Path(os.environ.get("HOME") or Path.home()) / ".config"
    return base / "systemd" / "user" / "nanochat.slice"


def install_slice(destination: Path) -> None:
    """Install the aggregate ceiling, once, and report the result."""
    if destination.exists() or not SLICE_SRC.is_file():
        return
    reload_result = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SLICE_SRC, destination)
        reload_result = subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        reload_result = None
    if reload_result is not None and reload_result.returncode == 0:
        print(f"sandbox: installed aggregate slice at {destination}",
              file=sys.stderr)
    else:
        print(f"sandbox: warning: could not install {destination}; "
              "no aggregate ceiling", file=sys.stderr)


# -- launch ----------------------------------------------------------------


def parse_argv(argv: list[str]) -> tuple[str, str, bool, list[str]] | int:
    """Parse the launcher options. On error, print and return the exit code.

    The interface is frozen. ``--run_under`` needs ``--profile=NAME``, and the
    host tool uses ``--profile NAME``.
    """
    profile = DEFAULT_PROFILE
    backend = os.environ.get("NANOCHAT_SANDBOX_BACKEND") or "auto"
    use_slice = True
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--profile":
            if index + 1 >= len(argv) or not argv[index + 1]:
                return error("sandbox: --profile needs a name")
            profile = argv[index + 1]
            index += 2
        elif argument.startswith("--profile="):
            profile = argument.split("=", 1)[1]
            index += 1
        elif argument == "--backend":
            if index + 1 >= len(argv) or not argv[index + 1]:
                return error("sandbox: --backend needs a value")
            backend = argv[index + 1]
            index += 2
        elif argument.startswith("--backend="):
            backend = argument.split("=", 1)[1]
            index += 1
        elif argument == "--no-slice":
            use_slice = False
            index += 1
        elif argument == "--":
            index += 1
            break
        elif argument.startswith("-"):
            return error(f"sandbox: unknown option '{argument}'")
        else:
            break
    command = argv[index:]
    if not command:
        return error("sandbox: no command given")
    return profile, backend, use_slice, command


def run_unsandboxed(profile: str, command: list[str]) -> int:
    """Set the sentinel and exec the command without a scope.

    A host without cgroup v2, such as a Kaggle Notebook, needs this mode. The
    container quota and the entry-point guard are the only controls, so the
    sentinel records the deliberate choice.
    """
    os.environ["NANOCHAT_SANDBOX"] = "off"
    os.environ["NANOCHAT_SANDBOX_PROFILE"] = profile
    threads = visible_cpus()
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["NANOCHAT_NUM_THREADS"] = str(threads)
    line = f"sandbox: backend=none profile={profile} (not sandboxed)"
    append_log(line)
    if not quiet():
        print(line, file=sys.stderr)
    return exec_replace(command)


def run_in_scope(profiles: dict, profile: str, values: dict,
                 use_slice: bool, command: list[str]) -> int:
    """Start the systemd scope, then the verify step, then the command."""
    if use_slice:
        install_slice(slice_destination())
    properties = [
        "-p", f"MemoryHigh={values['mem_high']}",
        "-p", f"MemoryMax={values['mem_max']}",
        "-p", "MemorySwapMax=0",
        "-p", f"CPUQuota={values['quota']}",
        "-p", f"TasksMax={values['pids']}",
    ]
    if values.get("runtime"):
        properties += ["-p", f"RuntimeMaxSec={values['runtime']}"]
    if use_slice:
        properties.append(f"--slice={profiles['slice']}")
    # Match the library thread pools to the CPU budget. The cgroup is the hard
    # ceiling; this is the polite hint that avoids oversubscription.
    threads = cpu_count(values["cpus"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["NANOCHAT_NUM_THREADS"] = str(threads)
    # The sentinel, read by RequireSandboxOrDie() and by the verifier.
    os.environ["NANOCHAT_SANDBOX"] = profile
    interpreter = sys.executable or shutil.which("python3") or "python3"
    verifier = [
        interpreter, str(Path(__file__).resolve()), "verify",
        values["mem_max"], values["quota"], values["cpus"], values["pids"],
        "--", *command,
    ]
    argv = [
        "systemd-run", "--user", "--scope", "--quiet", "--same-dir",
        *properties, "--",
        "taskset", "-c", values["cpus"],
        *verifier,
    ]
    return exec_replace(argv)


def launch(argv: list[str]) -> int:
    """Apply the profile and exec one job."""
    parsed = parse_argv(argv)
    if isinstance(parsed, int):
        return parsed
    profile, backend, use_slice, command = parsed
    # The profile check comes first, so a bad name fails before any probe.
    profiles = load_profiles()
    try:
        values = profile_values(profiles, profile)
    except UnknownProfile:
        print(f"sandbox: unknown profile '{profile}'", file=sys.stderr)
        return 1
    if backend == "auto":
        backend = "systemd" if systemd_available() else "none"
    if backend not in ("systemd", "none"):
        return error("sandbox: backend must be systemd, none, or auto")
    os.environ["NANOCHAT_SANDBOX_BACKEND"] = backend
    # The GPU worker selects a device by exporting CUDA_VISIBLE_DEVICES, but a
    # Bazel test action does not inherit the worker's environment. Translate
    # the forwarded value unless the caller already chose a device.
    if not os.environ.get("CUDA_VISIBLE_DEVICES") and \
            os.environ.get("NANOCHAT_GPU_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["NANOCHAT_GPU_DEVICES"]
    if backend == "none":
        return run_unsandboxed(profile, command)
    return run_in_scope(profiles, profile, values, use_slice, command)


# -- verify ----------------------------------------------------------------


def drop_core_limit() -> None:
    """Keep core dumps out of the worktree, the way ``prlimit`` did.

    The verifier sets the limit, so the job inherits it. It must not set the
    affinity mask: the check below reads that mask back from `/proc`, and a
    verifier that writes the value cannot witness it.
    """
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def verify(argv: list[str]) -> int:
    """Read the limits back, then exec the command. It runs inside the scope."""
    if len(argv) < 4:
        return error("sandbox: verify needs four values and a command")
    drop_core_limit()
    mem_max, quota, cpus, pids = argv[0], argv[1], argv[2], argv[3]
    command = argv[4:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        return error("sandbox: no command given")
    try:
        wanted_mem = to_bytes(mem_max)
    except ValueError:
        print(f"sandbox: cannot parse size '{mem_max}'", file=sys.stderr)
        return 1
    head = quota[:-1] if quota.endswith("%") else quota
    wanted_cpu = f"{int(head) * 1000} 100000"
    expected = {"mem": str(wanted_mem), "cpu": wanted_cpu,
                "pids": pids, "cpus": cpus}
    actual = read_limits()
    failure = check_limits(expected, actual)
    if failure:
        print(f"sandbox: limits not applied:{failure}", file=sys.stderr)
        print("sandbox: refusing to run the command unsandboxed.",
              file=sys.stderr)
        return 1
    profile = os.environ.get("NANOCHAT_SANDBOX", "?")
    append_log(f"sandbox: profile={profile} mem.max={actual['mem']} "
               f"cpu.max={actual['cpu']} pids.max={actual['pids']} "
               f"cpus={actual['cpus']}")
    if not quiet():
        print(f"sandbox: profile={profile} mem.max={actual['mem']} "
              f"cpu.max={actual['cpu']} cpus={actual['cpus']}", file=sys.stderr)
    return exec_replace(command)


def main(argv: list[str] | None = None) -> int:
    """The command line. ``verify`` is the inner step; ``launch`` is default."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["verify"]:
        return verify(arguments[1:])
    return launch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
