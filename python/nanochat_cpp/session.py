#!/usr/bin/env python3
"""The agent session: one record, one slice, one network, one read-back.

Stages S1 and S2 of ``docs/sandbox.md`` Part 2. A session is four things: a
name, a record file, a systemd user slice called
``nanochat-agent-<id>.slice``, and a network setting. Systemd nests that slice
under ``nanochat.slice`` by the slice name hierarchy, so the aggregate ceiling
in ``tools/sandbox/nanochat.slice`` still applies.

Four rules hold, and each one answers a failure:

- **The record is the session.** A slice exists only while a command runs
  inside it, so the record on disk is what a later command reads. ``status``
  reports ``idle`` for a record with no live slice.
- **The build enters the slice.** ``tools/nanochat build`` reads the record for
  its worktree and starts Bazel inside the slice. The Bazel client detaches a
  server that outlives it, and a cgroup keeps that server because membership
  follows the fork. That is the mechanism ``docs/sandbox.md`` ``### Builds``
  describes.
- **The build enters a private network namespace**, unless the session names
  ``host``. A build that cannot fetch is a build whose inputs are all
  committed. `netprep` in ``sandbox.py`` refuses to run when any interface
  besides loopback exists.
- **Both claims are read back.** A server outside the slice, or a server that
  shares the host network while the session denies it, is a failure. A server
  outside every cgroup exhausted host memory on 2026-10-01.

The module imports only the Python standard library. The record directory is
``NANOCHAT_SESSION_DIR``, then ``$XDG_STATE_HOME/nanochat/sessions``, then
``~/.local/state/nanochat/sessions``.

    tools/nanochat session start --agent pi
    tools/nanochat session status --json
    tools/nanochat session stop
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "BUILD_PROFILE",
    "NET_DENY",
    "NET_HOST",
    "NET_MODES",
    "SLICE_PREFIX",
    "SessionError",
    "check_placement",
    "default_id",
    "find_for_root",
    "list_records",
    "net_is_private",
    "net_namespace",
    "normalize_id",
    "output_base",
    "placement_of",
    "read_ready",
    "record_path",
    "run_via_supervisor",
    "socket_path",
    "start_supervisor",
    "stop_supervisor",
    "supervisor_unit",
    "remove_record",
    "slice_name",
    "start_record",
    "state_dir",
    "status",
    "stop",
    "write_record",
]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

#: The slice name prefix. Systemd reads the dash hierarchy, so
#: ``nanochat-agent-x.slice`` sits under ``nanochat-agent.slice``, and that sits
#: under ``nanochat.slice``.
SLICE_PREFIX = "nanochat-agent-"

#: The profile a session command uses. It carries a memory ceiling and no wall
#: clock, because a Bazel server outlives the client that started it.
BUILD_PROFILE = "build"

#: The network setting a session records. ``deny`` is the default: a build that
#: cannot fetch is a build whose inputs are all committed. ``host`` is the one
#: visible opt-in, for the first fetch on a fresh tree.
NET_DENY = "deny"
NET_HOST = "host"
NET_MODES = (NET_HOST, NET_DENY)

#: The environment variable that names the live session.
SESSION_ENV = "NANOCHAT_SESSION"

#: The supervisor: the transient unit name, its files, and its role programs.
SUPERVISOR_PREFIX = "nanochat-supervisor-"
SOCKET_SUFFIX = ".sock"
READY_SUFFIX = ".ready.json"
SUPERVISOR_PROGRAM = HERE / "supervisor.py"

#: How long a start waits for the ready file. The unit start plus an import.
READY_TIMEOUT_SEC = 20

#: The record format version. A record with another version is ignored.
RECORD_VERSION = 1

#: Where cgroup v2 is mounted on this host.
CGROUP_MOUNT = "/sys/fs/cgroup"

#: A systemd command must never hang the caller.
SD_TIMEOUT_SEC = 10

_NOT_ID = re.compile(r"[^a-z0-9]+")
_ID_LIMIT = 48


class SessionError(Exception):
    """A session name, record, or systemd call is not usable."""


# -- The name and the store ------------------------------------------------


def state_dir() -> Path:
    """The record directory."""
    explicit = os.environ.get("NANOCHAT_SESSION_DIR")
    if explicit:
        return Path(explicit)
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "nanochat" / "sessions"
    home = os.environ.get("HOME") or str(Path.home())
    return Path(home) / ".local" / "state" / "nanochat" / "sessions"


def normalize_id(text: str) -> str:
    """The systemd-safe form of a session name. Raise when nothing is left."""
    cleaned = _NOT_ID.sub("-", str(text).strip().lower()).strip("-")
    cleaned = cleaned[:_ID_LIMIT].strip("-")
    if not cleaned:
        raise SessionError(
            "a session name needs at least one letter or digit")
    return cleaned


def default_id(root: Path) -> str:
    """The session name of a worktree: its directory name."""
    return normalize_id(Path(root).name)


def slice_name(session_id: str) -> str:
    """The slice unit of a session."""
    return f"{SLICE_PREFIX}{normalize_id(session_id)}.slice"


def record_path(session_id: str) -> Path:
    """The record file of a session."""
    return state_dir() / f"{normalize_id(session_id)}.json"


def write_record(session_id: str, root: Path, *, agent: str = "",
                 net: str = NET_DENY, architect: bool = False) -> dict:
    """Write the record of one session, atomically. Return the record."""
    if net not in NET_MODES:
        raise SessionError(f"network mode must be one of {NET_MODES}")
    record = {
        "version": RECORD_VERSION,
        "id": normalize_id(session_id),
        "root": str(Path(root).resolve()),
        "slice": slice_name(session_id),
        "profile": BUILD_PROFILE,
        "agent": agent,
        "net": net,
        # The mount role, so a supervisor that a command starts on demand keeps
        # the frozen headers writable for the workstream that owns them.
        "architect": bool(architect),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = record_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.new")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return record


def read_record(session_id: str) -> dict | None:
    """The record of one session, or ``None`` when it is absent or unusable."""
    try:
        with open(record_path(session_id), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("version") != RECORD_VERSION:
        return None
    if not record.get("id") or not record.get("root"):
        return None
    return record


def remove_record(session_id: str) -> bool:
    """Delete one record. Return whether a record was there."""
    try:
        record_path(session_id).unlink()
    except OSError:
        return False
    return True


def list_records() -> list[dict]:
    """Every usable record, in name order."""
    directory = state_dir()
    try:
        names = sorted(directory.glob("*.json"))
    except OSError:
        return []
    records = []
    for path in names:
        record = read_record(path.stem)
        if record is not None:
            records.append(record)
    return records


def find_for_root(root: Path | None = None) -> dict | None:
    """The record whose worktree is ``root``, or ``None``."""
    wanted = str(Path(root or REPO_ROOT).resolve())
    for record in list_records():
        if record.get("root") == wanted:
            return record
    return None


def active() -> dict | None:
    """The session this command belongs to, or ``None``.

    The environment names the session when a command runs inside one. A command
    started in a session worktree finds the record instead, so a plain
    ``tools/nanochat build`` enters the session with no extra step.
    """
    named = os.environ.get(SESSION_ENV)
    if named:
        record = read_record(named)
        if record is not None:
            return record
        return {"id": normalize_id(named), "slice": slice_name(named),
                "root": str(REPO_ROOT), "version": RECORD_VERSION,
                "profile": BUILD_PROFILE, "started": ""}
    return find_for_root(REPO_ROOT)


# -- The slice -------------------------------------------------------------


def systemctl(*arguments: str) -> subprocess.CompletedProcess:
    """Run one ``systemctl --user`` call. Never raise and never hang."""
    try:
        return subprocess.run(
            ["systemctl", "--user", *arguments], capture_output=True,
            text=True, timeout=SD_TIMEOUT_SEC, check=False)
    except (OSError, subprocess.SubprocessError) as failure:
        return subprocess.CompletedProcess(
            ["systemctl", "--user", *arguments], 1, "", str(failure))


def _property(unit: str, name: str) -> str:
    """One property of one unit, or ``""`` when the unit is absent."""
    result = systemctl("show", "--property", name, "--value", unit)
    return result.stdout.strip() if result.returncode == 0 else ""


def slice_cgroup(session_id: str) -> str:
    """The cgroup path of the session slice, or ``""`` when it is absent."""
    unit = slice_name(session_id)
    if _property(unit, "LoadState") in ("", "not-found"):
        return ""
    return _property(unit, "ControlGroup")


def _read(path: str) -> str:
    """One cgroup value, or ``""``. ``path`` is relative to the mount point."""
    if not path:
        return ""
    try:
        with open(f"{CGROUP_MOUNT}{path}", encoding="utf-8",
                  errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _scopes(cgroup: str) -> list[dict]:
    """The limits of every command scope under the session slice.

    The slice holds the session ceiling. Each scope inside it holds one profile
    budget. The list shows where each limit sits, because a slice-level ``max``
    alone would hide the ceiling that actually stops a compile.
    """
    if not cgroup:
        return []
    top = f"{CGROUP_MOUNT}{cgroup}"
    found = []
    for directory, names, files in os.walk(top):
        names[:] = [name for name in sorted(names)
                    if not name.endswith(".slice")]
        if directory == top or "memory.max" not in files:
            continue
        relative = os.path.relpath(directory, CGROUP_MOUNT)
        found.append({
            "scope": os.path.basename(directory),
            "memory_current": _read(f"/{relative}/memory.current"),
            "memory_max": _read(f"/{relative}/memory.max"),
            "pids_max": _read(f"/{relative}/pids.max"),
        })
    return found


def status(session_id: str) -> dict:
    """The state of one session: the record, the slice, the supervisor, the server."""
    identifier = normalize_id(session_id)
    record = read_record(identifier) or {}
    cgroup = slice_cgroup(identifier)
    ready = read_ready(identifier)
    report = {
        "id": identifier,
        "slice": slice_name(identifier),
        "state": "active" if cgroup else "idle",
        "cgroup": cgroup,
        "root": record.get("root", ""),
        "started": record.get("started", ""),
        "memory_current": _read(f"{cgroup}/memory.current"),
        "memory_max": _read(f"{cgroup}/memory.max"),
        "memory_peak": _read(f"{cgroup}/memory.peak"),
        "pids_current": _read(f"{cgroup}/pids.current"),
        "pids_max": _read(f"{cgroup}/pids.max"),
        "scopes": _scopes(cgroup),
        "net": record.get("net", NET_DENY),
        "supervisor": "down",
        "supervisor_pid": "",
        "net_ns": "",
        "mnt_ns": "",
        "confinement": {},
        "server": "",
    }
    if ready is not None:
        report["supervisor"] = "up"
        report["supervisor_pid"] = str(ready.get("pid", ""))
        report["net_ns"] = str(ready.get("net_ns", ""))
        report["mnt_ns"] = str(ready.get("mnt_ns", ""))
        report["confinement"] = ready.get("confinement") or {}
    report["server"] = describe_placement(identifier, report["net_ns"])
    return report


def stop(session_id: str) -> int:
    """Stop the supervisor, then the slice, then remove the record.

    The slice stop is the teardown: a process in it dies even when it called
    ``setsid`` and moved to its own scope. There is no process namespace, so
    the slice is the mechanism and not the kernel's pid-1 rule.
    """
    identifier = normalize_id(session_id)
    unit = slice_name(identifier)
    was_active = bool(slice_cgroup(identifier))
    failure = stop_supervisor(identifier)
    if failure:
        print(f"session: cannot stop the supervisor: {failure}")
        return 1
    result = systemctl("stop", unit)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        print(f"session: cannot stop {unit}: {message}")
        return 1
    leftover = slice_cgroup(identifier)
    if leftover:
        print(f"session: {unit} still holds {leftover}")
        return 1
    remove_record(identifier)
    if was_active:
        print(f"session: stopped {unit}")
    else:
        print(f"session: {unit} was idle")
    return 0


def start_record(session_id: str, root: Path | None = None,
                 agent: str = "", net: str = NET_DENY,
                 architect: bool = False) -> dict:
    """Write the record of one session."""
    return write_record(session_id, root or REPO_ROOT, agent=agent, net=net,
                        architect=architect)


# -- The supervisor --------------------------------------------------------


def socket_path(session_id: str) -> Path:
    """The command socket of a session."""
    return state_dir() / f"{normalize_id(session_id)}{SOCKET_SUFFIX}"


def ready_path(session_id: str) -> Path:
    """The ready file of a session. It holds the namespace identifiers."""
    return state_dir() / f"{normalize_id(session_id)}{READY_SUFFIX}"


def supervisor_unit(session_id: str) -> str:
    """The transient unit of a session supervisor."""
    return f"{SUPERVISOR_PREFIX}{normalize_id(session_id)}.service"


def read_ready(session_id: str) -> dict | None:
    """The ready file of a session, or ``None`` when it is absent."""
    try:
        with open(ready_path(session_id), encoding="utf-8") as handle:
            ready = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(ready, dict) or not ready.get("pid"):
        return None
    if not Path(f"/proc/{ready['pid']}").is_dir():
        return None
    return ready


def interpreter() -> str:
    """The Python interpreter that runs the supervisor roles."""
    return sys.executable or shutil.which("python3") or "python3"


def supervisor_unit_properties(values: dict, session_id: str) -> list[str]:
    """The systemd arguments of one session supervisor."""
    return [
        "--unit", supervisor_unit(session_id),
        "--working-directory", str(Path.cwd()),
        "--slice", slice_name(session_id),
        "-p", f"MemoryHigh={values['mem_high']}",
        "-p", f"MemoryMax={values['mem_max']}",
        "-p", "MemorySwapMax=0",
        "-p", f"CPUQuota={values['quota']}",
        "-p", f"TasksMax={values['pids']}",
    ]


def start_supervisor(session_id: str, values: dict, net: str,
                     root: Path | None = None,
                     architect: bool = False) -> dict | None:
    """Start the session supervisor, or return the live one.

    The supervisor owns the user, mount, and network namespaces of the session.
    There is no process namespace: it breaks the systemd bus, and every test
    action needs that bus for its own scope (`docs/sandbox.md` section 17).
    """
    identifier = normalize_id(session_id)
    ready = read_ready(identifier)
    if ready is not None:
        return ready
    socket_file = socket_path(identifier)
    ready_file = ready_path(identifier)
    for stale in (ready_file, socket_file):
        try:
            stale.unlink()
        except OSError:
            pass
    namespace = ["unshare", "--user", "--map-root-user", "--mount"]
    if net == NET_DENY:
        namespace.append("--net")
    worktree = Path(root or REPO_ROOT).resolve()
    command = [
        interpreter(), str(SUPERVISOR_PROGRAM), "hold", str(socket_file),
        str(ready_file), net, str(worktree),
        "architect" if architect else "user",
    ]
    argv = ["systemd-run", "--user", "--quiet", "--collect",
            *supervisor_unit_properties(values, identifier),
            "--", "taskset", "-c", values["cpus"], *namespace, "--",
            *command]
    try:
        result = subprocess.run(argv, cwd=str(root or REPO_ROOT),
                                capture_output=True, text=True,
                                timeout=READY_TIMEOUT_SEC + 10, check=False)
    except (OSError, subprocess.SubprocessError) as failure:
        raise SessionError(f"cannot start the supervisor: {failure}") from failure
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise SessionError(f"cannot start the supervisor: {message}")
    deadline = time.monotonic() + READY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        ready = read_ready(identifier)
        if ready is not None:
            return ready
        time.sleep(0.1)
    raise SessionError(
        f"the supervisor did not become ready; see: journalctl --user -u "
        f"{supervisor_unit(identifier)}")


def stop_supervisor(session_id: str) -> str:
    """Stop the supervisor unit. Return the last error line, or ``""``."""
    identifier = normalize_id(session_id)
    result = systemctl("stop", supervisor_unit(identifier))
    for path in (ready_path(identifier), socket_path(identifier)):
        try:
            path.unlink()
        except OSError:
            pass
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if "not loaded" in message or "not found" in message:
            return ""
        return message
    return ""


def run_via_supervisor(session_id: str, command: list[str],
                       root: Path | None = None) -> int:
    """Run one command in the session namespaces and stream its output."""
    identifier = normalize_id(session_id)
    argv = [interpreter(), str(SUPERVISOR_PROGRAM), "send",
            str(socket_path(identifier)), str(root or REPO_ROOT), "--",
            *command]
    try:
        return subprocess.run(argv, check=False).returncode
    except OSError as failure:
        raise SessionError(f"cannot run the command: {failure}") from failure


# -- The build server ------------------------------------------------------


def output_base(root: Path | None = None) -> Path | None:
    """The Bazel output base of a workspace, or ``None``.

    ``bazel-bin`` is a symlink through ``<output_base>/execroot/...``, so the
    output base is the directory above ``execroot``. Reading the link starts no
    server, and that matters: a Bazel call from the host would start a host
    server, which is exactly the state this module must detect.
    """
    link = Path(root or REPO_ROOT) / "bazel-bin"
    try:
        target = Path(os.path.realpath(link))
    except OSError:
        return None
    for parent in [target, *target.parents]:
        if parent.name == "execroot":
            return parent.parent
    return None


def server_pid(root: Path | None = None) -> int | None:
    """The process identifier of the Bazel server, or ``None``.

    The server writes its own identifier to ``<output_base>/server/server.pid.txt``
    at startup. The file is the only honest source: it names the server of this
    output base whether or not the caller can reach it, and it costs no server
    start. A stale file names a dead process, and every caller treats a missing
    ``/proc`` entry as ``unknown``.
    """
    base = output_base(root)
    if base is None:
        return None
    try:
        text = (base / "server" / "server.pid.txt").read_text(
            encoding="utf-8").strip()
    except OSError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    if not Path(f"/proc/{pid}").is_dir():
        return None
    return pid


def server_cgroup(pid: int) -> str:
    """The cgroup path of one process, or ``""``."""
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8",
                  errors="replace") as handle:
            line = handle.readline().strip()
    except OSError:
        return ""
    fields = line.split(":", 2)
    return fields[2] if len(fields) == 3 else ""


def net_namespace(pid: int | None = None) -> str:
    """The network namespace of one process, of this one, or ``""``."""
    target = "/proc/self/ns/net" if pid is None else f"/proc/{pid}/ns/net"
    try:
        return os.readlink(target)
    except OSError:
        return ""


def net_is_private(pid: int) -> bool | None:
    """Whether one process is out of the host network namespace."""
    theirs = net_namespace(pid)
    if not theirs:
        return None
    return theirs != net_namespace()


def placement_of(pid: int, session_id: str) -> tuple[bool | None, str]:
    """Whether one process is in the session slice, and where it runs."""
    cgroup = server_cgroup(pid)
    if not cgroup:
        return None, ""
    wanted = slice_name(session_id)
    return f"/{wanted}/" in cgroup or cgroup.endswith(f"/{wanted}"), cgroup


def describe_placement(session_id: str, netns: str = "") -> str:
    """One line about the build server: the pid, the cgroup, and the network."""
    pid = server_pid()
    if pid is None:
        return "unavailable"
    inside, _ = placement_of(pid, session_id)
    if inside is None:
        return f"pid={pid} gone"
    if netns:
        where = "session" if net_namespace(pid) == netns else "other"
    else:
        where = {True: "private", False: "host", None: "unknown"}[
            net_is_private(pid)]
    return (f"pid={pid} in-session={'yes' if inside else 'no'} "
            f"net={where}")


def check_placement(session_id: str, net: str = NET_DENY,
                    netns: str = "") -> int:
    """Fail when the build server is outside the session. Return a status.

    Three claims are checked: the cgroup, the network when the session denies
    it, and the exact namespace when a supervisor owns one. A server outside
    any of them is the failure this stage exists to catch.
    """
    pid = server_pid()
    if pid is None:
        # Absent or busy. A busy server answers the probe late, so this is a
        # warning and not a failure: the build itself already succeeded.
        print(f"session: {session_id}: placement not verified "
              "(no build server, or a busy one)")
        return 0
    inside, cgroup = placement_of(pid, session_id)
    if inside is None:
        print(f"session: {session_id}: placement not verified "
              f"(server pid {pid} has gone)")
        return 0
    if not inside:
        print(f"session: {session_id}: build server is outside the session "
              "slice")
        print(f"session: {session_id}: it runs in {cgroup}")
        print("session: the memory ceiling did not cover this build; see "
              "docs/sandbox.md ### Builds")
        print("session: the command itself ran; the placement is the failure")
        return 1
    if net != NET_DENY:
        print(f"session: {session_id}: bazel server in {cgroup}")
        return 0
    if netns:
        theirs = net_namespace(pid)
        if theirs != netns:
            print(f"session: {session_id}: build server is in another "
                  "network namespace")
            print(f"session: {session_id}: it has {theirs or 'unknown'}, "
                  f"the session has {netns}")
            print("session: this build could fetch; see docs/sandbox.md "
                  "section 19")
            print("session: the command itself ran; the placement is the "
                  "failure")
            return 1
    elif net_is_private(pid) is False:
        print(f"session: {session_id}: build server shares the host network")
        print("session: this build could fetch; see docs/sandbox.md "
              "section 19")
        print("session: a fetch needs: tools/nanochat build --net host ...")
        print("session: the command itself ran; the placement is the failure")
        return 1
    print(f"session: {session_id}: bazel server in {cgroup}")
    print(f"session: {session_id}: bazel server has no network")
    return 0
