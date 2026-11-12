#!/usr/bin/env python3
"""The session supervisor: one namespace per session, one command at a time.

Stage S3 of ``docs/sandbox.md`` Part 2. The supervisor is a long-lived process
that owns the session's user, mount, and network namespaces, and runs each
command as a child. Containment then comes from process inheritance, so no
command needs a new namespace.

Two roles live here:

- ``hold`` is the long-lived side. It prepares the network, writes a ready
  file with the namespace identifiers, and serves one command at a time.
- ``send`` is the client side. It sends one command, streams the output, and
  returns the exit code.

The wire format is newline-delimited JSON. A request names ``argv``, ``cwd``,
and ``env``. A reply is a sequence of ``{"stream": "out"|"err", "data": ...}``
frames and one ``{"exit": N}`` frame, or an ``{"error": ...}`` frame and then
``{"exit": 125}``.

Why a supervisor and not a namespace per command: a network namespace cannot be
joined from outside once a user namespace owns it, and Bazel needs its server
and its clients in one namespace. The supervisor keeps that namespace alive, so
a denied build reuses the same Bazel server and costs no server start.

There is no process namespace. A process namespace made ``systemd-run --user
--scope`` fail with ``Failed to connect to bus`` on this host, and every test
action needs that scope (measured 2026-11-11, ``docs/sandbox.md`` section 17).
Orphan teardown comes from the slice instead: stopping it kills every process
in it, whether or not that process called ``setsid``.

The module imports only the Python standard library.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

__all__ = ["hold", "main", "send"]

#: The ready file name. It holds the namespace identifiers of the session.
READY_SUFFIX = ".ready.json"

#: The socket file name of a session.
SOCKET_SUFFIX = ".sock"

#: The limit of one command, and of a quiet stream. A build is minutes, and the
#: caller owns the real timeout, so the supervisor only refuses a request that
#: never ends.
COMMAND_TIMEOUT_SEC = 7200

#: The exit code of a command the supervisor could not start.
SPAWN_FAILURE = 127

#: Where the ``ip`` command may live.
IP_CANDIDATES = ("ip", "/usr/sbin/ip", "/sbin/ip")

LOOPBACK = "lo"

#: Credential paths the session must not see, when they exist. A directory gets
#: an empty one bound over it, and a file gets ``/dev/null``.
HIDDEN_PATHS = (
    "~/.ssh",
    "~/.aws",
    "~/.config/gh",
    "~/.netrc",
    "~/.git-credentials",
    "~/.docker/config.json",
)

#: The frozen headers of the contract. They are read-only outside an architect
#: session, because they are the most expensive files to change by accident.
FROZEN_HEADER_DIR = "include/nanochat"


def _send_frame(connection: socket.socket, payload: dict,
                lock: threading.Lock) -> None:
    """Write one JSON frame, so two reader threads cannot interleave."""
    line = json.dumps(payload) + "\n"
    with lock:
        try:
            connection.sendall(line.encode("utf-8"))
        except OSError:
            pass


def _pump(stream, connection: socket.socket, name: str,
          lock: threading.Lock) -> None:
    """Copy one child stream to frames until it ends."""
    try:
        for chunk in iter(lambda: stream.read(8192), b""):
            if not chunk:
                break
            _send_frame(connection, {"stream": name,
                                     "data": chunk.decode("utf-8",
                                                          "replace")}, lock)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def serve_one(connection: socket.socket, base_env: dict) -> int:
    """Read one request, run it, stream the output, and answer. Return the code."""
    connection.settimeout(COMMAND_TIMEOUT_SEC)
    handle = connection.makefile("rb")
    try:
        line = handle.readline()
    except OSError:
        return SPAWN_FAILURE
    if not line:
        return SPAWN_FAILURE
    try:
        request = json.loads(line.decode("utf-8"))
        argv = [str(value) for value in request["argv"]]
    except (ValueError, KeyError, TypeError):
        _send_frame(connection, {"error": "malformed request"},
                    threading.Lock())
        _send_frame(connection, {"exit": SPAWN_FAILURE}, threading.Lock())
        return SPAWN_FAILURE
    environment = dict(base_env)
    environment.update({str(k): str(v)
                        for k, v in (request.get("env") or {}).items()})
    cwd = request.get("cwd") or None
    lock = threading.Lock()
    try:
        child = subprocess.Popen(argv, cwd=cwd, env=environment,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 start_new_session=True, bufsize=0)
    except OSError as failure:
        _send_frame(connection, {"error": f"cannot run {argv[0]}: "
                                          f"{failure.strerror}"}, lock)
        _send_frame(connection, {"exit": SPAWN_FAILURE}, lock)
        return SPAWN_FAILURE
    readers = [threading.Thread(target=_pump,
                               args=(child.stdout, connection, "out", lock),
                               daemon=True),
               threading.Thread(target=_pump,
                               args=(child.stderr, connection, "err", lock),
                               daemon=True)]
    for reader in readers:
        reader.start()
    try:
        code = child.wait(timeout=COMMAND_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        _kill_group(child.pid)
        code = SPAWN_FAILURE
    for reader in readers:
        reader.join(timeout=10)
    _send_frame(connection, {"exit": code}, lock)
    return code


def _kill_group(pid: int) -> None:
    """Stop a command and every process in its group."""
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, signal_number)
        except OSError:
            return
        try:
            os.waitpid(pid, 0)
            return
        except OSError:
            pass


def serve(path: Path) -> int:
    """Serve the socket until the process is stopped."""
    try:
        path.unlink()
    except OSError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(8)
    base_env = dict(os.environ)
    while True:
        try:
            connection, _ = server.accept()
        except OSError:
            return 0
        except KeyboardInterrupt:
            return 0
        with connection:
            serve_one(connection, base_env)


def set_loopback_up() -> str:
    """Bring the loopback device up, or return the failure text."""
    for candidate in IP_CANDIDATES:
        binary = candidate if "/" in candidate else shutil.which(candidate)
        if not binary or not os.access(binary, os.X_OK):
            continue
        try:
            result = subprocess.run([binary, "link", "set", LOOPBACK, "up"],
                                    capture_output=True, text=True,
                                    timeout=10, check=False)
        except (OSError, subprocess.SubprocessError) as failure:
            return str(failure)
        if result.returncode == 0:
            return ""
        return (result.stderr or result.stdout).strip() or "ip link failed"
    return "no ip command on this host"


def interface_names(text: str | None = None) -> list[str]:
    """The interface names of this network namespace.

    ``/proc/self/net/dev`` follows the namespace of the reader, so the result is
    a read-back and not a claim. ``text`` overrides the read, for a test.
    """
    if text is None:
        try:
            with open("/proc/self/net/dev", encoding="utf-8",
                      errors="replace") as handle:
                text = handle.read()
        except OSError:
            return []
    names = []
    for line in text.splitlines()[2:]:
        name, separator, _ = line.partition(":")
        if separator and name.strip():
            names.append(name.strip())
    return names


def namespace_id(name: str) -> str:
    """The identifier of one namespace of this process, or ``""``."""
    try:
        return os.readlink(f"/proc/self/ns/{name}")
    except OSError:
        return ""


def _mount(argv: list[str], timeout: int = 20) -> str:
    """Run one ``mount`` command. Return the failure text, or ``""``."""
    binary = shutil.which("mount") or "/usr/bin/mount"
    try:
        result = subprocess.run([binary, *argv], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as failure:
        return str(failure)
    if result.returncode == 0:
        return ""
    return (result.stderr or result.stdout).strip() or "mount failed"


def bind_read_only(path: Path) -> str:
    """Bind one path onto itself and make that mount read-only."""
    failure = _mount(["--bind", str(path), str(path)])
    if failure:
        return failure
    return _mount(["-o", "remount,bind,ro", str(path), str(path)])


def bind_read_write(path: Path) -> str:
    """Bind one path onto itself with write access, inside a read-only parent.

    A bind mount inherits the read-only flag of its source mount, so the flag
    must be cleared after the bind. Without the second step the worktree stays
    read-only and every build fails.
    """
    failure = _mount(["--bind", str(path), str(path)])
    if failure:
        return failure
    return _mount(["-o", "remount,bind,rw", str(path), str(path)])


def hide_path(path: Path, empty: Path) -> str:
    """Put an empty path over a credential path, so the session cannot read it."""
    source = Path("/dev/null") if path.is_file() else empty
    failure = _mount(["--bind", str(source), str(path)])
    if failure:
        return failure
    if source == empty:
        return _mount(["-o", "remount,bind,ro", str(path), str(path)])
    return ""


def confine(worktree: Path, state: Path, architect: bool) -> dict[str, str]:
    """Apply the writable-set plan of the session. Return one claim per path.

    The plan protects what the session can actually damage. A user namespace
    already makes ``/usr`` and ``/etc`` unwritable, because root inside it is
    the host user and the host user does not own them. What is reachable is the
    home directory: the sibling worktrees, the frozen headers, and credentials.
    """
    claims: dict[str, str] = {}
    parent = worktree.parent
    if parent != Path(parent.root) and parent.is_dir():
        claims[str(parent)] = "ro" if not bind_read_only(parent) else "failed"
        # The worktree comes back to read-write inside its read-only parent.
        claims[str(worktree)] = (
            "rw" if not bind_read_write(worktree) else "failed")
    headers = worktree / FROZEN_HEADER_DIR
    if architect:
        claims[str(headers)] = "rw"
    else:
        # One mount for the directory, so a new header cannot be created either.
        claims[str(headers)] = "ro" if not bind_read_only(headers) else "failed"
    empty = state / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    hidden = []
    for name in HIDDEN_PATHS:
        path = Path(name).expanduser()
        if not path.exists():
            continue
        if hide_path(path, empty):
            hidden.append(f"{path}: failed")
        else:
            hidden.append(f"{path}: hidden")
    claims["hidden"] = ", ".join(hidden) if hidden else "none"
    return claims


def _append_fails(path: Path) -> bool:
    """True when appending to ``path`` is refused. It writes no bytes."""
    try:
        with open(path, "a", encoding="utf-8"):
            return False
    except OSError:
        return True


def _write_fails(path: Path) -> bool:
    """True when a write to ``path`` is refused."""
    try:
        with open(path / ".nanochat-probe", "w", encoding="utf-8"):
            return False
    except OSError:
        return True


def verify_confinement(worktree: Path, claims: dict[str, str]) -> list[str]:
    """Read every claim back. Return the failures.

    A claim that is not applied is a failure, as for every other budget. The
    test is an attempt and not a table lookup: the kernel decides.
    """
    failures = []
    for path, claim in claims.items():
        if path in ("hidden",) or claim in ("rw", "none"):
            continue
        target = Path(path)
        if claim == "failed":
            failures.append(f"{path}: the mount was refused")
            continue
        if not target.exists():
            failures.append(f"{path}: missing after the mount")
            continue
        if claim == "ro":
            refused = (_append_fails(target) if target.is_file()
                       else _write_fails(target))
            if not refused:
                failures.append(f"{path}: still writable")
    hidden = claims.get("hidden", "none")
    if hidden not in ("none", ""):
        for entry in hidden.split(", "):
            if entry.endswith(": failed"):
                failures.append(f"{entry}: the hide was refused")
                continue
            path = Path(entry.rsplit(": hidden", 1)[0])
            if path.is_dir() and any(path.iterdir()):
                failures.append(f"{path}: still readable")
            if path.is_file() and path.stat().st_size:
                failures.append(f"{path}: still readable")
    probe = worktree / ".nanochat-probe"
    try:
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError:
        failures.append(f"{worktree}: no write access")
    return failures


def write_ready(path: Path, net: str, extra: dict | None = None) -> dict:
    """Write the ready file, which proves the namespaces exist."""
    ready = {
        "pid": os.getpid(),
        "net": net,
        "user_ns": namespace_id("user"),
        "net_ns": namespace_id("net"),
        "mnt_ns": namespace_id("mnt"),
        "interfaces": interface_names(),
        "socket": "",
    }
    ready.update(extra or {})
    temporary = path.with_suffix(".new")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(ready, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return ready


def hold(argv: list[str]) -> int:
    """The ``hold`` role: prepare the namespaces, confine, then serve.

    ``argv`` is ``SOCKET READY NET WORKTREE [ARCHITECT]``. The confinement runs
    after the namespaces exist, because a mount needs its own mount namespace.
    """
    socket_path = Path(argv[0]) if argv else None
    ready_path = Path(argv[1]) if len(argv) > 1 else None
    net = argv[2] if len(argv) > 2 else "host"
    worktree = Path(argv[3]).resolve() if len(argv) > 3 else None
    architect = len(argv) > 4 and argv[4] == "architect"
    if socket_path is None or ready_path is None:
        print("supervisor: hold needs a socket path and a ready path",
              file=sys.stderr)
        return 2
    if worktree is None or not worktree.is_dir():
        # The writable set is relative to the worktree, so a missing one would
        # silently confine nothing. Refuse instead.
        print(f"supervisor: hold needs a worktree directory, not {worktree}",
              file=sys.stderr)
        return 2
    if net == "deny":
        failure = set_loopback_up()
        if failure:
            print(f"supervisor: net: {failure}", file=sys.stderr)
            return 1
        names = interface_names()
        if names != [LOOPBACK]:
            print("supervisor: net: expected only loopback, found "
                  f"{', '.join(names) or 'nothing'}", file=sys.stderr)
            return 1
    claims = confine(worktree, ready_path.parent, architect)
    failures = verify_confinement(worktree, claims)
    if failures:
        print("supervisor: the writable set is not as claimed:", file=sys.stderr)
        for failure in failures:
            print(f"supervisor:   {failure}", file=sys.stderr)
        return 1
    write_ready(ready_path, net, {"socket": str(socket_path),
                                  "worktree": str(worktree),
                                  "confinement": claims})
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    return serve(socket_path)


def send(argv: list[str]) -> int:
    """The ``send`` role: one command to one session supervisor."""
    socket_path = Path(argv[0]) if argv else None
    cwd = argv[1] if len(argv) > 1 else ""
    command = argv[2:]
    if command[:1] == ["--"]:
        command = command[1:]
    if socket_path is None or not command:
        print("supervisor: send needs a socket path and a command",
              file=sys.stderr)
        return 2
    request = {"argv": command, "cwd": cwd, "env": dict(os.environ)}
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(socket_path))
    except OSError as failure:
        print(f"supervisor: cannot reach {socket_path}: {failure.strerror}",
              file=sys.stderr)
        return SPAWN_FAILURE
    client.sendall((json.dumps(request) + "\n").encode("utf-8"))
    handle = client.makefile("rb")
    code = SPAWN_FAILURE
    try:
        for line in handle:
            try:
                frame = json.loads(line.decode("utf-8"))
            except ValueError:
                break
            if "exit" in frame:
                code = int(frame["exit"])
                break
            if "error" in frame:
                print(f"supervisor: {frame['error']}", file=sys.stderr)
            elif frame.get("stream") == "out":
                sys.stdout.write(frame.get("data", ""))
                sys.stdout.flush()
            else:
                sys.stderr.write(frame.get("data", ""))
                sys.stderr.flush()
    finally:
        client.close()
    return code


def main(argv: list[str] | None = None) -> int:
    """Dispatch one role."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["hold"]:
        return hold(arguments[1:])
    if arguments[:1] == ["send"]:
        return send(arguments[1:])
    print("supervisor: roles are hold and send", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
