"""The ``tools/nanochat`` command line.

``tools/nanochat`` is the Python entry point that starts this module. The
command table,
the argument assembly, the Bazel wiring, the build lock, and the sandbox and
queue routing live here. The resource launcher is
``python/nanochat_cpp/sandbox.py``. See ``docs/python.md``
and ``docs/sandbox.md``.

The observable contract is frozen (decision 4 of the plan): the environment
variable names, the two log paths and their line formats, the exit codes, the
two ``--run_under`` strings, the ``NANOCHAT_SANDBOX`` sentinel, the
``doctor --json`` keys, the fixture subcommand names, and the fixture binary
layouts. ``python/tests/contract_test.py`` pins them.

The ``doctor --json`` keys are ``repo``, ``revision``, ``branch``, ``nvcc``,
``bazel``, ``gpu``, ``gpu_procs``, ``sandbox``, ``sandbox_backend``,
``gpu_queue``, ``sim``, and ``plan``, in that order. The tuple is append-only:
a key is added at the end, never renamed, reordered, or dropped, because both
the text printer and the typed client read it in order.

The ``plan`` value is the device-and-graph plan that ``//src:plan_main`` prints
for the probe spec (``docs/model-and-device-bringup.md`` section 9 rule 6 and
section 9.1 rule 3): the user sees which implementation each composition
resolved to on this host, before a model loads, without reading source. It is
an object with ``available``, ``reason``, ``ok``, ``spec``, ``graph``,
``device_index``, ``choices``, ``gaps``, and ``why_not``. ``choices`` holds one
entry per composition the graph runs - the chosen ``implementation``, whether a
``fused`` family won, and the ``declined`` fallbacks with the reason each one
lost - and ``gaps`` lists the compositions that could not be placed at all when
``ok`` is false. Collection is non-fatal: on a partial build, or when the probe
spec is missing, ``available`` is false and ``reason`` says why, because a
``doctor`` run must not fail on a partial checkout. ``nanochat_cpp.plan`` is
the training-hyperparameter planner, unrelated to this object.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# `tools/nanochat` runs this file directly, so `python/` is not on the path.
# A `-m nanochat_cpp.cli` run already has it.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from nanochat_cpp import _entry, gpu_queue, lint, prune, sandbox, session  # noqa: E402

__all__ = ["main"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TOOLS = REPO_ROOT / "tools"
ENTRY = TOOLS / "nanochat"
SANDBOX = HERE / "sandbox.py"
BAZEL_BIN = REPO_ROOT / "bazel-bin"

BUILD_LOCK = os.environ.get("NANOCHAT_BUILD_LOCK", "/tmp/nanochat-build.lock")

#: The commands that `--dry-run` covers.
DRY_RUN_COMMANDS = ("build", "test", "check", "run", "simulate")

#: The `doctor --json` keys, in output order. Append-only.
DOCTOR_KEYS = ("repo", "revision", "branch", "nvcc", "bazel", "gpu",
               "gpu_procs", "sandbox", "sandbox_backend", "gpu_queue", "sim",
               "plan")

#: The model package the doctor's plan probe asks about. A package name rather
#: than a path, so `plan_main` resolves it the same way `Model::Create` does.
DOCTOR_PLAN_SPEC = "nanochat-default"

#: The graph the doctor's plan probe resolves. Training is the graph a bring-up
#: check looks at first, and it is `plan_main`'s own default.
DOCTOR_PLAN_GRAPH = "train"

#: The sandbox profiles that `run` sends to the queue instead of the launcher.
DIRECT_PROFILE = "t0-cpu"


class Failure(Exception):
    """A command-line error. ``main`` prints it and returns 2."""


def die(message: str) -> None:
    """Raise the command-line error that ``main`` reports."""
    raise Failure(message)


def in_sandbox() -> bool:
    """True inside a sandbox scope or a queued job."""
    return bool(os.environ.get("NANOCHAT_SANDBOX") or
                os.environ.get("NANOCHAT_GPU_QUEUED"))


def sandbox_backend() -> str:
    """The configured backend name, or ``auto`` when unset."""
    return os.environ.get("NANOCHAT_SANDBOX_BACKEND", "auto")


def find_bazel() -> str:
    """The Bazel launcher. Prefer Bazelisk so `.bazelversion` is honored."""
    path = _bazel_or_none()
    if path:
        return path
    die("neither bazelisk nor bazel found on PATH")
    raise AssertionError("unreachable")


def _bazel_or_none() -> str | None:
    """The Bazel launcher, or ``None``. A probe must not fail the command."""
    for name in ("bazelisk", "bazel"):
        path = shutil.which(name)
        if path:
            return path
    return None


@contextlib.contextmanager
def build_lock():
    """Hold the build lock.

    The lock lives on a descriptor that Python opens close-on-exec, so the
    detached Bazel server cannot inherit it. That inheritance deadlocked the
    old GPU broker.
    """
    with open(BUILD_LOCK, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def with_build_lock(argv: list[str], stdout=None, stderr=None) -> int:
    """Run ``argv`` under the build lock."""
    with build_lock():
        return subprocess.run(argv, check=False, stdout=stdout,
                              stderr=stderr).returncode


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run ``argv`` and never raise on a nonzero status."""
    return subprocess.run(argv, check=False, **kwargs)


def _capture(argv: list[str]) -> str:
    """The trimmed stdout of ``argv``, or ``""`` when it fails."""
    result = _run(argv, capture_output=True, text=True)
    return result.stdout.strip()


def exec_replace(argv: list[str]) -> None:
    """Replace this process with ``argv``."""
    os.execvp(argv[0], argv)


class Context:
    """The shared state of one command run."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def plan(self, layer: str, *argv: str) -> None:
        """Print one ``--dry-run`` line: the layer and the argument list."""
        print(json.dumps({"layer": layer, "argv": [str(a) for a in argv]}))

    def bazel(self, argv: list[str]) -> int:
        """Run Bazel under the lock, or print the plan line in dry-run mode."""
        if self.dry_run:
            self.plan("bazel", "bazel", *argv)
            return 0
        return with_build_lock([find_bazel(), *argv])

    def sandbox(self, profile: str, cmd: list[str]) -> int:
        """Run ``cmd`` under the resource launcher."""
        if self.dry_run:
            self.plan("sandbox", str(SANDBOX), "--profile", profile, "--",
                      *cmd)
            return 0
        return exec_replace([str(SANDBOX), "--profile", profile, "--", *cmd])

    def session_scope(self, name: str, cmd: list[str], net: str) -> int:
        """Run ``cmd`` inside one agent session slice and network."""
        argv = [str(SANDBOX), "--profile", session.BUILD_PROFILE,
                "--session", name, "--net", net, "--", *cmd]
        if self.dry_run:
            self.plan("session", *argv)
            return 0
        return exec_replace(argv)

    def submit(self, profile: str, cmd: list[str]) -> int:
        """Submit one GPU job, or run the command when already inside one."""
        if self.dry_run:
            self.plan("gpuq", str(ENTRY), "gpuq", "submit", "--profile",
                      profile, "--", *cmd)
            return 0
        if in_sandbox():
            return exec_replace(cmd)
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *cmd])


# -- Commands -------------------------------------------------------------

def cmd_help(argv: list[str], ctx: Context) -> int:
    sys.stdout.write(_entry.usage())
    return 0


#: The cache flags a build may state itself. An explicit flag wins, because the
#: caller named it and the appended flags would otherwise be the last word.
_CACHE_FLAGS = ("--disk_cache", "--repository_cache", "--remote_cache")


def _build_argv(argv: list[str]) -> list[str]:
    """The Bazel argument list for ``build``, with the cache flags appended.

    ``NANOCHAT_BAZEL_CACHE`` reached the Python API and the Modal image and not
    a hand-run build, because this command forwards its arguments straight to
    Bazel. Appending the flags from ``nanochat_cpp._build`` gives the flag line
    one home and the variable one meaning.
    """
    from nanochat_cpp import _build  # noqa: PLC0415 - the build path only

    arguments = list(argv) or ["//..."]
    if not [part for part in arguments
            if part.split("=", 1)[0] in _CACHE_FLAGS]:
        arguments = [*arguments, *_build.cache_flags()]
    return ["build", *arguments]


def cmd_build(argv: list[str], ctx: Context) -> int:
    """Build under the lock, inside the session boundary when there is one.

    The Bazel client starts a server and exits, and the server keeps compiling.
    A cgroup keeps that server because membership follows the fork, so the
    build enters the session. A server outside every cgroup exhausted host
    memory on 2026-10-01; see ``docs/sandbox.md`` ``### Builds``.
    """
    bazel_argv = _build_argv(argv)
    net, bazel_argv = _take_net_flag(bazel_argv)
    record = _active_session()
    if record is None:
        if net is not None:
            die("--net needs a session; run: tools/nanochat session start")
        return ctx.bazel(bazel_argv)
    return _bazel_in_boundary(bazel_argv, record, ctx,
                              net or record.get("net", session.NET_DENY),
                              "the build had no network")


def _active_session() -> dict | None:
    """The session a command belongs to, or ``None``.

    A command already inside a sandbox, a host with no cgroup v2, and a host with
    no systemd user manager all skip the session. A portable host would
    otherwise fail the placement read-back for a scope it cannot create.
    """
    if in_sandbox() or sandbox_backend() == "none":
        return None
    if not sandbox.systemd_available():
        return None
    record = session.active()
    if record is None or not record.get("id"):
        return None
    return record


def _bazel_in_boundary(bazel_argv: list[str], record: dict, ctx: Context,
                       net: str, hint: str) -> int:
    """Pick the cheapest boundary that honours ``net``, then check it."""
    ready = None
    if net == session.NET_DENY:
        ready = session.read_ready(record["id"])
    if ready is not None:
        return _bazel_supervised(bazel_argv, record, ctx, net, ready, hint)
    return _bazel_in_session(bazel_argv, record, ctx, net, hint)


def _bazel_supervised(bazel_argv: list[str], record: dict, ctx: Context,
                      net: str, ready: dict, hint: str) -> int:
    """Run Bazel as a child of the session supervisor.

    The supervisor owns one network namespace for the whole session, so the
    Bazel server stays reachable between commands and no run pays a server
    start. That is the whole reason the supervisor exists.
    """
    name = record["id"]
    want_ns = ready.get("net_ns", "")
    if ctx.dry_run:
        ctx.plan("bazel", "bazel", *bazel_argv)
        ctx.plan("supervisor", str(session.SUPERVISOR_PROGRAM), "send",
                 str(session.socket_path(name)), "bazel", *bazel_argv)
        return 0
    _reconcile_server(name, net, want_ns)
    with build_lock():
        try:
            code = session.run_via_supervisor(name, [find_bazel(), *bazel_argv])
        except session.SessionError as failure:
            die(str(failure))
    if code != 0:
        if net == session.NET_DENY:
            print(f"session: {hint}; one run with --net host fetches the "
                  "dependency")
        return code
    return session.check_placement(name, net, want_ns)


def _bazel_in_session(bazel_argv: list[str], record: dict, ctx: Context,
                      net: str, hint: str) -> int:
    """Run one Bazel command in the session slice, on the host network.

    This is the path for ``--net host`` and for a session whose supervisor is
    down. A denied command always belongs to the supervisor, because only a
    long-lived process can own a namespace that outlives a command.
    """
    name = record["id"]
    launcher = [str(SANDBOX), "--profile", session.BUILD_PROFILE,
                "--session", name, "--", find_bazel(), *bazel_argv]
    if ctx.dry_run:
        ctx.plan("bazel", "bazel", *bazel_argv)
        ctx.plan("session", *launcher)
        return 0
    _reconcile_server(name, net)
    code = with_build_lock(launcher)
    if code != 0:
        if net == session.NET_DENY:
            print(f"session: {hint}; one run with --net host fetches the "
                  "dependency")
        return code
    return session.check_placement(name, net)


def _reconcile_server(name: str, net: str, want_ns: str = "") -> None:
    """Stop a server that cannot serve this command, before the command starts.

    A server outside the session slice, or in a network namespace this command
    cannot reach, cannot serve it. Bazel would kill it and start another, but
    that recovery is a race, so the stop is explicit. The read-back then
    confirms a placement instead of rescuing one.
    """
    pid = session.server_pid()
    if pid is None:
        return
    inside, where = session.placement_of(pid, name)
    good = bool(inside)
    if good and net == session.NET_DENY:
        if want_ns:
            good = session.net_namespace(pid) == want_ns
        else:
            good = session.net_is_private(pid) is True
    elif good:
        good = session.net_is_private(pid) is False
    if good:
        return
    actual = "private" if session.net_is_private(pid) else "host"
    reason = ("outside the session slice" if not inside
              else f"on the {actual} network")
    print(f"session: stopping the build server ({reason}): pid={pid} "
          f"cgroup={where or 'unknown'}")
    try:
        _run([find_bazel(), "shutdown"], capture_output=True, text=True,
             timeout=120)
    except subprocess.SubprocessError:
        die("cannot stop the stale build server; run: tools/nanochat shutdown")


def _test_argv(argv: list[str]) -> list[str]:
    """The Bazel argument list for ``test``."""
    gpu = False
    targets: list[str] = []
    configs: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--gpu":
            gpu = True
            index += 1
        elif argument == "--config":
            if index + 1 >= len(argv):
                die("--config needs a name")
            configs += ["--config", argv[index + 1]]
            index += 2
        elif argument.startswith("--config="):
            configs.append(argument)
            index += 1
        else:
            targets.append(argument)
            index += 1
    if not targets:
        targets = ["//..."]
    if gpu:
        profile, tag_filter = "t1-gpu", "gpu"
        run_under = f"{ENTRY} gpuq submit --profile={profile} --"
    else:
        profile, tag_filter = "t0-cpu", "-gpu"
        run_under = f"{SANDBOX} --profile={profile} --"
    arguments = [*targets, f"--test_tag_filters={tag_filter}",
                 f"--run_under={run_under}", "--spawn_strategy=local",
                 *configs]
    if gpu:
        # A GPU run selects the CUDA backend, so the shared workflow
        # (//src:model) links the kernels under test. One action at a time.
        arguments += ["--config=cuda", "--local_test_jobs=1"]
    return ["test", *arguments]


_NET_FLAG = "--net"


def _take_net_flag(argv: list[str]) -> tuple[str | None, list[str]]:
    """Remove ``--net host|deny`` from an argument list."""
    net = None
    kept: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == _NET_FLAG:
            if index + 1 >= len(argv) or argv[index + 1] not in \
                    session.NET_MODES:
                die("--net needs host or deny")
            net = argv[index + 1]
            index += 2
        elif argument.startswith(f"{_NET_FLAG}="):
            value = argument.split("=", 1)[1]
            if value not in session.NET_MODES:
                die("--net needs host or deny")
            net = value
            index += 1
        else:
            kept.append(argument)
            index += 1
    return net, kept


def cmd_test(argv: list[str], ctx: Context) -> int:
    """Run the tests, inside the session slice when there is one.

    The build phase of a test run compiles, so it belongs in the same memory
    boundary as a build. The network stays on the host unless ``--net deny`` is
    named, because a denied run enters a fresh network namespace and the Bazel
    server then restarts on every invocation. Measured on 2026-11-11: a warm
    suite costs 0.6 s with the server alive and 21 s with a restart per run.
    """
    net, argv = _take_net_flag(argv)
    bazel_argv = _test_argv(argv)
    record = _active_session()
    if record is None:
        if net is not None:
            die("--net needs a session; run: tools/nanochat session start")
        return ctx.bazel(bazel_argv)
    if net is None:
        # A supervisor holds one namespace for the session, so a denied run
        # reuses its server and costs nothing. Without one, a denied run would
        # restart the server every time, so the network stays on the host.
        if session.read_ready(record["id"]) is not None:
            net = record.get("net", session.NET_DENY)
        else:
            net = session.NET_HOST
            print("session: tests have the host network, because no "
                  "supervisor is up; a hermetic run needs: "
                  "tools/nanochat session start")
    return _bazel_in_boundary(bazel_argv, record, ctx, net,
                              "the test build had no network")


def cmd_check(argv: list[str], ctx: Context) -> int:
    gpu = argv[:1] == ["--gpu"]
    if ctx.dry_run:
        ctx.bazel(_build_argv([]))
        ctx.bazel(_test_argv([]))
        if gpu:
            ctx.bazel(_test_argv(["--gpu"]))
        return 0
    children = [["build"], ["test"]]
    if gpu:
        children.append(["test", "--gpu"])
    for child in children:
        code = _run([str(ENTRY), *child]).returncode
        if code != 0:
            return code
    return 0


def cmd_lint(argv: list[str], ctx: Context) -> int:
    return lint.main(argv)


def cmd_shutdown(argv: list[str], ctx: Context) -> int:
    return exec_replace([find_bazel(), "shutdown"])


def cmd_prune(argv: list[str], ctx: Context) -> int:
    return prune.main(argv)


def cmd_gpuq(argv: list[str], ctx: Context) -> int:
    return gpu_queue.main(argv)


def cmd_session(argv: list[str], ctx: Context) -> int:
    """The agent session scope: start, status, stop."""
    if not argv:
        die("session needs an action: start, status, or stop")
    action, rest = argv[0], argv[1:]
    if action == "start":
        return _session_start(rest, ctx)
    if action == "status":
        return _session_status(rest)
    if action == "stop":
        return _session_stop(rest)
    die(f"unknown session action '{action}' (try: start, status, stop)")
    raise AssertionError("unreachable")


def _parse_session_args(
        rest: list[str]) -> tuple[str, str, bool, str, list[str]]:
    """The ``--id``, ``--agent``, ``--json``, ``--net`` options and the command."""
    identifier, agent, net, as_json, architect = "", "", session.NET_DENY, False, False
    index = 0
    while index < len(rest):
        argument = rest[index]
        if argument == "--":
            index += 1
            break
        if argument in ("--id", "--agent", "--net"):
            if index + 1 >= len(rest) or not rest[index + 1]:
                die(f"{argument} needs a value")
            value = rest[index + 1]
            if argument == "--id":
                identifier = value
            elif argument == "--agent":
                agent = value
            else:
                if value not in session.NET_MODES:
                    die("--net needs host or deny")
                net = value
            index += 2
        elif argument.startswith("--id="):
            identifier = argument.split("=", 1)[1]
            index += 1
        elif argument.startswith("--agent="):
            agent = argument.split("=", 1)[1]
            index += 1
        elif argument.startswith("--net="):
            net = argument.split("=", 1)[1]
            if net not in session.NET_MODES:
                die("--net needs host or deny")
            index += 1
        elif argument == "--architect":
            architect = True
            index += 1
        elif argument == "--json":
            as_json = True
            index += 1
        else:
            die(f"unknown session option '{argument}'")
    return identifier, agent, as_json, net, architect, rest[index:]


def _session_name(identifier: str) -> str:
    """The session name: the option, or the worktree directory name."""
    try:
        if identifier:
            return session.normalize_id(identifier)
        return session.default_id(REPO_ROOT)
    except session.SessionError as error:
        die(str(error))
        raise AssertionError("unreachable")


def _session_start(rest: list[str], ctx: Context) -> int:
    """Write the record, then enter the slice with the command, if any."""
    identifier, agent, _, net, architect, command = _parse_session_args(rest)
    name = _session_name(identifier)
    if sandbox_backend() == "none":
        die("session needs cgroup v2; NANOCHAT_SANDBOX_BACKEND=none "
            "turns the session scope off")
    try:
        record = session.start_record(name, REPO_ROOT, agent=agent, net=net)
    except OSError as error:
        die(f"cannot write the session record: {error}")
    print(f"session: {record['id']} slice={record['slice']} net={record['net']}")
    print(f"session: root={record['root']}")
    if record["net"] == session.NET_DENY:
        ready = _start_supervisor(record, _profile_values(),
                                  architect=architect)
        print(f"session: supervisor pid={ready['pid']} "
              f"net_ns={ready.get('net_ns', '?')}")
        print("session: one namespace for the session, so a denied build "
              "reuses its server")
    if not command:
        print("session: a build in this worktree now runs inside the boundary")
        if record["net"] == session.NET_DENY:
            print("session: the build has no network; use `build --net host` "
                  "for a fetch")
        return 0
    if record["net"] == session.NET_DENY:
        code = 0
        try:
            with build_lock():
                code = session.run_via_supervisor(name, command, REPO_ROOT)
        except session.SessionError as failure:
            die(str(failure))
        return code
    return ctx.session_scope(name, command, net)


def _profile_values() -> dict:
    """The resource values of the session build profile."""
    try:
        profiles = sandbox.load_profiles()
        return sandbox.profile_values(profiles, session.BUILD_PROFILE)
    except (OSError, ValueError, sandbox.UnknownProfile) as failure:
        die(f"cannot read the build profile: {failure}")
        raise AssertionError("unreachable")


def _start_supervisor(record: dict, values: dict,
                      architect: bool = False) -> dict:
    """Start the session supervisor, or report why it did not start."""
    try:
        ready = session.start_supervisor(record["id"], values,
                                         record.get("net", session.NET_DENY),
                                         REPO_ROOT, architect)
    except session.SessionError as failure:
        die(str(failure))
        raise AssertionError("unreachable")
    if ready is None:
        die("the session supervisor did not start")
    return ready


def _session_status(rest: list[str]) -> int:
    """Report the record, the slice, and the build server placement."""
    identifier, _, as_json, _, _, _ = _parse_session_args(rest)
    name = _session_name(identifier)
    if session.read_record(name) is None:
        print(f"session: no record for '{name}' in {session.state_dir()}")
        print("session: run: tools/nanochat session start")
        return 1
    report = session.status(name)
    if as_json:
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    print(f"session: id={report['id']} state={report['state']} "
          f"net={report['net']}")
    print(f"session: slice={report['slice']}")
    if not report["cgroup"]:
        print("session: no command runs in the slice; a build creates it")
        return 0
    print(f"session: cgroup={report['cgroup']}")
    if report["supervisor"] == "up":
        print(f"session: supervisor pid={report['supervisor_pid']} "
              f"net_ns={report['net_ns']}")
        for path, claim in report["confinement"].items():
            print(f"session: confine {path} = {claim}")
    else:
        print("session: supervisor down; a build starts a fresh namespace")
    print(f"session: memory.current={report['memory_current']} "
          f"memory.max={report['memory_max']} "
          f"memory.peak={report['memory_peak']}")
    print(f"session: pids.current={report['pids_current']}")
    for scope in report["scopes"]:
        print(f"session: scope {scope['scope']} "
              f"memory.current={scope['memory_current']} "
              f"memory.max={scope['memory_max']} pids.max={scope['pids_max']}")
    print(f"session: bazel server {report['server']}")
    return 0


def _session_stop(rest: list[str]) -> int:
    """Stop the slice, which stops every command and server inside it."""
    identifier, _, _, _, _, _ = _parse_session_args(rest)
    name = _session_name(identifier)
    if session.read_record(name) is None:
        print(f"session: no record for '{name}' in {session.state_dir()}")
        return 1
    return session.stop(name)


def cmd_run(argv: list[str], ctx: Context) -> int:
    if not argv:
        die("run needs a profile")
    profile, rest = argv[0], argv[1:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if not rest:
        die(f"no command given; use: tools/nanochat run {profile} -- <cmd...>")
    if profile == DIRECT_PROFILE:
        return ctx.sandbox(profile, rest)
    return ctx.submit(profile, rest)


def _alias(profile: str):
    """A command that submits its argument list under one profile."""

    def command(argv: list[str], ctx: Context) -> int:
        rest = argv[1:] if argv[:1] == ["--"] else argv
        if not rest:
            die("no command given; use: tools/nanochat -- <cmd...>")
        return ctx.submit(profile, rest)

    return command


def _int(value: str, message: str) -> int:
    """Parse a base-10 integer, or report a command-line error."""
    try:
        return int(value, 10)
    except ValueError:
        die(message)
        raise AssertionError("unreachable")


def cmd_dist(argv: list[str], ctx: Context) -> int:
    nproc = 2
    device = ""
    device_given = False
    profile = "train"
    master = "127.0.0.1"
    port = 29500
    log_dir = ""
    as_json = False
    command: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--nproc-per-node", "--device", "--profile", "--master",
                        "--port", "--log-dir"):
            if index + 1 >= len(argv):
                die(f"dist: {argument} needs a value")
            value = argv[index + 1]
            if argument == "--nproc-per-node":
                nproc = _int(value, f"dist: --nproc-per-node needs a number, "
                                    f"not '{value}'")
            elif argument == "--device":
                device, device_given = value, True
            elif argument == "--profile":
                profile = value
            elif argument == "--master":
                master = value
            elif argument == "--port":
                port = _int(value, f"dist: --port needs a number, "
                                   f"not '{value}'")
            else:
                log_dir = value
            index += 2
        elif argument == "--architect":
            architect = True
            index += 1
        elif argument == "--json":
            as_json = True
            index += 1
        elif argument == "--":
            command = argv[index + 1:]
            break
        else:
            die(f"dist: unknown option {argument}")
    if not command:
        die("dist: no command given; use: tools/nanochat dist -- <cmd...>")
    if nproc < 1:
        die("dist: --nproc-per-node must be >= 1")
    if not device and os.environ.get("NANOCHAT_GPU_DEVICES"):
        pool = os.environ["NANOCHAT_GPU_DEVICES"].split(",")
        if len(pool) < nproc:
            die(f"dist: the worker assigned {len(pool)} device(s), need {nproc}")
        device = ",".join(pool[:nproc])
    if not device:
        device = ",".join(str(rank) for rank in range(nproc))
    devices = device.split(",")
    if len(devices) != nproc:
        die(f"dist: --device has {len(devices)} entries, need {nproc}")
    if not log_dir:
        log_dir = _tempdir("nanochat-dist.")
    os.makedirs(log_dir, exist_ok=True)

    # One queue job for the whole rank group. Kaggle has no queue, so skip
    # when the backend is `none` or the job is already queued.
    if not in_sandbox() and sandbox_backend() != "none":
        reexec = [str(ENTRY), "dist", "--nproc-per-node", str(nproc),
                  "--profile", profile, "--master", master,
                  "--port", str(port), "--log-dir", log_dir]
        if device_given:
            reexec += ["--device", device]
        if as_json:
            reexec.append("--json")
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *reexec, "--", *command])

    # Inside a queued job the outer scope applies the budget and the affinity.
    launcher: list[str] = []
    if not os.environ.get("NANOCHAT_GPU_QUEUED"):
        launcher = [str(SANDBOX), "--profile", profile, "--"]

    start = time.monotonic()
    processes = []
    stdio_paths = []
    for rank in range(nproc):
        expanded = [value.replace("{rank}", str(rank)) for value in command]
        stdio = Path(log_dir) / f"rank{rank}.stdio.log"
        stdio_paths.append(str(stdio))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devices[rank]
        env["NANOCHAT_WORLD_SIZE"] = str(nproc)
        env["NANOCHAT_RANK"] = str(rank)
        env["NANOCHAT_LOCAL_RANK"] = str(rank)
        env["NANOCHAT_MASTER_ADDR"] = master
        env["NANOCHAT_MASTER_PORT"] = str(port)
        handle = open(stdio, "w", encoding="utf-8")
        processes.append(subprocess.Popen([*launcher, *expanded], env=env,
                                          stdout=handle, stderr=handle))
        # Rank 0 binds the rendezvous port; give it a moment before the peers
        # start so the connect does not race the bind.
        if rank == 0 and nproc > 1:
            time.sleep(0.5)

    codes = [process.wait() for process in processes]
    seconds = f"{time.monotonic() - start:.3f}"
    returncode = next((code for code in codes if code != 0), 0)

    if as_json:
        ranks = ",".join(
            json.dumps({"rank": rank, "returncode": codes[rank],
                        "stdio": stdio_paths[rank], "seconds": float(seconds)},
                       separators=(",", ":"))
            for rank in range(nproc))
        print(f'{{"ranks":[{ranks}]}}')
    else:
        for rank in range(nproc):
            print(f"dist: rank {rank} exit {codes[rank]} "
                  f"(stdio {stdio_paths[rank]})")
    return returncode


def cmd_client(argv: list[str], ctx: Context) -> int:
    device = ""
    device_given = False
    profile = "train"
    command: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--device", "--profile"):
            if index + 1 >= len(argv):
                die(f"client: {argument} needs a value")
            if argument == "--device":
                device, device_given = argv[index + 1], True
            else:
                profile = argv[index + 1]
            index += 2
        elif argument.startswith("--device="):
            device, device_given = argument[len("--device="):], True
            index += 1
        elif argument.startswith("--profile="):
            profile = argument[len("--profile="):]
            index += 1
        elif argument == "--":
            command = argv[index + 1:]
            break
        else:
            die(f"client: unknown option {argument}")
    if not command:
        die("client: no command given; use: tools/nanochat client -- <cmd...>")
    if not device:
        if os.environ.get("NANOCHAT_GPU_DEVICES"):
            device = os.environ["NANOCHAT_GPU_DEVICES"]
        elif shutil.which("nvidia-smi"):
            device = _nvidia_smi_indices()
        else:
            device = "0"
    # One queue job for the whole device set. The worker owns the devices, so
    # the client does not name one.
    if not in_sandbox() and sandbox_backend() != "none":
        reexec = [str(ENTRY), "client", "--profile", profile]
        if device_given:
            reexec += ["--device", device]
        return exec_replace([str(ENTRY), "gpuq", "submit", "--profile",
                             profile, "--", *reexec, "--", *command])
    if os.environ.get("NANOCHAT_GPU_QUEUED"):
        return exec_replace(command)
    return ctx.sandbox(profile, command)


def _nvidia_smi_indices() -> str:
    output = _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                  capture_output=True, text=True).stdout
    indices = ",".join(line.strip() for line in output.splitlines()
                       if line.strip())
    return indices or "0"


def cmd_profile(argv: list[str], ctx: Context) -> int:
    code = ctx.bazel(["build", "//backends/cuda/kernels:attention_benchmark"])
    if ctx.dry_run:
        return code
    if code != 0:
        return code
    binary = BAZEL_BIN / "backends" / "cuda" / "kernels" / "attention_benchmark"
    return exec_replace([str(ENTRY), "gpuq", "submit", "--profile", "t3-bench",
                         "--", str(binary), *argv])


def cmd_roofline(argv: list[str], ctx: Context) -> int:
    code = ctx.bazel(["build", "//src:nanochat_cli"])
    if code != 0:
        return code
    binary = BAZEL_BIN / "src" / "nanochat_cli"
    return exec_replace([str(SANDBOX), "--profile", "t0-cpu", "--",
                         str(binary), "roofline", *argv])


def cmd_attain(argv: list[str], ctx: Context) -> int:
    """Join a bound with a measurement (docs/performance-model.md section 10).

    The record is derived, not transcribed: the two reports the model and the
    benchmark already write carry the identity keys a join needs, so this
    command is the only place that puts them side by side.
    """
    from . import attain

    del ctx  # A join is arithmetic over two files; it needs no build.
    return attain.main(argv)


def cmd_devices(argv: list[str], ctx: Context) -> int:
    """Print the declared device names (docs/simulator.md section 7)."""
    del ctx
    names = declared_device_names()
    if argv[:1] == ["--json"]:
        print(json.dumps({"devices": names}))
    else:
        print("\n".join(names))
    return 0


def _tempdir(prefix: str) -> str:
    import tempfile
    return tempfile.mkdtemp(prefix=prefix, dir="/tmp")


def cmd_pgo(argv: list[str], ctx: Context) -> int:
    """The profile-guided loop (docs/performance.md).

    The loop is Python and the host tool is Python, so the command forwards to
    the module instead of starting a process. The Modal SDK is the one thing the
    loop needs that this tool does not, and ``NANOCHAT_MODAL`` names its
    executable when the host keeps it outside ``PATH``.
    """
    from . import pgo
    return pgo.main(argv)


def cmd_doctor(argv: list[str], ctx: Context) -> int:
    as_json = argv[:1] == ["--json"]
    values = _doctor_values()
    if as_json:
        print(json.dumps({key: values[key] for key in DOCTOR_KEYS}))
        return 0
    _print_doctor(values)
    return 0


def _doctor_values() -> dict:
    """Collect the environment report. Every field is a string or an object."""
    revision = branch = ""
    if _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True).returncode == 0:
        revision = _capture(["git", "-C", str(REPO_ROOT), "rev-parse",
                             "--short", "HEAD"])
        branch = _capture(["git", "-C", str(REPO_ROOT), "branch",
                           "--show-current"])
    if shutil.which("nvcc"):
        found = re.search(r"release [0-9.]*",
                          _run(["nvcc", "--version"],
                               capture_output=True, text=True).stdout)
        nvcc = found.group(0) if found else ""
    else:
        nvcc = "NOT FOUND"
    bazel = ""
    if shutil.which("bazelisk"):
        bazel = _run(["bazelisk", "version"], capture_output=True,
                     text=True).stdout.splitlines()[:1]
        bazel = bazel[0] if bazel else ""
    if shutil.which("nvidia-smi"):
        gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,"
                    "memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader"], capture_output=True,
                   text=True).stdout.splitlines()[:1]
        gpu = gpu[0] if gpu else ""
        gpu_procs = _run(["nvidia-smi", "--query-compute-apps=pid,"
                          "process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.splitlines()
        gpu_procs = "; ".join(line.strip() for line in gpu_procs
                              if line.strip())
    else:
        gpu, gpu_procs = "nvidia-smi NOT FOUND", ""
    sandbox = " ".join(_run(
        ["systemctl", "--user", "show", "nanochat.slice", "-p",
         "MemoryCurrent", "-p", "MemoryMax", "-p", "CPUQuotaPerSecUSec"],
        capture_output=True, text=True).stdout.splitlines())
    sim = os.environ.get("NANOCHAT_SIM_PROFILE", "unset")
    if os.environ.get("NANOCHAT_SANDBOX_BACKEND"):
        backend = os.environ["NANOCHAT_SANDBOX_BACKEND"]
    elif _run(["systemd-run", "--user", "--scope", "--quiet", "true"],
              capture_output=True).returncode == 0:
        backend = "systemd"
    else:
        backend = "none (systemd unavailable)"
    return {
        "repo": str(REPO_ROOT),
        "revision": revision,
        "branch": branch,
        "nvcc": nvcc,
        "bazel": bazel,
        "gpu": gpu,
        "gpu_procs": gpu_procs,
        "sandbox": sandbox,
        "sandbox_backend": backend,
        "gpu_queue": _queue_status(),
        "sim": sim,
        "plan": _plan_status(),
    }


def _plan_status() -> dict:
    """The device-and-graph plan for the probe spec, or why there is none.

    `//src:plan_main` prints the plan that `nanochat::BuildPlan` resolves for
    `DOCTOR_PLAN_SPEC` and `DOCTOR_PLAN_GRAPH` (section 9.1 rule 3). The plan
    is pure, so the probe needs no device, no checkpoint, and no tokenizer.

    Non-fatal by design: `doctor` runs on a partial build, so a binary that is
    not built or a spec that is not readable reports the reason instead of
    failing the command. `available` is false in that case and `reason` says
    why; the rest of the object keeps its shape so a reader never has to guess
    which keys exist.

    The probe inherits the environment, so `NANOCHAT_SIM_PROFILE` makes it
    resolve against the simulated device, the same reading the ``sim`` key
    reports.
    """
    plan = {
        "available": False,
        "reason": "",
        "ok": False,
        "spec": DOCTOR_PLAN_SPEC,
        "graph": DOCTOR_PLAN_GRAPH,
        "device_index": 0,
        "choices": [],
        "gaps": [],
        "why_not": "",
    }
    binary = BAZEL_BIN / "src" / "plan_main"
    if not binary.exists():
        plan["reason"] = (f"{binary} is not built; "
                          "run: tools/nanochat build //src:plan_main")
        return plan
    # The working directory is the repository root so the binary finds
    # `models/<name>/spec.json` relative to it; a plan is rejected with exit 1
    # and still prints JSON, so only an unparseable report means "unavailable".
    try:
        result = _run([str(binary), f"--spec={DOCTOR_PLAN_SPEC}",
                       f"--graph={DOCTOR_PLAN_GRAPH}", "--json"],
                      capture_output=True, text=True, cwd=str(REPO_ROOT))
    except OSError as error:
        plan["reason"] = f"plan probe could not run {binary}: {error}"
        return plan
    try:
        payload = json.loads(result.stdout or "")
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        detail = " ".join(result.stderr.split())
        plan["reason"] = (f"plan probe failed: {detail}" if detail else
                          f"plan probe exited {result.returncode} "
                          "without a report")
        return plan
    device_index = payload.get("device_index")
    plan.update({
        "available": True,
        "ok": bool(payload.get("ok")),
        "spec": str(payload.get("spec") or DOCTOR_PLAN_SPEC),
        "graph": str(payload.get("graph") or DOCTOR_PLAN_GRAPH),
        "device_index": device_index if isinstance(device_index, int) else 0,
        "choices": _plan_choices(payload.get("choices")),
        "gaps": _plan_gaps(payload.get("gaps")),
        "why_not": str(payload.get("why_not") or ""),
    })
    return plan


def _plan_choices(choices) -> list[dict]:
    """The plan's per-composition decisions, with a fixed shape per entry."""
    out = []
    for choice in choices if isinstance(choices, list) else []:
        if not isinstance(choice, dict):
            continue
        declined = []
        for entry in choice.get("declined") or []:
            if not isinstance(entry, dict):
                continue
            declined.append({
                "implementation": str(entry.get("implementation") or ""),
                "reason": str(entry.get("reason") or ""),
            })
        out.append({"composition": str(choice.get("composition") or ""),
                    "implementation": str(choice.get("implementation") or ""),
                    "fused": bool(choice.get("fused")),
                    "declined": declined})
    return out


def _plan_gaps(gaps) -> list[dict]:
    """The compositions the plan could not place, and why."""
    out = []
    for gap in gaps if isinstance(gaps, list) else []:
        if not isinstance(gap, dict):
            continue
        out.append({"composition": str(gap.get("composition") or ""),
                    "reason": str(gap.get("reason") or "")})
    return out


def _queue_status() -> dict:
    """The queue status object, or an empty object when the query fails."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(io.StringIO()):
            gpu_queue.main(["status", "--json"])
    except SystemExit:
        return {}
    try:
        payload = json.loads(buffer.getvalue() or "{}")
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _print_doctor(values: dict) -> None:
    print("nanochat.cpp doctor")
    print("====================")
    print(f"repo:      {values['repo']}")
    if values["revision"]:
        print(f"revision:  {values['revision']} "
              f"({values['branch'] or 'detached'})")
    print(f"nvcc:      {values['nvcc']}")
    if values["bazel"]:
        print(f"bazel:     {values['bazel']}")
    print(f"gpu:       {values['gpu']}")
    if values["gpu_procs"]:
        print(f"gpu procs: {values['gpu_procs']}")
    if values["sandbox"]:
        print(f"sandbox:   {values['sandbox']}")
    print(f"sandbox backend: {values['sandbox_backend']}")
    print(f"sim profile: {values['sim']}")
    queue = values["gpu_queue"]
    if queue.get("alive"):
        print(f"gpu queue: worker {queue.get('worker_id', '')} "
              f"devices {queue.get('devices', '')} "
              f"queued {queue.get('queued', 0)} "
              f"running {queue.get('running_job') or '-'}")
    else:
        print("gpu queue: idle (no worker)")
    if queue:
        print(f"gpu logs:  {queue.get('log_files', 0)} files, "
              f"{queue.get('log_bytes', 0)} bytes")
    _print_plan(values["plan"])
    for log in ("/tmp/nanochat-gpu.log", "/tmp/nanochat-sandbox.log"):
        if not os.path.isfile(log):
            continue
        print(f"--- tail {log} ---")
        for line in _tail(log, 5):
            print(line)
    for binary in (BAZEL_BIN / "src" / "nanochat_cli",
                   BAZEL_BIN / "backends" / "cuda" / "kernels"
                   / "attention_benchmark"):
        if not binary.exists():
            continue
        stamp = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(binary.stat().st_mtime))
        print(f"built:     {binary} ({stamp})")


def _print_plan(plan: dict) -> None:
    """The plan section: the probe, the resolution, then one line per piece.

    It reads like `//src:plan_main`'s own text report, so a user who runs the
    binary by hand sees the same sentences `doctor` prints.
    """
    if not plan["available"]:
        print(f"plan:      unavailable ({plan['reason']})")
        return
    print(f"plan:      {plan['spec']}, graph {plan['graph']}, "
          f"device {plan['device_index']}: "
          f"{'ok' if plan['ok'] else 'rejected'}")
    if not plan["ok"]:
        # The rejection message already names the missing bits and the
        # consequence, so it is printed as the binary wrote it.
        if plan["why_not"]:
            print(plan["why_not"].rstrip())
        for gap in plan["gaps"]:
            print(f"           gap {gap['composition']}: {gap['reason']}")
        return
    for choice in plan["choices"]:
        fused = " (fused)" if choice["fused"] else ""
        print(f"           {choice['composition']}: "
              f"{choice['implementation']}{fused}")
        for declined in choice["declined"]:
            print(f"             declined {declined['implementation']}: "
                  f"{declined['reason']}")


def _tail(path: str, count: int) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle.readlines()[-count:]]


def declared_devices() -> dict:
    """The accelerator declarations, read from `backends/*/caps.json`.

    The declaration is the one source of the device set: a simulated run, a
    roofline report, and a benchmark all name a device this mapping holds.
    Reading it here keeps the command line from carrying a second copy that
    drifts from the declarations, which is what a hard-coded tuple did
    (docs/simulator.md section 7). The mapping keeps the file order, so the
    `devices` report is stable.
    """
    import json
    found: dict = {}
    for path in sorted(REPO_ROOT.glob("backends/*/caps.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                body = json.load(handle)
        except (OSError, ValueError):
            continue
        for device in body.get("devices", []):
            name = device.get("name")
            if isinstance(name, str) and name and device.get("is_device"):
                found[name] = device
    return found


def declared_device_names() -> list[str]:
    """The device names the tree declares, in declaration order."""
    return list(declared_devices())


SIM_SUITES = {
    "correctness": ["//tests:oracle_test", "//tests:train_parity_test",
                    "//tests:sim_numerics_test", "//tests:numerics_trace_test"],
    "fp8-fast": ["//tests:fp8_rounding_test", "//tests:fp8_gemm_test",
                 "//tests:fp8_graph_test", "//tests:device_profile_test"],
    "kernel": ["//tests:kernel_emu_test"],
    # The emulated hand GEMM of docs/tensor-core-gemm-plan.md. Its own target,
    # because the tile is bf16 by construction and //tests:kernel_emu_test is
    # fp32.
    "gemm": ["//tools/cuda_sim/emu:gemm_emu_test"],
    "profile": ["//tests:device_profile_test"],
    "all": ["//tests:oracle_test", "//tests:train_parity_test",
            "//tests:sim_numerics_test", "//tests:numerics_trace_test",
            "//tests:fp8_rounding_test", "//tests:fp8_gemm_test",
            "//tests:fp8_graph_test",
            "//tests:kernel_emu_test", "//tests:device_profile_test",
            "//tools/cuda_sim/emu:gemm_emu_test"],
}
SIM_GATE_TARGETS = ["//backends/cuda:all_kernels",
                    "//backends/cuda/collective:native_sync",
                    "//backends/cuda:cuda_runtime_gpu_test",
                    "//backends/cuda/kernels:precision_gpu_test",
                    "//backends/cuda/kernels:row_kernels_gpu_test",
                    "//backends/cuda/kernels:sequence_oracle_gpu_test"]


def _simulate_args(argv: list[str]) -> tuple[str, str, str, str]:
    mode, device, suite, precision = "run", "h100", "all", "fp32"
    for argument in argv:
        if argument.startswith("--device="):
            device = argument.split("=", 1)[1]
        elif argument.startswith("--suite="):
            suite = argument.split("=", 1)[1]
        elif argument.startswith("--mode="):
            mode = argument.split("=", 1)[1]
        elif argument.startswith("--precision="):
            precision = argument.split("=", 1)[1]
        else:
            die(f"simulate: unknown argument '{argument}'")
    return mode, device, suite, precision


def cmd_simulate(argv: list[str], ctx: Context) -> int:
    # `--name value` comes as separate tokens; fold them onto the `=` form.
    folded: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        match = re.fullmatch(r"--(device|suite|mode|precision)", argument)
        if match:
            if index + 1 >= len(argv):
                die(f"simulate: {argument} needs a value")
            folded.append(f"{argument}={argv[index + 1]}")
            index += 2
        else:
            folded.append(argument)
            index += 1
    mode, device, suite, precision = _simulate_args(folded)
    devices = declared_device_names()
    if device not in devices:
        die(f"simulate: unknown device '{device}' ({' '.join(devices)})")

    if mode == "compile":
        # The gate builds for the declared card, so the architecture and the
        # precision come from the declaration rather than from a named config.
        # The tree keeps one config per device family that a hand run uses, and
        # this mode must work for every declared device.
        declaration = declared_devices()[device]
        arch = f"sm_{declaration['compute_major']}{declaration['compute_minor']}"
        precision_name = declaration.get("reference_precision", "fp32")
        ptxas_log = os.environ.get(
            "NANOCHAT_SIM_PTXAS_LOG", f"/tmp/nanochat-sim-ptxas-{device}.log")
        gate_argv = ["build", "--config=cuda",
                     f"--@rules_cuda//cuda:archs={arch}",
                     f"--define=precision={precision_name}",
                     "--config=sim-gate", *SIM_GATE_TARGETS]
        if ctx.dry_run:
            ctx.plan("bazel", "bazel", *gate_argv)
            return 0
        return _simulate_compile(gate_argv, ptxas_log, device)
    if mode != "run":
        die(f"simulate: unknown mode '{mode}' (run compile)")

    if suite in ("api", "all"):
        code = _simulate_api(ctx, device)
        if code != 0:
            return code
    if suite == "api":
        return 0
    if suite not in SIM_SUITES:
        die(f"simulate: unknown suite '{suite}'")
    configs = ["--config=sim"]
    if precision == "bf16":
        configs.append("--config=bf16")
    elif precision == "fp8":
        configs.append("--config=fp8")
    elif precision != "fp32":
        die(f"simulate: unknown precision '{precision}' (fp32 bf16 fp8)")
    return ctx.bazel([
        "test", *SIM_SUITES[suite], "--test_tag_filters=-gpu",
        f"--run_under={SANDBOX} --profile=t0-cpu --", "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}", *configs])


def _simulate_compile(gate_argv: list[str], ptxas_log: str,
                      device: str) -> int:
    """Run the compile gate, keep the ptxas resource lines, and report."""
    import tempfile
    handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="nanochat-sim-gate.", suffix=".log", delete=False,
        encoding="utf-8")
    try:
        code = with_build_lock([find_bazel(), *gate_argv], stdout=handle,
                               stderr=handle)
    finally:
        handle.close()
    text = Path(handle.name).read_text(encoding="utf-8", errors="replace")
    if code != 0:
        sys.stderr.write(text)
        os.unlink(handle.name)
        return code
    # Keep the ptxas resource lines for the record. A cached build compiles
    # nothing, so keep the previous log instead of erasing it.
    fresh = [line for line in text.splitlines(keepends=True)
             if "ptxas info" in line or _REGISTERS.search(line)]
    fresh_count = sum(1 for line in fresh if _REGISTERS.search(line))
    if fresh_count > 0:
        Path(ptxas_log).write_text("".join(fresh), encoding="utf-8")
        count, note = fresh_count, "fresh"
    elif os.path.isfile(ptxas_log) and os.path.getsize(ptxas_log) > 0:
        previous = Path(ptxas_log).read_text(encoding="utf-8",
                                             errors="replace")
        count = sum(1 for line in previous.splitlines()
                    if _REGISTERS.search(line))
        note = "build cached; kept the previous log"
    else:
        count, note = 0, "build cached and no earlier log"
    print(f"simulate: compile gate OK ({device}, -Werror, ptxas records: "
          f"{count}; {note})")
    print(f"simulate: ptxas resource log -> {ptxas_log}")
    os.unlink(handle.name)
    return 0


_REGISTERS = re.compile(r"Used [0-9]+ registers")


def _simulate_api(ctx: Context, device: str) -> int:
    """Build the mock, run the symbol gate, and run the two API tests."""
    mock = (BAZEL_BIN / "tools" / "cuda_sim"
            / "libcuda_sim_interposer.so").resolve()
    real = (BAZEL_BIN / "backends" / "cuda" / "cuda_runtime_gpu_test").resolve()
    gate = str(TOOLS / "cuda_sim" / "check_symbols.py")
    sim_log = "<sim-api>.jsonl" if ctx.dry_run else str(
        Path(_tempdir("nanochat-sim-api.")) / "events.jsonl")
    build_argv = ["build", "--config=cuda",
                  "//tools/cuda_sim:cuda_sim_interposer",
                  "//backends/cuda:cuda_runtime_gpu_test",
                  "//tests:cuda_sim_preload_test"]
    preload_argv = [
        "test", "//tests:cuda_sim_preload_test", "--config=cuda",
        "--test_tag_filters=sim",
        f"--run_under={SANDBOX} --profile=t0-cpu --",
        "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}",
        f"--test_env=LD_PRELOAD={mock}",
        f"--test_env=NANOCHAT_SIM_LOG={sim_log}"]
    unit_argv = [
        "test", "//tests:cuda_sim_test", "--test_tag_filters=sim",
        f"--run_under={SANDBOX} --profile=t0-cpu --",
        "--spawn_strategy=local",
        f"--test_env=NANOCHAT_SIM_PROFILE={device}"]
    if ctx.dry_run:
        ctx.plan("bazel", "bazel", *build_argv)
        for extra in ([], [str(real)], ["--negative-self-test"]):
            ctx.plan("host", gate, *extra)
        ctx.plan("bazel", "bazel", *preload_argv)
        ctx.plan("bazel", "bazel", *unit_argv)
        return 0
    code = ctx.bazel(build_argv)
    if code != 0:
        return code
    if not mock.is_file():
        die(f"simulate: the interposer .so was not built ({mock})")
    if not real.is_file():
        die(f"simulate: {real} was not built")
    for extra in ([], [str(real)], ["--negative-self-test"]):
        status = _run([gate, *extra]).returncode
        if status != 0:
            return status
    code = ctx.bazel(preload_argv)
    if code != 0:
        return code
    return ctx.bazel(unit_argv)


def cmd_gpu(argv: list[str], ctx: Context) -> int:
    if in_sandbox():
        profile = "t1-gpu"
        rest = list(argv)
        while rest:
            argument = rest[0]
            if argument == "--profile":
                if len(rest) < 2:
                    die("--profile needs a name")
                profile, rest = rest[1], rest[2:]
            elif argument.startswith("--profile="):
                profile, rest = argument.split("=", 1)[1], rest[1:]
            elif argument == "--":
                rest = rest[1:]
                break
            else:
                break
        if not rest:
            die("gpu: no command given")
        if os.environ.get("NANOCHAT_GPU_QUEUED"):
            return exec_replace(rest)
        return ctx.sandbox(profile, rest)
    return exec_replace([str(ENTRY), "gpuq", "submit", *argv])


#: The handler for every ``_entry`` row, in the table order. The keys must
#: equal `_entry.names()`; `python/tests/contract_test.py` compares them.
COMMANDS = {
    "build": cmd_build,
    "test": cmd_test,
    "check": cmd_check,
    "lint": cmd_lint,
    "shutdown": cmd_shutdown,
    "prune": cmd_prune,
    "gpuq": cmd_gpuq,
    "session": cmd_session,
    "run": cmd_run,
    "dist": cmd_dist,
    "client": cmd_client,
    "train": _alias("train"),
    "rl": _alias("train"),
    "eval": _alias("eval"),
    "bench": _alias("t3-bench"),
    "profile": cmd_profile,
    "roofline": cmd_roofline,
    "attain": cmd_attain,
    "devices": cmd_devices,
    "pgo": cmd_pgo,
    "doctor": cmd_doctor,
    "simulate": cmd_simulate,
    "verify": _alias("t2-parity"),
    "gpu": cmd_gpu,
    "help": cmd_help,
}


def _split_dry_run(argv: list[str]) -> tuple[bool, list[str]]:
    """Remove ``--dry-run`` up to the first ``--`` and report whether it ran."""
    dry_run = False
    kept: list[str] = []
    seen_separator = False
    for argument in argv:
        if seen_separator:
            kept.append(argument)
            continue
        if argument == "--dry-run":
            dry_run = True
        elif argument == "--":
            seen_separator = True
            kept.append(argument)
        else:
            kept.append(argument)
    return dry_run, kept


def main(argv: list[str] | None = None) -> int:
    """Run one command and report a command-line error as exit code 2."""
    try:
        return _dispatch(argv)
    except Failure as error:
        sys.stderr.write(f"nanochat: {error}\n")
        return 2


def _dispatch(argv: list[str] | None) -> int:
    """Split the flags, resolve the command, and run it."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    dry_run, kept = _split_dry_run(arguments)
    if not kept:
        sys.stdout.write(_entry.usage())
        return 2
    command, rest = kept[0], kept[1:]
    if command in ("-h", "--help"):
        command, rest = "help", []
    handler = COMMANDS.get(command)
    if handler is None:
        die(f"unknown command '{command}' (try: tools/nanochat help)")
    ctx = Context(dry_run=dry_run)
    if dry_run:
        if command not in DRY_RUN_COMMANDS:
            die("--dry-run covers build, test, check, run, and simulate")
        # A nested invocation (from `check`) prints only its external commands.
        nested = os.environ.get("NANOCHAT_PLAN_NESTED") == "1"
        if not nested:
            ctx.plan("argv", command, *rest)
    return handler(rest, ctx)


if __name__ == "__main__":
    sys.exit(main())
