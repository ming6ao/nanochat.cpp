#!/usr/bin/env python3
"""The GPU job queue for nanochat.cpp (docs/gpu-queue-plan.md).

One worker process owns the GPU. A client submits a job, and the worker runs
the jobs in first-in, first-out order. A SQLite database holds the job records
and the state machine. Plain files hold the logs.

The module imports only the Python standard library. Run it as a script, or
through ``tools/nanochat gpuq <action>``.

Every job runs under ``tools/sandbox.sh --profile <profile> -- <command>``.
The worker owns the devices and assigns them. The client does not name one.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

#: The state machine. A terminal state never changes.
TERMINAL = ("done", "failed", "cancelled")

#: The heartbeat interval and the lease timeout, in seconds.
HEARTBEAT_SEC = 5.0
LEASE_SEC = 15.0

#: How long a client waits for a worker to publish its first heartbeat.
START_TIMEOUT_SEC = 20.0

#: The poll interval while a client waits for a job, in seconds.
POLL_SEC = 0.2

#: The grace period before the worker escalates to SIGKILL, in seconds.
KILL_GRACE_SEC = 5.0

#: The maximum queue length. A full queue refuses a new job.
MAX_QUEUE = 64

#: The default idle exit, in seconds.
DEFAULT_IDLE_SEC = 1800

#: The shared GPU log. The contract test pins its line format. Override the
#: path with ``NANOCHAT_GPU_LOG`` for a test or a second host.
GPU_LOG = Path("/tmp/nanochat-gpu.log")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id        TEXT PRIMARY KEY,
  state     TEXT NOT NULL,
  command   TEXT NOT NULL,
  directory TEXT NOT NULL,
  profile   TEXT NOT NULL,
  env       TEXT NOT NULL,
  timeout   INTEGER NOT NULL DEFAULT 0,
  submitter TEXT,
  submitted INTEGER NOT NULL,
  started   INTEGER,
  ended     INTEGER,
  devices   TEXT,
  worker    TEXT,
  exit_code INTEGER,
  note      TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue_order ON jobs(state, submitted);
CREATE TABLE IF NOT EXISTS workers (
  id        TEXT PRIMARY KEY,
  pid       INTEGER NOT NULL,
  devices   TEXT NOT NULL,
  started   INTEGER NOT NULL,
  heartbeat INTEGER NOT NULL
);
"""


def repo_root() -> Path:
    """The repository root: ``python/nanochat_cpp`` is two levels down."""
    return Path(__file__).resolve().parents[2]


def queue_base() -> Path:
    """The queue directory: ``$XDG_STATE_HOME/nanochat/gpu-queue``."""
    override = os.environ.get("NANOCHAT_GPU_QUEUE_DIR")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME") or str(
        Path.home() / ".local" / "state")
    return Path(state) / "nanochat" / "gpu-queue"


def logs_dir(base: Path) -> Path:
    """The log directory inside the queue directory."""
    return base / "logs"


def _now_ns() -> int:
    return time.time_ns()


def _seconds(started_ns: int, ended_ns: int) -> str:
    return f"{(ended_ns - started_ns) / 1e9:.3f}s"


def _log(message: str) -> None:
    """Write one worker line to standard error."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    sys.stderr.write(f"gpuq[{os.getpid()}] {stamp} {message}\n")
    sys.stderr.flush()


def gpu_log(message: str) -> None:
    """Append one line to the shared GPU log. Never fail the caller."""
    path = Path(os.environ.get("NANOCHAT_GPU_LOG") or GPU_LOG)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


# -- Store -------------------------------------------------------------------


def open_db(base: Path | None = None) -> sqlite3.Connection:
    """Open the queue database, create the schema, and set the pragmas."""
    base = base or queue_base()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    logs_dir(base).mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(str(base / "queue.db"), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def new_job_id() -> str:
    """A short, filesystem-safe job identifier, unique on this host."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def new_worker_id() -> str:
    """A short, filesystem-safe worker identifier, unique on this host."""
    return f"w{os.getpid()}-{secrets.token_hex(3)}"


def insert_job(conn: sqlite3.Connection, *, command: list[str],
               directory: str, profile: str, env: dict[str, str],
               timeout: int, submitter: str) -> str:
    """Insert one queued job in one transaction and return its identifier."""
    job_id = new_job_id()
    conn.execute(
        "INSERT INTO jobs (id, state, command, directory, profile, env,"
        " timeout, submitter, submitted)"
        " VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
        (job_id, json.dumps(command), directory, profile,
         json.dumps(env), int(timeout), submitter, _now_ns()))
    conn.commit()
    return job_id


def fetch_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    """One job row, or None."""
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def queue_depth(conn: sqlite3.Connection) -> int:
    """The number of queued jobs."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE state='queued'").fetchone()
    return int(row["n"])


def queue_position(conn: sqlite3.Connection, job_id: str) -> int:
    """The number of jobs ahead of this one: running plus earlier queued."""
    row = fetch_job(conn, job_id)
    if row is None:
        return 0
    if row["state"] != "queued":
        return 0
    ahead = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs"
        " WHERE state='queued' AND submitted < ?",
        (row["submitted"],)).fetchone()
    running = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE state='running'").fetchone()
    return int(ahead["n"]) + int(running["n"])


def finish_job(conn: sqlite3.Connection, job_id: str, state: str,
               exit_code: int | None, note: str | None,
               ended: int | None = None) -> None:
    """Write one terminal state in one transaction."""
    conn.execute(
        "UPDATE jobs SET state=?, exit_code=?, note=?, ended=? WHERE id=?",
        (state, exit_code, note, ended or _now_ns(), job_id))
    conn.commit()


# -- Worker lease ------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """True when the process exists and is not a zombie.

    A killed worker stays a zombie until its parent reaps it. ``os.kill(pid,
    0)`` still succeeds for a zombie, so read the state from ``/proc``.
    """
    if pid <= 0:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        stat = ""
    if stat:
        try:
            state = stat.rsplit(")", 1)[1].split()[0]
        except IndexError:
            state = ""
        if state:
            return state != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_workers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """The worker rows whose lease is fresh and whose process is alive."""
    cutoff = _now_ns() - int(LEASE_SEC * 1e9)
    rows = conn.execute("SELECT * FROM workers").fetchall()
    return [row for row in rows
            if row["heartbeat"] >= cutoff and _pid_alive(int(row["pid"]))]


def register_worker(conn: sqlite3.Connection, worker_id: str, pid: int,
                    devices: str) -> None:
    """Insert one worker lease row."""
    now = _now_ns()
    conn.execute(
        "INSERT OR REPLACE INTO workers (id, pid, devices, started, heartbeat)"
        " VALUES (?, ?, ?, ?, ?)",
        (worker_id, int(pid), devices, now, now))
    conn.commit()


def heartbeat(conn: sqlite3.Connection, worker_id: str) -> None:
    """Refresh one worker lease."""
    conn.execute("UPDATE workers SET heartbeat=? WHERE id=?",
                 (_now_ns(), worker_id))
    conn.commit()


def unregister_worker(conn: sqlite3.Connection, worker_id: str) -> None:
    """Delete one worker lease row."""
    conn.execute("DELETE FROM workers WHERE id=?", (worker_id,))
    conn.commit()


def kill_orphans(job_id: str) -> int:
    """Kill the surviving process group of a dead worker's job.

    The worker marks its jobs with ``NANOCHAT_GPU_JOB``, so a later worker can
    find the orphan in ``/proc`` and stop it. Without this step a killed worker
    leaves a job that still owns the GPU.
    """
    marker = f"NANOCHAT_GPU_JOB={job_id}".encode()
    killed = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker not in environ.split(b"\0"):
            continue
        try:
            os.killpg(int(entry.name), signal.SIGTERM)
            killed += 1
        except OSError:
            pass
    return killed


def reconcile(conn: sqlite3.Connection) -> list[str]:
    """Fail the jobs of dead workers and drop the dead leases.

    A recovered job is not re-queued: a training run is not idempotent.
    """
    alive = {row["id"] for row in live_workers(conn)}
    failed: list[str] = []
    for job in conn.execute(
            "SELECT id, worker FROM jobs WHERE state='running'").fetchall():
        if job["worker"] not in alive:
            finish_job(conn, job["id"], "failed", None, "worker died")
            kill_orphans(job["id"])
            failed.append(job["id"])
    for row in conn.execute("SELECT id FROM workers").fetchall():
        if row["id"] not in alive:
            conn.execute("DELETE FROM workers WHERE id=?", (row["id"],))
    conn.commit()
    return failed


# -- Devices -----------------------------------------------------------------


def detect_devices() -> str:
    """The local device pool: ``NANOCHAT_GPU_DEVICES`` or ``nvidia-smi``."""
    value = os.environ.get("NANOCHAT_GPU_DEVICES")
    if value:
        return value
    if shutil.which("nvidia-smi"):
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=index",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            ids = [line.strip() for line in done.stdout.splitlines()
                   if line.strip()]
            if ids:
                return ",".join(ids)
        except (OSError, subprocess.SubprocessError):
            pass
    return "0"


def device_pids(devices: str) -> list[str]:
    """The foreign process identifiers on the given devices.

    The result is empty when ``nvidia-smi`` is absent or the device is free.
    """
    if not shutil.which("nvidia-smi"):
        return []
    try:
        done = subprocess.run(
            ["nvidia-smi", f"--id={devices}", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


# -- Run one job -------------------------------------------------------------


def resolve_sandbox(directory: str) -> str:
    """The sandbox launcher for a job: the worktree copy, or the worker copy."""
    candidate = Path(directory) / "tools" / "sandbox.sh"
    if candidate.is_file():
        return str(candidate)
    return str(repo_root() / "tools" / "sandbox.sh")


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the whole process group, then escalate to SIGKILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=KILL_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=KILL_GRACE_SEC)
    except subprocess.TimeoutExpired:
        pass


def run_job(conn: sqlite3.Connection, base: Path, job: sqlite3.Row,
            devices: str, stop: dict) -> None:
    """Run one claimed job and write its terminal state."""
    job_id = job["id"]
    command = json.loads(job["command"])
    extra_env = json.loads(job["env"] or "{}")
    profile = job["profile"]
    directory = job["directory"] or os.getcwd()
    if not Path(directory).is_dir():
        directory = str(repo_root())
    timeout = int(job["timeout"] or 0)
    cancel_path = logs_dir(base) / f"{job_id}.cancel"
    try:
        cancel_path.unlink()
    except OSError:
        pass

    gpu_log(f"gpuq: started {job_id} GPU {devices} ({profile}): "
            f"{' '.join(command)}")

    busy = device_pids(devices)
    if busy and not stop.get("force"):
        note = f"device {devices} busy with pid {', '.join(busy)}"
        finish_job(conn, job_id, "failed", None, note)
        gpu_log(f"gpuq: finished {job_id} exit 1 ({note})")
        return

    launch = [resolve_sandbox(directory), "--profile", profile, "--", *command]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in extra_env.items()})
    env["NANOCHAT_GPU_QUEUED"] = "1"
    env["NANOCHAT_GPU_JOB"] = job_id
    env["NANOCHAT_GPU_DEVICES"] = devices
    env["CUDA_VISIBLE_DEVICES"] = devices

    out_path = logs_dir(base) / f"{job_id}.stdout"
    err_path = logs_dir(base) / f"{job_id}.stderr"
    with open(out_path, "ab") as out, open(err_path, "ab") as err:
        proc = subprocess.Popen(
            launch, cwd=directory, env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)

        deadline = (time.monotonic() + timeout) if timeout > 0 else None
        state, note, code = "done", None, 0
        while True:
            code = proc.poll()
            if code is not None:
                break
            if cancel_path.exists():
                _terminate(proc)
                state, note, code = "cancelled", "cancelled", proc.returncode
                break
            if deadline is not None and time.monotonic() > deadline:
                _terminate(proc)
                state = "failed"
                note = f"timeout after {timeout}s"
                code = proc.returncode
                break
            time.sleep(POLL_SEC)

    try:
        cancel_path.unlink()
    except OSError:
        pass

    if state == "done" and code != 0:
        state = "failed"
    finish_job(conn, job_id, state, code, note)
    ended = fetch_job(conn, job_id)["ended"]
    gpu_log(f"gpuq: finished {job_id} exit {code} "
            f"({_seconds(int(job['started'] or ended), int(ended))}, {state})")


# -- The worker --------------------------------------------------------------


def claim(conn: sqlite3.Connection, worker_id: str,
          devices: str) -> sqlite3.Row | None:
    """Claim the oldest queued job in one transaction, or return None."""
    now = _now_ns()
    conn.execute("BEGIN IMMEDIATE")
    cursor = conn.execute(
        "UPDATE jobs SET state='running', started=?, worker=?, devices=?"
        " WHERE id = (SELECT id FROM jobs WHERE state='queued'"
        "              ORDER BY submitted LIMIT 1)"
        " AND state='queued'",
        (now, worker_id, devices))
    changed = cursor.rowcount
    conn.commit()
    if changed != 1:
        return None
    return conn.execute(
        "SELECT * FROM jobs WHERE state='running' AND worker=?",
        (worker_id,)).fetchone()


def run_worker(idle_sec: int, force: bool) -> int:
    """Own the GPU and run the queued jobs in order until idle or stopped."""
    base = queue_base()
    conn = open_db(base)
    devices = detect_devices()
    worker_id = new_worker_id()
    force = force or os.environ.get("NANOCHAT_GPU_QUEUE_FORCE") == "1"

    failed = reconcile(conn)
    for job_id in failed:
        _log(f"reconciled {job_id} to failed (worker died)")

    register_worker(conn, worker_id, os.getpid(), devices)
    _log(f"worker {worker_id} owns devices {devices}")

    stop = {"stop": False, "force": force}
    lease_conn = open_db(base)

    def beat() -> None:
        while not stop["stop"]:
            time.sleep(HEARTBEAT_SEC)
            try:
                heartbeat(lease_conn, worker_id)
            except sqlite3.Error:
                pass

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()

    def on_term(signum: int, frame: object) -> None:
        stop["stop"] = True
        _log("stop requested; draining after the current job")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    idle_since = time.monotonic()
    try:
        while True:
            job = claim(conn, worker_id, devices)
            if job is None:
                if stop["stop"]:
                    break
                if time.monotonic() - idle_since >= idle_sec:
                    _log(f"idle for {idle_sec}s; exiting")
                    break
                time.sleep(POLL_SEC)
                continue
            idle_since = time.monotonic()
            run_job(conn, base, job, devices, stop)
    finally:
        stop["stop"] = True
        unregister_worker(conn, worker_id)
        _log(f"worker {worker_id} stopped")
    return 0


# -- Worker start (the client side) -----------------------------------------


def _guard(base: Path) -> Path:
    return base / "worker.starting"


def ensure_worker(conn: sqlite3.Connection, base: Path,
                  idle_sec: int = DEFAULT_IDLE_SEC,
                  force: bool = False) -> sqlite3.Row | None:
    """Return a live worker, and start one when none is alive.

    One atomic directory create guards the start, so two clients do not start
    two workers. The guard is not a GPU lock: it protects one process identity.
    """
    live = live_workers(conn)
    if live:
        return live[0]

    guard = _guard(base)
    for _ in range(2):
        try:
            guard.mkdir(mode=0o700)
            break
        except FileExistsError:
            live = live_workers(conn)
            if live:
                return live[0]
            age = time.time() - guard.stat().st_mtime
            if age > START_TIMEOUT_SEC:
                shutil.rmtree(guard, ignore_errors=True)
                continue
            time.sleep(0.5)
    else:
        return None

    log_path = base / "worker.log"
    worker_env = os.environ.copy()
    if force:
        worker_env["NANOCHAT_GPU_QUEUE_FORCE"] = "1"
    try:
        with open(log_path, "ab") as log:
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "worker",
                 "--idle", str(int(idle_sec))],
                env=worker_env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=log, start_new_session=True, close_fds=True)
    except OSError as error:
        _log(f"cannot start the worker: {error}")
        shutil.rmtree(guard, ignore_errors=True)
        return None

    deadline = time.monotonic() + START_TIMEOUT_SEC
    worker = None
    while time.monotonic() < deadline:
        live = live_workers(conn)
        if live:
            worker = live[0]
            break
        time.sleep(POLL_SEC)
    shutil.rmtree(guard, ignore_errors=True)
    return worker


# -- Client actions ----------------------------------------------------------


def _pump(path: Path, stream, offset: int) -> int:
    """Copy the new bytes of a log file to a stream, and return the offset."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return offset
    if data:
        stream.write(data.decode("utf-8", "replace"))
        stream.flush()
    return offset + len(data)


def exit_code_of(job: sqlite3.Row) -> int:
    """The job exit code. A failed job without a code returns 1."""
    if job["exit_code"] is not None:
        return int(job["exit_code"])
    return 0 if job["state"] == "done" else 1


def wait_and_follow(conn: sqlite3.Connection, base: Path, job_id: str,
                    follow: bool = True) -> int:
    """Block until the job is terminal, follow the logs, return its code.

    A queued job with no live worker means the worker died. Start a new worker
    so the queue does not stall (docs/gpu-queue-plan.md section 4.1).
    """
    out_offset = 0
    err_offset = 0
    out_path = logs_dir(base) / f"{job_id}.stdout"
    err_path = logs_dir(base) / f"{job_id}.stderr"
    while True:
        job = fetch_job(conn, job_id)
        if job is None:
            return 1
        if follow:
            out_offset = _pump(out_path, sys.stdout, out_offset)
            err_offset = _pump(err_path, sys.stderr, err_offset)
        if job["state"] in TERMINAL:
            if follow:
                _pump(out_path, sys.stdout, out_offset)
                _pump(err_path, sys.stderr, err_offset)
            return exit_code_of(job)
        if job["state"] == "queued" and not live_workers(conn):
            ensure_worker(conn, base)
        time.sleep(POLL_SEC)


def cmd_submit(args: argparse.Namespace) -> int:
    """Enqueue one job, start a worker, and optionally wait for the result."""
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.stderr.write("gpuq: submit needs a command after '--'\n")
        return 2
    base = queue_base()
    conn = open_db(base)
    if queue_depth(conn) >= MAX_QUEUE:
        sys.stderr.write(f"gpuq: the queue is full ({MAX_QUEUE} jobs)\n")
        return 2
    if not ensure_worker(conn, base, args.idle, args.force):
        sys.stderr.write("gpuq: cannot start the worker\n")
        return 1
    submitter = args.submitter or os.environ.get("NANOCHAT_SUBMITTER") \
        or getpass.getuser()
    job_id = insert_job(
        conn, command=command, directory=args.directory or os.getcwd(),
        profile=args.profile, env={}, timeout=args.timeout,
        submitter=submitter)
    gpu_log(f"gpuq: queued {job_id} ({args.profile}) {submitter}: "
            f"{' '.join(command)}")
    if args.no_wait:
        if args.json:
            print(json.dumps({"id": job_id, "state": "queued",
                              "profile": args.profile}))
        else:
            print(job_id)
        return 0
    if not args.json:
        position = queue_position(conn, job_id)
        print(f"gpuq: job {job_id} queued ({position} ahead, "
              f"profile {args.profile})", file=sys.stderr)
    code = wait_and_follow(conn, base, job_id, follow=not args.json)
    if args.json:
        job = fetch_job(conn, job_id)
        print(json.dumps({key: job[key] for key in job.keys()}))
    return code


def cmd_start(args: argparse.Namespace) -> int:
    """Start the worker when none is alive."""
    conn = open_db()
    worker = ensure_worker(conn, queue_base(), args.idle, args.force)
    if worker is None:
        sys.stderr.write("gpuq: cannot start the worker\n")
        return 1
    if args.json:
        print(json.dumps({"worker_id": worker["id"],
                          "pid": int(worker["pid"]),
                          "devices": worker["devices"]}))
    else:
        print(f"gpuq: worker {worker['id']} owns devices "
              f"{worker['devices']} (pid {worker['pid']})")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the worker after the current job, or cancel it now."""
    conn = open_db()
    live = live_workers(conn)
    if not live:
        if not args.json:
            print("gpuq: no live worker")
        return 0
    if args.now:
        running = conn.execute(
            "SELECT id FROM jobs WHERE state='running'").fetchone()
        if running is not None:
            cancel_path = logs_dir(queue_base()) / f"{running['id']}.cancel"
            cancel_path.touch()
    for row in live:
        try:
            os.kill(int(row["pid"]), signal.SIGTERM)
        except OSError:
            pass
        if not args.json:
            print(f"gpuq: stop requested for worker {row['id']}")
    if args.json:
        print(json.dumps({"stopped": [row["id"] for row in live]}))
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    """Cancel a queued job, or request the cancellation of a running job."""
    conn = open_db()
    job = fetch_job(conn, args.job)
    if job is None:
        sys.stderr.write(f"gpuq: no such job {args.job}\n")
        return 1
    if job["state"] == "queued":
        finish_job(conn, args.job, "cancelled", None, "cancelled before start")
        if not args.json:
            print(f"gpuq: cancelled {args.job}")
        return 0
    if job["state"] == "running":
        (logs_dir(queue_base()) / f"{args.job}.cancel").touch()
        if not args.json:
            print(f"gpuq: cancellation requested for {args.job}")
        return 0
    if not args.json:
        print(f"gpuq: {args.job} is already {job['state']}")
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    """Block until a job finishes and return its exit code."""
    conn = open_db()
    job = fetch_job(conn, args.job)
    if job is None:
        sys.stderr.write(f"gpuq: no such job {args.job}\n")
        return 1
    return wait_and_follow(conn, queue_base(), args.job, follow=not args.json)


def cmd_logs(args: argparse.Namespace) -> int:
    """Print the job logs, optionally following them."""
    base = queue_base()
    conn = open_db(base)
    if fetch_job(conn, args.job) is None:
        sys.stderr.write(f"gpuq: no such job {args.job}\n")
        return 1
    if args.follow:
        return wait_and_follow(conn, base, args.job)
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        path = logs_dir(base) / f"{args.job}.{name}"
        try:
            data = path.read_bytes()
        except OSError:
            continue
        stream.write(data.decode("utf-8", "replace"))
        stream.flush()
    return 0


def _age(heartbeat_ns: int | None) -> str:
    if heartbeat_ns is None:
        return "-"
    return f"{(time.time_ns() - heartbeat_ns) / 1e9:.1f}s"


def cmd_list(args: argparse.Namespace) -> int:
    """List the queued, running, and finished jobs."""
    conn = open_db()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY submitted DESC LIMIT ?",
        (args.limit,)).fetchall()
    if args.json:
        print(json.dumps([{key: row[key] for key in row.keys()}
                          for row in rows]))
        return 0
    if not rows:
        print("gpuq: the queue is empty")
        return 0
    print(f"{'id':<20} {'state':<10} {'profile':<10} {'submitter':<12} "
          f"{'command'}")
    for row in rows:
        command = " ".join(json.loads(row["command"]))
        if len(command) > 60:
            command = command[:57] + "..."
        print(f"{row['id']:<20} {row['state']:<10} {row['profile']:<10} "
              f"{(row['submitter'] or '-'):<12} {command}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show the queue depth, the worker lease, and one job."""
    conn = open_db()
    live = live_workers(conn)
    worker = live[0] if live else None
    running = conn.execute(
        "SELECT id FROM jobs WHERE state='running'").fetchone()
    payload = {
        "worker_id": worker["id"] if worker else "",
        "alive": worker is not None,
        "devices": worker["devices"] if worker else "",
        "heartbeat_age": _age(worker["heartbeat"]) if worker else "",
        "depth": queue_depth(conn),
        "running_job": running["id"] if running else None,
        "queued": queue_depth(conn),
    }
    if args.job:
        job = fetch_job(conn, args.job)
        if job is None:
            sys.stderr.write(f"gpuq: no such job {args.job}\n")
            return 1
        payload["job"] = {key: job[key] for key in job.keys()}
    if args.json:
        print(json.dumps(payload))
        return 0
    print(f"worker:  {payload['worker_id'] or '(none)'}"
          + (f" age {payload['heartbeat_age']}" if worker else ""))
    print(f"devices: {payload['devices'] or '-'}")
    print(f"queued:  {payload['depth']}")
    print(f"running: {payload['running_job'] or '-'}")
    if args.job:
        job = payload["job"]
        print(f"job:     {job['id']} {job['state']} "
              f"exit={job['exit_code']} note={job['note'] or '-'}")
        print(f"command: {' '.join(json.loads(job['command']))}")
    return 0


# -- Entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for every action."""
    parser = argparse.ArgumentParser(
        prog="tools/nanochat gpuq",
        description="The nanochat GPU job queue (docs/gpu-queue-plan.md).")
    sub = parser.add_subparsers(dest="action", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text)

    p = add("submit", "enqueue a job and wait for it")
    p.add_argument("--profile", default="t1-gpu")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--directory", default="")
    p.add_argument("--submitter", default="")
    p.add_argument("--idle", type=int, default=DEFAULT_IDLE_SEC)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_submit)

    p = add("start", "start the worker")
    p.add_argument("--idle", type=int, default=DEFAULT_IDLE_SEC)
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_start)

    p = add("stop", "stop the worker after the current job")
    p.add_argument("--now", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stop)

    p = add("cancel", "request cancellation of a job")
    p.add_argument("job")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cancel)

    p = add("wait", "block until a job finishes")
    p.add_argument("job")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_wait)

    p = add("logs", "print the job logs")
    p.add_argument("job")
    p.add_argument("--follow", action="store_true")
    p.set_defaults(func=cmd_logs)

    p = add("list", "list the jobs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = add("status", "show the queue depth and one job")
    p.add_argument("job", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = add("worker", "run the worker loop (internal)")
    p.add_argument("--idle", type=int, default=DEFAULT_IDLE_SEC)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run one action."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "worker":
        return run_worker(args.idle, args.force)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
