"""T0: the GPU job queue store, worker, and state machine.

The test is hermetic and CPU-only. It uses a temporary queue directory, a fake
``nvidia-smi`` that reports no compute processes, and a fake
``python/nanochat_cpp/sandbox.py`` that execs the command directly. No GPU and
no systemd scope are involved. See docs/sandbox.md.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import gpu_queue

FAKE_SANDBOX = """#!/bin/sh
# Skip the option list up to the first `--`, then exec the job command.
while [ "$1" != "--" ]; do
  [ "$#" -gt 0 ] || exit 2
  shift
done
shift
exec "$@"
"""


class GpuQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_gpuq_")
        self.root = Path(self.tmp.name)
        self.base = self.root / "queue"
        self.work = self.root / "work"
        (self.work / "python" / "nanochat_cpp").mkdir(parents=True)
        sandbox = self.work / "python" / "nanochat_cpp" / "sandbox.py"
        sandbox.write_text(FAKE_SANDBOX, encoding="utf-8")
        sandbox.chmod(0o755)

        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "nvidia-smi"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)

        self.env = {
            "NANOCHAT_GPU_QUEUE_DIR": str(self.base),
            "NANOCHAT_GPU_DEVICES": "0",
            "NANOCHAT_GPU_LOG": str(self.root / "gpu.log"),
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(self.root)),
            "USER": os.environ.get("USER", "tester"),
        }
        self._patch = mock.patch.dict(os.environ, self.env)
        self._patch.start()
        self.conn = gpu_queue.open_db(self.base)

    def tearDown(self) -> None:
        for row in self.conn.execute("SELECT pid FROM workers").fetchall():
            with contextlib.suppress(OSError):
                os.kill(int(row["pid"]), signal.SIGKILL)
        self._patch.stop()
        self.conn.close()
        self.tmp.cleanup()

    # -- Helpers ---------------------------------------------------------

    def submit(self, command: list[str], *, profile: str = "t0-cpu",
               timeout: int = 0, wait: bool = True) -> str:
        args = gpu_queue.build_parser().parse_args(
            ["submit", "--profile", profile, "--timeout", str(timeout),
             "--directory", str(self.work), *([] if wait else ["--no-wait"]),
             "--", *command])
        gpu_queue.cmd_submit(args)
        return self.conn.execute(
            "SELECT id FROM jobs ORDER BY submitted DESC LIMIT 1").fetchone()["id"]

    def wait_state(self, job_id: str, states: tuple[str, ...],
                   timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = gpu_queue.fetch_job(self.conn, job_id)
            if job["state"] in states:
                return job["state"]
            time.sleep(0.05)
        self.fail(f"{job_id} did not reach {states}")

    def live_pid(self) -> int:
        workers = gpu_queue.live_workers(self.conn)
        self.assertTrue(workers, "no live worker")
        return int(workers[0]["pid"])

    # -- Tests -----------------------------------------------------------

    def test_schema_and_pragmas(self) -> None:
        tables = {row["name"] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("jobs", tables)
        self.assertIn("workers", tables)
        mode = self.conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")
        self.assertEqual(self.base.stat().st_mode & 0o777, 0o700)

    def test_submit_and_run(self) -> None:
        job_id = self.submit(["/bin/sh", "-c", "echo hello-queue"])
        job = gpu_queue.fetch_job(self.conn, job_id)
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["exit_code"], 0)
        self.assertEqual(job["profile"], "t0-cpu")
        log = (self.base / "logs" / f"{job_id}.stdout").read_text()
        self.assertIn("hello-queue", log)

    def test_failure_records_the_exit_code(self) -> None:
        job_id = self.submit(["/bin/sh", "-c", "exit 7"])
        job = gpu_queue.fetch_job(self.conn, job_id)
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["exit_code"], 7)

    def test_jobs_run_in_order(self) -> None:
        first = self.submit(
            ["/bin/sh", "-c", "date +%s.%N; sleep 1; date +%s.%N"],
            wait=False)
        second = self.submit(["/bin/sh", "-c", "date +%s.%N"], wait=False)
        self.wait_state(second, gpu_queue.TERMINAL)
        first_log = (self.base / "logs" / f"{first}.stdout").read_text()
        second_log = (self.base / "logs" / f"{second}.stdout").read_text()
        start_first, end_first = [float(x) for x in first_log.split()]
        start_second = float(second_log.split()[0])
        self.assertLess(start_first, end_first)
        self.assertGreaterEqual(start_second, end_first - 0.05)

    def test_timeout_stops_the_job(self) -> None:
        job_id = self.submit(
            ["/bin/sh", "-c", "echo start; sleep 30; echo end"], timeout=1)
        job = gpu_queue.fetch_job(self.conn, job_id)
        self.assertEqual(job["state"], "failed")
        self.assertIn("timeout", job["note"])
        log = (self.base / "logs" / f"{job_id}.stdout").read_text()
        self.assertIn("start", log)
        self.assertNotIn("end", log)

    def test_cancel_stops_a_running_job(self) -> None:
        job_id = self.submit(
            ["/bin/sh", "-c", "echo begin; sleep 30; echo finish"], wait=False)
        self.wait_state(job_id, ("running",))
        gpu_queue.cmd_cancel(
            gpu_queue.build_parser().parse_args(["cancel", job_id]))
        self.wait_state(job_id, gpu_queue.TERMINAL)
        job = gpu_queue.fetch_job(self.conn, job_id)
        self.assertEqual(job["state"], "cancelled")
        log = (self.base / "logs" / f"{job_id}.stdout").read_text()
        self.assertNotIn("finish", log)

    def test_cancel_a_queued_job(self) -> None:
        first = self.submit(["/bin/sh", "-c", "sleep 10"], wait=False)
        second = self.submit(["/bin/sh", "-c", "echo second"], wait=False)
        self.wait_state(first, ("running",))
        gpu_queue.cmd_cancel(
            gpu_queue.build_parser().parse_args(["cancel", second]))
        self.wait_state(second, gpu_queue.TERMINAL)
        self.assertEqual(gpu_queue.fetch_job(self.conn, second)["state"],
                         "cancelled")

    def test_worker_restart_fails_the_job_and_clears_the_queue(self) -> None:
        job_id = self.submit(["/bin/sh", "-c", "sleep 60"], wait=False)
        self.wait_state(job_id, ("running",))
        os.kill(self.live_pid(), signal.SIGKILL)
        time.sleep(0.5)
        follow = self.submit(["/bin/sh", "-c", "echo after-restart"])
        job = gpu_queue.fetch_job(self.conn, job_id)
        self.assertEqual(job["state"], "failed")
        self.assertIn("worker died", job["note"])
        self.assertEqual(gpu_queue.fetch_job(self.conn, follow)["state"], "done")

    def test_reconcile_marks_a_dead_worker_job(self) -> None:
        job_id = gpu_queue.insert_job(
            self.conn, command=["/bin/true"], directory=str(self.work),
            profile="t0-cpu", env={}, timeout=0, submitter="tester")
        self.conn.execute(
            "UPDATE jobs SET state='running', worker='ghost' WHERE id=?",
            (job_id,))
        self.conn.commit()
        self.conn.execute(
            "INSERT INTO workers (id, pid, devices, started, heartbeat)"
            " VALUES ('ghost', 999999, '0', 0, 0)")
        self.conn.commit()
        failed = gpu_queue.reconcile(self.conn)
        self.assertIn(job_id, failed)
        self.assertEqual(gpu_queue.fetch_job(self.conn, job_id)["state"],
                         "failed")
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM workers").fetchone()[0], 0)

    def test_gpu_log_line_format(self) -> None:
        job_id = self.submit(["/bin/sh", "-c", "echo logged"])
        lines = (self.root / "gpu.log").read_text().splitlines()
        queued = [line for line in lines if line.startswith("gpuq: queued")]
        started = [line for line in lines if line.startswith("gpuq: started")]
        finished = [line for line in lines if line.startswith("gpuq: finished")]
        self.assertEqual(len(queued), 1)
        self.assertEqual(len(started), 1)
        self.assertEqual(len(finished), 1)
        self.assertTrue(queued[0].startswith(
            f"gpuq: queued {job_id} (t0-cpu) "))
        self.assertEqual(
            started[0],
            f"gpuq: started {job_id} GPU 0 (t0-cpu): /bin/sh -c echo logged")
        self.assertTrue(finished[0].startswith(
            f"gpuq: finished {job_id} exit 0 ("),
            finished[0])
        self.assertTrue(finished[0].endswith("s, done)"), finished[0])

    def test_queue_position(self) -> None:
        first = self.submit(["/bin/sh", "-c", "sleep 2"], wait=False)
        second = self.submit(["/bin/sh", "-c", "sleep 2"], wait=False)
        self.wait_state(first, ("running",))
        self.assertGreaterEqual(gpu_queue.queue_position(self.conn, second), 1)
        self.assertEqual(gpu_queue.queue_position(self.conn, first), 0)

    # -- The lease, the claim lock, and retention ------------------------

    def test_heartbeat_reports_a_lost_lease(self) -> None:
        """A missing lease row is a lost lease, not a silent no-op."""
        self.assertFalse(gpu_queue.heartbeat(self.conn, "nobody"))

    def test_a_locked_database_is_reported_not_raised(self) -> None:
        """A lease write that cannot land returns False instead of raising."""
        gpu_queue.register_worker(self.conn, "w-lock", 999999, "0")
        holder = sqlite3.connect(str(self.base / "queue.db"), timeout=5.0)
        holder.execute("BEGIN EXCLUSIVE")
        blocked = sqlite3.connect(str(self.base / "queue.db"), timeout=0.1)
        blocked.execute("PRAGMA busy_timeout=100")
        try:
            self.assertFalse(gpu_queue.heartbeat(blocked, "w-lock"))
        finally:
            blocked.close()
            holder.rollback()
            holder.close()
            self.conn.execute("DELETE FROM workers WHERE id='w-lock'")
            self.conn.commit()

    def test_retire_stale_workers_never_signals_this_process(self) -> None:
        """A stale row that names this process must not stop the client."""
        self.conn.execute(
            "INSERT INTO workers (id, pid, devices, started, heartbeat)"
            " VALUES ('self', ?, '0', 0, 0)", (os.getpid(),))
        self.conn.commit()
        self.assertEqual(gpu_queue.retire_stale_workers(self.conn), [])
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM workers").fetchone()[0], 0)

    def test_claim_takes_no_write_lock_when_idle(self) -> None:
        """An idle poll must not hold the writer five times a second."""
        self.assertIsNone(gpu_queue.claim(self.conn, "w-idle", "0"))
        self.assertFalse(self.conn.in_transaction)

    def test_retire_stale_workers_stops_the_orphan(self) -> None:
        """A live process with a stale lease is stopped, not left to compete."""
        orphan = subprocess.Popen(["/bin/sh", "-c", "sleep 30"])
        try:
            self.conn.execute(
                "INSERT INTO workers (id, pid, devices, started, heartbeat)"
                " VALUES ('ghost', ?, '0', 0, 0)", (orphan.pid,))
            self.conn.commit()
            retired = gpu_queue.retire_stale_workers(self.conn)
            self.assertEqual(retired, [orphan.pid])
            self.assertEqual(self.conn.execute(
                "SELECT COUNT(*) FROM workers").fetchone()[0], 0)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and orphan.poll() is None:
                time.sleep(0.05)
            self.assertIsNotNone(orphan.poll(), "the orphan kept running")
        finally:
            with contextlib.suppress(OSError):
                orphan.kill()
            orphan.wait()

    def test_worker_exits_when_its_lease_row_disappears(self) -> None:
        """An orphan must step down instead of polling beside a replacement."""
        worker = gpu_queue.ensure_worker(self.conn, self.base)
        self.assertIsNotNone(worker)
        pid = int(worker["pid"])
        self.assertNotEqual(pid, os.getpid())
        try:
            self.conn.execute("DELETE FROM workers WHERE id=?",
                              (worker["id"],))
            self.conn.commit()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and gpu_queue._pid_alive(pid):
                time.sleep(0.05)
            self.assertFalse(gpu_queue._pid_alive(pid),
                             "the worker kept polling after losing its lease")
        finally:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

    def test_submit_inherits_the_profile_wall_clock(self) -> None:
        """A job with no explicit timeout gets the profile backstop."""
        args = gpu_queue.build_parser().parse_args(
            ["submit", "--profile", "t0-cpu", "--directory", str(self.work),
             "--timeout", "0", "--", "/bin/true"])
        gpu_queue.cmd_submit(args)
        zero = self.conn.execute(
            "SELECT timeout FROM jobs ORDER BY submitted DESC LIMIT 1"
        ).fetchone()["timeout"]
        self.assertEqual(int(zero), 0)

        args = gpu_queue.build_parser().parse_args(
            ["submit", "--profile", "t0-cpu", "--directory", str(self.work),
             "--", "/bin/true"])
        gpu_queue.cmd_submit(args)
        inherited = self.conn.execute(
            "SELECT timeout FROM jobs ORDER BY submitted DESC LIMIT 1"
        ).fetchone()["timeout"]
        self.assertEqual(int(inherited), gpu_queue.profile_runtime("t0-cpu"))
        self.assertGreater(int(inherited), 0)

    def test_prune_reaps_old_logs_and_keeps_the_records(self) -> None:
        """The logs are reaped; the job rows are the record and stay."""
        logs = self.base / "logs"
        job_id = gpu_queue.insert_job(
            self.conn, command=["/bin/true"], directory=str(self.work),
            profile="t0-cpu", env={}, timeout=0, submitter="tester")
        fresh = logs / f"{job_id}.stdout"
        fresh.write_text("fresh\n", encoding="utf-8")
        old = logs / "20200101-000000-aaaaaa.stdout"
        old.write_text("old\n", encoding="utf-8")
        stale = time.time() - 40 * 86400
        os.utime(old, (stale, stale))

        removed, freed = gpu_queue.prune_logs(
            self.conn, self.base, gpu_queue.LOG_KEEP_DAYS)
        self.assertEqual(removed, 1)
        self.assertEqual(freed, len("old\n"))
        self.assertFalse(old.exists())
        self.assertIsNotNone(gpu_queue.fetch_job(self.conn, job_id))

        # A queued job keeps its logs whatever their age.
        os.utime(fresh, (stale, stale))
        removed, _ = gpu_queue.prune_logs(self.conn, self.base, 1)
        self.assertEqual(removed, 0)
        self.assertTrue(fresh.exists())

    def test_status_reports_the_log_footprint(self) -> None:
        """The log footprint is visible before it becomes a problem."""
        logs = self.base / "logs"
        (logs / "20200101-000000-aaaaaa.stderr").write_text(
            "x" * 32, encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(gpu_queue.cmd_status(
                gpu_queue.build_parser().parse_args(
                    ["status", "--json"])), 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["log_files"], 1)
        self.assertEqual(payload["log_bytes"], 32)


if __name__ == "__main__":
    unittest.main()
