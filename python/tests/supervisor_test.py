#!/usr/bin/env python3
"""T0: the session supervisor, stage S3 of ``docs/sandbox.md`` Part 2.

The suite is hermetic and CPU-only. It exercises the real wire protocol over a
temporary Unix socket, with a real child process, and it never creates a
namespace: what must not be wrong is the framing, the streaming, the exit
code, and the fact that a command is a child of the holder.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from nanochat_cpp import session, supervisor


class OneShotServer:
    """A server that serves one request from a real socket, then stops."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        connection, _ = self.server.accept()
        with connection:
            supervisor.serve_one(connection, dict(os.environ))
        self.server.close()

    def __enter__(self) -> "OneShotServer":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.thread.join(timeout=30)
        try:
            self.server.close()
        except OSError:
            pass


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_sup_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "s.sock"

    def _send(self, argv: list[str]) -> tuple[int, str, str]:
        with OneShotServer(self.path):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = supervisor.send([str(self.path), self.tmp.name, "--",
                                        *argv])
        return code, out.getvalue(), err.getvalue()

    def test_a_command_runs_and_reports_its_exit_code(self) -> None:
        code, out, _ = self._send(
            [sys.executable, "-c", "print('hello'); raise SystemExit(3)"])
        self.assertEqual(code, 3)
        self.assertIn("hello", out)

    def test_stderr_is_a_separate_stream(self) -> None:
        code, out, err = self._send(
            [sys.executable, "-c",
             "import sys; print('o'); print('e', file=sys.stderr)"])
        self.assertEqual(code, 0)
        self.assertIn("o", out)
        self.assertNotIn("e\n", out)
        self.assertIn("e", err)

    def test_a_missing_command_is_a_clear_failure(self) -> None:
        code, _, err = self._send(["no-such-command-nanochat"])
        self.assertEqual(code, supervisor.SPAWN_FAILURE)
        self.assertIn("cannot run", err)

    def test_a_missing_socket_is_a_failure_not_a_hang(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = supervisor.send([str(Path(self.tmp.name) / "absent.sock"),
                                    self.tmp.name, "--", "true"])
        self.assertEqual(code, supervisor.SPAWN_FAILURE)
        self.assertIn("cannot reach", err.getvalue())

    def test_send_needs_a_command(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.send([str(self.path), self.tmp.name]),
                             2)

    def test_a_malformed_request_answers_with_a_failure_frame(self) -> None:
        with OneShotServer(self.path):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(self.path))
            client.sendall(b"not json\n")
            frames = client.makefile("rb").read().decode("utf-8")
            client.close()
        self.assertIn('"error"', frames)
        self.assertIn('"exit": 127', frames)


class HoldTest(unittest.TestCase):
    def test_hold_needs_its_paths(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(supervisor.hold([]), 2)


    def test_the_interface_list_is_read_from_the_namespace(self) -> None:
        text = """Inter-|   Receive   |  Transmit
 face |bytes    packets|bytes    packets
    lo:    1024       8    1024       8
  eth0: 9000000    9000 1000000    8000
"""
        self.assertEqual(supervisor.interface_names(text), ["lo", "eth0"])
        self.assertEqual(supervisor.interface_names(""), [])
        one = "\n".join(text.splitlines()[:3]) + "\n"
        self.assertEqual(supervisor.interface_names(one), ["lo"])

    def test_hold_refuses_a_namespace_with_a_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with mock.patch.object(supervisor, "set_loopback_up",
                                   return_value=""), \
                    mock.patch.object(supervisor, "interface_names",
                                      return_value=["lo", "eth0"]), \
                    redirect_stderr(io.StringIO()) as err:
                code = supervisor.hold([str(path / "s.sock"),
                                        str(path / "r.json"), "deny",
                                        str(path)])
        self.assertEqual(code, 1)
        self.assertIn("expected only loopback", err.getvalue())

    def test_hold_refuses_to_guess_the_worktree(self) -> None:
        """A missing worktree would confine nothing, so it is an error."""
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(supervisor.hold(["/tmp/s", "/tmp/r", "deny"]), 2)
        self.assertIn("worktree", err.getvalue())

    def test_hold_writes_the_ready_file_before_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ready = path / "r.json"
            with mock.patch.object(supervisor, "set_loopback_up",
                                   return_value=""), \
                    mock.patch.object(supervisor, "interface_names",
                                      return_value=["lo"]), \
                    mock.patch.object(supervisor, "confine",
                                      return_value={"hidden": "none"}), \
                    mock.patch.object(supervisor, "serve", return_value=0) as \
                    serve, mock.patch.object(supervisor, "signal"):
                self.assertEqual(supervisor.hold(
                    [str(path / "s.sock"), str(ready), "deny",
                     str(path)]), 0)
            self.assertTrue(ready.is_file())
            payload = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(payload["net"], "deny")
            self.assertEqual(payload["interfaces"], ["lo"])
            self.assertIn("confinement", payload)
            self.assertEqual(payload["worktree"], str(path.resolve()))
            self.assertTrue(serve.called)


class SessionSupervisorTest(unittest.TestCase):
    """The parts of `session` that name and start the supervisor."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_sup_")
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {"NANOCHAT_SESSION_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_paths_and_the_unit_are_derived_from_the_name(self) -> None:
        self.assertEqual(session.socket_path("nanochat.cpp").name,
                         "nanochat-cpp.sock")
        self.assertEqual(session.ready_path("nanochat.cpp").name,
                         "nanochat-cpp.ready.json")
        self.assertEqual(session.supervisor_unit("nanochat.cpp"),
                         "nanochat-supervisor-nanochat-cpp.service")

    def test_a_ready_file_with_a_live_pid_is_ready(self) -> None:
        path = session.ready_path("agent1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid(), "net": "deny",
                                    "net_ns": "net:[1]"}), encoding="utf-8")
        ready = session.read_ready("agent1")
        self.assertEqual(ready["net_ns"], "net:[1]")

    def test_a_dead_pid_is_not_ready(self) -> None:
        path = session.ready_path("agent1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": 4194304}), encoding="utf-8")
        self.assertIsNone(session.read_ready("agent1"))

    def test_the_unit_properties_carry_the_session_budget(self) -> None:
        values = {"quota": "400%", "mem_high": "3G", "mem_max": "4G",
                  "pids": "512", "cpus": "0-3"}
        properties = session.supervisor_unit_properties(values, "agent1")
        self.assertIn("MemoryMax=4G", properties)
        self.assertIn("TasksMax=512", properties)
        self.assertIn("nanochat-agent-agent1.slice", properties)
        self.assertIn("nanochat-supervisor-agent1.service", properties)

    def test_start_builds_a_namespace_holder(self) -> None:
        values = {"quota": "400%", "mem_high": "3G", "mem_max": "4G",
                  "pids": "512", "cpus": "0-3"}
        recorded: dict[str, list[str]] = {}

        def fake_run(argv, **kwargs):
            recorded["argv"] = argv

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        ready_states = [None, {"pid": os.getpid(), "net": "deny",
                               "net_ns": "net:[1]"}]

        with mock.patch.object(session.subprocess, "run",
                               side_effect=fake_run), \
                mock.patch.object(session, "read_ready",
                                  side_effect=ready_states):
            ready = session.start_supervisor("agent1", values, "deny")
        self.assertEqual(ready["net_ns"], "net:[1]")
        argv = recorded["argv"]
        self.assertEqual(argv[0], "systemd-run")
        self.assertIn("--user", argv)
        self.assertIn("--slice", argv)
        self.assertIn("unshare", argv)
        self.assertIn("--map-root-user", argv)
        self.assertIn("--net", argv)
        self.assertNotIn("--pid", argv, "a process namespace breaks the bus")
        self.assertIn("hold", argv)

    def test_a_host_session_has_no_namespace_flag(self) -> None:
        values = {"quota": "400%", "mem_high": "3G", "mem_max": "4G",
                  "pids": "512", "cpus": "0-3"}
        recorded: dict[str, list[str]] = {}

        def fake_run(argv, **kwargs):
            recorded["argv"] = argv

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        with mock.patch.object(session.subprocess, "run",
                               side_effect=fake_run), \
                mock.patch.object(
                    session, "read_ready",
                    side_effect=[None, {"pid": os.getpid(), "net": "host"}]):
            session.start_supervisor("agent1", values, "host")
        self.assertIn("--mount", recorded["argv"])
        self.assertNotIn("--net", recorded["argv"])

    def test_stop_removes_the_files(self) -> None:
        for path in (session.ready_path("agent1"),
                     session.socket_path("agent1")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        with mock.patch.object(session, "systemctl",
                               return_value=mock.Mock(returncode=0, stdout="",
                                                      stderr="")):
            self.assertEqual(session.stop_supervisor("agent1"), "")
        self.assertFalse(session.ready_path("agent1").exists())
        self.assertFalse(session.socket_path("agent1").exists())


class ConfinementTest(unittest.TestCase):
    """The writable set: the plan, and the read-back that proves it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_conf_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.worktree = self.root / "repos" / "nanochat.cpp"
        (self.worktree / "include" / "nanochat").mkdir(parents=True)

    def test_the_plan_names_every_claim(self) -> None:
        with mock.patch.object(supervisor, "_mount", return_value=""), \
                mock.patch.object(supervisor, "HIDDEN_PATHS", ()):
            claims = supervisor.confine(self.worktree, self.root, False)
        self.assertEqual(claims[str(self.worktree.parent)], "ro")
        self.assertEqual(claims[str(self.worktree)], "rw")
        self.assertEqual(
            claims[str(self.worktree / "include" / "nanochat")], "ro")
        self.assertEqual(claims["hidden"], "none")

    def test_an_architect_session_keeps_the_headers_writable(self) -> None:
        with mock.patch.object(supervisor, "_mount", return_value=""), \
                mock.patch.object(supervisor, "HIDDEN_PATHS", ()):
            claims = supervisor.confine(self.worktree, self.root, True)
        self.assertEqual(
            claims[str(self.worktree / "include" / "nanochat")], "rw")

    def test_a_refused_mount_is_a_failed_claim(self) -> None:
        with mock.patch.object(supervisor, "_mount",
                               return_value="wrong fs type"), \
                mock.patch.object(supervisor, "HIDDEN_PATHS", ()):
            claims = supervisor.confine(self.worktree, self.root, False)
        self.assertEqual(claims[str(self.worktree.parent)], "failed")

    def test_verify_reports_a_failed_claim(self) -> None:
        claims = {"/a": "failed", "hidden": "none"}
        failures = supervisor.verify_confinement(self.worktree, claims)
        self.assertTrue([item for item in failures if "/a" in item])

    def test_verify_reports_a_writable_read_only_claim(self) -> None:
        claims = {str(self.worktree): "rw",
                  str(self.worktree / "include" / "nanochat"): "ro",
                  "hidden": "none"}
        with mock.patch.object(supervisor, "_write_fails",
                               return_value=False):
            failures = supervisor.verify_confinement(self.worktree, claims)
        self.assertTrue([item for item in failures if "still writable" in item])

    def test_verify_accepts_a_clean_plan(self) -> None:
        claims = {str(self.worktree): "rw",
                  str(self.worktree / "include" / "nanochat"): "ro",
                  "hidden": "none"}
        with mock.patch.object(supervisor, "_write_fails", return_value=True):
            self.assertEqual(
                supervisor.verify_confinement(self.worktree, claims), [])

    def test_verify_reports_a_hidden_path_that_is_not_empty(self) -> None:
        hidden = self.root / "secret"
        hidden.mkdir()
        (hidden / "key").write_text("k", encoding="utf-8")
        claims = {str(self.worktree): "rw", "hidden": f"{hidden}: hidden"}
        with mock.patch.object(supervisor, "_write_fails", return_value=True):
            failures = supervisor.verify_confinement(self.worktree, claims)
        self.assertTrue([item for item in failures if "still readable" in item])


if __name__ == "__main__":
    unittest.main()
