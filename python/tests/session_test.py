#!/usr/bin/env python3
"""T0: the agent session scope, stages S1 and S2 of ``docs/sandbox.md`` Part 2.

The suite is hermetic and CPU-only. It needs no systemd, no Bazel, and no GPU:
``systemctl`` and the Bazel server probe are mocked, and the record directory is
a temporary directory. What it covers is the part that must not be wrong:

- the name and the slice, because the slice hierarchy is what keeps the
  aggregate ceiling in force;
- the record round trip, because the record is the session;
- the scope arguments, because ``--slice`` is what puts a detached Bazel server
  in a cgroup with a memory ceiling;
- the placement check, because a server outside the slice was the 2026-10-01
  crash.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from nanochat_cpp import cli, sandbox, session


def _completed(stdout: str = "", returncode: int = 0,
               stderr: str = "") -> subprocess.CompletedProcess:
    """One fake ``systemctl --user`` result."""
    return subprocess.CompletedProcess(["systemctl"], returncode, stdout, stderr)


class SessionNameTest(unittest.TestCase):
    def test_a_worktree_name_becomes_a_slice_name(self) -> None:
        self.assertEqual(session.normalize_id("nanochat.cpp"), "nanochat-cpp")
        self.assertEqual(session.normalize_id(" Nanochat.CPP.architect "),
                         "nanochat-cpp-architect")
        self.assertEqual(session.default_id(Path("/w/nanochat.cpp")),
                         "nanochat-cpp")

    def test_a_name_with_no_letter_or_digit_is_an_error(self) -> None:
        for bad in ("", "   ", "---", "..."):
            with self.assertRaises(session.SessionError):
                session.normalize_id(bad)

    def test_the_slice_nests_under_the_aggregate_slice(self) -> None:
        """The dash hierarchy is what keeps the aggregate ceiling in force."""
        self.assertEqual(session.slice_name("agent1"),
                         "nanochat-agent-agent1.slice")
        self.assertTrue(session.slice_name("agent1").startswith(
            session.SLICE_PREFIX))


class RecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_session_")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        patcher = mock.patch.dict(
            os.environ, {"NANOCHAT_SESSION_DIR": str(self.work / "sessions")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_store_follows_the_environment(self) -> None:
        self.assertEqual(session.state_dir(), self.work / "sessions")

    def test_a_record_round_trips(self) -> None:
        root = self.work / "nanochat.cpp"
        root.mkdir()
        record = session.start_record("agent1", root, agent="pi")
        self.assertEqual(record["version"], session.RECORD_VERSION)
        self.assertEqual(record["slice"], "nanochat-agent-agent1.slice")
        self.assertEqual(record["profile"], session.BUILD_PROFILE)
        self.assertEqual(session.read_record("agent1"), record)
        self.assertEqual(session.list_records(), [record])
        self.assertEqual(session.find_for_root(root)["id"], "agent1")
        self.assertIsNone(session.find_for_root(self.work / "elsewhere"))

    def test_remove_deletes_one_record(self) -> None:
        session.start_record("agent1", self.work, agent="")
        self.assertTrue(session.remove_record("agent1"))
        self.assertFalse(session.remove_record("agent1"))
        self.assertIsNone(session.read_record("agent1"))

    def test_a_broken_or_foreign_record_is_ignored(self) -> None:
        directory = session.state_dir()
        directory.mkdir(parents=True)
        (directory / "broken.json").write_text("{not json", encoding="utf-8")
        (directory / "future.json").write_text(
            json.dumps({"version": 99, "id": "future", "root": "/w"}),
            encoding="utf-8")
        (directory / "empty.json").write_text("{}", encoding="utf-8")
        self.assertEqual(session.list_records(), [])

    def test_the_environment_names_the_session_first(self) -> None:
        """A command inside a session needs no record of its own."""
        with mock.patch.dict(os.environ, {"NANOCHAT_SESSION": "from-env"}):
            record = session.active()
        self.assertEqual(record["id"], "from-env")
        self.assertIsNone(session.read_record("from-env"))

    def test_the_worktree_finds_its_record(self) -> None:
        session.start_record("agent1", session.REPO_ROOT, agent="")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_SESSION", None)
            self.assertEqual(session.active()["id"], "agent1")


class SliceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_session_")
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {"NANOCHAT_SESSION_DIR": str(Path(self.tmp.name) / "sessions")})
        patcher.start()
        self.addCleanup(patcher.stop)
        session.start_record("agent1", Path(self.tmp.name), agent="")

    def test_status_reports_an_idle_session(self) -> None:
        with mock.patch.object(session, "systemctl",
                               return_value=_completed("not-found\n")):
            report = session.status("agent1")
        self.assertEqual(report["state"], "idle")
        self.assertEqual(report["cgroup"], "")
        self.assertEqual(report["slice"], "nanochat-agent-agent1.slice")

    def test_status_reads_the_limits_back(self) -> None:
        cgroup = ("/user.slice/user-1000.slice/user@1000.service/nanochat"
                  ".slice/nanochat-agent.slice/nanochat-agent-agent1.slice")

        def fake(*arguments: str) -> subprocess.CompletedProcess:
            if "LoadState" in arguments:
                return _completed("loaded\n")
            return _completed(f"{cgroup}\n")

        values = {"memory.current": "2048", "memory.max": "4294967296",
                  "memory.peak": "8192", "pids.current": "7",
                  "pids.max": "512"}
        with mock.patch.object(session, "systemctl", side_effect=fake), \
                mock.patch.object(session, "_read",
                                  side_effect=lambda path: values[
                                      path.rsplit("/", 1)[-1]]):
            report = session.status("agent1")
        self.assertEqual(report["state"], "active")
        self.assertEqual(report["cgroup"], cgroup)
        self.assertEqual(report["memory_max"], "4294967296")
        self.assertEqual(report["pids_current"], "7")

    def test_scopes_report_each_scope_ceiling(self) -> None:
        """The slice holds the session ceiling; each scope holds a budget."""
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory) / "nanochat-agent-x.slice"
            scope = top / "nanochat-x-1.scope"
            scope.mkdir(parents=True)
            (top / "memory.max").write_text("4294967296", encoding="utf-8")
            (scope / "memory.max").write_text("1073741824", encoding="utf-8")
            (scope / "memory.current").write_text("10", encoding="utf-8")
            (scope / "pids.max").write_text("512", encoding="utf-8")
            with mock.patch.object(session, "CGROUP_MOUNT", directory):
                found = session._scopes("/nanochat-agent-x.slice")
        self.assertEqual([entry["scope"] for entry in found],
                         ["nanochat-x-1.scope"])
        self.assertEqual(found[0]["memory_max"], "1073741824")
        self.assertEqual(found[0]["memory_current"], "10")

    def test_stop_reports_a_slice_it_cannot_stop(self) -> None:
        with mock.patch.object(session, "systemctl",
                               return_value=_completed("", 1, "denied")), \
                mock.patch.object(session, "slice_cgroup",
                                  return_value=""), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(session.stop("agent1"), 1)
        self.assertIn("cannot stop", output.getvalue())
        self.assertIsNotNone(session.read_record("agent1"))

    def test_stop_removes_the_record(self) -> None:
        with mock.patch.object(session, "systemctl",
                               return_value=_completed()), \
                mock.patch.object(session, "slice_cgroup",
                                  return_value=""), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(session.stop("agent1"), 0)
        self.assertIsNone(session.read_record("agent1"))


class PlacementTest(unittest.TestCase):
    """The build read-back. A server outside the slice is a failure."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_session_")
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {"NANOCHAT_SESSION_DIR": str(Path(self.tmp.name) / "sessions")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _placement(self, cgroup: str, pid: int | None = 4242):
        """Where one process runs, through the function the check uses."""
        if pid is None:
            return None, ""
        with mock.patch.object(session, "server_cgroup",
                               return_value=cgroup):
            return session.placement_of(pid, "agent1")

    def test_a_server_in_the_slice_is_accepted(self) -> None:
        inside, cgroup = self._placement(
            "/user.slice/user-1000.slice/user@1000.service/nanochat.slice"
            "/nanochat-agent.slice/nanochat-agent-agent1.slice"
            "/nanochat-agent1-99.scope")
        self.assertTrue(inside)
        self.assertIn("nanochat-agent-agent1.slice", cgroup)

    def test_a_server_in_another_slice_is_refused(self) -> None:
        """The aggregate slice is not enough: the session slice must hold it."""
        inside, _ = self._placement(
            "/user.slice/user-1000.slice/user@1000.service/app.slice"
            "/run-abc.scope")
        self.assertFalse(inside)
        inside, _ = self._placement(
            "/user.slice/user-1000.slice/user@1000.service/nanochat.slice"
            "/nanochat-agent.slice/nanochat-agent-other.slice/run-abc.scope")
        self.assertFalse(inside)

    def test_no_server_is_unknown_not_a_pass(self) -> None:
        self.assertEqual(self._placement("", pid=None), (None, ""))
        self.assertEqual(self._placement("", pid=4242), (None, ""))

    def test_check_placement_fails_the_build(self) -> None:
        with mock.patch.object(session, "server_pid", return_value=4242), \
                mock.patch.object(
                    session, "server_cgroup",
                    return_value="/app.slice/run-abc.scope"), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                session.check_placement("agent1", "deny"), 1)
        self.assertIn("outside the session", output.getvalue())

    def test_check_placement_passes_inside_and_when_absent(self) -> None:
        inside = ("/nanochat.slice/nanochat-agent-agent1.slice/"
                  "nanochat-agent1-99.scope")
        with mock.patch.object(session, "server_pid", return_value=4242), \
                mock.patch.object(session, "server_cgroup",
                                  return_value=inside), \
                mock.patch.object(session, "net_is_private",
                                  return_value=True), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                session.check_placement("agent1", "deny"), 0)
        self.assertIn("no network", output.getvalue())
        with mock.patch.object(session, "server_pid", return_value=None), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                session.check_placement("agent1", "deny"), 0)
        self.assertIn("not verified", output.getvalue())

    def test_a_denied_build_that_kept_the_host_network_fails(self) -> None:
        """The network claim is read back, like the memory claim."""
        inside = ("/nanochat.slice/nanochat-agent-agent1.slice/"
                  "nanochat-agent1-99.scope")
        with mock.patch.object(session, "server_pid", return_value=4242), \
                mock.patch.object(session, "server_cgroup",
                                  return_value=inside), \
                mock.patch.object(session, "net_is_private",
                                  return_value=False), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                session.check_placement("agent1", "deny"), 1)
        self.assertIn("shares the host network", output.getvalue())

    def test_a_host_session_does_not_check_the_network(self) -> None:
        inside = ("/nanochat.slice/nanochat-agent-agent1.slice/"
                  "nanochat-agent1-99.scope")
        with mock.patch.object(session, "server_pid", return_value=4242), \
                mock.patch.object(session, "server_cgroup",
                                  return_value=inside), \
                mock.patch.object(session, "net_is_private",
                                  return_value=False), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                session.check_placement("agent1", "host"), 0)
        self.assertNotIn("network", output.getvalue())


class NetworkTest(unittest.TestCase):
    """The deny rule as the session and the CLI state it.

    The namespace itself belongs to the supervisor, so its own tests live in
    ``supervisor_test.py``.
    """

    def test_the_record_keeps_the_network_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ,
                                 {"NANOCHAT_SESSION_DIR": directory}):
                record = session.start_record("agent1", Path(directory),
                                              net="host")
                self.assertEqual(record["net"], "host")
                self.assertEqual(session.read_record("agent1")["net"], "host")
                self.assertEqual(session.start_record(
                    "agent1", Path(directory))["net"], session.NET_DENY)
                with self.assertRaises(session.SessionError):
                    session.start_record("agent1", Path(directory),
                                         net="nonsense")

    def test_the_cli_reads_a_net_flag_and_keeps_the_rest(self) -> None:
        net, rest = cli._take_net_flag(["--net", "host", "//src:plan_main"])
        self.assertEqual((net, rest), ("host", ["//src:plan_main"]))
        net, rest = cli._take_net_flag(["--net=deny", "//..."])
        self.assertEqual((net, rest), ("deny", ["//..."]))
        self.assertEqual(cli._take_net_flag(["//..."]), (None, ["//..."]))
        with self.assertRaises(cli.Failure):
            cli._take_net_flag(["--net", "nonsense"])

    def test_the_cli_reads_the_session_net_option(self) -> None:
        identifier, agent, as_json, net, architect, command = \
            cli._parse_session_args(["--net", "host", "--", "pi"])
        self.assertEqual((identifier, agent, as_json, net, architect, command),
                         ("", "", False, "host", False, ["pi"]))
        self.assertEqual(cli._parse_session_args(["--json"])[3],
                         session.NET_DENY)
        with self.assertRaises(cli.Failure):
            cli._parse_session_args(["--net", "nonsense"])


class ServerFileTest(unittest.TestCase):
    """The server pid comes from a file, never from a Bazel call."""

    def _workspace(self, directory: str, pid_text: str) -> tuple[Path, Path]:
        base = Path(directory) / "base"
        marker = base / "execroot" / "_main" / "bazel-out" / "k8" / "bin"
        marker.mkdir(parents=True)
        (base / "server").mkdir()
        (base / "server" / "server.pid.txt").write_text(
            pid_text, encoding="utf-8")
        root = Path(directory) / "ws"
        root.mkdir()
        (root / "bazel-bin").symlink_to(marker)
        return root, base

    def test_the_output_base_is_the_directory_above_execroot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, base = self._workspace(directory, "1")
            self.assertEqual(session.output_base(root), base)

    def test_the_pid_is_read_from_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._workspace(directory, f"{os.getpid()}\n")
            self.assertEqual(session.server_pid(root), os.getpid())

    def test_a_stale_pid_file_is_not_a_server(self) -> None:
        """A dead pid is `unknown`, so a read-back cannot pass on a ghost."""
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._workspace(directory, "4194304")
            self.assertIsNone(session.server_pid(root))

    def test_no_bazel_bin_means_no_output_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(session.output_base(Path(directory)))


class ReconcileTest(unittest.TestCase):
    """A server that cannot serve this build is stopped before the build."""

    def _reconcile(self, inside: bool, private: bool | None,
                   net: str) -> tuple[bool, str]:
        with mock.patch.object(cli.session, "server_pid", return_value=4242), \
                mock.patch.object(cli.session, "placement_of",
                                  return_value=(inside, "/cgroup/x.scope")), \
                mock.patch.object(cli.session, "net_is_private",
                                  return_value=private), \
                mock.patch.object(cli, "find_bazel", return_value="/bin/true"), \
                mock.patch.object(cli, "_run") as run, \
                redirect_stdout(io.StringIO()) as output:
            cli._reconcile_server("agent1", net)
        return run.called, output.getvalue()

    def test_a_good_server_is_left_alone(self) -> None:
        self.assertEqual(self._reconcile(True, True, session.NET_DENY),
                         (False, ""))

    def test_a_server_outside_the_slice_is_stopped(self) -> None:
        called, text = self._reconcile(False, True, session.NET_DENY)
        self.assertTrue(called)
        self.assertIn("outside the session slice", text)

    def test_a_server_on_the_wrong_network_is_stopped(self) -> None:
        called, text = self._reconcile(True, False, session.NET_DENY)
        self.assertTrue(called)
        self.assertIn("on the host network", text)
        called, text = self._reconcile(True, True, session.NET_HOST)
        self.assertTrue(called)
        self.assertIn("on the private network", text)

    def test_no_server_needs_no_stop(self) -> None:
        with mock.patch.object(cli.session, "server_pid", return_value=None), \
                mock.patch.object(cli, "_run") as run, \
                redirect_stdout(io.StringIO()):
            cli._reconcile_server("agent1", session.NET_DENY)
        self.assertFalse(run.called)


class ScopeArgumentsTest(unittest.TestCase):
    """``--session`` is what puts the detached server in a bounded cgroup."""

    def test_parse_accepts_both_session_forms(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NANOCHAT_SANDBOX_BACKEND": "auto",
                              "NANOCHAT_SANDBOX_NET": "host"}):
            self.assertEqual(
                sandbox.parse_argv(["--session", "a", "--", "true"]),
                ("t0-cpu", "auto", True, "a", ["true"]))
            self.assertEqual(
                sandbox.parse_argv(["--session=a", "--", "true"]),
                ("t0-cpu", "auto", True, "a", ["true"]))
            self.assertEqual(sandbox.parse_argv(["--", "true"]),
                             ("t0-cpu", "auto", True, "", ["true"]))

    def test_parse_rejects_a_session_with_no_name(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NANOCHAT_SANDBOX_BACKEND": "auto"}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    sandbox.parse_argv(["--session", "--", "true"]), 2)
                self.assertEqual(
                    sandbox.parse_argv(["--session=", "--", "true"]), 2)

    def test_the_scope_lands_in_the_session_slice(self) -> None:
        profiles = sandbox.load_profiles()
        values = sandbox.profile_values(profiles, "build")
        recorded: dict[str, list[str]] = {}

        def record(argv: list[str]) -> int:
            recorded["argv"] = argv
            return 0

        with mock.patch.dict(os.environ), \
                mock.patch.object(sandbox, "install_slice"), \
                mock.patch.object(sandbox, "exec_replace", side_effect=record):
            os.environ.pop("NANOCHAT_SESSION", None)
            sandbox.run_in_scope(profiles, "build", values, True, ["true"],
                                 "agent1")
            self.assertEqual(os.environ["NANOCHAT_SESSION"], "agent1")
            self.assertEqual(os.environ["NANOCHAT_SESSION_SLICE"],
                             "nanochat-agent-agent1.slice")

        argv = recorded["argv"]
        self.assertIn("--slice=nanochat-agent-agent1.slice", argv)
        self.assertNotIn(f"--slice={profiles['slice']}", argv)
        self.assertTrue([part for part in argv
                         if part.startswith("--unit=nanochat-agent1-")], argv)
        self.assertIn(f"MemoryMax={values['mem_max']}", argv)

    def test_the_build_profile_has_no_wall_clock(self) -> None:
        """A wall clock would stop the server in the middle of a later build."""
        values = sandbox.profile_values(sandbox.load_profiles(), "build")
        self.assertNotIn("runtime", values)

    def test_run_in_scope_without_a_session_keeps_the_aggregate_slice(
            self) -> None:
        profiles = sandbox.load_profiles()
        values = sandbox.profile_values(profiles, "t0-cpu")
        recorded: dict[str, list[str]] = {}

        def record(argv: list[str]) -> int:
            recorded["argv"] = argv
            return 0

        with mock.patch.dict(os.environ), \
                mock.patch.object(sandbox, "install_slice"), \
                mock.patch.object(sandbox, "exec_replace", side_effect=record):
            sandbox.run_in_scope(profiles, "t0-cpu", values, True, ["true"])
        argv = recorded["argv"]
        self.assertIn("--slice=nanochat.slice", argv)
        self.assertFalse([part for part in argv
                          if part.startswith("--unit=")], argv)


class SessionGuaranteesTest(unittest.TestCase):
    """The two claims the session verifier must prove, inside the scope."""

    def test_a_read_back_that_disagrees_is_a_failure(self) -> None:
        with mock.patch.object(sandbox, "read_cgroup_path",
                               return_value=Path("/cgroup/x")), \
                mock.patch.object(sandbox, "read_text", return_value="0"), \
                mock.patch("builtins.open", mock.mock_open()):
            self.assertIn("reads", sandbox.group_the_oom_kill())

    def test_a_successful_write_reports_nothing(self) -> None:
        with mock.patch.object(sandbox, "read_cgroup_path",
                               return_value=Path("/cgroup/x")), \
                mock.patch.object(sandbox, "read_text", return_value="1"), \
                mock.patch("builtins.open", mock.mock_open()):
            self.assertEqual(sandbox.group_the_oom_kill(), "")

    def test_a_write_that_fails_reports_the_path(self) -> None:
        with mock.patch.object(sandbox, "read_cgroup_path",
                               return_value=Path("/cgroup/x")), \
                mock.patch("builtins.open",
                           side_effect=OSError(13, "Permission denied")):
            self.assertIn("Permission denied", sandbox.group_the_oom_kill())

    def test_the_session_ceiling_is_set_on_the_slice(self) -> None:
        """One session, one ceiling: the slice above the scope carries it."""
        env = {"NANOCHAT_SESSION_SLICE": "nanochat-agent-x.slice",
               "NANOCHAT_SESSION_MEM_MAX": "4G"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(
                    sandbox, "read_cgroup_path",
                    return_value=Path("/sys/fs/cgroup/slices/"
                                      "nanochat-agent-x.slice/run-1.scope")), \
                mock.patch.object(sandbox, "read_text",
                                  return_value=str(4 * 1024 ** 3)), \
                mock.patch("builtins.open", mock.mock_open()) as handle:
            self.assertEqual(sandbox.apply_session_ceiling(), "")
        self.assertEqual(handle.call_args[0][0].name, "memory.max")

    def test_a_scope_outside_its_slice_is_a_failure(self) -> None:
        env = {"NANOCHAT_SESSION_SLICE": "nanochat-agent-x.slice",
               "NANOCHAT_SESSION_MEM_MAX": "4G"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(sandbox, "read_cgroup_path",
                                  return_value=Path("/sys/fs/cgroup/app.slice"
                                                    "/run-1.scope")):
            self.assertIn("is not the session slice",
                          sandbox.apply_session_ceiling())

    def test_no_session_means_no_extra_work(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sandbox.apply_session_ceiling(), "")


if __name__ == "__main__":
    unittest.main()
