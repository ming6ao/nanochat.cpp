"""T0: the resource sandbox launcher and the limit verifier.

The launcher runs as `--run_under` for every CPU test action, so this suite is
the only T0 test of the trust path. It is hermetic and CPU-only.

The `none` backend needs no cgroup and no systemd, so most cases run there. The
limit comparison is tested through its pure function, because a real cgroup
read needs a scope. The contract suite runs the two real modes.

See docs/sandbox.md and docs/python-sandbox-plan.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from nanochat_cpp import sandbox

#: The fixed log line that the `none` backend appends.
NONE_LINE = "sandbox: backend=none profile=t0-cpu (not sandboxed)"

#: The environment keys the launcher sets. The contract pins the whole set.
PROBE_KEYS = ("NANOCHAT_SANDBOX", "NANOCHAT_SANDBOX_PROFILE",
              "NANOCHAT_SANDBOX_BACKEND", "OMP_NUM_THREADS",
              "NANOCHAT_NUM_THREADS", "CUDA_VISIBLE_DEVICES")

PROBE = ("import json, os; print(json.dumps({k: os.environ.get(k) "
         f"for k in {list(PROBE_KEYS)}}}))")


class SandboxTest(unittest.TestCase):
    """The launcher interface, the frozen lines, and the limit check."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = Path(sandbox.__file__).resolve()
        cls.env = {key: value for key, value in os.environ.items()
                   if not key.startswith("NANOCHAT_")}
        cls.env.pop("CUDA_VISIBLE_DEVICES", None)

    def run_launcher(self, *argv: str, env: dict | None = None,
                     probe: bool = False) -> subprocess.CompletedProcess:
        command = [sys.executable, str(self.module), *argv]
        if probe:
            command += [sys.executable, "-c", PROBE]
        return subprocess.run(command, capture_output=True, text=True,
                              env=env or self.env, timeout=120, check=False)

    # -- the arguments ----------------------------------------------------

    def test_both_profile_forms_select_one_profile(self) -> None:
        outputs = []
        for form in (["--profile", "t0-cpu", "--"], ["--profile=t0-cpu", "--"]):
            result = self.run_launcher("--backend", "none", *form, probe=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["NANOCHAT_SANDBOX_PROFILE"], "t0-cpu")

    def test_none_backend_sets_the_sentinel_and_the_profile(self) -> None:
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu",
                                   "--", probe=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(values["NANOCHAT_SANDBOX"], "off")
        self.assertEqual(values["NANOCHAT_SANDBOX_PROFILE"], "t0-cpu")
        self.assertEqual(values["NANOCHAT_SANDBOX_BACKEND"], "none")
        # `nproc` respects the affinity mask; os.cpu_count() would not.
        threads = max(len(os.sched_getaffinity(0)), 1)
        self.assertEqual(values["OMP_NUM_THREADS"], str(threads))
        self.assertEqual(values["NANOCHAT_NUM_THREADS"], str(threads))

    def test_none_backend_appends_the_frozen_log_line(self) -> None:
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu",
                                   "--", "true")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = sandbox.SANDBOX_LOG.read_text(encoding="utf-8").splitlines()
        self.assertIn(NONE_LINE, lines)
        self.assertIn(NONE_LINE, result.stderr)

    def test_quiet_hides_the_progress_line_but_keeps_the_log(self) -> None:
        env = dict(self.env, NANOCHAT_SANDBOX_QUIET="1")
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu",
                                   "--", "true", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("not sandboxed", result.stderr)
        lines = sandbox.SANDBOX_LOG.read_text(encoding="utf-8").splitlines()
        self.assertIn(NONE_LINE, lines)

    def test_the_devices_translation_fills_only_an_empty_slot(self) -> None:
        env = dict(self.env, NANOCHAT_GPU_DEVICES="3")
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu",
                                   "--", probe=True, env=env)
        values = json.loads(result.stdout)
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "3")
        chosen = dict(env, CUDA_VISIBLE_DEVICES="1")
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu",
                                   "--", probe=True, env=chosen)
        values = json.loads(result.stdout)
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "1")

    # -- the errors -------------------------------------------------------

    def test_unknown_profile_is_an_error(self) -> None:
        result = self.run_launcher("--backend", "none", "--profile",
                                   "nonsense", "--", "true")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr,
                         "sandbox: unknown profile 'nonsense'\n")

    def test_unknown_option_is_an_error(self) -> None:
        result = self.run_launcher("--backend", "none", "--bogus", "--", "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option '--bogus'", result.stderr)

    def test_a_missing_command_is_an_error(self) -> None:
        result = self.run_launcher("--backend", "none", "--profile", "t0-cpu")
        self.assertEqual(result.returncode, 2)
        self.assertIn("sandbox: no command given", result.stderr)

    def test_a_bad_backend_is_an_error(self) -> None:
        result = self.run_launcher("--backend", "bogus", "--", "true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("backend must be systemd, none, or auto", result.stderr)

    def test_a_missing_option_value_is_an_error(self) -> None:
        result = self.run_launcher("--profile")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--profile needs a name", result.stderr)

    # -- the verifier -----------------------------------------------------

    def test_verify_refuses_an_unapplied_budget(self) -> None:
        """Outside a scope the limits never match, so the run must stop."""
        result = subprocess.run(
            [sys.executable, str(self.module), "verify",
             "2G", "800%", "0-1", "128", "--", "true"],
            capture_output=True, text=True, env=self.env, timeout=120,
            check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("sandbox: limits not applied:", result.stderr)
        self.assertIn("refusing to run the command unsandboxed.", result.stderr)

    def test_verify_rejects_a_size_it_cannot_parse(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.module), "verify",
             "xM", "200%", "0-1", "128", "--", "true"],
            capture_output=True, text=True, env=self.env, timeout=120,
            check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("sandbox: cannot parse size 'xM'", result.stderr)

    def test_verify_needs_a_command(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.module), "verify",
             "2G", "200%", "0-1", "128"],
            capture_output=True, text=True, env=self.env, timeout=120,
            check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("sandbox: no command given", result.stderr)

    def test_the_verifier_clears_the_core_limit(self) -> None:
        """The job inherits `RLIMIT_CORE=0`, so no core lands in a worktree."""
        probe = ("from nanochat_cpp import sandbox; import resource; "
                 "sandbox.drop_core_limit(); "
                 "print(resource.getrlimit(resource.RLIMIT_CORE))")
        # The child needs the package on the path, as the tool's entry point
        # sets it. The test action's own PYTHONPATH is not inherited here.
        python_dir = str(self.module.parents[1])
        env = dict(self.env, PYTHONPATH=python_dir)
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True, env=env,
                                timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "(0, 0)")

    # -- the pure functions -----------------------------------------------

    def test_the_profile_table_holds_the_eight_profiles(self) -> None:
        profiles = sandbox.load_profiles()["profiles"]
        self.assertEqual(sorted(profiles), [
            "eval", "kaggle-cpu", "kaggle-gpu", "t0-cpu", "t1-gpu",
            "t2-parity", "t3-bench", "train"])
        self.assertEqual(profiles["t0-cpu"], {
            "quota": "200%", "mem_high": "1536M", "mem_max": "2G",
            "pids": "128", "cpus": "0-1", "runtime": None})
        self.assertEqual(sandbox.load_profiles()["slice"], "nanochat.slice")

    def test_an_unknown_profile_raises(self) -> None:
        profiles = sandbox.load_profiles()
        with self.assertRaises(sandbox.UnknownProfile):
            sandbox.profile_values(profiles, "nonsense")

    def test_to_bytes(self) -> None:
        self.assertEqual(sandbox.to_bytes("2"), 2)
        self.assertEqual(sandbox.to_bytes("2K"), 2 * 1024)
        self.assertEqual(sandbox.to_bytes("1536M"), 1536 * 1024 * 1024)
        self.assertEqual(sandbox.to_bytes("2G"), 2 * 1024 ** 3)
        self.assertEqual(sandbox.to_bytes("2g"), 2 * 1024 ** 3)
        with self.assertRaises(ValueError):
            sandbox.to_bytes("xM")

    def test_cpu_count(self) -> None:
        self.assertEqual(sandbox.cpu_count("0-1"), 2)
        self.assertEqual(sandbox.cpu_count("0-11"), 12)
        self.assertEqual(sandbox.cpu_count("0,2"), 2)
        self.assertEqual(sandbox.cpu_count("0-3,8-9"), 6)

    def test_check_limits_reports_every_mismatch(self) -> None:
        good = {"mem": "2", "cpu": "200000 100000", "pids": "128",
                "cpus": "0-1"}
        self.assertEqual(sandbox.check_limits(good, dict(good)), "")
        actual = {"mem": "max", "cpu": "max 100000", "pids": "max",
                  "cpus": "0-11"}
        self.assertEqual(
            sandbox.check_limits(good, actual),
            " memory.max=max(want 2) cpu.max='max 100000'(want '200000 "
            "100000') pids.max=max(want 128) cpus=0-11(want 0-1)")
        self.assertEqual(
            sandbox.check_limits(good, {"mem": "max", "cpu": good["cpu"],
                                       "pids": good["pids"],
                                       "cpus": good["cpus"]}),
            " memory.max=max(want 2)")

    def test_the_launcher_is_executable(self) -> None:
        """The `--run_under` string names this file, so the bit must be set."""
        self.assertTrue(os.access(self.module, os.X_OK))


if __name__ == "__main__":
    unittest.main()
