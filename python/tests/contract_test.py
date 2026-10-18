"""T0: the observable contract of ``tools/nanochat``.

Pin the behavior that the language move must not change
(``docs/script-consolidation-plan.md`` section 7, P2). The suite runs the real
entry script and the real sandbox launcher, so it is hermetic but not a fake:
it needs no library, no Bazel run, no GPU, and no network.

It covers the command list, the ``doctor --json`` key set, the sandbox
interface, the ``/tmp/nanochat-sandbox.log`` line, and the ``--dry-run`` plan.
The ``/tmp/nanochat-gpu.log`` line belongs to
``python/tests/gpu_queue_test.py::test_gpu_log_line_format``, which owns the
queue store.

A fake ``bazelisk`` on ``PATH`` keeps the entry script's startup lookup happy
without letting the test run Bazel.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _entry, cli, toolchain

#: The frozen ``doctor --json`` keys.
DOCTOR_KEYS = (
    "repo", "revision", "branch", "nvcc", "bazel", "gpu", "gpu_procs",
    "sandbox", "sandbox_backend", "gpu_queue", "sim",
)

#: Every fixture kind that the merged generator must keep.
GENERATOR_COMMANDS = (
    "oracle", "train", "rl", "generate", "eval", "plan",
    "tokenizer", "loader", "chat", "api", "numerics",
)

#: The frozen line that the ``none`` backend appends to the sandbox log.
SANDBOX_LOG_NONE_LINE = "sandbox: backend=none profile=t0-cpu (not sandboxed)"

#: The profile values that the sandbox exports. The contract pins the sentinel
#: ``NANOCHAT_SANDBOX`` and the informal ``NANOCHAT_SANDBOX_PROFILE``.
SANDBOX_ENV_KEYS = ("NANOCHAT_SANDBOX", "NANOCHAT_SANDBOX_PROFILE",
                    "NANOCHAT_SANDBOX_BACKEND")

FAKE_BAZELISK = """#!/bin/sh
# A stub so the entry script's startup lookup succeeds without Bazel.
exit 0
"""


def _usage_names(text: str) -> tuple[str, ...]:
    """The command names in the ``tools/nanochat help`` usage block.

    A command line starts with exactly two spaces, and the name starts in
    column three. A continuation line is indented further, so it does not
    match.
    """
    names = []
    for line in text.splitlines():
        match = re.match(r"^  ([a-z][a-z0-9_-]*)", line)
        if match:
            names.append(match.group(1))
    return tuple(names)


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The runfiles tree keeps the repository layout, so the same file
        # position gives the root in a Bazel test and in a direct run.
        cls.root = Path(__file__).resolve().parents[2]
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_contract_")
        cls.work = Path(cls.tmp.name)
        cls.bin_dir = cls.work / "bin"
        cls.bin_dir.mkdir()
        shim = cls.bin_dir / "bazelisk"
        shim.write_text(FAKE_BAZELISK, encoding="utf-8")
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
        cls.entry = cls.root / "tools" / "nanochat"
        cls.sandbox = cls.root / "tools" / "sandbox.sh"
        cls.env = dict(os.environ)
        for name in ("NANOCHAT_SANDBOX", "NANOCHAT_SANDBOX_BACKEND",
                     "NANOCHAT_SANDBOX_PROFILE", "NANOCHAT_GPU_QUEUED"):
            cls.env.pop(name, None)
        cls.env["PATH"] = f"{cls.bin_dir}:{os.environ.get('PATH', '')}"
        # Keep the queue store and the GPU log out of the real host state.
        cls.env["NANOCHAT_GPU_QUEUE_DIR"] = str(cls.work / "queue")
        cls.env["NANOCHAT_GPU_LOG"] = str(cls.work / "gpu.log")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_script(self, script: Path, *argv: str,
                   env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(script), *argv], cwd=str(self.root), env=env or self.env,
            capture_output=True, text=True, timeout=300, check=False)

    def plan(self, *argv: str) -> list[dict]:
        """Run ``--dry-run`` and return the parsed plan lines."""
        result = self.run_script(self.entry, *argv)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()
                if line.strip()]

    @staticmethod
    def _bazel_line(plan: list[dict]) -> list[str]:
        return next(step["argv"] for step in plan
                    if step["layer"] == "bazel")

    @staticmethod
    def _run_under(bazel_argv: list[str]) -> str:
        return next(value for value in bazel_argv
                    if value.startswith("--run_under="))

    def test_command_list(self) -> None:
        """The shell command list equals the one command table."""
        result = self.run_script(self.entry, "help")
        self.assertEqual(result.returncode, 0, result.stderr)
        names = _usage_names(result.stdout)
        self.assertEqual(names, _entry.names())

    def test_handler_table_matches_the_command_table(self) -> None:
        """The CLI has one handler per command table row, in table order."""
        names = _entry.names()
        self.assertEqual(tuple(cli.COMMANDS), names)
        self.assertEqual(len(names), len(set(names)), "a command name repeats")

    def test_doctor_json_keys(self) -> None:
        result = self.run_script(self.entry, "doctor", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(sorted(payload), sorted(DOCTOR_KEYS))
        self.assertIsInstance(payload["gpu_queue"], dict)

    def test_unknown_command_is_an_error(self) -> None:
        result = self.run_script(self.entry, "nonsense")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command", result.stderr)

    def test_sandbox_accepts_both_profile_forms(self) -> None:
        """``--profile NAME`` and ``--profile=NAME`` select the same profile."""
        probe = ("import json, os; print(json.dumps({k: os.environ.get(k) "
                 f"for k in {list(SANDBOX_ENV_KEYS)}}}))")
        outputs = []
        for form in (["--profile", "t0-cpu", "--"], ["--profile=t0-cpu", "--"]):
            result = self.run_script(
                self.sandbox, "--backend", "none", *form,
                "python3", "-c", probe)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["NANOCHAT_SANDBOX_PROFILE"], "t0-cpu")
        self.assertEqual(outputs[0]["NANOCHAT_SANDBOX"], "off")
        self.assertEqual(outputs[0]["NANOCHAT_SANDBOX_BACKEND"], "none")

    def test_sandbox_rejects_an_unknown_profile(self) -> None:
        result = self.run_script(
            self.sandbox, "--backend", "none", "--profile", "nonsense", "--",
            "true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown profile", result.stderr)

    def test_sandbox_log_line_format(self) -> None:
        """``/tmp/nanochat-sandbox.log`` keeps its ``none`` backend line."""
        self.run_script(
            self.sandbox, "--backend", "none", "--profile", "t0-cpu", "--",
            "true")
        log = Path("/tmp/nanochat-sandbox.log")
        self.assertTrue(log.is_file(), "the sandbox wrote no log")
        self.assertIn(SANDBOX_LOG_NONE_LINE,
                      log.read_text(encoding="utf-8").splitlines())

    def test_dry_run_argv_matches_the_python_argument_layer(self) -> None:
        """The ``argv`` plan line equals the list ``toolchain.py`` builds."""
        recorded: list[tuple[str, ...]] = []

        class Recorder(toolchain.Toolchain):
            def _run(self, argv, check=None):
                arguments = tuple(str(value) for value in argv)
                recorded.append(arguments)
                return toolchain.CommandResult(arguments, 0, "", "", 0.0)

        tool = Recorder(entry=self.entry, repo_root=self.root, env=self.env)
        tool.build(["//src:train_main"])
        tool.test(["//bindings:all"], gpu=True)
        tool.check(gpu=True)
        tool.run("t2-parity", ["binary"])
        for arguments in recorded:
            plan = self.plan("--dry-run", *arguments)
            self.assertEqual(plan[0], {"layer": "argv",
                                       "argv": list(arguments)})

    def test_dry_run_test_run_under_strings(self) -> None:
        """The CPU and GPU test paths keep their ``--run_under`` wrapper."""
        cpu = self._bazel_line(self.plan("--dry-run", "test", "//bindings:all"))
        self.assertTrue(self._run_under(cpu).endswith(
            "sandbox.sh --profile=t0-cpu --"), self._run_under(cpu))
        self.assertIn("--test_tag_filters=-gpu", cpu)
        gpu = self._bazel_line(self.plan("--dry-run", "test", "--gpu",
                                         "//bindings:all"))
        self.assertTrue(self._run_under(gpu).endswith(
            "gpuq submit --profile=t1-gpu --"), self._run_under(gpu))
        self.assertIn("--test_tag_filters=gpu", gpu)
        self.assertIn("--config=cuda", gpu)

    def test_dry_run_plans_the_sandbox_and_the_queue(self) -> None:
        sandbox = self.plan("--dry-run", "run", "t0-cpu", "--", "binary")
        self.assertEqual(sandbox[1]["layer"], "sandbox")
        self.assertEqual(sandbox[1]["argv"][1:],
                         ["--profile", "t0-cpu", "--", "binary"])
        queue = self.plan("--dry-run", "run", "t2-parity", "--", "binary")
        self.assertEqual(queue[1]["layer"], "gpuq")
        self.assertEqual(queue[1]["argv"][1:5],
                         ["gpuq", "submit", "--profile", "t2-parity"])

    def test_dry_run_covers_five_commands_only(self) -> None:
        plan = self.plan("--dry-run", "simulate", "--device", "h100",
                         "--suite", "api")
        self.assertEqual(plan[0], {
            "layer": "argv",
            "argv": ["simulate", "--device", "h100", "--suite", "api"]})
        self.assertEqual({step["layer"] for step in plan[1:]},
                         {"bazel", "host"})
        rejected = self.run_script(self.entry, "--dry-run", "lint")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("--dry-run", rejected.stderr)

    def test_fixture_generator_subcommands(self) -> None:
        """The merged generator keeps every fixture kind."""
        generator = self.root / "tools" / "dump_fixtures.py"
        # The generator imports its sibling `fixture_common`. A py_test may set
        # PYTHONSAFEPATH, which drops the script directory from `sys.path`, so
        # name the directory here.
        env = dict(self.env)
        env.pop("PYTHONSAFEPATH", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.root / "tools"), env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep)
        result = self.run_script(generator, "--help", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        match = re.search(r"\{([^}]*)\}", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertEqual(tuple(match.group(1).split("|")), GENERATOR_COMMANDS)


if __name__ == "__main__":
    unittest.main()
