"""T0: the orchestration layer builds the right ``tools/nanochat`` argv.

The test is hermetic: it needs no library, no Bazel, and no GPU. It writes a
fake entry script that echoes its argument list as JSON, then checks the argv
for every command in the table, and it gives that script a fake
``doctor --json`` report (environment plus the device-and-graph plan) so the
parser assertions describe a real report. See docs/python.md sections 6 and 11.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import _entry, toolchain

#: Why a fused family won its composition (``src/plan.cc``): the device
#: declares the bit that gates it, so the composition path is what lost.
REASON_FUSED_WINS = ('the device declares override_rms_norm and the fused '
                     'family "rms_norm" implements the same composition faster')

#: Why a registered family lost (``src/plan.cc``): the bit is the switch, so a
#: family that landed before its gate is declined by name instead of chosen.
REASON_BIT_CLEAR = ('the device does not declare override_qk_prep, so this '
                    'fused family is not offered here')

#: The fake ``doctor --json`` report. The ``plan`` object is the graph's plan
#: that ``//src:plan_main --json`` prints, wrapped in the availability fields
#: ``cli._plan_status`` adds, so no build runs and no GPU is needed.
FAKE_ENTRY = """#!{python}
import json
import sys

REASON_FUSED_WINS = {fused_wins!r}
REASON_BIT_CLEAR = {bit_clear!r}

argv = sys.argv[1:]
if argv and argv[0] == "doctor":
    plan = {{"available": True, "reason": "", "ok": True,
            "spec": "nanochat-default", "graph": "train",
            "device_index": 0, "gaps": [], "why_not": "",
            "choices": [
                {{"composition": "rms_norm", "implementation": "rms_norm",
                  "fused": True,
                  "declined": [{{"implementation": "composition",
                                "reason": REASON_FUSED_WINS}}]}},
                {{"composition": "qk_prep", "implementation": "composition",
                  "fused": False,
                  "declined": [{{"implementation": "qk_prep",
                                "reason": REASON_BIT_CLEAR}}]}}]}}
    print(json.dumps({{"repo": "/fake", "revision": "abc", "branch": "main",
                       "nvcc": "release 12.0", "gpu": "fake gpu",
                       "sandbox_backend": "systemd",
                       "gpu_queue": {{"worker_id": "w1", "alive": True,
                                     "devices": "0", "depth": 0,
                                     "running_job": None, "queued": 0}},
                       "plan": plan}}))
elif argv and argv[0] == "shutdown":
    sys.stderr.write("boom\\n")
    sys.exit(3)
else:
    print(json.dumps(argv))
"""

#: The keys the ``doctor --json`` report's ``plan`` object carries, in the order
#: ``cli._plan_status`` builds them. Append-only, like ``cli.DOCTOR_KEYS``.
DOCTOR_PLAN_KEYS = (
    "available", "reason", "ok", "spec", "graph", "device_index",
    "choices", "gaps", "why_not",
)


class ToolchainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_toolchain_")
        cls.entry = Path(cls.tmp.name) / "fake-nanochat"
        cls.entry.write_text(
            FAKE_ENTRY.format(python=sys.executable,
                              fused_wins=REASON_FUSED_WINS,
                              bit_clear=REASON_BIT_CLEAR),
            encoding="utf-8")
        cls.entry.chmod(cls.entry.stat().st_mode | stat.S_IEXEC)
        cls.env = dict(os.environ)
        cls.env.pop("NANOCHAT_SANDBOX", None)
        cls.toolchain = toolchain.Toolchain(
            entry=cls.entry, repo_root=cls.tmp.name, env=cls.env)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def argv_of(self, result) -> list:
        return json.loads(result.stdout)

    def check(self, result, expected) -> None:
        self.assertEqual(self.argv_of(result), expected)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(list(result.argv), expected)

    def test_build(self) -> None:
        self.check(self.toolchain.build(), ["build"])
        self.check(self.toolchain.build(["//src:train_main"]),
                   ["build", "//src:train_main"])

    def test_test(self) -> None:
        self.check(self.toolchain.test(), ["test"])
        self.check(self.toolchain.test(["//bindings:all"], gpu=True),
                   ["test", "--gpu", "//bindings:all"])

    def test_check(self) -> None:
        self.check(self.toolchain.check(), ["check"])
        self.check(self.toolchain.check(gpu=True), ["check", "--gpu"])

    def test_lint(self) -> None:
        self.check(self.toolchain.lint(), ["lint"])
        self.check(self.toolchain.lint(["src/train.cc"]),
                   ["lint", "src/train.cc"])

    def test_shutdown(self) -> None:
        with self.assertRaises(toolchain.ToolError) as caught:
            self.toolchain.shutdown()
        self.assertEqual(caught.exception.returncode, 3)
        self.assertEqual(list(caught.exception.argv), ["shutdown"])

    def test_prune(self) -> None:
        self.check(self.toolchain.prune(), ["prune"])
        self.check(self.toolchain.prune(apply=True, worktree="/tmp/w"),
                   ["prune", "--apply", "--worktree", "/tmp/w"])

    def test_run_and_aliases(self) -> None:
        self.check(self.toolchain.run("t2-parity", ["binary", "--x"]),
                   ["run", "t2-parity", "--", "binary", "--x"])
        self.check(self.toolchain.train(["binary"]),
                   ["train", "--", "binary"])
        # The RL entry is an alias of the `train` profile.
        self.check(self.toolchain.rl(["rl_worker"]),
                   ["train", "--", "rl_worker"])
        self.check(self.toolchain.eval(["binary"]),
                   ["eval", "--", "binary"])
        self.check(self.toolchain.bench(["binary"]),
                   ["bench", "--", "binary"])
        self.check(self.toolchain.verify(["binary"]),
                   ["verify", "--", "binary"])

    def test_profile(self) -> None:
        self.check(self.toolchain.profile(), ["profile"])
        self.check(self.toolchain.profile(["--json"]), ["profile", "--json"])

    def test_gpu(self) -> None:
        self.check(self.toolchain.gpu(["binary"]),
                   ["gpu", "--", "binary"])
        self.check(self.toolchain.gpu(["binary"], profile="t1-gpu"),
                   ["gpu", "--profile", "t1-gpu", "--", "binary"])

    def test_dist(self) -> None:
        self.check(
            self.toolchain.dist(["binary", "--log", "rank{rank}.log"],
                                nproc_per_node=2, device="0,1",
                                master="127.0.0.1", port=29500,
                                profile="train", log_dir="/tmp/run", json=True),
            ["dist", "--nproc-per-node", "2", "--device", "0,1",
             "--profile", "train", "--master", "127.0.0.1",
             "--port", "29500", "--log-dir", "/tmp/run", "--json", "--",
             "binary", "--log", "rank{rank}.log"])
        self.check(self.toolchain.dist(["binary"]),
                   ["dist", "--nproc-per-node", "2", "--profile", "train",
                    "--master", "127.0.0.1", "--port", "29500", "--",
                    "binary"])

    def test_client(self) -> None:
        self.check(
            self.toolchain.client(["python", "train.py"], device="0,1"),
            ["client", "--device", "0,1", "--profile", "train", "--",
             "python", "train.py"])
        self.check(self.toolchain.client(["python", "train.py"]),
                   ["client", "--profile", "train", "--", "python",
                    "train.py"])

    def test_gpu_inside_sandbox_skips_the_queue(self) -> None:
        env = dict(self.env)
        env["NANOCHAT_SANDBOX"] = "t2-parity"
        inside = toolchain.Toolchain(entry=self.entry, repo_root=self.tmp.name,
                                     env=env)
        self.check(inside.gpu(["binary"]), ["binary"])
        queued_env = dict(self.env)
        queued_env["NANOCHAT_GPU_QUEUED"] = "1"
        queued = toolchain.Toolchain(entry=self.entry, repo_root=self.tmp.name,
                                     env=queued_env)
        self.check(queued.gpu(["binary"]), ["binary"])

    def test_gpuq(self) -> None:
        self.check(self.toolchain.gpuq(["status", "--json"]),
                   ["gpuq", "status", "--json"])
        self.check(self.toolchain.gpuq(["submit", "--", "binary"]),
                   ["gpuq", "submit", "--", "binary"])

    def test_dispatch_and_commands(self) -> None:
        self.check(self.toolchain.dispatch("build", "//a"),
                   ["build", "//a"])
        self.assertEqual(self.toolchain.commands(), _entry.names())
        with self.assertRaises(KeyError):
            self.toolchain.dispatch("nonsense")

    def test_doctor_json(self) -> None:
        report = self.toolchain.doctor()
        self.assertEqual(report.repo, "/fake")
        self.assertEqual(report.revision, "abc")
        self.assertEqual(report.branch, "main")
        self.assertEqual(report.nvcc, "release 12.0")
        self.assertEqual(report.sandbox_backend, "systemd")
        self.assertEqual(report.gpu_queue["worker_id"], "w1")
        self.assertTrue(report.gpu_queue["alive"])
        self.assertTrue(report.raw)

    def test_doctor_json_reports_the_plan(self) -> None:
        """The report names the chosen implementation and the declined ones.

        Section 9 rule 6 of ``docs/model-and-device-bringup.md`` makes the plan
        observable: a user proves why their path changed without reading
        source, and rule 4 says a plan is the chosen implementation for each
        composition plus the fallbacks it declined. The fake payload stands in
        for ``//src:plan_main --json`` in the shape ``cli._plan_status``
        publishes, so the suite stays hermetic: no Bazel run and no GPU.
        ``DoctorReport`` carries the object in ``raw``; it has no ``plan``
        field yet, which is why the assertion reads the payload.
        """
        report = self.toolchain.doctor()
        plan = json.loads(report.raw)["plan"]
        self.assertEqual(sorted(plan), sorted(DOCTOR_PLAN_KEYS))
        self.assertTrue(plan["available"])
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["spec"], "nanochat-default")
        choices = {choice["composition"]: choice
                   for choice in plan["choices"]}
        # The fused family won, and the composition path it beat is named next
        # to the reason it lost.
        fused = choices["rms_norm"]
        self.assertTrue(fused["fused"])
        self.assertEqual(fused["implementation"], "rms_norm")
        self.assertEqual([(entry["implementation"], entry["reason"])
                          for entry in fused["declined"]],
                         [("composition", REASON_FUSED_WINS)])
        # A family whose capability bit is clear is declined by name, so the
        # composition path is the choice the reader sees.
        declined = choices["qk_prep"]
        self.assertFalse(declined["fused"])
        self.assertEqual(declined["implementation"], "composition")
        self.assertEqual([(entry["implementation"], entry["reason"])
                          for entry in declined["declined"]],
                         [("qk_prep", REASON_BIT_CLEAR)])


if __name__ == "__main__":
    unittest.main()
