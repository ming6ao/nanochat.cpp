"""T0: the orchestration layer builds the right ``tools/nanochat`` argv.

The test is hermetic: it needs no library, no Bazel, and no GPU. It writes a
fake entry script that echoes its argument list as JSON, then checks the argv
for every command in the table. See docs/python.md sections 6 and 11.
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

FAKE_ENTRY = """#!{python}
import json
import sys

argv = sys.argv[1:]
if argv and argv[0] == "doctor":
    print(json.dumps({{"repo": "/fake", "revision": "abc", "branch": "main",
                       "nvcc": "release 12.0", "gpu": "fake gpu",
                       "sandbox_backend": "systemd", "gpu_lock": "free"}}))
elif argv and argv[0] == "shutdown":
    sys.stderr.write("boom\\n")
    sys.exit(3)
else:
    print(json.dumps(argv))
"""


class ToolchainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_toolchain_")
        cls.entry = Path(cls.tmp.name) / "fake-nanochat"
        cls.entry.write_text(FAKE_ENTRY.format(python=sys.executable),
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

    def test_gpu_inside_sandbox_skips_the_broker(self) -> None:
        env = dict(self.env)
        env["NANOCHAT_SANDBOX"] = "t2-parity"
        inside = toolchain.Toolchain(entry=self.entry, repo_root=self.tmp.name,
                                     env=env)
        self.check(inside.gpu(["binary"]), ["binary"])

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
        self.assertEqual(report.gpu_lock, "free")
        self.assertTrue(report.raw)


if __name__ == "__main__":
    unittest.main()
