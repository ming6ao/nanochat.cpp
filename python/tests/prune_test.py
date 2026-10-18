"""T0: the Bazel output-base reaper, ``python/nanochat_cpp/prune.py``.

Hermetic: the test builds a synthetic Bazel output user root under a temporary
directory and asserts the classification and removal semantics. It needs no
GPU and no network, and it never touches the real ``~/.cache/bazel``, because
every call passes ``--output-root``.

A live Bazel server is simulated with a process whose executable name is
``java`` and whose command line carries ``A-server.jar`` and
``--output_base=<base>``. That is exactly what the liveness scan looks for.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nanochat_cpp import prune


class PruneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="nanochat_prune_"))
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.server: subprocess.Popen | None = None

    def tearDown(self) -> None:
        self.stop_server()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stop_server(self) -> None:
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None

    def make_base(self, workspace: Path) -> Path:
        """The output base that Bazel names for ``workspace``."""
        name = hashlib.md5(str(workspace).encode("utf-8")).hexdigest()
        base = self.root / name
        (base / "execroot").mkdir(parents=True)
        (base / "execroot" / "DO_NOT_BUILD_HERE").write_text(
            str(workspace), encoding="utf-8")
        return base

    def run_prune(self, *argv: str) -> tuple[int, str, str]:
        """Run the reaper over the synthetic root and capture the output."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = prune.main(["--output-root", str(self.root), *argv])
            except SystemExit as exit_code:
                code = int(exit_code.code or 0)
        return code, out.getvalue(), err.getvalue()

    def fake_server(self, base: Path) -> None:
        """Start a process that the liveness scan reads as a live server."""
        java = self.tmp / "java"
        shutil.copy("/bin/bash", java)
        # The trailing `true` stops bash from replacing itself with `sleep`, so
        # the process keeps the executable name `java` and the full command
        # line, exactly as a Bazel server does.
        self.server = subprocess.Popen(
            ["A-server.jar", "-c", "sleep 30; true",
             f"--output_base={base}"], executable=str(java))
        time.sleep(0.3)

    def test_classification_and_removal(self) -> None:
        live_ws = self.tmp / "live-ws" / "nanochat.cpp"
        wtree_ws = self.tmp / "wtree-keep" / "nanochat.cpp"
        other_ws = self.tmp / "other-ws" / "nanochat.cpp"
        for workspace in (live_ws, wtree_ws, other_ws):
            workspace.mkdir(parents=True)
        base_live = self.make_base(live_ws)
        base_gone = self.make_base(self.tmp / "gone-ws" / "nanochat.cpp")
        base_wtree = self.make_base(wtree_ws)
        base_other = self.make_base(other_ws)
        base_fake = self.make_base(self.tmp / "fake-ws" / "nanochat.cpp")

        # 1. A global dry run targets only the vanished workspaces.
        code, out, err = self.run_prune()
        self.assertEqual(code, 0, err)
        self.assertIn(str(base_gone), out)
        self.assertIn(str(base_fake), out)
        self.assertNotIn(str(base_live), out)
        self.assertNotIn(str(base_other), out)
        self.assertTrue(base_gone.is_dir(), "a dry run must remove nothing")

        # 2. --worktree scopes to a directory, existing or not.
        code, out, err = self.run_prune("--worktree", str(wtree_ws.parent))
        self.assertEqual(code, 0, err)
        self.assertIn(str(base_wtree), out)
        self.assertNotIn(str(base_gone), out)

        # 3. --json reports the status and the workspace.
        code, out, err = self.run_prune("--json")
        self.assertEqual(code, 0, err)
        self.assertIn('"status": "target"', out)
        self.assertIn('"workspace":', out)
        self.assertIn(str(self.tmp / "gone-ws" / "nanochat.cpp"), out)

        # 4. A base held by a live server is skipped, even with --apply.
        self.fake_server(base_fake)
        code, out, err = self.run_prune("--apply")
        self.assertEqual(code, 0, err)
        self.assertIn(f"skip (live server) {base_fake}", out)
        self.assertIn(f"removed {base_gone}", out)
        self.assertTrue(base_fake.is_dir(),
                        "a live-server base must survive --apply")
        self.assertFalse(base_gone.is_dir(),
                         "--apply must remove the non-live orphan")
        self.stop_server()

        # 5. Once the server is gone, the last orphan is reaped.
        code, out, err = self.run_prune("--apply")
        self.assertEqual(code, 0, err)
        self.assertIn(f"removed {base_fake}", out)
        self.assertFalse(base_fake.is_dir())
        self.assertTrue(base_live.is_dir())
        self.assertTrue(base_wtree.is_dir())
        self.assertTrue(base_other.is_dir())

        # 6. An empty sweep prints an empty array.
        code, out, err = self.run_prune("--json")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "[]")

    def test_error_paths(self) -> None:
        code, _, _ = self.run_prune("--nonsense")
        self.assertNotEqual(code, 0, "an unknown argument must fail")
        missing = self.tmp / "does-not-exist"
        code, _, err = self.run_prune("--output-root", str(missing))
        self.assertNotEqual(code, 0, "a missing output root must fail")
        self.assertIn("no output user root", err)


if __name__ == "__main__":
    unittest.main()
