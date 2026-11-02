"""T0: the four wrappers of this skill reach the loop they name.

The wrappers own almost nothing, so the test pins the one thing they do own: the
import path from a checkout, and the subcommand each wrapper forwards to.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPERS = ("trace", "measure", "analyze", "compare")


class WrapperTest(unittest.TestCase):
    def test_every_wrapper_prints_its_own_usage(self) -> None:
        for name in WRAPPERS:
            result = subprocess.run(
                [sys.executable, str(HERE / f"{name}.py"), "--help"],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
            self.assertIn(name, result.stdout)

    def test_a_wrapper_forwards_to_the_named_subcommand(self) -> None:
        # A target that does not exist is the loop's own fault message, which
        # proves the wrapper reached it with the right subcommand.
        result = subprocess.run(
            [sys.executable, str(HERE / "compare.py"), "--stage", "base",
             "--device", "h100", "--root", "/tmp/pgo-does-not-exist"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("pgo:", result.stderr)
        self.assertIn("loop", result.stderr)

    def test_the_repository_is_found_from_the_wrapper(self) -> None:
        sys.path.insert(0, str(HERE))
        import _cli  # noqa: PLC0415 - the module under test

        self.assertTrue((_cli.ROOT / "MODULE.bazel").is_file())
        self.assertTrue((_cli.ROOT / "python" / "nanochat_cpp" / "pgo.py")
                        .is_file())


if __name__ == "__main__":
    unittest.main()
