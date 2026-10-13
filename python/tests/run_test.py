"""T0: the client entry point re-enters ``tools/nanochat client``.

Hermetic: a recording tool stands in for the toolchain, so the test needs no
library, no Bazel, and no GPU. See docs/client-runtime-design.md section 14.
"""

from __future__ import annotations

import sys
import unittest

from nanochat_cpp import run


class _Result:
    returncode = 0


class _RecordingTool:
    """A stand-in for ``Toolchain`` that records each ``client`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def client(self, argv, *, device=None, profile="train"):
        self.calls.append({"argv": list(argv), "device": device,
                           "profile": profile})
        return _Result()


class RunTest(unittest.TestCase):
    def test_main_reenters_the_client(self) -> None:
        tool = _RecordingTool()
        code = run.main(["train.py", "--steps", "3"], tool=tool)
        self.assertEqual(code, 0)
        self.assertEqual(tool.calls[0]["argv"],
                         [sys.executable, "train.py", "--steps", "3"])
        self.assertIsNone(tool.calls[0]["device"])
        self.assertEqual(tool.calls[0]["profile"], "train")

    def test_device_is_passed_through(self) -> None:
        tool = _RecordingTool()
        code = run.main(["--device", "0,1", "train.py"], tool=tool)
        self.assertEqual(code, 0)
        self.assertEqual(tool.calls[0]["device"], "0,1")
        self.assertEqual(tool.calls[0]["argv"], [sys.executable, "train.py"])

    def test_empty_arguments_report_usage(self) -> None:
        self.assertEqual(run.main([], tool=_RecordingTool()), 2)

    def test_help_is_success(self) -> None:
        self.assertEqual(run.main(["--help"], tool=_RecordingTool()), 0)

    def test_missing_device_value(self) -> None:
        self.assertEqual(run.main(["--device"], tool=_RecordingTool()), 2)


if __name__ == "__main__":
    unittest.main()
