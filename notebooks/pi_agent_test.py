"""Tests for the Pi event-stream runner."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import pi_agent


class _FakeProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class RunPiTest(unittest.TestCase):
    def test_prints_text_and_tool_progress_and_removes_hf_token(self) -> None:
        records = [
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "delta": "Checking the repository.",
                },
            },
            {"type": "tool_execution_start", "toolName": "read"},
            {"type": "tool_execution_end", "toolName": "read"},
            {"type": "agent_settled"},
        ]
        stdout = "".join(json.dumps(record) + "\n" for record in records)
        process = _FakeProcess(stdout, stderr="warning\n")
        with (
            mock.patch.dict(os.environ, {"HF_TOKEN": "not-forwarded"}),
            mock.patch.object(pi_agent.shutil, "which", return_value="/usr/bin/pi"),
            mock.patch.object(pi_agent.subprocess, "Popen", return_value=process) as popen,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            result = pi_agent.run_pi("Inspect the repository.", cwd=Path("/repo"))

        self.assertEqual(result, 0)
        self.assertIn("Checking the repository.", output.getvalue())
        self.assertIn("starting tool: read", output.getvalue())
        self.assertIn("finished tool: read", output.getvalue())
        self.assertIn("[pi stderr] warning", output.getvalue())
        arguments = popen.call_args.args[0]
        self.assertEqual(arguments[-1], "Inspect the repository.")
        self.assertEqual(popen.call_args.kwargs["cwd"], "/repo")
        self.assertNotIn("HF_TOKEN", popen.call_args.kwargs["env"])

    def test_raises_on_pi_error(self) -> None:
        process = _FakeProcess("", returncode=3)
        with (
            mock.patch.object(pi_agent.shutil, "which", return_value="/usr/bin/pi"),
            mock.patch.object(pi_agent.subprocess, "Popen", return_value=process),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo")

    def test_requires_pi_cli(self) -> None:
        with (
            mock.patch.object(pi_agent.shutil, "which", return_value=None),
            self.assertRaises(FileNotFoundError),
        ):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo")

    def test_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            pi_agent.run_pi("  ", cwd="/repo")

    def test_rejects_invalid_heartbeat(self) -> None:
        with self.assertRaises(ValueError):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            heartbeat_seconds=0)


if __name__ == "__main__":
    unittest.main()
