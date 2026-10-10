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


def _stream(*records: dict) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _text_delta(delta: str) -> dict:
    return {
        "type": "message_update",
        "assistantMessageEvent": {"type": "text_delta", "delta": delta},
    }


@contextlib.contextmanager
def _pi(process: _FakeProcess, env: dict | None = None):
    """Patch the Pi CLI, capture the stream, and yield the Popen mock."""
    output = io.StringIO()
    with (
        mock.patch.dict(os.environ, env or {}),
        mock.patch.object(pi_agent.shutil, "which",
                          return_value="/usr/bin/pi"),
        mock.patch.object(pi_agent.subprocess, "Popen",
                          return_value=process) as popen,
    ):
        yield popen, output


class RunPiTest(unittest.TestCase):
    def test_prints_text_and_tool_progress_and_removes_hf_token(self) -> None:
        process = _FakeProcess(
            _stream(
                _text_delta("Checking the repository."),
                {"type": "tool_execution_start", "toolCallId": "c1",
                 "toolName": "read"},
                {"type": "tool_execution_update", "toolCallId": "c1",
                 "toolName": "read"},
                {"type": "tool_execution_end", "toolCallId": "c1",
                 "toolName": "read", "durationMs": 12},
                {"type": "agent_settled"},
            ),
            stderr="warning\n",
        )
        with _pi(process, {"HF_TOKEN": "not-forwarded"}) as (popen, output):
            result = pi_agent.run_pi(
                "Inspect the repository.", cwd=Path("/repo"), stream=output)

        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("Checking the repository.", text)
        self.assertIn("running read", text)
        self.assertIn("tools 1 (read 1)", text)
        self.assertIn("[pi stderr] warning", text)
        arguments = popen.call_args.args[0]
        self.assertEqual(arguments[-2:], ["--", "Inspect the repository."])
        self.assertEqual(popen.call_args.kwargs["cwd"], "/repo")
        self.assertNotIn("HF_TOKEN", popen.call_args.kwargs["env"])

    def test_passes_provider_model_and_thinking(self) -> None:
        process = _FakeProcess(_stream({"type": "agent_settled"}))
        with _pi(process) as (popen, output):
            pi_agent.run_pi(
                "Inspect the repository.",
                cwd="/repo",
                provider="opencode-go",
                model="deepseek-v4.1-flash",
                thinking="low",
                stream=output,
            )

        self.assertEqual(
            popen.call_args.args[0][1:],
            ["--mode", "json", "--provider", "opencode-go",
             "--model", "deepseek-v4.1-flash", "--thinking", "low",
             "--", "Inspect the repository."],
        )

    def test_tool_updates_do_not_add_lines(self) -> None:
        records = [
            {"type": "tool_execution_start", "toolCallId": "c1",
             "toolName": "bash"},
            *({"type": "tool_execution_update", "toolCallId": "c1",
               "toolName": "bash"} for _ in range(50)),
            {"type": "tool_execution_end", "toolCallId": "c1",
             "toolName": "bash"},
        ]
        with _pi(_FakeProcess(_stream(*records))) as (_, output):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

        text = output.getvalue()
        # The updates stay on one carriage-return line. Only the final
        # cumulative summary ends with a newline.
        self.assertEqual(text.count("\n"), 1)
        self.assertIn("tools 1 (bash 1)", text)
        self.assertNotIn("tool update", text)

    def test_two_tools_share_the_status_line(self) -> None:
        records = [
            {"type": "tool_execution_start", "toolCallId": "c1",
             "toolName": "bash"},
            {"type": "tool_execution_start", "toolCallId": "c2",
             "toolName": "read"},
            {"type": "tool_execution_end", "toolCallId": "c1",
             "toolName": "bash"},
            {"type": "tool_execution_end", "toolCallId": "c2",
             "toolName": "read"},
        ]
        with _pi(_FakeProcess(_stream(*records))) as (_, output):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

        text = output.getvalue()
        self.assertIn("running bash, read", text)
        self.assertIn("tools 2 (bash 1, read 1)", text)

    def test_reports_tokens_cost_and_thinking(self) -> None:
        records = [
            {"type": "message_update", "assistantMessageEvent": {
                "type": "thinking_start", "contentIndex": 0}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "thinking_delta", "contentIndex": 0,
                "delta": "checking"}},
            {"type": "message_update", "assistantMessageEvent": {
                "type": "thinking_end", "contentIndex": 0}},
            {"type": "turn_end", "message": {
                "role": "assistant", "stopReason": "stop",
                "usage": {"input": 1200, "output": 340,
                          "reasoning": 120,
                          "cost": {"total": 0.0123}}}},
        ]
        with _pi(_FakeProcess(_stream(*records))) as (_, output):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

        text = output.getvalue()
        self.assertIn("turns 1", text)
        self.assertIn("think 0.0s", text)
        self.assertIn("in 1.2k / out 340", text)
        self.assertIn("reasoning 120", text)
        self.assertIn("$0.0123", text)

    def test_failed_tool_is_counted(self) -> None:
        records = [
            {"type": "tool_execution_start", "toolCallId": "c1",
             "toolName": "bash"},
            {"type": "tool_execution_end", "toolCallId": "c1",
             "toolName": "bash", "isError": True},
        ]
        with _pi(_FakeProcess(_stream(*records))) as (_, output):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

        text = output.getvalue()
        self.assertIn("failed 1", text)

    def test_raises_on_pi_error(self) -> None:
        process = _FakeProcess("", returncode=3)
        with (
            _pi(process) as (_, output),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

    def test_raises_when_the_model_call_fails(self) -> None:
        process = _FakeProcess(_stream(
            {"type": "turn_end", "message": {
                "role": "assistant", "stopReason": "error",
                "errorMessage": "503 overloaded"}},
            {"type": "agent_end", "willRetry": True},
            {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
             "delayMs": 200, "errorMessage": "503 overloaded"},
            {"type": "turn_end", "message": {
                "role": "assistant", "stopReason": "error",
                "errorMessage": "503 overloaded"}},
            {"type": "agent_end", "willRetry": False},
            {"type": "auto_retry_end", "success": False, "attempt": 1,
             "finalError": "503 overloaded"},
            {"type": "agent_settled", "aborted": False},
        ))
        with (
            _pi(process) as (_, output),
            self.assertRaises(RuntimeError) as raised,
        ):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            stream=output)

        self.assertIn("503 overloaded", str(raised.exception))
        self.assertIn("503 overloaded", output.getvalue())

    def test_a_recovered_retry_does_not_raise(self) -> None:
        process = _FakeProcess(_stream(
            {"type": "turn_end", "message": {
                "role": "assistant", "stopReason": "error",
                "errorMessage": "503 overloaded"}},
            {"type": "agent_end", "willRetry": True},
            {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
             "delayMs": 200, "errorMessage": "503 overloaded"},
            {"type": "turn_end", "message": {
                "role": "assistant", "stopReason": "stop"}},
            {"type": "agent_end", "willRetry": False},
            {"type": "auto_retry_end", "success": True, "attempt": 1},
            {"type": "agent_settled", "aborted": False},
        ))
        with _pi(process) as (_, output):
            result = pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                                     stream=output)

        self.assertEqual(result, 0)

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

    def test_rejects_provider_without_model(self) -> None:
        with self.assertRaises(ValueError):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            provider="opencode-go")

    def test_rejects_invalid_thinking(self) -> None:
        with self.assertRaises(ValueError):
            pi_agent.run_pi("Inspect the repository.", cwd="/repo",
                            model="deepseek-v4.1-flash", thinking="fast")


if __name__ == "__main__":
    unittest.main()
