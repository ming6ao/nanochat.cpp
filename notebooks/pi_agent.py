"""Run the Pi coding agent and report its JSON event stream.

Assistant text streams to the output as it arrives. Tool activity updates one
status line with cumulative counts, so a notebook does not grow one line for
each event. The status line uses a carriage return, the same convention as
``lab.ProgressReporter``.

Pass ``provider`` and ``model`` explicitly. A settings file is not applied
when its model ID does not match the catalog, and the automatic fallback can
select a provider that the credential cannot use.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

__all__ = ["run_pi"]

#: The thinking levels accepted by ``pi --thinking``.
_THINKING_LEVELS = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max"})

#: The width of the tool status line. It overwrites the previous status.
_STATUS_WIDTH = 100


def _read_lines(name: str, stream: TextIO, events: queue.Queue) -> None:
    for line in stream:
        events.put((name, line))
    events.put((name, None))


class _Status:
    """Stream assistant text and one cumulative tool-status line."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.running: dict[str, str] = {}
        self.calls = 0
        self.failed = 0
        self.total_ms = 0.0
        self.shown = False
        self.text_open = False

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _erase(self) -> None:
        if self.shown:
            self._write("\r" + " " * _STATUS_WIDTH + "\r")
            self.shown = False

    def _close_text(self) -> None:
        if self.text_open:
            self._write("\n")
            self.text_open = False

    def text(self, delta: str) -> None:
        self._erase()
        self._write(delta)
        self.text_open = True

    def totals(self) -> str:
        parts = [f"[pi] tools {self.calls}"]
        if self.failed:
            parts.append(f"failed {self.failed}")
        parts.append(f"{self.total_ms / 1000.0:.1f}s")
        return " \u00b7 ".join(parts)

    def live(self) -> None:
        self._erase()
        self._close_text()
        if not self.calls and not self.running:
            return
        line = self.totals()
        if self.running:
            names = list(self.running.values())
            shown = ", ".join(names[:4])
            if len(names) > 4:
                shown += f", +{len(names) - 4}"
            line += f" \u00b7 running {shown}"
        self._write("\r" + line.ljust(_STATUS_WIDTH))
        self.shown = True

    def line(self, text: str) -> None:
        self._erase()
        self._close_text()
        self._write(text + "\n")

    def tool_start(self, event: dict) -> None:
        self.running[str(event.get("toolCallId", ""))] = str(
            event.get("toolName", "?"))
        self.live()

    def tool_end(self, event: dict) -> None:
        self.running.pop(str(event.get("toolCallId", "")), None)
        self.calls += 1
        if event.get("isError"):
            self.failed += 1
        duration = event.get("durationMs")
        if isinstance(duration, (int, float)):
            self.total_ms += float(duration)
        self.live()

    def finish(self) -> None:
        if self.calls:
            self.line(self.totals())
        else:
            self._erase()
            self._close_text()


def _report_event(event: dict, status: _Status) -> None:
    """Render one event of the Pi JSON stream."""
    kind = event.get("type")
    if kind == "message_update":
        update = event.get("assistantMessageEvent", {})
        if update.get("type") == "text_delta":
            status.text(update.get("delta", ""))
    elif kind == "tool_execution_start":
        status.tool_start(event)
    elif kind == "tool_execution_end":
        status.tool_end(event)
    elif kind == "turn_end":
        error = event.get("message", {}).get("errorMessage")
        if error:
            status.line(f"[pi] turn failed: {error}")
    elif kind == "agent_end":
        status.line(f"[pi] agent_end willRetry={event.get('willRetry')}")
    elif kind == "auto_retry_start":
        status.line(
            f"[pi] retry {event.get('attempt')}/{event.get('maxAttempts')}")
    elif kind == "agent_settled":
        status.line("[pi] agent_settled")


def run_pi(
    prompt: str,
    *,
    cwd: str | Path,
    provider: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    heartbeat_seconds: float = 15.0,
    stream: TextIO | None = None,
) -> int:
    """Run Pi in JSON mode with a live tool-status line.

    The heartbeat reports that the process is still running. It does not
    confirm that the model or a tool is making progress.

    ``provider`` requires ``model``. When both are omitted, Pi selects the
    model, and its automatic choice can fail with "Model access is disabled".
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be greater than zero")
    if provider and not model:
        raise ValueError("provider requires model")
    if thinking is not None and thinking not in _THINKING_LEVELS:
        raise ValueError(f"invalid thinking level: {thinking!r}")

    pi_path = shutil.which("pi")
    if pi_path is None:
        raise FileNotFoundError("Pi CLI is not installed or is not on PATH")

    command = [pi_path, "--mode", "json"]
    if provider:
        command += ["--provider", provider]
    if model:
        command += ["--model", model]
    if thinking:
        command += ["--thinking", thinking]
    command += ["--", prompt]

    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("Pi did not provide its output streams")

    events: queue.Queue = queue.Queue()
    streams = (("stdout", process.stdout), ("stderr", process.stderr))
    for name, source in streams:
        threading.Thread(
            target=_read_lines,
            args=(name, source, events),
            daemon=True,
        ).start()

    status = _Status(stream if stream is not None else sys.stdout)
    started = last_event = time.monotonic()
    closed_streams = 0
    last_error: str | None = None

    try:
        while closed_streams < 2:
            try:
                name, line = events.get(timeout=heartbeat_seconds)
            except queue.Empty:
                quiet = time.monotonic() - last_event
                elapsed = time.monotonic() - started
                return_code = process.poll()
                if return_code is None:
                    status.line(
                        f"[pi] process is running; no output event for "
                        f"{quiet:.0f}s ({elapsed:.0f}s elapsed)")
                else:
                    status.line(
                        f"[pi] process exited with code {return_code}; "
                        "waiting for output streams to close")
                continue

            if line is None:
                closed_streams += 1
                continue

            last_event = time.monotonic()
            if name == "stderr":
                status.line(f"[pi stderr] {line.rstrip()}")
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                status.line(f"[pi raw] {line.rstrip()}")
                continue
            _report_event(event, status)
            if event.get("type") == "turn_end":
                last_error = event.get("message", {}).get("errorMessage")

        return_code = process.wait()
    except BaseException:
        status.finish()
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise

    status.finish()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    # Pi exits with code zero after a failed model call. Check the stream too.
    if last_error is not None:
        raise RuntimeError(f"Pi run failed: {last_error}")
    return return_code
