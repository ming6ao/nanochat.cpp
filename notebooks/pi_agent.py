"""Run the Pi coding agent and report its JSON event stream.

Assistant text streams to the output as it arrives. Agent activity updates one
status line with cumulative metrics, so a notebook does not grow one line for
each event. The final line reports the turn count, the per-tool breakdown, the
thinking time, the tool time, the tokens, and the cost.

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

#: The minimum width of the status line. It overwrites the previous status.
_STATUS_WIDTH = 100


def _read_lines(name: str, stream: TextIO, events: queue.Queue) -> None:
    for line in stream:
        events.put((name, line))
    events.put((name, None))


def _tokens(value: int) -> str:
    """Format a token count as ``1.2k`` or ``3.4M``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


class _Status:
    """Stream assistant text and one cumulative metrics line."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started = time.monotonic()
        self.running: dict[str, str] = {}
        self.turns = 0
        self.calls = 0
        self.failed = 0
        self.tool_ms = 0.0
        self.tool_counts: dict[str, int] = {}
        self.think_ms = 0.0
        self.think_started: float | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost = 0.0
        self.shown = False
        self.text_open = False
        self.last_line = ""
        self.last_len = 0

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _erase(self) -> None:
        if self.shown:
            width = max(_STATUS_WIDTH, self.last_len)
            self._write("\r" + " " * width + "\r")
            self.shown = False
            self.last_line = ""
            self.last_len = 0

    def _draw(self, line: str) -> None:
        if self.shown and line == self.last_line:
            return
        width = max(_STATUS_WIDTH, self.last_len)
        self._write("\r" + line.ljust(width))
        self.shown = True
        self.last_line = line
        self.last_len = len(line)

    def _close_text(self) -> None:
        if self.text_open:
            self._write("\n")
            self.text_open = False

    def close_thinking(self) -> None:
        if self.think_started is not None:
            self.think_ms += (time.monotonic() - self.think_started) * 1000.0
            self.think_started = None

    def thinking_seconds(self) -> float:
        elapsed = self.think_ms
        if self.think_started is not None:
            elapsed += (time.monotonic() - self.think_started) * 1000.0
        return elapsed / 1000.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def text(self, delta: str) -> None:
        self.close_thinking()
        self._erase()
        self._write(delta)
        self.text_open = True

    def thinking(self) -> None:
        if self.think_started is None:
            self.think_started = time.monotonic()
        self.live()

    def thinking_end(self) -> None:
        self.close_thinking()
        self.live()

    def _breakdown(self) -> str:
        if not self.tool_counts:
            return str(self.calls)
        items = sorted(self.tool_counts.items(),
                       key=lambda item: (-item[1], item[0]))
        shown = ", ".join(f"{name} {count}" for name, count in items[:3])
        if len(items) > 3:
            shown += f", +{len(items) - 3}"
        return f"{self.calls} ({shown})"

    def _counters(self, live: bool) -> str:
        thought = self.thinking_seconds()
        think = f"{int(thought)}s" if live else f"{thought:.1f}s"
        parts = [f"turns {self.turns}", f"tools {self._breakdown()}"]
        if self.failed:
            parts.append(f"failed {self.failed}")
        parts.append(f"think {think}")
        return " \u00b7 ".join(parts)

    def live(self) -> None:
        self._close_text()
        active = (self.turns or self.calls or self.running or self.think_ms
                  or self.think_started is not None)
        if not active:
            self._erase()
            return
        line = f"[pi] {self._counters(True)} \u00b7 {self.elapsed():.0f}s"
        if self.running:
            names = list(self.running.values())
            shown = ", ".join(names[:4])
            if len(names) > 4:
                shown += f", +{len(names) - 4}"
            line += f" \u00b7 running {shown}"
        self._draw(line)

    def totals(self) -> str:
        parts = [f"[pi] {self._counters(False)}"]
        if self.tool_ms:
            parts.append(f"tool {self.tool_ms / 1000.0:.1f}s")
        if self.input_tokens or self.output_tokens:
            parts.append(f"in {_tokens(self.input_tokens)} / "
                         f"out {_tokens(self.output_tokens)}")
        if self.reasoning_tokens:
            parts.append(f"reasoning {_tokens(self.reasoning_tokens)}")
        if self.cost:
            parts.append(f"${self.cost:.4f}")
        parts.append(f"{self.elapsed():.0f}s")
        return " \u00b7 ".join(parts)

    def line(self, text: str) -> None:
        self.close_thinking()
        self._erase()
        self._close_text()
        self._write(text + "\n")

    def tool_start(self, event: dict) -> None:
        self.close_thinking()
        self._close_text()
        self.running[str(event.get("toolCallId", ""))] = str(
            event.get("toolName", "?"))
        self.live()

    def tool_end(self, event: dict) -> None:
        self.close_thinking()
        name = str(event.get("toolName", "?"))
        self.running.pop(str(event.get("toolCallId", "")), None)
        self.calls += 1
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        if event.get("isError"):
            self.failed += 1
        duration = event.get("durationMs")
        if isinstance(duration, (int, float)):
            self.tool_ms += float(duration)
        self.live()

    def _add_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.input_tokens += int(usage.get("input") or 0)
        self.output_tokens += int(usage.get("output") or 0)
        self.reasoning_tokens += int(usage.get("reasoning") or 0)
        cost = usage.get("cost") or {}
        self.cost += float(cost.get("total") or 0.0)

    def turn_end(self, event: dict) -> None:
        self.close_thinking()
        self.turns += 1
        message = event.get("message") or {}
        self._add_usage(message.get("usage"))
        for result in event.get("toolResults") or []:
            self._add_usage(result.get("usage"))
        error = message.get("errorMessage")
        if error:
            self.line(f"[pi] turn failed: {error}")

    def finish(self) -> None:
        self.close_thinking()
        if self.turns or self.calls:
            self.line(self.totals())
        else:
            self._erase()
            self._close_text()


def _report_event(event: dict, status: _Status) -> None:
    """Render one event of the Pi JSON stream."""
    kind = event.get("type")
    if kind == "message_update":
        update = event.get("assistantMessageEvent", {})
        subtype = update.get("type")
        if subtype == "text_delta":
            status.text(update.get("delta", ""))
        elif subtype in ("thinking_start", "thinking_delta"):
            status.thinking()
        elif subtype == "thinking_end":
            status.thinking_end()
    elif kind == "tool_execution_start":
        status.tool_start(event)
    elif kind == "tool_execution_end":
        status.tool_end(event)
    elif kind == "turn_end":
        status.turn_end(event)
    elif kind == "auto_retry_start":
        status.line(
            f"[pi] retry {event.get('attempt')}/{event.get('maxAttempts')}")


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
    """Run Pi in JSON mode with a live metrics line.

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
