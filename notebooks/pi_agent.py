"""Run the Pi coding agent and report its JSON event stream."""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TextIO

__all__ = ["run_pi"]


def _read_lines(name: str, stream: TextIO, events: queue.Queue) -> None:
    for line in stream:
        events.put((name, line))
    events.put((name, None))


def _report_event(event: dict) -> None:
    """Print one event of the Pi JSON stream."""
    event_type = event.get("type")

    if event_type == "message_update":
        update = event.get("assistantMessageEvent", {})
        if update.get("type") == "text_delta":
            print(update.get("delta", ""), end="", flush=True)
    elif event_type == "tool_execution_start":
        print(f"\n[pi] starting tool: {event.get('toolName')}", flush=True)
    elif event_type == "tool_execution_update":
        print(f"\n[pi] tool update: {event.get('toolName')}", flush=True)
    elif event_type == "tool_execution_end":
        print(f"\n[pi] finished tool: {event.get('toolName')}", flush=True)
    elif event_type == "turn_end":
        error = event.get("message", {}).get("errorMessage")
        print(f"\n[pi] turn failed: {error}" if error else "\n[pi] turn_end",
              flush=True)
    elif event_type == "agent_end":
        print(f"\n[pi] agent_end willRetry={event.get('willRetry')}", flush=True)
    elif event_type == "auto_retry_start":
        print(f"\n[pi] retry {event.get('attempt')}/{event.get('maxAttempts')}",
              flush=True)
    elif event_type == "agent_settled":
        print(f"\n[pi] {event_type}", flush=True)


def run_pi(
    prompt: str,
    *,
    cwd: str | Path,
    heartbeat_seconds: float = 15.0,
) -> int:
    """Run Pi in JSON mode and print text, tool events, and idle heartbeats.

    The heartbeat reports that the process is still running. It does not
    confirm that the model or a tool is making progress.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be greater than zero")

    pi_path = shutil.which("pi")
    if pi_path is None:
        raise FileNotFoundError("Pi CLI is not installed or is not on PATH")

    command = [pi_path, "--mode", "json", "--", prompt]
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
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        threading.Thread(
            target=_read_lines,
            args=(name, stream, events),
            daemon=True,
        ).start()

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
                    print(
                        f"\n[pi] process is running; no output event for "
                        f"{quiet:.0f}s ({elapsed:.0f}s elapsed)",
                        flush=True,
                    )
                else:
                    print(
                        f"\n[pi] process exited with code {return_code}; "
                        "waiting for output streams to close",
                        flush=True,
                    )
                continue

            if line is None:
                closed_streams += 1
                continue

            last_event = time.monotonic()
            if name == "stderr":
                print(f"\n[pi stderr] {line.rstrip()}", flush=True)
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"\n[pi raw] {line.rstrip()}", flush=True)
                continue
            _report_event(event)
            if event.get("type") == "turn_end":
                last_error = event.get("message", {}).get("errorMessage")

        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    # Pi exits with code zero after a failed model call. Check the stream too.
    if last_error is not None:
        raise RuntimeError(f"Pi run failed: {last_error}")
    return return_code
