"""The child-process launcher for the C++ mesh training driver.

The C++ driver owns the step loop, the metric record, the summary, the
checkpoint interval, and the resume (docs/client-runtime-design.md section
10.1). The client is :class:`nanochat_cpp.runtime.MeshTrainConfig`. This module
holds only the Kaggle process seam.

:func:`launch` runs one config in a child process through
``tools/nanochat client``. The notebook cell returns, so the cell status stays
correct, and the parent polls ``metrics.jsonl`` and ``worker.log`` instead. See
``docs/client-runtime-design.md``.

The module imports ``nanochat_cpp`` inside the functions, not at module load.
The pure helpers then import without the shared library.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["MeshLiveRun", "launch", "read_metrics", "read_summary"]


def read_metrics(path) -> list[dict]:
    """Read the per-step JSON lines. A missing file returns an empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def read_summary(out_dir) -> dict:
    """Read the run summary. A missing or bad file returns an empty dict."""
    path = Path(out_dir) / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


class MeshLiveRun:
    """A mesh driver that runs in a child process.

    The parent starts the child with :func:`launch` and polls it. The child
    writes ``metrics.jsonl`` and ``worker.log`` in the run directory, so the
    parent never reads a pipe and the notebook cell never blocks.
    """

    def __init__(self, config, process: subprocess.Popen, *,
                 log_path, timeout: float | None = None) -> None:
        self.config = config
        self.process = process
        self.log_path = Path(log_path)
        self._timeout = timeout

    def running(self) -> bool:
        """True while the child process is alive."""
        return self.process.poll() is None

    def tail(self, lines: int = 15) -> str:
        """The last ``lines`` lines of the child log. Missing returns ""."""
        if not self.log_path.is_file():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def rows(self) -> list[dict]:
        """The per-step metrics written so far."""
        return read_metrics(self.config.metrics_path())

    def summary(self) -> dict:
        """The final summary, or an empty dict before the child finishes."""
        return read_summary(self.config.out_dir)

    def stream(self, interval: float = 5.0) -> None:
        """Print the child log as it grows, until the child exits.

        A Control-C stops the worker and returns. The C++ driver owns the
        interrupt path, so the worker saves the checkpoint before it exits.
        """
        position = 0
        try:
            while self.running():
                text = (self.log_path.read_text(encoding="utf-8",
                                                errors="replace")
                        if self.log_path.is_file() else "")
                if len(text) > position:
                    print(text[position:], end="", flush=True)
                    position = len(text)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\ninterrupt received; stopping the worker", flush=True)
            self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        """Block until the child exits and return its return code."""
        timeout = self._timeout if timeout is None else timeout
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.terminate()
            raise

    def terminate(self, timeout: float = 30.0) -> None:
        """Stop the child group. Send SIGINT first, so the worker can save.

        Further SIGINT delivery is ignored until the stop finishes, so a second
        cancel click does not break the cleanup.
        """
        if self.process.poll() is not None:
            return
        previous = None
        try:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            previous = None
        try:
            self._stop(timeout)
        finally:
            if previous is not None:
                try:
                    signal.signal(signal.SIGINT, previous)
                except (ValueError, OSError):
                    pass

    def _stop(self, timeout: float) -> None:
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            self.process.terminate()
            return
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.process.kill()


def launch(config, root: str | Path = ".", *,
           entry: str = "tools/nanochat", profile: str = "train",
           device: str | None = None, timeout: float | None = None,
           extra_env: dict | None = None) -> MeshLiveRun:
    """Start one config in a child process and return a pollable handle.

    ``config`` is a :class:`nanochat_cpp.runtime.MeshTrainConfig`. The child
    runs through ``tools/nanochat client``, so one broker lock and one sandbox
    scope cover the whole mesh. The child stdout and stderr go to
    ``worker.log``, so the parent never blocks on a full pipe.
    """
    root = Path(root)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    job_path.write_text(config.job_json() + "\n", encoding="utf-8")

    worker = Path(__file__).with_name("mesh_worker.py")
    env = dict(os.environ)
    env.update(extra_env or {})
    paths = [str(root / "python"), str(root / "notebooks")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    command = [str(root / entry), "client", "--profile", profile]
    if device:
        command += ["--device", str(device)]
    command += ["--", sys.executable, str(worker), str(job_path)]

    log_path = out_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(root), env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return MeshLiveRun(config, process, log_path=log_path, timeout=timeout)
