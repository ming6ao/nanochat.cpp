"""Locate, build, and launch the C++ training binary under the sandbox.

The C++ entry points refuse to start without ``NANOCHAT_SANDBOX`` (see
``include/nanochat/sandbox.h``), so every launch goes through ``tools/nanochat``.
When the bridge itself is already running inside the sandbox (the user invoked
``tools/nanochat gpu -- python -m nanochat_cpp.base_train``), the binary is
exec'd directly so the existing cgroup covers the whole process tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def nanochat_entry(repo_root: Path) -> Path:
    return repo_root / "tools" / "nanochat"


def binary_path(repo_root: Path, name: str = "train_main") -> Path:
    return repo_root / "bazel-bin" / "src" / name


def ensure_built(repo_root: Path, cuda: bool, name: str = "train_main") -> Path:
    binary = binary_path(repo_root, name)
    if binary.is_file():
        return binary
    command = [str(nanochat_entry(repo_root)), "build"]
    if cuda:
        command.append("--config=cuda")
    print(f"[nanochat_cpp] building {name} ({'cuda' if cuda else 'cpu'})...",
          file=sys.stderr)
    subprocess.run(command, check=True)
    if not binary.is_file():
        raise SystemExit(f"build finished but {binary} is missing")
    return binary


def gpu_name() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL)
        return output.decode().strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - MFU is best-effort
        return ""


def launch(repo_root: Path, binary: Path, arguments: list[str], cuda: bool,
           profile: str) -> int:
    """Run the binary, acquiring the sandbox and (for CUDA) the GPU broker."""
    if os.environ.get("NANOCHAT_SANDBOX"):
        # Already inside a sandbox (and possibly the broker); exec in place.
        os.execv(str(binary), [str(binary)] + arguments)

    if cuda:
        command = [str(nanochat_entry(repo_root)), "gpu", "--profile", profile,
                   "--", str(binary)]
    else:
        command = [str(nanochat_entry(repo_root)), "run", "train", "--",
                   str(binary)]
    return subprocess.call(command + arguments)
