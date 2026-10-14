"""Supervised fine-tuning bridge (docs/post-training.md sections 4 and 8).

Python renders and packs the conversations. It then hands the packed shard to
the C++ SFT driver, launched through ``tools/nanochat train``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable

from . import _build, toolchain
from . import sft_data

__all__ = ["pack", "run"]

_MAGIC = b"NCSFT001"


def pack(output_path, conversations, tokenizer, seq_len: int,
         render: Callable | None = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as file:
        file.write(_MAGIC)
        file.write(struct.pack("<QQ", seq_len, 0))
        for inputs, targets in sft_data.iter_packed_rows(
                conversations, tokenizer, seq_len, render=render):
            if len(inputs) != seq_len or len(targets) != seq_len:
                continue
            for value in inputs:
                file.write(struct.pack("<i", int(value)))
            for value in targets:
                file.write(struct.pack("<i", int(value)))
            count += 1
        size = file.tell()
        file.seek(16)
        file.write(struct.pack("<Q", count))
        file.seek(size)
        file.flush()
    return path


def run(shard, checkpoint, *, binary: str | None = None,
        num_iterations: int = 100, batch: int = 2, seq_len: int | None = None,
        seed: int = 42, resume: str | None = None,
        device: str | None = None) -> int:
    if binary is None:
        env_binary = os.environ.get("NANOCHAT_SFT_MAIN")
        if env_binary:
            binary = env_binary
        else:
            binary = str(_build.repo_root() / "bazel-bin" / "src" / "sft_main")
    arguments = [
        "--rows", str(shard),
        "--checkpoint", str(checkpoint),
        "--num-iterations", str(num_iterations),
        "--batch", str(batch),
        "--seed", str(seed),
    ]
    if seq_len is not None:
        arguments.extend(["--seq", str(seq_len)])
    if resume is not None:
        arguments.extend(["--resume", str(resume)])
    if device is not None:
        arguments.extend(["--device", device])
    argv = [str(binary), *arguments]
    result = toolchain.train(argv)
    return result.returncode