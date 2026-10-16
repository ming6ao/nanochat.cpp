"""Supervised fine-tuning bridge (docs/post-training.md sections 4 and 8).

Python renders and packs the conversations. It then hands the packed shard to
the C++ SFT driver, launched through ``tools/nanochat train``.
"""

from __future__ import annotations

import os
import sys
from array import array
from pathlib import Path
from typing import Callable

from . import _build, eval_fixture, sft_data, toolchain

__all__ = ["pack", "run"]

#: The magic of the packed SFT shard. The layout is the shared record file
#: (`src/record_file.h`), so the C++ reader owns it.
_MAGIC = b"NCSFT001"


def _pack_int32(values: array) -> bytes:
    """Little-endian bytes for one ``int32`` payload."""
    buffer = array("i", values)
    if buffer.itemsize != 4:
        raise ValueError("this host's C int is not 32 bits")
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer.tobytes()


def pack(output_path, conversations, tokenizer, seq_len: int,
         render: Callable | None = None) -> Path:
    """Write the packed rows as one ``NCSFT001`` record file.

    The file holds two ``int32`` records, ``tokens`` and ``targets``, each of
    shape ``[rows, seq_len]``. A target of ``-1`` marks an ignored position.
    The C++ mesh training driver and ``sft_main`` read the file through
    ``src/record_file.h`` (docs/training-seam.md section 6.2).
    """
    seq_len = int(seq_len)
    inputs = array("i")
    targets = array("i")
    rows = 0
    for row_inputs, row_targets in sft_data.iter_packed_rows(
            conversations, tokenizer, seq_len, render=render):
        if len(row_inputs) != seq_len or len(row_targets) != seq_len:
            continue
        inputs.extend(int(value) for value in row_inputs)
        targets.extend(int(value) for value in row_targets)
        rows += 1
    shape = (rows, seq_len)
    records = [
        eval_fixture.Record("tokens", eval_fixture.DTYPE_INT32, shape,
                            _pack_int32(inputs)),
        eval_fixture.Record("targets", eval_fixture.DTYPE_INT32, shape,
                            _pack_int32(targets)),
    ]
    return eval_fixture.write(output_path, records, magic=_MAGIC)


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
