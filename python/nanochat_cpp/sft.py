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
        device: str | None = None, warm_start: str | None = None,
        val_rows: str | None = None,
        eval_every: int | None = None, eval_tokens: int | None = None,
        chatcore_every: int | None = None) -> int:
    """Run the C++ SFT driver over one packed shard.

    ``shard`` is the ``NCSFT001`` file :func:`pack` writes and ``checkpoint``
    is the checkpoint container the driver saves to. The remaining keyword
    arguments are the driver flags: ``num_iterations``, ``batch``,
    ``seq_len``, ``seed``, ``device``, and the ones below.

    ``warm_start`` loads a base checkpoint's weights and optimizer moments and
    is **not** a resume (design decision D5): the rates are the fresh SFT ones
    and the step counter starts at 0, whereas ``resume`` restores this run's
    stored step and continues its schedule. The driver rejects both flags at
    once, and rejects ``resume`` with a base file.

    The run is scheduled by *progress* rather than by a step count (design
    decision D1): the driver turns the packed rows consumed so far into an
    ``approx_progress`` and feeds its running maximum to
    ``LrMultiplierForProgress``, which is what the reference SFT loop does.
    The driver also reads the recipe (the learning rates, adam epsilon, the
    warmdown ratio, and the Muon momentum schedule) from the checkpoint it
    loads (design decision D7), so a warm start carries the reference SFT
    settings rather than the pretraining ones.

    ``val_rows`` is the packed validation shard the evaluation reads and
    ``eval_tokens`` is the token budget of one pass; without ``val_rows`` the
    driver has nothing to evaluate and skips the bits-per-byte pass entirely.
    ``eval_every`` and ``chatcore_every`` set the bits-per-byte and ChatCORE
    cadence in steps, and the driver always evaluates at its final step. It
    reports the ChatCORE sentinel ``-1``, because it does not measure that
    score itself. Each of these is forwarded as its own flag and only when it
    is not ``None``, so the driver's own defaults apply otherwise.

    Returns the driver's exit status.
    """
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
    # Design decision D5: a warm start and a resume are different loads. The
    # driver rejects a run that names both, so the bridge only forwards them.
    if warm_start is not None:
        arguments.extend(["--warm-start", str(warm_start)])
    if device is not None:
        arguments.extend(["--device", device])
    if val_rows is not None:
        arguments.extend(["--val-rows", str(val_rows)])
    # The intervals and the token budget are integers; `int()` here keeps a
    # float or a numeric string from reaching the driver as a bad flag value,
    # which the driver would read as 0 (and so as "disabled").
    if eval_every is not None:
        arguments.extend(["--eval-every", str(int(eval_every))])
    if eval_tokens is not None:
        arguments.extend(["--eval-tokens", str(int(eval_tokens))])
    if chatcore_every is not None:
        arguments.extend(["--chatcore-every", str(int(chatcore_every))])
    argv = [str(binary), *arguments]
    result = toolchain.train(argv)
    return result.returncode
