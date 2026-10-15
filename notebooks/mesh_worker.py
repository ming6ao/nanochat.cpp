"""Run one mesh training job from a JSON job file in a child process.

:func:`mesh_train.launch` starts this module through ``tools/nanochat
client``, so one broker lock and one sandbox scope cover the whole mesh. The
parent notebook cell returns, and the parent polls ``metrics.jsonl`` and
``worker.log``. See ``docs/client-runtime-design.md``.

The C++ driver owns the loop and the interrupt path, so it saves the checkpoint
before it returns. This module only reports the outcome.

Usage: ``python mesh_worker.py <job.json>``. The exit code is 0 on success and
130 after an interrupt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["main"]


def main(argv=None) -> int:
    """Run the job file in ``argv[0]``. Return the process exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: mesh_worker.py <job.json>", file=sys.stderr)
        return 2

    from nanochat_cpp import runtime as rt

    job_path = Path(arguments[0])
    text = job_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError as error:
        print(f"mesh_worker: bad job file: {error}", file=sys.stderr)
        return 2
    config = rt.MeshTrainConfig.from_dict(data)
    try:
        result = config.run()
    except KeyboardInterrupt:
        print("mesh_worker: interrupted; the checkpoint is saved", flush=True)
        return 130
    finally:
        config.close()
    if result.interrupted:
        print("mesh_worker: interrupted; the checkpoint is saved", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
