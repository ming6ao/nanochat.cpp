"""Run one mesh training plan from a JSON job file in a child process.

:func:`mesh_train.launch` starts this module through ``tools/nanochat
client``, so one broker lock and one sandbox scope cover the whole mesh. The
parent notebook cell returns, and the parent polls ``metrics.jsonl`` and
``worker.log``. See ``docs/notebook-workflow.md``.

Usage: ``python mesh_worker.py <job.json>``. The exit code is 0 on success and
130 after an interrupt. On an interrupt, :func:`mesh_train.run` saves the
checkpoint before the error propagates.
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

    import mesh_train

    job_path = Path(arguments[0])
    plan = mesh_train.MeshTrainPlan.from_job(
        json.loads(job_path.read_text(encoding="utf-8")))
    try:
        mesh_train.run(plan, progress=mesh_train.log_progress)
    except KeyboardInterrupt:
        print("mesh_worker: interrupted; the checkpoint is saved",
              flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
