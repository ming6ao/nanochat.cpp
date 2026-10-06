#!/usr/bin/env python3
"""Convert a reference nanochat ``.pt`` checkpoint into the NCHKPT01 container.

The C++ runtime reads only its own self-describing container and never imports
PyTorch, so a released base or SFT checkpoint has to be remapped once on the
host. The logic lives in ``python/nanochat_cpp/checkpoint.py``; this script only
puts ``python/`` on the path and forwards to its command line, so the same code
is importable (and testable without torch). See ``docs/python.md``.

A released checkpoint keeps the parameters under the same names the C++ model
registers (``src/model.cc``), so the remap is mostly normalising the
``_orig_mod.``/``module.`` wrappers; the dtype policy casts every tensor to the
container build's storage dtype (fp32, or fp16 for the sm_75 build).

    # Explicit file.
    python3 tools/convert_checkpoint.py \
        --input ~/.cache/nanochat/base_checkpoints/d20/model_001000.pt \
        --dtype fp32 --verify

    # Resolve source/tag/step like the reference checkpoint manager.
    python3 tools/convert_checkpoint.py \
        --source sft --model-tag d20 --step 499 --output model.nchkpt01

The script needs the reference virtual environment (torch + numpy); run it with
``python3`` from that environment. ``--print-path`` works without torch::

    python3 tools/convert_checkpoint.py --source base --model-tag d20 --print-path
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives at <repo>/tools/convert_checkpoint.py; the package lives under
# <repo>/python. Put it on sys.path before importing, so the script works when
# invoked from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from nanochat_cpp import checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return checkpoint.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
