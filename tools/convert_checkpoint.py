#!/usr/bin/env python3
"""Convert a reference nanochat ``.pt`` checkpoint into the checkpoint container.

The C++ runtime reads only its own self-describing container and never imports
PyTorch, so a released base or SFT checkpoint has to be remapped once on the
host. The container writer stays in ``python/nanochat_cpp/checkpoint.py`` and
the PyTorch half lives in ``nanochat_cpp_bridge.torch_checkpoint`` (the adapter
package of ``docs/dependency-layering-plan.md`` section 4); this script only
puts ``python/`` on the path and forwards to that module's command line, so the
conversion is importable (and testable without torch) and shares one entry
point with the ``nanochat-cpp-convert`` console script. See ``docs/python.md``.

A released checkpoint keeps the parameters under the same names the C++ model
registers (``src/model.cc``), so the remap is mostly normalising the
``_orig_mod.``/``module.`` wrappers; the dtype policy casts every tensor to the
requested storage dtype (fp32 by default, bf16 for a compact artifact).

The container bytes come from the C library, so the conversion runs inside the
sandbox (``docs/sandbox.md``).

    # Explicit file.
    tools/nanochat run t0-cpu -- python3 tools/convert_checkpoint.py \
        --input ~/.cache/nanochat/base_checkpoints/d20/model_001000.pt \
        --dtype fp32 --verify

    # A compact bf16 artifact.
    tools/nanochat run t0-cpu -- python3 tools/convert_checkpoint.py \
        --input ~/.cache/nanochat/base_checkpoints/d20/model_001000.pt \
        --dtype bf16 --output model-bf16.nchkpt01

    # Resolve source/tag/step like the reference checkpoint manager.
    tools/nanochat run t0-cpu -- python3 tools/convert_checkpoint.py \
        --source sft --model-tag d20 --step 499 --output model.nchkpt01

The conversion needs the reference virtual environment (torch + numpy), which
``pip install 'nanochat-cpp[reference]'`` supplies; run it with ``python3``
from that environment. Set ``NANOCHAT_ALLOW_UNSANDBOXED=1`` for a pure
host-side run that touches no device. ``--print-path`` and ``--help`` work
without torch and without the library, because the module imports torch inside
the functions that need it::

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

from nanochat_cpp_bridge import torch_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Convert one checkpoint; the adapter module owns the command line."""
    return torch_checkpoint.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
