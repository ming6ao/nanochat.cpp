"""The one import path every wrapper of this skill shares.

The wrappers are thin on purpose: the loop itself is
``nanochat_cpp.pgo``, which the Bazel test covers. A wrapper only finds the
repository and calls it, so a host that installed no wheel still runs the loop
from a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The repository root, from this file: ``<root>/.agents/skills/<skill>/scripts``.
ROOT = Path(__file__).resolve().parents[4]


def main(command: str, argv: list[str] | None = None) -> int:
    """Run one ``pgo`` subcommand against the tree this skill belongs to."""
    library = ROOT / "python"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from nanochat_cpp import pgo  # noqa: PLC0415 - after the path is set

    return pgo.main([command, *(sys.argv[1:] if argv is None else argv)])
