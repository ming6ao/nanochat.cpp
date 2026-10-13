"""Run a client script as the single controller.

``python -m nanochat_cpp.run train.py`` re-enters ``tools/nanochat client``.
That command takes one broker lock for the full device set, applies one sandbox
scope, and starts the script as one process. The script then sees all local
accelerators as one mesh. See ``docs/client-runtime-design.md`` section 14.
"""

from __future__ import annotations

import sys
from typing import Sequence

from . import toolchain

__all__ = ["main"]

_USAGE = ("usage: python -m nanochat_cpp.run [--device LIST] "
          "<script.py> [args...]")


def main(argv: Sequence[str] | None = None,
         tool: toolchain.Toolchain | None = None) -> int:
    """Re-enter ``tools/nanochat client`` with the given script.

    ``argv`` defaults to ``sys.argv[1:]``. ``tool`` injects a toolchain; the
    test uses it to record the call. The return value is the script exit code,
    or 2 for a usage error.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    device = None
    if arguments[:1] == ["--device"]:
        if len(arguments) < 3:
            print(_USAGE, file=sys.stderr)
            return 2
        device, arguments = arguments[1], arguments[2:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print(_USAGE, file=sys.stdout if arguments else sys.stderr)
        return 0 if arguments else 2
    if tool is None:
        tool = toolchain.Toolchain(check=False)
    result = tool.client([sys.executable, *arguments], device=device)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
