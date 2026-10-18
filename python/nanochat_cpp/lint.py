#!/usr/bin/env python3
"""The nanochat.cpp style gate: the Google C++ Style Guide checks.

It runs clang-format in check mode over every tracked C++ source, then the text
checks that clang-format cannot make:

- no using-directive,
- every project include is project-relative,
- no C-style cast,
- no non-ASCII character.

    tools/nanochat lint           # every tracked C++ source
    tools/nanochat lint a.cc b.h  # the named files

`python/nanochat_cpp/cli.py` calls :func:`main` in process. See
CONTRIBUTING.md, .clang-format, and docs/testing.md.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["main"]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The include prefixes that resolve from the repository root.
INCLUDE_PREFIXES = ("nanochat/", "backends/", "src/", "include/", "tools/",
                    "tests/", "dev/")

#: The tracked source patterns, in the order ``git ls-files`` receives them.
SOURCE_PATTERNS = ("*.h", "*.cc", "*.cu", "*.cuh")

USING_DIRECTIVE = re.compile(r"using namespace")
BARE_INCLUDE = re.compile(r'^#include "')
C_STYLE_CAST = re.compile(
    r"\((int|float|double|char|unsigned|long|size_t|std::[a-z_0-9]+)\)"
    r"[A-Za-z_0-9(]")

#: The checks, in report order. Each entry pairs the message with its key.
CHECKS = (
    ("using-directive; Google bans 'using namespace'", "using"),
    ("bare project include; use a path from the repository root", "include"),
    ("C-style cast; use static_cast or a brace initialization", "cast"),
    ("non-ASCII character", "ascii"),
)


def tracked_sources() -> list[str]:
    """The tracked C++ sources, or an empty list when git fails."""
    result = subprocess.run(["git", "ls-files", *SOURCE_PATTERNS],
                            cwd=REPO_ROOT, capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def read_source(path: str) -> str | None:
    """The file text, or ``None`` when the read fails."""
    try:
        with open(REPO_ROOT / path, encoding="utf-8",
                  errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def scan(path: str) -> dict[str, list[str]]:
    """The violations of one file, keyed by check name.

    Each hit is ``path:line:text``, the format that ``grep -nH`` prints.
    """
    text = read_source(path)
    if text is None:
        return {}
    hits: dict[str, list[str]] = {key: [] for _, key in CHECKS}
    for number, line in enumerate(text.splitlines(), start=1):
        where = f"{path}:{number}:{line}"
        if USING_DIRECTIVE.search(line):
            hits["using"].append(where)
        if BARE_INCLUDE.match(line):
            quoted = line.split('"', 2)[1]
            if not quoted.startswith(INCLUDE_PREFIXES):
                hits["include"].append(where)
        if C_STYLE_CAST.search(line):
            hits["cast"].append(where)
        if any(ord(character) > 127 for character in line):
            hits["ascii"].append(where)
    return hits


def report(message: str, hits: list[str]) -> bool:
    """Print one check's hits. True when the check failed."""
    if not hits:
        return False
    print(f"lint: {message}", file=sys.stderr)
    print("\n".join(hits), file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    """Check the named files, or every tracked C++ source."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    files = arguments or tracked_sources()
    if not files:
        print("lint: no C++ sources to check", file=sys.stderr)
        return 1
    formatter = shutil.which("clang-format")
    if not formatter:
        print("lint: clang-format is not on PATH", file=sys.stderr)
        return 1
    failed = False
    result = subprocess.run([formatter, "--dry-run", "--Werror", *files],
                            cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print("lint: clang-format reported differences", file=sys.stderr)
        failed = True
    scans = {path: scan(path) for path in files}
    for message, key in CHECKS:
        hits = [hit for path in files for hit in scans[path].get(key, [])]
        if report(message, hits):
            failed = True
    if failed:
        print("lint: FAILED", file=sys.stderr)
        return 1
    print(f"lint: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
