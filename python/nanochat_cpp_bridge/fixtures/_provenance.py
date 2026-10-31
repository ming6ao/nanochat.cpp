"""The environment record a parity producer stamps into the manifest (7.2).

A parity claim that names no environment rests on an environment a reader
cannot rebuild. The record of ``docs/dependency-layering-plan.md`` section 7.2
names four facts::

    "producer": {
      "torch": "2.5.1",
      "numpy": "1.26.4",
      "nanochat_commit": "<sha>",
      "python": "3.12.4"
    },
    "generated_by": "nanochat_cpp_bridge.fixtures.train_parity"

Truth over completeness: a fact is recorded only when the run establishes it.
A host without torch, or a reference checkout that is not a git working tree,
leaves the key out instead of guessing a version, and the caller stamps a gap
it can name rather than a version that never existed.

:func:`emit` prints one record per line in a stable, parseable form, so a Model
owner reads the run instead of retyping it::

    producer {"generated_by": "...", "producer": {...}}
    fixture {"path": "...", "sha256": "...", "size_bytes": 161760}

Extras: none at import; ``torch`` and ``numpy`` are read only to report their
version, and an absent package is left out of the record.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

__all__ = [
    "environment",
    "emit",
    "fixture_record",
    "nanochat_commit",
    "package_version",
    "producer_record",
    "sha256",
]

#: The packages whose version the record carries, in report order.
VERSIONED_PACKAGES = ("torch", "numpy")


def package_version(package: str) -> str | None:
    """The installed version of ``package``, or ``None`` when it is absent.

    A probe keeps the record honest and cheap: a package that is not installed
    contributes no key, and one that is installed contributes its own version.
    """
    try:
        if importlib.util.find_spec(package) is None:
            return None
        module = importlib.import_module(package)
    except (ImportError, ValueError):
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def nanochat_commit() -> str | None:
    """The reference checkout's ``HEAD`` commit, or ``None``.

    The commit comes from the checkout the producer imported, so the record
    names the reference code that produced the bytes. A checkout that is not a
    git working tree establishes nothing, and the key is left out.
    """
    try:
        from nanochat_cpp.reference import find_reference_repo  # noqa: PLC0415

        repo = find_reference_repo()
    except (ImportError, SystemExit):
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def environment() -> dict[str, str]:
    """The environment facts this run establishes, in the section 7.2 shape."""
    facts: dict[str, str] = {}
    for package in VERSIONED_PACKAGES:
        version = package_version(package)
        if version:
            facts[package] = version
    commit = nanochat_commit()
    if commit:
        facts["nanochat_commit"] = commit
    facts["python"] = platform.python_version()
    return facts


def producer_record(generated_by: str) -> dict[str, Any]:
    """The ``producer`` and ``generated_by`` pair a manifest record carries."""
    return {"generated_by": str(generated_by), "producer": environment()}


def sha256(path: str | Path) -> str:
    """The SHA-256 of one file, in chunks so a large fixture stays cheap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_record(path: str | Path) -> dict[str, Any]:
    """The size and hash of one produced fixture, as a manifest records them."""
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def emit(tag: str, record: Mapping[str, Any],
         stream: TextIO | None = None) -> None:
    """Print one record as ``<tag> <json>`` with sorted keys.

    One line, one record, keys sorted: a reader greps the tag and parses the
    rest with ``json.loads``, and two runs of the same environment print the
    same text.
    """
    stream = sys.stdout if stream is None else stream
    print(f"{tag} {json.dumps(dict(record), sort_keys=True)}", file=stream)
