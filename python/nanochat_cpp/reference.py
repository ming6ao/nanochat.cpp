"""Locate the reference PyTorch nanochat checkout.

The bridge uses the reference package for two things: the BPE tokenizer and its
per-token byte lengths, and the model/parameter-count math that sets the
training horizon. Both must match ``scripts.base_train.py`` exactly, so the
reference implementation is imported rather than reimplemented.

Set ``NANOCHAT_REPO`` to override the search, and ``NANOCHAT_BASE_DIR`` to
override the data/tokenizer cache root (defaults to ``~/.cache/nanochat``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_reference_repo() -> Path:
    """Return the directory that contains the reference ``nanochat`` package."""
    candidates = []
    if os.environ.get("NANOCHAT_REPO"):
        candidates.append(Path(os.environ["NANOCHAT_REPO"]))
    # python/nanochat_cpp/reference.py -> repository root is parents[2].
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / "nanochat")  # sibling checkout
    candidates.append(repo_root / "nanochat")
    candidates.append(Path.home() / "repos" / "nanochat")
    for candidate in candidates:
        if (candidate / "nanochat" / "gpt.py").is_file():
            return candidate.resolve()
    raise SystemExit(
        "could not locate the PyTorch nanochat checkout; set NANOCHAT_REPO to "
        "the directory that contains the `nanochat` package")


def ensure_reference_on_path() -> Path:
    """Make ``import nanochat`` work and pin the base data directory."""
    repo = find_reference_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault(
        "NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    return repo


def repository_root() -> Path:
    """The nanochat.cpp repository root (``python/nanochat_cpp`` -> up two)."""
    return Path(__file__).resolve().parents[2]
