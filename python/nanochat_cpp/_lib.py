"""Load the nanochat shared library once.

``load`` finds the library, declares the ctypes function table, applies the
``nanochat_init`` sandbox check, and caches the loaded library for the process.

The search order follows docs/python-api.md sections 5.1 and 8.3:

1. the path in ``NANOCHAT_CPP_LIB``;
2. the Bazel runfiles tree (a ``py_test`` or a ``bazel run``);
3. the development tree under ``bazel-bin/bindings``;
4. a library bundled next to the package;
5. the cache or an on-demand build from ``nanochat_cpp._build``.

This module imports only the Python standard library.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

from . import _build, _core

__all__ = [
    "ensure_library",
    "find_in_runfiles",
    "find_development_library",
    "find_bundled_library",
    "library_backend",
    "library_path",
    "load",
    "reset",
]

#: The relative runfiles path of the shared library.
RUNFILES_RELATIVE = f"bindings/{_build.LIBRARY_NAME}"

_lock = threading.Lock()
_library = None
_library_path: Path | None = None
_library_backend: str | None = None


def _wanted_backend(device) -> str | None:
    """Map a Python device request to ``"cpu"``, ``"cuda"``, or None.

    ``None`` means no constraint: use whatever backend the loaded library
    provides. A string of ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or a bare
    index selects that backend. When ``device`` is None, an explicit
    ``NANOCHAT_CPP_BACKEND`` value selects the backend; the unset default
    leaves the choice to the found library.
    """
    if device is None:
        device = os.environ.get("NANOCHAT_CPP_BACKEND")
        if device is None or not str(device).strip():
            return None
    if isinstance(device, int):
        return "cuda"
    text = str(device).strip().lower()
    if text in ("", "auto"):
        return None
    if text == "cpu":
        return "cpu"
    if text == "cuda" or text.startswith("cuda:"):
        return "cuda"
    if text.isdigit():
        return "cuda"
    raise ValueError(
        f"unknown device {device!r}; use 'cpu', 'cuda', 'cuda:N', or an index")


def _runfiles_roots() -> list[Path]:
    """Candidate runfiles directories, most specific first."""
    roots: list[Path] = []

    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        base = Path(runfiles)
        roots.append(base)
        if base.is_dir():
            roots.extend(child for child in sorted(base.iterdir())
                         if child.is_dir())

    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        base = Path(srcdir)
        workspace = os.environ.get("TEST_WORKSPACE")
        if workspace:
            roots.append(base / workspace)
        roots.append(base)
        if base.is_dir():
            roots.extend(child for child in sorted(base.iterdir())
                         if child.is_dir())

    argv0 = None
    if sys.argv and sys.argv[0]:
        argv0 = Path(sys.argv[0])
    if argv0 is not None:
        for suffix in (".runfiles", ".runfiles_manifest"):
            candidate = Path(str(argv0) + suffix)
            if candidate.is_dir():
                roots.append(candidate)

    # Remove duplicates while keeping the order.
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_in_runfiles(relative: str) -> Path | None:
    """Resolve ``relative`` in the Bazel runfiles tree, or return None."""
    relative_path = Path(relative)
    for root in _runfiles_roots():
        candidate = root / relative_path
        if candidate.is_file():
            return candidate

    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest:
        try:
            lines = Path(manifest).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        wanted = relative_path.as_posix()
        for line in lines:
            entry, _, value = line.partition(" ")
            if entry == wanted or entry.endswith("/" + wanted):
                candidate = Path(value)
                if candidate.is_file():
                    return candidate
    return None


def find_development_library() -> Path | None:
    """The library in the development tree, or None."""
    candidate = _build.repo_root() / "bazel-bin" / "bindings" / \
        _build.LIBRARY_NAME
    return candidate if candidate.is_file() else None


def find_bundled_library() -> Path | None:
    """A library placed next to this package, or None."""
    candidate = Path(__file__).resolve().parent / _build.LIBRARY_NAME
    return candidate if candidate.is_file() else None


def ensure_library(device=None) -> Path:
    """Return a usable library path without loading it.

    ``device`` constrains the backend. An explicit backend request prefers the
    Bazel runfiles library and then an on-demand build, because the
    development tree under ``bazel-bin`` may hold the other backend.
    """
    explicit = _build.explicit_library()
    if explicit is not None:
        return explicit

    runfiles = find_in_runfiles(RUNFILES_RELATIVE)
    if runfiles is not None:
        return runfiles

    want = _wanted_backend(device)
    if want is not None:
        try:
            return _build.ensure_library(backend_name=want)
        except _build.BuildError:
            # No build is possible; fall through to the search path and let
            # ``load`` verify the backend.
            pass

    development = find_development_library()
    if development is not None:
        return development

    bundled = find_bundled_library()
    if bundled is not None:
        return bundled

    return _build.ensure_library(backend_name=want)


def load(device=None):
    """Load the shared library once and return the declared handle.

    The first call applies ``nanochat_init``. A nonzero status becomes a
    ``NanochatError`` with the ``nanochat_last_error`` message. ``device``
    selects the backend on the first call and is checked on every later call,
    so a CPU process cannot silently create a CUDA model.
    """
    global _library, _library_path, _library_backend
    want = _wanted_backend(device)
    with _lock:
        if _library is None:
            path = ensure_library(device)
            library = ctypes.CDLL(str(path))
            _core.declare(library)
            _core.check_status(library, library.nanochat_init(),
                               "nanochat_init")
            actual = _core.backend_name(library)
            if want is not None and actual and actual != want:
                raise _core.NanochatError(
                    "nanochat_init",
                    f"the loaded library uses the {actual or 'unknown'} "
                    f"backend, but device {device!r} asks for {want}; build or "
                    f"point NANOCHAT_CPP_LIB at the {want} library")
            _library = library
            _library_path = path
            _library_backend = actual
        elif want is not None and _library_backend and _library_backend != want:
            raise _core.NanochatError(
                "nanochat_init",
                f"the process already loaded the {_library_backend} backend; "
                f"one process cannot use both backends")
    return _library


def library_path() -> Path:
    """The path of the loaded library, loading it when necessary."""
    if _library_path is None:
        load()
    return _library_path


def library_backend() -> str:
    """The backend of the loaded library: ``"cpu"`` or ``"cuda"``."""
    if _library is None:
        load()
    return _library_backend or ""


def reset() -> None:
    """Forget the cached library. Use only in a test that reloads it."""
    global _library, _library_path, _library_backend
    with _lock:
        _library = None
        _library_path = None
        _library_backend = None
