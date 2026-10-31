"""Load the nanochat shared library once.

``load`` finds the library, declares the ctypes function table, applies the
``nanochat_init`` sandbox check, and caches the loaded library for the process.

The search order follows docs/python.md sections 5.1 and 8.3, with the
installed-wheel entry of ``docs/dependency-layering-plan.md`` section 6.3 as
item 5:

1. the path in ``NANOCHAT_CPP_LIB``;
2. the Bazel runfiles tree (a ``py_test`` or a ``bazel run``);
3. the development tree under ``bazel-bin/bindings``;
4. a library bundled next to the package;
5. an installed ``nanochat_cpp_cuda_<arch>`` wheel;
6. the cache or an on-demand build from ``nanochat_cpp._build``.

The two stage executables of the chain follow the same shape, because
:func:`nanochat_cpp._build.stage_binary` resolves them in this module's order
and not in the shorter list of ``docs/python.md`` section 3.2 that omits them:
the override, the runfiles tree, ``bazel-bin/src/<name>``, and the
``nanochat-<name>`` entry point of a future wheel. A library and a stage
executable are then found by one rule, so a reader who trusts one docstring
trusts the other (``docs/pipeline-plan.md`` B6).

Item 5 is how a host that installed a binary wheel finds its library without a
repository clone (section 6.4). A wheel is named for the CUDA architecture it
was built for, ``nanochat_cpp_cuda_sm75`` for example, so a request for another
architecture fails at ``nanochat_init`` with the ``pip install`` line that
fixes it. Nothing here builds or publishes a wheel: when none is installed the
search falls through to item 6, and a build failure says so in the message.

This module imports only the Python standard library.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import sys
import threading
from pathlib import Path

from . import _build, _core

__all__ = [
    "WHEEL_DISTRIBUTION_PREFIX",
    "ensure_library",
    "find_in_runfiles",
    "find_development_library",
    "find_bundled_library",
    "find_wheel_library",
    "library_architecture",
    "library_backend",
    "library_path",
    "load",
    "normalize_arch",
    "reset",
    "wheel_architectures",
    "wheel_architecture",
    "wheel_distribution_name",
]

#: The relative runfiles path of the shared library.
RUNFILES_RELATIVE = f"bindings/{_build.LIBRARY_NAME}"

#: The distribution name prefix of the per-architecture binary wheel.
WHEEL_DISTRIBUTION_PREFIX = "nanochat_cpp_cuda_"

#: ``sm_75``, ``sm75``, and ``7.5`` all name one architecture.
_ARCH_PREFIXES = ("sm_", "sm-", "sm")

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


def normalize_arch(arch: str) -> str:
    """Normalise a CUDA architecture to the ``sm_75`` form of a wheel name.

    ``sm_75``, ``sm75``, and ``7.5`` all name one architecture; a trailing
    letter survives (``sm90a`` -> ``sm_90a``). An empty value answers an empty
    string, because no architecture was asked for.
    """
    text = str(arch).strip().lower()
    for prefix in _ARCH_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.replace(".", "")
    return f"sm_{text}" if text else ""


def wheel_distribution_name(arch: str) -> str:
    """The distribution name of the wheel for ``arch``.

    Section 6.3 spells the wheel ``nanochat_cpp_cuda_sm75``: the prefix and the
    architecture without its separator. An empty architecture yields the
    prefix plus ``<arch>``, so a message can print it as a placeholder.
    """
    tag = normalize_arch(arch).replace("_", "")
    return f"{WHEEL_DISTRIBUTION_PREFIX}{tag or '<arch>'}"


def wheel_architecture() -> str:
    """The CUDA architecture a caller asks a wheel for, or an empty string.

    ``NANOCHAT_CUDA_ARCH`` names it by hand, the same variable the Bazel build
    reads (``nanochat_cpp._build.architecture``), and the value is normalised
    to the wheel spelling of section 6.3: ``sm_75``. Unset means no
    preference, so any installed wheel is usable. There is one environment
    variable for the architecture and not two, so the build and the wheel
    search cannot disagree.
    """
    requested = _build.architecture()
    return normalize_arch(requested) if requested.strip() else ""


def wheel_architectures() -> dict[str, Path]:
    """The installed ``nanochat_cpp_cuda_<arch>`` wheels, arch -> library path.

    A wheel is named for the architecture it was built for, and it carries the
    shared library as a packaged file, so the RECORD of the distribution says
    where the library landed without importing anything. A distribution with
    no library in its RECORD is skipped.
    """
    found: dict[str, Path] = {}
    try:
        distributions = list(importlib.metadata.distributions())
    except Exception:  # noqa: BLE001 - a broken metadata tree is not fatal
        return found
    for distribution in distributions:
        # A distribution name is case-insensitive and treats ``-``, ``_`` and
        # ``.`` alike, so a wheel named ``nanochat-cpp-cuda-sm75`` and one
        # named ``nanochat_cpp_cuda_sm75`` are the same wheel.
        name = _distribute_name(distribution.metadata["Name"] or "")
        if not name.startswith(WHEEL_DISTRIBUTION_PREFIX):
            continue
        arch = normalize_arch(name[len(WHEEL_DISTRIBUTION_PREFIX):])
        library = None
        for entry in distribution.files or ():
            if Path(entry.name).name == _build.LIBRARY_NAME:
                library = Path(distribution.locate_file(entry))
                break
        if library is not None and library.is_file():
            found[arch] = library
    return found


def _distribute_name(name: str) -> str:
    """The normalised distribution name, comparable to a prefix."""
    text = name.strip().lower()
    for separator in ("-", "."):
        text = text.replace(separator, "_")
    return text


def find_wheel_library(arch: str | None = None) -> Path | None:
    """The shared library of the installed binary wheel, or None.

    ``arch`` asks for one architecture (``sm_75`` or any spelling
    :func:`normalize_arch` accepts); omitting it takes any installed wheel,
    lowest architecture first so a result is deterministic. An absent wheel
    returns None, and the search falls through to the cache and the on-demand
    build of ``nanochat_cpp._build``.
    """
    wheels = wheel_architectures()
    if not wheels:
        return None
    if arch:
        return wheels.get(normalize_arch(arch))
    return wheels[sorted(wheels)[0]]


def library_architecture(path) -> str:
    """The wheel architecture a library path came from, or an empty string.

    A library found in the repository, the cache, or the runfiles tree has no
    wheel behind it, so it answers an empty string and no architecture check
    applies.
    """
    resolved = Path(path).resolve()
    for arch, library in wheel_architectures().items():
        if library.resolve() == resolved:
            return arch
    return ""


def _wrong_arch_message(asked: str, built: str) -> str:
    """The error of a wheel built for another architecture (section 6.3)."""
    return (f"the installed wheel carries a library built for {built}, but "
            f"NANOCHAT_CUDA_ARCH asks for {asked}; install the matching "
            f"wheel: pip install 'nanochat-cpp' "
            f"{wheel_distribution_name(asked)}")


def _absent_wheel_message(want: str | None) -> str:
    """The hint appended to a failed build when no usable wheel was found.

    Two shapes, both with the ``pip install`` line that fixes it: a wheel built
    for another architecture is the likely cause when one is installed, and an
    absent wheel is the cause when none is (section 6.3).
    """
    asked = wheel_architecture()
    wheels = wheel_architectures()
    if asked and wheels and asked not in wheels:
        installed = ", ".join(wheel_distribution_name(arch)
                              for arch in sorted(wheels))
        return (f"the installed wheel {installed} is built for another "
                f"architecture than the requested {asked}; install the "
                f"matching wheel: pip install 'nanochat-cpp' "
                f"{wheel_distribution_name(asked)}")
    name = wheel_distribution_name(asked)
    backend = want or "cpu or cuda"
    return (f"no {name} wheel is installed for the {backend} backend either: "
            f"pip install 'nanochat-cpp' {name}")


def ensure_library(device=None) -> Path:
    """Return a usable library path without loading it.

    The order is the one in the module docstring: ``NANOCHAT_CPP_LIB``, the
    runfiles tree, the development tree, a library next to the package, the
    installed binary wheel, and finally the cache or an on-demand build.

    ``device`` constrains the backend. An explicit backend request prefers the
    wheel and then an on-demand build before the development tree, because the
    development tree under ``bazel-bin`` may hold the other backend.
    """
    explicit = _build.explicit_library()
    if explicit is not None:
        return explicit

    runfiles = find_in_runfiles(RUNFILES_RELATIVE)
    if runfiles is not None:
        return runfiles

    wheel = find_wheel_library(wheel_architecture() or None)
    want = _wanted_backend(device)
    if want is not None:
        if wheel is not None:
            return wheel
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

    if wheel is not None:
        return wheel

    try:
        return _build.ensure_library(backend_name=want)
    except _build.BuildError as error:
        raise _build.BuildError(
            f"{error}\n{_absent_wheel_message(want)}") from error


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
            asked = wheel_architecture()
            built = library_architecture(path)
            if want == "cuda" and asked and built and built != asked:
                raise _core.NanochatError(
                    "nanochat_init", _wrong_arch_message(asked, built))
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
