"""Compile the nanochat shared library on demand.

The function ``ensure_library`` is the last step of the library search: the
runfiles tree, the development tree, a library beside the package, and an
installed binary wheel are tried first by ``nanochat_cpp._lib``, and what is
left here is

1. the path in ``NANOCHAT_CPP_LIB`` when the file exists;
2. a cached library when the build key matches;
3. a build of the library through ``tools/nanochat``;
4. a stop with a clear message when no compiler and no prebuilt library exist.

The stage executables use the same shape: :func:`stage_binary` resolves the
environment override, the Bazel runfiles tree, ``bazel-bin/src/<name>``, and
the ``nanochat-<name>`` entry point of a future wheel before it stops, so the
library and the two children of the chain are found by one rule
(docs/pipeline-plan.md B6).

The build key is a hash of the source contents, the compiler version, the
optional CUDA architecture, the precision, and the backend (section 3.3). Two
inputs that differ give a different cache entry, so a stale library is never
reused by accident.

This module imports only the Python standard library.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

__all__ = [
    "BuildError",
    "BuildOptions",
    "STAGE_NAMES",
    "SUPPORTED_PRECISIONS",
    "architecture",
    "backend",
    "build",
    "build_key",
    "build_stages",
    "cache_dir",
    "cached_library",
    "compiler_version",
    "ensure_library",
    "explicit_library",
    "find_compiler",
    "precision",
    "repo_root",
    "source_hash",
    "stage_binary",
]

#: The cache directory name under the user cache home.
CACHE_DIRNAME = "nanochat_cpp"

#: The library name that both the Bazel target and the cache use.
LIBRARY_NAME = "libnanochat_shared.so"

#: The Bazel target that builds the shared library.
SHARED_TARGET = "//bindings:nanochat_shared"

#: The two stage executables of the chain, in pipeline order
#: (docs/pipeline-plan.md B6). A name is both the Bazel label under ``//src``
#: and the program name of a future wheel entry point.
STAGE_NAMES = ("sft_main", "rl_worker")

#: Source directories whose contents change the compiled library.
_SOURCE_DIRS = ("bindings", "src", "include", "backends")

#: Root files whose contents change the compiled library.
_SOURCE_FILES = ("nanochat.bzl", "MODULE.bazel", ".bazelrc", "BUILD.bazel")

#: File suffixes that hold C, C++, or Bazel build input.
_SOURCE_SUFFIXES = (".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cu", ".bzl")

#: Compiler programs to look for, in preference order.
_COMPILERS = ("c++", "g++", "clang++", "nvcc")

#: Environment values that mean "true".
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class BuildError(RuntimeError):
    """The shared library is absent and no build can supply it."""


def repo_root() -> Path:
    """The repository root that holds ``tools/nanochat``.

    This file is ``<root>/python/nanochat_cpp/_build.py``, so the root is three
    parents up.
    """
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    """The build cache: ``NANOCHAT_CPP_CACHE`` or ``~/.cache/nanochat_cpp``."""
    value = os.environ.get("NANOCHAT_CPP_CACHE")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / CACHE_DIRNAME


#: The build precisions the tree supports (docs/precision.md).
SUPPORTED_PRECISIONS = ("fp32", "bf16")


def precision() -> str:
    """The requested precision: ``"fp32"`` or ``"bf16"``."""
    return os.environ.get("NANOCHAT_CPP_PRECISION", "fp32").strip().lower()


def backend(override: str | None = None) -> str:
    """The requested backend: ``"cpu"`` by default, or ``"cuda"``.

    ``override`` names the backend directly, so a caller such as ``Model``
    can select the CUDA library without changing the process environment.
    """
    if override is not None:
        return str(override).strip().lower()
    return os.environ.get("NANOCHAT_CPP_BACKEND", "cpu").strip().lower()


def architecture() -> str:
    """The requested CUDA architecture, or an empty string.

    ``NANOCHAT_CUDA_ARCH`` names the architecture by hand, for example
    ``sm_75``. When it is unset the builder leaves the choice to Bazel.
    """
    return os.environ.get("NANOCHAT_CUDA_ARCH", "").strip()


@dataclasses.dataclass(frozen=True)
class BuildOptions:
    """The explicit inputs of one on-demand build.

    ``from_env`` reads the current environment, so ``resolve`` with no
    arguments reproduces the historical behavior. ``resolve`` starts from the
    environment and overrides each field that the caller passes.
    """

    backend: str
    precision: str
    arch: str
    cache: Path

    @classmethod
    def from_env(cls, backend_name: str | None = None) -> "BuildOptions":
        """Read the build inputs from the current environment."""
        return cls(
            backend=backend(backend_name),
            precision=precision(),
            arch=architecture(),
            cache=cache_dir(),
        )

    @classmethod
    def resolve(cls, backend: str | None = None,
                precision: str | None = None, arch: str | None = None,
                cache: Path | None = None,
                backend_name: str | None = None) -> "BuildOptions":
        """Start from the environment and override each supplied field."""
        base = cls.from_env(backend_name)
        return cls(
            backend=str(backend).strip().lower() if backend is not None
            else base.backend,
            precision=str(precision).strip().lower() if precision is not None
            else base.precision,
            arch=str(arch).strip() if arch is not None else base.arch,
            cache=Path(cache).expanduser() if cache is not None else base.cache,
        )


def _is_true(value: str) -> bool:
    return value.strip().lower() in _TRUE_VALUES


def prebuilt_only() -> bool:
    """True when ``NANOCHAT_CPP_PREBUILT`` forbids a build."""
    return _is_true(os.environ.get("NANOCHAT_CPP_PREBUILT", ""))


def source_files(root: Path | None = None) -> list[Path]:
    """Every source file that can change the compiled library, sorted."""
    root = Path(root) if root is not None else repo_root()
    found: set[Path] = set()
    for name in _SOURCE_DIRS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
                found.add(path)
    for name in _SOURCE_FILES:
        path = root / name
        if path.is_file():
            found.add(path)
    return sorted(found)


def source_hash(root: Path | None = None) -> str:
    """A SHA-256 hex digest over the relative path and the bytes of each source."""
    root = Path(root) if root is not None else repo_root()
    digest = hashlib.sha256()
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def find_compiler() -> str | None:
    """The first C or C++ compiler on ``PATH``, or None."""
    for name in _COMPILERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def compiler_version() -> str:
    """A short compiler identity string for the build key."""
    compiler = find_compiler()
    if compiler is None:
        return "none"
    try:
        result = subprocess.run([compiler, "--version"], capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return Path(compiler).name
    lines = (result.stdout or result.stderr).splitlines()
    if lines:
        return lines[0].strip()
    return Path(compiler).name


def build_key(root: Path | None = None,
              backend_name: str | None = None,
              options: BuildOptions | None = None) -> str:
    """A short key that changes when any build input changes."""
    if options is None:
        options = BuildOptions.from_env(backend_name)
    material = "\n".join((
        source_hash(root),
        compiler_version(),
        options.arch,
        options.precision,
        options.backend,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def explicit_library() -> Path | None:
    """The readable path in ``NANOCHAT_CPP_LIB``, or None."""
    value = os.environ.get("NANOCHAT_CPP_LIB")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def cached_library(key: str | None = None,
                   directory: Path | None = None) -> Path:
    """The cache path for a build key (the current key by default)."""
    if key is None:
        key = build_key()
    directory = Path(directory) if directory is not None else cache_dir()
    return directory / f"libnanochat_shared_{key}.so"


def _config_flag(options: BuildOptions) -> str:
    """The named Bazel configuration for the requested backend."""
    if options.precision not in SUPPORTED_PRECISIONS:
        raise BuildError(
            f"{options.precision} is not a build precision: "
            f"{' or '.join(SUPPORTED_PRECISIONS)} (docs/precision.md)")
    if options.backend == "cpu":
        return "--config=cpu"
    return "--config=cuda"


def _no_library_message(cache: Path) -> str:
    entry = repo_root() / "tools" / "nanochat"
    return (
        "no prebuilt nanochat shared library and no C++ compiler: set "
        f"NANOCHAT_CPP_LIB to a prebuilt {LIBRARY_NAME}, place one in "
        f"{cache}, install c++ or g++, or run "
        f"{entry} build {SHARED_TARGET} by hand"
    )


def _build_command(entry: Path, options: BuildOptions,
                   targets: Sequence[str] = (SHARED_TARGET,)) -> list[str]:
    """The ``tools/nanochat build`` command for one set of options.

    ``targets`` defaults to the shared library. :func:`build_stages` passes the
    library and the two stage executables, so this function stays the one home
    of the command line and of the configuration flags it carries.

    A command-line flag wins over the ``.bazelrc`` value, so an explicit CUDA
    architecture selects the compiled target and not only the cache key.
    """
    command = [str(entry), "build", *targets, _config_flag(options)]
    # A command-line flag wins over a value in `.bazelrc`, so the precision is
    # explicit even though `--config=cpu` and `--config=cuda` also set one
    # (docs/precision.md).
    command.append(f"--define=precision={options.precision}")
    if options.backend == "cuda" and options.arch:
        command.append(f"--@rules_cuda//cuda:archs={options.arch}")
    return command


def build(key: str | None = None, root: Path | None = None,
          backend_name: str | None = None,
          options: BuildOptions | None = None) -> Path:
    """Build the shared library and return its path.

    A CPU build copies the library into the cache and returns the cached path.
    A CUDA build returns the ``bazel-bin`` path, because the library needs the
    runpath beside it. Raises ``BuildError`` when the entry point is missing or
    the build fails.
    """
    if options is None:
        options = BuildOptions.from_env(backend_name)
    root = Path(root) if root is not None else repo_root()
    entry = root / "tools" / "nanochat"
    if not entry.is_file():
        raise BuildError(
            f"cannot build the shared library: {entry} is missing; set "
            f"NANOCHAT_CPP_LIB to a prebuilt {LIBRARY_NAME}")
    command = _build_command(entry, options)
    try:
        result = subprocess.run(command, cwd=str(root), capture_output=True,
                                text=True)
    except OSError as error:
        raise BuildError(f"cannot run {' '.join(command)}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise BuildError(
            f"the build command failed ({result.returncode}): "
            f"{' '.join(command)}\n{detail}")
    built = root / "bazel-bin" / "bindings" / LIBRARY_NAME
    if not built.is_file():
        raise BuildError(
            f"the build reported success but {built} is missing")
    if options.backend == "cuda":
        # The CUDA shared library carries a runpath into the Bazel solib tree
        # next to `bazel-bin`. A copy in the cache would lose that runpath and
        # fail to find libcudart and libcublas, so return the built path.
        return built
    directory = options.cache
    directory.mkdir(parents=True, exist_ok=True)
    if key is None:
        key = build_key(root, backend_name, options)
    destination = cached_library(key, directory)
    shutil.copy2(built, destination)
    return destination


def stage_override(name: str) -> str:
    """The environment variable that overrides the stage executable ``name``.

    ``sft_main`` is ``NANOCHAT_SFT_MAIN`` and ``rl_worker`` is
    ``NANOCHAT_RL_WORKER``: the program name upper-cased behind the prefix, so
    a new stage needs no second table (docs/pipeline-plan.md B6).
    """
    return "NANOCHAT_" + str(name).upper()


def stage_target(name: str) -> str:
    """The Bazel label of the stage executable ``name``: ``//src:<name>``."""
    return f"//src:{name}"


def _stages_command(entry: Path, options: BuildOptions) -> list[str]:
    """The one ``tools/nanochat build`` command of :func:`build_stages`."""
    targets = (SHARED_TARGET, *(stage_target(name) for name in STAGE_NAMES))
    return _build_command(entry, options, targets)


def stage_binary(name: str, options: BuildOptions | None = None,
                 root: Path | None = None) -> Path:
    """Resolve one stage executable, or raise :class:`BuildError`.

    The order is the one B6 fixes, which is the order of ``nanochat_cpp._lib``
    rather than the shorter list of ``docs/python.md`` section 3.2:

    1. the ``NANOCHAT_SFT_MAIN`` or ``NANOCHAT_RL_WORKER`` override, which wins
       even when the file is absent, because the caller named it;
    2. the Bazel runfiles tree, so a ``py_test`` or a ``bazel run`` finds the
       executable its data dependencies carry;
    3. ``bazel-bin/src/<name>``, the development tree;
    4. ``shutil.which("nanochat-" + name)``, the entry point of a future wheel;
    5. an error that names the build command, so a tree with no build fails
       with one line a reader can paste.

    A wheel install has no ``bazel-bin``, so a facade that takes a default
    binary fails here with this message and not with a file-not-found from the
    launcher (documents decision D8).

    ``root`` overrides the tree the development build is looked up in, and
    ``options`` only shapes the message, so a test can resolve a fake tree
    without building one.
    """
    override = os.environ.get(stage_override(name), "").strip()
    if override:
        return Path(override).expanduser()

    # The runfiles search lives in ``_lib``, which imports this module, so the
    # import is local: one runfiles implementation for the library and for the
    # stage executables.
    from . import _lib

    runfiles = _lib.find_in_runfiles(f"src/{name}")
    if runfiles is not None:
        return runfiles

    tree = Path(root) if root is not None else repo_root()
    development = tree / "bazel-bin" / "src" / str(name)
    if development.is_file():
        return development

    installed = shutil.which("nanochat-" + str(name))
    if installed:
        return Path(installed)

    if options is None:
        options = BuildOptions.from_env()
    command = _command_text(_stages_command(
        tree / "tools" / "nanochat", options))
    raise BuildError(
        f"cannot find the {name} executable: {stage_override(name)} is unset, "
        f"there is no runfiles entry, {development} does not exist, and "
        f"nanochat-{name} is not on PATH; build it with: {command}")


def _command_text(command: Sequence[str]) -> str:
    """One shell-readable line for a command list."""
    return " ".join(str(part) for part in command)


def build_stages(options: BuildOptions | None = None,
                 root: Path | None = None) -> tuple[Path, Path, Path]:
    """Build the shared library and both stage executables in one call.

    The chain needs the library and the two children, and one
    ``tools/nanochat build`` command names all three, so a host runs one build
    instead of three (docs/pipeline-plan.md B6 and D2). ``_build_command``
    stays the one home of the command line: this function adds the two stage
    targets and nothing else.

    Returns ``(library, sft_main, rl_worker)``, resolved the way the rest of
    the package resolves them. Raises :class:`BuildError` when the entry point
    is missing, the build fails, or an output is absent afterwards.
    """
    if options is None:
        options = BuildOptions.from_env()
    root = Path(root) if root is not None else repo_root()
    entry = root / "tools" / "nanochat"
    if not entry.is_file():
        raise BuildError(
            f"cannot build the stage executables: {entry} is missing; a tree "
            f"checkout is the only build route until a binary wheel exists "
            f"(documents decision D8)")
    command = _stages_command(entry, options)
    try:
        result = subprocess.run(command, cwd=str(root), capture_output=True,
                                text=True)
    except OSError as error:
        raise BuildError(
            f"cannot run {_command_text(command)}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise BuildError(
            f"the stage build failed ({result.returncode}): "
            f"{_command_text(command)}\n{detail}")
    library = root / "bazel-bin" / "bindings" / LIBRARY_NAME
    stages = tuple(root / "bazel-bin" / "src" / name for name in STAGE_NAMES)
    missing = [str(path) for path in (library, *stages) if not path.is_file()]
    if missing:
        verb = "are" if len(missing) > 1 else "is"
        raise BuildError(
            f"the stage build reported success but {', '.join(missing)} "
            f"{verb} missing; the command was {_command_text(command)}")
    return (library,
            *(stage_binary(name, options, root) for name in STAGE_NAMES))


def ensure_library(root: Path | None = None,
                   backend_name: str | None = None,
                   options: BuildOptions | None = None) -> Path:
    """Return a usable library path, or raise ``BuildError``.

    The order is the one in docs/python.md section 3.2: the cache, then a
    build. ``nanochat_cpp._lib`` performs the runfiles and development-tree
    search before it calls this function.
    """
    if options is None:
        options = BuildOptions.from_env(backend_name)
    root = Path(root) if root is not None else repo_root()

    explicit = explicit_library()
    if explicit is not None:
        return explicit

    key = build_key(root, backend_name, options)
    directory = options.cache
    cached = cached_library(key, directory)
    if cached.is_file() and options.backend != "cuda":
        return cached

    if prebuilt_only():
        raise BuildError(
            f"NANOCHAT_CPP_PREBUILT is set but no library was found: set "
            f"NANOCHAT_CPP_LIB to a prebuilt {LIBRARY_NAME} or place one in "
            f"{directory}")

    if find_compiler() is None:
        raise BuildError(_no_library_message(directory))

    return build(key, root, backend_name, options)
