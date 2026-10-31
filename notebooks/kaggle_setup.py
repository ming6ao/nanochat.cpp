"""The Kaggle session setup for the notebook workflow.

A notebook installs the package with one ``pip install`` cell and configures
the environment with one cell, so this module has two paths and prefers the
first (``docs/dependency-layering-plan.md`` section 6.5):

* the installed package. ``pip install "nanochat-cpp[hub]"`` gives the two
  import packages, and the library search order of section 6.3 finds the shared
  library, so no clone and no build run;
* the repository. When the package is absent, or a caller asks for the
  fallback, the session fetches the tree, runs ``tools/kaggle/bootstrap.sh``,
  reads the host contract, and builds the library.

The pure helpers are testable without the shared library and without network
access. See ``docs/host-portability.md``, section 8.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_REPO_DEST",
    "DEFAULT_REPO_URL",
    "DISTRIBUTION",
    "Host",
    "MODE_INSTALLED",
    "MODE_SOURCE",
    "apply_host_path",
    "build_library",
    "ensure_repository",
    "installed_package",
    "installed_revision",
    "installed_root",
    "installed_version",
    "load_host_env",
    "parse_env_file",
    "revision",
    "run_bootstrap",
    "setup",
]

#: The default repository to clone into a Kaggle session.
DEFAULT_REPO_URL = "https://github.com/ming6ao/nanochat.cpp"

#: The default repository location on Kaggle.
DEFAULT_REPO_DEST = "/kaggle/working/nanochat.cpp"

#: The distribution that carries both import packages (D1 and D2).
DISTRIBUTION = "nanochat-cpp"

#: A session prepared from the installed distribution: no clone, no build.
MODE_INSTALLED = "installed"

#: A session prepared from a source-tree clone: the documented fallback.
MODE_SOURCE = "source"

#: The host contract file that ``tools/kaggle/bootstrap.sh`` writes.
HOST_ENV_NAME = ".nanochat.env"


def _unquote(value: str) -> str:
    """Remove one pair of matching quotes from ``value``."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``export KEY=value`` lines into a mapping.

    Blank lines, comment lines, and other lines are ignored. A value may be
    quoted with single or double quotes.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, separator, raw = line[len("export "):].partition("=")
        key = key.strip()
        if not key or not separator:
            continue
        values[key] = _unquote(raw.strip())
    return values


def load_host_env(path: str | Path | None = None) -> dict[str, str]:
    """Read the host contract file and apply it to ``os.environ``.

    ``path`` defaults to ``~/.nanochat.env``. A missing file is not an error,
    because the bootstrap may not have run yet.
    """
    path = Path(path) if path is not None else Path.home() / HOST_ENV_NAME
    if not path.is_file():
        return {}
    values = parse_env_file(path.read_text(encoding="utf-8"))
    os.environ.update(values)
    return values


def apply_host_path(home: str | Path | None = None) -> list[str]:
    """Prepend the Kaggle tool directories to ``PATH``.

    The directories hold Bazelisk, Node.js, and the image CUDA toolkit. The
    call uses ``CUDA_HOME`` when the host contract set it.
    """
    home = Path(home) if home is not None else Path.home()
    entries = [str(home / ".local" / "bin"),
               str(home / ".local" / "node" / "bin")]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        entries.append(str(Path(cuda_home) / "bin"))
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(entries + ([current] if current else []))
    return entries


def installed_package():
    """The installed ``nanochat_cpp`` module, or ``None`` when it is absent.

    A failed import is the signal that the session has no installed package, so
    :func:`setup` takes the repository fallback. The import runs here, inside
    the call, so a host that only needs the pure helpers pays nothing for it.
    """
    try:
        return importlib.import_module("nanochat_cpp")
    except ImportError:
        return None


def installed_root(package=None) -> Path:
    """The directory of the installed package: site-packages, or ``python/``.

    An installed session has no checkout, so this directory stands in for the
    repository root. A job that needs the source tree itself, such as the mesh
    driver that runs ``tools/nanochat client``, takes the
    :func:`setup` fallback instead.
    """
    package = installed_package() if package is None else package
    if package is None:
        raise RuntimeError(
            "no installed package: run pip install 'nanochat-cpp[hub]', or "
            "prepare the session from the source tree")
    location = getattr(package, "__file__", None)
    if not location:
        return Path.cwd()
    return Path(location).resolve().parent.parent


def installed_version() -> str:
    """The version of the installed distribution, or ``""`` when unknown.

    The version is read from the installed metadata, never from the tree, so an
    unknown version stays unknown (``docs/dependency-layering-plan.md`` section
    7.2).
    """
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return ""


def installed_revision() -> str:
    """The revision an installed session records: the distribution it runs.

    A wheel session has no checkout to ask, so the record names the
    distribution and its version, ``nanochat-cpp 0.1.0``. An unknown version
    stays an empty string, because a guessed version would name an environment
    that never existed (section 7.2).
    """
    version = installed_version()
    return f"{DISTRIBUTION} {version}" if version else ""


def _git(root: Path, *args: str, check: bool = True) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=check)


def ensure_repository(url: str = DEFAULT_REPO_URL,
                      dest: str | Path = DEFAULT_REPO_DEST,
                      ref: str | None = None) -> Path:
    """Clone the repository when absent, then fetch and pull."""
    dest = Path(dest)
    if dest.is_dir():
        _git(dest, "fetch", "--all", "--prune")
        if ref:
            _git(dest, "checkout", ref)
        _git(dest, "pull", "--ff-only", check=False)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    arguments = ["clone"]
    if ref:
        arguments += ["--branch", ref]
    subprocess.run(["git", *arguments, url, str(dest)], check=True)
    return dest


def revision(root: str | Path) -> str:
    """The short Git revision of the working tree at ``root``."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def run_bootstrap(root: str | Path) -> None:
    """Run ``tools/kaggle/bootstrap.sh`` for the session."""
    script = Path(root) / "tools" / "kaggle" / "bootstrap.sh"
    subprocess.run(["bash", str(script)], check=True)


def build_library(backend: str = "cuda", precision: str = "fp32",
                  arch: str = "sm_75",
                  cache: str | Path | None = None,
                  package=None) -> tuple[Path, str]:
    """Resolve the shared library and pin ``NANOCHAT_CPP_LIB``.

    Return the library path and the build key. The build key lets a trial
    fingerprint match the library the session actually runs.

    The call asks the package for the library, so the search order of
    ``docs/python.md`` decides: ``NANOCHAT_CPP_LIB``, the runfiles, the
    development tree, a library beside the package, an installed binary wheel,
    and only then the cache or an on-demand build (section 6.3). A session that
    installed the wheel therefore builds nothing.
    """
    module = installed_package() if package is None else package
    if module is None:
        import nanochat_cpp as module  # noqa: PLC0415

    options = module._build.BuildOptions.resolve(
        backend=backend, precision=precision, arch=arch, cache=cache)
    path = Path(module._build.ensure_library(options=options))
    os.environ["NANOCHAT_CPP_LIB"] = str(path)
    return path, module._build.build_key(options=options)


@dataclass
class Host:
    """The prepared Kaggle session.

    ``mode`` says which path prepared it: :data:`MODE_INSTALLED` for the
    installed distribution and :data:`MODE_SOURCE` for the clone fallback.
    """

    repo: Path
    library: Path
    build_key: str
    env: dict[str, str] = field(default_factory=dict)
    mode: str = MODE_SOURCE


def setup(url: str = DEFAULT_REPO_URL, dest: str | Path = DEFAULT_REPO_DEST,
          ref: str | None = None, backend: str = "cuda",
          precision: str = "fp32", arch: str = "sm_75",
          cache: str | Path | None = None,
          env_path: str | Path | None = None, package=None,
          prefer_installed: bool = True) -> Host:
    """Prepare the session: the installed package first, else the clone.

    ``prefer_installed`` is the default of section 6.5, because a host that ran
    the install cell must not clone or build. ``prefer_installed=False`` forces
    the repository path, which is the documented fallback and is what an
    offline session with a source tree uses. ``package`` is the module to use,
    and ``None`` looks the installed one up.
    """
    module = package
    if module is None and prefer_installed:
        module = installed_package()
    if module is not None:
        env = load_host_env(env_path)
        apply_host_path()
        library, build_key = build_library(
            backend=backend, precision=precision, arch=arch, cache=cache,
            package=module)
        return Host(repo=installed_root(module), library=library,
                    build_key=build_key, env=env, mode=MODE_INSTALLED)
    repo = ensure_repository(url=url, dest=dest, ref=ref)
    run_bootstrap(repo)
    env = load_host_env(env_path)
    apply_host_path()
    library, build_key = build_library(backend=backend, precision=precision,
                                       arch=arch, cache=cache)
    return Host(repo=repo, library=library, build_key=build_key, env=env,
                mode=MODE_SOURCE)
