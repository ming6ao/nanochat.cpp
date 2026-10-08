"""The Kaggle session setup for the notebook workflow.

The notebook loader cell clones the repository and adds ``python/`` and
``notebooks/`` to ``sys.path``. This module does the rest: fetch, bootstrap,
the host environment, and the library build. The pure helpers are testable
without the shared library. See ``docs/host-portability.md``, section 8.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_REPO_DEST",
    "DEFAULT_REPO_URL",
    "Host",
    "apply_host_path",
    "build_library",
    "ensure_repository",
    "load_host_env",
    "parse_env_file",
    "run_bootstrap",
    "setup",
]

#: The default repository to clone into a Kaggle session.
DEFAULT_REPO_URL = "https://github.com/ming6ao/nanochat.cpp"

#: The default repository location on Kaggle.
DEFAULT_REPO_DEST = "/kaggle/working/nanochat.cpp"

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


def load_host_env(path=None) -> dict[str, str]:
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


def apply_host_path(home=None) -> list[str]:
    """Prepend the Kaggle tool directories to ``PATH``.

    The directories hold Bazelisk, Node.js, and the image CUDA toolkit. The
    call uses ``CUDA_HOME`` when the host contract set it.
    """
    home = Path(home) if home is not None else Path.home()
    directories = [home / ".local" / "bin", home / ".local" / "node" / "bin"]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        directories.append(Path(cuda_home) / "bin")
    entries = [str(directory) for directory in directories]
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(entries + ([current] if current else []))
    return entries


def _git(root: Path, *args: str, check: bool = True) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=check)


def ensure_repository(url=DEFAULT_REPO_URL, dest=DEFAULT_REPO_DEST,
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


def run_bootstrap(root) -> Path:
    """Run ``tools/kaggle/bootstrap.sh`` for the session."""
    script = Path(root) / "tools" / "kaggle" / "bootstrap.sh"
    subprocess.run(["bash", str(script)], check=True)
    return script


def build_library(backend: str = "cuda", precision: str = "fp16",
                  arch: str = "sm_75", cache=None) -> tuple[Path, str]:
    """Build the shared library and pin ``NANOCHAT_CPP_LIB``.

    Return the library path and the build key. The build key lets a trial
    fingerprint match the library that the session actually built.
    """
    import nanochat_cpp as nc

    path = Path(nc.build(backend=backend, precision=precision, arch=arch,
                         cache=cache))
    os.environ["NANOCHAT_CPP_LIB"] = str(path)
    options = nc._build.BuildOptions.resolve(
        backend=backend, precision=precision, arch=arch, cache=cache)
    return path, nc._build.build_key(options=options)


@dataclass
class Host:
    """The prepared Kaggle session."""

    repo: Path
    library: Path
    build_key: str
    env: dict[str, str] = field(default_factory=dict)


def setup(url: str = DEFAULT_REPO_URL, dest=DEFAULT_REPO_DEST,
          ref: str | None = None, backend: str = "cuda",
          precision: str = "fp16", arch: str = "sm_75", cache=None,
          env_path=None) -> Host:
    """Prepare the session: fetch, bootstrap, host environment, and build."""
    repo = ensure_repository(url=url, dest=dest, ref=ref)
    run_bootstrap(repo)
    env = load_host_env(env_path)
    apply_host_path()
    library, build_key = build_library(backend=backend, precision=precision,
                                       arch=arch, cache=cache)
    return Host(repo=repo, library=library, build_key=build_key, env=env)
