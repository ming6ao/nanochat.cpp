"""The shared notebook session setup for the three notebooks.

A notebook installs the package with one cell and configures the environment
with one cell (``docs/dependency-layering-plan.md`` section 6.5). The bodies
live here, so the code has one home and the tests can drive it.

:func:`setup` is the one entry point. It prefers the installed distribution: a
host that can import ``nanochat_cpp`` prepares the session from it, so no clone
and no Bazel build run, and the shared library comes from the search order of
section 6.3, which reads an installed binary wheel before it builds anything.
When the package is absent, :func:`clone_and_build` is the documented fallback,
unchanged in behaviour: it fetches the repository, runs the bootstrap, and
builds the library, so an offline session with a source tree still works. The
choice is visible as :attr:`Session.mode`.

The module wraps ``notebooks/kaggle_setup.py``, which owns the fetch, the
bootstrap, the host environment, and the library build. This module adds the
notebook secrets, the staging root, the device check, and the run host.

The module imports ``kaggle_secrets`` inside :func:`load_secrets` and
``nanochat_cpp`` inside :func:`require_devices`, so the tests import without
the shared library and without network access. See
``docs/checkpoint-format.md`` section 15.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import kaggle_setup
import target as targets

__all__ = [
    "DEFAULT_BASE",
    "SECRET_NAMES",
    "Session",
    "bootstrap",
    "clone_and_build",
    "describe",
    "installed_session",
    "load_secrets",
    "require_devices",
    "setup",
]

#: The secret names a notebook reads. Only ``HF_TOKEN`` is required. The other
#: names authenticate the ``pi`` agent in ``setup.ipynb``.
SECRET_NAMES = ("HF_TOKEN",)

#: The staging root a notebook uses when the caller names none.
DEFAULT_BASE = targets.NOTEBOOK_BASE


@dataclasses.dataclass(frozen=True)
class Session:
    """One prepared notebook session.

    ``host`` is the :class:`kaggle_setup.Host`, whose mode says whether the
    session came from the installed package or from a source tree. ``secrets``
    holds the values the loader found. ``devices`` holds the device records the
    notebook needs. ``base`` is the staging root, also set as
    ``NANOCHAT_BASE_DIR``. ``git_revision`` is the revision a session without a
    checkout records, and it is empty when the repository supplies one.
    ``target`` is the run host, and it defaults to the local host.
    """

    host: kaggle_setup.Host
    secrets: dict
    devices: list
    base: Path
    target: targets.Target = dataclasses.field(
        default_factory=targets.LocalTarget)
    git_revision: str = ""

    @property
    def repo(self) -> Path:
        """The repository path, or the installed package's directory."""
        return Path(self.host.repo)

    @property
    def library(self) -> Path:
        """The path of the shared library."""
        return Path(self.host.library)

    @property
    def build_key(self) -> str:
        """The build key of the shared library."""
        return self.host.build_key

    @property
    def mode(self) -> str:
        """How the session was prepared, ``installed`` or ``source``."""
        return getattr(self.host, "mode", kaggle_setup.MODE_SOURCE)

    @property
    def installed(self) -> bool:
        """True when the session came from the installed distribution."""
        return self.mode == kaggle_setup.MODE_INSTALLED

    @property
    def revision(self) -> str:
        """The revision to record: the checkout's, or the distribution's.

        An installed session has no repository to ask, so it records the
        distribution it runs. An unknown version stays empty rather than
        guessed (``docs/dependency-layering-plan.md`` section 7.2).
        """
        if self.git_revision:
            return self.git_revision
        if self.installed:
            return ""
        return kaggle_setup.revision(self.repo)


def load_secrets(names=SECRET_NAMES, client=None, environ=None) -> dict:
    """Read the named Kaggle secrets and put them in the environment.

    A missing secret prints one line and is absent from the result. The caller
    decides whether the secret is required. ``client`` defaults to
    ``kaggle_secrets.UserSecretsClient``. Pass ``client`` and ``environ`` to
    test without the Kaggle package.
    """
    if client is None:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

        client = UserSecretsClient()
    environ = os.environ if environ is None else environ
    values = {}
    for name in names:
        try:
            value = client.get_secret(name)
        except Exception:  # noqa: BLE001 - a missing secret is reported
            print(f"missing secret {name}")
            continue
        values[name] = value
        environ[name] = value
    return values


def require_devices(count=1, devices=None) -> list:
    """The local devices, or a ``RuntimeError`` when there are too few.

    ``devices`` defaults to ``nanochat_cpp.devices()``. Pass a list to test
    the count check without the shared library.
    """
    if devices is None:
        import nanochat_cpp as nc  # noqa: PLC0415

        devices = nc.devices()
    devices = list(devices)
    if len(devices) < int(count):
        raise RuntimeError(
            f"this notebook needs {int(count)} device(s), "
            f"but the host has {len(devices)}")
    return devices


def installed_session(package, *, arch="sm_75", devices=1, base=None,
                      backend="cuda", precision="fp32", env_path=None,
                      secrets=None, target=None) -> Session:
    """Prepare the session from the installed distribution.

    No clone and no build run: the library comes from the search order of
    ``docs/dependency-layering-plan.md`` section 6.3, which ends at the
    installed ``nanochat_cpp_cuda_<arch>`` wheel. ``package`` is the imported
    ``nanochat_cpp`` module. ``base`` becomes ``NANOCHAT_BASE_DIR`` and
    defaults to the staging root of the run host.
    """
    if secrets is None:
        secrets = load_secrets()
    resolved = targets.resolve(target)
    base_path = Path(base) if base is not None else resolved.cache_root()
    os.environ["NANOCHAT_BASE_DIR"] = str(base_path)
    host = kaggle_setup.setup(backend=backend, precision=precision, arch=arch,
                              env_path=env_path, package=package,
                              prefer_installed=True)
    device_list = (require_devices(devices)
                   if resolved.needs_local_devices else [])
    return Session(host=host, secrets=secrets, devices=device_list,
                   base=base_path, target=resolved,
                   git_revision=kaggle_setup.installed_revision())


def clone_and_build(repo_url=kaggle_setup.DEFAULT_REPO_URL,
                    repo_path=kaggle_setup.DEFAULT_REPO_DEST, ref=None,
                    arch="sm_75", devices=1, base=None, backend="cuda",
                    precision="fp32", env_path=None, secrets=None,
                    target=None) -> Session:
    """Fetch, bootstrap, build, and check the session.

    This is the documented fallback of section 6.5: it runs when the package is
    not installed, and it keeps an offline session with a source tree working.
    It always takes the repository path, so it never silently becomes the
    default of :func:`setup`.

    ``base`` becomes ``NANOCHAT_BASE_DIR`` and defaults to the staging root of
    the run host. ``devices`` is the number of cards the notebook needs. A
    remote run host runs the job on its own card set, so it needs none.
    """
    if secrets is None:
        secrets = load_secrets()
    resolved = targets.resolve(target)
    base_path = Path(base) if base is not None else resolved.cache_root()
    os.environ["NANOCHAT_BASE_DIR"] = str(base_path)
    host = kaggle_setup.setup(url=repo_url, dest=repo_path, ref=ref,
                              backend=backend, precision=precision,
                              arch=arch, env_path=env_path,
                              prefer_installed=False)
    device_list = (require_devices(devices)
                   if resolved.needs_local_devices else [])
    return Session(host=host, secrets=secrets, devices=device_list,
                   base=base_path, target=resolved)


def setup(repo_url=kaggle_setup.DEFAULT_REPO_URL,
          repo_path=kaggle_setup.DEFAULT_REPO_DEST, *, arch="sm_75",
          devices=1, base=None, backend="cuda", precision="fp32", ref=None,
          env_path=None, secrets=None, target=None, package=None,
          prefer_installed=True) -> Session:
    """The one entry point a notebook calls: the installed package first.

    A host that can import ``nanochat_cpp`` prepares the session from the
    installed distribution and skips the clone and the Bazel build. Otherwise
    :func:`clone_and_build` runs. ``package`` is the ``nanochat_cpp`` module,
    and ``None`` looks it up; ``prefer_installed=False`` takes the repository
    path even when the package imports.
    """
    module = package
    if module is None and prefer_installed:
        module = kaggle_setup.installed_package()
    if module is None:
        return clone_and_build(repo_url=repo_url, repo_path=repo_path, ref=ref,
                               arch=arch, devices=devices, base=base,
                               backend=backend, precision=precision,
                               env_path=env_path, secrets=secrets,
                               target=target)
    return installed_session(module, arch=arch, devices=devices, base=base,
                             backend=backend, precision=precision,
                             env_path=env_path, secrets=secrets,
                             target=target)


def bootstrap(repo_url=kaggle_setup.DEFAULT_REPO_URL,
              repo_path=kaggle_setup.DEFAULT_REPO_DEST, *, arch="sm_75",
              devices=1, base=None, backend="cuda", precision="fp32",
              ref=None, env_path=None, secrets=None,
              target=None) -> Session:
    """Add the helper paths and prepare the session from the source tree.

    This is the entry of a notebook that kept a checkout: it adds
    ``python/`` and ``notebooks/`` to ``sys.path`` before the import, adds the
    model-fit helper scripts when the checkout holds them, and then runs the
    repository path of :func:`setup`, because a session that holds a checkout
    runs the host bootstrap. A notebook that installed the package calls
    :func:`setup` directly. ``target`` names the run host, from :mod:`target`.
    """
    helper = (Path(repo_path) / ".agents" / "skills" / "model-fit"
              / "scripts")
    if helper.is_dir() and str(helper) not in sys.path:
        sys.path.insert(0, str(helper))
    return setup(repo_url=repo_url, repo_path=repo_path, arch=arch,
                 devices=devices, base=base, backend=backend,
                 precision=precision, ref=ref, env_path=env_path,
                 secrets=secrets, target=target, prefer_installed=False)


def describe(prepared: "Session") -> "Session":
    """Print the preparation mode, the target, the library, and the devices."""
    print("target:", prepared.target.name)
    print("mode:", prepared.mode)
    print("library:", prepared.library)
    if prepared.git_revision:
        print("revision:", prepared.git_revision)
    print("secrets:", sorted(prepared.secrets))
    for device in prepared.devices:
        memory = float(getattr(device, "total_memory_bytes", 0))
        print(device, getattr(device, "name", ""), f"{memory / 2**30:.1f} GiB")
    return prepared
