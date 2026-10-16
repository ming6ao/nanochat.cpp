"""The shared Kaggle session setup for the three notebooks.

Every notebook runs the setup before its own cells. The bodies live here, so
the code has one home and the tests can drive it. The module wraps
``notebooks/kaggle_setup.py``, which keeps the repository fetch, the
bootstrap, and the build. This module adds the Kaggle secrets, the cache root,
and the device check.

The module imports ``kaggle_secrets`` inside :func:`load_secrets` and
``nanochat_cpp`` inside :func:`require_devices`, so the tests import without
the shared library and without network access. See ``docs/checkpoint-format.md``
section 15.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import kaggle_setup

__all__ = [
    "DEFAULT_BASE",
    "SECRET_NAMES",
    "Session",
    "bootstrap",
    "clone_and_build",
    "describe",
    "load_secrets",
    "require_devices",
    "setup",
]

#: The secret names a notebook reads. Only ``HF_TOKEN`` is required. The other
#: names authenticate the ``pi`` agent in ``setup.ipynb``.
SECRET_NAMES = ("HF_TOKEN",)

#: The cache root a notebook uses when the caller names none.
DEFAULT_BASE = Path("/kaggle/working/nanochat-cache")


@dataclasses.dataclass(frozen=True)
class Session:
    """One prepared notebook session.

    ``host`` is the :class:`kaggle_setup.Host`. ``secrets`` holds the values
    the loader found. ``devices`` holds the device records the notebook needs.
    ``base`` is the cache root, also set as ``NANOCHAT_BASE_DIR``.
    """

    host: kaggle_setup.Host
    secrets: dict
    devices: list
    base: Path

    @property
    def repo(self) -> Path:
        """The repository path."""
        return Path(self.host.repo)

    @property
    def library(self) -> Path:
        """The path of the built shared library."""
        return Path(self.host.library)

    @property
    def build_key(self) -> str:
        """The build key of the shared library."""
        return self.host.build_key

    @property
    def revision(self) -> str:
        """The short Git revision of the session repository."""
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


def clone_and_build(repo_url=kaggle_setup.DEFAULT_REPO_URL,
                    repo_path=kaggle_setup.DEFAULT_REPO_DEST, ref=None,
                    arch="sm_75", devices=1, base=None, backend="cuda",
                    precision="fp32", env_path=None, secrets=None) -> Session:
    """Fetch, bootstrap, build, and check the session.

    ``base`` becomes ``NANOCHAT_BASE_DIR``. ``devices`` is the number of cards
    the notebook needs.
    """
    if secrets is None:
        secrets = load_secrets()
    base_path = Path(base) if base is not None else DEFAULT_BASE
    os.environ["NANOCHAT_BASE_DIR"] = str(base_path)
    host = kaggle_setup.setup(url=repo_url, dest=repo_path, ref=ref,
                              backend=backend, precision=precision,
                              arch=arch, env_path=env_path)
    device_list = require_devices(devices)
    return Session(host=host, secrets=secrets, devices=device_list,
                   base=base_path)


def setup(repo_url=kaggle_setup.DEFAULT_REPO_URL,
          repo_path=kaggle_setup.DEFAULT_REPO_DEST, *, arch="sm_75",
          devices=1, base=None, backend="cuda", precision="fp32", ref=None,
          env_path=None, secrets=None) -> Session:
    """The one entry point a notebook calls."""
    return clone_and_build(repo_url=repo_url, repo_path=repo_path, ref=ref,
                           arch=arch, devices=devices, base=base,
                           backend=backend, precision=precision,
                           env_path=env_path, secrets=secrets)


def bootstrap(repo_url=kaggle_setup.DEFAULT_REPO_URL,
              repo_path=kaggle_setup.DEFAULT_REPO_DEST, *, arch="sm_75",
              devices=1, base=None, backend="cuda", precision="fp32",
              ref=None, env_path=None, secrets=None) -> Session:
    """Add the helper paths and run the session setup.

    The notebook clones the repository and adds ``python/`` and
    ``notebooks/`` to ``sys.path`` before it imports this module. This call
    adds the model-fit helper scripts, then runs :func:`setup`.
    """
    helper = (Path(repo_path) / ".agents" / "skills" / "model-fit"
              / "scripts")
    if helper.is_dir() and str(helper) not in sys.path:
        sys.path.insert(0, str(helper))
    return setup(repo_url=repo_url, repo_path=repo_path, arch=arch,
                 devices=devices, base=base, backend=backend,
                 precision=precision, ref=ref, env_path=env_path,
                 secrets=secrets)


def describe(prepared: "Session") -> "Session":
    """Print the library, the secret names, and the devices of a session."""
    print("library:", prepared.library)
    print("secrets:", sorted(prepared.secrets))
    for device in prepared.devices:
        memory = float(getattr(device, "total_memory_bytes", 0))
        print(device, getattr(device, "name", ""), f"{memory / 2**30:.1f} GiB")
    return prepared
