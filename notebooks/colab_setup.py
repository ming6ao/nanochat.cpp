"""The Colab session setup: the source tree, the library, and the two stages.

A Colab session that runs the chain builds from the source tree and never from
the installed distribution (documents decision D8), so :func:`setup` calls
:func:`kaggle_setup.setup` with ``prefer_installed=False``. The installed
package would answer with its ``site-packages`` root, and that root carries no
``tools/nanochat`` and no stage executable: the wrapped entry of D10 and the
two children of the chain would then be missing, which is the failure the
default ``prefer_installed=True`` produces on a session that already installed
the wheel.

``setup`` resolves the library only, as its sibling does. The chain also needs
``sft_main`` and ``rl_worker``, so the session makes a second call,
:func:`stages`, which forwards to
:func:`nanochat_cpp._build.build_stages`. Two calls, not one, because the two
questions have two answers: where the tree is, and what the build produces.

:func:`stages` passes the Colab defaults to the builder instead of reading the
environment, because the build cell runs before the environment cell sets
``NANOCHAT_CPP_BACKEND``; an environment read there would compile the CPU
library.

:func:`probe` maps the card to an architecture name. Free Colab gives a T4, so
``sm_75``; a paid session can give a V100, ``sm_70``, or an A100, ``sm_80``.
The build is pinned to the probed value (decision D3), so a card that does not
match fails inside ``nanochat_init`` with a message about the architecture and
not about the job.

``kaggle_setup.py`` keeps its name, because ``notebooks/session.py`` and five
documents name it. This module is additive: it holds the ``/content`` defaults
of Colab and no setup logic of its own.
"""

from __future__ import annotations

import subprocess

import kaggle_setup

__all__ = [
    "CAPABILITY_QUERY",
    "COLAB_DEST",
    "COLAB_URL",
    "DEFAULT_BACKEND",
    "DEFAULT_PRECISION",
    "Host",
    "arch_from_capability",
    "probe",
    "setup",
    "stages",
]

#: The repository a Colab session clones, the same one Kaggle uses.
COLAB_URL = kaggle_setup.DEFAULT_REPO_URL

#: The Colab destination of the clone. ``/content`` is the working directory of
#: a Colab session, where ``/kaggle/working`` is Kaggle's.
COLAB_DEST = "/content/nanochat.cpp"

#: The backend of the pinned triple (decision D3): one T4, ``sm_75``, fp32.
DEFAULT_BACKEND = "cuda"
DEFAULT_PRECISION = "fp32"

#: The query that names the compute capability, such as ``7.5``.
CAPABILITY_QUERY = ("nvidia-smi", "--query-gpu=compute_cap",
                    "--format=csv,noheader")

#: The prepared session, as ``kaggle_setup.setup`` returns it.
Host = kaggle_setup.Host


def arch_from_capability(text: str) -> str:
    """Map a compute capability, ``7.5``, to an architecture, ``sm_75``.

    The pair ``(major, minor)`` is concatenated without a separator, which is
    how ``NANOCHAT_CUDA_ARCH`` and the ``--@rules_cuda//cuda:archs`` flag name
    an architecture. An empty value or a value that is not a pair of numbers
    raises ``ValueError``, so a surprising ``nvidia-smi`` answer stops here and
    not inside the build.
    """
    value = str(text).strip()
    if not value:
        raise ValueError("nvidia-smi reported no compute capability")
    major, _, minor = value.partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise ValueError(
            f"{value!r} is not a compute capability such as 7.5")
    return f"sm_{major}{minor}"


def probe(run=None) -> str:
    """The architecture of the card: ``sm_75`` for a T4, ``sm_70`` for a V100.

    ``run`` is the subprocess call, so a test drives the mapping without a
    card. The first card answers, because the triple of D3 is one card; a
    multi-card session still builds one architecture, which is what the two
    stage executables run.

    A session with no ``nvidia-smi`` raises ``RuntimeError`` naming the query:
    a notebook that cannot name its card must not fall back to a pinned
    architecture, because the build would then fail inside ``nanochat_init``
    with a message about the wrong thing.
    """
    run = subprocess.run if run is None else run
    command = list(CAPABILITY_QUERY)
    try:
        result = run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot read the card: {' '.join(command)} failed: {error}") \
            from error
    lines = [line.strip()
             for line in str(getattr(result, "stdout", "")).splitlines()
             if line.strip()]
    if not lines:
        raise RuntimeError(
            f"cannot read the card: {' '.join(command)} printed nothing")
    return arch_from_capability(lines[0])


def setup(backend: str = DEFAULT_BACKEND, precision: str = DEFAULT_PRECISION,
          arch: str | None = None, *, url: str = COLAB_URL,
          dest: str = COLAB_DEST, ref: str | None = None,
          cache=None, env_path=None,
          prefer_installed: bool = False) -> Host:
    """Prepare the Colab session from the source tree and return the host.

    The clone, ``tools/kaggle/bootstrap.sh``, the host contract, and the
    library resolution are :func:`kaggle_setup.setup`'s work; this call fixes
    the ``/content`` destination and the source-tree path. ``arch`` defaults to
    :func:`probe`, so the build matches the card the session actually has.

    ``prefer_installed`` is accepted so the call site reads the choice the
    plan writes, and ``True`` raises: the install root has no
    ``tools/nanochat`` and no stage executable, so it contradicts D8 and
    breaks D10. The session therefore builds from the tree, which is the only
    route until a binary wheel exists.
    """
    if prefer_installed:
        raise ValueError(
            "a Colab session builds from the source tree: the install root "
            "has no tools/nanochat and no stage executable (D8 and D10), so "
            "pass prefer_installed=False")
    return kaggle_setup.setup(
        url=url, dest=dest, ref=ref, backend=backend, precision=precision,
        arch=arch if arch else probe(), cache=cache, env_path=env_path,
        prefer_installed=False)


def stages(backend: str = DEFAULT_BACKEND,
           precision: str = DEFAULT_PRECISION, arch: str | None = None,
           cache=None, root=None) -> tuple:
    """Build the library and both stage executables, and return their paths.

    The return value is ``(library, sft_main, rl_worker)``, the tuple of
    :func:`nanochat_cpp._build.build_stages`. The options are explicit for the
    reason in the module docstring: this cell runs before the environment cell,
    so a read of ``NANOCHAT_CPP_BACKEND`` here would answer ``cpu`` and build
    the wrong library. ``arch`` defaults to :func:`probe`.

    The import of the package stays inside the call, as the sibling module's
    imports do, so reading this module costs nothing.
    """
    from nanochat_cpp import _build  # noqa: PLC0415 - keeps the import lazy

    options = _build.BuildOptions.resolve(
        backend=backend, precision=precision,
        arch=arch if arch else probe(), cache=cache)
    return _build.build_stages(options=options, root=root)
