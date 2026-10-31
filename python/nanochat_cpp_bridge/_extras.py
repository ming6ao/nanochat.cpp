"""The extras map of Z2: name the package, name the install line.

Every module of this package imports its heavy package inside the function
that needs it, so ``import nanochat_cpp_bridge`` costs nothing and needs no
extra (sections 4 and 6.1 of ``docs/dependency-layering-plan.md``). When the
import fails, the caller gets :class:`MissingExtra` naming the extra of
section 6.2 instead of a bare ``ModuleNotFoundError``::

    pip install 'nanochat-cpp[reference]'

This module imports the standard library only, so it can be imported from any
of the Z2 modules without a cost.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

__all__ = [
    "EXTRAS",
    "INSTALL_LINE",
    "MissingExtra",
    "extra_for",
    "import_package",
    "missing",
    "reference_on_path",
]

#: The extra of section 6.2 that supplies each heavy package.
EXTRAS = {
    "torch": "reference",
    "numpy": "reference",
    "tiktoken": "reference",
    "nanochat": "reference,nanochat",
    "pyarrow": "hfdata",
    "filelock": "hfdata",
    "datasets": "hfdata",
    "tokenizers": "hfdata",
    "huggingface_hub": "hub",
    "safetensors": "hub",
    "modal": "modal",
}

#: The install shape of section 6.2, with ``{extra}`` for the extra name.
INSTALL_LINE = "pip install 'nanochat-cpp[{extra}]'"


class MissingExtra(RuntimeError):
    """A heavy package is absent; the message names the extra to install."""


def extra_for(package: str) -> str:
    """The extra that supplies ``package``; a dotted name uses its root."""
    root = package.split(".", 1)[0]
    return EXTRAS.get(root, root)


def missing(package: str, detail: str | None = None) -> MissingExtra:
    """The error a caller raises when ``package`` is not installed.

    Call sites keep the cause: ``raise _extras.missing("torch") from error``.
    ``detail`` carries the underlying import error when the package is present
    but one of its own dependencies is not, so the message never sends a reader
    to install something they already have.
    """
    extra = extra_for(package)
    message = (
        f"{package} cannot be imported; the extra that supplies it is "
        f"'{extra}': " + INSTALL_LINE.format(extra=extra))
    if detail:
        message += f" (the import failed with: {detail})"
    return MissingExtra(message)


def import_package(package: str) -> Any:
    """Import a heavy package, or raise :class:`MissingExtra`.

    The import happens in the calling function, so a module that uses this
    helper still imports with no extra installed.
    """
    try:
        return importlib.import_module(package)
    except ImportError as error:
        raise missing(package, str(error)) from error


def reference_on_path() -> Path:
    """Put the reference PyTorch checkout on ``sys.path``.

    The core owns the search order and the ``NANOCHAT_REPO`` override
    (``nanochat_cpp.reference``), so Z2 asks the core instead of repeating the
    candidate paths. A missing checkout becomes a :class:`MissingExtra` that
    names the ``reference,nanochat`` extras line.
    """
    from nanochat_cpp import reference  # noqa: PLC0415 - Z1, imported on use

    try:
        return reference.ensure_reference_on_path()
    except SystemExit as error:
        raise MissingExtra(
            "the reference PyTorch nanochat checkout was not found; the extra "
            "that supplies it is 'reference,nanochat': "
            + INSTALL_LINE.format(extra="reference,nanochat")
            + " (or point NANOCHAT_REPO at the checkout)") from error
