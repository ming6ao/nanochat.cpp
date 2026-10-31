"""The registration seam: the two callables the core cannot own.

Rule 1 of ``docs/dependency-layering-plan.md`` section 4 says
``python/nanochat_cpp/**`` imports only the standard library. Two jobs still
need a third-party package, so the core reaches each one by being *handed* a
callable at run time instead of importing it (rule 5 of section 8, gate G5):

``hub_loader``
    ``loader(repo_id, subset="default", split="train")`` returns a dataset with
    ``__len__``, ``__getitem__`` and ``shuffle(seed=...)``. It backs the
    task datasets of :mod:`nanochat_cpp.tasks` (ARC, MMLU, GSM8K, HumanEval)
    and the SmolTalk half of :mod:`nanochat_cpp.sft_data`, which need
    ``pyarrow`` and ``filelock``.
``code_executor``
    ``executor(program)`` runs a generated program and returns a result with a
    ``success`` attribute. It backs the HumanEval criterion, which needs the
    reference guards (a fresh interpreter, rlimits, a scrubbed environment, a
    timeout) rather than a reimplementation.

A host registers the callable once, before it builds a task::

    from nanochat_cpp import seam

    seam.register_hub_loader(my_loader)         # or seam.use(hub_loader=...)
    seam.register_code_executor(my_executor)

A test registers a fake one, so the task classes stay hermetic: no network, no
wheel, no GPU. An unregistered adapter raises :class:`MissingAdapter`, which
names the extra of section 6.2 to install and the call that registers one,
instead of failing on an ``ImportError`` deep inside a task. Nothing here
imports a heavy package, so this module is standard-library only like the rest
of the core.

The registry is process-wide, which matches the rest of the core: the session
binds a library and a backend for the process. :func:`use` is the scoped form
for a caller that must not leak a fake adapter into a later test.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any

__all__ = [
    "ADAPTERS",
    "EXTRAS",
    "MissingAdapter",
    "code_executor",
    "describe",
    "install_line",
    "installed",
    "register",
    "register_code_executor",
    "register_hub_loader",
    "registered",
    "registered_code_executor",
    "registered_hub_loader",
    "require",
    "reset",
    "unregister",
    "use",
]

#: The adapter names a caller can register, in the order a host supplies them.
ADAPTERS = ("hub_loader", "code_executor")

#: The extra of section 6.2 that supplies each adapter's package.
EXTRAS = {
    "hub_loader": "hfdata",
    "code_executor": "reference,nanochat",
}

#: What each adapter does, for the error message of :class:`MissingAdapter`.
_ROLES = {
    "hub_loader": "hub dataset loader",
    "code_executor": "code executor",
}

#: What each adapter takes and returns, for the same message.
_CONTRACTS = {
    "hub_loader": "loader(repo_id, subset=\"default\", split=\"train\") -> "
                  "a dataset with __len__, __getitem__ and shuffle(seed=...)",
    "code_executor": "executor(program) -> a result with a success attribute",
}

#: The install shape of section 6.2, with ``{extra}`` for the extra name.
_INSTALL_LINE = "pip install 'nanochat-cpp[{extra}]'"

_registry: dict[str, Callable[..., Any]] = {}


class MissingAdapter(RuntimeError):
    """A core call needs an adapter that no caller registered.

    The message names the adapter, the extra that supplies the package behind
    it (section 6.2), and the call that registers one, so the reader is not
    sent to an ``ImportError`` in a module the core is not allowed to import.
    """

    def __init__(self, adapter: str) -> None:
        self.adapter = adapter
        self.extra = EXTRAS.get(adapter, adapter)
        message = (
            f"no {_ROLES.get(adapter, adapter)} is registered; the core "
            f"imports only the standard library, so the caller hands the "
            f"adapter in. Install the package with "
            f"{install_line(adapter)}, then register one with "
            f"nanochat_cpp.seam.register_{adapter}(...) or pass {adapter}=... "
            f"to the class that needs it. The contract is "
            f"{_CONTRACTS.get(adapter, adapter)}.")
        super().__init__(message)


def install_line(adapter: str) -> str:
    """The ``pip install`` command that supplies ``adapter``'s package."""
    return _INSTALL_LINE.format(extra=EXTRAS.get(adapter, adapter))


def register(adapter: str, implementation: Callable[..., Any]) -> None:
    """Register ``implementation`` as ``adapter``, replacing any earlier one.

    ``adapter`` is one of :data:`ADAPTERS`; a caller supplying something else
    is a programming error, so it raises ``ValueError`` rather than being
    stored where nothing reads it.
    """
    if adapter not in ADAPTERS:
        raise ValueError(
            f"unknown adapter {adapter!r}; the seam carries "
            + ", ".join(ADAPTERS))
    if not callable(implementation):
        raise TypeError(
            f"the {adapter} adapter must be callable, got "
            f"{type(implementation).__name__}")
    _registry[adapter] = implementation


def unregister(adapter: str) -> None:
    """Forget one adapter. A later :func:`require` raises again."""
    _registry.pop(adapter, None)


def registered(adapter: str) -> Callable[..., Any] | None:
    """The registered ``adapter``, or None when nothing is registered."""
    return _registry.get(adapter)


def installed() -> tuple[str, ...]:
    """The registered adapters, in :data:`ADAPTERS` order."""
    return tuple(name for name in ADAPTERS if name in _registry)


def require(adapter: str) -> Callable[..., Any]:
    """The registered ``adapter``, or raise :class:`MissingAdapter`."""
    implementation = _registry.get(adapter)
    if implementation is None:
        raise MissingAdapter(adapter)
    return implementation


def reset() -> None:
    """Forget every adapter. Use it in a test that registers a fake one."""
    _registry.clear()


@contextlib.contextmanager
def use(**adapters: Callable[..., Any]) -> Iterator[None]:
    """Register ``adapters`` for the duration of the block, then restore.

    The previous registrations come back exactly as they were, so a test that
    hands in a fake loader cannot leak it into the next test in the process::

        with seam.use(hub_loader=fake_loader):
            task = chat.build_task("GSM8K")
    """
    previous = dict(_registry)
    try:
        for adapter, implementation in adapters.items():
            register(adapter, implementation)
        yield
    finally:
        _registry.clear()
        _registry.update(previous)


# -- The named adapters -----------------------------------------------------


def register_hub_loader(loader: Callable[..., Any]) -> None:
    """Register the hub dataset loader the task classes call."""
    register("hub_loader", loader)


def registered_hub_loader() -> Callable[..., Any] | None:
    """The hub dataset loader, or None when no caller registered one."""
    return registered("hub_loader")


def hub_loader() -> Callable[..., Any]:
    """The registered hub dataset loader, or raise :class:`MissingAdapter`."""
    return require("hub_loader")


def register_code_executor(executor: Callable[..., Any]) -> None:
    """Register the code executor the HumanEval criterion calls."""
    register("code_executor", executor)


def registered_code_executor() -> Callable[..., Any] | None:
    """The code executor, or None when no caller registered one."""
    return registered("code_executor")


def code_executor() -> Callable[..., Any]:
    """The registered code executor, or raise :class:`MissingAdapter`."""
    return require("code_executor")


def describe() -> Mapping[str, Any]:
    """The seam as one mapping, for a report or a dry run."""
    return {name: ("registered" if name in _registry else "absent")
            for name in ADAPTERS}
