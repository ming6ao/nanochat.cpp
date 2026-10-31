"""The offline parity producers of D11: a fixture is a file, not an import.

A comparison with the PyTorch world runs in two halves
(``docs/dependency-layering-plan.md`` section 7.1)::

          producer (Z2, offline)              consumer (Z1, no torch)
    torch nanochat  -->  fixtures/*.bin  -->  tests/
    HF datasets     -->  packed rows     -->  src/sft_main

The producer runs in a virtual environment that holds the reference checkout
and the extras, writes the bytes, and records the environment it ran in
(section 7.2). The consumer reads the committed file with the standard library
only. This package is the producer half, and ``tools/nanochat parity produce
<kind>`` is its one command (section 7.3); no producer runs under ``tools/
nanochat test``, so the lock and the sandbox still cover every long job.

Kinds
-----
:data:`KINDS` maps a kind to the module that produces it, as an import path,
so importing this package imports no producer and needs no extra. One kind
exists today:

``train_parity``
    ``models/<name>/fixtures/train_parity.bin``, the training trajectory
    fixture of :mod:`nanochat_cpp_bridge.fixtures.train_parity`.

:data:`ALIASES` maps the shorter name a host's generator already uses
(``train``) onto the kind, so a caller does not have to know both spellings.

The SFT rung fixtures of section 7.4 (``sft_rows_fixture.bin``,
``sft_trajectory.bin``, ``sft_val.bin``) are deliberately absent: they must
land on the consumer side once stage S4 has landed, and a producer written
before the consumer exists would pin bytes nothing reads.

Usage
-----
::

    from nanochat_cpp_bridge import fixtures

    path = fixtures.produce("train_parity")          # the kind name
    path = fixtures.produce("train", out="/tmp/t.bin")  # the alias

A producer prints two lines a Model owner can stamp into a manifest
(:func:`produce` forwards them): the ``producer`` record of section 7.2 and the
``fixture`` record with its path, size, and SHA-256.

Extras: none at import. A producer's own module names the extras it needs.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ._provenance import environment, fixture_record, producer_record

__all__ = [
    "ALIASES",
    "KINDS",
    "canonical_kind",
    "environment",
    "fixture_record",
    "produce",
    "producer_module",
    "producer_record",
]

#: Every producer kind and the module that implements it, as an import path.
KINDS = {
    "train_parity": "nanochat_cpp_bridge.fixtures.train_parity",
}

#: The shorter names a host's generator uses, mapped onto a kind.
ALIASES = {
    "train": "train_parity",
}


def canonical_kind(kind: str) -> str:
    """The kind name for ``kind``, accepting the alias of :data:`ALIASES`."""
    name = ALIASES.get(kind, kind)
    if name not in KINDS:
        known = sorted(KINDS) + sorted(ALIASES)
        raise ValueError(
            f"unknown fixture kind {kind!r}; the producers are "
            + ", ".join(sorted(set(known))))
    return name


def producer_module(kind: str):
    """Import and return the producer module of ``kind``."""
    return importlib.import_module(KINDS[canonical_kind(kind)])


def produce(kind: str = "train_parity", **options: Any) -> Path:
    """Run one producer and return the path it wrote.

    ``options`` are the producer's own keyword arguments (``out``, ``model``,
    ``seed``, ...), so a host passes through whatever the command line parsed.
    The producer prints its ``producer`` and ``fixture`` records; nothing else
    here prints, so a caller can capture the run.
    """
    return Path(producer_module(kind).produce(**options))
