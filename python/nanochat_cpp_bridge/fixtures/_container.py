"""The little-endian record container the parity fixtures use (pinned copy).

Every model fixture is the same container (``NANOORC1``): an eight-byte magic,
a ``u32`` version, a ``u32`` record count, and then a flat list of
self-describing records::

    header   magic[8]    = b"NANOORC1"
             version u32 = 1
             count   u32 = number of records
    record   name_len u16
             name     name_len bytes, UTF-8, no terminator
             dtype    u8
             ndim     u8
             shape    ndim * u64, most significant dimension first
             payload  numel * element_size bytes, C order

Dtype codes are shared with the C++ reader (``tests/oracle_fixture.h``):
``0`` fp32, ``1`` int32, ``2`` fp64, ``3`` int64, ``4`` uint8.

The layout is single-sourced in ``tools/fixture_format.py``, which a host
reads. A package must not import a host (rule 4 of
``docs/dependency-layering-plan.md`` section 4), so this module is a pinned
copy of that writer: the same header, the same record framing, and the same
dtype table, so a producer here writes the bytes a host writes. A change to
the container belongs in both files in the same commit; the fixture hashes of
``models/<name>/fixtures/manifest.json`` fail the isolation gate when they
drift apart.

Extras: none. The writer takes NumPy arrays but never imports NumPy.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Sequence

__all__ = ["DTYPE_CODES", "MAGIC", "VERSION", "FixtureWriter"]

#: The container magic of the model fixtures.
MAGIC = b"NANOORC1"

#: The container version this writer emits.
VERSION = 1

#: The dtype code and element size of every record dtype.
DTYPE_CODES = {
    "float32": (0, 4),
    "int32": (1, 4),
    "float64": (2, 8),
    "int64": (3, 8),
    "uint8": (4, 1),
    "f4": (0, 4),
    "i4": (1, 4),
    "f8": (2, 8),
    "i8": (3, 8),
    "u1": (4, 1),
}


class FixtureWriter:
    """Write typed records with an eight-byte magic and a version header."""

    def __init__(self, path: str | Path, magic: bytes = MAGIC) -> None:
        if len(magic) != 8:
            raise ValueError("fixture magic must contain exactly eight bytes")
        self._path = Path(path)
        self._magic = magic
        self._records: list[bytes] = []

    @property
    def path(self) -> Path:
        """The file this writer fills."""
        return self._path

    def add(
        self,
        name: str,
        value: Any,
        shape: Sequence[int] | None = None,
        payload: bytes | None = None,
    ) -> None:
        """Add an array or a payload with its dtype code and shape."""
        if shape is None and payload is None:
            self._add_array(name, value)
            return
        if shape is None or payload is None:
            raise ValueError("shape and payload must be provided together")
        self._add_record(name, str(value), shape, payload)

    def add_int32(self, name: str, values: Sequence[int]) -> None:
        """Add one int32 record from a sequence of integers."""
        payload = struct.pack("<" + "i" * len(values), *values)
        self._add_record(name, "i4", (len(values),), payload)

    def add_uint8(self, name: str, payload: bytes) -> None:
        """Add one uint8 record from raw bytes."""
        self._add_record(name, "u1", (len(payload),), bytes(payload))

    def _add_array(self, name: str, array: Any) -> None:
        little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
        dtype = little_endian.dtype.name
        self._add_record(name, dtype, little_endian.shape,
                         little_endian.tobytes(order="C"))

    def _add_record(
        self,
        name: str,
        dtype: str,
        shape: Sequence[int],
        payload: bytes,
    ) -> None:
        try:
            dtype_code, element_size = DTYPE_CODES[dtype]
        except KeyError as error:
            raise TypeError(f"unsupported fixture dtype {dtype!r}") from error

        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"record name too long: {name!r}")
        if len(shape) > 0xFF:
            raise ValueError(f"record rank too high: {name!r}")
        if any(extent < 0 for extent in shape):
            raise ValueError(f"record shape has a negative extent: {name!r}")
        expected_size = element_size
        for extent in shape:
            expected_size *= int(extent)
        if len(payload) != expected_size:
            raise ValueError(
                f"record {name!r} has {len(payload)} payload bytes; "
                f"expected {expected_size}")

        parts = [
            struct.pack("<H", len(name_bytes)),
            name_bytes,
            struct.pack("<BB", dtype_code, len(shape)),
        ]
        parts.extend(struct.pack("<Q", int(extent)) for extent in shape)
        parts.append(payload)
        self._records.append(b"".join(parts))

    def write(self) -> None:
        """Write the header and every record to the output path."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as handle:
            handle.write(self._magic)
            handle.write(struct.pack("<II", VERSION, len(self._records)))
            for record in self._records:
                handle.write(record)
