"""Write the little-endian record containers used by offline fixtures."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Sequence

_DTYPE_CODES = {
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

    def __init__(self, path: Path, magic: bytes = b"NANOORC1") -> None:
        if len(magic) != 8:
            raise ValueError("fixture magic must contain exactly eight bytes")
        self._path = Path(path)
        self._magic = magic
        self._records: list[bytes] = []

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
        dtype = str(value)
        self._add_record(name, dtype, shape, payload)

    def add_int32(self, name: str, values: Sequence[int]) -> None:
        payload = struct.pack("<" + "i" * len(values), *values)
        self._add_record(name, "i4", (len(values),), payload)

    def add_uint8(self, name: str, payload: bytes) -> None:
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
            dtype_code, element_size = _DTYPE_CODES[dtype]
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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as handle:
            handle.write(self._magic)
            handle.write(struct.pack("<II", 1, len(self._records)))
            for record in self._records:
                handle.write(record)
