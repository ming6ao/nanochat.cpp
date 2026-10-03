"""Evaluation fixture wire format for ``ScoreBatch`` (docs/eval.md section 3.2).

This module is the single source of truth for the byte layout of the
evaluation fixtures that the Python bridge writes, that the C++ ``score_main``
binary reads and writes, and that the committed parity dumper mirrors. Every
import here is stdlib-only, so the bridge self-test can round-trip a fixture
without importing torch, numpy, or the reference package.

The model is a causal language model. The logits at *prediction position*
``p`` predict the token at ``p + 1``. A fixture describes a batch of ``N``
token sequences padded to a common width ``T``; for each sequence it carries a
candidate span and an optional focus set. The results are the per-position
negative log-likelihood, the per-position argmax, and the focused logits.

Container
---------
Everything is little-endian. The file is a 16-byte header followed by a flat
list of self-describing records::

    header   magic[8]     = b"NANOEVL1"
             version u32  = 1
             count   u32  = number of records
    record   name_len u16
             name     name_len bytes, UTF-8, no terminator
             dtype    u8
             ndim     u8
             shape    ndim * u64, most significant dimension first
             payload  numel * elem_size bytes, C order

``numel`` is the product of ``shape`` (a scalar record has ``ndim == 0`` and a
one-element payload). ``elem_size`` is ``4`` for fp32/int32, ``8`` for
fp64/int64, and ``1`` for uint8. The payload has no length field: it is derived
from the shape and dtype. Trailing bytes after the last record are an error.

Dtype codes (identical to ``tests/oracle_fixture.h`` so one C++ parser can read
either container):

    ===  ======  ==========
    code name    elem_size
    ===  ======  ==========
    0    fp32    4
    1    int32   4
    2    fp64    8
    3    int64   8
    4    uint8   1
    ===  ======  ==========

Semantic schema
---------------
A fixture carries the input *cases*, the *results*, or both. Records are keyed
by name, so a reader does not depend on the order, but ``serialize`` writes them
in the order below: ``config/`` first, then ``input/``, then ``result/``.

``config`` (always present)::

    config/version  int32 [1]   = 1
    config/batch    int32 [1]   = N
    config/seq      int32 [1]   = T
    config/pad_id   int32 [1]   padding token id used in input/tokens

``input`` (present only when the fixture carries cases)::

    input/tokens           int32 [N, T]  real ids followed by pad_id
    input/lengths          int32 [N]     number of valid ids per row
    input/spans            int32 [N, 2]  candidate [start, end) in token space
    input/focus_positions  int32 [N]     prediction position, or -1 for none
    input/focus_offsets    int32 [N+1]   ragged offsets into input/focus_ids
    input/focus_ids        int32 [K]     concatenated focus token ids

``result`` (present only when the fixture carries results)::

    result/nll             fp32  [N, T]  per prediction position
    result/argmax          int32 [N, T]  per prediction position
    result/focus_offsets   int32 [N+1]   ragged offsets into result/focus_logits
    result/focus_logits    fp32  [K]     aligned with input/focus_ids

Row semantics, for row ``i`` with ``L = input/lengths[i]`` and padded width
``T``:

* ``input/tokens[i]`` holds the ``L`` real ids followed by ``T - L`` copies of
  ``config/pad_id``.
* A prediction position ``p`` is valid when ``p < L - 1``; positions
  ``L - 1 <= p < T`` (the row's final position and all padding) are masked.
* ``result/nll[i][p]`` is the negative log-likelihood in nats of the target
  ``input/tokens[i][p+1]``; masked positions carry ``MASKED_NLL`` (``0.0``).
* ``result/argmax[i][p]`` is the argmax token id of the raw, pre-softcap logits
  over ``[0, vocab_size)``; masked positions carry ``IGNORE_INDEX`` (``-1``).
* ``input/spans[i] = [start, end)`` is the *candidate span* in token-index
  space, exactly as the CORE reference computes it
  (``nanochat/core_eval.py``): the scored prediction positions are
  ``[start - 1, end - 1)``. A non-empty span satisfies
  ``1 <= start < end <= L``; an empty span satisfies ``start == end``.
* ``input/focus_positions[i]`` is a prediction position ``p`` in
  ``[0, L)``, or ``-1``. ``input/focus_ids`` is the candidate set of token ids
  at that position (the answer letters for categorical chat).
  ``result/focus_logits[i]`` holds the raw logits at ``p`` for those ids, in
  request order, so ``len(result/focus_logits[i]) == len(input/focus_ids[i])``.
  A sequence with ``focus_positions[i] == -1`` has no focus ids and no focused
  logits. Focus is allowed at ``p == L - 1``, where the loss is masked but the
  logits are still defined (they predict the token after the row).

The module exposes a generic :func:`serialize`/:func:`parse` over named
records plus the semantic :func:`write_fixture`/:func:`read_fixture` and the
cases-only and results-only helpers. Every writer validates before it touches
disk and replaces the target atomically.
"""

from __future__ import annotations

import dataclasses
import os
import struct
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path

#: Container magic (eight bytes, version encoded in the last character).
MAGIC = b"NANOEVL1"

#: Container version written in the header.
VERSION = 1

#: Semantic schema version in ``config/version``.
FORMAT_VERSION = 1

#: Maximum record rank (matches ``tests/oracle_fixture.h``).
MAX_RANK = 16

#: ``argmax`` value at a masked prediction position.
IGNORE_INDEX = -1

#: ``nll`` value at a masked prediction position.
MASKED_NLL = 0.0

#: Dtype codes, identical to ``nanochat::oracle::DType``.
DTYPE_FP32 = 0
DTYPE_INT32 = 1
DTYPE_FP64 = 2
DTYPE_INT64 = 3
DTYPE_UINT8 = 4

DTYPE_NAMES = {
    DTYPE_FP32: "fp32",
    DTYPE_INT32: "int32",
    DTYPE_FP64: "fp64",
    DTYPE_INT64: "int64",
    DTYPE_UINT8: "uint8",
}
DTYPE_SIZES = {
    DTYPE_FP32: 4,
    DTYPE_INT32: 4,
    DTYPE_FP64: 8,
    DTYPE_INT64: 8,
    DTYPE_UINT8: 1,
}


class EvalFixtureError(Exception):
    """An evaluation fixture is malformed, inconsistent, or unsupported."""


# ---------------------------------------------------------------------------
# Generic record container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Record:
    """One named, self-describing tensor in the container."""

    name: str
    dtype: int
    shape: tuple[int, ...]
    data: bytes

    @property
    def dtype_name(self) -> str:
        try:
            return DTYPE_NAMES[self.dtype]
        except KeyError as error:
            raise EvalFixtureError(f"unknown dtype code: {self.dtype!r}") from error

    @property
    def element_size(self) -> int:
        try:
            return DTYPE_SIZES[self.dtype]
        except KeyError as error:
            raise EvalFixtureError(f"unknown dtype code: {self.dtype!r}") from error

    def numel(self) -> int:
        total = 1
        for extent in self.shape:
            total *= int(extent)
        return total

    def expected_bytes(self) -> int:
        return self.numel() * self.element_size


def _pack_record(record: Record) -> bytes:
    name_bytes = record.name.encode("utf-8")
    if not name_bytes:
        raise EvalFixtureError("record name is empty")
    if len(name_bytes) > 0xFFFF:
        raise EvalFixtureError(f"record name is too long: {record.name!r}")
    if record.dtype not in DTYPE_NAMES:
        raise EvalFixtureError(
            f"unknown dtype code for {record.name!r}: {record.dtype!r}")
    if len(record.shape) > MAX_RANK:
        raise EvalFixtureError(f"record rank is too large: {record.name!r}")
    for extent in record.shape:
        if int(extent) < 0:
            raise EvalFixtureError(
                f"negative dimension in {record.name!r}: {extent}")
    if len(record.data) != record.expected_bytes():
        raise EvalFixtureError(
            f"payload size mismatch for {record.name!r}: "
            f"{len(record.data)} != {record.expected_bytes()}")
    parts = [
        struct.pack("<H", len(name_bytes)),
        name_bytes,
        struct.pack("<BB", record.dtype, len(record.shape)),
    ]
    for extent in record.shape:
        parts.append(struct.pack("<Q", int(extent)))
    parts.append(record.data)
    return b"".join(parts)


def serialize(records: Sequence[Record]) -> bytes:
    """Serialise ``records`` into the ``NANOEVL1`` byte layout."""
    header = MAGIC + struct.pack("<II", VERSION, len(records))
    return header + b"".join(_pack_record(record) for record in records)


def _need(data: bytes, offset: int, count: int, what: str) -> None:
    if count < 0 or offset + count > len(data):
        raise EvalFixtureError(f"truncated fixture while reading {what}")


def parse(data: bytes) -> list[Record]:
    """Parse a ``NANOEVL1`` byte string into records.

    The inverse of :func:`serialize`. Rejects a bad magic, an unknown version,
    unknown dtype codes, duplicate names, truncation, and trailing bytes.
    """
    if len(data) < 16 or data[:8] != MAGIC:
        raise EvalFixtureError("bad fixture magic (expected NANOEVL1)")
    version, count = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise EvalFixtureError(f"unsupported fixture version: {version}")

    offset = 16
    records: list[Record] = []
    seen: set[str] = set()
    for _ in range(count):
        _need(data, offset, 2, "record name length")
        (name_len,) = struct.unpack_from("<H", data, offset)
        offset += 2
        _need(data, offset, name_len, "record name")
        try:
            name = data[offset:offset + name_len].decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvalFixtureError("record name is not valid UTF-8") from error
        offset += name_len
        if not name:
            raise EvalFixtureError("record name is empty")
        if name in seen:
            raise EvalFixtureError(f"duplicate record name: {name!r}")
        seen.add(name)

        _need(data, offset, 2, "record header")
        dtype, ndim = struct.unpack_from("<BB", data, offset)
        offset += 2
        if dtype not in DTYPE_NAMES:
            raise EvalFixtureError(
                f"unknown dtype code for {name!r}: {dtype!r}")
        if ndim > MAX_RANK:
            raise EvalFixtureError(f"record rank is too large: {name!r}")

        if ndim:
            _need(data, offset, 8 * ndim, "record shape")
            shape = struct.unpack_from(f"<{ndim}Q", data, offset)
            offset += 8 * ndim
        else:
            shape = ()

        numel = 1
        for extent in shape:
            numel *= int(extent)
        payload_bytes = numel * DTYPE_SIZES[dtype]
        _need(data, offset, payload_bytes, f"record payload of {name!r}")
        payload = data[offset:offset + payload_bytes]
        offset += payload_bytes
        records.append(Record(name, dtype, tuple(int(d) for d in shape), payload))

    if offset != len(data):
        raise EvalFixtureError(
            f"trailing bytes after {count} records: {len(data) - offset}")
    return records


def write(path: str | os.PathLike[str], records: Sequence[Record]) -> Path:
    """Write ``records`` to ``path`` atomically and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(serialize(records))
    os.replace(temporary, target)
    return target


def read(path: str | os.PathLike[str]) -> list[Record]:
    """Read and parse a ``NANOEVL1`` file."""
    return parse(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Typed payload helpers
# ---------------------------------------------------------------------------


def _signed_32(values: Sequence[int]) -> bytes:
    buf = array("i", (int(value) for value in values))
    if buf.itemsize != 4:
        raise EvalFixtureError("this host's C int is not 32 bits")
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _float_32(values: Sequence[float]) -> bytes:
    buf = array("f", (float(value) for value in values))
    if buf.itemsize != 4:
        raise EvalFixtureError("this host's C float is not 32 bits")
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _read_signed_32(record: Record) -> list[int]:
    if record.dtype != DTYPE_INT32:
        raise EvalFixtureError(f"{record.name!r} is not int32")
    buf = array("i")
    buf.frombytes(record.data)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tolist()


def _read_float_32(record: Record) -> list[float]:
    if record.dtype != DTYPE_FP32:
        raise EvalFixtureError(f"{record.name!r} is not fp32")
    buf = array("f")
    buf.frombytes(record.data)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tolist()


# ---------------------------------------------------------------------------
# Semantic dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvalCase:
    """One padded-sequence input: tokens, a candidate span, and optional focus.

    ``tokens`` holds only the real token ids (no padding); ``length`` is
    ``len(tokens)``. ``start``/``end`` are the candidate span in token-index
    space; the scored prediction positions are ``[start - 1, end - 1)``.
    ``focus_position`` is a prediction position ``p`` in ``[0, len(tokens))``
    whose logits are requested for ``focus_ids``, or ``-1`` for no focus.
    """

    tokens: tuple[int, ...]
    start: int
    end: int
    focus_position: int = IGNORE_INDEX
    focus_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(int(t) for t in self.tokens))
        object.__setattr__(self,
                           "focus_ids", tuple(int(t) for t in self.focus_ids))
        length = len(self.tokens)
        if not (0 <= self.start <= self.end <= length):
            raise EvalFixtureError(
                f"candidate span [{self.start}, {self.end}) is outside "
                f"[0, {length}]")
        if self.start != self.end and self.start < 1:
            raise EvalFixtureError(
                "a non-empty candidate span must start at token index >= 1 "
                "(the first token has no context to be predicted from)")
        if self.focus_position < 0:
            if self.focus_ids:
                raise EvalFixtureError(
                    "focus_ids is non-empty but focus_position is disabled")
        elif not self.focus_ids:
            raise EvalFixtureError(
                "focus_position is set but focus_ids is empty")
        elif not (0 <= self.focus_position < length):
            raise EvalFixtureError(
                f"focus_position {self.focus_position} is outside [0, {length})")

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def has_focus(self) -> bool:
        return self.focus_position >= 0

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Per-position losses, argmax, and focused logits for one sequence.

    ``nll`` and ``argmax`` each have ``seq`` entries (the padded width);
    masked positions carry :data:`MASKED_NLL` and :data:`IGNORE_INDEX`.
    ``focus_logits`` is aligned with the matching case's ``focus_ids`` and is
    empty when the case has no focus.
    """

    nll: tuple[float, ...]
    argmax: tuple[int, ...]
    focus_logits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nll", tuple(float(x) for x in self.nll))
        object.__setattr__(self, "argmax", tuple(int(x) for x in self.argmax))
        object.__setattr__(self,
                           "focus_logits",
                           tuple(float(x) for x in self.focus_logits))
        if len(self.nll) != len(self.argmax):
            raise EvalFixtureError(
                f"result nll has {len(self.nll)} entries but argmax has "
                f"{len(self.argmax)}")


@dataclasses.dataclass
class Fixture:
    """A parsed fixture: its padded width, cases, and results."""

    seq: int
    batch: int
    pad_id: int = 0
    cases: list[EvalCase] | None = None
    results: list[EvalResult] | None = None

    @property
    def has_cases(self) -> bool:
        return self.cases is not None

    @property
    def has_results(self) -> bool:
        return self.results is not None


# ---------------------------------------------------------------------------
# Records <-> dataclasses
# ---------------------------------------------------------------------------


def _i32_record(name: str, shape: Sequence[int],
                values: Sequence[int]) -> Record:
    return Record(name, DTYPE_INT32, tuple(int(d) for d in shape),
                  _signed_32(values))


def _fp32_record(name: str, shape: Sequence[int],
                 values: Sequence[float]) -> Record:
    return Record(name, DTYPE_FP32, tuple(int(d) for d in shape),
                  _float_32(values))


def _config_records(batch: int, seq: int, pad_id: int) -> list[Record]:
    return [
        _i32_record("config/version", (1,), (FORMAT_VERSION,)),
        _i32_record("config/batch", (1,), (batch,)),
        _i32_record("config/seq", (1,), (seq,)),
        _i32_record("config/pad_id", (1,), (pad_id,)),
    ]


def _cases_to_records(cases: Sequence[EvalCase], seq: int,
                      pad_id: int) -> list[Record]:
    if not cases:
        raise EvalFixtureError("a case fixture needs at least one row")
    batch = len(cases)
    for case in cases:
        if case.length > seq:
            raise EvalFixtureError(
                f"a case has {case.length} tokens, wider than seq {seq}")

    lengths: list[int] = []
    tokens: list[int] = []
    spans: list[int] = []
    focus_positions: list[int] = []
    focus_offsets: list[int] = [0]
    focus_ids: list[int] = []
    for case in cases:
        lengths.append(case.length)
        tokens.extend(case.tokens)
        tokens.extend([pad_id] * (seq - case.length))
        spans.extend((case.start, case.end))
        focus_positions.append(case.focus_position)
        focus_ids.extend(case.focus_ids)
        focus_offsets.append(len(focus_ids))

    return [
        _i32_record("input/tokens", (batch, seq), tokens),
        _i32_record("input/lengths", (batch,), lengths),
        _i32_record("input/spans", (batch, 2), spans),
        _i32_record("input/focus_positions", (batch,), focus_positions),
        _i32_record("input/focus_offsets", (batch + 1,), focus_offsets),
        _i32_record("input/focus_ids", (len(focus_ids),), focus_ids),
    ]


def _results_to_records(results: Sequence[EvalResult], seq: int) -> list[Record]:
    if not results:
        raise EvalFixtureError("a result fixture needs at least one row")
    batch = len(results)
    for result in results:
        if len(result.nll) != seq:
            raise EvalFixtureError(
                f"a result has {len(result.nll)} positions, expected seq {seq}")

    nll: list[float] = []
    argmax: list[int] = []
    focus_offsets: list[int] = [0]
    focus_logits: list[float] = []
    for result in results:
        nll.extend(result.nll)
        argmax.extend(result.argmax)
        focus_logits.extend(result.focus_logits)
        focus_offsets.append(len(focus_logits))

    return [
        _fp32_record("result/nll", (batch, seq), nll),
        _i32_record("result/argmax", (batch, seq), argmax),
        _i32_record("result/focus_offsets", (batch + 1,), focus_offsets),
        _fp32_record("result/focus_logits", (len(focus_logits),), focus_logits),
    ]


def _records_by_name(records: Sequence[Record]) -> dict[str, Record]:
    by_name: dict[str, Record] = {}
    for record in records:
        if record.name in by_name:
            raise EvalFixtureError(f"duplicate record name: {record.name!r}")
        by_name[record.name] = record
    return by_name


def _get_int_scalar(by_name: dict[str, Record], name: str) -> int:
    try:
        record = by_name[name]
    except KeyError as error:
        raise EvalFixtureError(f"missing required record {name!r}") from error
    values = _read_signed_32(record)
    if record.shape not in ((), (1,)) or len(values) != 1:
        raise EvalFixtureError(f"{name!r} must be a scalar int32")
    return values[0]


def _records_to_cases(by_name: dict[str, Record], batch: int,
                      seq: int) -> list[EvalCase]:
    try:
        tokens_record = by_name["input/tokens"]
        lengths_record = by_name["input/lengths"]
        spans_record = by_name["input/spans"]
    except KeyError as error:
        raise EvalFixtureError(
            f"missing required input record {error.args[0]!r}") from error

    if tokens_record.shape != (batch, seq):
        raise EvalFixtureError(
            f"input/tokens shape {tokens_record.shape} != ({batch}, {seq})")
    if lengths_record.shape != (batch,):
        raise EvalFixtureError(
            f"input/lengths shape {lengths_record.shape} != ({batch},)")
    if spans_record.shape != (batch, 2):
        raise EvalFixtureError(
            f"input/spans shape {spans_record.shape} != ({batch}, 2)")

    tokens = _read_signed_32(tokens_record)
    lengths = _read_signed_32(lengths_record)
    spans = _read_signed_32(spans_record)

    # Focus is optional: a fixture without focus records carries none.
    if "input/focus_positions" in by_name:
        positions = _read_signed_32(by_name["input/focus_positions"])
        offsets = _read_signed_32(by_name["input/focus_offsets"])
        ids = _read_signed_32(by_name["input/focus_ids"])
        if len(positions) != batch:
            raise EvalFixtureError(
                f"input/focus_positions has {len(positions)} entries, "
                f"expected {batch}")
        if len(offsets) != batch + 1:
            raise EvalFixtureError(
                f"input/focus_offsets has {len(offsets)} entries, "
                f"expected {batch + 1}")
        if offsets[0] != 0 or offsets[-1] != len(ids):
            raise EvalFixtureError("input focus offsets do not span focus_ids")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise EvalFixtureError("input focus offsets are not monotonic")
    else:
        positions = [IGNORE_INDEX] * batch
        offsets = [0] * (batch + 1)
        ids = []

    cases: list[EvalCase] = []
    for row in range(batch):
        length = lengths[row]
        if not (0 <= length <= seq):
            raise EvalFixtureError(
                f"input/lengths[{row}] = {length} is outside [0, {seq}]")
        row_tokens = tokens[row * seq:row * seq + length]
        start = spans[row * 2]
        end = spans[row * 2 + 1]
        focus_start, focus_end = offsets[row], offsets[row + 1]
        cases.append(EvalCase(
            tokens=tuple(row_tokens),
            start=start,
            end=end,
            focus_position=positions[row],
            focus_ids=tuple(ids[focus_start:focus_end]),
        ))
    return cases


def _records_to_results(by_name: dict[str, Record], batch: int,
                        seq: int) -> list[EvalResult]:
    try:
        nll_record = by_name["result/nll"]
        argmax_record = by_name["result/argmax"]
    except KeyError as error:
        raise EvalFixtureError(
            f"missing required result record {error.args[0]!r}") from error

    if nll_record.shape != (batch, seq):
        raise EvalFixtureError(
            f"result/nll shape {nll_record.shape} != ({batch}, {seq})")
    if argmax_record.shape != (batch, seq):
        raise EvalFixtureError(
            f"result/argmax shape {argmax_record.shape} != ({batch}, {seq})")

    nll = _read_float_32(nll_record)
    argmax = _read_signed_32(argmax_record)

    if "result/focus_offsets" in by_name or "result/focus_logits" in by_name:
        if "result/focus_offsets" not in by_name or \
                "result/focus_logits" not in by_name:
            raise EvalFixtureError(
                "result focus needs both result/focus_offsets and "
                "result/focus_logits")
        offsets = _read_signed_32(by_name["result/focus_offsets"])
        logits = _read_float_32(by_name["result/focus_logits"])
        if len(offsets) != batch + 1:
            raise EvalFixtureError(
                f"result/focus_offsets has {len(offsets)} entries, "
                f"expected {batch + 1}")
        if offsets[0] != 0 or offsets[-1] != len(logits):
            raise EvalFixtureError(
                "result focus offsets do not span result/focus_logits")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise EvalFixtureError("result focus offsets are not monotonic")
    else:
        offsets = [0] * (batch + 1)
        logits = []

    results: list[EvalResult] = []
    for row in range(batch):
        focus_start, focus_end = offsets[row], offsets[row + 1]
        results.append(EvalResult(
            nll=tuple(nll[row * seq:(row + 1) * seq]),
            argmax=tuple(argmax[row * seq:(row + 1) * seq]),
            focus_logits=tuple(logits[focus_start:focus_end]),
        ))
    return results


def _infer_width(cases: Sequence[EvalCase] | None,
                 results: Sequence[EvalResult] | None,
                 seq: int | None) -> tuple[int, int]:
    """Return ``(seq, batch)``, inferring and cross-checking the two sides.

    The padded width is the result width when results are present (it can be
    wider than the longest real sequence), then an explicit ``seq``, then the
    longest case. The batch is the number of rows on whichever side is present.
    """
    if results:
        width = len(results[0].nll)
        for result in results:
            if len(result.nll) != width:
                raise EvalFixtureError(
                    "results disagree on the padded width")
        if seq is not None and int(seq) != width:
            raise EvalFixtureError(
                f"explicit seq {seq} disagrees with the result width {width}")
    elif seq is not None:
        width = int(seq)
    elif cases:
        width = max(case.length for case in cases)
    else:
        raise EvalFixtureError("cannot infer seq from an empty fixture")

    batch = len(results) if results else (len(cases) if cases else 0)
    if batch <= 0:
        raise EvalFixtureError("a fixture needs at least one row")
    if cases:
        for case in cases:
            if case.length > width:
                raise EvalFixtureError(
                    f"a case has {case.length} tokens, wider than seq {width}")
    return width, batch


def _cross_check(cases: Sequence[EvalCase] | None,
                 results: Sequence[EvalResult] | None) -> None:
    if cases is None or results is None:
        return
    if len(cases) != len(results):
        raise EvalFixtureError(
            f"cases and results disagree on batch: {len(cases)} != "
            f"{len(results)}")
    for row, (case, result) in enumerate(zip(cases, results)):
        expected = len(case.focus_ids)
        if len(result.focus_logits) != expected:
            raise EvalFixtureError(
                f"row {row} requested {expected} focused logits but has "
                f"{len(result.focus_logits)}")


# ---------------------------------------------------------------------------
# Fixture-level API
# ---------------------------------------------------------------------------


def cases_to_records(cases: Sequence[EvalCase], seq: int | None = None,
                     pad_id: int = 0) -> list[Record]:
    """Build the ``config/`` and ``input/`` records for ``cases``."""
    width, batch = _infer_width(cases, None, seq)
    return _config_records(batch, width, pad_id) + \
        _cases_to_records(cases, width, pad_id)


def results_to_records(results: Sequence[EvalResult],
                       seq: int | None = None) -> list[Record]:
    """Build the ``config/`` and ``result/`` records for ``results``."""
    width, batch = _infer_width(None, results, seq)
    return _config_records(batch, width, 0) + _results_to_records(results, width)


def fixture_to_records(cases: Sequence[EvalCase] | None = None,
                       results: Sequence[EvalResult] | None = None,
                       seq: int | None = None, pad_id: int = 0) -> list[Record]:
    """Build every record for a fixture carrying cases, results, or both."""
    if not cases and not results:
        raise EvalFixtureError("a fixture needs cases, results, or both")
    width, batch = _infer_width(cases, results, seq)
    _cross_check(cases, results)
    records = _config_records(batch, width, pad_id)
    if cases:
        records += _cases_to_records(cases, width, pad_id)
    if results:
        records += _results_to_records(results, width)
    return records


def records_to_fixture(records: Sequence[Record]) -> Fixture:
    """Parse a record list into a :class:`Fixture`."""
    by_name = _records_by_name(records)
    version = _get_int_scalar(by_name, "config/version")
    if version != FORMAT_VERSION:
        raise EvalFixtureError(f"unsupported format version: {version}")
    batch = _get_int_scalar(by_name, "config/batch")
    seq = _get_int_scalar(by_name, "config/seq")
    pad_id = (_get_int_scalar(by_name, "config/pad_id")
              if "config/pad_id" in by_name else 0)
    if batch < 0 or seq < 0:
        raise EvalFixtureError("config/batch and config/seq must be >= 0")

    has_input = any(name.startswith("input/") for name in by_name)
    has_result = any(name.startswith("result/") for name in by_name)
    if not has_input and not has_result:
        raise EvalFixtureError("fixture has neither input nor result records")

    cases = _records_to_cases(by_name, batch, seq) if has_input else None
    results = _records_to_results(by_name, batch, seq) if has_result else None
    _cross_check(cases, results)
    return Fixture(seq=seq, batch=batch, pad_id=pad_id, cases=cases,
                   results=results)


def write_cases(path: str | os.PathLike[str], cases: Sequence[EvalCase],
                seq: int | None = None, pad_id: int = 0) -> Path:
    """Write a cases-only fixture."""
    return write(path, cases_to_records(cases, seq, pad_id))


def write_results(path: str | os.PathLike[str],
                  results: Sequence[EvalResult],
                  seq: int | None = None) -> Path:
    """Write a results-only fixture."""
    return write(path, results_to_records(results, seq))


def write_fixture(path: str | os.PathLike[str],
                  cases: Sequence[EvalCase] | None = None,
                  results: Sequence[EvalResult] | None = None,
                  seq: int | None = None, pad_id: int = 0) -> Path:
    """Write a combined fixture and return the path."""
    return write(path, fixture_to_records(cases, results, seq, pad_id))


def read_fixture(path: str | os.PathLike[str]) -> Fixture:
    """Read a fixture written by any of the writers above."""
    return records_to_fixture(read(path))


def read_cases(path: str | os.PathLike[str]) -> list[EvalCase]:
    """Read the cases from a fixture, requiring that it carries them."""
    fixture = read_fixture(path)
    if fixture.cases is None:
        raise EvalFixtureError(f"fixture has no input records: {path}")
    return fixture.cases


def read_results(path: str | os.PathLike[str]) -> list[EvalResult]:
    """Read the results from a fixture, requiring that it carries them."""
    fixture = read_fixture(path)
    if fixture.results is None:
        raise EvalFixtureError(f"fixture has no result records: {path}")
    return fixture.results


__all__ = [
    "MAGIC",
    "VERSION",
    "FORMAT_VERSION",
    "MAX_RANK",
    "IGNORE_INDEX",
    "MASKED_NLL",
    "DTYPE_FP32",
    "DTYPE_INT32",
    "DTYPE_FP64",
    "DTYPE_INT64",
    "DTYPE_UINT8",
    "DTYPE_NAMES",
    "DTYPE_SIZES",
    "EvalFixtureError",
    "Record",
    "EvalCase",
    "EvalResult",
    "Fixture",
    "serialize",
    "parse",
    "write",
    "read",
    "cases_to_records",
    "results_to_records",
    "fixture_to_records",
    "records_to_fixture",
    "write_cases",
    "write_results",
    "write_fixture",
    "read_fixture",
    "read_cases",
    "read_results",
]
