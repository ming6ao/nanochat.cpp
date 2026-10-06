"""The portable ``NCTOKEN1`` tokenizer for the C++ runtime.

The C++ runtime reads parquet documents and tokenizes during the run
(``docs/parquet-native.md``), so the bridge no longer materializes ``NANO``
shards. This module supplies the two things the runtime still needs:

- ``parquet_files`` -- the parquet paths for a split, in dataset order.
- ``nctoken1_path`` -- the portable ``NCTOKEN1`` artifact.

It also reads that artifact and builds a ``tiktoken`` encoding from its merge
list (docs/tokenizer.md section 5), so the fixture generator and the self-test
can share the tokenizer with the C++ side. ``tiktoken`` is imported lazily, so
the torch-free self-test can import this module.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

# From include/nanochat/data.h.
NCTOKEN1_MAGIC = b"NCTOKEN1"
NCTOKEN1_VERSION = 1
NCTOKEN1_NAME = "tokenizer.nctoken"
#: Number of single-byte tokens; the first merge gets rank ``BASE_VOCAB_SIZE``.
BASE_VOCAB_SIZE = 256


def _tokenizer_dir() -> Path:
    base = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" /
                                                     "nanochat"))
    return Path(base) / "tokenizer"


def parquet_files(split: str) -> list[str]:
    """The parquet paths for a split. Train is all but the last file, and val
    is the last file, matching the reference dataloader. The C++ reader expands
    and orders them (docs/parquet-native.md)."""
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    from nanochat.dataset import list_parquet_files
    paths = [str(p) for p in list_parquet_files()]
    return paths[:-1] if split == "train" else paths[-1:]


def nctoken1_path() -> Path | None:
    """The portable ``NCTOKEN1`` artifact, or None when it is absent. When it
    is present the C++ runtime tokenizes parquet directly, so no shard is
    written (docs/parquet-native.md)."""
    path = _tokenizer_dir() / NCTOKEN1_NAME
    return path if path.is_file() else None


class Nctoken1Error(ValueError):
    """Raised when an ``NCTOKEN1`` artifact is missing or malformed."""


@dataclass(frozen=True)
class Nctoken1Artifact:
    """The portable tokenizer: split pattern, merge pairs, special tokens."""

    pattern: str
    merge_pairs: tuple[tuple[int, int], ...]
    special_tokens: tuple[tuple[str, int], ...]


def _read_u32(raw: bytes, offset: int) -> tuple[int, int]:
    """Read one little-endian ``u32``; return the value and the next offset."""
    if offset + 4 > len(raw):
        raise Nctoken1Error("truncated NCTOKEN1 artifact")
    return struct.unpack_from("<I", raw, offset)[0], offset + 4


def read_nctoken1(path) -> Nctoken1Artifact:
    """Parse an ``NCTOKEN1`` artifact (docs/tokenizer.md section 5).

    The layout is: magic, version, pattern, merge pairs, then special tokens.
    Every integer is little-endian. A bad file raises :class:`Nctoken1Error`.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 16 or raw[:8] != NCTOKEN1_MAGIC:
        raise Nctoken1Error(f"not an NCTOKEN1 artifact: {path}")
    version, offset = _read_u32(raw, 8)
    if version != NCTOKEN1_VERSION:
        raise Nctoken1Error(f"unsupported NCTOKEN1 version {version}: {path}")

    pattern_length, offset = _read_u32(raw, offset)
    if offset + pattern_length > len(raw):
        raise Nctoken1Error("truncated NCTOKEN1 pattern")
    try:
        pattern = raw[offset:offset + pattern_length].decode("utf-8")
    except UnicodeDecodeError as error:
        raise Nctoken1Error("invalid UTF-8 in the NCTOKEN1 pattern") from error
    offset += pattern_length

    merge_count, offset = _read_u32(raw, offset)
    if merge_count > (len(raw) - offset) // 8:
        raise Nctoken1Error("truncated NCTOKEN1 merge list")
    merge_pairs = []
    for _ in range(merge_count):
        left, offset = _read_u32(raw, offset)
        right, offset = _read_u32(raw, offset)
        merge_pairs.append((left, right))

    special_count, offset = _read_u32(raw, offset)
    special_tokens = []
    for _ in range(special_count):
        name_length, offset = _read_u32(raw, offset)
        if offset + name_length > len(raw):
            raise Nctoken1Error("truncated NCTOKEN1 special token name")
        try:
            name = raw[offset:offset + name_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise Nctoken1Error(
                "invalid UTF-8 in an NCTOKEN1 special token name") from error
        offset += name_length
        token_id, offset = _read_u32(raw, offset)
        special_tokens.append((name, token_id))

    if offset != len(raw):
        raise Nctoken1Error("trailing bytes after the NCTOKEN1 special tokens")
    return Nctoken1Artifact(pattern, tuple(merge_pairs), tuple(special_tokens))


def _derive_tokens(artifact: Nctoken1Artifact) -> list[bytes]:
    """Rebuild the bytes of every base token from the ordered merge pairs.

    Rank ``i`` is the token id. The first 256 ranks are the single-byte
    tokens; rank ``256 + index`` concatenates the two parts of merge
    ``index``. This mirrors ``LoadTokenizer`` in ``src/tokenizer.cc``.
    """
    tokens = [bytes([value]) for value in range(BASE_VOCAB_SIZE)]
    for index, (left, right) in enumerate(artifact.merge_pairs):
        rank = BASE_VOCAB_SIZE + index
        if left >= rank or right >= rank:
            raise Nctoken1Error(
                f"merge {index} refers to a token at or after its own rank")
        tokens.append(tokens[left] + tokens[right])
    return tokens


def mergeable_ranks(artifact: Nctoken1Artifact) -> dict[bytes, int]:
    """Map every base token byte string to its rank, the tiktoken input."""
    return {token: rank for rank, token in enumerate(_derive_tokens(artifact))}


def token_byte_lengths(artifact: Nctoken1Artifact) -> list[int]:
    """Source byte length of every token id; a special token has length zero."""
    lengths = [len(token) for token in _derive_tokens(artifact)]
    table = [0] * (len(lengths) + len(artifact.special_tokens))
    table[:len(lengths)] = lengths
    return table


def build_tiktoken_encoding(artifact: Nctoken1Artifact):
    """Build a ``tiktoken.Encoding`` from an ``NCTOKEN1`` artifact.

    ``tiktoken`` is imported here, not at module import, so the torch-free
    self-test can import this module. The reference environment supplies the
    package.
    """
    import tiktoken

    return tiktoken.Encoding(
        name="nctoken1",
        pat_str=artifact.pattern,
        mergeable_ranks=mergeable_ranks(artifact),
        special_tokens=dict(artifact.special_tokens),
    )


class Nctoken1Tokenizer:
    """The reference tokenizer interface, backed by the portable artifact.

    ``get_vocab_size``, ``get_bos_token_id``, and ``encode`` match the
    reference ``nanochat.tokenizer`` interface, so the same object drives the
    reference dataloader in the loader-parity fixture.
    """

    def __init__(self, artifact: Nctoken1Artifact) -> None:
        self.artifact = artifact
        self.encoding = build_tiktoken_encoding(artifact)
        self._bos_id = self.encoding.encode_single_token("<|bos|>")

    def get_vocab_size(self) -> int:
        return self.encoding.n_vocab

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def encode_special(self, name: str) -> int:
        return self.encoding.encode_single_token(name)

    def decode(self, ids) -> str:
        return self.encoding.decode(ids)

    def encode(self, text, prepend=None, append=None, num_threads=1):
        if isinstance(text, str):
            ids = self.encoding.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend)
            if append is not None:
                ids.append(append)
            return ids
        rows = self.encoding.encode_ordinary_batch(
            text, num_threads=num_threads)
        if prepend is not None:
            for row in rows:
                row.insert(0, prepend)
        if append is not None:
            for row in rows:
                row.append(append)
        return rows

    def token_byte_lengths(self) -> list[int]:
        return token_byte_lengths(self.artifact)


def load_nctoken1(path) -> Nctoken1Tokenizer:
    """Read an ``NCTOKEN1`` artifact and wrap its ``tiktoken`` encoding."""
    return Nctoken1Tokenizer(read_nctoken1(path))
