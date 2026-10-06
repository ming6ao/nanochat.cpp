"""Just-in-time parquet -> NANO shard materialization.

The C++ runtime reads only pre-tokenized ``NANO`` shards (see
``include/nanochat/data.h``). This module tokenizes exactly the tokens a run
needs from the reference parquet dataset, writes a shard in that format plus
the ``<shard>.bytes`` sidecar that ``DataLoader`` reads for bits-per-byte, and
caches the result so repeated runs reuse it.

The token stream is BOS-separated documents in dataset order. nanochat packs
batches with BOS-aligned best-fit; the C++ ``DataLoader`` does its own packing,
so the two are close but not bit-identical. The training-parity harness is the
exact gate; this path is for training.

The tokenizer comes from the reference pickle when the reference directory
holds one. The bridge also reads the portable ``NCTOKEN1`` artifact and builds
a ``tiktoken`` encoding from its merge list (docs/tokenizer.md section 5), so a
run can use the portable file alone. ``tiktoken`` is imported lazily, so the
torch-free self-test can import this module.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

# From include/nanochat/data.h.
MAGIC = 0x4F4E414E  # "NANO"
VERSION = 1
_U16 = "H"
_U32 = "I"

# From docs/tokenizer.md section 5 and tools/convert_tokenizer.py. The bridge
# reads this portable artifact when the reference directory holds one.
NCTOKEN1_MAGIC = b"NCTOKEN1"
NCTOKEN1_VERSION = 1
NCTOKEN1_NAME = "tokenizer.nctoken"
#: Number of single-byte tokens; the first merge gets rank ``BASE_VOCAB_SIZE``.
BASE_VOCAB_SIZE = 256


def cache_root() -> Path:
    root = os.environ.get("NANOCHAT_CPP_CACHE")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "nanochat_cpp" / "shards"


def _fingerprint(parquet_paths: list[Path], tokenizer_dir: Path, max_tokens: int,
                 width: int, split: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"split={split};max={max_tokens};width={width};".encode())
    for path in parquet_paths:
        try:
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)};"
                          .encode())
        except OSError:
            digest.update(f"{path.name}:missing;".encode())
    for name in ("tokenizer.pkl", "token_bytes.pt", NCTOKEN1_NAME):
        path = tokenizer_dir / name
        if path.is_file():
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{int(stat.st_mtime)};".encode())
    return digest.hexdigest()[:16]


def _write_header(handle, num_tokens: int, width: int) -> None:
    handle.seek(0)
    handle.write(struct.pack("<IIQII", MAGIC, VERSION, num_tokens, width, 0))


def _as_bytes(values: list[int], width: int) -> bytes:
    code = _U16 if width == 2 else _U32
    buf = array(code, values)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _tokenizer_dir() -> Path:
    base = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" /
                                                     "nanochat"))
    return Path(base) / "tokenizer"


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

    ``materialize`` needs only ``get_vocab_size``, ``get_bos_token_id``, and
    ``encode``. ``token_byte_lengths`` also lets the shard sidecar skip the
    ``torch`` tensor.
    """

    def __init__(self, artifact: Nctoken1Artifact) -> None:
        self.artifact = artifact
        self.encoding = build_tiktoken_encoding(artifact)
        self._bos_id = self.encoding.encode_single_token("<|bos|>")

    def get_vocab_size(self) -> int:
        return self.encoding.n_vocab

    def get_bos_token_id(self) -> int:
        return self._bos_id

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


def _load_bridge_tokenizer():
    """Return the tokenizer the bridge tokenizes with.

    The portable ``NCTOKEN1`` artifact wins when the reference directory holds
    one. Otherwise the reference pickle supplies the tokenizer, which needs the
    reference environment.
    """
    artifact_path = _tokenizer_dir() / NCTOKEN1_NAME
    if artifact_path.is_file():
        return load_nctoken1(artifact_path)
    from nanochat.tokenizer import get_tokenizer
    return get_tokenizer()


class ShardWriter:
    """Streaming writer for the NANO shard format (header + token stream)."""

    def __init__(self, path, width: int = 2) -> None:
        self.path = Path(path)
        self.width = width
        self.count = 0
        self._tmp = str(self.path) + ".tmp"
        self._handle = open(self._tmp, "wb")
        _write_header(self._handle, 0, width)

    def write(self, ids: list[int]) -> None:
        if not ids:
            return
        self._handle.write(_as_bytes(ids, self.width))
        self.count += len(ids)

    def close(self) -> Path:
        _write_header(self._handle, self.count, self.width)
        self._handle.close()
        os.replace(self._tmp, self.path)
        return self.path

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_shard(path) -> tuple[int, int, list[int]]:
    """Parse a NANO shard; returns ``(num_tokens, width, tokens)``."""
    with open(path, "rb") as handle:
        header = handle.read(24)
        magic, version, num_tokens, width, _reserved = struct.unpack(
            "<IIQII", header)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"bad shard header: {path}")
        code = _U16 if width == 2 else _U32
        raw = handle.read(num_tokens * width)
    tokens = array(code)
    tokens.frombytes(raw)
    if sys.byteorder != "little":
        tokens.byteswap()
    return num_tokens, width, tokens.tolist()


def materialize(split: str, max_tokens: int, width: int = 2, force: bool = False):
    """Tokenize ``max_tokens`` tokens of ``split`` into a cached NANO shard.

    Returns the shard path. ``split`` is ``"train"`` or ``"val"`` (the last
    parquet file, matching nanochat). The tokenizer comes from the portable
    ``NCTOKEN1`` artifact when present, otherwise from the reference checkout.
    The parquet reader comes from the reference checkout, which must already be
    importable.

    Documents are read in small pyarrow record batches and tokenized in small
    chunks, so a short run only touches the first few batches of the dataset
    instead of a whole parquet row group.
    """
    import pyarrow.parquet as pq

    from nanochat.dataset import list_parquet_files

    tokenizer = _load_bridge_tokenizer()
    bos = tokenizer.get_bos_token_id()
    vocab_size = tokenizer.get_vocab_size()

    parquet_paths = [Path(p) for p in list_parquet_files()]
    if split == "train":
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]

    out_dir = cache_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _fingerprint(parquet_paths, _tokenizer_dir(), max_tokens, width, split)
    out_path = out_dir / f"{split}_{max_tokens}_{width}_{key}.bin"
    if out_path.is_file() and not force:
        return out_path

    tmp_path = out_path.with_suffix(".bin.tmp")
    writer = ShardWriter(tmp_path, width)
    try:
        done = False
        for path in parquet_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
                documents = batch.column("text").to_pylist()
                token_lists = tokenizer.encode(documents, prepend=bos,
                                               num_threads=1)
                for ids in token_lists:
                    take = min(len(ids), max_tokens - writer.count)
                    if take <= 0:
                        done = True
                        break
                    writer.write(ids[:take])
                    if writer.count >= max_tokens:
                        done = True
                        break
                if done:
                    break
            if done:
                break
    finally:
        writer.close()
    os.replace(tmp_path, out_path)

    byte_lengths = None
    if isinstance(tokenizer, Nctoken1Tokenizer):
        byte_lengths = tokenizer.token_byte_lengths()
    _write_token_bytes(out_path, vocab_size, byte_lengths)
    return out_path


def _write_token_bytes(shard_path: Path, vocab_size: int,
                       byte_lengths: list[int] | None = None) -> None:
    """Write the ``<shard>.bytes`` sidecar (u32 vocab + one byte per token).

    ``byte_lengths`` comes from a portable ``NCTOKEN1`` artifact. When it is
    absent, the reference ``token_bytes.pt`` tensor supplies the table, which
    needs ``torch``.
    """
    if byte_lengths is None:
        try:
            from nanochat.tokenizer import get_token_bytes
            byte_lengths = get_token_bytes(device="cpu").to("cpu").tolist()
        except Exception:  # noqa: BLE001 - bpb is optional; fall back
            return
    table = byte_lengths
    if len(table) < vocab_size:
        return
    payload = struct.pack("<I", vocab_size)
    payload += bytes(int(min(max(v, 0), 255)) for v in table[:vocab_size])
    with open(str(shard_path) + ".bytes", "wb") as handle:
        handle.write(payload)


def tokens_for_run(total_batch_size: int, num_iterations: int, seq_len: int,
                   margin: int = 2048) -> int:
    """Tokens to tokenize for a run (one extra sequence for the target shift)."""
    return total_batch_size * num_iterations + seq_len + margin
