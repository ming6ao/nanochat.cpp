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
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from array import array
from pathlib import Path

# From include/nanochat/data.h.
MAGIC = 0x4F4E414E  # "NANO"
VERSION = 1
_U16 = "H"
_U32 = "I"


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
    for name in ("tokenizer.pkl", "token_bytes.pt"):
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
    parquet file, matching nanochat). The tokenizer and dataset come from the
    reference checkout, which must already be importable.

    Documents are read in small pyarrow record batches and tokenized in small
    chunks, so a short run only touches the first few batches of the dataset
    instead of a whole parquet row group.
    """
    import pyarrow.parquet as pq

    from nanochat.dataset import list_parquet_files
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
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

    _write_token_bytes(out_path, vocab_size)
    return out_path


def _write_token_bytes(shard_path: Path, vocab_size: int) -> None:
    """Write the ``<shard>.bytes`` sidecar (u32 vocab + one byte per token)."""
    try:
        from nanochat.tokenizer import get_token_bytes
        table = get_token_bytes(device="cpu").to("cpu").tolist()
    except Exception:  # noqa: BLE001 - bpb is optional; fall back to one byte
        return
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
