#!/usr/bin/env python3
"""Convert a reference tokenizer into the portable ``NCTOKEN1`` artifact.

The reference nanochat pipeline saves a pickled ``tiktoken`` encoding next to a
``torch`` byte-length tensor. Neither file is portable: the pickle needs the
exact ``tiktoken`` build, and the tensor needs ``torch``. This script reads the
pickle and writes the small, little-endian ``NCTOKEN1`` container that the C++
tokenizer loads (docs/tokenizer.md section 5).

Artifact layout (all integers little-endian)::

    offset  size   field
    0       8      magic "NCTOKEN1"
    8       4      version (1)
    12      4      pattern length
    16      n      pattern bytes (UTF-8)
    16+n    4      merge count
    ...     8*m    merge pairs (u32 left, u32 right); rank = 256 + index
    ...     4      special count
    ...     ...    special records (u32 name length, name bytes, u32 id)

The merge pairs must reconstruct every token. The script recovers a pair for
each rank from the token bytes to rank map. A token has one valid pair when the
two parts are tokens with a lower rank. When two splits are valid, the script
picks the smallest left rank and then the smallest right rank, so the output is
stable. Either split loads to the same token bytes.

Run it inside the reference environment, which needs ``tiktoken``::

    python3 tools/convert_tokenizer.py --tokenizer <dir> --out tokenizer.nctoken

The module also exposes :func:`recover_merge_pairs` and
:func:`write_nctoken1`, so ``tools/dump_tokenizer_fixture.py`` can reuse the
same recovery rule without importing ``tiktoken`` twice.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

MAGIC = b"NCTOKEN1"
VERSION = 1
#: Number of single-byte tokens. The first merge gets rank ``BASE_VOCAB_SIZE``.
BASE_VOCAB_SIZE = 256


def recover_merge_pairs(
    mergeable_ranks: dict[bytes, int],
) -> list[tuple[int, int]]:
    """Recover the ordered merge pairs from a token-bytes-to-rank map.

    ``mergeable_ranks`` maps the bytes of every base token to its rank. Rank
    ``i`` is the token id. The first 256 ranks are the single byte tokens. This
    function returns one ``(left, right)`` pair for every rank at or above
    ``BASE_VOCAB_SIZE``, in rank order. The pair reconstructs the token bytes:
    ``token[left] + token[right] == token[rank]``.

    Raises:
        ValueError: if the ranks are not contiguous or a rank has no pair.
    """
    ranks = dict(mergeable_ranks)
    inverse: dict[int, bytes] = {}
    for token, rank in ranks.items():
        if rank in inverse:
            raise ValueError(f"duplicate rank {rank}")
        inverse[rank] = bytes(token)
    if sorted(inverse) != list(range(len(ranks))):
        raise ValueError("ranks must be contiguous and start at zero")

    pairs: list[tuple[int, int]] = []
    for rank in range(BASE_VOCAB_SIZE, len(ranks)):
        token = inverse[rank]
        candidates: list[tuple[int, int]] = []
        for split in range(1, len(token)):
            left = ranks.get(token[:split])
            right = ranks.get(token[split:])
            if left is None or right is None:
                continue
            if left < rank and right < rank:
                candidates.append((left, right))
        if not candidates:
            raise ValueError(f"no merge pair reconstructs rank {rank}")
        # Smallest left rank, then smallest right rank: a stable choice.
        candidates.sort()
        pairs.append(candidates[0])
    return pairs


def write_nctoken1(
    stream: BinaryIO,
    pattern: str,
    merge_pairs: Sequence[tuple[int, int]],
    special_tokens: Iterable[tuple[str, int]],
) -> None:
    """Write one ``NCTOKEN1`` artifact to a binary stream."""
    pattern_bytes = pattern.encode("utf-8")
    ordered_specials = sorted(special_tokens, key=lambda item: item[1])
    stream.write(MAGIC)
    stream.write(struct.pack("<I", VERSION))
    stream.write(struct.pack("<I", len(pattern_bytes)))
    stream.write(pattern_bytes)
    stream.write(struct.pack("<I", len(merge_pairs)))
    for left, right in merge_pairs:
        stream.write(struct.pack("<II", left, right))
    stream.write(struct.pack("<I", len(ordered_specials)))
    for name, token_id in ordered_specials:
        name_bytes = name.encode("utf-8")
        stream.write(struct.pack("<I", len(name_bytes)))
        stream.write(name_bytes)
        stream.write(struct.pack("<I", token_id))


def load_reference(path: Path):
    """Load a reference tokenizer from a pickle file or a directory.

    A directory must hold ``tokenizer.pkl``. The pickle is the reference
    ``tiktoken.Encoding`` or a ``RustBPETokenizer`` that holds one in ``enc``.
    """
    if path.is_dir():
        path = path / "tokenizer.pkl"
    with path.open("rb") as handle:
        encoding = pickle.load(handle)
    return getattr(encoding, "enc", encoding)


def extract(encoding) -> tuple[str, list[tuple[int, int]], list[tuple[str, int]]]:
    """Read the pattern, the merge pairs, and the special tokens."""
    try:
        pattern = encoding._pat_str
        mergeable_ranks = encoding._mergeable_ranks
        special_tokens = encoding._special_tokens
    except AttributeError as error:
        raise ValueError("not a reference tiktoken encoding") from error
    pairs = recover_merge_pairs(dict(mergeable_ranks))
    specials = list(special_tokens.items())
    return pattern, pairs, specials


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a reference tokenizer into NCTOKEN1.")
    parser.add_argument(
        "--tokenizer", required=True, type=Path,
        help="reference tokenizer directory or tokenizer.pkl file")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output artifact (default: <tokenizer dir>/tokenizer.nctoken)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    encoding = load_reference(args.tokenizer)
    pattern, merge_pairs, special_tokens = extract(encoding)
    out = args.out
    if out is None:
        base = args.tokenizer if args.tokenizer.is_dir() else args.tokenizer.parent
        out = base / "tokenizer.nctoken"
    with out.open("wb") as handle:
        write_nctoken1(handle, pattern, merge_pairs, special_tokens)
    print(f"wrote {out}: pattern={len(pattern)} bytes, "
          f"{len(merge_pairs)} merges, {len(special_tokens)} special tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
