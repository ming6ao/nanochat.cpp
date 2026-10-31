"""The reference data surface: the parquet listing and the tiktoken tokenizer.

``nanochat_cpp.data`` (Z1) reads the portable ``NCTOKEN1`` artifact with the
standard library: the magic, the merge list, the derived token bytes, and the
per-token byte lengths. Two things it cannot do without an extra live here
(``docs/dependency-layering-plan.md`` section 2.1):

``parquet_files(split)``
    The parquet paths of a split in dataset order, from
    ``nanochat.dataset.list_parquet_files``. The C++ reader expands and orders
    them (``docs/parquet-native.md``).
``build_tiktoken_encoding(artifact)``, ``Nctoken1Tokenizer``, ``load_nctoken1``
    The ``tiktoken`` encoding built from an ``NCTOKEN1`` artifact, and the
    reference tokenizer interface over it (``docs/tokenizer.md`` section 5).

The artifact reader, :func:`nanochat_cpp.data.read_nctoken1`, and the pure
derivations :func:`nanochat_cpp.data.mergeable_ranks` and
:func:`nanochat_cpp.data.token_byte_lengths` stay in the core, so this module
imports them instead of copying them.

:func:`parquet_lister` returns :func:`parquet_files` as a callable, which is
the seam the core is handed when a caller wants the reference dataset order
without importing the reference package itself.

Extras: ``reference`` (``tiktoken`` and ``nanochat``).
"""

from __future__ import annotations

from pathlib import Path

from nanochat_cpp.data import (
    Nctoken1Artifact,
    Nctoken1Error,
    mergeable_ranks,
    read_nctoken1,
    token_byte_lengths,
)

from . import _extras

__all__ = [
    "Nctoken1Artifact",
    "Nctoken1Error",
    "Nctoken1Tokenizer",
    "build_tiktoken_encoding",
    "load_nctoken1",
    "mergeable_ranks",
    "parquet_files",
    "parquet_lister",
    "read_nctoken1",
    "token_byte_lengths",
]


def parquet_files(split: str) -> list[str]:
    """The parquet paths for a split.

    Train is all but the last file and val is the last file, matching the
    reference dataloader. The C++ reader expands and orders them
    (``docs/parquet-native.md``). The reference package supplies the listing,
    so the reference checkout goes on ``sys.path`` first.
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    _extras.reference_on_path()
    dataset = _extras.import_package("nanochat.dataset")
    paths = [str(path) for path in dataset.list_parquet_files()]
    return paths[:-1] if split == "train" else paths[-1:]


def parquet_lister():
    """The dataset-order lister, as one callable a core caller can be handed.

    The core never names this package (gate G5), so a host that wants the
    reference order injects :func:`parquet_files` through its own seam.
    """
    return parquet_files


def build_tiktoken_encoding(artifact: Nctoken1Artifact):
    """Build a ``tiktoken.Encoding`` from an ``NCTOKEN1`` artifact.

    ``tiktoken`` is imported here, not at module import, so the package imports
    with no extra installed. The merge list comes from the core's pure
    derivation, so the encoding and the C++ tokenizer read one artifact the
    same way.
    """
    tiktoken = _extras.import_package("tiktoken")

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


def load_nctoken1(path: str | Path) -> Nctoken1Tokenizer:
    """Read an ``NCTOKEN1`` artifact and wrap its ``tiktoken`` encoding."""
    return Nctoken1Tokenizer(read_nctoken1(path))
