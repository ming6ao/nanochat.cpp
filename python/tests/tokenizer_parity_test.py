"""T0: the ``data.py`` reader and the C tokenizer agree on one artifact.

``nanochat_cpp.data`` stays standard library only, because five callers read
the ``NCTOKEN1`` artifact where no shared library exists
(``docs/api-surface.md`` R12). The duplicate reader is therefore deliberate, and
this test is the gate it lacked: it loads one artifact twice, once through the C
ABI (``nanochat_tokenizer_load`` with ``nanochat_encode`` and
``nanochat_decode``) and once through :mod:`nanochat_cpp.data`, then compares
the mergeable ranks, the token byte lengths, and the encode result of a small
corpus. The artifact is the one the tokenizer tests already read, so both
readers see the same bytes and no new fixture is committed.

The encoder is what makes the comparison exact. For one id out of
``nanochat_encode``, the C tokenizer consumed exactly the bytes the caller
passed, so the id a rank encodes to and the length of its token are both
observable for every rank without reading C memory. ``nanochat_decode`` repairs
invalid UTF-8, so it only supplies the two rows a C string cannot carry: rank
zero, whose byte is NUL, and the special tokens, which have no source bytes.
"""

from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

from nanochat_cpp import _build, _core, _lib, data

ARTIFACT = "tests/data/loader_tokenizer.nctoken"

#: The two readers meet on text, so the corpus needs an empty string, ASCII,
#: and a string whose merge spans two UTF-8 bytes (``Caf`` plus U+00E9).
CORPUS = (
    "",
    "hello",
    "Hello, world!",
    "Caf\u00e9",
    "  \n",
    "0",
)


def resolve(relative: str) -> Path:
    """A runfiles path when the test runs under Bazel, else a source path."""
    found = _lib.find_in_runfiles(relative)
    if found is not None:
        return found
    return _build.repo_root() / relative


class TokenizerParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _lib.load()
        cls.path = resolve(ARTIFACT)
        cls.artifact = data.read_nctoken1(cls.path)
        cls.ranks = data.mergeable_ranks(cls.artifact)
        cls.lengths = data.token_byte_lengths(cls.artifact)
        cls.handle = _core.check_handle(
            cls.lib, cls.lib.nanochat_tokenizer_load(str(cls.path).encode()),
            "nanochat_tokenizer_load", _core.KIND_TOKENIZER)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lib.nanochat_tokenizer_free(cls.handle)

    def encode_bytes(self, raw: bytes) -> list[int]:
        """The ids the C tokenizer encodes ``raw`` to."""
        needed = _core.check_code(
            self.lib, self.lib.nanochat_encode(self.handle, raw, None, 0),
            "nanochat_encode")
        if needed == 0:
            return []
        buffer = (ctypes.c_int * needed)()
        _core.check_code(
            self.lib,
            self.lib.nanochat_encode(self.handle, raw, buffer, needed),
            "nanochat_encode")
        return [int(value) for value in buffer]

    def decode_bytes(self, ids) -> bytes:
        """The bytes the C tokenizer decodes ``ids`` to."""
        count = len(ids)
        buffer = (ctypes.c_int * count)(*ids) if count else None
        needed = _core.check_code(
            self.lib,
            self.lib.nanochat_decode(self.handle, buffer, count, None, 0),
            "nanochat_decode")
        out = ctypes.create_string_buffer(needed + 1)
        _core.check_code(
            self.lib,
            self.lib.nanochat_decode(self.handle, buffer, count,
                                     ctypes.cast(out, ctypes.c_void_p),
                                     needed + 1),
            "nanochat_decode")
        return out.raw[:needed]

    def test_the_corpus_holds_the_two_required_cases(self) -> None:
        # The comparison is only as good as the corpus, and the byte-level
        # checks below need an empty string and a non-ASCII one to mean
        # anything.
        self.assertIn("", CORPUS)
        self.assertTrue(any(text and not text.isascii() for text in CORPUS))

    def test_mergeable_ranks_agree_with_the_c_encoder(self) -> None:
        # One entry per rank: a merge that repeated a token would leave the
        # dict with fewer entries than the C reader has ranks.
        self.assertEqual(len(self.ranks),
                         data.BASE_VOCAB_SIZE + len(self.artifact.merge_pairs))
        # The non-ASCII merges are the interesting rows: `Caf` plus U+00E9
        # reaches one.
        self.assertTrue(any(rank >= data.BASE_VOCAB_SIZE
                            and not token.isascii()
                            for token, rank in self.ranks.items()))
        # Rank zero is the NUL byte, and a C string cannot carry it, so the
        # decoder answers that one row.
        self.assertEqual(self.decode_bytes([0]), b"\x00")
        for token, rank in self.ranks.items():
            if rank == 0:
                continue
            # One id out means the C tokenizer holds these bytes as one token
            # and holds it under this rank.
            self.assertEqual(self.encode_bytes(token), [rank], f"rank {rank}")

    def test_token_byte_lengths_sum_to_the_source_bytes(self) -> None:
        self.assertEqual(len(self.lengths),
                         len(self.ranks) + len(self.artifact.special_tokens))
        # A special token has no source bytes of its own, and the C decoder
        # answers its name in place of text.
        for name, token_id in self.artifact.special_tokens:
            self.assertEqual(self.lengths[token_id], 0, name)
            self.assertEqual(self.decode_bytes([token_id]),
                             name.encode("utf-8"), name)
        # The table is what the byte budget is computed from, so the sum over
        # one encoding must be the source length the C encoder consumed.
        for text in CORPUS:
            with self.subTest(text=text):
                source = text.encode("utf-8")
                ids = self.encode_bytes(source)
                self.assertEqual(sum(self.lengths[value] for value in ids),
                                 len(source))

    def test_encode_of_the_corpus_rebuilds_from_the_reader(self) -> None:
        tokens = {rank: token for token, rank in self.ranks.items()}
        for text in CORPUS:
            with self.subTest(text=text):
                source = text.encode("utf-8")
                ids = self.encode_bytes(source)
                # The C encoder chose these ids; the reader's rank table must
                # spell the same source bytes.
                self.assertEqual(b"".join(tokens[value] for value in ids),
                                 source)
                self.assertEqual(self.decode_bytes(ids), source)


if __name__ == "__main__":
    unittest.main()
