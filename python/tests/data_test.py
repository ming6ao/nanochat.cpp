"""T0: the ``NCTOKEN1`` reader, and the reference tokenizer over it.

The test serializes a tiny artifact, reads it back, and checks the token-byte
reconstruction and the corruption handling. It is hermetic: standard library
only. See docs/python.md section 11.1.

The reader stays in the core, because it is the standard library alone. The
``tiktoken`` encoding built from it is the Z2 half of
``docs/dependency-layering-plan.md`` section 2.1, so
``build_tiktoken_encoding`` and ``Nctoken1Tokenizer`` now live in
``nanochat_cpp_bridge.reference_data`` and the coverage below follows them
there. A fake ``tiktoken`` module keeps the test hermetic: no wheel, no
network, and no shared library.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import data
from nanochat_cpp_bridge import _extras
from nanochat_cpp_bridge.reference_data import load_nctoken1


def nctoken1_bytes(pattern: str, merges, specials) -> bytes:
    """Serialize a tiny ``NCTOKEN1`` artifact."""
    payload = bytearray(data.NCTOKEN1_MAGIC)
    payload += struct.pack("<I", data.NCTOKEN1_VERSION)
    encoded = pattern.encode("utf-8")
    payload += struct.pack("<I", len(encoded)) + encoded
    payload += struct.pack("<I", len(merges))
    for left, right in merges:
        payload += struct.pack("<II", left, right)
    payload += struct.pack("<I", len(specials))
    for name, token_id in specials:
        name_bytes = name.encode("utf-8")
        payload += struct.pack("<I", len(name_bytes)) + name_bytes
        payload += struct.pack("<I", token_id)
    return bytes(payload)


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = "a+"
        self.merges = [(97, 98), (256, 99)]  # b"ab", then b"abc"
        self.specials = [("<|bos|>", 258), ("<|user_start|>", 259)]
        self.raw = nctoken1_bytes(self.pattern, self.merges, self.specials)
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_data_")
        self.path = Path(self.tmp.name) / data.NCTOKEN1_NAME
        self.path.write_bytes(self.raw)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_round_trip(self) -> None:
        artifact = data.read_nctoken1(self.path)
        self.assertEqual(artifact.pattern, self.pattern)
        self.assertEqual(artifact.merge_pairs, tuple(self.merges))
        self.assertEqual(artifact.special_tokens, tuple(self.specials))

    def test_derived_tokens(self) -> None:
        artifact = data.read_nctoken1(self.path)
        ranks = data.mergeable_ranks(artifact)
        self.assertEqual(ranks[b"\x00"], 0)
        self.assertEqual(ranks[b"ab"], 256)
        self.assertEqual(ranks[b"abc"], 257)
        lengths = data.token_byte_lengths(artifact)
        self.assertEqual(lengths[256], 2)
        self.assertEqual(lengths[257], 3)
        self.assertEqual(lengths[258], 0)
        self.assertEqual(lengths[259], 0)

    def test_corruption_is_rejected(self) -> None:
        bad = self.path.with_name("bad.nctoken")
        bad.write_bytes(b"NOTOKEN1" + self.raw[8:])
        with self.assertRaises(data.Nctoken1Error):
            data.read_nctoken1(bad)
        bad.write_bytes(self.raw[:-1])
        with self.assertRaises(data.Nctoken1Error):
            data.read_nctoken1(bad)

    def test_forward_merge_is_rejected(self) -> None:
        forward = data.Nctoken1Artifact("a+", ((257, 0),), ())
        with self.assertRaises(data.Nctoken1Error):
            data.mergeable_ranks(forward)


class _FakeTiktoken:
    """The slice of ``tiktoken`` the reference tokenizer calls.

    ``Encoding`` records the four arguments the module builds it from and
    answers the one lookup the tokenizer makes at construction, so the test
    reads the encoding the module produced without a wheel and without a real
    BPE merge.
    """

    class Encoding:
        def __init__(self, name, pat_str, mergeable_ranks, special_tokens):
            self.name = name
            self.pattern = pat_str
            self.ranks = dict(mergeable_ranks)
            self.specials = dict(special_tokens)
            self.n_vocab = len(self.ranks) + len(self.specials)

        def encode_single_token(self, name):
            return self.specials[name]


class ReferenceTokenizerTest(unittest.TestCase):
    """The Z2 tokenizer wraps the core reader, on a fake ``tiktoken``."""

    def setUp(self) -> None:
        self.pattern = "a+"
        self.merges = [(97, 98), (256, 99)]
        self.specials = [("<|bos|>", 258), ("<|user_start|>", 259)]
        self.raw = nctoken1_bytes(self.pattern, self.merges, self.specials)
        self.tmp = tempfile.TemporaryDirectory(prefix="nanochat_refdata_")
        self.path = Path(self.tmp.name) / data.NCTOKEN1_NAME
        self.path.write_bytes(self.raw)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load(self):
        """Load the artifact as the reference tokenizer, on a fake tiktoken."""
        with mock.patch.object(_extras, "import_package",
                               lambda name: _FakeTiktoken):
            return load_nctoken1(self.path)

    def test_the_encoding_is_built_from_the_core_derivations(self) -> None:
        tokenizer = self.load()
        self.assertEqual(tokenizer.encoding.name, "nctoken1")
        self.assertEqual(tokenizer.encoding.pattern, self.pattern)
        self.assertEqual(tokenizer.encoding.ranks,
                         data.mergeable_ranks(tokenizer.artifact))
        self.assertEqual(tokenizer.encoding.specials, dict(self.specials))

    def test_the_tokenizer_answers_the_artifact_ids(self) -> None:
        tokenizer = self.load()
        self.assertEqual(tokenizer.get_bos_token_id(), 258)
        self.assertEqual(tokenizer.encode_special("<|user_start|>"), 259)
        self.assertEqual(
            tokenizer.get_vocab_size(),
            len(data.mergeable_ranks(tokenizer.artifact))
            + len(self.specials))
        self.assertEqual(tokenizer.token_byte_lengths(),
                         data.token_byte_lengths(tokenizer.artifact))

    def test_the_core_no_longer_offers_the_tiktoken_entry_points(self) -> None:
        """The names moved with their import, and the reader stayed."""
        for name in ("build_tiktoken_encoding", "Nctoken1Tokenizer",
                     "load_nctoken1"):
            self.assertFalse(hasattr(data, name),
                             f"{name} is still in the core package")
        self.assertTrue(hasattr(data, "read_nctoken1"))


if __name__ == "__main__":
    unittest.main()
