"""T0: the ``NCTOKEN1`` reader (replaces the removed self-test check).

The test serializes a tiny artifact, reads it back, and checks the token-byte
reconstruction and the corruption handling. It is hermetic: standard library
only. See docs/python.md section 11.1.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from nanochat_cpp import data


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


if __name__ == "__main__":
    unittest.main()
