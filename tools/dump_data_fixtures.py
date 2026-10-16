#!/usr/bin/env python3
"""Generate grouped offline fixtures: tokenizer, loader, chat."""

from __future__ import annotations

import sys
import textwrap

_COMMANDS = ('tokenizer', 'loader', 'chat')
_USAGE = "usage: dump_data_fixtures.py {tokenizer|loader|chat} [options]"
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_USAGE)
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    _command = sys.argv[1]
    if _command not in _COMMANDS:
        print("unknown fixture kind: " + _command, file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)
else:
    _command = None

if _command == 'tokenizer':
    __doc__ = textwrap.dedent("""\
        Generate the reference tokenizer fixture, ``tests/data/tokenizer_fixture.bin``.

        The fixture is *data*: the script trains a small reference tokenizer with
        ``rustbpe`` and records the inference behavior of the matching ``tiktoken``
        encoding. Runtime C++ tests never import Python; they parse this file with
        ``tests/oracle_fixture.h``.

        What it contains
        ----------------
        * ``config/*``          the format version, the split pattern, the vocabulary
                                sizes, the merge hash, the corpus hash, and the
                                ``rustbpe`` commit and ``tiktoken`` version;
        * ``merges/*``          the ordered merges: the left and right ids, and the
                                token bytes of every merge;
        * ``tokens/*``          the token bytes of every base vocabulary token;
        * ``special/*``         the nine special token names and their ids;
        * ``split/*``           the reference split of a curated string set;
        * ``encode/*``          encode cases, including a leading or trailing special;
        * ``decode/*``          decode cases, including specials and partial UTF-8;
        * ``token_bytes/*``     the source byte length of every token.

        Binary format (little-endian)
        -----------------------------
        ::

            header   : magic[8] = b"NANOTOK1", uint32 version, uint32 num_records
            record   : uint16 name_len, name bytes,
                       uint8 dtype (1=int32, 4=uint8),
                       uint8 ndim, uint64 shape[ndim],
                       payload (product(shape) * dtype_size bytes)

        The container matches the ``NANOORC1`` oracle fixture, so the same reader
        (``tests/oracle_fixture.h``) parses both.

        Determinism
        -----------
        The corpus is fixed and the reference merge loop is deterministic. The script
        trains the reference twice and compares the two merge hashes. It fails when the
        hashes differ. The split check encodes each piece of the Python ``regex`` split
        and compares the result with the full ``tiktoken`` encoding for every curated
        string.

        The curated set stays inside the Unicode 16.0.0 letter, number, and whitespace
        ranges, so the Python split and the tiktoken split agree for these strings. The
        Python ``regex`` module may use a newer Unicode release, so do not add a
        version-sensitive code point to the curated set without a check.

        Run it from the repository root::

            python3 tools/dump_data_fixtures.py tokenizer
            python3 tools/dump_data_fixtures.py tokenizer --vocab-size 512 --out /tmp/f.bin

        The script needs ``rustbpe``, ``tiktoken``, and ``regex``. It never imports
        ``torch`` and never touches the GPU.

    """)
    from fixture_format import FixtureWriter
    import argparse
    import hashlib
    import struct
    import sys
    from pathlib import Path

    import regex
    import rustbpe
    import tiktoken

    # --- reference constants ---------------------------------------------------

    MERGE_BASE_VOCAB_SIZE = 256


    def recover_merge_pairs(
        mergeable_ranks: dict[bytes, int],
    ) -> list[tuple[int, int]]:
        """Recover the ordered merge pairs from a token-bytes-to-rank map.

        Rank ``i`` is the token id. The first 256 ranks are the single byte tokens.
        The function returns one ``(left, right)`` pair for every rank at or above
        256, in rank order.
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
        for rank in range(MERGE_BASE_VOCAB_SIZE, len(ranks)):
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
            candidates.sort()
            pairs.append(candidates[0])
        return pairs


    #: The nine special tokens, in the reference order (docs/tokenizer.md 2.4).
    SPECIAL_TOKENS = [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    ]

    #: The nanochat split pattern from docs/tokenizer.md section 2.5.
    SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"
        r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )

    #: The pinned rustbpe source. The tag v0.1.0 of karpathy/rustbpe.
    RUSTBPE_COMMIT = "8bcf3131ab2ed157ac1510201e620cc36068aa66"
    RUSTBPE_VERSION = "0.1.0"

    DEFAULT_VOCAB_SIZE = 512

    #: A fixed English-like corpus. It is long enough to train the small fixture
    #: vocabulary and it exercises letters, digits, punctuation, and whitespace.
    CORPUS = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "Sphinx of black quartz, judge my vow.",
        "The five boxing wizards jump quickly.",
        "Bright vixens jump; dozy fowl quack.",
        "Jinxed wizards pluck ivy from the big quilt.",
        "Waltz, bad nymph, for quick jigs vex.",
        "Quick zephyrs blow, vexing daft Jim.",
        "Two driven jocks help fax my big quiz.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "It is a truth universally acknowledged, that a single man in possession of "
        "a good fortune, must be in want of a wife.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "I've seen things you people wouldn't believe.",
        "Don't panic. The answer is 42, not 1234.",
        "We're going to need a bigger boat, aren't we?",
        'She said, "Hello, World!" and waved. Numbers: 1 22 333 4444.',
        "cafe naive resume jalapeno cooperate",
        "The value is 3.14 and 2,718, plus 100% of it.",
        "Line one\nline two\r\nline three\n",
        "   leading spaces and trailing spaces   ",
        "Tabs\tand\tspaces\t mixed together.",
        "Symbols: @#$%^&*()_+-=[]{}|;:',.<>/?`~",
        "def train_model(x, y): return x + y  # comment",
        "for i in range(10): print(i * 2)",
        "The rain in Spain stays mainly in the plain.",
        "Peter Piper picked a peck of pickled peppers.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "A man, a plan, a canal: Panama!",
        "Was it a car or a cat I saw?",
        "Never odd or even. Madam, I'm Adam.",
        "The llama is a quadruped, related to the camel.",
        "Pack my box with five dozen liquor jugs, again.",
    ]

    #: A curated string set that covers the seven pattern alternatives and the
    #: known edge cases in docs/tokenizer.md section 6.
    CURATED_STRINGS = [
        # Apostrophe and contraction.
        "don't", "I'll", "we've", "they're", "he's", "won't", "Mary's", "isn't",
        # Optional symbol, then letters.
        " hello", " !abc", "  !abc", "-word", "@user", "_private",
        # One or two digits.
        "1", "12", "123", "1234", "3.14", "007",
        # Optional space, symbols, and line ends.
        "!!!", "??\n", "...", " \t  ", "```", "a!!",
        # Whitespace and line ends.
        "\n", "\r\n", "  \n", "a\nb", "one\r\ntwo",
        # Trailing whitespace.
        "word ", "a  ", "end\t",
        # Whitespace runs.
        "   ", "\t\t", " \t ",
        # Mixed text.
        "Hello, World!", "naive cafe", "\u65e5\u672c\u8a9e", "123 456", "",
    ]

    #: Encode cases: ``(text, prepend special name or None, append name or None)``.
    ENCODE_CASES = (
        [(text, None, None) for text in CURATED_STRINGS]
        + [
            ("Hello, World!", "<|bos|>", None),
            ("Hello, World!", None, "<|assistant_end|>"),
            ("", "<|bos|>", "<|assistant_end|>"),
            ("naive cafe", "<|bos|>", "<|assistant_end|>"),
            ("def f(x): return x", "<|python_start|>", "<|python_end|>"),
            ("user message", "<|user_start|>", "<|user_end|>"),
        ]
    )

    # Dtype codes shared with tests/oracle_fixture.h.
    _MAGIC = b"NANOTOK1"


    def _flatten(items: list[bytes]) -> tuple[bytes, list[int]]:
        """Concatenate byte strings and return the payload plus prefix offsets."""
        payload = bytearray()
        offsets = [0]
        for item in items:
            payload += item
            offsets.append(len(payload))
        return bytes(payload), offsets


    def _flatten_ids(rows: list[list[int]]) -> tuple[list[int], list[int]]:
        """Concatenate integer lists and return the payload plus prefix offsets."""
        payload: list[int] = []
        offsets = [0]
        for row in rows:
            payload.extend(row)
            offsets.append(len(payload))
        return payload, offsets


    def merge_hash(mergeable_ranks: list[tuple[bytes, int]]) -> str:
        """Hash the ordered merges (rank >= 256), so two runs can be compared."""
        digest = hashlib.sha256()
        for token, rank in mergeable_ranks:
            if rank < 256:
                continue
            digest.update(struct.pack("<II", rank, len(token)))
            digest.update(token)
        return digest.hexdigest()


    def train_reference(vocab_size: int) -> list[tuple[bytes, int]]:
        """Train the reference tokenizer and return the mergeable ranks."""
        base_vocab_size = vocab_size - len(SPECIAL_TOKENS)
        if base_vocab_size < 256:
            raise ValueError("vocab_size must leave at least 256 base tokens")
        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            iter(CORPUS), base_vocab_size, pattern=SPLIT_PATTERN)
        if tokenizer.vocab_size != base_vocab_size:
            raise AssertionError(
                f"rustbpe returned {tokenizer.vocab_size} tokens, "
                f"expected {base_vocab_size}")
        pattern = tokenizer.get_pattern()
        if pattern != SPLIT_PATTERN:
            raise AssertionError("rustbpe returned a different split pattern")
        return list(tokenizer.get_mergeable_ranks())


    def build_encoding(
        mergeable_ranks: list[tuple[bytes, int]], vocab_size: int
    ) -> tiktoken.Encoding:
        """Build the matching reference tiktoken encoding."""
        ranks = {bytes(token): rank for token, rank in mergeable_ranks}
        offset = len(ranks)
        specials = {
            name: offset + index for index, name in enumerate(SPECIAL_TOKENS)
        }
        encoding = tiktoken.Encoding(
            name="nanochat-fixture",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=ranks,
            special_tokens=specials,
        )
        if encoding.n_vocab != vocab_size:
            raise AssertionError(
                f"tiktoken vocabulary is {encoding.n_vocab}, expected {vocab_size}")
        return encoding


    def check_split(encoding: tiktoken.Encoding) -> None:
        """Prove that the Python split agrees with the tiktoken split."""
        for text in CURATED_STRINGS:
            pieces = regex.findall(SPLIT_PATTERN, text)
            manual: list[int] = []
            for piece in pieces:
                manual.extend(encoding.encode_ordinary(piece))
            reference = encoding.encode_ordinary(text)
            if manual != reference:
                raise AssertionError(
                    f"split mismatch for {text!r}: {pieces} -> {manual} "
                    f"!= {reference}")


    def _resolve(value: int | str, specials: dict[str, int]) -> int:
        if isinstance(value, str):
            return specials[value]
        return value


    def decode_cases(specials: dict[str, int]) -> list[list[int]]:
        """Return the fixed decode cases, with specials resolved to ids."""
        raw: list[list[int | str]] = [
            [0xE2, 0x80],                      # partial UTF-8: one U+FFFD
            [0xE2, 0x82, 0xAC],                # complete euro sign
            [ord("h"), ord("i")],              # "hi"
            ["<|bos|>"],
            ["<|assistant_end|>"],
            ["<|user_start|>", ord("d"), "<|user_end|>"],
            ["<|bos|>", "<|user_start|>", ord("H"), ord("i"),
             "<|user_end|>", "<|assistant_start|>", ord("y"),
             "<|assistant_end|>"],
        ]
        return [[_resolve(value, specials) for value in row] for row in raw]


    def build_fixture(args: argparse.Namespace) -> tuple[Path, str]:
        first = train_reference(args.vocab_size)
        second = train_reference(args.vocab_size)
        first_hash = merge_hash(first)
        second_hash = merge_hash(second)
        if first_hash != second_hash:
            raise AssertionError(
                f"reference runs disagree: {first_hash} != {second_hash}")

        encoding = build_encoding(first, args.vocab_size)
        check_split(encoding)

        specials = {
            name: encoding.encode_single_token(name) for name in SPECIAL_TOKENS
        }
        base_vocab_size = len(first)
        mergeable = {bytes(token): rank for token, rank in first}
        merge_pairs = recover_merge_pairs(mergeable)
        if len(merge_pairs) != base_vocab_size - 256:
            raise AssertionError("merge pair count does not match the vocabulary")

        # Base token bytes, in rank order.
        inverse = {rank: bytes(token) for token, rank in first}
        base_bytes = [inverse[rank] for rank in range(base_vocab_size)]
        merge_bytes = [inverse[rank] for rank in range(256, base_vocab_size)]

        # Source byte length of every token. A special token has length zero.
        special_ids = set(specials.values())
        token_lengths = []
        for token_id in range(args.vocab_size):
            if token_id in special_ids:
                token_lengths.append(0)
            else:
                token_lengths.append(
                    len(encoding.decode_single_token_bytes(token_id)))

        # The reference split of the curated set.
        split_rows = [regex.findall(SPLIT_PATTERN, text) for text in CURATED_STRINGS]
        split_pieces = [piece.encode("utf-8") for row in split_rows for piece in row]
        string_piece_offsets = [0]
        for row in split_rows:
            string_piece_offsets.append(string_piece_offsets[-1] + len(row))

        # Encode cases.
        encode_inputs = [text.encode("utf-8") for text, _, _ in ENCODE_CASES]
        encode_ids: list[list[int]] = []
        encode_prepend: list[int] = []
        encode_append: list[int] = []
        for text, prepend_name, append_name in ENCODE_CASES:
            ids = list(encoding.encode_ordinary(text))
            prepend = -1 if prepend_name is None else specials[prepend_name]
            append = -1 if append_name is None else specials[append_name]
            if prepend >= 0:
                ids.insert(0, prepend)
            if append >= 0:
                ids.append(append)
            encode_ids.append(ids)
            encode_prepend.append(prepend)
            encode_append.append(append)

        # Decode cases.
        raw_decode = decode_cases(specials)
        decode_outputs = [encoding.decode(row).encode("utf-8") for row in raw_decode]

        corpus_text = "\n".join(CORPUS).encode("utf-8")
        pattern_bytes = SPLIT_PATTERN.encode("utf-8")
        commit_bytes = RUSTBPE_COMMIT.encode("ascii")
        version_bytes = tiktoken.__version__.encode("ascii")
        names_payload, names_offsets = _flatten(
            [name.encode("utf-8") for name in SPECIAL_TOKENS])
        strings_payload, strings_offsets = _flatten(
            [text.encode("utf-8") for text in CURATED_STRINGS])
        pieces_payload, piece_offsets = _flatten(split_pieces)
        encode_inputs_payload, encode_input_offsets = _flatten(encode_inputs)
        encode_ids_payload, encode_id_offsets = _flatten_ids(encode_ids)
        decode_ids_payload, decode_id_offsets = _flatten_ids(raw_decode)
        decode_outputs_payload, decode_output_offsets = _flatten(decode_outputs)
        base_bytes_payload, base_bytes_offsets = _flatten(base_bytes)
        merge_bytes_payload, merge_bytes_offsets = _flatten(merge_bytes)

        writer = FixtureWriter(args.out, magic=_MAGIC)
        writer.add_int32("config/format_version", [1])
        writer.add_uint8("config/pattern", pattern_bytes)
        writer.add_uint8("config/rustbpe_commit", commit_bytes)
        writer.add_uint8("config/rustbpe_version", RUSTBPE_VERSION.encode("ascii"))
        writer.add_uint8("config/tiktoken_version", version_bytes)
        writer.add_uint8("config/merge_hash", first_hash.encode("ascii"))
        writer.add_uint8("config/corpus_sha256",
                         hashlib.sha256(corpus_text).hexdigest().encode("ascii"))
        writer.add_int32("config/vocab_size", [args.vocab_size])
        writer.add_int32("config/base_vocab_size", [base_vocab_size])
        writer.add_int32("config/special_count", [len(SPECIAL_TOKENS)])
        writer.add_int32("config/merge_count", [len(merge_pairs)])

        writer.add_int32("merges/ranks",
                         [256 + index for index in range(len(merge_pairs))])
        writer.add_int32("merges/left", [left for left, _ in merge_pairs])
        writer.add_int32("merges/right", [right for _, right in merge_pairs])
        writer.add_int32("merges/token_bytes_offsets", merge_bytes_offsets)
        writer.add_uint8("merges/token_bytes", merge_bytes_payload)

        writer.add_int32("tokens/bytes_offsets", base_bytes_offsets)
        writer.add_uint8("tokens/bytes", base_bytes_payload)

        writer.add_int32("special/names_offsets", names_offsets)
        writer.add_uint8("special/names", names_payload)
        writer.add_int32("special/ids",
                         [specials[name] for name in SPECIAL_TOKENS])

        writer.add_int32("split/strings_offsets", strings_offsets)
        writer.add_uint8("split/strings", strings_payload)
        writer.add_int32("split/string_piece_offsets", string_piece_offsets)
        writer.add_int32("split/piece_offsets", piece_offsets)
        writer.add_uint8("split/pieces", pieces_payload)

        writer.add_int32("encode/input_offsets", encode_input_offsets)
        writer.add_uint8("encode/inputs", encode_inputs_payload)
        writer.add_int32("encode/prepend", encode_prepend)
        writer.add_int32("encode/append", encode_append)
        writer.add_int32("encode/id_offsets", encode_id_offsets)
        writer.add_int32("encode/ids", encode_ids_payload)

        writer.add_int32("decode/id_offsets", decode_id_offsets)
        writer.add_int32("decode/ids", decode_ids_payload)
        writer.add_int32("decode/output_offsets", decode_output_offsets)
        writer.add_uint8("decode/outputs", decode_outputs_payload)

        writer.add_uint8("token_bytes/lengths", bytes(token_lengths))

        writer.write()
        return args.out, first_hash


    def parse_args(argv: list[str]) -> argparse.Namespace:
        root = Path(__file__).resolve().parents[1]
        parser = argparse.ArgumentParser(
            description="Generate the reference tokenizer fixture.")
        parser.add_argument(
            "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
            help="full vocabulary size, including the nine special tokens")
        parser.add_argument(
            "--out", type=Path,
            default=root / "tests" / "data" / "tokenizer_fixture.bin",
            help="output fixture path")
        return parser.parse_args(argv)


    def main(argv: list[str]) -> int:
        args = parse_args(argv)
        path, merge_digest = build_fixture(args)
        payload = path.read_bytes()
        print(f"wrote {path} ({len(payload)} bytes)")
        print(f"merge hash: {merge_digest}")
        print(f"fixture sha256: {hashlib.sha256(payload).hexdigest()}")
        return 0
    _selected_main = main

if _command == 'loader':
    __doc__ = textwrap.dedent("""\
        Dump the loader-parity fixture (docs/parquet-native.md phase 8).

        The fixture pins the C++ document-mode loader against the reference
        ``tokenizing_distributed_data_loader_bos_bestfit`` (``nanochat/dataloader.py``).
        Only the document source is replaced; the reference packing loop runs
        unchanged.

        The tokenizer comes from a portable ``NCTOKEN1`` artifact, so the C++ side and
        the ``tiktoken`` side use the same merge list. Run this tool in the reference
        virtual environment (torch, tiktoken) because ``nanochat.dataloader`` imports
        torch.

            # 1. Dump the fixed document corpus for the trainer.
            python tools/dump_data_fixtures.py loader --dump-docs /tmp/loader_docs.txt

            # 2. Build the NCTOKEN1 artifact with the native trainer.
            tools/nanochat run t0-cpu -- <tok_train_main> --text /tmp/loader_docs.txt \
                --vocab-size 512 --out tests/data/loader_tokenizer.nctoken

            # 3. Write the fixture with the reference venv.
            /home/egrader/repos/nanochat/.venv/bin/python tools/dump_data_fixtures.py loader \
                --tokenizer tests/data/loader_tokenizer.nctoken \
                --out tests/data/loader_parity.bin

        The output format is little-endian:

            magic       8 bytes  "NCLDRP01"
            version     u32      1
            batch       u32
            seq         u32
            buffer      u32      the encoded-document buffer size
            source_batch u32     documents per source batch
            doc_count   u32
            documents   doc_count * (u32 length, bytes)
            steps       u32
            rows        steps * (batch*seq i32 tokens, batch*seq i32 targets)

    """)
    import argparse
    import struct
    import sys
    from pathlib import Path

    MAGIC = b"NCLDRP01"
    VERSION = 1

    # The fixed loader shape. `buffer` and `source_batch` must match the C++ test.
    BATCH = 2
    SEQ = 16
    BUFFER = 16
    SOURCE_BATCH = 256
    STEPS = 3
    DOC_COUNT = 256

    BASE_DOCS = [
        "The quick brown fox jumps over the lazy dog.",
        "a",
        "ab cd ef",
        "Caf\u00e9 \u2615 time.",
        "Short and sweet.",
        "A document that is intentionally very long so that it exceeds one single "
        "row and must be cropped somewhere in the middle of the text.",
        "0123456789",
        "Hello, world!",
    ]


    def documents() -> list[str]:
        """The fixed corpus: every base document with a unique suffix."""
        return [f"{BASE_DOCS[i % len(BASE_DOCS)]} [{i}]" for i in range(DOC_COUNT)]


    def dump_docs(path: Path) -> None:
        path.write_text("\n".join(documents()) + "\n", encoding="utf-8")
        print(f"wrote {DOC_COUNT} documents to {path}")


    def reference_rows(reference: Path, bridge: Path, tokenizer_path: Path):
        """Runs the reference best-fit loader over the fixed corpus."""
        sys.path.insert(0, str(reference))
        sys.path.insert(0, str(bridge))
        import torch
        from nanochat import dataloader
        from nanochat_cpp.data import load_nctoken1

        tokenizer = load_nctoken1(str(tokenizer_path))
        docs = documents()

        def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
            while True:
                for start in range(0, len(docs), tokenizer_batch_size):
                    batch = docs[start:start + tokenizer_batch_size]
                    yield batch, (0, 0, 1)

        dataloader._document_batches = fake_document_batches

        loader = dataloader.tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, BATCH, SEQ, split="train",
            tokenizer_threads=1, tokenizer_batch_size=SOURCE_BATCH,
            device="cpu", buffer_size=BUFFER)
        rows = []
        for _ in range(STEPS):
            inputs, targets = next(loader)
            assert inputs.shape == (BATCH, SEQ), inputs.shape
            rows.append((inputs.flatten().tolist(), targets.flatten().tolist()))
        return tokenizer.get_vocab_size(), rows


    def write_fixture(path: Path, rows) -> None:
        docs = documents()
        with path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<IIIII", VERSION, BATCH, SEQ, BUFFER,
                                     SOURCE_BATCH))
            handle.write(struct.pack("<I", len(docs)))
            for doc in docs:
                data = doc.encode("utf-8")
                handle.write(struct.pack("<I", len(data)))
                handle.write(data)
            handle.write(struct.pack("<I", len(rows)))
            for tokens, targets in rows:
                handle.write(struct.pack(f"<{len(tokens)}i", *tokens))
                handle.write(struct.pack(f"<{len(targets)}i", *targets))
        print(f"wrote {len(rows)} steps of {BATCH}x{SEQ} to {path}")


    def main(argv: list[str]) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--dump-docs", type=Path,
                            help="write the fixed corpus and exit")
        parser.add_argument("--tokenizer", type=Path,
                            help="the NCTOKEN1 artifact")
        parser.add_argument("--out", type=Path, help="the fixture to write")
        parser.add_argument("--reference", type=Path,
                            default=Path("/home/egrader/repos/nanochat"),
                            help="the reference nanochat checkout")
        parser.add_argument("--bridge", type=Path, default=Path("python"),
                            help="the directory that holds the nanochat_cpp bridge")
        args = parser.parse_args(argv)

        if args.dump_docs is not None:
            dump_docs(args.dump_docs)
            return 0
        if args.tokenizer is None or args.out is None:
            parser.error("give --dump-docs, or --tokenizer and --out")

        vocab_size, rows = reference_rows(args.reference, args.bridge,
                                          args.tokenizer)
        print(f"reference tokenizer vocab size: {vocab_size}")
        write_fixture(args.out, rows)
        return 0
    _selected_main = main

if _command == 'chat':
    __doc__ = textwrap.dedent("""\
        Generate the chat prompt fixture, ``tests/data/chat_fixture.bin``.

        The fixture pins the completion prompt that the reference
        ``nanochat.tokenizer.Tokenizer.render_for_completion`` produces for a few
        synthetic conversations. The test ``python/tests/chat_test.py`` renders the
        same conversations with :class:`nanochat_cpp.chat._PromptRenderer` and compares
        the ids. See docs/python.md sections 5 and 11.

        The file is a small container::

            header : magic[8] = b"NANCHAT1", uint32 version
            body   : UTF-8 JSON

        The generation needs the reference package and ``tiktoken`` only:

            /path/to/reference/.venv/bin/python tools/dump_data_fixtures.py chat         --out tests/data/chat_fixture.bin

    """)
    import argparse
    import json
    import struct
    import sys
    from pathlib import Path

    MAGIC = b"NANCHAT1"
    VERSION = 1

    #: The committed tokenizer artifact the fixture is rendered with.
    TOKENIZER = "tests/data/loader_tokenizer.nctoken"

    #: Synthetic conversations. Each renders with the reference template.
    CASES = [
        {
            "name": "simple",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ],
            },
        },
        {
            "name": "system",
            "conversation": {
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Name a color."},
                    {"role": "assistant", "content": "Blue"},
                ],
            },
        },
        {
            "name": "tool_parts",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": [
                        {"type": "python", "text": "2+2"},
                        {"type": "python_output", "text": "4"},
                        {"type": "text", "text": " #### 4"},
                    ]},
                    {"role": "user", "content": "Now double it."},
                    {"role": "assistant", "content": "#### 8"},
                ],
            },
        },
        {
            "name": "multiple_choice",
            "conversation": {
                "messages": [
                    {"role": "user", "content": (
                        "Multiple Choice question: What is 1+1?\n"
                        "- 1=A\n- 2=B\n\n"
                        "Respond only with the letter of the correct answer.")},
                    {"role": "assistant", "content": "B"},
                ],
            },
        },
    ]


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", default="tests/data/chat_fixture.bin")
        parser.add_argument("--tokenizer", default=TOKENIZER)
        args = parser.parse_args(argv)

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "python"))

        from nanochat_cpp import data  # noqa: E402
        from nanochat_cpp.reference import ensure_reference_on_path  # noqa: E402

        ensure_reference_on_path()
        from nanochat.tokenizer import RustBPETokenizer  # noqa: E402

        artifact = data.read_nctoken1(repo_root / args.tokenizer)
        encoding = data.build_tiktoken_encoding(artifact)
        tokenizer = RustBPETokenizer(encoding, "<|bos|>")

        cases = []
        for case in CASES:
            prompt_ids = tokenizer.render_for_completion(
                json.loads(json.dumps(case["conversation"])))
            cases.append({
                "name": case["name"],
                "conversation": case["conversation"],
                "prompt_ids": [int(token) for token in prompt_ids],
            })

        body = json.dumps({"version": VERSION, "tokenizer": args.tokenizer,
                           "cases": cases},
                          sort_keys=True, indent=2).encode("utf-8")
        payload = MAGIC + struct.pack("<I", VERSION) + body
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        print(f"wrote {out} ({len(cases)} cases, {len(payload)} bytes)")
        return 0
    _selected_main = main

if __name__ == "__main__":
    raise SystemExit(_selected_main(sys.argv[2:]))
