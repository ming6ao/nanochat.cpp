#!/usr/bin/env python3
"""Dump the loader-parity fixture (docs/parquet-native.md phase 8).

The fixture pins the C++ document-mode loader against the reference
``tokenizing_distributed_data_loader_bos_bestfit`` (``nanochat/dataloader.py``).
Only the document source is replaced; the reference packing loop runs
unchanged.

The tokenizer comes from a portable ``NCTOKEN1`` artifact, so the C++ side and
the ``tiktoken`` side use the same merge list. Run this tool in the reference
virtual environment (torch, tiktoken) because ``nanochat.dataloader`` imports
torch.

    # 1. Dump the fixed document corpus for the trainer.
    python tools/dump_loader_fixture.py --dump-docs /tmp/loader_docs.txt

    # 2. Build the NCTOKEN1 artifact with the native trainer.
    tools/nanochat run t0-cpu -- <tok_train_main> --text /tmp/loader_docs.txt \\
        --vocab-size 512 --out tests/data/loader_tokenizer.nctoken

    # 3. Write the fixture with the reference venv.
    /home/egrader/repos/nanochat/.venv/bin/python tools/dump_loader_fixture.py \\
        --tokenizer tests/data/loader_tokenizer.nctoken \\
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
"""

from __future__ import annotations

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
