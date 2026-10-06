#!/usr/bin/env python3
"""Full-vocabulary tokenizer parity: native trainer vs reference rustbpe.

The native trainer `tok_train_main` and the reference `rustbpe` must produce the
same ordered merge list for the same corpus and vocabulary. The fixture test in
`src/tokenizer/tokenizer_parity_test.cc` proves this at small scale. This script
proves it at the full vocabulary size, where the native trainer reaches the same
32503 merges as the reference and the base token bytes match one for one.

Run it with the reference environment, from the repository root (see
``docs/python.md`` for the Python surface):

    python3 tools/check_tokenizer_parity.py --max-chars 40000000

The script builds a one-document-per-line corpus from the parquet shards,
trains the reference tokenizer, runs the native trainer under `tools/nanochat`
(the trainer refuses to start outside the sandbox), and compares every base
token byte. It exits nonzero on the first mismatch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
REFERENCE_ROOT = Path(os.environ.get("NANOCHAT_REPO",
                                    str(REPO_ROOT.parent / "nanochat")))
sys.path.insert(0, str(REFERENCE_ROOT))

import pyarrow.parquet as pq  # noqa: E402
import rustbpe  # noqa: E402
from nanochat.tokenizer import SPLIT_PATTERN  # noqa: E402
from nanochat_cpp import data  # noqa: E402

SPECIAL_COUNT = 9


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NANOCHAT_BASE_DIR",
                               str(Path.home() / ".cache" / "nanochat")))
    parser.add_argument("--parquet-dir", type=Path,
                        default=base / "base_data_climbmix")
    parser.add_argument("--max-chars", type=int, default=40_000_000)
    parser.add_argument("--doc-cap", type=int, default=10000)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--binary", type=Path,
                        default=REPO_ROOT / "bazel-bin/src/tokenizer/tok_train_main")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp"))
    return parser.parse_args(argv)


def build_corpus(parquet_dir: Path, max_chars: int, out: Path) -> tuple[int, int]:
    """Write one document per line; replace in-document newlines with spaces."""
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {parquet_dir}")
    chars = 0
    docs = 0
    with open(out, "w", encoding="utf-8") as handle:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=512, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    text = text.replace("\r", " ").replace("\n", " ")
                    handle.write(text + "\n")
                    chars += len(text)
                    docs += 1
                    if chars >= max_chars:
                        return docs, chars
    return docs, chars


def reference_tokens(corpus: Path, vocab_size: int,
                     doc_cap: int) -> list[bytes]:
    docs = [doc[:doc_cap]
            for doc in corpus.read_text(encoding="utf-8").splitlines()]
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(docs, vocab_size - SPECIAL_COUNT,
                                  pattern=SPLIT_PATTERN)
    merges = tokenizer.get_mergeable_ranks()
    return [bytes(token) for token, _ in sorted(merges, key=lambda i: i[1])]


def native_tokens(binary: Path, corpus: Path, vocab_size: int, doc_cap: int,
                  container: Path, profile: str) -> list[bytes]:
    command = [
        str(REPO_ROOT / "tools" / "nanochat"), "run", profile, "--",
        str(binary), "--text", str(corpus), "--doc-cap", str(doc_cap),
        "--vocab-size", str(vocab_size), "--out", str(container),
    ]
    subprocess.run(command, check=True)
    ranks = data.mergeable_ranks(data.read_nctoken1(container))
    return [token for token, _ in sorted(ranks.items(), key=lambda i: i[1])]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.work_dir / "p4_corpus.txt"
    container = args.work_dir / "p4_native.nctoken"

    docs, chars = build_corpus(args.parquet_dir, args.max_chars, corpus)
    print(f"corpus: {docs} documents, {chars} characters -> {corpus}")

    reference = reference_tokens(corpus, args.vocab_size, args.doc_cap)
    native = native_tokens(args.binary, corpus, args.vocab_size, args.doc_cap,
                           container, "t2-parity")

    print(f"reference tokens {len(reference)}  native tokens {len(native)}")
    if len(reference) != len(native):
        print("FAIL: token count differs", file=sys.stderr)
        return 1
    mismatches = [i for i in range(len(reference)) if reference[i] != native[i]]
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches; first at {mismatches[0]}",
              file=sys.stderr)
        return 1
    print(f"PASS: {len(native)} base tokens match the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
