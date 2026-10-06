#!/usr/bin/env python3
"""Generate the chat prompt fixture, ``tests/data/chat_fixture.bin``.

The fixture pins the completion prompt that the reference
``nanochat.tokenizer.Tokenizer.render_for_completion`` produces for a few
synthetic conversations. The test ``python/tests/chat_test.py`` renders the
same conversations with :class:`nanochat_cpp.chat._PromptRenderer` and compares
the ids. See docs/python.md sections 5 and 11.

The file is a small container::

    header : magic[8] = b"NANCHAT1", uint32 version
    body   : UTF-8 JSON

The generation needs the reference package and ``tiktoken`` only:

    /path/to/reference/.venv/bin/python tools/dump_chat_fixture.py \
        --out tests/data/chat_fixture.bin
"""

from __future__ import annotations

import argparse
import json
import os
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


def _find_reference_repo() -> Path:
    candidates = []
    if os.environ.get("NANOCHAT_REPO"):
        candidates.append(Path(os.environ["NANOCHAT_REPO"]))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root.parent / "nanochat")
    candidates.append(Path.home() / "repos" / "nanochat")
    for candidate in candidates:
        if (candidate / "nanochat" / "tokenizer.py").is_file():
            return candidate.resolve()
    raise SystemExit(
        "cannot find the reference nanochat checkout; set NANOCHAT_REPO")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/data/chat_fixture.bin")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "python"))
    reference = _find_reference_repo()
    sys.path.insert(0, str(reference))

    from nanochat.tokenizer import RustBPETokenizer  # noqa: E402
    from nanochat_cpp import data  # noqa: E402

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
