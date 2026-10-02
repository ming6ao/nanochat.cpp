"""Torch-free self-test for the bridge's pure logic.

Run with ``tools/nanochat_cpp selftest`` (or
``PYTHONPATH=python python3 -m nanochat_cpp.selftest``). It checks the
configuration math, the token-count estimate, and the NANO shard round-trip
without importing torch or the reference package, so it runs anywhere.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import config, data


def _check(condition: bool, message: str) -> int:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    failures = 0

    # model_shape mirrors build_model_meta: ceil(depth*aspect/head_dim)*head_dim.
    cases = [
        (2, 128, 1),
        (6, 384, 3),
        (8, 512, 4),
        (20, 1280, 10),
    ]
    for depth, expected_dim, expected_heads in cases:
        dim, heads = config.model_shape(depth, 64, 128)
        failures += _check(dim == expected_dim and heads == expected_heads,
                           f"model_shape({depth}) = {dim}/{heads}")

    failures += _check(config.padded_vocab(32768) == 32768,
                       "padded_vocab(32768)")
    failures += _check(config.padded_vocab(32769) == 32832,
                       "padded_vocab(32769)")
    failures += _check(config.padded_vocab(1) == 64, "padded_vocab(1)")

    failures += _check(
        data.tokens_for_run(4096, 50, 512) == 4096 * 50 + 512 + 2048,
        "tokens_for_run")

    # NANO shard round-trip, both widths.
    with tempfile.TemporaryDirectory() as tmp:
        for width, limit in ((2, 65535), (4, 1 << 20)):
            path = Path(tmp) / f"shard{width}.bin"
            tokens = [i % limit for i in range(1000)]
            with data.ShardWriter(path, width) as writer:
                writer.write(tokens[:400])
                writer.write(tokens[400:])
            count, got_width, got = data.read_shard(path)
            failures += _check(count == len(tokens), f"shard{width} count")
            failures += _check(got_width == width, f"shard{width} width")
            failures += _check(got == tokens, f"shard{width} tokens")

    if failures:
        print(f"selftest: {failures} failure(s)", file=sys.stderr)
        return 1
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
