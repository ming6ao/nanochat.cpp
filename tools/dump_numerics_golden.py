#!/usr/bin/env python3
"""Generate the staged numeric golden trace, ``tests/data/numerics_golden_10l.bin``.

The trace of ``docs/numerics-integration.md`` is one flat list of named records
in the little-endian container that ``tests/oracle_fixture.h`` reads, under the
magic ``NANONUM1``. The C++ emitter owns the exact bytes, because it computes
the values; this script builds the emitter, runs it once through
``tools/nanochat`` under the ``t0-cpu`` sandbox profile, verifies the container,
and moves the file into place. The emitter pins the host thread count to one, so
the file does not depend on the sandbox thread budget.

The fixture is the fp32 CPU reference of section 6.2: 10 layers, 6 heads, 3 KV
heads, hidden dimension 384, sequence length 256, batch 4, vocabulary 512, the
window pattern ``SSSL``, value embeddings on, seed 42, and five optimizer steps.
The trace holds the training trajectory (loss, gradient norm, and the L2 norm
and byte hash of every parameter and gradient at every step), the evaluation
section (bits per byte, per-batch loss, logits hash, greedy argmax identifiers),
the greedy generation section (identifiers, mask, and logit margins), the SFT
arithmetic, and the RL arithmetic.

Run the generator from the repository root::

    python3 tools/dump_numerics_golden.py

Then read the file with ``tests/numerics_trace_test.cc`` (the CPU and simulator
legs compare byte for byte; the GPU leg compares within the tolerance table) or
with ``read_records`` below.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# The repository root holds the `tools/` entry point and the emitter binary.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# The fixture magic and version of the container.
MAGIC = b"NANONUM1"
VERSION = 1

#: One element size per dtype code (`tests/oracle_fixture.h`).
_ELEM_SIZE = {0: 4, 1: 4, 2: 8, 3: 8, 4: 1}

DEFAULT_OUT = "tests/data/numerics_golden_10l.bin"
DEFAULT_TOKENIZER = "tests/data/loader_tokenizer.nctoken"
EMIT_TARGET = "//tests:numerics_trace_emit"
EMIT_BINARY = "bazel-bin/tests/numerics_trace_emit"


def read_records(path: Path) -> list[tuple[str, int, tuple, bytes]]:
    """Reads the flat record container: name, dtype, shape, payload."""
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic {data[:8]!r}")
    version, count = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise SystemExit(f"{path}: unsupported version {version}")
    offset = 16
    records = []
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", data, offset)
        offset += 2
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        dtype = data[offset]
        ndim = data[offset + 1]
        offset += 2
        shape = struct.unpack_from("<" + "q" * ndim, data, offset) if ndim else ()
        offset += 8 * ndim
        numel = 1
        for extent in shape:
            numel *= extent
        nbytes = numel * _ELEM_SIZE[dtype]
        payload = data[offset:offset + nbytes]
        offset += nbytes
        records.append((name, dtype, shape, payload))
    if offset != len(data):
        raise SystemExit(f"{path}: {len(data) - offset} trailing bytes")
    return records


def scalar(values: dict[str, tuple], name: str) -> float:
    dtype, _shape, payload = values[name]
    if dtype == 0:
        return struct.unpack("<f", payload)[0]
    if dtype == 1:
        return struct.unpack("<i", payload)[0]
    if dtype == 3:
        return struct.unpack("<q", payload)[0]
    raise SystemExit(f"{name}: not a scalar record (dtype {dtype})")


def run_nanochat(repo_root: Path, arguments) -> subprocess.CompletedProcess:
    """Runs one command through the single entry point and captures its output."""
    command = [str(repo_root / "tools" / "nanochat"), *arguments]
    result = subprocess.run(command, cwd=str(repo_root), capture_output=True,
                            text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"command failed ({result.returncode}): " + " ".join(command))
    return result


def summarize(records, out: Path) -> None:
    """Prints the fixture summary and the checks a reviewer needs."""
    values = {name: (dtype, shape, payload)
              for name, dtype, shape, payload in records}
    backend = values["header/backend"][2].decode()
    precision = values["header/precision"][2].decode()
    steps = int(scalar(values, "header/steps"))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(records)} records, "
          f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]})")
    print(f"identity: backend {backend}, precision {precision}, "
          f"seed {int(scalar(values, 'header/seed'))}, "
          f"parameters {int(scalar(values, 'config/parameters'))}")
    losses = [scalar(values, f"train/step{step}/loss") for step in range(steps + 1)]
    print("train loss: " + ", ".join(f"{loss:.6f}" for loss in losses))
    print("train gradient norm: %.6f" % scalar(values, "train/step1/grad_norm"))
    print("eval: bpb %.6f, loss %.6f, min logit margin %.6g"
          % (scalar(values, "eval/bpb"), scalar(values, "eval/loss/0"),
             scalar(values, "eval/margin_min")))
    print("generation: min logit margin %.6g"
          % scalar(values, "gen/margin_min"))
    print("sft: loss %.6f, gradient norm %.6f, valid targets %d"
          % (scalar(values, "sft/loss"), scalar(values, "sft/grad_norm"),
             int(scalar(values, "sft/valid_targets"))))
    print("rl: loss %.6g, gradient norm %.6g, mean reward %.6f, "
          "min logit margin %.6g"
          % (scalar(values, "rl/loss"), scalar(values, "rl/grad_norm"),
             scalar(values, "rl/mean_reward"), scalar(values, "rl/margin_min")))
    for row in range(4):
        print("rl row %d: reward %.6f, advantage %.6g"
              % (row, scalar(values, f"rl/reward/{row}"),
                 scalar(values, f"rl/advantage/{row}")))
    if backend != "cpu" or precision != "fp32":
        raise SystemExit("the golden trace must come from the fp32 CPU build")


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output fixture path")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="the committed NCTOKEN1 artifact of the eval phase")
    parser.add_argument("--no-build", action="store_true",
                        help="use the current bazel-bin emitter")
    return parser.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    # The fixture is the CPU reference. Keep the shared GPU free.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = _REPO_ROOT
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out
    tokenizer = Path(args.tokenizer)
    if not tokenizer.is_absolute():
        tokenizer = repo_root / tokenizer
    if not tokenizer.is_file():
        raise SystemExit(f"missing tokenizer file: {tokenizer}")
    if not args.no_build:
        run_nanochat(repo_root, ["build", EMIT_TARGET])
    emitter = repo_root / EMIT_BINARY
    if not emitter.is_file():
        raise SystemExit(f"missing emitter {emitter}; run "
                         f"tools/nanochat build {EMIT_TARGET}")

    with tempfile.TemporaryDirectory(prefix="numerics_golden_") as workdir:
        staged = Path(workdir) / "numerics_golden_10l.bin"
        result = run_nanochat(repo_root, [
            "run", "t0-cpu", "--", str(emitter),
            "--out", str(staged),
            "--tokenizer", str(tokenizer),
        ])
        sys.stdout.write(result.stdout)
        records = read_records(staged)
        out.parent.mkdir(parents=True, exist_ok=True)
        # The temporary directory may live on another filesystem, so stage
        # beside the target and replace inside one directory.
        beside = out.parent / (out.name + ".tmp")
        beside.write_bytes(staged.read_bytes())
        os.replace(beside, out)

    # Re-read the committed path, so the summary describes the file on disk.
    summarize(read_records(out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
