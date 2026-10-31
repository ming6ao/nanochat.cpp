#!/usr/bin/env python3
"""Shared helpers for the offline fixture generators.

The merged generator ``tools/dump_fixtures.py`` keeps one dispatch and one
usage line for every fixture kind (``docs/script-consolidation-plan.md``
section 7, P4). This module holds the code that the kinds used to repeat:
the subcommand selection, the ``--model`` selector of
``docs/model-and-device-bringup.md`` section 12.2, and the small value
helpers.

The module imports no third-party package at import time. Each helper imports
the reference package it needs, so a kind that does not touch PyTorch stays
fast. The ``--model`` path helpers use only ``hashlib``, ``importlib.util``,
``json``, ``pathlib``, and ``sys``, so the manifest-backed ``--check``
fallback runs in a tree with no torch installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

__all__ = [
    "COMMANDS",
    "USAGE",
    "DEFAULT_MODEL",
    "MODEL_FIXTURES",
    "REPO_ROOT",
    "add_check_argument",
    "add_model_argument",
    "as_f32",
    "as_i32",
    "build_config",
    "check_committed_manifest",
    "check_generated",
    "compare_fixture_bytes",
    "model_fixtures_dir",
    "model_from_argv",
    "model_spec_path",
    "read_model_spec",
    "resolve_out",
    "scalar",
    "selected_command",
    "torch_available",
]

#: Every fixture subcommand, in help order.
COMMANDS = (
    "oracle",
    "train",
    "rl",
    "generate",
    "eval",
    "plan",
    "tokenizer",
    "loader",
    "chat",
    "api",
    "numerics",
    "checkpoint",
)

#: The usage line for the merged generator. The ``--model`` and ``--check``
#: options are shown, because they are the P4 surface
#: (``docs/model-and-device-bringup.md`` sections 12.2 and 16).
USAGE = (
    "usage: dump_fixtures.py {"
    + "|".join(COMMANDS)
    + "} [--model NAME] [--check] [options]"
)

#: The repository root, derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The model package selected when ``--model`` is absent.
DEFAULT_MODEL = "nanochat-default"

#: The fixture files owned by a model package, by the generator kind that
#: produces them. A kind in this mapping is *model-scoped*: its default output
#: is ``models/<name>/fixtures/<file>`` and ``--check`` can compare its bytes
#: with the committed fixture. The tokenizer, loader, chat, plan, API, and
#: checkpoint fixtures are harness data, not model definitions, so they stay
#: in ``tests/data/`` (``models/nanochat-default/README.md``).
MODEL_FIXTURES = {
    "oracle": ("debug_state.bin",),
    "train": ("train_parity.bin",),
    "rl": ("rl_parity.bin",),
    "generate": ("generate_parity.bin",),
    "eval": ("core_eval.bin",),
    "numerics": ("numerics_golden_10l.bin", "numerics_golden_10l_fp8.bin"),
}


def selected_command(argv: list[str]) -> str:
    """The fixture kind from ``argv``, or ``SystemExit`` with the usage line."""
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        raise SystemExit(0 if argv else 2)
    command = argv[0]
    if command not in COMMANDS:
        print("unknown fixture kind: " + command, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    return command


def as_f32(tensor):
    """The tensor as a contiguous fp32 NumPy array."""
    import torch
    return tensor.detach().to(torch.float32).cpu().numpy()


def as_i32(tensor):
    """The tensor as a contiguous int32 NumPy array."""
    import torch
    return tensor.detach().to(torch.int32).cpu().numpy()


def scalar(value: float):
    """One float as a one-element fp32 NumPy array."""
    import numpy as np
    return np.asarray([value], dtype=np.float32)


def build_config(args, config_class):
    """The reference ``GPTConfig`` from the parsed arguments."""
    return config_class(
        sequence_len=args.seq,
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
        window_pattern=args.window,
    )


# --- the --model selector ---------------------------------------------------


def model_spec_path(name: str) -> Path:
    """The spec path for ``name``, under the repository's ``models/`` axis."""
    return REPO_ROOT / "models" / name / "spec.json"


def read_model_spec(name: str) -> dict:
    """Read ``models/<name>/spec.json`` and check its declared identity.

    The model axis is data (``docs/model-and-device-bringup.md`` section 8), so
    a misspelled ``--model`` is a missing directory, not a guess. The returned
    mapping is the parsed spec; callers that need only the path use
    :func:`model_spec_path`.
    """
    path = model_spec_path(name)
    print(f"read  {path}", file=sys.stderr)
    if not path.is_file():
        raise SystemExit(
            f"unknown model {name!r}: {path} is missing "
            "(the --model selector names a directory under models/)")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: invalid JSON: {error.msg}") from error
    declared = spec.get("name")
    if declared != name:
        raise SystemExit(
            f"{path}: spec declares name {declared!r}, not {name!r}")
    return spec


def model_fixtures_dir(name: str) -> Path:
    """The committed fixture directory of ``name``, after validating its spec."""
    read_model_spec(name)
    return REPO_ROOT / "models" / name / "fixtures"


def add_model_argument(parser) -> None:
    """Add the shared ``--model NAME`` option to one fixture kind's parser."""
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=("model package under models/ whose spec.json and fixtures are "
              f"used (default: {DEFAULT_MODEL})"))


def add_check_argument(parser) -> None:
    """Add ``--check``: regenerate to a temporary directory and compare bytes."""
    parser.add_argument(
        "--check",
        action="store_true",
        help=("regenerate into a temporary directory and compare the bytes "
              "with the committed model fixture; never writes into models/"))


def model_from_argv(argv: list[str]) -> str:
    """Read ``--model`` from raw arguments before argparse runs.

    The merged generator needs the model name before a kind imports torch, so
    the ``--check`` fallback can verify the committed manifest in a tree where
    torch is not installed.
    """
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--model":
            if index + 1 >= len(argv):
                raise SystemExit("--model needs a name")
            return argv[index + 1]
        if argument.startswith("--model="):
            return argument.split("=", 1)[1]
        index += 1
    return DEFAULT_MODEL


def resolve_out(args, filename: str) -> Path:
    """The output path for a model-scoped kind.

    An explicit ``--out`` wins. Otherwise the output is
    ``models/<args.model>/fixtures/<filename>``. Reading the spec here is the
    validation: a model that does not exist fails before any generation runs.
    """
    explicit = getattr(args, "out", None)
    fixtures_dir = model_fixtures_dir(getattr(args, "model", DEFAULT_MODEL))
    if explicit:
        return Path(explicit)
    return fixtures_dir / filename


# --- --check: byte identity against the committed fixture -------------------


def torch_available() -> bool:
    """Whether the reference PyTorch/numpy pair is importable.

    The import is not attempted: a probe keeps the check fallback fast and
    avoids side effects on a host that may have a partially installed torch.
    """
    try:
        return (importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("numpy") is not None)
    except (ImportError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fnv1a64(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "0x%016x" % value


def _manifest_entries(manifest: dict) -> dict[str, dict]:
    return {Path(entry["file"]).name: entry for entry in manifest["fixtures"]}


def check_committed_manifest(kind: str, model: str) -> int:
    """Verify a committed model fixture against its manifest, without torch.

    ``models/<name>/fixtures/manifest.json`` records the size, SHA-256, and
    FNV-1a 64 of every committed fixture. This is the honest fallback for a
    tree with no torch: the generator cannot re-derive the bytes here, so the
    check proves the committed file still matches the manifest that pins it,
    and says so. A generator run in a torch-equipped environment performs the
    stronger byte comparison in :func:`check_generated`.
    """
    committed_dir = model_fixtures_dir(model)
    manifest_path = committed_dir / "manifest.json"
    print(f"read  {manifest_path}", file=sys.stderr)
    if not manifest_path.is_file():
        print(f"check: missing manifest {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != model:
        print(f"check: {manifest_path}: model field is "
              f"{manifest.get('model')!r}, not {model!r}", file=sys.stderr)
        return 1
    entries = _manifest_entries(manifest)
    failed = 0
    checked = 0
    for filename in MODEL_FIXTURES[kind]:
        entry = entries.get(filename)
        path = committed_dir / filename
        print(f"read  {path}", file=sys.stderr)
        if entry is None:
            print(f"check: {manifest_path}: no manifest entry for {filename}",
                  file=sys.stderr)
            failed += 1
            continue
        if not path.is_file():
            print(f"check: missing committed fixture {path}", file=sys.stderr)
            failed += 1
            continue
        data = path.read_bytes()
        problems = []
        if "size_bytes" in entry and len(data) != int(entry["size_bytes"]):
            problems.append(f"size {len(data)} != manifest "
                            f"{entry['size_bytes']}")
        if "sha256" in entry:
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"]:
                problems.append(f"sha256 {digest} != manifest {entry['sha256']}")
        if "fnv1a64" in entry:
            digest = _fnv1a64(data)
            if digest != entry["fnv1a64"]:
                problems.append(f"fnv1a64 {digest} != manifest "
                                f"{entry['fnv1a64']}")
        if problems:
            print(f"check: {path}: " + "; ".join(problems), file=sys.stderr)
            failed += 1
        else:
            checked += 1
    if failed:
        print(f"check: {failed} committed fixture(s) disagree with the manifest",
              file=sys.stderr)
        return 1
    print(f"check: torch/numpy are not installed; verified {checked} committed "
          f"fixture(s) for model {model!r} against "
          f"{manifest_path.name} instead of re-deriving them. Run the same "
          "command in the reference environment for the byte-identity check.",
          file=sys.stderr)
    return 0


def compare_fixture_bytes(generated: Path, committed: Path) -> bool:
    """Compare two fixture files byte for byte and report the first difference."""
    print(f"read  {generated}", file=sys.stderr)
    print(f"read  {committed}", file=sys.stderr)
    if not committed.is_file():
        print(f"check: missing committed fixture {committed}", file=sys.stderr)
        return False
    generated_bytes = generated.read_bytes()
    committed_bytes = committed.read_bytes()
    if generated_bytes == committed_bytes:
        print(f"check: byte-identical to {committed} "
              f"({len(committed_bytes)} bytes, sha256 "
              f"{hashlib.sha256(committed_bytes).hexdigest()})",
              file=sys.stderr)
        return True
    limit = min(len(generated_bytes), len(committed_bytes))
    offset = next((index for index in range(limit)
                   if generated_bytes[index] != committed_bytes[index]), limit)
    print(f"check: {generated} differs from {committed} at byte {offset} "
          f"(generated {len(generated_bytes)} bytes, committed "
          f"{len(committed_bytes)} bytes)", file=sys.stderr)
    return False


def check_generated(kind: str, args, generate) -> int:
    """Run ``generate`` into a temporary directory and compare its bytes.

    ``generate`` is the kind's own generation entry point. It receives ``args``
    with ``args.out`` pointing inside a temporary directory, so the check never
    writes into ``models/`` or ``tests/``.
    """
    import tempfile

    committed_dir = model_fixtures_dir(args.model)
    filename = Path(getattr(args, "out", "")).name
    if filename not in MODEL_FIXTURES[kind]:
        # A caller may pass an explicit --out with a non-canonical name; use
        # the first committed fixture of the kind as the comparison target and
        # the requested basename for the generated file.
        filename = MODEL_FIXTURES[kind][0]
    committed = committed_dir / filename
    with tempfile.TemporaryDirectory(prefix=f"nanochat-{kind}-check-") as tmp:
        args.out = str(Path(tmp) / filename)
        print(f"write {args.out}", file=sys.stderr)
        generated = Path(generate(args))
        if not compare_fixture_bytes(generated, committed):
            return 1
    return 0
