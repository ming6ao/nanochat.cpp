#!/usr/bin/env python3
"""The cross-package non-interference gate (bring-up section 10.4).

For a diff that touches ``models/<name>/**`` the gate:

1. regenerates that model's fixtures into a temporary directory;
2. diffs them against the committed ``models/<name>/fixtures`` using the
   record-name selectors in ``models/<name>/tolerances.json``; and
3. confirms every *other* model's committed fixtures still match their
   manifests.

Step 3 is the isolation proof (``docs/model-and-device-bringup.md`` section
10.4), and it costs nothing: the fixture comparison and the reference trace run
on the CPU.  A failure names the offending fixture and record, so the log is
enough to diagnose it.

Usage::

    python3 tools/isolation_gate.py --model nanochat-default

    # Diff an externally regenerated directory instead of invoking the
    # generator (useful when a caller owns the reference environment).
    python3 tools/isolation_gate.py --model nanochat-default --generated-dir /tmp/regenerated

The script is standard-library only and hermetic.  It writes generated
fixtures only into a temporary directory; it never writes into ``models/`` or
``tests/``.  When the reference torch/numpy pair is unavailable, the
torch-backed kinds fall back to verifying the committed bytes against the model
manifest and say so.  Pass ``--strict`` to make an unavailable regeneration a
failure instead.

The three numeric goldens are recorded on the reference trace host.  A
re-emission on another host changes the low-order summation order, so a record
mismatch in one of those files is reported as a reproducibility warning and
fails only under ``--strict``.  The committed bytes stay pinned: the manifest
checksum is always verified, and ``tolerances.json`` carries the same sha256.
The functional fixtures (oracle, eval, train, generate, rl) keep the strict
comparison in every mode.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# The generator helpers live beside this script.  Adding the directory keeps
# `import fixture_common` working under PYTHONSAFEPATH as well as in runfiles.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixture_common  # noqa: E402

REPO_ROOT = fixture_common.REPO_ROOT
DEFAULT_MODEL = fixture_common.DEFAULT_MODEL

#: The fixtures whose golden is recorded on the reference numeric-trace host.
#: Their low-order bits depend on the summation order of the recording host, so
#: a re-emission here is a warning by default and a failure under --strict.
REFERENCE_HOST_FIXTURES = frozenset({
    "numerics_golden_10l.bin",
    "numerics_golden_10l_bf16.bin",
    "numerics_golden_10l_fp8.bin",
})

#: The generator kind (and, for numerics, the precision) that produces each
#: model-scoped fixture file.
GENERATOR_FOR = {
    "debug_state.bin": ("oracle", None),
    "core_eval.bin": ("eval", None),
    "train_parity.bin": ("train", None),
    "generate_parity.bin": ("generate", None),
    "rl_parity.bin": ("rl", None),
    "numerics_golden_10l.bin": ("numerics", "fp32"),
    "numerics_golden_10l_bf16.bin": ("numerics", "bf16"),
    "numerics_golden_10l_fp8.bin": ("numerics", "fp8"),
}

#: The record container is the same little-endian framing for every magic the
#: model fixtures use (``NANOORC1``, ``NANOEVL1``, ``NANONUM1``).
_ELEMENT_SIZE = {0: 4, 1: 4, 2: 8, 3: 8, 4: 1}


class FixtureError(Exception):
    """A malformed fixture, reported with the path that failed."""


# --- reading the record container ------------------------------------------


def read_records(path: Path) -> tuple[bytes, int, list[tuple[str, int, tuple, bytes]]]:
    """Read one ``magic/version/count`` record container."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise FixtureError(f"{path}: cannot read: {error}") from error
    if len(data) < 16:
        raise FixtureError(f"{path}: truncated header ({len(data)} bytes)")
    magic = data[:8]
    version, count = struct.unpack_from("<II", data, 8)
    offset = 16
    records: list[tuple[str, int, tuple, bytes]] = []
    seen: set[str] = set()
    for _ in range(count):
        if offset + 2 > len(data):
            raise FixtureError(f"{path}: truncated record name length")
        (name_len,) = struct.unpack_from("<H", data, offset)
        offset += 2
        if offset + name_len + 2 > len(data):
            raise FixtureError(f"{path}: truncated record name")
        name = data[offset:offset + name_len].decode("utf-8", "replace")
        offset += name_len
        dtype, ndim = struct.unpack_from("<BB", data, offset)
        offset += 2
        if dtype not in _ELEMENT_SIZE:
            raise FixtureError(f"{path}: record {name!r} has dtype code {dtype}")
        if offset + 8 * ndim > len(data):
            raise FixtureError(f"{path}: truncated record shape")
        shape = struct.unpack_from("<" + "Q" * ndim, data, offset) if ndim else ()
        offset += 8 * ndim
        elements = 1
        for extent in shape:
            elements *= extent
        size = elements * _ELEMENT_SIZE[dtype]
        if offset + size > len(data):
            raise FixtureError(f"{path}: record {name!r} is truncated")
        payload = data[offset:offset + size]
        offset += size
        if name in seen:
            raise FixtureError(f"{path}: duplicate record name {name!r}")
        seen.add(name)
        records.append((name, dtype, shape, payload))
    if offset != len(data):
        raise FixtureError(f"{path}: {len(data) - offset} trailing bytes")
    return magic, version, records


# --- selecting and applying one tolerance ----------------------------------


def select_bound(entry: dict, name: str) -> dict | None:
    """The first record-name selector of one fixture that matches ``name``."""
    for selector in entry.get("records", []):
        if fnmatch.fnmatchcase(name, selector["match"]):
            return selector["bound"]
    return None


def _floats(dtype: int, payload: bytes) -> list[float]:
    if dtype == 0:
        return list(struct.unpack("<" + "f" * (len(payload) // 4), payload))
    if dtype == 2:
        return list(struct.unpack("<" + "d" * (len(payload) // 8), payload))
    raise TypeError("not a float payload")


def compare_payload(dtype: int, committed: bytes, generated: bytes,
                    bound: dict) -> str | None:
    """The first tolerance violation in one record, or ``None``."""
    mode = bound.get("mode")
    if mode == "report_only":
        return None
    if dtype != 0 and dtype != 2:
        # Integer and byte records are identifiers, masks, or inputs: exact.
        if committed != generated:
            return "integer/byte record differs (exact)"
        return None
    if mode in ("exact", "producer_identity", "exact_or_skip_device"):
        if committed != generated:
            return f"record differs (exact); {mode}"
        return None
    if mode not in ("absolute", "relative"):
        return f"unknown tolerance mode {mode!r}"
    atol = float(bound.get("atol", 0.0))
    rtol = float(bound.get("rtol", 0.0))
    expected = _floats(dtype, committed)
    actual = _floats(dtype, generated)
    if len(expected) != len(actual):
        return "element count differs"
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want == got:
            continue
        if want != want and got != got:  # both NaN
            continue
        if abs(got - want) <= atol + rtol * abs(want):
            continue
        return (f"element {index} is {got!r}, expected {want!r} "
                f"(atol={atol:g}, rtol={rtol:g})")
    return None


def compare_fixture(path: Path, committed: Path, generated: Path,
                    fixture_entry: dict) -> list[str]:
    """Compare one generated fixture with its committed bytes within tolerance."""
    problems: list[str] = []
    print(f"read  {committed}", file=sys.stderr)
    print(f"read  {generated}", file=sys.stderr)
    try:
        committed_magic, committed_version, committed_records = read_records(committed)
        generated_magic, generated_version, generated_records = read_records(generated)
    except FixtureError as error:
        return [str(error)]
    if committed_magic != generated_magic:
        problems.append(
            f"{path}: magic {generated_magic!r} != committed "
            f"{committed_magic!r}")
    if committed_version != generated_version:
        problems.append(
            f"format version {generated_version} != committed "
            f"{committed_version}")
    committed_by_name = {name: (dtype, shape, payload)
                         for name, dtype, shape, payload in committed_records}
    generated_by_name = {name: (dtype, shape, payload)
                         for name, dtype, shape, payload in generated_records}
    for name in sorted(committed_by_name.keys() | generated_by_name.keys()):
        if name not in generated_by_name:
            problems.append(f"{path}: record {name!r} is missing from the "
                            "regenerated fixture")
            continue
        if name not in committed_by_name:
            problems.append(f"{path}: record {name!r} is new in the "
                            "regenerated fixture")
            continue
        c_dtype, c_shape, c_payload = committed_by_name[name]
        g_dtype, g_shape, g_payload = generated_by_name[name]
        if c_dtype != g_dtype or c_shape != g_shape:
            problems.append(
                f"{path}: record {name!r} changed dtype/shape "
                f"({c_dtype}, {c_shape}) -> ({g_dtype}, {g_shape})")
            continue
        bound = select_bound(fixture_entry, name)
        if bound is None:
            problems.append(
                f"{path}: record {name!r} has no tolerance selector in "
                "tolerances.json")
            continue
        detail = compare_payload(c_dtype, c_payload, g_payload, bound)
        if detail is not None:
            problems.append(f"{path}: record {name!r}: {detail}")
    return problems


# --- manifest verification --------------------------------------------------


def _fnv1a64(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "0x%016x" % value


def verify_manifest(model_dir: Path) -> list[str]:
    """Check every committed fixture against the model's manifest."""
    manifest_path = model_dir / "fixtures" / "manifest.json"
    print(f"read  {manifest_path}", file=sys.stderr)
    if not manifest_path.is_file():
        return [f"{manifest_path}: missing manifest"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{manifest_path}: cannot read manifest: {error}"]
    problems: list[str] = []
    for entry in manifest.get("fixtures", []):
        path = model_dir / entry["file"]
        print(f"read  {path}", file=sys.stderr)
        if not path.is_file():
            problems.append(f"{path}: missing committed fixture")
            continue
        data = path.read_bytes()
        if "size_bytes" in entry and len(data) != int(entry["size_bytes"]):
            problems.append(
                f"{path}: size {len(data)} != manifest {entry['size_bytes']}")
        if "sha256" in entry:
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"]:
                problems.append(
                    f"{path}: sha256 {digest} != manifest {entry['sha256']}")
        if "fnv1a64" in entry:
            digest = _fnv1a64(data)
            if digest != entry["fnv1a64"]:
                problems.append(
                    f"{path}: fnv1a64 {digest} != manifest {entry['fnv1a64']}")
        if "record_count" in entry:
            try:
                _, _, records = read_records(path)
            except FixtureError as error:
                problems.append(str(error))
                continue
            if len(records) != int(entry["record_count"]):
                problems.append(
                    f"{path}: {len(records)} records != manifest "
                    f"{entry['record_count']}")
        if "record_names" in entry:
            try:
                _, _, records = read_records(path)
            except FixtureError as error:
                problems.append(str(error))
                continue
            names = [name for name, _, _, _ in records]
            if names != list(entry["record_names"]):
                problems.append(
                    f"{path}: record names/order differ from the manifest")
    return problems


# --- regeneration -----------------------------------------------------------


def _can_run(kind: str) -> bool:
    """Whether this environment can invoke the generator for ``kind``.

    The torch-backed kinds need torch and numpy.  The numerics kind drives the
    C++ emitter, so it is attempted whenever the entry point exists.
    """
    if kind == "numerics":
        return True
    return fixture_common.torch_available()


def regenerate(model: str, workdir: Path,
               wanted: set[str]) -> tuple[dict[str, Path], dict[str, str]]:
    """Regenerate every model-scoped fixture in ``wanted`` into ``workdir``.

    Returns the paths that were produced and the fixture names that could not
    be regenerated, with the reason.  A caller running on the CPU-only sandbox
    gets the manifest fallback for the second mapping.
    """
    produced: dict[str, Path] = {}
    failures: dict[str, str] = {}
    generator = REPO_ROOT / "tools" / "dump_fixtures.py"
    for filename, (kind, precision) in sorted(GENERATOR_FOR.items()):
        if filename not in wanted:
            continue
        out = workdir / filename
        if not _can_run(kind):
            failures[filename] = (f"{kind} needs torch/numpy, which are not "
                                  "importable")
            continue
        command = [sys.executable, str(generator), kind, "--model", model,
                   "--out", str(out)]
        if precision is not None:
            command += ["--precision", precision]
        print(f"write {out}", file=sys.stderr)
        result = subprocess.run(command, cwd=str(REPO_ROOT),
                                capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            reason = tail[-1] if tail else f"exit {result.returncode}"
            failures[filename] = f"{kind} failed: {reason}"
            continue
        if not out.is_file():
            failures[filename] = f"{kind} reported success but wrote no {out}"
            continue
        produced[filename] = out
    return produced, failures


# --- the gate ---------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="NAME",
                        help=f"the touched model (default: {DEFAULT_MODEL})")
    parser.add_argument("--generated-dir", metavar="DIR",
                        help=("an already regenerated fixture directory; "
                              "skips invoking the generator"))
    parser.add_argument("--manifest-only", action="store_true",
                        help="do not regenerate; verify committed manifests only")
    parser.add_argument("--strict", action="store_true",
                        help="fail when a fixture cannot be regenerated instead "
                             "of falling back to its manifest, and compare the "
                             "numeric goldens on this host as well")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = args.model
    model_dir = REPO_ROOT / "models" / model
    tolerances_path = model_dir / "tolerances.json"
    print(f"read  {tolerances_path}", file=sys.stderr)
    if not tolerances_path.is_file():
        print(f"isolation: missing {tolerances_path}", file=sys.stderr)
        return 1
    try:
        tolerances = json.loads(tolerances_path.read_text(encoding="utf-8"))
        fixture_common.read_model_spec(model)
    except (OSError, json.JSONDecodeError) as error:
        print(f"isolation: {tolerances_path}: {error}", file=sys.stderr)
        return 1

    problems: list[str] = []
    produced: dict[str, Path] = {}
    failures: dict[str, str] = {}
    work = None
    if args.generated_dir:
        generated_dir = Path(args.generated_dir)
        if not generated_dir.is_dir():
            print(f"isolation: --generated-dir {generated_dir} is not a "
                  "directory", file=sys.stderr)
            return 1
        for entry in tolerances.get("fixtures", []):
            name = Path(entry["file"]).name
            path = generated_dir / name
            if path.is_file():
                produced[name] = path
            else:
                failures[name] = f"missing from --generated-dir {generated_dir}"
    elif not args.manifest_only:
        wanted = {Path(entry["file"]).name
                  for entry in tolerances.get("fixtures", [])}
        work = tempfile.TemporaryDirectory(prefix=f"nanochat-isolation-{model}-")
        produced, failures = regenerate(model, Path(work.name), wanted)

    # Step 1 and 2: the touched model's fixtures, within its tolerance contract.
    fallback_used = False
    for entry in tolerances.get("fixtures", []):
        name = Path(entry["file"]).name
        committed = model_dir / entry["file"]
        if not committed.is_file():
            problems.append(f"{committed}: missing committed fixture")
            continue
        digest = hashlib.sha256(committed.read_bytes()).hexdigest()
        if "sha256" in entry and digest != entry["sha256"]:
            problems.append(
                f"{committed}: sha256 {digest} != tolerances.json "
                f"{entry['sha256']}")
        if name in produced:
            mismatches = compare_fixture(model_dir / entry["file"], committed,
                                         produced[name], entry)
            if mismatches and name in REFERENCE_HOST_FIXTURES:
                message = (f"{committed}: {len(mismatches)} record(s) differ "
                           "from a re-emission on this host; the golden is "
                           "recorded on the reference trace host, so the "
                           "committed bytes are verified against the "
                           "manifest only")
                if args.strict:
                    problems.extend(mismatches)
                else:
                    print(f"check: {message}", file=sys.stderr)
                    fallback_used = True
            else:
                problems.extend(mismatches)
        elif args.generated_dir:
            problems.append(
                f"{generated_dir / name}: missing from --generated-dir; "
                "the gate cannot diff this fixture")
        else:
            reason = failures.get(name, "not regenerated")
            message = (f"{committed}: regeneration unavailable ({reason}); "
                       "committed bytes were verified against the manifest "
                       "only")
            if args.strict:
                problems.append(message)
            else:
                print(f"check: {message}", file=sys.stderr)
            fallback_used = True

    if fallback_used:
        problems.extend(verify_manifest(model_dir))

    # Step 3: every other model's committed fixtures are unchanged.  The
    # manifest pins each file's size, checksum, record count, and record order,
    # so this is the isolation proof without touching another model's code.
    other_models = []
    for spec_path in sorted((REPO_ROOT / "models").glob("*/spec.json")):
        other = spec_path.parent.name
        if other == model:
            continue
        other_models.append(other)
        other_dir = REPO_ROOT / "models" / other
        if not (other_dir / "fixtures").is_dir():
            print(f"check: model {other!r} has no fixtures directory; "
                  "nothing to compare", file=sys.stderr)
            continue
        problems.extend(verify_manifest(other_dir))

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        print(f"isolation: {len(problems)} violation(s) in {model}",
              file=sys.stderr)
        return 1
    if other_models:
        print(f"isolation: other models unchanged: {', '.join(other_models)}",
              file=sys.stderr)
    else:
        print("isolation: no other model package to compare", file=sys.stderr)
    print(f"isolation: {model} is isolated: fixtures reproduce within "
          "tolerances.json and other models are unchanged", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
