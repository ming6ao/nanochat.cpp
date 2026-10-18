#!/usr/bin/env python3
"""The symbol gate for the S0 API interposer (docs/simulator.md, invariant 6).

The gate reads the version script as the required set and compares it against
two things:

- the mock's dynamic symbol table: every name in a ``global:`` block must be
  exported, and the version nodes must be present;
- the undefined CUDA symbols of a binary under test: every runtime or cuBLAS
  symbol that binary asks for must be one the mock exports. The mock may export
  more than the binary needs.

    tools/cuda_sim/check_symbols.py [options] [binary...]

      --map FILE             use FILE as the version script
                             (default tools/cuda_sim/cuda_sim.map)
      --mock FILE            check FILE as the mock
                             (default the built interposer)
      --negative-self-test   prove the gate fails when a symbol is missing,
                             then exit. `tools/nanochat simulate --suite api`
                             runs it.

With no binary arguments only the mock is checked. The mock's path is
``bazel-bin/tools/cuda_sim/libcuda_sim_interposer.so``. Build it first with
``tools/nanochat simulate --suite api`` or ``tools/nanochat build``.

Exit status: 0 when the gate passes, 1 when a required symbol is missing, and 2
for a missing input.

See docs/simulator.md sections 5.2 and 12.2, docs/correctness-gates.md, and
docs/simulator-precision.md.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["main", "required_symbols"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MAP = HERE / "cuda_sim.map"
DEFAULT_MOCK = (REPO_ROOT / "bazel-bin" / "tools" / "cuda_sim"
                / "libcuda_sim_interposer.so")

#: The version nodes the binary under test references.
VERSION_NODES = ("libcudart.so.12", "libcublas.so.12")

#: A required name: a line that holds only an identifier and a semicolon.
NAMED_ENTRY = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*);\s*$")

BLOCK_END = re.compile(r"^\s*};\s*$")
GLOBAL_LINE = re.compile(r"^\s*global:\s*$")
LOCAL_LINE = re.compile(r"^\s*local:\s*$")

#: The CUDA symbol families a binary under test may ask for.
CUDA_SYMBOL = re.compile(r"^(cuda|__cuda|cublas)")

#: The name the negative self-test injects, which the mock must not export.
MISSING_NAME = "cudaSimMissingSymbol"


def fail(message: str) -> int:
    """Print one error line and return the failure status."""
    print(message, file=sys.stderr)
    return 1


def complaint(message: str) -> int:
    """Print one input error and return the usage status."""
    print(message, file=sys.stderr)
    return 2


def required_symbols(text: str) -> list[str]:
    """The names in the ``global:`` blocks of a version script, sorted.

    The gate reads only a ``global:`` block. A name under ``local:`` is not
    required, because ``local:`` lists the names the library keeps private.
    """
    names: set[str] = set()
    in_global = False
    for line in text.splitlines():
        if GLOBAL_LINE.match(line):
            in_global = True
        elif LOCAL_LINE.match(line):
            in_global = False
        elif BLOCK_END.match(line):
            in_global = False
        elif in_global:
            entry = NAMED_ENTRY.match(line)
            if entry:
                names.add(entry.group(1))
    return sorted(names)


def read_text(path: Path) -> str:
    """The file text, or ``""`` when the read fails."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def symbol_table(argv: list[str]) -> str:
    """The stdout of one tool call, or ``""`` when it fails."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                check=False)
    except OSError:
        return ""
    return result.stdout


def defined_symbols(mock: Path) -> set[str]:
    """The exported names of the mock, without their version suffix."""
    names = set()
    for line in symbol_table(["nm", "-D", "--defined-only",
                              str(mock)]).splitlines():
        fields = line.split()
        if fields:
            names.add(fields[-1].split("@", 1)[0])
    return names


def version_nodes(mock: Path) -> set[str]:
    """The version-node names of the mock."""
    names = set()
    for line in symbol_table(["readelf", "-V", str(mock)]).splitlines():
        match = re.search(r"Name:\s*(\S+)", line)
        if match:
            names.add(match.group(1))
    return names


def undefined_cuda_symbols(binary: Path) -> list[str]:
    """The CUDA names one binary asks for, sorted and unique."""
    names = set()
    for line in symbol_table(["nm", "-D", "--undefined-only",
                              str(binary)]).splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[-1].split("@", 1)[0]
        if CUDA_SYMBOL.match(name):
            names.add(name)
    return sorted(names)


def negative_self_test(map_file: Path, mock: Path) -> int:
    """Prove the gate rejects a required name the mock does not export."""
    text = read_text(map_file)
    lines = text.splitlines(keepends=True)
    injected: list[str] = []
    done = False
    for line in lines:
        injected.append(line)
        entry = NAMED_ENTRY.match(line)
        if not done and entry and entry.group(1) == "cudaMalloc":
            injected.append(f"    {MISSING_NAME};\n")
            done = True
    fixture = "".join(injected)
    if not done or MISSING_NAME not in fixture:
        return fail("check_symbols: negative self-test could not build its "
                    "fixture")
    with tempfile.NamedTemporaryFile("w", suffix=".map", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(fixture)
        temporary = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--map", str(temporary), "--mock", str(mock)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    finally:
        temporary.unlink()
    if result.returncode == 0:
        return fail("check_symbols: negative self-test FAILED (a missing "
                    "symbol was accepted)")
    print("check_symbols: negative self-test OK (a missing symbol is "
          "rejected)")
    return 0


def check(map_file: Path, mock: Path, binaries: list[str]) -> int:
    """Compare the version script against the mock and the named binaries."""
    if not map_file.is_file():
        return complaint(f"check_symbols: {map_file} is missing")
    if not mock.is_file():
        return complaint(f"check_symbols: {mock} is missing; build "
                         "//tools/cuda_sim:cuda_sim_interposer")
    required = required_symbols(read_text(map_file))
    if not required:
        return complaint("check_symbols: the version script lists no global "
                         "symbols")
    defined = defined_symbols(mock)
    failed = False
    missing = [name for name in required if name not in defined]
    if missing:
        print("check_symbols: the mock does not export:", file=sys.stderr)
        for name in missing:
            print(name, file=sys.stderr)
        failed = True
    nodes = version_nodes(mock)
    for node in VERSION_NODES:
        if node not in nodes:
            print(f"check_symbols: the mock has no version node '{node}'",
                  file=sys.stderr)
            failed = True
    for binary in binaries:
        path = Path(binary)
        if not path.is_file():
            return complaint(f"check_symbols: {binary} is missing")
        unresolved = [name for name in undefined_cuda_symbols(path)
                      if name not in defined]
        if unresolved:
            print(f"check_symbols: {binary} asks for symbols the mock lacks:",
                  file=sys.stderr)
            for name in unresolved:
                print(name, file=sys.stderr)
            failed = True
    if failed:
        return fail("check_symbols: FAILED")
    print(f"check_symbols: OK ({len(required)} required symbols)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    map_file = DEFAULT_MAP
    mock = DEFAULT_MOCK
    binaries: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--map":
            if index + 1 >= len(arguments):
                return complaint("check_symbols: --map needs a path")
            map_file = Path(arguments[index + 1])
            index += 2
        elif argument.startswith("--map="):
            map_file = Path(argument.split("=", 1)[1])
            index += 1
        elif argument == "--mock":
            if index + 1 >= len(arguments):
                return complaint("check_symbols: --mock needs a path")
            mock = Path(arguments[index + 1])
            index += 2
        elif argument.startswith("--mock="):
            mock = Path(argument.split("=", 1)[1])
            index += 1
        elif argument == "--negative-self-test":
            return negative_self_test(map_file, mock)
        else:
            binaries.append(argument)
            index += 1
    return check(map_file, mock, binaries)


if __name__ == "__main__":
    raise SystemExit(main())
