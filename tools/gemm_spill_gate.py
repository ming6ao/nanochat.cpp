"""Rejects a GEMM candidate that spills its accumulator or serializes it.

``docs/kernels.md`` section 11.2 records the defect of the first CUTLASS
instantiation: every kernel stores the accumulator to local memory inside the k
loop. Sixty-four stores and sixty-four loads sit between the multiply
instructions, and the measurement of section 11.1 reads 2.6 to 9.4 percent of the
card peak because of it.

Section 11.4 item 2 states the gate: a host check reads the machine code of each
candidate and rejects a candidate whose ``LDL`` or ``STL`` instructions fall
between its ``HGMMA`` instructions. That gate needs no card, so a defect of this
shape costs a rebuild instead of a sweep.

Section 11.9 states the cause of that defect, and it adds a second rule. A
call between two multiplies forces ptxas to serialize the ``wgmma``
instructions around it, and the accumulator then leaves the register file. The
gate therefore refuses a kernel body that calls out from inside its k loop, and
it names the calls it found.

The tool reads one object file for each candidate. One source file is compiled
once for each candidate, so the label is the index that the object's directory
carries, and the registry of ``backends/cuda/gemm/cutlass_candidates.h`` names
the candidate at that index.

Usage::

    tools/gemm_spill_gate.py                       # the built objects
    tools/gemm_spill_gate.py --objects DIR         # another build's objects
    tools/gemm_spill_gate.py PATH [PATH ...]       # named objects
    tools/gemm_spill_gate.py --sass FILE           # analyse disassembly text

``--sass`` reads the text ``cuobjdump -sass`` writes, so the parser and the
verdict are testable with no CUDA toolchain and no card.

Exit codes: 0 when every candidate is clean, 1 when a candidate spills, 2 for a
usage error or a missing toolchain.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: The build's object tree. Each candidate is one target, so each one has its
#: own directory under `_objs` and the index is in the directory name.
DEFAULT_OBJECTS = Path("bazel-bin/backends/cuda/gemm/_objs")

#: The file that names the candidates, in registry order. The table is
#: `kGemmCandidates` in the header, because a candidate's translation unit reads
#: its own row as a compile-time constant.
DEFAULT_REGISTRY = Path("backends/cuda/gemm/cutlass_candidates.h")

#: One disassembled instruction: the address comment, an optional predicate, and
#: the mnemonic.
INSTRUCTION = re.compile(
    r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)"
)

#: One function header, in either of the two forms `cuobjdump` writes.
FUNCTION = re.compile(r"^\s*(?:Function\s*:\s*(\S+)|\.section\s+\.text\.(\S+))")

#: The multiply instructions of the mainloop.
MULTIPLY = "HGMMA"

#: The instructions that read or write local memory, which is where a spilled
#: accumulator goes.
LOAD = "LDL"
STORE = "STL"

#: A call to a function ptxas could not inline. Inside a k loop it serializes
#: the multiplies and forces the accumulator out of the register file.
CALL = "CALL"


class GateError(Exception):
    """A tree, a toolchain, or an object the gate cannot read."""


@dataclass(frozen=True)
class FunctionScan:
    """What one kernel body does with its accumulator."""

    name: str
    multiplies: int
    loads: int
    stores: int
    calls: int = 0

    @property
    def is_mainloop(self) -> bool:
        """True when the body holds a k loop: two or more multiplies."""
        return self.multiplies >= 2

    @property
    def spills(self) -> bool:
        """True when local memory is touched between two multiplies."""
        return self.is_mainloop and (self.loads + self.stores) > 0

    @property
    def serializes(self) -> bool:
        """True when a call sits between two multiplies."""
        return self.is_mainloop and self.calls > 0

    @property
    def is_clean(self) -> bool:
        """True when the k loop keeps its accumulator and its instruction order."""
        return not self.spills and not self.serializes


def analyze(text: str) -> list[FunctionScan]:
    """One scan for each function body in disassembly text."""
    scans: list[FunctionScan] = []
    name: str | None = None
    mnemonics: list[str] = []

    def finish() -> None:
        if name is None:
            return
        positions = [
            index
            for index, mnemonic in enumerate(mnemonics)
            if mnemonic.startswith(MULTIPLY)
        ]
        loads = 0
        stores = 0
        calls = 0
        if positions:
            first = positions[0]
            last = positions[-1]
            for mnemonic in mnemonics[first : last + 1]:
                if mnemonic.startswith(LOAD):
                    loads += 1
                elif mnemonic.startswith(STORE):
                    stores += 1
                elif mnemonic.startswith(CALL):
                    calls += 1
        scans.append(
            FunctionScan(
                name=name,
                multiplies=len(positions),
                loads=loads,
                stores=stores,
                calls=calls,
            )
        )

    for line in text.splitlines():
        header = FUNCTION.match(line)
        if header is not None:
            finish()
            name = header.group(1) or header.group(2)
            mnemonics = []
            continue
        instruction = INSTRUCTION.match(line)
        if instruction is not None:
            mnemonics.append(instruction.group(1))
    finish()
    return scans


def verdict(scans: list[FunctionScan]) -> bool:
    """True when every k loop keeps its accumulator and its instruction order."""
    return all(scan.is_clean for scan in scans)


def read_candidates(path: Path) -> list[str]:
    """The registry's candidate names, in registry order.

    ``backends/cuda/gemm/cutlass_candidates.h`` holds one row for each
    candidate and its first field is the name. The index of the name is the
    candidate's index, which is the label the build puts on the object's
    directory.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GateError(f"cannot read the registry {path}: {error}") from error
    start = text.find("kGemmCandidates[]")
    end = text.find("};", start) if start >= 0 else -1
    if start < 0 or end < 0:
        raise GateError(f"the registry {path} holds no kGemmCandidates table")
    names = re.findall(r'^\s*\{"([a-z0-9-]+)",', text[start:end], re.MULTILINE)
    if not names:
        raise GateError(f"the registry {path} names no candidate")
    return names


def sass_of(cuobjdump: str, path: Path) -> str:
    """The disassembly text of one object file."""
    try:
        completed = subprocess.run(
            [cuobjdump, "-sass", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise GateError(f"cannot run {cuobjdump}: {error}") from error
    if completed.returncode != 0:
        raise GateError(
            f"{cuobjdump} -sass {path} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def candidate_index(path: Path) -> int:
    """The registry index a path states.

    One source file is compiled once for each candidate, so the object carries
    the source's name and its directory carries the index.
    """
    stamp = re.search(r"cutlass_candidate_(\d+)", str(path))
    if stamp is None:
        raise GateError(f"{path} names no candidate")
    return int(stamp.group(1))


def discover(directory: Path) -> list[Path]:
    """The candidate objects of one build's object tree."""
    if not directory.is_dir():
        raise GateError(
            f"no object directory at {directory}; build the candidate library "
            "first"
        )
    objects = sorted(directory.rglob("cutlass_candidate_*[0-9]/*.o"))
    if not objects:
        objects = sorted(directory.rglob("cutlass_candidate_*.o"))
    if not objects:
        raise GateError(f"{directory} holds no candidate object")
    return objects


def report_candidate(
    label: str, path: Path, scans: list[FunctionScan], full_names: bool = False
) -> bool:
    """One line for a candidate, and the worst kernel of a refused one.

    A candidate is eight instantiations and its mangled names are hundreds of
    characters, so a refused candidate states the count of its refused kernels
    and the worst of them. `--full-names` prints every one.
    """
    loops = [scan for scan in scans if scan.is_mainloop]
    if not loops:
        print(f"gemm spill gate: {label} ({path.name}): clean, no k loop to check")
        return True
    refused = [scan for scan in loops if not scan.is_clean]
    if not refused:
        print(
            f"gemm spill gate: {label} ({path.name}): clean, "
            f"{len(loops)} k loop(s)"
        )
        return True
    spilled = [scan for scan in refused if scan.spills]
    serialized = [scan for scan in refused if scan.serializes]
    if spilled:
        print(f"gemm spill gate: {label} ({path.name}): SPILLS the accumulator")
        print(
            f"  {len(spilled)} of {len(loops)} kernel(s) touch local memory "
            "inside a k loop"
        )
        worst = max(spilled, key=lambda scan: scan.loads + scan.stores)
        for scan in [worst] if not full_names else spilled:
            print(
                f"  {_short(scan.name)}: {scan.multiplies} {MULTIPLY}, "
                f"{scan.loads} {LOAD}, {scan.stores} {STORE}"
            )
    if serialized:
        print(f"gemm spill gate: {label} ({path.name}): SERIALIZES the mainloop")
        print(
            f"  {len(serialized)} of {len(loops)} kernel(s) call out from "
            "inside a k loop"
        )
        worst = max(serialized, key=lambda scan: scan.calls)
        for scan in [worst] if not full_names else serialized:
            print(
                f"  {_short(scan.name)}: {scan.multiplies} {MULTIPLY}, "
                f"{scan.calls} {CALL}"
            )
    return False


def _short(name: str) -> str:
    """A mangled name a reader can compare, at the width of one line."""
    if len(name) <= 72:
        return name
    return name[:69] + "..."


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cuobjdump", default="")
    parser.add_argument("--sass", type=Path, default=None)
    parser.add_argument("--full-names", action="store_true")
    options = parser.parse_args(argv)

    # The text mode is what a test drives: it reads disassembly the caller
    # already wrote, so the parser and the verdict need no toolchain.
    if options.sass is not None:
        try:
            text = options.sass.read_text(encoding="utf-8")
        except OSError as error:
            print(f"gemm spill gate: {error}", file=sys.stderr)
            return 2
        scans = analyze(text)
        clean = verdict(scans)
        report_candidate(options.sass.stem, options.sass, scans, options.full_names)
        return 0 if clean else 1

    toolchain = options.cuobjdump or shutil.which("cuobjdump") or ""
    if not toolchain:
        print(
            "gemm spill gate: no cuobjdump; install the CUDA toolkit or name "
            "one with --cuobjdump",
            file=sys.stderr,
        )
        return 2

    try:
        names = read_candidates(options.registry)
        paths = list(options.paths) if options.paths else discover(options.objects)
    except GateError as error:
        print(f"gemm spill gate: {error}", file=sys.stderr)
        return 2

    clean = True
    for path in sorted(paths):
        try:
            index = candidate_index(path)
            text = sass_of(toolchain, path)
        except GateError as error:
            print(f"gemm spill gate: {error}", file=sys.stderr)
            return 2
        label = names[index] if index < len(names) else f"candidate {index}"
        if not report_candidate(label, path, analyze(text), options.full_names):
            clean = False
    if clean:
        print(f"gemm spill gate: {len(paths)} candidate(s) checked, none spills")
        return 0
    print(f"gemm spill gate: {len(paths)} candidate(s) checked, one or more refused")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
