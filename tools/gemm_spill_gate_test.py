"""T0: the candidate spill gate reads machine code and refuses a spill.

``docs/kernels.md`` section 11.4 item 2 states the gate: a candidate whose
``LDL`` or ``STL`` instructions fall between its ``HGMMA`` instructions spills
its accumulator and leaves the registry. Section 11.2 records the defect the gate
exists to catch, which a card run took 2.6 to 9.4 percent of the peak to show.

The gate itself runs ``cuobjdump -sass`` over a built object, so this suite
drives its ``--sass`` mode instead: the same text, a file the test writes, and no
CUDA toolchain and no card. Seven properties are checked:

   1. A body whose multiplies hold no local access is clean.
   2. A body with a load between two multiplies spills.
   3. A body with a store between two multiplies spills.
   4. A single multiply is not a mainloop, so its local access is not a spill.
   5. The exit code states the verdict, and the refuted function is named.
   6. A call between two multiplies refuses the body as well.
   7. A call outside the multiplies is clean.

Hermetic and CPU-only: the standard library and a text file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The tool under test, relative to the repository root.
TOOL = "tools/gemm_spill_gate.py"


def _runfiles_root() -> Path:
    """The root as this test sees it: runfiles first, the source tree second."""
    candidates: list[Path] = []
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        workspace = os.environ.get("TEST_WORKSPACE") or "_main"
        candidates.append(Path(runfiles) / workspace)
        candidates.append(Path(runfiles) / "_main")
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        if (candidate / TOOL).is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(f"no runfiles root carries {TOOL}: tried {tried}")


def _instruction(address: int, text: str) -> str:
    return f"\t\t/*{address:04x}*/                   {text} ;"


def _function(name: str, body: list[str]) -> str:
    """One function body, in the form ``cuobjdump -sass`` writes it."""
    lines = [f"Function : {name}"]
    for index, text in enumerate(body):
        lines.append(_instruction(index * 16, text))
    return "\n".join(lines)


class GemmSpillGateTest(unittest.TestCase):
    """The parser and the verdict, over synthetic disassembly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _runfiles_root() / TOOL

    def _analyze(self, function: str) -> subprocess.CompletedProcess:
        directory = tempfile.mkdtemp(prefix="gemm-spill-")
        self.addCleanup(shutil.rmtree, directory)
        path = Path(directory) / "cutlass_candidate_0.sass"
        path.write_text(function + "\n", encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(self.tool), "--sass", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_a_mainloop_without_local_access_is_clean(self) -> None:
        function = _function(
            "_ZN3gemm4mainEv",
            [
                "S2R R0, SR_TID.X",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "LDSM.16.M88.4 R8, [R2]",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "EXIT",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("clean", completed.stdout)

    def test_a_load_between_multiplies_spills(self) -> None:
        function = _function(
            "_ZN3gemm4mainEv",
            [
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "LDL.128 R16, [R1+0x150]",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "STL.128 [R1+0x160], R16",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("SPILLS", completed.stdout)
        self.assertIn("_ZN3gemm4mainEv", completed.stdout)
        self.assertIn("1 LDL", completed.stdout)

    def test_a_store_between_multiplies_spills(self) -> None:
        function = _function(
            "_ZN3gemm4mainEv",
            [
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "STL.64 [R1+0x18], R16",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("1 STL", completed.stdout)

    def test_one_multiply_is_not_a_mainloop(self) -> None:
        # No k loop, so no accumulator can leave the register file in a loop.
        function = _function(
            "_ZN3gemm4tailEv",
            [
                "LDL.128 R16, [R1+0x150]",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "STL.128 [R1+0x160], R16",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("clean", completed.stdout)

    def test_a_call_between_multiplies_serializes(self) -> None:
        # The defect of docs/kernels.md section 11.9: an `assert` in device code
        # becomes a call to `__assertfail`, ptxas serializes the multiplies
        # around it, and the accumulator leaves the register file.
        function = _function(
            "_ZN3gemm4mainEv",
            [
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "CALL.ABS.NOINC 0x0",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("SERIALIZES", completed.stdout)
        self.assertIn("1 CALL", completed.stdout)

    def test_a_call_outside_the_multiplies_is_clean(self) -> None:
        # The epilogue and the scheduler may call; the k loop is what decides.
        function = _function(
            "_ZN3gemm5blockEv",
            [
                "CALL.ABS.NOINC 0x0",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_local_access_outside_the_multiplies_is_clean(self) -> None:
        # The epilogue may use local memory; only the k loop decides the spill.
        function = _function(
            "_ZN3gemm5blockEv",
            [
                "LDL.128 R16, [R1+0x150]",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "HGMMA.16816.F32.BF16 R4, R8, R12, R4",
                "STL.128 [R1+0x160], R16",
            ],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_a_body_without_multiplies_is_clean(self) -> None:
        function = _function(
            "_ZN3gemm4fillEv",
            ["LDL.128 R16, [R1+0x150]", "STL.128 [R1+0x160], R16"],
        )
        completed = self._analyze(function)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
