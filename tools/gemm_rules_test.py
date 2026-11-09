"""T0: the GEMM decision table is written from a sweep report, and checked.

``docs/kernels.md`` section 11.4 item 5 states the design: the table of
``backends/cuda/gemm/cutlass_rules.h`` is data, ``tools/gemm_rules.py --write``
writes it from a ``gemm_bench --json`` report, and ``--check`` compares the
committed block with the report that produced it. This suite drives both modes on
a report the test writes, so no card and no CUDA toolchain are needed.

Six properties are checked:

   1. The order of section 11.4 holds: a candidate that beats cuBLAS and the hand
      tile is promoted, a hand tile that beats cuBLAS but no candidate is
      recorded as the hand route, and cuBLAS keeps every other shape.
   2. A ``:median`` companion row is not a measurement and never decides a row.
   3. ``--write`` is a fixed point: a second run changes no byte.
   4. ``--check`` accepts the block ``--write`` wrote and names the header when
      the block and the report disagree.
   5. A report of the measurement family is refused, because those rows answer a
      question about a kernel and not about a model.
   6. A candidate name the registry does not hold is refused.

Hermetic and CPU-only: the standard library, no torch, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The tool under test, relative to the repository root.
TOOL = "tools/gemm_rules.py"

#: The two files the tool reads and rewrites.
HEADER = "backends/cuda/gemm/cutlass_rules.h"
REGISTRY = "backends/cuda/gemm/cutlass_candidates.h"


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
        if ((candidate / TOOL).is_file() and (candidate / HEADER).is_file()
                and (candidate / REGISTRY).is_file()):
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(f"no runfiles root carries {TOOL}: tried {tried}")


def _report(rows: list[tuple[str, str, float]],
            family: str = "production") -> dict:
    """One sweep report, as ``gemm_bench --json`` writes it."""
    return {
        "schema": "nanochat.bench.v3",
        "device": "NVIDIA H100 80GB HBM3",
        "precision": "bf16",
        "row_family": family,
        "rows": [
            {"name": name, "shape": shape, "ms": ms, "gflops": 0.0}
            for name, shape, ms in rows
        ],
    }


#: One square Muon-shaped row and one row with a transposed operand and an fp32
#: store, so the key covers every field the table names.
SQUARE = "1280x1280x1280 batch 40"
TALL = "4096x1024x4096 tn fp32 out"

BASE_ROWS = [
    # The square row: a candidate beats both other arms, so it is promoted.
    ("gemm:muon:square:cublas", SQUARE, 0.2230),
    ("gemm:muon:square:wide-n:cutlass", SQUARE, 0.3000),
    ("gemm:muon:square:balanced:cutlass", SQUARE, 0.2000),
    ("gemm:muon:square:128x128x64x2s8:hand", SQUARE, 0.7414),
    # The median companions are higher and must not decide anything.
    ("gemm:muon:square:balanced:cutlass:median", SQUARE, 0.9000),
    ("gemm:muon:square:cublas:median", SQUARE, 0.2300),
    # The tall row: the hand tile beats cuBLAS and no candidate does.
    ("gemm:muon:tall:cublas", TALL, 0.4000),
    ("gemm:muon:tall:wide-n:cutlass", TALL, 0.6000),
    ("gemm:muon:tall:balanced:cutlass", TALL, 0.5000),
    ("gemm:muon:tall:128x128x64x2s8:hand", TALL, 0.3500),
]


class GemmRulesTest(unittest.TestCase):
    """The writer, its check mode, and the order of section 11.4."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _runfiles_root()
        cls.tool = cls.root / TOOL

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="gemm-rules-")
        self.addCleanup(shutil.rmtree, self.directory)
        self.header = Path(self.directory) / "cutlass_rules.h"
        self.registry = Path(self.directory) / "cutlass_candidates.h"
        shutil.copy(self.root / HEADER, self.header)
        shutil.copy(self.root / REGISTRY, self.registry)

    def _write_report(self, rows, family: str = "production") -> Path:
        path = Path(self.directory) / "bench.json"
        path.write_text(json.dumps(_report(rows, family)), encoding="utf-8")
        return path

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.tool), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def _generate(self) -> None:
        report = self._write_report(BASE_ROWS)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--write",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_write_promotes_by_the_order_of_section_11_4(self) -> None:
        self._generate()
        text = self.header.read_text(encoding="utf-8")
        # One field per line, so a comparison states the block without its
        # layout: the values and their order are the contract.
        block = " ".join(text.split())
        self.assertIn("std::array<GemmRuleRow, 2> kGemmRules{{", block)
        # The square row is promoted to candidate 1 (`balanced`), whose measured
        # fraction of cuBLAS is 0.2230 / 0.2000.
        self.assertIn(
            "{ 1280, 1280, 1280, 40, false, false, false, 1, "
            "GemmRoute::kCandidate, 1.1150, 0.3008, },",
            block,
        )
        # The tall row records the hand tile and names no candidate.
        self.assertIn(
            "{ 4096, 1024, 4096, 1, true, false, true, -1, "
            "GemmRoute::kHand, 1.1429, 1.1429, },",
            block,
        )

    def test_cublas_keeps_a_row_nothing_beats(self) -> None:
        rows = [
            ("gemm:muon:square:cublas", SQUARE, 0.1000),
            ("gemm:muon:square:wide-n:cutlass", SQUARE, 0.3000),
            ("gemm:muon:square:128x128x64x2s8:hand", SQUARE, 0.2000),
        ]
        report = self._write_report(rows)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--write",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = self.header.read_text(encoding="utf-8")
        block = " ".join(text.split())
        self.assertIn("GemmRoute::kCublas", block)
        self.assertIn(
            "{ 1280, 1280, 1280, 40, false, false, false, -1, "
            "GemmRoute::kCublas, 1.0000, 0.5000, },",
            block,
        )

    def test_a_seam_report_is_refused(self) -> None:
        # `gemm_benchmark --arms seam` measures the shipped path, which is the
        # gate on a promotion and not an input to it.
        rows = [
            ("gemm:muon:square:seam", SQUARE, 0.2000),
        ]
        report = self._write_report(rows)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--write",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("verification row", completed.stderr)

    def test_a_win_inside_the_margin_promotes_nothing(self) -> None:
        # The candidate is faster than cuBLAS by 2.2 percent, and two runs of one
        # sweep can disagree by more than that, so the row keeps cuBLAS.
        rows = [
            ("gemm:muon:square:cublas", SQUARE, 0.2230),
            ("gemm:muon:square:wide-n:cutlass", SQUARE, 0.2180),
            ("gemm:muon:square:128x128x64x2s8:hand", SQUARE, 0.7414),
        ]
        report = self._write_report(rows)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--write",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        block = " ".join(self.header.read_text(encoding="utf-8").split())
        self.assertIn(
            "{ 1280, 1280, 1280, 40, false, false, false, -1, "
            "GemmRoute::kCublas, 1.0000, 0.3008, },",
            block,
        )

    def test_the_margin_is_a_flag(self) -> None:
        # A caller that states a smaller margin promotes the same 2.2 percent
        # win, which shows the flag reaches the decision.
        rows = [
            ("gemm:muon:square:cublas", SQUARE, 0.2230),
            ("gemm:muon:square:wide-n:cutlass", SQUARE, 0.2180),
            ("gemm:muon:square:128x128x64x2s8:hand", SQUARE, 0.7414),
        ]
        report = self._write_report(rows)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--min-margin", "0.01",
            "--write",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        block = " ".join(self.header.read_text(encoding="utf-8").split())
        self.assertIn("GemmRoute::kCandidate", block)

    def test_write_is_a_fixed_point_and_check_accepts_it(self) -> None:
        self._generate()
        first = self.header.read_bytes()
        self._generate()
        self.assertEqual(first, self.header.read_bytes())
        report = self._write_report(BASE_ROWS)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--check",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("2 row(s) agree", completed.stdout)
        self.assertIn("1 promoted", completed.stdout)

    def test_check_names_the_header_a_hand_edit_drifted(self) -> None:
        self._generate()
        text = self.header.read_text(encoding="utf-8")
        edited = text.replace("1.1150", "1.2000")
        self.assertNotEqual(text, edited)
        self.header.write_text(edited, encoding="utf-8")
        report = self._write_report(BASE_ROWS)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--check",
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn(str(self.header), completed.stderr)
        self.assertIn("have", completed.stderr)
        self.assertIn("want", completed.stderr)

    def test_measurement_family_is_refused(self) -> None:
        report = self._write_report(BASE_ROWS, family="measurement")
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--check",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("measurement", completed.stderr)
        self.assertIn("production", completed.stderr)

    def test_unknown_candidate_is_refused(self) -> None:
        rows = list(BASE_ROWS) + [
            ("gemm:muon:square:no-such-candidate:cutlass", SQUARE, 0.0100),
        ]
        report = self._write_report(rows)
        completed = self._run(
            "--report", str(report),
            "--header", str(self.header),
            "--registry", str(self.registry),
            "--check",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("no-such-candidate", completed.stderr)


if __name__ == "__main__":
    unittest.main()
