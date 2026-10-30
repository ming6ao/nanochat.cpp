"""T0: the conformance matrix and the isolation gate, as checks rather than files.

``docs/model-and-device-bringup.md`` makes two artifacts *generated*, and this
suite is the gate that keeps them generated:

* section 11's model/device conformance matrix is not a maintained document. It
  is derived from ``models/<name>/spec.json`` and ``backends/<name>/caps.json``
  by ``tools/gen_matrix.py``, and a hand-added row fails ``--check`` with the
  offending label named;
* section 10.4's cross-package isolation gate is not a checklist. It is
  ``tools/isolation_gate.py``: the flagship model's fixtures are verified
  against its manifest, and a mutated fixture fails with the offending record
  named, so a log is enough to diagnose the break.

Both scripts run in *check* mode and neither writes into ``models/`` or
``tests/``: the matrix is rendered into the test's temporary directory, and the
mutated model package is a copy. The suite is hermetic and CPU-only - the
standard library, no torch, no GPU, no network - in the style of the other
Python suites in the tree.

The scripts travel in the runfiles of this target through ``//tools:matrix``
and ``//tools:isolation_gate``; the model package and the two capability
declarations travel with the data they read, so the test reads exactly the
bytes the generator and the gate read in a build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The section 11 matrix generator, relative to the repository root.
GEN_MATRIX = "tools/gen_matrix.py"

#: The section 10.4 cross-package isolation gate, relative to the same root.
ISOLATION_GATE = "tools/isolation_gate.py"

#: The scripts the isolation gate needs beside it: it dispatches the fixture
#: generator, and the generator's shared helpers import the model package
#: selector. The copy in the mutated-package test carries all four, so the
#: copied tree is a working one.
GATE_SCRIPTS = (
    "isolation_gate.py",
    "dump_fixtures.py",
    "fixture_common.py",
    "fixture_format.py",
)

#: The one model package in this release.
MODEL = "nanochat-default"

#: The fixture the mutation case edits, and a record near the end of it: the
#: last record's payload byte is an fp32 exponent byte, so flipping it moves the
#: value far outside every bound the model declares and the gate has to name
#: the record rather than "a file changed".
MUTATED_FIXTURE = "debug_state.bin"

#: A hand-edited matrix: one true row and one row for a model the generator
#: cannot derive. Section 15 makes such a row a failure, and the failure must
#: name the label.
HAND_EDITED_MATRIX = """\
| Model | Tier | Device | Graphs | Status |
|---|---|---|---|---|
| nanochat-default | T0 CPU | cpu | train step, prefill, decode, eval, generate | pass |
| nanochat-second | T0 CPU | cpu | train step, decode | pass |
"""

#: A hand-edited matrix whose only fault is a tier the tier rule does not
#: produce. The generator names the row and the tier.
HAND_WRITTEN_TIER_MATRIX = """\
| Model | Tier | Device | Graphs | Status |
|---|---|---|---|---|
| nanochat-default | T0 CPU | cpu | train step, prefill, decode, eval, generate | pass |
| nanochat-default | T3 nightly | h100 | train step, decode | pending |
"""


def _runfiles_root() -> Path:
    """The repository root as this test sees it: runfiles first, source second.

    The runfiles tree keeps the repository layout, so the same relative position
    gives the root in a Bazel test (where ``RUNFILES_DIR`` is set and the test
    file also sits at ``<root>/tests/``) and in a direct run from the checkout.
    """
    candidates: list[Path] = []
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        workspace = os.environ.get("TEST_WORKSPACE") or "_main"
        candidates.append(Path(runfiles) / workspace)
        candidates.append(Path(runfiles) / "_main")
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        if (candidate / GEN_MATRIX).is_file() and (candidate / "models").is_dir():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(
        f"no runfiles root carries {GEN_MATRIX} and models/: tried {tried}")


class ScriptTest(unittest.TestCase):
    """Runs a repository script the way ``tools/nanochat`` would: as a process."""

    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _runfiles_root()

    def run_script(self, script: Path, *argv: str,
                   cwd: Path | None = None) -> subprocess.CompletedProcess:
        command = [sys.executable, str(script), *argv]
        return subprocess.run(
            command,
            cwd=str(cwd if cwd is not None else self.root),
            capture_output=True,
            text=True,
            timeout=180,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    def run_gen_matrix(self, *argv: str,
                       cwd: Path | None = None) -> subprocess.CompletedProcess:
        return self.run_script(self.root / GEN_MATRIX, *argv, cwd=cwd)

    def run_isolation_gate(self, script: Path, *argv: str,
                           cwd: Path) -> subprocess.CompletedProcess:
        return self.run_script(script, *argv, cwd=cwd)


class MatrixCheckTest(ScriptTest):
    """The section 11 matrix is a check over a derivation, not a maintained file."""

    def test_the_derivation_is_valid(self) -> None:
        # No matrix document is supplied, so the generator validates its own
        # rows: every model has a row, every passing row covers the minimum
        # graphs. There is no document to hand-edit, which is the point.
        result = self.run_gen_matrix("--check", "--root", str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated", result.stderr)

    def test_a_generated_document_passes_its_own_check(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-matrix-") as tmp:
            document = Path(tmp) / "conformance-matrix.md"
            render = self.run_gen_matrix("--out", str(document),
                                         "--root", str(self.root))
            self.assertEqual(render.returncode, 0, render.stderr)
            self.assertTrue(document.is_file(), "the generator wrote no matrix")
            check = self.run_gen_matrix("--check", "--matrix", str(document),
                                        "--root", str(self.root))
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertIn("agree with the matrix document", check.stderr)

    def test_a_hand_added_row_fails_with_the_label_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-matrix-") as tmp:
            document = Path(tmp) / "hand-edited.md"
            document.write_text(HAND_EDITED_MATRIX, encoding="utf-8")
            result = self.run_gen_matrix("--check", "--matrix", str(document),
                                         "--root", str(self.root))
            self.assertNotEqual(
                result.returncode, 0,
                "a row for an undeclared model must not pass the check")
            self.assertIn("nanochat-second", result.stderr,
                          "the failure names the offending label")
            self.assertIn("cannot derive", result.stderr)

    def test_a_hand_written_tier_fails_with_the_row_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-matrix-") as tmp:
            document = Path(tmp) / "hand-written-tier.md"
            document.write_text(HAND_WRITTEN_TIER_MATRIX, encoding="utf-8")
            result = self.run_gen_matrix("--check", "--matrix", str(document),
                                         "--root", str(self.root))
            self.assertNotEqual(result.returncode, 0,
                                "a tier the tier rule does not produce must fail")
            self.assertIn("T3 nightly", result.stderr,
                          "the failure names the offending row's tier")

    def test_a_passing_row_that_skips_decode_fails(self) -> None:
        # The minimum for a passing row is the train step *and* decode
        # (section 11). A row that drops decode is not a passing row.
        document = (
            "| Model | Tier | Device | Graphs | Status |\n"
            "|---|---|---|---|---|\n"
            f"| {MODEL} | T0 CPU | cpu | train step | pass |\n")
        with tempfile.TemporaryDirectory(prefix="nanochat-matrix-") as tmp:
            path = Path(tmp) / "short-row.md"
            path.write_text(document, encoding="utf-8")
            result = self.run_gen_matrix("--check", "--matrix", str(path),
                                         "--root", str(self.root))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decode", result.stderr)


class IsolationCheckTest(ScriptTest):
    """The section 10.4 gate passes on the shipped bytes and names a break."""

    def test_the_flagship_package_passes_its_manifest_check(self) -> None:
        result = self.run_isolation_gate(
            self.root / ISOLATION_GATE, "--model", MODEL, "--manifest-only",
            cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("isolated", result.stderr)

    def test_a_mutated_fixture_fails_with_the_record_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-isolation-") as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            for name in GATE_SCRIPTS:
                shutil.copy2(self.root / "tools" / name,
                             root / "tools" / name)
            package = root / "models" / MODEL
            shutil.copytree(self.root / "models" / MODEL, package)
            # The pristine bytes stand in for a regeneration, which is what the
            # gate diffs in its steps 1 and 2.
            regenerated = root / "regenerated"
            shutil.copytree(package / "fixtures", regenerated)

            script = root / ISOLATION_GATE
            clean = self.run_isolation_gate(
                script, "--model", MODEL, "--generated-dir", str(regenerated),
                cwd=root)
            self.assertEqual(
                clean.returncode, 0,
                "the unmodified copy of the package must pass: " + clean.stderr)

            target = package / "fixtures" / MUTATED_FIXTURE
            data = bytearray(target.read_bytes())
            # The last two payload bytes are the exponent byte of the final
            # float record, so this is a value change, not a framing change:
            # the gate has to compare a record rather than report a parse error.
            data[-2] ^= 0xFF
            target.write_bytes(bytes(data))

            broken = self.run_isolation_gate(
                script, "--model", MODEL, "--generated-dir", str(regenerated),
                cwd=root)
            self.assertNotEqual(
                broken.returncode, 0,
                "a mutated fixture must not pass the isolation gate")
            self.assertIn(MUTATED_FIXTURE, broken.stderr,
                          "the failure names the offending fixture")
            self.assertIn("record", broken.stderr,
                          "the failure names the offending record")
            # The last record of the fixture, whose exponent byte the mutation
            # moved: the log is enough to diagnose the break without opening
            # the file.
            self.assertIn("value_embeds.1.weight", broken.stderr,
                          "the failure names the record that moved")
            self.assertIn("FAIL:", broken.stderr)

    def test_the_committed_bytes_are_not_touched_by_the_suite(self) -> None:
        # The gate writes only into a temporary directory. Re-running it on the
        # real package leaves the committed bytes alone: the digests the model
        # package pins still hold afterwards.
        fixture = self.root / "models" / MODEL / "fixtures" / MUTATED_FIXTURE
        before = fixture.read_bytes()
        result = self.run_isolation_gate(
            self.root / ISOLATION_GATE, "--model", MODEL, "--manifest-only",
            cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, fixture.read_bytes(),
                         "the gate rewrote a committed fixture byte")


if __name__ == "__main__":
    unittest.main()
