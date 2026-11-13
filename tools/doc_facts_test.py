"""T0: the build document agrees with the build it describes.

``tools/doc_facts.py`` reads three files: ``docs/build.md``, the tracked Bazel
configuration, and the subcommand table of the stage binary. The build document
holds two tables that a reader can derive from the build itself, and this suite
proves the comparison works in both directions.

The cases are of three kinds. The first reads the committed tree, which must
pass. The second mutates one cell, one row, one label, or one link of a copy,
and must fail with the row or the line named. The third proves that ``--write``
repairs a mutated table and reaches the committed bytes, so a reader can fix a
stale table with one command.

The copy holds exactly the files the check reads, so a write in it cannot reach
the checkout. Standard library only: no torch, no GPU, no network, so the suite
rides the default CPU test leg.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The script under test, relative to the repository root.
TOOL = "tools/doc_facts.py"

#: The files ``--check`` reads, relative to the same root.
CHECKED_FILES = (
    "docs/build.md",
    ".bazelrc",
    "src/commands.cc",
)

#: The log heading, before which a calendar date is a failure.
LOG_HEADING = "## Logs"

#: A heading of the design text that another document links to, so its wording
#: is stable. A prose case adds one sentence after it.
PROSE_ANCHOR = "## Layers and the one seam\n"


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
        if ((candidate / TOOL).is_file()
                and (candidate / "docs/build.md").is_file()
                and (candidate / "src/commands.cc").is_file()):
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(f"no runfiles root carries {TOOL}: tried {tried}")


class DocFactsTest(unittest.TestCase):
    """Runs the script the way ``tools/nanochat`` would: as a process."""

    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _runfiles_root()

    def run_tool(self, root: Path, *arguments: str) -> subprocess.CompletedProcess:
        command = [sys.executable, str(self.root / TOOL), *arguments,
                   "--root", str(root)]
        return subprocess.run(command, cwd=str(root), capture_output=True,
                              text=True, timeout=120,
                              stdin=subprocess.DEVNULL, check=False)

    def copy_tree(self, target: Path) -> Path:
        """A working tree that holds the three files the check reads."""
        names = list(CHECKED_FILES) + _linked(self.root, self.root / "docs/build.md")
        for name in names:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_find(self.root, name), destination)
        return target

    def mutate(self, root: Path, old: str, new: str) -> None:
        """Replace one exact string of the build document in a copy."""
        doc = root / "docs/build.md"
        text = doc.read_text(encoding="utf-8")
        self.assertIn(old, text, "the mutation target is not in the document")
        doc.write_text(text.replace(old, new, 1), encoding="utf-8")

    def check_fails(self, root: Path, expected: str) -> str:
        result = self.run_tool(root, "--check")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(expected, result.stdout)
        return result.stdout

    # -- the committed tree ------------------------------------------------

    def test_committed_tree_passes(self) -> None:
        result = self.run_tool(self.root, "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("agree with the tree", result.stdout)

    def test_help_states_the_two_modes(self) -> None:
        result = self.run_tool(self.root, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check", result.stdout)
        self.assertIn("--write", result.stdout)

    def test_unknown_argument_is_a_usage_error(self) -> None:
        result = self.run_tool(self.root, "--bogus")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    # -- a stale table -----------------------------------------------------

    def test_changed_configuration_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root,
                        "| `--config=hopper` | `sm_90` | `bf16` | `cuda` |",
                        "| `--config=hopper` | `sm_90` | `bf16` | `cpu` |")
            self.check_fails(
                root, "row `--config=hopper` states `cpu`, the configuration "
                      "sets `cuda`")

    def test_inherited_value_in_place_of_a_dash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, "| `--config=cuda` | — |",
                        "| `--config=cuda` | `sm_75` |")
            self.check_fails(
                root, "row `--config=cuda` states `sm_75` for Architecture; "
                      "the configuration sets none")

    def test_row_for_an_undeclared_configuration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, "| `--config=hopper` |",
                        "| `--config=h100x` |")
            self.check_fails(root, "does not declare")

    def test_missing_configuration_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            doc = root / "docs/build.md"
            text = doc.read_text(encoding="utf-8")
            kept = [line for line in text.splitlines()
                    if "`--config=hopper`" not in line]
            self.assertLess(len(kept), len(text.splitlines()))
            doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
            self.check_fails(root, "no row for `--config=hopper`")

    def test_changed_subcommand_name_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, "| `rl_step` |", "| `rlstep` |")
            self.check_fails(root, "subcommand row 7 is `rlstep`")

    def test_changed_subcommand_summary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, "| `train` | one training run over packed parquet rows |",
                        "| `train` | one training run |")
            self.check_fails(root, "subcommand `train` reads")

    def test_missing_subcommand_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            doc = root / "docs/build.md"
            text = doc.read_text(encoding="utf-8")
            kept = [line for line in text.splitlines()
                    if not line.startswith("| `eval_bench` |")]
            doc.write_text("\n".join(kept) + "\n", encoding="utf-8")
            self.check_fails(root, "8 subcommand rows")

    # -- the prose rules ---------------------------------------------------

    def test_retired_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, PROSE_ANCHOR,
                        PROSE_ANCHOR + "\nThe Z2 package holds every import.\n")
            self.check_fails(root, "`Z2` is a retired zone label")

    def test_date_before_the_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, PROSE_ANCHOR,
                        PROSE_ANCHOR + "\nMeasured on 2026-01-01.\n")
            self.check_fails(root, "a calendar date before the log")

    def test_file_position_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, PROSE_ANCHOR,
                        PROSE_ANCHOR + "\nThe reader is in checkpoint.py:12.\n")
            self.check_fails(root, "a file position")

    def test_line_number_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, PROSE_ANCHOR,
                        PROSE_ANCHOR + "\nThe reader starts at line 39.\n")
            self.check_fails(root, "a line number")

    def test_log_out_of_order_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            doc = root / "docs/build.md"
            text = doc.read_text(encoding="utf-8")
            head, _, log = text.partition("## Logs")
            entries = log.split("\n### ")
            self.assertGreaterEqual(len(entries), 3, "the log holds entries")
            entries[1], entries[2] = entries[2], entries[1]
            doc.write_text(head + "## Logs" + "\n### ".join(entries),
                           encoding="utf-8")
            self.check_fails(root, "the log holds its entries newest first")

    def test_date_inside_the_log_passes(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            doc = root / "docs/build.md"
            text = doc.read_text(encoding="utf-8")
            body, _, _ = text.partition(LOG_HEADING)
            self.assertTrue(body, "the mutation needs a log heading")
            doc.write_text(text + "\nA later run on 2026-02-02 agreed.\n",
                           encoding="utf-8")
            result = self.run_tool(root, "--check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    # -- the links ---------------------------------------------------------

    def test_broken_internal_anchor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root,
                        "The host budget that these bounds protect is in "
                        "[sandbox.md](sandbox.md).",
                        "The host budget that these bounds protect is in "
                        "[sandbox.md](sandbox.md); see [the bounds](#bounds).")
            self.check_fails(root, "has no heading for #bounds")

    def test_internal_anchor_passes(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root,
                        "The host budget that these bounds protect is in "
                        "[sandbox.md](sandbox.md).",
                        "The host budget that these bounds protect is in "
                        "[sandbox.md](sandbox.md); see [the "
                        "bounds](#resource-bounds).")
            result = self.run_tool(root, "--check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_broken_document_anchor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            self.mutate(root, "(python.md#32-kaggle-delivery)",
                        "(python.md#no-such-heading)")
            self.check_fails(root, "no heading for #no-such-heading")

    # -- the write mode ----------------------------------------------------

    def test_write_repairs_a_stale_table(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            committed = (self.root / "docs/build.md").read_text(encoding="utf-8")
            self.mutate(root,
                        "| `--config=hopper` | `sm_90` | `bf16` | `cuda` |",
                        "| `--config=hopper` | `sm_75` | `fp32` | `cpu` |")

            repaired = self.run_tool(root, "--write")
            self.assertEqual(repaired.returncode, 0, repaired.stderr)
            self.assertIn("rewrote the marked regions", repaired.stdout)

            result = self.run_tool(root, "--check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            again = self.run_tool(root, "--write")
            self.assertIn("already agree", again.stdout)
            self.assertEqual(
                (root / "docs/build.md").read_text(encoding="utf-8"),
                committed)

    def test_write_keeps_a_row_for_every_configuration(self) -> None:
        """A write restates every named configuration of the tracked file."""
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            result = self.run_tool(root, "--write")
            self.assertEqual(result.returncode, 0, result.stderr)
            names = re.findall(r"^build:([A-Za-z0-9_-]+) ",
                               (root / ".bazelrc").read_text(encoding="utf-8"),
                               re.M)
            self.assertIn("sim", names)
            table = (root / "docs/build.md").read_text(encoding="utf-8")
            for name in sorted(set(names)):
                self.assertIn(f"| `--config={name}` |", table,
                              f"the write dropped the row of {name}")

    def test_write_touches_no_prose(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = self.copy_tree(Path(scratch))
            doc = root / "docs/build.md"
            original = doc.read_text(encoding="utf-8")
            marker = "A sentence that the write mode must not touch."
            doc.write_text(original.replace("## Layers and the one seam",
                                            marker + "\n\n## Layers and the "
                                            "one seam", 1), encoding="utf-8")
            result = self.run_tool(root, "--write")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(marker, doc.read_text(encoding="utf-8"))


def _linked(root: Path, doc: Path) -> list[str]:
    """Every local document the build document links to, as a relative path."""
    names = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)\s]+)\)", doc.read_text()):
        target = match.group(1).split("#", 1)[0]
        if target and not target.startswith(("http://", "https://")):
            names.append(str((doc.parent / target).relative_to(root)))
    return names


def _find(root: Path, name: str) -> Path:
    """A checked file, found beside the tool when the runfiles lack the top."""
    for candidate in (root / name, root / "docs" / Path(name).name):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{name} is not in {root}")


if __name__ == "__main__":
    unittest.main()
