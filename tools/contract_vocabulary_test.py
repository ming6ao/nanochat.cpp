"""T0: the capability vocabulary is generated where it is data, checked where
it is prose.

``spec/api-surface.md`` R4a splits the capability tables by content, and
``tools/contract_vocabulary.py`` is the one script that does both halves:
``--write`` rewrites the rows that hold nothing but a name, a bit, and a
position, and ``--check`` compares the rows that also carry hand-written prose.

This suite runs ``--check`` against the committed tree, which must agree with
``spec/capability_bits.md`` section 3 and ``include/nanochat/tensor.h``, and
then against mutated copies, which must fail with the file and the row named.
The script travels in the runfiles through ``//tools:contract_vocabulary`` and
the files it reads travel as ``data``, so the test compares exactly the bytes a
build reads. Hermetic and CPU-only: the standard library, no torch, no GPU, no
network.
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

#: The script under test, relative to the repository root.
TOOL = "tools/contract_vocabulary.py"

#: The files ``--check`` reads, relative to the same root. The mutation cases
#: copy exactly these, so a copy is a working tree for the check and a write
#: into it cannot reach the checkout.
CHECKED_FILES = (
    "spec/capability_bits.md",
    "spec/block_kinds.md",
    "include/nanochat/tensor.h",
    "include/nanochat/kernels.h",
    "src/capability_names.h",
    "src/model_spec.cc",
    "backends/cpu/caps.json",
    "backends/cuda/caps.json",
)


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
                and (candidate / "spec/capability_bits.md").is_file()):
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(f"no runfiles root carries {TOOL}: tried {tried}")


class ContractVocabularyTest(unittest.TestCase):
    """Runs the script the way ``tools/nanochat`` would: as a process."""

    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _runfiles_root()

    def run_tool(self, root: Path) -> subprocess.CompletedProcess:
        command = [sys.executable, str(self.root / TOOL), "--check",
                   "--root", str(root)]
        return subprocess.run(command, cwd=str(root), capture_output=True,
                              text=True, timeout=180,
                              stdin=subprocess.DEVNULL, check=False)

    def copy_tree(self, target: Path) -> Path:
        """A writable copy of the files the check reads."""
        for relative in CHECKED_FILES:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / relative, destination)
        return target

    def test_the_committed_rows_agree_with_the_spec(self) -> None:
        result = self.run_tool(self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("agree with", result.stderr)
        self.assertIn("the two directions agree", result.stderr)
        # The one vocabulary bit the frozen header has no constant for is
        # recorded on every run rather than silently dropped.
        self.assertIn("override_value_gate", result.stderr)
        # So is the one accepted spelling with no prose row (F17).
        self.assertIn("accepts `none`", result.stderr)

    def test_a_renamed_row_fails_with_the_file_and_row_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            path = root / "src/capability_names.h"
            text = path.read_text(encoding="utf-8")
            # `sliding_window` is row 3 of the spec's table. Renaming it moves
            # the name and leaves the bit and the position alone, so the check
            # has to compare names and not just the shape of the row.
            mutated = text.replace('{"sliding_window",',
                                   '{"sliding_windowed",', 1)
            self.assertNotEqual(mutated, text, "the mutation did not apply")
            path.write_text(mutated, encoding="utf-8")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a renamed row must not pass the check")
            self.assertIn("src/capability_names.h:", result.stderr,
                          "the failure names the file and the line")
            self.assertIn("sliding_windowed", result.stderr,
                          "the failure names the spelling it found")

    def test_a_swapped_bit_fails_with_the_row_named(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            path = root / "src/capability_names.h"
            text = path.read_text(encoding="utf-8")
            # The header gives `fp32` bit 5 and `bf16` bit 4. Swapping the two
            # constants keeps every name in place and moves the values, which is
            # the failure mode spec/capability_bits.md section 2 calls a silent
            # lie.
            mutated = text.replace('{"fp32", kFeatureFp32, "kFeatureFp32",',
                                   '{"fp32", kFeatureBf16, "kFeatureBf16",', 1)
            self.assertNotEqual(mutated, text, "the mutation did not apply")
            path.write_text(mutated, encoding="utf-8")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a row on the wrong bit must not pass")
            self.assertIn("src/capability_names.h:", result.stderr)
            self.assertIn("kFeatureBf16", result.stderr)

    def test_a_feature_out_of_order_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            path = root / "backends/cpu/caps.json"
            declaration = json.loads(path.read_text(encoding="utf-8"))
            features = declaration["devices"][0]["features"]
            features[0], features[1] = features[1], features[0]
            path.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a features[] array out of the spec's order "
                                "must not pass")
            self.assertIn("backends/cpu/caps.json", result.stderr)
            self.assertIn("order", result.stderr)

    def test_an_unknown_feature_name_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            path = root / "backends/cuda/caps.json"
            declaration = json.loads(path.read_text(encoding="utf-8"))
            declaration["devices"][0]["features"].append("not_a_bit")
            path.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a name outside the vocabulary must not pass")
            self.assertIn("backends/cuda/caps.json", result.stderr)
            self.assertIn("not_a_bit", result.stderr)

    def mutate(self, root: Path, relative: str, old: str, new: str) -> None:
        """Replace `old` with `new` once, and refuse to pass on a no-op."""
        path = root / relative
        text = path.read_text(encoding="utf-8")
        mutated = text.replace(old, new, 1)
        self.assertNotEqual(mutated, text, f"the mutation of {relative} "
                                           f"did not apply")
        path.write_text(mutated, encoding="utf-8")

    def test_a_prose_kind_the_reader_rejects_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            # A kind row with no reader spelling and no alias row: the failure a
            # model package hits when it names a kind the spec promised.
            self.mutate(root, "spec/block_kinds.md",
                        "| `relu2_mlp` |",
                        "| `geglu_mlp` |")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a kind the reader does not accept must not "
                                "pass")
            self.assertIn("geglu_mlp", result.stderr)
            self.assertIn("spec/block_kinds.md", result.stderr)

    def test_a_reader_spelling_with_no_prose_row_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            # The reverse direction of F17: the reader accepts a spelling the
            # contract has no row for.
            self.mutate(root, "src/model_spec.cc",
                        '{"relu_square", MlpKind::kReluSquare},',
                        '{"relu_square", MlpKind::kReluSquare},\n'
                        '    {"geglu", MlpKind::kReluSquare},')
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a spelling with no prose row must not pass")
            self.assertIn("kMlpRows", result.stderr,
                          "the failure names the reader table")
            self.assertIn("`geglu`", result.stderr)
            self.assertIn("src/model_spec.cc", result.stderr)

    def test_a_primitives_entry_with_no_seam_symbol_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "backends/cpu/caps.json",
                        '"gemm",\n        "gemm_wgrad",',
                        '"gemm",\n        "flash_attention",\n'
                        '        "gemm_wgrad",')
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a primitive with no seam symbol must not pass")
            self.assertIn("backends/cpu/caps.json", result.stderr)
            self.assertIn("flash_attention", result.stderr)
            self.assertIn("FlashAttention", result.stderr,
                          "the failure names the symbol it looked for")

    def test_a_primitive_tier_symbol_with_no_entry_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "backends/cuda/caps.json",
                        '"row_reduce",\n        "span_reduce",',
                        '"span_reduce",')
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "a primitive the declaration does not claim "
                                "must not pass")
            self.assertIn("backends/cuda/caps.json", result.stderr)
            self.assertIn("RowReduce", result.stderr)
            self.assertIn("no primitives[] entry", result.stderr)

    def test_a_seam_symbol_with_no_tier_row_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-vocabulary-") as tmp:
            root = self.copy_tree(Path(tmp))
            # A seam symbol nobody classified: the failure that makes a new
            # entry point ask who supplies it before it lands.
            self.mutate(root, "include/nanochat/kernels.h",
                        "void Synchronize();",
                        "void Synchronize();\nvoid FlashAttentionForward();")
            result = self.run_tool(root)
            self.assertNotEqual(result.returncode, 0,
                                "an unclassified seam symbol must not pass")
            self.assertIn("FlashAttentionForward", result.stderr)
            self.assertIn("add a tier row", result.stderr,
                          "the failure says what a repair has to add")


if __name__ == "__main__":
    unittest.main()
