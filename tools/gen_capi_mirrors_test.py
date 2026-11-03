"""T0: the three C ABI mirrors are generated from one header.

``docs/api-surface.md`` R11b: ``tools/gen_capi_mirrors.py`` reads
``include/nanochat/capi.h``, writes the ``_SIGNATURES`` table and the
``static_assert`` block, and compares the hand-written ``_LAYOUT`` table of
``python/nanochat_cpp/_core.py`` with the header through ``_core.layout()``.
This suite runs ``--check`` against the committed tree, which must agree with
the header, and then against mutated copies, which must fail naming the file,
the structure, and the field.

It also pins the property the design rests on: ``--write`` is a fixed point, so
a second run leaves every byte alone and a regeneration never shows up in a
diff.

The script travels in the runfiles through ``//tools:gen_capi_mirrors`` and the
three files it reads travel as ``data``, so the test compares exactly the bytes
a build reads. Hermetic and CPU-only: the standard library, no torch, no GPU, no
network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The script under test, relative to the repository root.
TOOL = "tools/gen_capi_mirrors.py"

#: The files ``--check`` and ``--write`` read and rewrite. The mutation cases
#: copy exactly these, so a copy is a working tree and a write into it cannot
#: reach the checkout.
MIRRORED_FILES = (
    "include/nanochat/capi.h",
    "python/nanochat_cpp/_core.py",
    "bindings/nanochat_capi_test.cc",
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
                and (candidate / "include/nanochat/capi.h").is_file()):
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(f"no runfiles root carries {TOOL}: tried {tried}")


class GenCapiMirrorsTest(unittest.TestCase):
    """Runs the script the way ``tools/nanochat`` would: as a process."""

    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _runfiles_root()

    def run_tool(self, root: Path, mode: str) -> subprocess.CompletedProcess:
        command = [sys.executable, str(self.root / TOOL), mode,
                   "--root", str(root)]
        return subprocess.run(command, cwd=str(root), capture_output=True,
                              text=True, timeout=180,
                              stdin=subprocess.DEVNULL, check=False)

    def copy_tree(self, target: Path) -> Path:
        """A writable copy of the files the generator reads and rewrites."""
        for relative in MIRRORED_FILES:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / relative, destination)
        return target

    def mutate(self, root: Path, relative: str, old: str, new: str) -> None:
        """Replace `old` with `new` once, and refuse to pass on a no-op."""
        path = root / relative
        text = path.read_text(encoding="utf-8")
        mutated = text.replace(old, new, 1)
        self.assertNotEqual(mutated, text,
                            f"the mutation of {relative} did not apply")
        path.write_text(mutated, encoding="utf-8")

    def test_the_committed_mirrors_agree_with_the_header(self) -> None:
        result = self.run_tool(self.root, "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("18 structures", result.stderr)
        self.assertIn("68 functions", result.stderr)
        self.assertIn("the three mirrors agree with", result.stderr)

    def test_write_is_a_fixed_point(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            before = {relative: (root / relative).read_bytes()
                      for relative in MIRRORED_FILES}
            first = self.run_tool(root, "--write")
            self.assertEqual(first.returncode, 0, first.stderr)
            after_first = {relative: (root / relative).read_bytes()
                           for relative in MIRRORED_FILES}
            self.assertEqual(after_first, before,
                             "a committed tree is already what --write emits, "
                             "so --write must change no byte")
            second = self.run_tool(root, "--write")
            self.assertEqual(second.returncode, 0, second.stderr)
            after_second = {relative: (root / relative).read_bytes()
                            for relative in MIRRORED_FILES}
            self.assertEqual(after_second, after_first,
                             "a second --write must leave the tree unchanged")

    def test_a_renamed_field_fails_with_the_structure_and_field_named(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '("padded_vocab_size", ctypes.c_int),',
                        '("padded_vocabulary_size", ctypes.c_int),')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a renamed field must not pass the check")
            self.assertIn("python/nanochat_cpp/_core.py:", result.stderr)
            self.assertIn("nanochat_config", result.stderr)
            self.assertIn("padded_vocab_size", result.stderr)

    def test_a_field_the_mirror_dropped_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '("value_embedding", ctypes.c_int),\n', "")
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a field the header declares and the mirror "
                                "lacks must not pass")
            self.assertIn("nanochat_config", result.stderr)
            self.assertIn("`nanochat_config.value_embedding`", result.stderr)
            self.assertIn("has no row for it", result.stderr)

    def test_a_field_the_mirror_invented_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '("rope_base", ctypes.c_float),',
                        '("rope_base", ctypes.c_float),\n'
                        '        ("rope_scale", ctypes.c_float),')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a field no header declares must not pass")
            self.assertIn("nanochat_config", result.stderr)
            self.assertIn("rope_scale", result.stderr)

    def test_a_changed_field_type_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '("total_memory_bytes", ctypes.c_int64),',
                        '("total_memory_bytes", ctypes.c_int32),')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a widened or narrowed mirror field must not "
                                "pass")
            self.assertIn("nanochat_device.total_memory_bytes", result.stderr)
            self.assertIn("spells it `c_int", result.stderr)
            self.assertIn("int64_t", result.stderr)

    def test_a_swapped_pair_of_rows_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '("num_layers", ctypes.c_int),\n        '
                        '("num_heads", ctypes.c_int),',
                        '("num_heads", ctypes.c_int),\n        '
                        '("num_layers", ctypes.c_int),')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "two rows in the wrong order must not pass")
            self.assertIn("nanochat_config.num_layers", result.stderr)
            self.assertIn("at position 2", result.stderr)
            self.assertIn("nanochat_config.num_heads", result.stderr)

    def test_a_structure_the_table_does_not_classify_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        'Record: "nanochat_record",',
                        'Record: "nanochat_records",')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a structure on one side alone must not pass")
            self.assertIn("the `_LAYOUT` table names `nanochat_records`",
                          result.stderr)
            self.assertIn("`nanochat_record`: the `_LAYOUT` table does not "
                          "classify it", result.stderr)

    def test_a_dropped_signature_row_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '    ("nanochat_runtime_step", (RuntimeHandle,), '
                        'ctypes.c_int),\n', "")
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a function with no row must not pass")
            self.assertIn("`nanochat_runtime_step` has no row", result.stderr)

    def test_a_changed_argument_type_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "python/nanochat_cpp/_core.py",
                        '        (ctypes.c_int, ctypes.POINTER(Device)),',
                        '        (ctypes.c_int64, ctypes.POINTER(Device)),')
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "an argument whose type moved must not pass")
            self.assertIn("`nanochat_device_at`: argument 1", result.stderr)
            self.assertIn("ctypes.c_int64", result.stderr)

    def test_a_dropped_drift_row_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "bindings/nanochat_capi_test.cc",
                        'static_assert(sizeof(nanochat_param::rows) ==\n'
                        '                  sizeof(nanochat::ParamView::rows),\n'
                        '              "nanochat_param.rows drifted from '
                        'nanochat::ParamView");\n', "")
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a drift row that disappeared must not pass")
            self.assertIn("bindings/nanochat_capi_test.cc:", result.stderr)
            self.assertIn("`nanochat_param.rows`", result.stderr)

    def test_a_field_added_to_the_header_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "include/nanochat/capi.h",
                        "  int device_count;\n",
                        "  int device_count;\n  int device_rank;\n")
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a header field with no mirror must not pass")
            self.assertIn("nanochat_metrics", result.stderr)
            self.assertIn("device_rank", result.stderr)

    def test_an_unparsed_form_fails_with_the_offending_line(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nanochat-mirrors-") as tmp:
            root = self.copy_tree(Path(tmp))
            self.mutate(root, "include/nanochat/capi.h",
                        "  int device_count;\n",
                        "  int device_count;\n  int device_grid[2][2];\n")
            result = self.run_tool(root, "--check")
            self.assertNotEqual(result.returncode, 0,
                                "a form the parser does not know must stop "
                                "the run")
            self.assertIn("include/nanochat/capi.h:", result.stderr)
            self.assertIn("does not understand", result.stderr)
            self.assertIn("device_grid", result.stderr)


if __name__ == "__main__":
    unittest.main()
