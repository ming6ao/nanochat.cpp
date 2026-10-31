"""G5: the core never names the Z2 package.

Gate G5 of ``docs/dependency-layering-plan.md`` section 8, and rule 5 of the
five that hold the zones: ``python/nanochat_cpp`` reaches the adapter package
by being handed a callable, never by naming it. A text scan of every file under
the package fails on one mention of ``nanochat_cpp_bridge`` and reports the
file and the line.

The scan reads the text and not the syntax tree on purpose. A docstring example
that imports the Z2 package invites the next reader to do it (section 11), so a
mention in prose is a failure too, and a comment or a string is not exempt.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

#: The package the scan covers, relative to the repository root.
PACKAGE = "nanochat_cpp"

#: The string the core may not contain, for any reason.
FORBIDDEN = "nanochat_cpp_bridge"

#: The one directory the scan skips: byte code is a build artifact, and the
#: file that carried the name is the source the scan already reads.
SKIP = {"__pycache__"}


def _runfiles_paths() -> list[Path]:
    """The repository root in the Bazel runfiles tree, if there is one."""
    paths: list[Path] = []
    bases = (os.environ.get("TEST_SRCDIR"), os.environ.get("RUNFILES_DIR"))
    for base in bases:
        if not base:
            continue
        workspace = os.environ.get("TEST_WORKSPACE")
        if workspace:
            paths.append(Path(base) / workspace)
        paths.append(Path(base))
    return paths


def package_dir() -> Path:
    """The package directory: the runfiles tree under Bazel, else above us."""
    for root in _runfiles_paths():
        candidate = root / "python" / PACKAGE
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[1] / PACKAGE


def scanned_files(root: Path) -> list[Path]:
    """Every source file under ``root``, in path order."""
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and not SKIP & set(path.parts))


def mentions(root: Path) -> list[str]:
    """Every ``file:line`` in ``root`` that names :data:`FORBIDDEN`."""
    reports: list[str] = []
    for path in scanned_files(root):
        text = path.read_bytes().decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN in line:
                reports.append(f"{path}:{number}: {line.strip()}")
    return reports


class CoreIndependenceTest(unittest.TestCase):
    def test_the_core_never_names_the_adapter_package(self) -> None:
        root = package_dir()
        self.assertTrue((root / "__init__.py").is_file(),
                        f"the package is not under {root}")
        reports = mentions(root)
        self.assertEqual(
            reports, [],
            "the core names the Z2 package; hand its callable in through "
            "nanochat_cpp.seam instead (rule 5 of section 4):\n"
            + "\n".join(reports))

    def test_the_scan_finds_a_mention_in_a_docstring(self) -> None:
        """The gate can fail, and prose counts."""
        with tempfile.TemporaryDirectory(prefix="nanochat_g5_") as tmp:
            path = Path(tmp) / "convert.py"
            path.write_text(
                '"""The converter.\n'
                "\n"
                "    from nanochat_cpp_bridge import torch_checkpoint\n"
                '"""\n',
                encoding="utf-8")
            reports = mentions(Path(tmp))
        self.assertEqual(len(reports), 1, reports)
        self.assertIn("convert.py:3", reports[0])


if __name__ == "__main__":
    unittest.main()
