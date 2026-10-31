"""G2: every import in the core is in the standard library.

Gate G2 of ``docs/dependency-layering-plan.md`` section 8, and the load-bearing
one: an abstract syntax tree walk over every ``.py`` file under
``python/nanochat_cpp``, failing when an absolute import's root name is neither
in :data:`sys.stdlib_module_names` nor the package itself. Every leak of
section 2.1 lands here, including one hidden inside a function, because the
walk descends into function and class bodies where a lazy import sits.

A relative import (``from . import _lib``) is inside the package by
construction, so it is allowed. A violation reports the file, the line, and
the module name, which is what the reader needs to move the import to the Z2
package of section 4.

The scan needs no third-party wheel and no network. Gate G1 is the independent
check: a scan reads the source, so it cannot see an import the source builds at
run time, and an isolated interpreter can.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

#: The package the scan covers, relative to the repository root.
PACKAGE = "nanochat_cpp"


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


def imported_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Every absolute import in ``tree``, as ``(line, module name)``.

    ``ast.walk`` visits nested statements, so an import inside a function or a
    class body is found exactly like a module-level one. A relative import
    carries ``level > 0`` and stays inside the package, so it is skipped.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module))
    return found


def allowed(module: str) -> bool:
    """True when ``module`` is the standard library or this package."""
    root = module.split(".")[0]
    return root in sys.stdlib_module_names or root == PACKAGE


def violations(directory) -> list[str]:
    """Every non-standard-library import under ``directory``, as text."""
    reports: list[str] = []
    for path in sorted(Path(directory).rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for line, module in imported_names(ast.parse(source, str(path))):
            if not allowed(module):
                reports.append(f"{path}:{line}: import {module}")
    return reports


class StdlibImportTest(unittest.TestCase):
    def test_every_core_import_is_in_the_standard_library(self) -> None:
        directory = package_dir()
        self.assertTrue((directory / "__init__.py").is_file(),
                        f"the package is not under {directory}")
        reports = violations(directory)
        self.assertEqual(reports, [], "the core imports outside the standard "
                                      "library; move each one to the Z2 "
                                      "package of section 4:\n"
                                      + "\n".join(reports))

    def test_the_scan_sees_an_import_inside_a_function(self) -> None:
        """The gate can fail, and a lazy import is the case that matters."""
        with tempfile.TemporaryDirectory(prefix="nanochat_g2_") as tmp:
            path = Path(tmp) / "lazy.py"
            path.write_text(
                "def convert(state):\n"
                "    import torch  # noqa: PLC0415 - the leak of section 2.1\n"
                "    return torch\n",
                encoding="utf-8")
            reports = violations(tmp)
        self.assertEqual(len(reports), 1, reports)
        self.assertIn("lazy.py:2", reports[0])
        self.assertIn("import torch", reports[0])


if __name__ == "__main__":
    unittest.main()
