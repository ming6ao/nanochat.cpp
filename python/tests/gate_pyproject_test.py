"""G3: the base install carries no third-party dependency.

Gate G3 of ``docs/dependency-layering-plan.md`` section 8:
``python/pyproject.toml`` declares ``dependencies = []`` (the line of section
6.1), so ``pip install nanochat-cpp`` pulls no third-party package and the
wheel stays pure (rule 1 of section 4). The optional dependencies below that
line carry what a task needs.

The gate parses the file with ``tomllib`` and fails when
``project.dependencies`` is missing, is not a list, or is not empty, naming the
offending entry. It is red on the tree that has no ``pyproject.toml`` yet and
turns green in the packaging node, which is the intended order: the invariant
is asserted before the file that satisfies it. The gate therefore never
softens into an import-or-skip -- a missing file is a failure, not a skip.

``tomllib`` is the standard-library TOML reader from Python 3.11 on, and this
tree's toolchain is newer.
"""

from __future__ import annotations

import os
import tomllib
import unittest
from pathlib import Path

#: The file that declares the base dependency list, from the repo root.
PYPROJECT = "python/pyproject.toml"


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


def pyproject_path() -> Path:
    """``python/pyproject.toml``: the runfiles tree under Bazel, else here."""
    for root in _runfiles_paths():
        candidate = root / PYPROJECT
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / PYPROJECT


def load(path: Path) -> dict:
    """The parsed ``pyproject.toml``."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


class BaseDependencyTest(unittest.TestCase):
    """The empty base dependency list of section 6.1, as a test."""

    def project(self) -> dict:
        """The ``[project]`` table, or a failure naming what is absent."""
        path = pyproject_path()
        self.assertTrue(
            path.is_file(),
            f"{path} is missing: the base install declares its empty "
            f"dependency list there (section 6.1), and without the file a "
            f"reader cannot tell an extra from a requirement")
        project = load(path).get("project")
        self.assertIsNotNone(project, f"{path} declares no [project] table")
        return project

    def test_the_pyproject_exists(self) -> None:
        self.project()

    def test_the_base_dependency_list_is_empty(self) -> None:
        dependencies = self.project().get("dependencies")
        self.assertIsInstance(
            dependencies, list,
            "project.dependencies is missing; declare it as an empty list "
            "(section 6.1), because an absent key reads as 'unknown' and the "
            "pure wheel rests on that line")
        self.assertEqual(
            dependencies, [],
            "the base install must carry no third-party package; each entry "
            "below belongs in [project.optional-dependencies] under the extra "
            "of section 6.2 (pip install 'nanochat-cpp[reference]', for "
            "example):\n"
            + "\n".join(str(entry) for entry in dependencies))


if __name__ == "__main__":
    unittest.main()
