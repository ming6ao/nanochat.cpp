"""G1: the core imports inside an isolated interpreter.

Gate G1 of ``docs/dependency-layering-plan.md`` section 8. The scan of gate G2
reads the source; this gate runs it. A fresh ``python3 -I -S`` subprocess puts
``python/`` on ``sys.path`` and imports ``nanochat_cpp``, then resolves every
public name and every submodule the package offers, so the lazy ``__getattr__``
hook of section 2.3 loads each module. A third-party import anywhere on that
path fails the gate with the subprocess's own traceback, including an import
the scan could miss because it is built at run time.

``-I`` drops the current directory from ``sys.path`` and ignores the
environment, and ``-S`` keeps ``site-packages`` off the path, so the import
resolves against the standard library and this tree alone. The ``-c`` source
inserts the path itself, because ``-I`` has already removed the working
directory by then.

The gate needs no wheel, no library, and no network: it is green on a host with
none of the extras installed, which is the pure-wheel invariant of section 6.1.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: The interpreter flag set of section 8: isolated, and no ``site``.
FLAGS = ("-I", "-S", "-c")

#: The probe. ``python`` is the absolute path of ``python/``, which goes on
#: ``sys.path`` from inside the source because ``-I`` drops the working
#: directory. Every public name and every submodule is resolved, so one probe
#: covers the whole package rather than a chosen import.
PROBE = """\
import sys
sys.path.insert(0, {python!r})
import nanochat_cpp
for name in sorted(set(nanochat_cpp.__all__) | set(nanochat_cpp._SUBMODULES)):
    getattr(nanochat_cpp, name)
print("imported", nanochat_cpp.__name__)
"""

#: The marker a failed interpreter writes to stderr.
TRACEBACK = "Traceback (most recent call last)"


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


def python_dir() -> Path:
    """``python/``: the runfiles tree under Bazel, else the directory above us.

    A ``py_test`` sees the same repository layout in the runfiles tree and in
    the source tree, so one lookup serves both: the runfiles win when they
    carry the package, and the directory this file lives in answers a direct
    run.
    """
    for root in _runfiles_paths():
        candidate = root / "python"
        if (candidate / "nanochat_cpp" / "__init__.py").is_file():
            return candidate
    return Path(__file__).resolve().parents[1]


def run_probe(directory) -> subprocess.CompletedProcess:
    """Import the package found at ``directory/nanochat_cpp`` in a subprocess.

    ``directory`` is the one the package lives in -- ``python/`` for the real
    tree -- because that is what ``sys.path`` needs to resolve
    ``import nanochat_cpp``.
    """
    source = PROBE.format(python=str(Path(directory)))
    return subprocess.run(
        [sys.executable or "python3", *FLAGS, source],
        capture_output=True, text=True, timeout=300, check=False)


class ImportIsolationTest(unittest.TestCase):
    def test_the_package_imports_with_no_third_party_package(self) -> None:
        directory = python_dir()
        self.assertTrue((directory / "nanochat_cpp" / "__init__.py").is_file(),
                        f"the package is not under {directory}")
        result = run_probe(directory)
        self.assertNotIn(TRACEBACK, result.stderr,
                         f"an isolated import raised:\n{result.stderr}")
        self.assertEqual(result.returncode, 0,
                         f"an isolated import failed:\n{result.stderr}")

    def test_the_probe_notices_a_third_party_import(self) -> None:
        """The gate can fail: a package that imports one fails the same probe.

        The fake package imports ``numpy``, which ``-S`` keeps off the path, so
        the probe has to exit nonzero. A green result here would mean the
        isolation flags stopped isolating.
        """
        with tempfile.TemporaryDirectory(prefix="nanochat_g1_") as tmp:
            package = Path(tmp) / "nanochat_cpp"
            package.mkdir()
            (package / "__init__.py").write_text(
                "__all__ = []\n_SUBMODULES = ()\nimport numpy\n",
                encoding="utf-8")
            result = run_probe(tmp)
        self.assertNotEqual(result.returncode, 0,
                            f"the probe accepted a third-party import:\n"
                            f"{result.stdout}")
        self.assertIn("numpy", result.stderr)


if __name__ == "__main__":
    unittest.main()
