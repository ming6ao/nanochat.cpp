"""T0: the style gate.

The gate is mostly text checks, so the suite needs no compiler and no real
clang-format. It puts a fake `clang-format` on `PATH` and gives the gate one
file per violation. Hermetic: standard library only.

See docs/testing.md and CONTRIBUTING.md.
"""

from __future__ import annotations

import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanochat_cpp import lint

FORMATTER = """#!/bin/sh
exit {code}
"""


class LintTest(unittest.TestCase):
    """The four text checks, the formatter call, and the exit codes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="nanochat_lint_")
        cls.root = Path(cls.tmp.name)
        cls.bin_dir = cls.root / "bin"
        cls.bin_dir.mkdir()
        cls.formatter = cls.bin_dir / "clang-format"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def write_formatter(self, code: int) -> None:
        self.formatter.write_text(FORMATTER.format(code=code), encoding="utf-8")
        self.formatter.chmod(stat.S_IRWXU)

    def write_source(self, name: str, text: str) -> str:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def run_lint(self, *paths: str, code: int = 0,
                 on_path: bool = True) -> tuple[int, str, str]:
        self.write_formatter(code)
        path = str(self.bin_dir) if on_path else str(self.root / "empty")
        if not on_path:
            (self.root / "empty").mkdir()
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"PATH": path}), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            status = lint.main(list(paths))
        return status, stdout.getvalue(), stderr.getvalue()

    # -- the pass case ----------------------------------------------------

    def test_a_clean_file_passes(self) -> None:
        source = self.write_source("clean.cc", '#include "nanochat/x.h"\n')
        status, out, err = self.run_lint(source)
        self.assertEqual(status, 0, err)
        self.assertEqual(out, "lint: OK (1 files)\n")
        self.assertEqual(err, "")

    def test_a_project_include_is_allowed(self) -> None:
        source = self.write_source(
            "project.cc", '#include "backends/cuda/kernels/x.cuh"\n')
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 0, err)

    # -- the four checks --------------------------------------------------

    def test_a_using_directive_fails(self) -> None:
        source = self.write_source("using.cc", "using namespace std;\n")
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 1)
        self.assertIn("using-directive; Google bans 'using namespace'", err)
        self.assertIn(f"{source}:1:using namespace std;", err)
        self.assertIn("lint: FAILED", err)

    def test_a_bare_project_include_fails(self) -> None:
        source = self.write_source("bare.cc", '#include "tensor.h"\n')
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 1)
        self.assertIn("bare project include", err)
        self.assertIn('#include "tensor.h"', err)

    def test_a_system_include_is_allowed(self) -> None:
        source = self.write_source("system.cc", "#include <vector>\n")
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 0, err)

    def test_a_c_style_cast_fails(self) -> None:
        source = self.write_source("cast.cc", "int n = (int)value;\n")
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 1)
        self.assertIn("C-style cast", err)

    def test_a_non_ascii_character_fails(self) -> None:
        source = self.write_source("ascii.cc", "// caf\u00e9\n")
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 1)
        self.assertIn("non-ASCII character", err)

    def test_every_check_reports_in_one_run(self) -> None:
        source = self.write_source(
            "all.cc",
            'using namespace std;\n#include "tensor.h"\nint n = (int)v;\n'
            "// caf\u00e9\n")
        status, _, err = self.run_lint(source)
        self.assertEqual(status, 1)
        for message, _ in lint.CHECKS:
            self.assertIn(message, err)

    # -- the formatter ----------------------------------------------------

    def test_a_formatter_failure_fails_the_gate(self) -> None:
        source = self.write_source("formatted.cc", "int a;\n")
        status, _, err = self.run_lint(source, code=1)
        self.assertEqual(status, 1)
        self.assertIn("lint: clang-format reported differences", err)

    def test_a_missing_formatter_fails_the_gate(self) -> None:
        source = self.write_source("nofmt.cc", "int a;\n")
        status, _, err = self.run_lint(source, on_path=False)
        self.assertEqual(status, 1)
        self.assertIn("lint: clang-format is not on PATH", err)


if __name__ == "__main__":
    unittest.main()
