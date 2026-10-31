"""T0: the ``tools/nanochat parity`` dispatch of section 7.3.

Gap found by the campaign review: the campaign added ``parity produce`` to
``tools/nanochat`` and no test ran it, so a broken kind lookup or a broken
re-exec would still pass every phase gate. This module closes that gap without
an extra installed.

The dispatch has two halves and only the first is testable here. The second half
runs a producer under the reference interpreter and needs PyTorch, so this test
covers the first half: the usage, the argument errors, and the producer's own
``--help``, which the dispatch serves in process from the producer's parser.
``test_extras_absent`` is the canary that makes the second half meaningful: when
no ``torch`` is importable, an exit code of 0 proves the help path reached the
producer without importing it.

See ``docs/dependency-layering-plan.md`` section 7.3 and section 8.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

#: The producer kinds the usage must list, and the alias of the host generator.
KINDS = ("train", "train_parity")

#: One option that only the producer's own parser defines.
PRODUCER_OPTION = "--nonzero-projections"

#: The environment variable that marks the child of a re-exec.
INPROCESS_ENV = "NANOCHAT_PARITY_INPROCESS"


def _repository_root() -> Path:
    """The repository root, in the runfiles tree or in the source tree."""
    bases = (os.environ.get("TEST_SRCDIR"), os.environ.get("RUNFILES_DIR"))
    for base in bases:
        if not base:
            continue
        workspace = os.environ.get("TEST_WORKSPACE")
        candidate = Path(base) / workspace if workspace else Path(base) / "_main"
        if (candidate / "tools" / "nanochat").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


class ParityDispatchTest(unittest.TestCase):
    """The dispatch answers without the reference environment."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _repository_root()
        cls.entry = cls.root / "tools" / "nanochat"

    def run_parity(self, *arguments: str) -> subprocess.CompletedProcess:
        """Run ``tools/nanochat parity`` with a clean parity environment."""
        env = dict(os.environ)
        env.pop(INPROCESS_ENV, None)
        return subprocess.run(
            [sys.executable, str(self.entry), "parity", *arguments],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )

    def test_extras_absent(self) -> None:
        """The canary: this interpreter has no torch, so no help path needs it."""
        self.assertIsNone(importlib.util.find_spec("torch"))

    def test_help_lists_the_kinds_and_names_the_design(self) -> None:
        result = self.run_parity("help")
        self.assertEqual(0, result.returncode, result.stderr)
        for kind in KINDS:
            self.assertIn(kind, result.stdout)
        self.assertIn("section 7.3", result.stdout)
        self.assertIn("No producer runs under `test`", result.stdout)

    def test_no_action_prints_the_usage_and_fails(self) -> None:
        result = self.run_parity()
        self.assertEqual(2, result.returncode)
        self.assertIn("parity produce <kind>", result.stdout)

    def test_unknown_action_is_named(self) -> None:
        result = self.run_parity("invent")
        self.assertEqual(2, result.returncode)
        self.assertIn("unknown parity action 'invent'", result.stderr)
        self.assertIn("nanochat: ", result.stderr)

    def test_produce_without_a_kind_prints_its_usage(self) -> None:
        result = self.run_parity("produce")
        self.assertEqual(2, result.returncode)
        self.assertIn("usage: tools/nanochat parity produce <kind>", result.stderr)

    def test_unknown_kind_names_the_producers(self) -> None:
        result = self.run_parity("produce", "invent")
        self.assertEqual(2, result.returncode)
        self.assertIn("unknown fixture kind 'invent'", result.stderr)
        for kind in KINDS:
            self.assertIn(kind, result.stderr)

    def test_each_kind_serves_its_own_help(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                result = self.run_parity("produce", kind, "--help")
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn(PRODUCER_OPTION, result.stdout)
                self.assertIn("--check", result.stdout)


if __name__ == "__main__":
    unittest.main()
